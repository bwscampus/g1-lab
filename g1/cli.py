"""``g1``: run a task, on the robot or on the twin.

    g1 run TASK --mode standing|gantry     # the robot (UNITREE_IFACE, UNITREE_ROBOT_IP from .env)
    g1 run TASK --mode standing --view     # ... and its head camera live in a browser
    g1 run TASK --sim [--mode gantry]      # the twin, in the MuJoCo window: the same robot code, over DDS
    g1 status                              # is the robot reachable, and what state is it in
    g1 twin fetch | status                 # the twin's model, walker and room assets
    g1 trace compare RUN_A RUN_B | show RUN   # how far the twin is from the robot on the same task
    g1 limits [SOURCE]                     # every number and default, from configs/limits.json

TASK is a folder under tasks/ (or its path): an ``instruction`` the vision model
works on, or a fixed ``tools`` chain (``g1/agent/task.py``). ``--mode standing``
lets the base walk; ``gantry`` never does. Every other number and default is in
configs/limits.json (``g1 limits``) or the task's task.json; .env holds only what is
private to this machine (the token, the robot's address and key). ``G1_VERBOSE=1``
prints every SDK call and a health line a second.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

from g1.core import limits
from g1.core.action import Action, Obs, Runnable, Segment
from g1.envs import Env, EnvAbort
from g1.envs.base import shield_sigint
from g1.tools import Tool, parse_chain
from g1.vlm import load_dotenv

ROOT = limits.ROOT


# --------------------------------------------------------------------------
# The loop, and the safe return
# --------------------------------------------------------------------------

TO_STAND = limits.get("return_to_stand_s")      # the safe return's move to STAND
RAMP = limits.get("return_ramp_s")              # then the weight fades to 0 (the onboard controller takes the arms)
RELEASED = limits.get("return_released_s")      # held at exactly weight 0 before the env is left


def safe_return(env: Env, last: Action, obs: Obs) -> str:
    """Bring the robot to a safe state after an interrupted run, at 50 Hz with
    no gap: from the last *commanded* pose (never the measured one) to STAND
    over TO_STAND, then the arm_sdk weight from where it was to 0 over RAMP,
    base stopped from the first tick. Ctrl-C is ignored until it is done."""
    from g1.core.config import STAND_Q
    joints = list(last.joints)
    stand = {j: float(STAND_Q[j]) for j in joints}
    w0 = float(last.weight)
    segs = [Segment(stand, TO_STAND, weight=lambda a, w=w0: w, label="returning to stand"),
            Segment(stand, RAMP, weight=lambda a, w=w0: w * (1.0 - a), label="handing back"),
            Segment(stand, RELEASED, weight=lambda a: 0.0, label="released")]
    ret = Tool.of(segs, joints=joints, name="safe-return")
    ret.reset(Obs(last.q))
    with shield_sigint("interrupt ignored: returning to a safe state, wait for the hand-over"):
        n = 0
        while (a := ret.step(n * ret.dt, obs)) is not None:
            obs = env.observe(env.step(a))
            n += 1
    return "completed"


def _return_to_safety(program: Runnable, env: Env, last: Optional[Action], obs: Obs, reason: str,
                      detail: str = "") -> None:
    if last is None:
        return                                  # nothing was commanded yet
    program.on_interrupt(reason, detail)
    print(f"returning to a safe state ({TO_STAND:.0f} s to stand, {RAMP:.0f} s hand-over)...")
    try:
        outcome = safe_return(env, last, obs)
    except Exception as e:                      # never mask the original failure
        print(f"safe return failed: {e}")
        outcome = "failed"
    program.on_returned(outcome)


def run(program: Runnable, env: Env, max_time: float = limits.get("max_time_s")) -> bool:
    """The one loop every env shares:

        with env: obs = env.observe(env.reset()); program.reset(obs)
                  loop: action = program.step(t, obs); obs = env.observe(env.step(action))
        env.report()

    The camera opens only when the program reads it (``Runnable.uses_camera``).
    """
    duration = getattr(program, "duration", None)
    if duration is not None and duration > max_time:
        print(f"warning: {program.name} lasts {duration:.1f}s but max_time_s is {max_time:.0f}s; "
              f"it will be cut short")
    camera = getattr(env.args, "camera", "auto")
    view = getattr(env.args, "view", False)
    env.use_camera = (program.uses_camera or view) if camera == "auto" else camera == "on"
    viewer = None
    try:
        with env:
            obs = env.observe(env.reset())
            if view:
                viewer = _start_view(env, program)
            if hasattr(program, "attach"):
                program.attach(env)
            program.reset(obs)
            n = 0
            last: Optional[Action] = None
            try:
                while True:
                    t = n * program.dt
                    if t > max_time:
                        print(f"aborted: the program exceeded max_time_s {max_time}s")
                        _return_to_safety(program, env, last, obs, "max_time")
                        return False
                    action = program.step(t, obs)
                    if action is None:
                        break
                    last = action
                    obs = env.observe(env.step(action))
                    n += 1
            except EnvAbort as e:
                print(f"\n{env.name}: {e}")
            except KeyboardInterrupt:
                print("\ninterrupted")
                _return_to_safety(program, env, last, obs, "ctrl_c")
                return False
            except Exception as e:
                print(f"\nerror: {e}")
                _return_to_safety(program, env, last, obs, "error", repr(e))
                raise
            finally:
                if viewer is not None:
                    viewer.stop()
    finally:
        program.close()
    return env.report()


VIEW_PORT = limits.get("view_port")


def _start_view(env: Env, program: Runnable):
    """``--view``: the head camera at http://127.0.0.1:VIEW_PORT (or a free port when that is taken)."""
    from g1.camera import Viewer
    source = env.source()
    if source is None:
        raise SystemExit("--view needs the head camera, but none opened")
    for port in (VIEW_PORT, 0):
        viewer = Viewer(source, port, title=f"{program.name} @ {env.name}")
        try:
            viewer.start()
            break
        except OSError:
            continue
    print(f"view: {viewer.url}")
    return viewer


# --------------------------------------------------------------------------
# g1 run TASK
# --------------------------------------------------------------------------

def run_settings(task: Optional[dict] = None, *, sim: bool = False, mode: Optional[str] = None,
                 view: bool = False, **overrides) -> argparse.Namespace:
    """Everything an env reads, from the task, configs/limits.json and the environment.
    The CLI decides only sim or robot, the mode, and the live view; ``overrides`` are for tests."""
    from g1.envs.scene import parse_object
    task = task or {}
    s = argparse.Namespace(
        sim=sim, mode=mode, view=view,
        iface=os.environ.get("UNITREE_IFACE"), domain=None,
        camera_ip=os.environ.get("UNITREE_ROBOT_IP"), camera_timeout=limits.get("camera_timeout_s"),
        camera_source="auto", camera="auto", trace=None,
        verbose=os.environ.get("G1_VERBOSE", "") not in ("", "0"),
        scene=task.get("scene") or "none", sim_objects=[parse_object(o) for o in task.get("objects") or []],
        camera_size=task.get("camera_size"),
        # the in-process twin (tests): headless, its own pacing and checks
        headless=True, realtime=None, hold_end=limits.get("sim_viewer_hold_s"), camera_dir=None,
        camera_fps=limits.get("twin_camera_fps"),
        camera_every=max(1, round(1.0 / (limits.get("twin_camera_fps") * limits.get("control_dt_s")))),
        sim_obstacle=None, twin_fsm=None, margin=limits.get("joint_margin_rad"),
        max_vel=limits.get("command_vel_max"), max_violations=int(limits.get("max_violations")),
    )
    for k, v in overrides.items():
        if not hasattr(s, k):
            raise TypeError(f"run_settings has no setting {k!r}")
        setattr(s, k, v)
    return s


def build_program(task: dict, env, runs: Path) -> Runnable:
    """The task's agent (an instruction) or its chain (tools)."""
    if task["tools"]:
        return parse_chain(task["tools"], pause=limits.get("chain_pause_s"))
    from g1.agent.agent import build_agent
    return build_agent(task, env_name=env.name, can_walk=env.can_walk, has_loco=env.has_loco, runs=runs)


def run_task(task: dict, *, sim: bool, mode: Optional[str], view: bool = False, env: Optional[Env] = None) -> int:
    """One run of a task, recorded. ``env`` (tests) is an in-process twin used instead of
    starting one over DDS."""
    from g1.agent.task import label_run, record_result, run_dir, write_status
    from g1.envs import RobotEnv
    if not sim and mode is None:
        raise SystemExit("the robot needs --mode gantry (hanging: Damp, stand, run, Damp; no walking) or "
                         "--mode standing (standing on its own: walks; never damps); or add --sim")
    mode = mode or "standing"
    in_process = env is not None
    if env is None:
        env = RobotEnv(run_settings(task, sim=sim, mode=mode, view=view))
    settings = env.args
    runs = task["dir"] / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in runs.iterdir()}
    try:
        program = build_program(task, env, runs)
    except (KeyError, ValueError, RuntimeError) as e:
        raise SystemExit(f"{task['dir'] / 'task.json'}: {e}") from None
    segments = getattr(program, "segments", None)
    if segments is not None and any(s.base for s in segments()) and not env.can_walk:
        raise SystemExit(f"{task['name']} drives the base, and --mode gantry never walks (use --mode standing)")
    recorder = getattr(program, "recorder", None)
    folder = Path(recorder.dir) if recorder is not None else run_dir(task, "sim" if sim else "robot")
    settings.trace = str(folder / "trace.npz")
    print(f"== {task['name']} ({'instruction' if task['instruction'] else 'tools'}) @ "
          f"{'the twin' if sim else 'the robot'}, mode {mode} ==")
    if sim and not in_process:
        from g1.twin.main import TwinProcess
        # the MuJoCo window always; G1_TWIN_HEADLESS=1 is for tests only
        window = os.environ.get("G1_TWIN_HEADLESS", "") in ("", "0")
        with TwinProcess(task, mode=mode, viewer=window, camera=program.uses_camera) as twin:
            settings.iface, settings.domain = twin.iface, twin.domain
            ok = run(program, env, max_time=task["max_time_s"])
    else:
        ok = run(program, env, max_time=task["max_time_s"])
    if recorder is not None:                     # the agent's close renamed its directory
        new = [p for p in runs.iterdir() if p.name not in before and p.is_dir()]
        folder = max(new, key=lambda p: p.stat().st_mtime) if new else folder
    else:
        write_status(folder, ok, "sim" if sim else "robot")
    if not sim:                                  # the robot: keep it? then the human's verdict
        from g1.agent import agent as prompts
        if not prompts.ask_save():
            shutil.rmtree(folder, ignore_errors=True)
            print("run not saved")
            return 0 if ok else 1
        human = prompts.ask_verdict()
        if human is not None:
            folder = label_run(folder, human)
    row = record_result(task, folder)
    if row is not None:
        shown = folder.relative_to(ROOT) if folder.is_relative_to(ROOT) else folder
        print(f"result: {row['outcome']} ({row['outcome_source']}) -> {shown}")
    return 0 if ok else 1


def run_main(argv: list[str]) -> int:
    from g1.agent.task import find, load_task
    p = argparse.ArgumentParser(prog="g1 run", description="run a task on the robot, or on the twin with --sim")
    p.add_argument("task", help="a folder under tasks/ (e.g. find_the_mug) or its path")
    p.add_argument("--sim", action="store_true", help="run on the twin instead of the robot, in the MuJoCo window")
    p.add_argument("--mode", choices=("standing", "gantry"), default=None,
                   help="standing: on its own feet, may walk (default with --sim). gantry: hanging; "
                        "Damp, stand, run, Damp; never walks. Required on the robot")
    p.add_argument("--view", action="store_true",
                   help=f"the robot: show its head camera live in a browser (http://127.0.0.1:{VIEW_PORT}). "
                        "--sim always shows the MuJoCo window")
    args = p.parse_args(argv)
    if args.view and args.sim:
        p.error("--view is the robot's camera; --sim already shows the twin in the MuJoCo window")
    try:
        task = load_task(find(args.task))
    except ValueError as e:
        p.error(str(e))
    return run_task(task, sim=args.sim, mode=args.mode, view=args.view)


# --------------------------------------------------------------------------
# g1
# --------------------------------------------------------------------------

COMMANDS = {
    "run": "run a task on the robot, or on the twin with --sim",
    "status": "is the robot reachable and what state is it in (read-only)",
    "twin": "fetch the twin's model, walker and room assets (fetch | status)",
    "trace": "compare two runs of a task, e.g. the twin's and the robot's (compare | show)",
    "limits": "print every limit and default in configs/limits.json as one table (or one source: g1 limits guess)",
}


def main(argv: Optional[list[str]] = None) -> int:
    load_dotenv()              # before anything reads $UNITREE_IFACE, $HF_TOKEN, ...
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(prog="g1", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=sorted(COMMANDS), nargs="?",
                   help="; ".join(f"{k}: {v}" for k, v in COMMANDS.items()))
    p.add_argument("args", nargs=argparse.REMAINDER)
    if not argv or argv[0] in ("-h", "--help"):
        p.print_help()
        return 0
    if argv[0] not in COMMANDS:
        p.error(f"unknown command {argv[0]!r}; choose from {', '.join(COMMANDS)}")
    command, rest = argv[0], argv[1:]
    if command == "run":
        return run_main(rest)
    if command == "status":
        from g1.envs.robot import status_main
        return status_main(rest)
    if command == "limits":
        return limits.main(rest)
    if command == "twin":
        from g1.twin.main import main as twin_main
        return twin_main(rest)
    from g1.envs.trace import main as trace_main
    return trace_main(rest)


if __name__ == "__main__":
    sys.exit(main())
