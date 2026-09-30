"""``g1``: one command, one subcommand per job.

    g1 run    --env sim --tools tpose --headless               # a tool (or a chain), fully checked
    g1 run    --env sim --tools walk_forward:0.5,hold:1,turn:45 --headless
    g1 run    --env sim --tools search --instruction "find the mug" ...   # the model decides
    g1 run    --env sim --tools replay --episode runs/<dir>     # a recorded run, no camera or model
    g1 run    --env robot --tools tpose --iface <iface> --mode standing
    g1 task   run tasks/find_the_mug --env sim --headless      # a task folder: instruction + context
    g1 tools  [--json] [--joints]                              # the menu; the catalog the model reads
    g1 limits [--source guess]                                 # every tunable number
    g1 camera | decide | episode | demo | scene                # the other entry points
    g1 new tool NAME | g1 new task NAME                        # scaffold a tool file or a task folder

``--env`` defaults to $G1_ENV, then ``sim``; ``--tools`` to $G1_TOOLS. On macOS
the sim viewer needs ``mjpython -m g1 run ...``; ``--headless`` needs nothing.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from g1.agent.agent import AGENTS
from g1.core import limits
from g1.core.action import Action, Obs, Runnable, Segment
from g1.envs import ENVS, Env, EnvAbort
from g1.envs.base import shield_sigint
from g1.tools import TOOLS, Tool, describe_menu, joint_table, menu, parse_chain
from g1.vlm import load_dotenv

ROOT = limits.ROOT


def run_parser(prog: str = "g1 run") -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description="run a tool, a chain of tools, the agent, or a replay")
    p.add_argument("--env", "-e", choices=sorted(ENVS), default=os.environ.get("G1_ENV", "sim"),
                   help="where to run (default: $G1_ENV or 'sim')")
    p.add_argument("--tools", "-t", default=os.environ.get("G1_TOOLS"),
                   help="comma-separated tools with arguments, e.g. tpose,turn:45 (see `g1 tools`), or "
                        "'search' (the model decides) or 'replay' (--episode DIR). Default: $G1_TOOLS")
    p.add_argument("--pause", type=float, default=1.0, help="seconds to hold between chained tools (default 1.0)")
    p.add_argument("--camera", choices=("auto", "on", "off"), default="auto",
                   help="open the env's camera: auto = only if the program uses it (default)")
    p.add_argument("--view", type=int, nargs="?", const=8765, default=None, metavar="PORT",
                   help="show the head camera live at http://127.0.0.1:PORT (default 8765; 0 = any free port); "
                        "opens the camera even for a chain")
    p.add_argument("--view-width", type=int, default=960, help="width of the live view's JPEGs (default 960)")
    p.add_argument("--record", nargs="?", const="auto", default=None, metavar="PATH",
                   help="record every camera frame to an .mp4 (PyAV): into the run directory as camera.mp4, or "
                        "runs/<ts>_<env>_<name>.mp4 for a chain, or the given PATH")
    p.add_argument("--max-time", type=float, default=limits.get("max_time_s"),
                   help="stop (and return to a safe state) if the program runs longer than this many seconds "
                        "(default: limit max_time_s)")
    a = p.add_argument_group("agent (--tools search / replay)")
    a.add_argument("--instruction", "-i", default=None, help="what the model is asked to do, e.g. \"find the mug\"")
    a.add_argument("--input-json", default=None, metavar="FILE",
                   help="a manifest: {\"instruction\", \"content\": [text, {\"image\"}, {\"video\", \"mode\"}]} "
                        "(--instruction may then be omitted)")
    a.add_argument("--demo", default=None, metavar="PATH",
                   help="a demonstration shown to the model on turn 0: a video file, a recorded runs/<dir>, "
                        "or a demo.json bundle (g1 demo prepare)")
    a.add_argument("--demo-mode", choices=("video", "video+action"), default=None,
                   help="video: images only (default for video files); video+action: also the tools, joint "
                        "angles and base poses (default for recorded runs)")
    a.add_argument("--demo-select", choices=("auto", "model", "uniform"), default="auto",
                   help="how keyframes are picked from a video file: the vision model (default when its key is set) "
                        "or evenly spaced")
    a.add_argument("--demo-frames", type=int, default=limits.get("demo_frames"),
                   help="keyframes to keep per demonstration (default: limit demo_frames, at most demo_frames_max)")
    a.add_argument("--ref", action="append", default=None, metavar="IMAGE",
                   help="a reference image (a photo of the goal) shown on turn 0; repeatable")
    a.add_argument("--episode", default=None, metavar="DIR", help="recorded run to replay (--tools replay)")
    a.add_argument("--model", default=None, help="model id to query (default: $VLM_MODEL or vlm.DEFAULT_MODEL)")
    a.add_argument("--echo", action="store_true", help="print the model's streamed text live")
    a.add_argument("--max-decisions", type=int, default=limits.get("max_decisions"),
                   help="model decisions per run, rejected replies included (default: limit max_decisions)")
    a.add_argument("--step-timeout", type=float, default=limits.get("step_timeout_s"),
                   help="wall seconds to wait for one decision before the run fails (default: limit step_timeout_s)")
    a.add_argument("--live-image-window", type=int, default=limits.get("live_image_window"),
                   help="observation images kept in the conversation; older turns keep their text "
                        "(default: limit live_image_window)")
    a.add_argument("--fresh-turns", action="store_true",
                   help="no conversation memory: every decision is a fresh chat with only the current observation")
    a.add_argument("--safety-note", action="append", default=None, metavar="TEXT",
                   help="a persistent physical fact the camera cannot see, added to the prompt (repeatable)")
    a.add_argument("--no-verdict", action="store_true",
                   help="do not ask for the human success/failed verdict after the run")
    a.add_argument("--log", default="runs", metavar="DIR", help="where search records its steps (default runs/)")
    a.add_argument("--no-log", action="store_true", help="do not record the run")
    for env_cls in ENVS.values():
        env_cls.add_args(p)
    return p


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


def _taps(args, env: Env, program: Runnable) -> list:
    """``--view`` / ``--record``: start the taps on the env's camera slot."""
    from g1.camera import Recorder, Viewer
    view = getattr(args, "view", None)
    record = getattr(args, "record", None)
    if view is None and record is None:
        return []
    source = env.source()
    if source is None:
        raise SystemExit("--view/--record need the camera, but the env opened none")
    taps = []
    if view is not None:
        viewer = Viewer(source, view, width=getattr(args, "view_width", 960), title=f"{program.name} @ {env.name}")
        viewer.start()
        print(f"view: {viewer.url}")
        taps.append(viewer)
    if record is not None:
        if record == "auto":
            recorder = getattr(program, "recorder", None)
            if recorder is not None:
                path = Path(recorder.dir) / "camera.mp4"
            else:
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                name = "".join(c if c.isalnum() or c in "+-_" else "_" for c in program.name)[:40]
                path = Path(getattr(args, "log", "runs")) / f"{stamp}_{env.name}_{name}.mp4"
        else:
            path = Path(record)
        try:
            rec = Recorder(source, path)
        except RuntimeError as e:
            raise SystemExit(str(e)) from None
        rec.start()
        print(f"recording: {path}")
        taps.append(rec)
    return taps


def run(program: Runnable, env: Env, max_time: float = limits.get("max_time_s")) -> bool:
    """The one loop every env shares:

        with env: obs = env.observe(env.reset()); program.reset(obs)
                  loop: action = program.step(t, obs); obs = env.observe(env.step(action))
        env.report()

    ``--view`` / ``--record`` taps start after the camera is up and stop before
    the env is left and before ``program.close()`` (which may rename the run
    directory the recording is in).
    """
    duration = getattr(program, "duration", None)
    if duration is not None and duration > max_time:
        print(f"warning: {program.name} lasts {duration:.1f}s but --max-time is {max_time:.0f}s; "
              f"it will be cut short")
    mode = getattr(env.args, "camera", "auto")
    tapped = getattr(env.args, "view", None) is not None or getattr(env.args, "record", None) is not None
    if mode == "off" and tapped:
        raise SystemExit("--camera off but --view/--record need frames; drop --camera off")
    env.use_camera = (program.uses_camera or tapped) if mode == "auto" else mode == "on"
    taps: list = []
    try:
        with env:
            obs = env.observe(env.reset())
            taps = _taps(env.args, env, program)
            program.reset(obs)
            n = 0
            last: Optional[Action] = None
            try:
                while True:
                    t = n * program.dt
                    if t > max_time:
                        print(f"aborted: the program exceeded --max-time {max_time}s")
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
                for tap in taps:
                    tap.stop()
                    if hasattr(tap, "summary"):
                        print(tap.summary())
    finally:
        program.close()
    return env.report()


# --------------------------------------------------------------------------
# g1 run
# --------------------------------------------------------------------------

def build_program(args, env: Env, parser: argparse.ArgumentParser) -> Runnable:
    """``--tools``: an agent by name, or a chain of tools."""
    try:
        if args.tools in AGENTS:
            return AGENTS[args.tools](args, env.can_walk, env.has_loco)
        return parse_chain(args.tools, pause=args.pause)
    except KeyError as e:
        parser.error(f"unknown tool {e}; agents: {', '.join(sorted(AGENTS))}; "
                     f"tools: {', '.join(n for n, t in TOOLS.items() if t.kind == 'motion' and n not in ('takeover', 'handback'))}")
    except (ValueError, RuntimeError) as e:
        parser.error(str(e))


def run_main(argv: list[str], parser: Optional[argparse.ArgumentParser] = None,
             defaults: Optional[dict] = None) -> int:
    parser = parser or run_parser()
    if defaults:
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.tools is None:
        parser.error("--tools is required (or set $G1_TOOLS); see `g1 tools`")
    env = ENVS[args.env](args)
    program = build_program(args, env, parser)
    segments = getattr(program, "segments", None)
    if segments is not None and any(s.base for s in segments()) and not env.can_walk:
        parser.error(f"{program.name} drives the base; {env.name} cannot walk here"
                     + (" (pass --walk after reading its pre-flight)" if args.env == "robot"
                        else " (drop --free-base)" if args.env == "sim" else ""))
    print(f"== {program.name} @ {env.name} ==")
    if args.tools == "search" and args.max_time <= 120.0:
        print("hint: search is open-ended (6-10 s per decision); raise --max-time, e.g. --max-time 600")
    ok = run(program, env, max_time=args.max_time)
    return 0 if ok else 1


# --------------------------------------------------------------------------
# g1 tools, g1 new
# --------------------------------------------------------------------------

def tools_main(argv: list[str]) -> int:
    from g1.tools import function_schemas
    p = argparse.ArgumentParser(prog="g1 tools", description="the tools, and what the model reads about them")
    p.add_argument("--json", action="store_true", help="the function schemas the model is given (every visible tool)")
    p.add_argument("--joints", action="store_true", help="the joints arm_path may command: limits and stand pose")
    p.add_argument("--prompts", action="store_true", help="also print each tool's prompt")
    args = p.parse_args(argv)
    import json
    if args.json:
        print(json.dumps(function_schemas(menu(True, True)), ensure_ascii=False, indent=1))
        return 0
    if args.joints:
        for r in joint_table():
            print(f"  {r['name']:<22} [{r['min']:+.3f}, {r['max']:+.3f}] rad   stand {r['stand']:+.3f}")
        return 0
    visible = [t for t in TOOLS.values() if t.visible]
    presets = [t for t in TOOLS.values() if not t.visible and t.kind == "motion" and t.name not in ("takeover", "handback")]
    print("offered to the model (menu(allow_base, has_loco) hides what an env cannot do):")
    print(describe_menu(visible))
    print("\npresets (CLI chains and replay only):")
    print(describe_menu(presets))
    print("\nagents: search --instruction \"...\" (the model decides), replay --episode DIR")
    print("chain them: g1 run --env sim --tools walk_forward:0.5,hold:1,turn:45,tpose --headless")
    if args.prompts:
        print()
        for t in visible:
            print(f"- {t.name}: {t.describe()}\n")
    return 0


def new_main(argv: list[str]) -> int:
    import re
    import shutil
    p = argparse.ArgumentParser(prog="g1 new", description="scaffold a tool file or a task folder")
    p.add_argument("kind", choices=("tool", "task"))
    p.add_argument("name", help="snake_case, e.g. bow or find_the_mug")
    args = p.parse_args(argv)
    if re.fullmatch(r"[a-z][a-z0-9_]*", args.name) is None:
        p.error("the name must be snake_case: letters, digits and underscores, starting with a letter")
    if args.kind == "tool":
        if args.name in TOOLS:
            p.error(f"a tool named {args.name!r} already exists ({TOOLS[args.name].__module__})")
        dest = ROOT / "g1" / "tools" / f"{args.name}.py"
        if dest.exists():
            p.error(f"{dest} already exists")
        cls = "".join(part.capitalize() for part in args.name.split("_"))
        text = (ROOT / "g1" / "tools" / "_template.py").read_text(encoding="utf-8")
        dest.write_text(text.replace("ClassName", cls).replace("NAME", args.name), encoding="utf-8")
        print(f"wrote {dest.relative_to(ROOT)}\nedit it, then:  g1 run --env sim --tools {args.name} --headless")
        return 0
    dest = ROOT / "tasks" / args.name
    if dest.exists():
        p.error(f"{dest} already exists")
    shutil.copytree(ROOT / "tasks" / "_template", dest)
    print(f"wrote {dest.relative_to(ROOT)}/task.json\nedit it, then:  g1 task run tasks/{args.name} --env sim --headless")
    return 0


# --------------------------------------------------------------------------
# g1
# --------------------------------------------------------------------------

COMMANDS = {
    "run": "run a tool, a chain of tools, the agent (search) or a replay",
    "task": "run, show or evaluate a task folder",
    "tools": "list the tools and the catalog the model reads",
    "limits": "print every tunable limit with its source",
    "camera": "smoke-test the robot's head camera (no robot control)",
    "decide": "one real model decision from a saved frame",
    "episode": "inspect a recorded run",
    "demo": "compile or inspect a demonstration",
    "scene": "fetch the room's assets",
    "new": "scaffold a tool file or a task folder",
}


def main(argv: Optional[list[str]] = None) -> int:
    load_dotenv()              # before any parser: defaults read $G1_ENV, $UNITREE_ROBOT_IP, ...
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
    if command == "task":
        from g1.agent.task import main as task_main
        return task_main(rest)
    if command == "tools":
        return tools_main(rest)
    if command == "limits":
        from g1.core.limits import main as limits_main
        return limits_main(rest)
    if command == "camera":
        from g1.camera import main as camera_main
        return camera_main(rest)
    if command == "decide":
        from g1.agent.decider import main as decide_main
        return decide_main(rest)
    if command == "episode":
        from g1.agent.episode import main as episode_main
        return episode_main(rest)
    if command == "demo":
        from g1.agent.demo import main as demo_main
        return demo_main(rest)
    if command == "scene":
        from g1.envs.scene import main as scene_main
        return scene_main(rest)
    return new_main(rest)


if __name__ == "__main__":
    sys.exit(main())
