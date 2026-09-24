"""Run a policy in an environment.

    python   run.py --env sim --policy tpose --headless     # fast, windowless, fully checked
    python   run.py --env sim --policy tpose,turn:45        # ad hoc chain of skills
    mjpython run.py --env sim --policy demo                 # macOS viewer
    python   run.py --env robot --policy demo --iface eth0 --mode standing

``--policy`` is a registered routine, policy or agent name, or a comma-separated
chain of skills. Every sim run is checked: the measured joint angles, the
commanded targets, the command speed, the weight and any base velocity (see
``envs/monitor.py``), so ``--headless`` is the pre-flight for a new policy.

Policies that use the camera (``look``, ``wave_on_red``) get frames from the
env: rendered in sim, or replayed / random with ``--camera-dir`` /
``--camera-noise``, the head camera on the robot (``--camera-ip``). ``describe``
asks the Hugging Face model (``$HF_TOKEN``) for a scene description; ``search``
asks it for the next skill every decision, feeds back what happened, records
everything under ``runs/`` and asks you for the verdict at the end. ``goto_red``
and the walking skills drive the base: sim slides it, the robot needs ``--walk``.
``--env`` defaults to $G1_ENV, then "sim".
"""
from __future__ import annotations

import argparse
import os
import sys

from envs import ENVS, Env, EnvAbort
from perception import VISION_MODES, build_perceiver
from policy import Policy
from routines import POLICIES, ROUTINES, build_policy
from skills import SKILLS, describe_menu, use_catalog
from agent import AGENTS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="g1", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", "-e", choices=sorted(ENVS), default=os.environ.get("G1_ENV", "sim"),
                   help="where to run the policy (default: $G1_ENV or 'sim')")
    p.add_argument("--policy", "-p", default=os.environ.get("G1_POLICY"),
                   help="routine, policy or agent name, or comma-separated skills e.g. tpose,turn:45 "
                        "(default: $G1_POLICY)")
    p.add_argument("--pause", type=float, default=1.0,
                   help="seconds to hold between chained skills (default 1.0)")
    p.add_argument("--list", action="store_true",
                   help="list envs, routines, policies, agents and skills, then exit")
    p.add_argument("--camera", choices=("auto", "on", "off"), default="auto",
                   help="open the env's camera: auto = only if the policy uses it (default)")
    v = p.add_argument_group("vision")
    v.add_argument("--vision", choices=VISION_MODES, default="auto",
                   help="vision model for policies that use one: auto = run it iff the policy asks "
                        "and $HF_TOKEN is set (default); api = require it; off = never")
    v.add_argument("--vision-model", default=os.environ.get("G1_VISION_MODEL"),
                   help="HF model id for --vision api (default: $G1_VISION_MODEL or perception.DEFAULT_MODEL)")
    v.add_argument("--vision-interval", type=float, default=1.0,
                   help="cost floor: requests closer together than this are ignored (default 1.0); "
                        "policies decide when to ask (their vision_refresh, default 2 s)")
    v.add_argument("--vision-echo", action="store_true", help="print the model's streamed text live")
    a = p.add_argument_group("agent (--policy search / replay)")
    a.add_argument("--goal", default=None, help="what search should look for, e.g. \"find the mug\"")
    a.add_argument("--input-json", default=None, metavar="FILE",
                   help="a manifest: {\"instruction\", \"content\": [text, {\"image\"}, {\"video\", \"mode\"}]} "
                        "(--goal may then be omitted)")
    a.add_argument("--demo", default=None, metavar="PATH",
                   help="a demonstration shown to the model on turn 0: a video file, a recorded runs/<dir>, "
                        "or a demo.json bundle (python -m demo prepare)")
    a.add_argument("--demo-mode", choices=("video", "video+action"), default=None,
                   help="video: images only (default for video files); video+action: also the skills, joint "
                        "angles and base poses (default for recorded runs)")
    a.add_argument("--demo-select", choices=("auto", "model", "uniform"), default="auto",
                   help="how keyframes are picked from a video file: the vision model (default with $HF_TOKEN) "
                        "or evenly spaced")
    a.add_argument("--demo-frames", type=int, default=12, help="keyframes to keep per demonstration (default 12, max 24)")
    a.add_argument("--ref", action="append", default=None, metavar="IMAGE",
                   help="a reference image (a photo of the goal) shown on turn 0; repeatable")
    a.add_argument("--episode", default=None, metavar="DIR", help="recorded run to replay (--policy replay)")
    a.add_argument("--max-decisions", type=int, default=30,
                   help="model decisions per run, rejected replies included (default 30)")
    a.add_argument("--step-timeout", type=float, default=60.0,
                   help="wall seconds to wait for one decision before the run fails (default 60)")
    a.add_argument("--skills", default=None, metavar="FILE",
                   help="skill catalog JSON to use instead of configs/skills.json (prompts, ranges)")
    a.add_argument("--live-image-window", type=int, default=8,
                   help="observation images kept in the conversation; older turns keep their text (default 8)")
    a.add_argument("--fresh-turns", action="store_true",
                   help="no conversation memory: every decision is a fresh chat with only the current observation")
    a.add_argument("--safety-note", action="append", default=None, metavar="TEXT",
                   help="a persistent physical fact the camera cannot see, added to the prompt (repeatable)")
    a.add_argument("--no-verdict", action="store_true",
                   help="do not ask for the human success/failed verdict after the run")
    a.add_argument("--log", default="runs", metavar="DIR", help="where search records its steps (default runs/)")
    a.add_argument("--no-log", action="store_true", help="do not record the run")
    p.add_argument("--max-time", type=float, default=120.0,
                   help="abort if the policy runs longer than this many seconds (default 120)")
    for env_cls in ENVS.values():
        env_cls.add_args(p)
    return p


def run(policy: Policy, env: Env, max_time: float = 120.0, perceiver=None) -> bool:
    """The one loop every stage shares. ``perceiver`` (or a pre-set
    ``env.perceiver``) runs the vision model beside the loop; it is started
    before the env and stopped after it, so it never gates the env's teardown."""
    duration = getattr(policy, "duration", None)
    if duration is not None and duration > max_time:
        print(f"warning: {policy.name} lasts {duration:.1f}s but --max-time is {max_time:.0f}s; "
              f"it will be cut short")
    if perceiver is not None:
        env.perceiver = perceiver
    perceiver = env.perceiver
    mode = getattr(env.args, "camera", "auto")
    wants_camera = policy.uses_camera or perceiver is not None
    if mode == "off" and perceiver is not None:
        raise SystemExit("--camera off but the vision model needs frames; drop --camera off or use --vision off")
    env.use_camera = wants_camera if mode == "auto" else mode == "on"
    policy.perceiver = perceiver
    if perceiver is not None:
        perceiver.start()
    try:
        with env:
            obs = env.observe(env.reset())
            policy.reset(obs)
            n = 0
            try:
                while True:
                    t = n * policy.dt
                    if t > max_time:
                        print(f"aborted: policy exceeded --max-time {max_time}s")
                        return False
                    action = policy.step(t, obs)
                    if action is None:
                        break
                    obs = env.observe(env.step(action))
                    n += 1
            except EnvAbort as e:
                print(f"\n{env.name}: {e}")
            except KeyboardInterrupt:
                print("\ninterrupted")
                return False
    finally:
        if perceiver is not None:
            perceiver.stop()
            print(perceiver.summary())
        policy.close()
    return env.report()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.skills:
        try:
            use_catalog(args.skills)
        except ValueError as e:
            parser.error(str(e))
    if args.list:
        print("envs:     " + ", ".join(sorted(ENVS)))
        print("routines: " + ", ".join(sorted(ROUTINES)))
        print("policies: " + ", ".join(sorted(POLICIES)))
        print("agents:   " + ", ".join(sorted(AGENTS)) + "   (search --goal ..., replay --episode DIR)")
        print("skills:   (chain them, e.g. --policy walk_forward:0.5,turn:45,tpose)")
        print(describe_menu([s for s in SKILLS.values() if not s.internal]))
        return 0
    if args.policy is None:
        parser.error("--policy is required (or set $G1_POLICY)")

    env = ENVS[args.env](args)
    try:
        if args.policy in AGENTS:
            policy = AGENTS[args.policy](args, env.can_walk)
        else:
            policy = build_policy(args.policy, pause=args.pause)
    except KeyError as e:
        parser.error(f"unknown policy {e}; routines: {', '.join(sorted(ROUTINES))}; "
                     f"agents: {', '.join(sorted(AGENTS))}; policies: {', '.join(sorted(POLICIES))}; "
                     f"skills: {', '.join(s for s in SKILLS if not SKILLS[s].internal)}")
    except (ValueError, RuntimeError) as e:
        parser.error(str(e))
    try:
        perceiver = build_perceiver(args, policy)
    except RuntimeError as e:
        parser.error(str(e))
    if args.vision == "api" and args.env == "sim" and args.headless and args.realtime is None:
        print("warning: this run is not paced to realtime, so vision-model results will describe "
              "frames from well before they arrive; use --vision fake here, or --realtime 1 in sim")
    segments = getattr(policy, "segments", ())
    if any(getattr(s, "base", None) for s in segments) and not env.can_walk:
        parser.error(f"{policy.name} drives the base; {env.name} cannot walk here"
                     + (" (pass --walk after reading its pre-flight)" if args.env == "robot"
                        else " (drop --free-base)" if args.env == "sim" else ""))
    print(f"== {policy.name} @ {env.name} ==")
    if args.policy == "search" and args.max_time <= 120.0:
        print("hint: search is open-ended (6-10 s per decision); raise --max-time, e.g. --max-time 600")
    ok = run(policy, env, max_time=args.max_time, perceiver=perceiver)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
