"""Run a routine in an environment.

    python run.py --env check --policy tpose
    python run.py --env check --policy tpose,sixseven   # ad hoc chain of motions
    python run.py --env check --policy demo             # registered routine
    mjpython run.py --env sim --policy demo             # macOS viewer
    python run.py --env robot --policy demo --iface eth0 --mode standing

``--policy`` is a registered routine or policy name, or a comma-separated list
of motions. Policies that use the camera (``look``, ``wave_on_red``) get frames
from the env: rendered in sim, replayed or random in check (``--camera-dir``,
``--camera-noise``), the head camera on the robot (``--camera-ip``). Policies
that use vision (``describe``, ``wave_on_person``) also get a scene description
from ``--vision``: the offline fake by default, a Hugging Face model with
``--vision api`` and ``$HF_TOKEN``. ``goto_red`` walks to its target: sim slides
the base, the robot needs ``--walk``.
``--env`` defaults to $G1_ENV, then "check", so ``G1_ENV=sim`` also works.
"""
from __future__ import annotations

import argparse
import os
import sys

from envs import ENVS, Env, EnvAbort
from motions import MOTIONS
from perception import VISION_MODES, build_perceiver
from policy import Policy
from routines import POLICIES, ROUTINES, build_policy
from skills import SKILLS, describe_menu
from agent import AGENTS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="g1", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", "-e", choices=sorted(ENVS), default=os.environ.get("G1_ENV", "check"),
                   help="where to run the policy (default: $G1_ENV or 'check')")
    p.add_argument("--policy", "-p", default=os.environ.get("G1_POLICY"),
                   help="routine name, or comma-separated motions e.g. tpose,sixseven "
                        "(default: $G1_POLICY)")
    p.add_argument("--pause", type=float, default=1.0,
                   help="seconds to hold between chained motions (default 1.0)")
    p.add_argument("--list", action="store_true",
                   help="list envs, routines, policies and motions, then exit")
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
    a.add_argument("--episode", default=None, metavar="DIR", help="recorded run to replay (--policy replay)")
    a.add_argument("--max-steps", type=int, default=30, help="decision steps before giving up (default 30)")
    a.add_argument("--step-timeout", type=float, default=30.0,
                   help="wall seconds to wait for a decision (default 30)")
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

    if args.list:
        print("envs:     " + ", ".join(sorted(ENVS)))
        print("routines: " + ", ".join(sorted(ROUTINES)))
        print("policies: " + ", ".join(sorted(POLICIES)))
        print("motions:  " + ", ".join(sorted(MOTIONS)))
        print("agents:   " + ", ".join(sorted(AGENTS)) + "   (search --goal ..., replay --episode DIR)")
        print("skills:   (chain them like motions, e.g. --policy walk_forward:0.5,turn:45,arms_up)")
        print(describe_menu(list(SKILLS.values())))
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
                     f"motions: {', '.join(sorted(MOTIONS))}; skills: {', '.join(SKILLS)}")
    except (ValueError, RuntimeError) as e:
        parser.error(str(e))
    try:
        perceiver = build_perceiver(args, policy)
    except RuntimeError as e:
        parser.error(str(e))
    if args.vision == "api" and (args.env == "check"
                                 or (args.env == "sim" and args.headless and args.realtime is None)):
        print("warning: this run is not paced to realtime, so vision-model results will describe "
              "frames from well before they arrive; use --vision fake here, or --realtime 1 in sim")
    segments = getattr(policy, "segments", ())
    if any(getattr(s, "base", None) for s in segments) and not env.can_walk:
        parser.error(f"{policy.name} drives the base; {env.name} cannot walk here"
                     + (" (pass --walk after reading its pre-flight)" if args.env == "robot"
                        else " (drop --free-base)" if args.env == "sim" else ""))
    print(f"== {policy.name} @ {env.name} ==")
    if args.policy == "search" and args.max_time <= 120.0:
        print("hint: search is open-ended (6-10 s per step); raise --max-time, e.g. --max-time 600")
    ok = run(policy, env, max_time=args.max_time, perceiver=perceiver)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
