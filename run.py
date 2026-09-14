"""Run a routine in an environment.

    python run.py --env check --policy tpose
    python run.py --env check --policy tpose,sixseven   # ad hoc chain of motions
    python run.py --env check --policy demo             # registered routine
    mjpython run.py --env sim --policy demo             # macOS viewer
    python run.py --env robot --policy demo --iface eth0 --mode standing

``--policy`` is a registered routine name or a comma-separated list of motions.
``--env`` defaults to $G1_ENV, then "check", so ``G1_ENV=sim`` also works.
"""
from __future__ import annotations

import argparse
import os
import sys

from envs import ENVS, Env, EnvAbort
from motions import MOTIONS
from policy import Policy
from routines import ROUTINES, build_policy


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
    p.add_argument("--list", action="store_true", help="list envs, routines and motions, then exit")
    p.add_argument("--max-time", type=float, default=120.0,
                   help="abort if the policy runs longer than this many seconds (default 120)")
    for env_cls in ENVS.values():
        env_cls.add_args(p)
    return p


def run(policy: Policy, env: Env, max_time: float = 120.0) -> bool:
    """The one loop every stage shares."""
    duration = getattr(policy, "duration", None)
    if duration is not None and duration > max_time:
        print(f"warning: {policy.name} lasts {duration:.1f}s but --max-time is {max_time:.0f}s; "
              f"it will be cut short")
    with env:
        q = env.reset()
        policy.reset(q)
        n = 0
        try:
            while True:
                t = n * policy.dt
                if t > max_time:
                    print(f"aborted: policy exceeded --max-time {max_time}s")
                    return False
                action = policy.step(t, q)
                if action is None:
                    break
                q = env.step(action)
                n += 1
        except EnvAbort as e:
            print(f"\n{env.name}: {e}")
        except KeyboardInterrupt:
            print("\ninterrupted")
            return False
    return env.report()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        print("envs:     " + ", ".join(sorted(ENVS)))
        print("routines: " + ", ".join(sorted(ROUTINES)))
        print("motions:  " + ", ".join(sorted(MOTIONS)))
        return 0
    if args.policy is None:
        parser.error("--policy is required (or set $G1_POLICY)")

    try:
        policy = build_policy(args.policy, pause=args.pause)
    except KeyError as e:
        parser.error(f"unknown policy {e}; routines: {', '.join(sorted(ROUTINES))}; "
                     f"motions: {', '.join(sorted(MOTIONS))}")
    env = ENVS[args.env](args)
    print(f"== {policy.name} @ {env.name} ==")
    ok = run(policy, env, max_time=args.max_time)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
