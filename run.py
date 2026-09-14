"""Run a policy in an environment.

    python run.py --env check --policy tpose
    mjpython run.py --env sim --policy tpose          # macOS viewer
    python run.py --env robot --policy tpose --iface eth0

``--env`` defaults to $G1_ENV, then "check", so ``G1_ENV=sim`` also works.
"""
from __future__ import annotations

import argparse
import os
import sys

from envs import ENVS, Env, EnvAbort
from policies import POLICIES
from policy import Policy


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="g1", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", "-e", choices=sorted(ENVS), default=os.environ.get("G1_ENV", "check"),
                   help="where to run the policy (default: $G1_ENV or 'check')")
    p.add_argument("--policy", "-p", choices=sorted(POLICIES), default=os.environ.get("G1_POLICY"),
                   help="which policy to run (default: $G1_POLICY)")
    p.add_argument("--list", action="store_true", help="list envs and policies, then exit")
    p.add_argument("--max-time", type=float, default=120.0,
                   help="abort if the policy runs longer than this many seconds (default 120)")
    for env_cls in ENVS.values():
        env_cls.add_args(p)
    return p


def run(policy: Policy, env: Env, max_time: float = 120.0) -> bool:
    """The one loop every stage shares."""
    with env:
        q = env.reset()
        policy.reset(q)
        t = 0.0
        try:
            while t <= max_time:
                action = policy.step(t, q)
                if action is None:
                    break
                q = env.step(action)
                t += policy.dt
            else:
                print(f"aborted: policy exceeded --max-time {max_time}s")
                return False
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
        print("policies: " + ", ".join(sorted(POLICIES)))
        return 0
    if args.policy is None:
        parser.error("--policy is required (or set $G1_POLICY)")

    policy = POLICIES[args.policy]()
    env = ENVS[args.env](args)
    print(f"== {policy.name} @ {env.name} ==")
    ok = run(policy, env, max_time=args.max_time)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
