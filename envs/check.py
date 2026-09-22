"""Stage 1: naive sanity check. No hardware, no GUI, no physics.

Runs the policy open-loop from the stand pose and verifies, per tick:

  * targets are finite and joint indices are valid
  * arm_sdk weight is in [0, 1]
  * every commanded target is inside the joint limits minus ``--margin``
  * the *effective* command (blend of hold pose and target by weight) never
    moves faster than ``--max-vel`` rad/s between ticks

There is no camera unless asked for: ``--camera-dir`` replays recorded frames
and ``--camera-noise`` feeds random ones, both on the check clock, so a camera
policy can be run against known input or fuzzed.

Exits non-zero on any violation.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from camera import DirCamera, NoiseCamera
from config import CONTROL_DT, JOINT_HI, JOINT_LO, JOINT_NAMES, NUM_JOINTS, STAND_Q
from policy import Action
from envs.base import Env, EnvAbort


@dataclass
class Violation:
    t: float
    joint: int
    kind: str
    value: float
    limit: float

    def __str__(self) -> str:
        return (f"t={self.t:6.2f}s  {JOINT_NAMES[self.joint]:<22} {self.kind:<8} "
                f"{self.value:+.3f}  (limit {self.limit:+.3f})")


class CheckEnv(Env):
    name = "check"

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("check")
        g.add_argument("--margin", type=float, default=0.05,
                       help="safety margin inside the joint limits, rad (default 0.05)")
        g.add_argument("--max-vel", type=float, default=4.0,
                       help="max allowed joint speed of the effective command, rad/s (default 4.0)")
        g.add_argument("--max-violations", type=int, default=20,
                       help="stop after this many violations (default 20)")
        g.add_argument("--camera-dir", type=Path, default=None, metavar="DIR",
                       help="replay the image files in DIR (sorted by name) as the camera feed")
        g.add_argument("--camera-noise", action="store_true",
                       help="feed random frames, to fuzz a camera policy")
        g.add_argument("--camera-fps", type=float, default=15.0,
                       help="frame rate of --camera-dir / --camera-noise (default 15)")

    def setup(self) -> None:
        self.violations: list[Violation] = []
        self.ticks = 0
        self.t = 0.0
        self.peak_vel = np.zeros(NUM_JOINTS)
        self.q_min = np.full(NUM_JOINTS, np.inf)
        self.q_max = np.full(NUM_JOINTS, -np.inf)
        self.camera = None
        if self.args.camera_dir is not None:
            self.camera = DirCamera(self.args.camera_dir, fps=self.args.camera_fps)
        elif self.args.camera_noise:
            self.camera = NoiseCamera(fps=self.args.camera_fps)
        if self.camera is not None:
            self.camera.start()

    def clock(self) -> float:
        return self.t

    def frame(self):
        return None if self.camera is None else self.camera.poll(self.t)

    def reset(self) -> np.ndarray:
        self.hold = STAND_Q.copy()
        self.q = STAND_Q.copy()
        self.prev_cmd = STAND_Q.copy()
        return self.q.copy()

    def step(self, action: Action) -> np.ndarray:
        dt = CONTROL_DT
        a = self.args
        joints = list(action.joints)

        if any(j < 0 or j >= NUM_JOINTS for j in joints):
            raise ValueError(f"policy commands invalid joint index: {joints}")
        if not (0.0 <= action.weight <= 1.0):
            self.violations.append(Violation(self.t, -1, "weight", action.weight, 1.0))

        tgt = action.q
        if not np.all(np.isfinite(tgt[joints])):
            raise ValueError(f"policy produced non-finite target at t={self.t:.2f}s")

        # Bounds on the raw target (what the policy asks for).
        lo, hi = JOINT_LO + a.margin, JOINT_HI - a.margin
        for j in joints:
            if tgt[j] < lo[j]:
                self.violations.append(Violation(self.t, j, "below", tgt[j], lo[j]))
            elif tgt[j] > hi[j]:
                self.violations.append(Violation(self.t, j, "above", tgt[j], hi[j]))

        # Effective command after the arm_sdk blend, and its speed.
        cmd = self.hold.copy()
        cmd[joints] = (1.0 - action.weight) * self.hold[joints] + action.weight * tgt[joints]
        vel = np.abs(cmd - self.prev_cmd) / dt
        self.peak_vel = np.maximum(self.peak_vel, vel)
        for j in joints:
            if vel[j] > a.max_vel:
                self.violations.append(Violation(self.t, j, "velocity", vel[j], a.max_vel))
        self.q_min[joints] = np.minimum(self.q_min[joints], cmd[joints])
        self.q_max[joints] = np.maximum(self.q_max[joints], cmd[joints])

        if len(self.violations) >= a.max_violations:
            raise EnvAbort(f"too many violations ({len(self.violations)}), stopping early")

        self.prev_cmd = cmd
        self.q = cmd            # perfect tracking: next state is the command
        self.t += dt
        self.ticks += 1
        return self.q.copy()

    def report(self) -> bool:
        used = [j for j in range(NUM_JOINTS) if np.isfinite(self.q_min[j])]
        print(f"\ncheck: {self.ticks} ticks, {self.t:.2f} s, {len(used)} joints commanded")
        print(f"{'joint':<22} {'min':>8} {'max':>8} {'peak vel':>9}   limits")
        for j in used:
            print(f"{JOINT_NAMES[j]:<22} {self.q_min[j]:8.3f} {self.q_max[j]:8.3f} "
                  f"{self.peak_vel[j]:9.2f}   [{JOINT_LO[j]:+.3f}, {JOINT_HI[j]:+.3f}]")
        if self.violations:
            print(f"\nFAIL: {len(self.violations)} violation(s)")
            for v in self.violations:
                print("  " + str(v))
            return False
        print("\nPASS: all targets in bounds")
        return True
