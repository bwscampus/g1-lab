"""The safety check every run carries: what the policy commanded, and where the
joints actually went.

`JointMonitor` watches a run tick by tick and reports afterwards:

  * the **measured** angle of every joint stayed inside its limits (minus
    ``margin``) — the check that matters, because physics, contact and gravity
    sag can put a joint somewhere the command never asked for
  * every commanded target was inside those limits too
  * the *effective* command (the arm_sdk blend of hold pose and target) never
    moved faster than ``max_vel`` rad/s between ticks
  * the arm_sdk weight stayed in [0, 1] (with the target gate) and a base
    velocity stayed within ``config.BASE_VEL_MAX``; the base pose is integrated
    kinematically

Sim runs it strictly (the run fails and stops early on too many violations).
The robot runs it report-only over the measured angles: teardown already
releases the arms, and stopping mid-run is its own risk.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from config import (BASE_VEL_MAX, CONTROL_DT, JOINT_HI, JOINT_LO, JOINT_NAMES, NUM_JOINTS,
                    STAND_Q)
from policy import Action

MAX_PRINTED = 12


@dataclass
class Violation:
    t: float
    joint: int
    kind: str
    value: float
    limit: float

    def __str__(self) -> str:
        name = JOINT_NAMES[self.joint] if self.joint >= 0 else "-"
        return (f"t={self.t:6.2f}s  {name:<22} {self.kind:<13} "
                f"{self.value:+.3f}  (limit {self.limit:+.3f})")


class TooManyViolations(Exception):
    """Raised by the monitor when a strict run should stop early."""


class JointMonitor:
    def __init__(self, *, margin: float = 0.05, max_vel: float = 4.0, max_violations: int = 20,
                 strict: bool = True, gate_targets: bool = True, gate_velocity: bool = True,
                 gate_base: bool = True) -> None:
        self.margin = margin
        self.max_vel = max_vel
        self.max_violations = max_violations
        self.strict = strict
        self.gate_targets = gate_targets
        self.gate_velocity = gate_velocity
        self.gate_base = gate_base
        self.reset()

    def reset(self, hold: np.ndarray | None = None) -> None:
        self.violations: list[Violation] = []
        self.ticks = 0
        self.t = 0.0
        self.q_min = np.full(NUM_JOINTS, np.inf)      # measured
        self.q_max = np.full(NUM_JOINTS, -np.inf)
        self.cmd_min = np.full(NUM_JOINTS, np.inf)    # commanded target
        self.cmd_max = np.full(NUM_JOINTS, -np.inf)
        self.peak_vel = np.zeros(NUM_JOINTS)
        self.commanded: set[int] = set()
        self.hold = STAND_Q.copy() if hold is None else np.array(hold, dtype=float)
        self.prev_cmd = self.hold.copy()
        self.base_pose_xy_yaw = np.zeros(3)
        self.base_path = 0.0
        self.base_ticks = 0
        self.base_peak = np.zeros(3)

    @property
    def lo(self) -> np.ndarray:
        return JOINT_LO + self.margin

    @property
    def hi(self) -> np.ndarray:
        return JOINT_HI - self.margin

    def _flag(self, joint: int, kind: str, value: float, limit: float) -> None:
        self.violations.append(Violation(self.t, joint, kind, float(value), float(limit)))
        if self.strict and len(self.violations) >= self.max_violations:
            raise TooManyViolations(f"too many violations ({len(self.violations)}), stopping early")

    def observe(self, q: np.ndarray, action: Action | None = None, dt: float = CONTROL_DT) -> None:
        """One tick: the measured joint angles, and the action that produced them."""
        q = np.asarray(q, dtype=float)
        if not np.all(np.isfinite(q)):
            raise ValueError(f"non-finite joint state at t={self.t:.2f}s")
        self.q_min = np.minimum(self.q_min, q)
        self.q_max = np.maximum(self.q_max, q)
        lo, hi = self.lo, self.hi
        for j in range(NUM_JOINTS):
            if q[j] < lo[j]:
                self._flag(j, "below", q[j], lo[j])
            elif q[j] > hi[j]:
                self._flag(j, "above", q[j], hi[j])

        if action is not None:
            joints = list(action.joints)
            if any(j < 0 or j >= NUM_JOINTS for j in joints):
                raise ValueError(f"policy commands invalid joint index: {joints}")
            self.commanded.update(joints)
            tgt = np.asarray(action.q, dtype=float)
            if not np.all(np.isfinite(tgt[joints])):
                raise ValueError(f"policy produced non-finite target at t={self.t:.2f}s")
            if self.gate_targets and not (0.0 <= action.weight <= 1.0):
                self._flag(-1, "weight", action.weight, 1.0)
            if self.gate_base and action.base is not None:
                self._base(action.base, dt)
            self.cmd_min[joints] = np.minimum(self.cmd_min[joints], tgt[joints])
            self.cmd_max[joints] = np.maximum(self.cmd_max[joints], tgt[joints])
            if self.gate_targets:
                for j in joints:
                    if tgt[j] < lo[j]:
                        self._flag(j, "target_below", tgt[j], lo[j])
                    elif tgt[j] > hi[j]:
                        self._flag(j, "target_above", tgt[j], hi[j])
            # the effective command after the arm_sdk blend, and its speed
            cmd = self.hold.copy()
            cmd[joints] = (1.0 - action.weight) * self.hold[joints] + action.weight * tgt[joints]
            vel = np.abs(cmd - self.prev_cmd) / dt
            self.peak_vel = np.maximum(self.peak_vel, vel)
            if self.gate_velocity:
                for j in joints:
                    if vel[j] > self.max_vel:
                        self._flag(j, "velocity", vel[j], self.max_vel)
            self.prev_cmd = cmd

        self.t += dt
        self.ticks += 1

    def _base(self, base, dt: float) -> None:
        v = np.asarray(base, dtype=float)
        for i, kind in enumerate(("base_vx", "base_vy", "base_vyaw")):
            if abs(v[i]) > BASE_VEL_MAX[i]:
                self._flag(-1, kind, v[i], BASE_VEL_MAX[i])
        self.base_peak = np.maximum(self.base_peak, np.abs(v))
        x, y, yaw = self.base_pose_xy_yaw
        yaw += v[2] * dt
        x += (math.cos(yaw) * v[0] - math.sin(yaw) * v[1]) * dt
        y += (math.sin(yaw) * v[0] + math.cos(yaw) * v[1]) * dt
        self.base_pose_xy_yaw = np.array([x, y, yaw])
        self.base_path += math.hypot(v[0], v[1]) * dt
        self.base_ticks += 1

    def base_pose(self) -> tuple[float, float, float]:
        return tuple(float(v) for v in self.base_pose_xy_yaw)

    # -- reporting ---------------------------------------------------------
    def rows(self) -> list[int]:
        """Joints worth printing: commanded, violated, or actually moved."""
        moved = np.isfinite(self.q_min) & ((self.q_max - self.q_min) > 5e-3)
        out = set(self.commanded) | {v.joint for v in self.violations if v.joint >= 0}
        out |= {j for j in range(NUM_JOINTS) if moved[j]}
        return sorted(out)

    def report(self, label: str = "run") -> bool:
        rows = self.rows()
        print(f"\n{label}: {self.ticks} ticks, {self.t:.2f} s, {len(self.commanded)} joints commanded")
        if rows:
            print(f"{'joint':<22} {'measured min':>12} {'max':>8} {'cmd min':>9} {'max':>8} "
                  f"{'peak vel':>9}   limits")
            for j in rows:
                mark = "!" if (np.isfinite(self.q_min[j]) and
                               (self.q_min[j] < self.lo[j] or self.q_max[j] > self.hi[j])) else " "
                cmd_lo = f"{self.cmd_min[j]:9.3f}" if np.isfinite(self.cmd_min[j]) else f"{'-':>9}"
                cmd_hi = f"{self.cmd_max[j]:8.3f}" if np.isfinite(self.cmd_max[j]) else f"{'-':>8}"
                print(f"{JOINT_NAMES[j]:<22} {self.q_min[j]:12.3f} {self.q_max[j]:8.3f} "
                      f"{cmd_lo} {cmd_hi} {self.peak_vel[j]:9.2f} {mark} "
                      f"[{JOINT_LO[j]:+.3f}, {JOINT_HI[j]:+.3f}]")
        if self.base_ticks:
            x, y, yaw = self.base_pose_xy_yaw
            print(f"base: {self.base_ticks} ticks commanded, path {self.base_path:.2f} m, ended at "
                  f"({x:+.2f}, {y:+.2f}) m yaw {math.degrees(yaw):+.0f} deg; peak "
                  f"vx {self.base_peak[0]:.2f} vy {self.base_peak[1]:.2f} vyaw {self.base_peak[2]:.2f}")
        if self.violations:
            head = "FAIL" if self.strict else "WARNING"
            print(f"\n{head}: {len(self.violations)} violation(s)")
            for v in self.violations[:MAX_PRINTED]:
                print("  " + str(v))
            if len(self.violations) > MAX_PRINTED:
                kinds = sorted({v.kind for v in self.violations})
                print(f"  ... and {len(self.violations) - MAX_PRINTED} more ({', '.join(kinds)})")
            return not self.strict
        print("\nPASS: every joint stayed in bounds")
        return True
