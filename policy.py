"""Policy interface.

A policy is a stateful function of time and the current joint state that
returns joint targets for the joints it controls. It knows nothing about
where those targets go (check / sim / robot), so the same policy object runs
unchanged through every stage.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from config import CONTROL_DT, NUM_JOINTS


@dataclass
class Action:
    """One control tick's worth of targets.

    q:      full-length (29,) target vector. Only entries at ``joints`` are
            meaningful; everything else is ignored by the env.
    joints: indices the policy is actually commanding this tick.
    weight: arm_sdk blend in [0, 1]. 1.0 = policy owns the joints, 0.0 = the
            onboard controller owns them. sim/check emulate the same blend.
    kp/kd:  PD gains sent to the robot (sim uses the model's own actuators).
    """

    q: np.ndarray
    joints: list[int]
    weight: float = 1.0
    kp: float = 60.0
    kd: float = 1.5

    def __post_init__(self) -> None:
        self.q = np.asarray(self.q, dtype=float)
        if self.q.shape != (NUM_JOINTS,):
            raise ValueError(f"Action.q must have shape ({NUM_JOINTS},), got {self.q.shape}")


class Policy:
    """Base class. Subclass and implement ``reset`` and ``step``."""

    name: str = "policy"
    joints: list[int] = []          # joints this policy commands
    dt: float = CONTROL_DT
    kp: float = 60.0
    kd: float = 1.5

    def reset(self, q0: np.ndarray) -> None:
        """Called once with the robot's current joint state before the first step."""

    def step(self, t: float, q: np.ndarray) -> Optional[Action]:
        """Return the Action for time ``t`` (seconds since reset), or None when finished."""
        raise NotImplementedError

    def action(self, q: np.ndarray, weight: float = 1.0) -> Action:
        return Action(q=q, joints=list(self.joints), weight=weight, kp=self.kp, kd=self.kd)


# --------------------------------------------------------------------------
# Segment helper: most scripted motions are "go from pose A to pose B over T
# seconds", chained. This turns such a list into a Policy.
# --------------------------------------------------------------------------

Pose = dict[int, float]
WeightFn = Callable[[float], float]


def ease(a: float) -> float:
    """Smooth cosine ease, 0 -> 1."""
    return 0.5 - 0.5 * math.cos(math.pi * a)


@dataclass
class Segment:
    goal: Pose | str                  # target pose, or "start" (pose at reset)
    duration: float
    weight: WeightFn = field(default=lambda a: 1.0)
    label: str = ""


class SegmentPolicy(Policy):
    """Chain of interpolated pose segments. Each segment starts from where the
    previous one ended; the very first starts from the pose observed at reset."""

    segments: Sequence[Segment] = ()

    def reset(self, q0: np.ndarray) -> None:
        self._q0 = np.array(q0, dtype=float)
        self._start = {j: float(q0[j]) for j in self.joints}
        # Resolve each segment's start/goal into concrete poses.
        self._plan: list[tuple[float, float, Pose, Pose, Segment]] = []
        t = 0.0
        prev = dict(self._start)
        for seg in self.segments:
            goal = dict(self._start) if seg.goal == "start" else {**prev, **seg.goal}
            self._plan.append((t, t + seg.duration, prev, goal, seg))
            prev = goal
            t += seg.duration
        self.total_time = t
        self._last_label: str | None = None

    def step(self, t: float, q: np.ndarray) -> Optional[Action]:
        for t0, t1, start, goal, seg in self._plan:
            if t < t1 - 1e-9:
                a = ease((t - t0) / (t1 - t0))
                if seg.label and seg.label != self._last_label:
                    print(f"[{self.name}] {seg.label}")
                    self._last_label = seg.label
                out = self._q0.copy()
                for j in self.joints:
                    out[j] = start[j] + a * (goal[j] - start[j])
                return self.action(out, weight=seg.weight(a))
        return None
