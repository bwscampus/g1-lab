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

from config import CONTROL_DT, NUM_JOINTS, UPPER_BODY


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
    previous one ended; the very first starts from the pose observed at reset.

    Segments, joints and name can be given to the constructor or, in subclass
    style, as class attributes."""

    segments: Sequence[Segment] = ()

    def __init__(self, segments: Sequence[Segment] | None = None, *,
                 joints: list[int] | None = None, name: str | None = None) -> None:
        if segments is not None:
            self.segments = tuple(segments)
        if joints is not None:
            self.joints = list(joints)
        if name is not None:
            self.name = name

    @property
    def duration(self) -> float:
        return sum(seg.duration for seg in self.segments)

    def reset(self, q0: np.ndarray) -> None:
        allowed = set(self.joints)
        for seg in self.segments:
            if isinstance(seg.goal, dict):
                bad = sorted(set(seg.goal) - allowed)
                if bad:
                    raise ValueError(f"[{self.name}] segment {seg.label!r} sets joints {bad} "
                                     f"outside this policy's joints")
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


class Motion:
    """A reusable building block: a factory for a tuple of Segments, with NO
    takeover/handback bookends. Routines (see routines.py) concatenate motions
    and add the bookends once.

    Contract:
      * never use the "start" goal (reserved for the Takeover bookend)
      * the first segment should specify the motion's full entry pose so it is
        robust to whatever motion preceded it
      * a motion may end anywhere; the Handback bookend returns to STAND
    Parametrise via __init__ (e.g. ``TPose(hold=5.0)``, ``SixSeven(reps=3)``).
    """

    name: str = "motion"
    joints: list[int] = UPPER_BODY

    def segments(self) -> tuple[Segment, ...]:
        raise NotImplementedError

    @property
    def duration(self) -> float:
        return sum(seg.duration for seg in self.segments())
