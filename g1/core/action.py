"""What the executor passes around each 50 Hz tick: the observation it hands
to a program, the action it gets back, and the segment that describes one
piece of a motion.

    Obs        joint state plus the latest camera frame
    Action     joint targets for one tick, the arm_sdk weight, a base velocity
    Segment    "go to this pose over this many seconds" (optionally with a
               base velocity or a one-shot onboard call), the unit every
               tool's motion is written in
    Runnable   the contract the env runs: reset(obs) once, step(t, obs) until
               it returns None. A Tool, a Chain of tools and the Agent are all
               Runnables.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from g1.camera import Frame
from g1.core.config import CONTROL_DT, NUM_JOINTS


@dataclass
class Action:
    """One control tick's worth of targets.

    q:      full-length (29,) target vector. Only entries at ``joints`` are
            meaningful; everything else is ignored by the env.
    joints: indices the program is actually commanding this tick.
    weight: arm_sdk blend in [0, 1]. 1.0 = the program owns the joints, 0.0 =
            the onboard controller owns them. Sim emulates the same blend.
    kp/kd:  PD gains sent to the robot (sim uses the model's own actuators).
    base:   optional (vx, vy, vyaw) base velocity in m/s, m/s, rad/s (forward,
            left, counter-clockwise), or None for no base command. The robot
            walks it via LocoClient.Move (needs --walk), sim slides the pinned
            base, the monitor enforces config.BASE_VEL_MAX.
    command: optional one-shot onboard call, ``("WaveHand", {"turn_flag": False})``:
            the robot env calls that LocoClient method once (allow-listed in
            config.LOCO_METHODS); sim has no equivalent and aborts.
    """

    q: np.ndarray
    joints: list[int]
    weight: float = 1.0
    kp: float = 60.0
    kd: float = 1.5
    base: Optional[tuple[float, float, float]] = None
    command: Optional[tuple[str, dict]] = None

    def __post_init__(self) -> None:
        self.q = np.asarray(self.q, dtype=float)
        if self.q.shape != (NUM_JOINTS,):
            raise ValueError(f"Action.q must have shape ({NUM_JOINTS},), got {self.q.shape}")
        if self.base is not None:
            b = tuple(float(v) for v in self.base)
            if len(b) != 3 or not all(math.isfinite(v) for v in b):
                raise ValueError(f"Action.base must be 3 finite floats, got {self.base!r}")
            self.base = b


@dataclass
class Obs:
    """What a program sees each tick.

    q:         current joint state (29,)
    frame:     latest camera frame, or None when the env has no camera
    frame_age: seconds since that frame was captured, on the env's clock
               (inf when there is no frame). Frames are latest-only: the same
               Frame (same ``seq``) is seen every tick until a newer one
               arrives, so key per-frame work on ``frame.seq``.
    base_pose: the env's own (x, y, yaw) base estimate (sim integrates the
               commanded velocity; the robot has none yet).
    qd, tau:   measured joint velocity (rad/s) and torque (N m), or None when
               the env cannot measure them. Sim reads qvel / actuator_force,
               the robot LowState dq / tau_est. Read them; never gate on them.
    """

    q: np.ndarray
    frame: Optional[Frame] = None
    frame_age: float = math.inf
    base_pose: Optional[tuple[float, float, float]] = None
    qd: Optional[np.ndarray] = None
    tau: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.q = np.asarray(self.q, dtype=float)


class Runnable:
    """The contract the env runs. Subclass and implement ``reset`` and ``step``."""

    name: str = "runnable"
    joints: list[int] = []          # joints this program commands
    dt: float = CONTROL_DT
    kp: float = 60.0
    kd: float = 1.5
    uses_camera: bool = False       # the runner only opens a camera for programs that ask

    def reset(self, obs: Obs) -> None:
        """Called once with the robot's current observation before the first step."""

    def step(self, t: float, obs: Obs) -> Optional[Action]:
        """Return the Action for time ``t`` (seconds since reset), or None when finished.

        Must not block: on the robot the whole tick is 20 ms. Do heavy per-frame
        work on another thread and read its latest result here."""
        raise NotImplementedError

    def close(self) -> None:
        """Called once by the runner after the env is torn down (also on Ctrl-C)."""

    def on_interrupt(self, reason: str, detail: str = "") -> None:
        """The runner is about to return the robot to a safe state because of
        ``reason`` (ctrl_c | max_time | error). Record it; do not act."""

    def on_returned(self, outcome: str) -> None:
        """The safe return finished with ``outcome`` (completed | failed)."""

    def action(self, q: np.ndarray, weight: float = 1.0,
               base: Optional[tuple[float, float, float]] = None,
               command: Optional[tuple[str, dict]] = None) -> Action:
        return Action(q=q, joints=list(self.joints), weight=weight, kp=self.kp, kd=self.kd,
                      base=base, command=command)


# --------------------------------------------------------------------------
# Segments: "go from pose A to pose B over T seconds", chained
# --------------------------------------------------------------------------

Pose = dict[int, float]
WeightFn = Callable[[float], float]
EPS = 1e-9          # phase-boundary tolerance against float drift in t = n * dt


def ease(a: float) -> float:
    """Smooth cosine ease, 0 -> 1."""
    return 0.5 - 0.5 * math.cos(math.pi * a)


@dataclass
class Segment:
    goal: Pose | str                  # target pose (merged onto the previous one), or "start" (pose at reset)
    duration: float
    weight: WeightFn = field(default=lambda a: 1.0)
    label: str = ""
    base: Optional[tuple[float, float, float]] = None   # base velocity held for the whole segment
    command: Optional[tuple[str, dict]] = None          # one-shot onboard call on the segment's first tick
