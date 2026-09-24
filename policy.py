"""Policy interface.

A policy is a stateful function of time and the current observation (joint
state plus, optionally, the latest camera frame) that returns joint targets
for the joints it controls. It knows nothing about where those targets go
(check / sim / robot), so the same policy object runs unchanged through every
stage.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional, Sequence

import numpy as np

from camera import Frame
from config import CONTROL_DT, JOINT_HI, JOINT_LO, NUM_JOINTS, STAND_Q, UPPER_BODY

if TYPE_CHECKING:
    from perception import Perceiver, Percept


@dataclass
class Action:
    """One control tick's worth of targets.

    q:      full-length (29,) target vector. Only entries at ``joints`` are
            meaningful; everything else is ignored by the env.
    joints: indices the policy is actually commanding this tick.
    weight: arm_sdk blend in [0, 1]. 1.0 = policy owns the joints, 0.0 = the
            onboard controller owns them. sim/check emulate the same blend.
    kp/kd:  PD gains sent to the robot (sim uses the model's own actuators).
    base:   optional (vx, vy, vyaw) base velocity in m/s, m/s, rad/s (forward,
            left, counter-clockwise), or None for no base command. The robot
            walks it via LocoClient.Move (needs --walk), sim slides the pinned
            base, check enforces config.BASE_VEL_MAX.
    """

    q: np.ndarray
    joints: list[int]
    weight: float = 1.0
    kp: float = 60.0
    kd: float = 1.5
    base: Optional[tuple[float, float, float]] = None

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
    """What a policy sees each tick.

    q:         current joint state (29,)
    frame:     latest camera frame, or None when the env has no camera
    frame_age: seconds since that frame was captured, on the env's clock
               (inf when there is no frame). Frames are latest-only: the same
               Frame (same ``seq``) is seen every tick until a newer one
               arrives, so key per-frame work on ``frame.seq``.
    percept:   latest ``perception.Percept`` (what a vision model said about a
               recent frame), or None. Latest-only too: key on ``percept.seq``.
    percept_age: env-clock age of the frame that percept describes, so it
               includes the model's latency (inf when there is no percept).
    perceiver: the vision model handle, or None. Percepts are produced only on
               request (non-blocking; the result lands in a later obs).
    base_pose: the env's own (x, y, yaw) base estimate (sim integrates the
               commanded velocity; the robot has none yet).
    qd, tau:   measured joint velocity (rad/s) and torque (N m), or None when
               the env cannot measure them. Sim reads qvel / actuator_force,
               the robot LowState dq / tau_est. Read them; never gate on them.
    """

    q: np.ndarray
    frame: Optional[Frame] = None
    frame_age: float = math.inf
    percept: Optional["Percept"] = None     # latest vision-model description, if any
    percept_age: float = math.inf           # age of the frame it describes (latency included)
    perceiver: Optional["Perceiver"] = None # ask for a fresh percept with .request(frame)
    base_pose: Optional[tuple[float, float, float]] = None   # env's (x, y, yaw) estimate, if any
    qd: Optional[np.ndarray] = None          # measured joint velocity, if the env has it
    tau: Optional[np.ndarray] = None         # measured joint torque, if the env has it

    def __post_init__(self) -> None:
        self.q = np.asarray(self.q, dtype=float)


class Policy:
    """Base class. Subclass and implement ``reset`` and ``step``."""

    name: str = "policy"
    joints: list[int] = []          # joints this policy commands
    dt: float = CONTROL_DT
    kp: float = 60.0
    kd: float = 1.5
    uses_camera: bool = False       # the runner only opens a camera for policies that ask
    uses_vision: bool = False       # ... and only runs a vision model for policies that ask
    perceiver: Optional["Perceiver"] = None   # set by the runner before reset()

    def reset(self, obs: Obs) -> None:
        """Called once with the robot's current observation before the first step."""

    def step(self, t: float, obs: Obs) -> Optional[Action]:
        """Return the Action for time ``t`` (seconds since reset), or None when finished.

        Must not block: on the robot the whole tick is 20 ms. Do heavy per-frame
        work on another thread and read its latest result here."""
        raise NotImplementedError

    def close(self) -> None:
        """Called once by the runner after the env is torn down (also on Ctrl-C)."""

    def action(self, q: np.ndarray, weight: float = 1.0,
               base: Optional[tuple[float, float, float]] = None) -> Action:
        return Action(q=q, joints=list(self.joints), weight=weight, kp=self.kp, kd=self.kd,
                      base=base)


# --------------------------------------------------------------------------
# Segment helper: most scripted motions are "go from pose A to pose B over T
# seconds", chained. This turns such a list into a Policy.
# --------------------------------------------------------------------------

Pose = dict[int, float]
WeightFn = Callable[[float], float]
EPS = 1e-9          # phase-boundary tolerance against float drift in t = n * dt


def ease(a: float) -> float:
    """Smooth cosine ease, 0 -> 1."""
    return 0.5 - 0.5 * math.cos(math.pi * a)


@dataclass
class Segment:
    goal: Pose | str                  # target pose, or "start" (pose at reset)
    duration: float
    weight: WeightFn = field(default=lambda a: 1.0)
    label: str = ""
    base: Optional[tuple[float, float, float]] = None   # base velocity held for the whole segment


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

    def reset(self, obs: Obs) -> None:
        allowed = set(self.joints)
        for seg in self.segments:
            if isinstance(seg.goal, dict):
                bad = sorted(set(seg.goal) - allowed)
                if bad:
                    raise ValueError(f"[{self.name}] segment {seg.label!r} sets joints {bad} "
                                     f"outside this policy's joints")
        self._q0 = np.array(obs.q, dtype=float)
        self._start = {j: float(self._q0[j]) for j in self.joints}
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

    def step(self, t: float, obs: Obs) -> Optional[Action]:
        for t0, t1, start, goal, seg in self._plan:
            if t < t1 - 1e-9:
                a = ease((t - t0) / (t1 - t0))
                if seg.label and seg.label != self._last_label:
                    print(f"[{self.name}] {seg.label}")
                    self._last_label = seg.label
                out = self._q0.copy()
                for j in self.joints:
                    out[j] = start[j] + a * (goal[j] - start[j])
                return self.action(out, weight=seg.weight(a), base=seg.base)
        return None


# --------------------------------------------------------------------------
# Closed-loop helper: a policy driven by the camera, with the bookends and a
# safety envelope built in.
# --------------------------------------------------------------------------


class ReactivePolicy(Policy):
    """A closed-loop policy: implement ``track(t, obs) -> Pose``.

    A run can only validate the frames it is shown, so a reactive policy never
    trusts ``track``: every target is clipped to the joint limits minus
    ``margin`` and rate-limited to ``max_vel`` rad/s from the *last commanded*
    pose. The defaults sit inside the sim monitor's ``--margin`` / ``--max-vel``
    so whatever ``track`` returns becomes a command the monitor accepts.

    ``track`` runs once per new input, keyed on ``fresh(obs)``: by default the
    frame's seq while the frame is younger than ``stale_after`` (override
    ``fresh`` to key on ``obs.percept.seq`` instead). Between inputs, and
    whenever there is no fresh input, the last pose is held. Its result merges
    onto the current command, so it may set only some joints.

    Phases mirror the Takeover/Handback bookends: hold the reset pose while
    ramping weight 0->1 (``ramp``) -> STAND (``to_stand``) -> track for
    ``duration`` -> STAND (``to_stand``) -> hold while ramping 1->0 (``ramp``).
    ``finish()`` ends the tracking phase early. ``drive(t, obs)`` runs every
    tracking tick and its result is the action's base velocity (None outside
    tracking, so the base always stops before the return-to-stand phase).
    ``phase`` names the current phase.
    """

    name = "reactive"
    joints = UPPER_BODY
    uses_camera = True

    def __init__(self, duration: float = 10.0, *, ramp: float = 2.0, to_stand: float = 3.0,
                 margin: float = 0.05, max_vel: float = 3.0, stale_after: float = 0.5,
                 vision_refresh: float = 2.0, name: str | None = None) -> None:
        self.track_time = duration
        self.ramp = ramp
        self.to_stand = to_stand
        self.margin = margin
        self.max_vel = max_vel
        self.stale_after = stale_after
        self.vision_refresh = vision_refresh    # seconds between vision-model requests
        if name is not None:
            self.name = name

    @property
    def duration(self) -> float:
        return 2 * (self.ramp + self.to_stand) + self.track_time

    @property
    def cmd(self) -> np.ndarray:
        """The last commanded full q vector (read-only view for ``track``)."""
        return self._cmd

    def track(self, t: float, obs: Obs) -> Pose:
        """Targets for a new input; ``t`` is seconds since tracking began."""
        raise NotImplementedError

    def fresh(self, obs: Obs) -> Optional[int]:
        """Key of the input ``track`` should see; ``track`` runs once per distinct
        non-None key. Default: the frame seq while the frame is fresh."""
        f = obs.frame
        if f is None or obs.frame_age > self.stale_after:
            return None
        return f.seq

    def drive(self, t: float, obs: Obs) -> Optional[tuple[float, float, float]]:
        """Base velocity for this tracking tick, or None. Called every tick."""
        return None

    def finish(self) -> None:
        """End the tracking phase; the return-to-stand phase starts next tick."""
        self._finished = True

    def reset(self, obs: Obs) -> None:
        self._q0 = np.array(obs.q, dtype=float)
        self._cmd = self._q0.copy()
        self._stand = self._q0.copy()
        self._stand[self.joints] = STAND_Q[self.joints]
        self._held: Pose = {}
        self._last_key: int | None = None
        self._exit: np.ndarray | None = None
        self._last_label: str | None = None
        self._finished = False
        self._track_end: float | None = None
        self.phase = "takeover"
        from perception import VisionQuery
        self.vision = VisionQuery(self.vision_refresh)

    def _label(self, label: str) -> None:
        if label != self._last_label:
            print(f"[{self.name}] {label}")
            self._last_label = label

    def step(self, t: float, obs: Obs) -> Optional[Action]:
        r, s, d = self.ramp, self.to_stand, self.track_time
        if t < r - EPS:
            self.phase = "takeover"
            self._label("taking over (hold)")
            return self._emit(self._q0, weight=ease(t / r))
        t -= r
        if t < s - EPS:
            self.phase = "to_stand"
            self._label("moving to stand")
            return self._emit(self._q0 + ease(t / s) * (self._stand - self._q0))
        t -= s
        if self._track_end is None:
            if t < d - EPS and not self._finished:
                self.phase = "track"
                self._label("tracking")
                if self.uses_vision:
                    self.vision.poll(self.perceiver, obs, t)   # ask the model; never waits
                key = self.fresh(obs)
                if key is not None and key != self._last_key:
                    self._last_key = key
                    pose = self.track(t, obs)
                    bad = sorted(set(pose) - set(self.joints))
                    if bad:
                        raise ValueError(f"[{self.name}] track() set joints {bad} outside this "
                                         f"policy's joints")
                    self._held = dict(pose)
                target = self._cmd.copy()
                for j, v in self._held.items():
                    target[j] = v
                base = self.drive(t, obs)
                return self._emit(target, base=None if self._finished else base)
            self._track_end = t
        t -= self._track_end
        if t < s - EPS:
            self.phase = "return"
            self._label("returning to stand")
            if self._exit is None:
                self._exit = self._cmd.copy()
            return self._emit(self._exit + ease(t / s) * (self._stand - self._exit))
        t -= s
        if t < r - EPS:
            self.phase = "handback"
            self._label("handing back")
            return self._emit(self._stand, weight=1.0 - ease(t / r))
        return None

    def _emit(self, target: np.ndarray, weight: float = 1.0,
              base: Optional[tuple[float, float, float]] = None) -> Action:
        """Clip to the limits, rate-limit from the last command, and record it."""
        j = self.joints
        want = np.clip(target[j], JOINT_LO[j] + self.margin, JOINT_HI[j] - self.margin)
        step = self.max_vel * self.dt
        out = self._cmd.copy()
        out[j] = self._cmd[j] + np.clip(want - self._cmd[j], -step, step)
        self._cmd = out
        return self.action(out.copy(), weight=weight, base=base)
