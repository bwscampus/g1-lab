"""Policies built on Targets.

  Face(target)   turn the waist toward the target; arms stay at STAND
  GoTo(target)   the top-level "walk to it" loop: turn to face, step forward,
                 hold when the target is lost, done when target.reached() says
                 so (confirmed twice) or the timeout passes. Steers with the
                 base (Action.base) only; the waist stays at 0 so camera bearing
                 is base bearing.

Both remember the commanded yaw per frame seq: a sighting (and a fortiori a
vision-model percept) lands after its frame, so bearings are applied relative
to the yaw at capture, not the current one.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Optional

import numpy as np

from config import CONTROL_DT, UPPER_BODY, joint_index
from policy import Obs, Pose, ReactivePolicy
from targets import RedDot, Salient, Target

WAIST_YAW = joint_index("waist_yaw")


class YawLog:
    """Commanded yaw per frame seq, bounded."""

    def __init__(self, cap: int = 300) -> None:
        self._d: "OrderedDict[int, float]" = OrderedDict()
        self.cap = cap

    def record(self, seq: int, yaw: float) -> None:
        self._d[seq] = yaw
        while len(self._d) > self.cap:
            self._d.popitem(last=False)

    def get(self, seq: int, default: float) -> float:
        return self._d.get(seq, default)


class Face(ReactivePolicy):
    """Turn the waist toward ``target``. ``narrate`` prints the vision model's
    summary whenever it changes."""

    def __init__(self, target: Target, duration: float = 15.0, *, gain: float = 0.6,
                 yaw_max: float = 0.8, narrate: bool = False, percept_stale_after: float = 6.0,
                 name: Optional[str] = None, **kw) -> None:
        super().__init__(duration, name=name or f"face_{target.name}", **kw)
        self.target = target
        self.uses_vision = target.uses_vision
        self.gain = gain
        self.yaw_max = yaw_max
        self.narrate = narrate
        self.percept_stale_after = percept_stale_after

    def reset(self, obs: Obs) -> None:
        super().reset(obs)
        self.target.reset()
        self._yaws = YawLog()
        self._last_frame_seq: Optional[int] = None
        self._last_summary: Optional[str] = None

    def fresh(self, obs: Obs) -> Optional[int]:
        f = obs.frame
        if f is not None and f.seq != self._last_frame_seq:
            self._last_frame_seq = f.seq
            self._yaws.record(f.seq, float(self.cmd[WAIST_YAW]))
        k = self.target.key(obs)
        if k is None:
            return None
        limit = self.percept_stale_after if self.target.uses_vision else self.stale_after
        return None if self.target.input_age(obs) > limit else k

    def track(self, t: float, obs: Obs) -> Pose:
        if self.narrate and obs.percept is not None and obs.percept.summary != self._last_summary:
            self._last_summary = obs.percept.summary
            print(f"[{self.name}] {obs.percept.summary} | path "
                  f"{'clear' if obs.percept.path_clear else 'blocked'}")
        s = self.target.update(obs, t)
        if s is None:
            return {}                       # not in view: hold
        yaw = self._yaws.get(s.frame_seq, float(self.cmd[WAIST_YAW])) - self.gain * s.bearing
        return {WAIST_YAW: float(np.clip(yaw, -self.yaw_max, self.yaw_max))}


class Look(Face):
    """Face the red dot (no vision model needed)."""

    def __init__(self, duration: float = 15.0, **kw) -> None:
        super().__init__(RedDot(), duration, name="look", **kw)


class Describe(Face):
    """Face whatever the vision model finds most salient, narrating the scene."""

    def __init__(self, duration: float = 30.0, **kw) -> None:
        kw.setdefault("narrate", True)
        super().__init__(Salient(), duration, name="describe", **kw)


class GoTo(ReactivePolicy):
    """Walk to ``target``. Phases: takeover -> seek (drive the base) -> return to
    stand -> handback. In seek, each tick: update the target; if it reports
    reached on ``confirm`` consecutive sightings, stop and finish; if it was
    seen within ``lost_after`` seconds, turn toward it (``k_yaw``) and, once
    roughly facing it (``face_within``), walk at ``v_fwd``; otherwise hold
    still and wait to reacquire. Velocities are rate-limited by ``accel`` /
    ``yaw_accel`` and never exceed the env's BASE_VEL_MAX. ``timeout`` is the
    seek phase's length.

    The base yaw is integrated from the commanded rate and logged per frame
    seq so a late sighting is applied relative to the yaw at capture."""

    joints = UPPER_BODY

    def __init__(self, target: Target, timeout: float = 60.0, *, v_fwd: float = 0.2,
                 k_yaw: float = 1.0, vyaw_max: float = 0.4, face_within: float = 0.35,
                 accel: float = 0.3, yaw_accel: float = 1.0, lost_after: float = 1.0,
                 confirm: int = 2, name: Optional[str] = None, **kw) -> None:
        if not target.can_reach:
            raise ValueError(f"{target.name} cannot tell when it is reached; GoTo refuses a "
                             f"target with no stop condition (use Face)")
        super().__init__(timeout, name=name or f"goto_{target.name}", **kw)
        self.target = target
        self.uses_vision = target.uses_vision
        self.v_fwd, self.k_yaw, self.vyaw_max = v_fwd, k_yaw, vyaw_max
        self.face_within, self.accel, self.yaw_accel = face_within, accel, yaw_accel
        self.lost_after, self.confirm = lost_after, confirm

    def reset(self, obs: Obs) -> None:
        super().reset(obs)
        self.target.reset()
        self._base = np.zeros(3)
        self._yaw_cmd = 0.0
        self._yaws = YawLog()
        self._last_frame_seq: Optional[int] = None
        self._hits = 0
        self._state: Optional[str] = None
        self.reached = False

    def _say(self, state: str) -> None:
        if state != self._state:
            self._state = state
            print(f"[{self.name}] {state}")

    def fresh(self, obs: Obs) -> Optional[int]:
        return None                         # the arms never move; steering is in drive()

    def track(self, t: float, obs: Obs) -> Pose:
        return {}

    def drive(self, t: float, obs: Obs):
        f = obs.frame
        if f is not None and f.seq != self._last_frame_seq:
            self._last_frame_seq = f.seq
            self._yaws.record(f.seq, self._yaw_cmd)
        s = self.target.update(obs, t)
        if s is not None:
            self._hits = self._hits + 1 if self.target.reached(s) else 0
        if self._hits >= self.confirm:
            self._say("reached")
            self.reached = True
            self._base[:] = 0.0
            self.finish()
            return None
        if self.target.age(t) > self.lost_after:
            want = np.zeros(3)
            self._say("lost, holding" if self.target.last is not None else "searching")
        else:
            last = self.target.last
            # bearing now = bearing at capture + how far the base has turned since
            b = last.bearing + (self._yaw_cmd - self._yaws.get(last.frame_seq, self._yaw_cmd))
            vyaw = float(np.clip(-self.k_yaw * b, -self.vyaw_max, self.vyaw_max))
            vx = self.v_fwd * max(0.0, math.cos(b)) if abs(b) <= self.face_within else 0.0
            want = np.array([vx, 0.0, vyaw])
            self._say("driving" if vx > 0 else "turning")
        lim = np.array([self.accel, self.accel, self.yaw_accel]) * CONTROL_DT
        self._base += np.clip(want - self._base, -lim, lim)
        self._yaw_cmd += self._base[2] * CONTROL_DT
        return (float(self._base[0]), float(self._base[1]), float(self._base[2]))
