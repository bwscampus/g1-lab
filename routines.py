"""Routines and other runnable policies.

A Routine is one SegmentPolicy whose segments are
    Takeover + motion_1 + pause + motion_2 + ... + Handback
so the arm_sdk bookends happen exactly once and every motion boundary is part
of the same continuous command stream (the check env's velocity gate therefore
covers every transition). Composition is deliberately done at the segment level
rather than by chaining Policy objects: chaining would restart each policy from
the env's *measured* state, which on the robot lags the command by gravity sag
and would produce a target jump at each boundary that the check stage cannot see.

A Selector is the camera-triggered counterpart: it idles at STAND until a
predicate on the latest frame fires, then runs one registered motion. It does
chain sub-policies, so it seeds each one from its own last *commanded* q.

Named routines live in ROUTINES, other named policies (camera examples) in
POLICIES; ad hoc routines come from ``--policy tpose,sixseven``.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Optional, Sequence

from camera import Frame
from motions import MOTIONS, Handback, Hold, SixSeven, Takeover, TPose
from policy import Action, Motion, Obs, Policy, Segment, SegmentPolicy
from vision import Look, red_blob


def motion_segments(part: Motion) -> list[Segment]:
    """A motion's segments with labels prefixed by its name. Only the Takeover
    bookend may use the reserved "start" goal."""
    out = []
    for seg in part.segments():
        if seg.goal == "start" and not isinstance(part, Takeover):
            raise ValueError(f"motion {part.name!r} uses the reserved 'start' goal")
        out.append(replace(seg, label=f"{part.name}: {seg.label}" if seg.label else ""))
    return out


class Routine(SegmentPolicy):
    def __init__(self, *motions: Motion, pause: float = 1.0, name: str | None = None) -> None:
        if not motions:
            raise ValueError("Routine needs at least one motion")
        parts: list[Motion] = [Takeover()]
        for i, m in enumerate(motions):
            if i > 0 and pause > 0:
                parts.append(Hold(pause))
            parts.append(m)
        parts.append(Handback())

        segments: list[Segment] = []
        joints: set[int] = set()
        for part in parts:
            joints.update(part.joints)
            segments.extend(motion_segments(part))

        self.motions = tuple(motions)
        super().__init__(segments, joints=sorted(joints),
                         name=name or "+".join(m.name for m in motions))


Rule = tuple[Callable[[Frame], bool], "str | Motion"]


class Selector(Policy):
    """Run a registered motion when a frame predicate fires.

    States: takeover (bookend) -> idle at STAND, evaluating ``rules`` on every
    new frame -> the first matching motion -> handback (``once``) or back to
    idle after ``cooldown`` seconds. Idle for ``timeout`` seconds ends the run.
    Every motion it can pick is an ordinary registered motion the check stage
    already validates; the only new boundaries are idle -> motion, and those
    start from STAND, which every motion enters from anyway.
    """

    name = "selector"
    uses_camera = True

    def __init__(self, rules: Sequence[Rule], *, timeout: float = 30.0, once: bool = True,
                 cooldown: float = 1.0, name: str | None = None) -> None:
        self.rules = [(pred, MOTIONS[m]() if isinstance(m, str) else m) for pred, m in rules]
        self.timeout = timeout
        self.once = once
        self.cooldown = cooldown
        joints: set[int] = set(Takeover().joints)
        for _, m in self.rules:
            joints.update(m.joints)
        self.joints = sorted(joints)
        if name is not None:
            self.name = name

    def reset(self, obs: Obs) -> None:
        self._cmd = obs.q.copy()
        self._last_seq: int | None = None
        self._sub: SegmentPolicy | None = None
        self._enter("takeover", 0.0, Takeover())

    def _enter(self, state: str, t: float, part: Motion | None = None) -> None:
        self._state, self._t0 = state, t
        if part is None:
            self._sub = None
            return
        segs = motion_segments(part)
        if state == "motion" and not self.once and self.cooldown > 0:
            segs.extend(motion_segments(Hold(self.cooldown)))
        self._sub = SegmentPolicy(segs, joints=self.joints, name=self.name)
        # Seed from the last *commanded* pose, never the measured one.
        self._sub.reset(Obs(self._cmd))

    def step(self, t: float, obs: Obs) -> Optional[Action]:
        if self._sub is not None:
            action = self._sub.step(t - self._t0, obs)
            if action is not None:
                self._cmd = action.q.copy()
                return action
            if self._state == "handback":
                return None
            if self._state == "motion" and self.once:
                self._enter("handback", t, Handback())
                return self.step(t, obs)
            self._enter("idle", t)
        if t - self._t0 >= self.timeout:
            print(f"[{self.name}] idle for {self.timeout:.0f}s, handing back")
            self._enter("handback", t, Handback())
            return self.step(t, obs)
        f = obs.frame
        if f is not None and f.seq != self._last_seq:
            self._last_seq = f.seq
            for pred, motion in self.rules:
                if pred(f):
                    print(f"[{self.name}] trigger -> {motion.name}")
                    self._enter("motion", t, motion)
                    return self.step(t, obs)
        return self.action(self._cmd.copy(), weight=1.0)


ROUTINES: dict[str, Callable[[], Routine]] = {
    "demo": lambda: Routine(TPose(), SixSeven(reps=3), name="demo"),
}

POLICIES: dict[str, Callable[[], Policy]] = {
    "look": lambda: Look(),
    "wave_on_red": lambda: Selector([(lambda f: red_blob(f.image) is not None, "sixseven")],
                                    name="wave_on_red"),
}


def build_policy(spec: str, pause: float = 1.0) -> Policy:
    """A registered routine or policy by name, or a comma-separated list of motion names."""
    if spec in ROUTINES:
        return ROUTINES[spec]()
    if spec in POLICIES:
        return POLICIES[spec]()
    names = [n.strip() for n in spec.split(",") if n.strip()]
    if not names:
        raise KeyError(spec)
    motions = []
    for n in names:
        if n not in MOTIONS:
            raise KeyError(n)
        motions.append(MOTIONS[n]())
    return Routine(*motions, pause=pause)
