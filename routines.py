"""Routines and other runnable policies.

A Routine is one SegmentPolicy whose segments are
    Takeover + skill_1 + pause + skill_2 + ... + Handback
so the arm_sdk bookends happen exactly once and every skill boundary is part of
the same continuous command stream (the monitor's velocity gate therefore covers
every transition). Composition is deliberately done at the segment level rather
than by chaining Policy objects: chaining would restart each policy from the
env's *measured* state, which on the robot lags the command by gravity sag and
would produce a target jump at each boundary that no gate can see.

A Selector is the camera-triggered counterpart: it idles at STAND until a
predicate on the observation (latest frame and/or vision-model percept) fires,
then runs one registered skill. It does chain sub-policies, so it seeds each one
from its own last *commanded* q.

Named routines live in ROUTINES, other named policies (camera examples) in
POLICIES; ad hoc ones come from ``--policy tpose,turn:45,sixseven``.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

from camera import Frame
from policy import Action, Obs, Policy, Segment, SegmentPolicy
from behaviors import Describe, Face, GoTo, Look
from perception import VisionQuery
from skills import (SKILLS, Handback, Hold, SixSeven, Skill, Takeover, TPose, parse_skill,
                    skill_segments, split_outside)
from targets import Doorway, Labeled, RedDot, seen


class Routine(SegmentPolicy):
    """Bookends once, then the given skills in order with a pause between."""

    def __init__(self, *skills: Skill, pause: float = 1.0, name: str | None = None) -> None:
        if not skills:
            raise ValueError("Routine needs at least one skill")
        parts: list[Skill] = [Takeover()]
        for i, s in enumerate(skills):
            if i > 0 and pause > 0:
                parts.append(Hold(seconds=pause))
            parts.append(s)
        parts.append(Handback())

        segments: list[Segment] = []
        joints: set[int] = set()
        for part in parts:
            joints.update(part.joints)
            segments.extend(skill_segments(part))

        self.skills = tuple(skills)
        super().__init__(segments, joints=sorted(joints),
                         name=name or "+".join(s.name for s in skills))


Rule = tuple[Callable[[Obs], bool], "str | Skill"]


class Selector(Policy):
    """Run a registered motion when a frame predicate fires.

    States: takeover (bookend) -> idle at STAND, evaluating ``rules`` on every
    new frame or new percept -> the first matching motion -> handback
    (``once``) or back to idle after ``cooldown`` seconds. Idle for ``timeout``
    seconds ends the run. Rules see the whole ``Obs``; ``uses_vision`` asks the
    runner for a vision model.
    Every motion it can pick is an ordinary registered motion the check stage
    already validates; the only new boundaries are idle -> motion, and those
    start from STAND, which every motion enters from anyway.
    """

    name = "selector"
    uses_camera = True

    def __init__(self, rules: Sequence[Rule], *, timeout: float = 30.0, once: bool = True,
                 cooldown: float = 1.0, uses_vision: bool = False, vision_refresh: float = 2.0,
                 name: str | None = None) -> None:
        self.rules = [(pred, SKILLS[m]() if isinstance(m, str) else m) for pred, m in rules]
        self.timeout = timeout
        self.once = once
        self.cooldown = cooldown
        self.uses_vision = uses_vision
        self.vision = VisionQuery(vision_refresh)
        joints: set[int] = set(Takeover().joints)
        for _, m in self.rules:
            joints.update(m.joints)
        self.joints = sorted(joints)
        if name is not None:
            self.name = name

    def reset(self, obs: Obs) -> None:
        self._cmd = obs.q.copy()
        self.vision.reset()
        self._last_seq: int | None = None
        self._last_pseq: int | None = None
        self._sub: SegmentPolicy | None = None
        self._enter("takeover", 0.0, Takeover())

    def _enter(self, state: str, t: float, part: Skill | None = None) -> None:
        self._state, self._t0 = state, t
        if part is None:
            self._sub = None
            return
        segs = skill_segments(part)
        if state == "motion" and not self.once and self.cooldown > 0:
            segs.extend(skill_segments(Hold(seconds=self.cooldown)))
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
        if self.uses_vision:
            self.vision.poll(self.perceiver, obs, t)       # ask for a fresh view while idle
        if t - self._t0 >= self.timeout:
            print(f"[{self.name}] idle for {self.timeout:.0f}s, handing back")
            self._enter("handback", t, Handback())
            return self.step(t, obs)
        f, p = obs.frame, obs.percept
        new = False
        if f is not None and f.seq != self._last_seq:
            self._last_seq, new = f.seq, True
        if p is not None and p.seq != self._last_pseq:
            self._last_pseq, new = p.seq, True
        if new:
            for pred, motion in self.rules:
                if pred(obs):
                    print(f"[{self.name}] trigger -> {motion.name}")
                    self._enter("motion", t, motion)
                    return self.step(t, obs)
        return self.action(self._cmd.copy(), weight=1.0)


ROUTINES: dict[str, Callable[[], Routine]] = {
    "demo": lambda: Routine(TPose(), SixSeven(reps=3), name="demo"),
}

POLICIES: dict[str, Callable[[], Policy]] = {
    "look": lambda: Look(),
    "describe": lambda: Describe(),
    "goto_red": lambda: GoTo(RedDot(), name="goto_red"),
    "face_door": lambda: Face(Doorway(), name="face_door"),   # Doorway is a stub: Face only
    "wave_on_red": lambda: Selector([(seen(RedDot()), "sixseven")], name="wave_on_red"),
    "wave_on_person": lambda: Selector([(seen(Labeled(("person",))), "sixseven")],
                                       uses_vision=True, name="wave_on_person"),
}


def build_policy(spec: str, pause: float = 1.0) -> Policy:
    """A registered routine or policy by name, or a comma-separated chain of
    skills (``walk_forward:0.5,turn:45,tpose``), run as one Routine from the
    start pose with the bookends once."""
    if spec in ROUTINES:
        return ROUTINES[spec]()
    if spec in POLICIES:
        return POLICIES[spec]()
    items = [n.strip() for n in split_outside(spec, ",") if n.strip()]
    if not items:
        raise KeyError(spec)
    parts = []
    for item in items:
        skill = parse_skill(item)
        if skill.terminal or skill.internal or not skill.segments():
            raise ValueError(f"{skill.name} cannot be chained; it is not a movement")
        parts.append(skill)
    return Routine(*parts, pause=pause)
