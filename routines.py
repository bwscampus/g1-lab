"""Routines: runnable compositions of motions.

A Routine is one SegmentPolicy whose segments are
    Takeover + motion_1 + pause + motion_2 + ... + Handback
so the arm_sdk bookends happen exactly once and every motion boundary is part
of the same continuous command stream (the check env's velocity gate therefore
covers every transition). Composition is deliberately done at the segment level
rather than by chaining Policy objects: chaining would restart each policy from
the env's *measured* state, which on the robot lags the command by gravity sag
and would produce a target jump at each boundary that the check stage cannot see.

Named routines live in ROUTINES; ad hoc ones come from ``--policy tpose,sixseven``.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable

from motions import MOTIONS, Handback, Hold, SixSeven, Takeover, TPose
from policy import Motion, Policy, Segment, SegmentPolicy


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
            for seg in part.segments():
                if seg.goal == "start" and not isinstance(part, Takeover):
                    raise ValueError(f"motion {part.name!r} uses the reserved 'start' goal")
                label = f"{part.name}: {seg.label}" if seg.label else ""
                segments.append(replace(seg, label=label))

        self.motions = tuple(motions)
        super().__init__(segments, joints=sorted(joints),
                         name=name or "+".join(m.name for m in motions))


ROUTINES: dict[str, Callable[[], Routine]] = {
    "demo": lambda: Routine(TPose(), SixSeven(reps=3), name="demo"),
}


def build_policy(spec: str, pause: float = 1.0) -> Policy:
    """A registered routine by name, or a comma-separated list of motion names."""
    if spec in ROUTINES:
        return ROUTINES[spec]()
    names = [n.strip() for n in spec.split(",") if n.strip()]
    if not names:
        raise KeyError(spec)
    motions = []
    for n in names:
        if n not in MOTIONS:
            raise KeyError(n)
        motions.append(MOTIONS[n]())
    return Routine(*motions, pause=pause)
