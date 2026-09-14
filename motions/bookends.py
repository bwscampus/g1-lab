"""Bookends a Routine wraps around its motions, plus the pause between them.

Takeover: ramp the arm_sdk weight 0->1 while holding the pose observed at reset
(nothing should move), then go to STAND.
Handback: go to STAND, then hold it while ramping the weight 1->0 so the
onboard controller takes the arms back smoothly.
"""
from __future__ import annotations

from motions.poses import STAND
from policy import Motion, Segment


class Takeover(Motion):
    name = "takeover"

    def __init__(self, ramp: float = 2.0, to_stand: float = 3.0) -> None:
        self.ramp = ramp
        self.to_stand = to_stand

    def segments(self) -> tuple[Segment, ...]:
        return (
            Segment("start", self.ramp, weight=lambda a: a, label="taking over (hold)"),
            Segment(STAND, self.to_stand, label="moving to stand"),
        )


class Handback(Motion):
    name = "handback"

    def __init__(self, to_stand: float = 3.0, ramp: float = 2.0) -> None:
        self.to_stand = to_stand
        self.ramp = ramp

    def segments(self) -> tuple[Segment, ...]:
        return (
            Segment(STAND, self.to_stand, label="returning to stand"),
            Segment(STAND, self.ramp, weight=lambda a: 1.0 - a, label="handing back"),
        )


class Hold(Motion):
    """Hold whatever pose the previous segment ended in."""

    name = "hold"

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def segments(self) -> tuple[Segment, ...]:
        return (Segment({}, self.seconds, label="pause"),)
