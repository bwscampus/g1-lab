"""T-pose: arms straight out to the sides at shoulder height, held."""
from __future__ import annotations

from motions.poses import ARMS_UP
from policy import Motion, Segment


class TPose(Motion):
    name = "tpose"

    def __init__(self, hold: float = 5.0, rise: float = 3.0) -> None:
        self.hold = hold
        self.rise = rise

    def segments(self) -> tuple[Segment, ...]:
        return (
            Segment(ARMS_UP, self.rise, label="arms up"),
            Segment(ARMS_UP, self.hold, label="holding T-pose"),
        )
