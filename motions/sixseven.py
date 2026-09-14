"""Six-seven: upper arms hanging, elbows bent 90 degrees so the forearms point
forward, palms up, hands shoulder-width apart, as if about to receive something.

Then the elbows oscillate: the left forearm swings up until the fingertips
reach shoulder height; as it starts back down the right forearm swings up, and
so on, overlapping like a see-saw for ``reps`` rounds each.

Joint conventions (verified by rendering the Menagerie model):
  * elbow 0.0 is the 90-degree bend with the forearm forward; ~1.57 is a straight arm;
    more negative bends the forearm up. Fingertips reach shoulder height at ~-0.45.
  * wrist roll rotates about the forearm axis; -1.57 (left) / +1.57 (right) turns
    the palms up, with the thumbs pointing outward
"""
from __future__ import annotations

from motions.poses import SIXSEVEN
from policy import Motion, Segment

L_ELBOW, R_ELBOW = 18, 25
ELBOW_REST = 0.0        # 90-degree bend, forearm horizontal
ELBOW_UP = -0.45        # fingertips at shoulder height


class SixSeven(Motion):
    name = "sixseven"

    def __init__(self, reps: int = 3, swing_time: float = 0.8,
                 settle: float = 3.0, hold: float = 1.0) -> None:
        self.reps = reps
        self.swing_time = swing_time
        self.settle = settle
        self.hold = hold

    def _swings(self) -> tuple[Segment, ...]:
        """Each segment raises one hand while lowering the other, so the next hand
        starts up exactly when the previous one starts down."""
        order = [("left", L_ELBOW), ("right", R_ELBOW)] * self.reps
        out: list[Segment] = []
        prev = None
        for i, (side, joint) in enumerate(order):
            goal = {joint: ELBOW_UP}
            if prev is not None:
                goal[prev] = ELBOW_REST
            out.append(Segment(goal, self.swing_time,
                               label=f"{side} hand up ({i // 2 + 1}/{self.reps})"))
            prev = joint
        if prev is not None:
            out.append(Segment({prev: ELBOW_REST}, self.swing_time))   # last hand back down
        return tuple(out)

    def segments(self) -> tuple[Segment, ...]:
        return (
            Segment(SIXSEVEN, self.settle, label="palms up, elbows 90"),
            Segment(SIXSEVEN, self.hold, label="holding"),
            *self._swings(),
            Segment(SIXSEVEN, self.hold, label="holding"),
        )
