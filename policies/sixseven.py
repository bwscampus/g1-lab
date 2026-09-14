"""Six-seven: upper arms hanging, elbows bent 90 degrees so the forearms point
forward, palms up, hands shoulder-width apart, as if about to receive something.

Then the elbows oscillate: the left forearm swings up until the fingertips
reach shoulder height; as it starts back down the right forearm swings up, and
so on, overlapping like a see-saw for ``REPS`` rounds each.

Joint conventions (verified by rendering the Menagerie model):
  * elbow 0.0 is the 90-degree bend with the forearm forward; ~1.57 is a straight arm;
    more negative bends the forearm up. Fingertips reach shoulder height at ~-0.45.
  * wrist roll rotates about the forearm axis; -1.57 (left) / +1.57 (right) turns
    the palms up, with the thumbs pointing outward
"""
from __future__ import annotations

from config import UPPER_BODY
from policy import Segment, SegmentPolicy
from policies.tpose import NEUTRAL

SIXSEVEN = {
    12: 0.0, 13: 0.0, 14: 0.0,                                            # waist
    15: 0.0, 16: 0.1, 17: 0.0, 18: 0.0, 19: -1.571, 20: 0.0, 21: 0.0,     # left arm
    22: 0.0, 23: -0.1, 24: 0.0, 25: 0.0, 26: 1.571, 27: 0.0, 28: 0.0,     # right arm
}


L_ELBOW, R_ELBOW = 18, 25
ELBOW_REST = 0.0        # 90-degree bend, forearm horizontal
ELBOW_UP = -0.45        # fingertips at shoulder height
REPS = 3                # swings per hand
SWING_TIME = 0.8        # seconds for each half swing (up, then down)


def _swings() -> tuple[Segment, ...]:
    """Each segment raises one hand while lowering the other, so the next hand
    starts up exactly when the previous one starts down."""
    order = [("left", L_ELBOW), ("right", R_ELBOW)] * REPS      # 2*REPS raises
    out = []
    prev = None
    for i, (side, joint) in enumerate(order):
        goal = {joint: ELBOW_UP}
        if prev is not None:
            goal[prev] = ELBOW_REST
        out.append(Segment(goal, SWING_TIME, label=f"{side} hand up ({i // 2 + 1}/{REPS})"))
        prev = joint
    out.append(Segment({prev: ELBOW_REST}, SWING_TIME))          # last hand back down
    return tuple(out)


class SixSeven(SegmentPolicy):
    name = "sixseven"
    joints = UPPER_BODY
    segments = (
        Segment("start", 2.0, weight=lambda a: a, label="taking over (hold)"),
        Segment(NEUTRAL, 3.0, label="moving to neutral"),
        Segment(SIXSEVEN, 3.0, label="palms up, elbows 90"),
        Segment(SIXSEVEN, 1.0, label="holding"),
        *_swings(),
        Segment(SIXSEVEN, 1.0, label="holding"),
        Segment(NEUTRAL, 3.0, label="returning to neutral"),
        Segment(NEUTRAL, 2.0, weight=lambda a: 1.0 - a, label="handing back"),
    )
