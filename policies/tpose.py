"""T-pose: neutral -> arms straight out to the sides -> neutral.

Port of arm_movements/tpose.py onto the Policy interface.
"""
from __future__ import annotations

from config import UPPER_BODY
from policy import Segment, SegmentPolicy

# Arms hanging at the sides, elbows slightly bent, waist zero.
NEUTRAL = {
    12: 0.0, 13: 0.0, 14: 0.0,                                          # waist
    15: 0.0, 16: 0.0, 17: 0.0, 18: 1.5, 19: 0.0, 20: 0.0, 21: 0.0,     # left arm
    22: 0.0, 23: 0.0, 24: 0.0, 25: 1.5, 26: 0.0, 27: 0.0, 28: 0.0,     # right arm
}

# Arms straight out to the sides at shoulder height.
ARMS_UP = {
    15: 0.0, 16: 1.57, 17: 0.0, 18: 1.47, 19: 0.0, 20: 0.0, 21: 0.0,   # left
    22: 0.0, 23: -1.57, 24: 0.0, 25: 1.47, 26: 0.0, 27: 0.0, 28: 0.0,  # right
}


class TPose(SegmentPolicy):
    name = "tpose"
    joints = UPPER_BODY
    segments = (
        # Ramp weight 0->1 while holding the observed pose. Nothing should move.
        Segment("start", 2.0, weight=lambda a: a, label="taking over (hold)"),
        Segment(NEUTRAL, 3.0, label="moving to neutral"),
        Segment(ARMS_UP, 3.0, label="arms up"),
        Segment(NEUTRAL, 3.0, label="returning to neutral"),
        # Ramp weight 1->0 at neutral so the controller takes the arms back smoothly.
        Segment(NEUTRAL, 2.0, weight=lambda a: 1.0 - a, label="handing back"),
    )
