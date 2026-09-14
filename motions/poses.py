"""Named upper-body poses shared by motions. Keys are DDS joint indices."""

from config import STAND_Q, UPPER_BODY

# Baseline every routine starts and ends at: the Menagerie ``stand`` keyframe
# (waist zero; arms 0.2, +/-0.2, 0, 1.28, 0, 0, 0 = relaxed hanging arms).
# Derived from config.STAND_Q so check/sim reset to exactly this pose.
STAND = {j: float(STAND_Q[j]) for j in UPPER_BODY}

# T-pose: arms straight out to the sides at shoulder height.
ARMS_UP = {
    15: 0.0, 16: 1.57, 17: 0.0, 18: 1.47, 19: 0.0, 20: 0.0, 21: 0.0,   # left
    22: 0.0, 23: -1.57, 24: 0.0, 25: 1.47, 26: 0.0, 27: 0.0, 28: 0.0,  # right
}

# Upper arms hanging, elbows bent 90 degrees, forearms forward, palms up.
# Conventions (verified by rendering the Menagerie model):
#   * elbow 0.0 is the 90-degree bend with the forearm forward; ~1.57 is a
#     straight arm; more negative bends the forearm up
#   * wrist roll -1.57 (left) / +1.57 (right) turns the palms up, thumbs outward
SIXSEVEN = {
    12: 0.0, 13: 0.0, 14: 0.0,                                            # waist
    15: 0.0, 16: 0.1, 17: 0.0, 18: 0.0, 19: -1.571, 20: 0.0, 21: 0.0,     # left arm
    22: 0.0, 23: -0.1, 24: 0.0, 25: 0.0, 26: 1.571, 27: 0.0, 28: 0.0,     # right arm
}
