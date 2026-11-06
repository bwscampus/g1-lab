"""Named upper-body poses shared by tools. Keys are DDS joint indices."""

from g1.core.config import STAND_Q, UPPER_BODY

# Baseline every chain starts and ends at: the stand pose
# (waist zero; arms 0.2, +/-0.2, 0, 1.28, 0, 0, 0 = relaxed hanging arms).
# Derived from config.STAND_Q, which the twin's onboard controller holds too.
STAND = {j: float(STAND_Q[j]) for j in UPPER_BODY}
