"""Sitting down: the onboard controller's sit (FSM 3), the one FSM a tool may enter.
RobotEnv sends it as SetFsmId(3) and the teardown stands the robot up (SetFsmId(801));
the agent offers nothing that moves the base or calls the onboard controller once it
sits. The twin plays a stand-in sit (``g1/twin/onboard.py``) and no stand-up."""
from __future__ import annotations

from g1.core import limits
from g1.core.action import Segment
from g1.core.poses import STAND
from g1.tools.base import Tool, limit, num


class Sit(Tool):
    """Hand the arms to the onboard controller (weight 1->0), sit (once), wait
    for the sit to finish at weight 0, then take the arms back (0->1)."""

    name = "sit"
    order = 42
    needs_base = True                    # the robot on its own feet, never on the gantry
    needs_loco = True
    command = "Sit"
    ramp_s = limits.get("gesture_ramp_s")
    prompt = ("Sit down where the robot stands, with the onboard controller's own sit; it does not aim for a chair. "
              "Only with a chair right behind the heels: first stand square in front of the chair, facing it, about "
              "{limit:sit_line_up_m} m from the seat's front edge (the edge then sits low in the image, centred); "
              "then move dyaw_deg 180 to turn your back to it. The turn is never exact, so look back (look "
              "{limit:look_yaw_max_deg} and -{limit:look_yaw_max_deg}) to check the chair is straight behind and "
              "square: then it shows at the left edge of the left look and, mirrored, at the right edge of the right "
              "look; if one side shows more of it, or it is turned, look 0, correct with a small move (dyaw_deg a few degrees, or dy_m a few cm) and "
              "look back again. Then look 0, move dx_m -{limit:sit_back_m} to back up to it, look back once more to "
              "see the seat right behind the heels, look 0, and sit. "
              "The host hands the arms to the onboard controller for seconds (default {limit:sit_seconds}), then "
              "takes them back. Seated, the robot cannot walk or sit again; "
              "end with done.")
    params = {"seconds": num(limit("sit_seconds_min"), limit("sit_seconds_max"), limit("sit_seconds"),
                             "how long the onboard sit takes")}

    def segments(self) -> tuple[Segment, ...]:
        assert self.command is not None
        return (Segment(STAND, self.ramp_s, weight=lambda a: 1.0 - a, label="handing the arms to the onboard controller"),
                Segment(STAND, self.seconds, weight=lambda a: 0.0, command=(self.command, {}), label="sit (onboard)"),
                Segment(STAND, self.ramp_s, weight=lambda a: a, label="taking the arms back"))
