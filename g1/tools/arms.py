"""Upper-body motion through arm_sdk: joint-space waypoints (the model's
``arm_path``, GPT-Policy's ``move_eef_chunk`` without IK) and the presets."""
from __future__ import annotations

import math

from g1.core import limits
from g1.core.action import Segment
from g1.core.config import JOINT_HI, JOINT_LO, joint_index
from g1.core.poses import ARMS_UP, SIXSEVEN, STAND
from g1.tools.base import Tool, integer, limit, num

WAIST_YAW = joint_index("waist_yaw")
L_ELBOW, R_ELBOW = joint_index("left_elbow"), joint_index("right_elbow")
ELBOW_REST = 0.0      # 90-degree bend, forearm horizontal
ELBOW_UP = -0.45      # fingertips at shoulder height


class ArmPath(Tool):
    """Joint-space waypoints over the arm_sdk joints. Each waypoint's
    ``joints`` merges onto the previous pose."""

    name = "arm_path"
    order = 20
    prompt = ("Move the waist and arms through 1 to {limit:arm_path_max_waypoints} joint-space waypoints; each gives "
              "joints (name -> radians, see the joint table) and seconds ({limit:arm_path_seconds_min} to "
              "{limit:arm_path_seconds_max}, default {limit:arm_path_seconds_default}). Omitted joints keep their "
              "current pose. Use waist_yaw alone (positive LEFT) to glance sideways without moving the feet; the "
              "next arm_path or move may recentre it. Keep the arms clear of the body and the head; the host rejects "
              "a path that leaves the joint limits or moves too fast, without moving. A gesture, a glance or a "
              "reach: not a search move.")
    params = {
        "waypoints": {
            "type": "array", "minItems": 1, "maxItems": limit("arm_path_max_waypoints"),
            "description": "joint-space waypoints, in order",
            "items": {"type": "object", "additionalProperties": False, "required": ["joints"],
                      "properties": {"joints": {"$template": "arm_joints"},
                                     "seconds": num(limit("arm_path_seconds_min"), limit("arm_path_seconds_max"),
                                                    limit("arm_path_seconds_default"), "seconds to reach it")}},
        },
    }

    def __init__(self, **args) -> None:
        super().__init__(**args)
        from g1.tools import ARM_JOINTS
        if not self.waypoints:
            raise ValueError("arm_path: at least one waypoint")
        lo, hi = limits.get("arm_path_seconds_min"), limits.get("arm_path_seconds_max")
        for i, wp in enumerate(self.waypoints):
            if not isinstance(wp, dict) or not isinstance(wp.get("joints"), dict) or not wp["joints"]:
                raise ValueError(f"arm_path: waypoint {i} needs a non-empty joints object")
            for name, v in wp["joints"].items():
                if name not in ARM_JOINTS:
                    raise ValueError(f"arm_path: unknown joint {name!r} (allowed: {ARM_JOINTS})")
                j = joint_index(name)
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    raise ValueError(f"arm_path: {name} must be a number") from None
                if not (JOINT_LO[j] <= v <= JOINT_HI[j]):
                    raise ValueError(f"arm_path: {name}={v} outside [{JOINT_LO[j]:.3f}, {JOINT_HI[j]:.3f}]")
            seconds = wp.get("seconds", limits.get("arm_path_seconds_default"))
            if not isinstance(seconds, (int, float)) or not (lo <= seconds <= hi):
                raise ValueError(f"arm_path: waypoint {i} seconds must be within [{lo}, {hi}]")
        if len(self.waypoints) > limits.get("arm_path_max_waypoints"):
            raise ValueError(f"arm_path: at most {limits.get('arm_path_max_waypoints')} waypoints")

    def segments(self) -> tuple[Segment, ...]:
        n = len(self.waypoints)
        default = limits.get("arm_path_seconds_default")
        return tuple(Segment({joint_index(k): float(v) for k, v in wp["joints"].items()},
                             float(wp.get("seconds", default)), label=f"waypoint {i + 1}/{n}")
                     for i, wp in enumerate(self.waypoints))


class Look(Tool):
    """A preset: turn only the waist and hold it, so the next image looks sideways."""

    name = "look"
    order = 62
    visible = False
    prompt = ("Turn only the waist by yaw_deg degrees (up to {limit:look_yaw_max_deg} either way), positive LEFT, "
              "over seconds, and hold it; the feet do not move and the next tool recentres the waist.")
    params = {"yaw_deg": num(-limit("look_yaw_max_deg"), limit("look_yaw_max_deg"), None, "degrees, + left / - right"),
              "seconds": num(0.5, 5.0, 1.5, "seconds to turn the waist")}

    def segments(self) -> tuple[Segment, ...]:
        return (Segment({**STAND, WAIST_YAW: math.radians(self.yaw_deg)}, self.seconds,
                        label=f"look {self.yaw_deg:+.0f} deg"),)


class TPose(Tool):
    """A preset: both arms straight out to the sides, hold, lower."""

    name = "tpose"
    order = 63
    visible = False
    prompt = ("Raise both arms straight out to the sides over rise_s seconds, hold for hold_s, then lower them over "
              "rise_s. Needs clear space to both sides at shoulder height.")
    params = {"hold_s": num(0.5, 30.0, 5.0, "seconds to hold the pose"),
              "rise_s": num(1.0, 10.0, 3.0, "seconds to raise and to lower")}

    def segments(self) -> tuple[Segment, ...]:
        return (Segment(ARMS_UP, self.rise_s, label="arms up"),
                Segment(ARMS_UP, self.hold_s, label="holding T-pose"),
                Segment(STAND, self.rise_s, label="arms down"))


class SixSeven(Tool):
    """A preset: palms up, elbows bent 90 degrees, then the forearms see-saw:
    the left swings up until the fingertips reach shoulder height, and as it
    starts back down the right swings up, overlapping, for ``reps`` rounds each.

    Joint conventions (verified by rendering the Menagerie model):
      * elbow 0.0 is the 90-degree bend with the forearm forward; ~1.57 is a
        straight arm; more negative bends the forearm up
      * wrist roll -1.57 (left) / +1.57 (right) turns the palms up, thumbs outward
    """

    name = "sixseven"
    order = 64
    visible = False
    prompt = ("A two-handed see-saw wave with the palms up: reps swings per hand of swing_s seconds each, after "
              "settle_s seconds to reach the palms-up pose and hold_s seconds before and after.")
    params = {"reps": integer(1, 10, 3, "swings per hand"),
              "swing_s": num(0.3, 2.0, 0.8, "seconds per swing"),
              "settle_s": num(0.5, 5.0, 3.0, "seconds to reach the palms-up pose"),
              "hold_s": num(0.1, 5.0, 1.0, "seconds to hold before and after")}

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
            out.append(Segment(goal, self.swing_s, label=f"{side} hand up ({i // 2 + 1}/{self.reps})"))
            prev = joint
        if prev is not None:
            out.append(Segment({prev: ELBOW_REST}, self.swing_s))   # last hand back down
        return tuple(out)

    def segments(self) -> tuple[Segment, ...]:
        return (Segment(SIXSEVEN, self.settle_s, label="palms up, elbows 90"),
                Segment(SIXSEVEN, self.hold_s, label="holding"),
                *self._swings(),
                Segment(SIXSEVEN, self.hold_s, label="holding"),
                Segment(STAND, 1.2, label="arms down"))
