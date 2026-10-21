"""Upper-body motion through arm_sdk: joint-space waypoints (the model's
``arm_path``, GPT-Policy's ``move_eef_chunk`` without IK) and the presets."""
from __future__ import annotations

import math

from g1.core import limits
from g1.core.action import Segment
from g1.core.config import HEAD_CAMERA_FOV_H, HEAD_CAMERA_PITCH, JOINT_HI, JOINT_LO, joint_index
from g1.core.poses import ARMS_UP, SIXSEVEN, STAND
from g1.tools.base import Tool, integer, limit, num

WAIST_YAW = joint_index("waist_yaw")
L_ELBOW, R_ELBOW = joint_index("left_elbow"), joint_index("right_elbow")
ELBOW_REST = 0.0      # 90-degree bend, forearm horizontal
ELBOW_UP = -0.45      # fingertips at shoulder height
# The right arm pointing (verified on the twin model): shoulder_pitch -1.57 with elbow 1.57
# holds a straight arm level and forward; each degree less of pitch tips it a degree down.
R_ARM = [joint_index(f"right_{j}") for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                                               "wrist_roll", "wrist_pitch", "wrist_yaw")]
ELBOW_STRAIGHT = 1.57
POINT_ROLL = -0.1     # the upper arm a little out, clear of the torso


class ArmPath(Tool):
    """Joint-space waypoints over the arm_sdk joints. Each waypoint's
    ``joints`` merges onto the previous pose."""

    name = "arm_path"
    order = 20
    prompt = ("Move the waist and arms through 1 to {limit:arm_path_max_waypoints} joint-space waypoints; each gives "
              "joints (name -> radians, see the joint table) and seconds ({limit:arm_path_seconds_min} to "
              "{limit:arm_path_seconds_max}, default {limit:arm_path_seconds_default}). Omitted joints keep their "
              "current pose, and joints stay where an arm_path leaves them until another arm_path moves them; "
              "walking does not change the arms. Use waist_yaw alone (positive LEFT) to glance sideways without "
              "moving the feet. Keep the arms clear of the body and the head; the host rejects "
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


class Point(Tool):
    """The right arm straight out along the torso's heading, tipped down by
    ``down_deg``, held, then lowered to the stand pose. The waist is left where
    it is, so the arm points where the camera looks."""

    name = "point"
    order = 25
    prompt = ("Point at something with the right arm: raise it straight out over rise_s seconds along the direction "
              "the torso (and the camera) faces, tipped down_deg below horizontal (0 to {limit:point_down_deg_max}), "
              "hold it hold_s seconds, then lower it. It points straight ahead of the torso, so first turn the waist "
              "(look) until the target is in the middle of the image (locate_point gives the look_yaw_deg that "
              "centres it, and its below_horizontal_deg); then aim down_deg by where it sits: the "
              f"image's centre row is about {math.degrees(HEAD_CAMERA_PITCH):.0f} degrees below horizontal, a target "
              "far across the floor about 15-25, one near the feet about 50-60, a raised one about 1 m "
              "away (on furniture) about 10-20 (aim higher than its place in the image suggests). Only the right arm moves. Point with "
              "the waist at most 90 degrees round; for a target further round, turn the feet first (move) where "
              "walking is allowed.")
    params = {"down_deg": num(0.0, limit("point_down_deg_max"), limit("point_down_deg_default"),
                              "degrees below horizontal to aim the arm"),
              "rise_s": num(limit("point_rise_s_min"), limit("point_rise_s_max"), limit("point_rise_s_default"),
                            "seconds to raise and to lower the arm"),
              "hold_s": num(limit("point_hold_s_min"), limit("point_hold_s_max"), limit("point_hold_s_default"),
                            "seconds to hold the point")}

    def segments(self) -> tuple[Segment, ...]:
        # a straight arm sags under gravity at arm_sdk's gains, most when level: aim that much higher
        lift = limits.get("point_sag_deg") * math.cos(math.radians(self.down_deg))
        aim = dict(zip(R_ARM, (-math.radians(90.0 - self.down_deg + lift), POINT_ROLL, 0.0, ELBOW_STRAIGHT,
                               0.0, 0.0, 0.0)))
        down = {j: STAND[j] for j in R_ARM}
        return (Segment(aim, self.rise_s, label=f"pointing {self.down_deg:.0f} deg down"),
                Segment(aim, self.hold_s, label="holding the point"),
                Segment(down, self.rise_s, label="arm down"))


class Look(Tool):
    """Turn only the waist and hold it, so the next image looks sideways."""

    name = "look"
    order = 22
    prompt = ("Turn only the waist to yaw_deg degrees from the feet (up to {limit:look_yaw_max_deg} either way), "
              "positive LEFT, over seconds, and hold it there; the feet and arms do not move. yaw_deg is where to "
              "look, not a step: look 0 faces the feet again. Search without walking: look in steps of 30-45 "
              f"degrees and observe between them (the image spans about {HEAD_CAMERA_FOV_H:.0f} degrees); to "
              "face a visible target, look to the look_yaw_deg locate_point gives for its pixel. Look back: yaw_deg "
              "{limit:look_yaw_max_deg} (over the left shoulder) or -{limit:look_yaw_max_deg} (the right) shows "
              "what is behind; straight behind the feet (180) lies the rest of the way toward the image's edge on "
              "the side you turned, so a thing straight behind sits off-centre toward that edge. Look back before "
              "anything that moves the robot into space it cannot see (walking backward or sideways, sitting) and "
              "after a large turn to check where a target ended up; give a look-back about 2 s per 90 degrees, then "
              "look 0 before walking (a waist held far round can twist the feet).")
    params = {"yaw_deg": num(-limit("look_yaw_max_deg"), limit("look_yaw_max_deg"), None, "degrees, + left / - right"),
              "seconds": num(limit("look_seconds_min"), limit("look_seconds_max"), limit("look_seconds_default"),
                              "seconds to turn the waist")}

    def segments(self) -> tuple[Segment, ...]:
        return (Segment({WAIST_YAW: math.radians(self.yaw_deg)}, self.seconds,
                        label=f"look {self.yaw_deg:+.0f} deg"),)


class TPose(Tool):
    """A preset: both arms straight out to the sides, hold, lower."""

    name = "tpose"
    order = 63
    visible = False
    prompt = ("Raise both arms straight out to the sides over rise_s seconds, hold for hold_s, then lower them over "
              "rise_s. Needs clear space to both sides at shoulder height.")
    params = {"hold_s": num(limit("tpose_hold_s_min"), limit("hold_seconds_max"), limit("tpose_hold_s_default"),
                            "seconds to hold the pose"),
              "rise_s": num(limit("tpose_rise_s_min"), limit("tpose_rise_s_max"), limit("tpose_rise_s_default"),
                            "seconds to raise and to lower")}

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
    params = {"reps": integer(1, limit("sixseven_reps_max"), limit("sixseven_reps_default"), "swings per hand"),
              "swing_s": num(limit("sixseven_swing_s_min"), limit("sixseven_swing_s_max"),
                             limit("sixseven_swing_s_default"), "seconds per swing"),
              "settle_s": num(limit("sixseven_settle_s_min"), limit("sixseven_settle_s_max"),
                              limit("sixseven_settle_s_default"), "seconds to reach the palms-up pose"),
              "hold_s": num(limit("sixseven_hold_s_min"), limit("sixseven_hold_s_max"),
                            limit("sixseven_hold_s_default"), "seconds to hold before and after")}

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
                Segment(STAND, limits.get("sixseven_lower_s"), label="arms down"))
