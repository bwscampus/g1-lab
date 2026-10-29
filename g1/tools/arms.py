"""Upper-body motion through arm_sdk: joint-space waypoints (the model's
``arm_path``, GPT-Policy's ``move_eef_chunk`` without IK) and the presets."""
from __future__ import annotations

import math

from g1.core import limits
from g1.core.action import Segment
from g1.core.config import (HEAD_CAMERA_FOV_H, HEAD_CAMERA_PITCH, JOINT_HI, JOINT_LO, R_SHOULDER_FROM_CAMERA,
                             joint_index)
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


def shoulder_parallax(distance_m: float) -> tuple[float, float]:
    """(left_deg, up_deg): how much further left and higher a target the camera looks straight at
    lies as seen from the right shoulder, ``distance_m`` away. The arm's line starts at the shoulder,
    so after centring a target, point that much further left (look) and less down (down_deg)."""
    _, dy, dz = R_SHOULDER_FROM_CAMERA
    return math.degrees(math.atan2(-dy, distance_m)), math.degrees(math.atan2(-dz, distance_m))


PARALLAX_AT_M = (0.8, 1.5, 3.0)     # the distances the point prompt quotes (examples, not settings)


def _parallax_text() -> str:
    _, dy, dz = R_SHOULDER_FROM_CAMERA
    rows = ", ".join(f"{d:g} m away: {left:.0f} left and {up:.0f} up" for d in PARALLAX_AT_M
                     for left, up in [shoulder_parallax(d)])
    return (f"The arm's line starts at the right shoulder, {-dy * 100:.0f} cm right of and {-dz * 100:.0f} cm below "
            "the camera, so a near target the camera has centred lies, from the shoulder, further LEFT and HIGHER: "
            "point with the waist at look_yaw_deg PLUS the left correction (positive is left), and down_deg = "
            f"below_horizontal_deg MINUS the up correction. Corrections in degrees: {rows} (judge the distance from "
            "how big the target looks; far targets need almost none). ")


def raises_arm(tool: Tool):
    """Whether a tool leaves the right arm raised (``point``: True), lowered (any other tool
    that moves it: False), or as it was (None). The agent and a chain lower a raised arm
    before they hand the arms back."""
    if isinstance(tool, Point):
        return True
    if any(j in seg.goal for seg in tool.segments() if isinstance(seg.goal, dict) for j in R_ARM):
        return False
    return None


def arm_down(seconds: float) -> Segment:
    """The right arm back to the stand pose."""
    return Segment({j: STAND[j] for j in R_ARM}, seconds, label="arm down")


class Point(Tool):
    """The right arm straight out along the torso's heading, tipped down by
    ``down_deg``, and left there: the next image shows where it points, so the
    model can correct the aim (look, or point again) and then lower it
    (``lower_arm``). The waist is left where it is, so the arm points where the
    camera looks. A raised arm is lowered before the arms are handed back,
    whatever ends the run."""

    name = "point"
    order = 25
    prompt = ("Point at something with the right arm: raise it straight out over rise_s seconds along the direction "
              "the torso (and the camera) faces, tipped down_deg below horizontal (0 to {limit:point_down_deg_max}), "
              "hold it still hold_s seconds, and KEEP IT UP: the next image shows where it points. Check it there: if "
              "it misses, correct it (look turns the waist and the raised arm with it; point again re-aims); "
              "when it points at the target, lower it with lower_arm (before walking: a raised arm cannot walk; "
              "it is lowered for you when the run ends). It points straight ahead of the torso, so first turn the waist "
              "(look) until the target is in the middle of the image (locate_point gives the look_yaw_deg that "
              "centres it, and its below_horizontal_deg); then aim down_deg by where it sits: the "
              f"image's centre row is about {math.degrees(HEAD_CAMERA_PITCH):.0f} degrees below horizontal, a target "
              "far across the floor about 15-25, one near the feet about 50-60. " + _parallax_text() +
              "Only the right arm moves. Point with "
              "the waist at most 90 degrees round; for a target further round, turn the feet first (move) where "
              "walking is allowed.")
    params = {"down_deg": num(0.0, limit("point_down_deg_max"), limit("point_down_deg_default"),
                              "degrees below horizontal to aim the arm"),
              "rise_s": num(limit("point_rise_s_min"), limit("point_rise_s_max"), limit("point_rise_s_default"),
                            "seconds to raise and to lower the arm"),
              "hold_s": num(limit("point_hold_s_min"), limit("point_hold_s_max"), limit("point_hold_s_default"),
                            "seconds to hold still before the next image")}

    def segments(self) -> tuple[Segment, ...]:
        # a straight arm sags under gravity at arm_sdk's gains, most when level: aim that much higher
        lift = limits.get("point_sag_deg") * math.cos(math.radians(self.down_deg))
        aim = dict(zip(R_ARM, (-math.radians(90.0 - self.down_deg + lift), POINT_ROLL, 0.0, ELBOW_STRAIGHT,
                               0.0, 0.0, 0.0)))
        return (Segment(aim, self.rise_s, label=f"pointing {self.down_deg:.0f} deg down"),
                Segment(aim, self.hold_s, label="holding the point"))


class LowerArm(Tool):
    """The right arm back down to the stand pose, after a point."""

    name = "lower_arm"
    order = 26
    prompt = ("Lower the right arm to the stand pose over seconds. Use it after point, once the image shows the "
              "arm pointing at the target (or to stop pointing before walking).")
    params = {"seconds": num(limit("point_rise_s_min"), limit("point_rise_s_max"), limit("point_rise_s_default"),
                             "seconds to lower the arm")}

    def segments(self) -> tuple[Segment, ...]:
        return (arm_down(self.seconds),)


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
