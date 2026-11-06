"""Upper-body motion through arm_sdk: joint-space waypoints (the model's
``arm_path``, GPT-Policy's ``move_eef_chunk`` without IK) and the presets."""
from __future__ import annotations

import math

import numpy as np

from g1.core import aim as aiming
from g1.core import limits
from g1.core.action import Segment
from g1.core.config import (HEAD_CAMERA_FOV_H, HEAD_CAMERA_PITCH, HEAD_CAMERA_SIZE, JOINT_HI, JOINT_LO,
                             R_SHOULDER_FROM_CAMERA, WAIST, joint_index)
from g1.core.poses import STAND
from g1.tools.base import Tool, limit, num
from g1.vlm import IMAGE_WIDTH

WAIST_YAW = joint_index("waist_yaw")
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
    prompt = ("When the target is in view, prefer point_to (it aims for you). "
              "Point at something with the right arm: raise it straight out over rise_s seconds along the direction "
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


class PointAt(Point):
    """``point`` aimed by the host: the model names a pixel of this turn's image and how far away
    the thing is; ``g1/core/aim.py`` turns that into a point in the feet frame (the camera's ray,
    from the waist joints and the IMU measured when the image was taken) and solves the waist yaw
    and shoulder pitch whose straight-arm line, from the shoulder (12 cm right of and 19 cm below the
    camera, on a shoulder joint mounted 16 degrees rolled), passes through it. The waist turns and
    the arm rises together; the arm is left up like point's. The Agent binds the turn's view before
    it plans the tool (``bind``); unbound (a tools chain, the tests) the pixel is read as if the
    image were the twin's, taken from the stand pose. ``aim_report`` says, after it, how far the
    arm's measured line misses that point."""

    name = "point_to"
    order = 24
    prompt = ("Point the right arm at something you see, in one move: give its pixel in this turn's image (x_px from "
              "the left edge, y_px from the top) and how far away it is (distance_m, from the camera; judge it from "
              "how big it looks, a rough guess is enough: it matters only up close). The host turns the waist and "
              "aims the arm so the arm's line passes through that point, allowing for the shoulder sitting below "
              "and to the right of the camera, and keeps the arm UP like point. The next observation's "
              "previous_result.aim says how far the arm's measured line misses the point (left/up degrees) and "
              "where it crosses that distance in the image (arm_line_px). Turning the waist far and raising the arm "
              "twists the robot a little after the image was taken, so the first point_to can miss by 10-20 "
              "degrees: then point_to again with the target's pixel in the new image (it allows for the twist and "
              "the arm's measured sag) and it lands within a few degrees. Use it instead of look + point when the "
              "target is in view; a "
              "target beyond {limit:point_to_yaw_max_deg} degrees of waist turn needs the feet turned first.")
    params = {"x_px": num(0, IMAGE_WIDTH, None, "the target's pixel column, from the left edge"),
              "y_px": num(0, IMAGE_WIDTH, None, "the target's pixel row, from the top edge"),
              "distance_m": num(limit("point_to_distance_min_m"), limit("point_to_distance_max_m"), None,
                                "how far the target is from the camera, metres (a rough estimate)"),
              "rise_s": Point.params["rise_s"], "hold_s": Point.params["hold_s"]}

    def __init__(self, **args) -> None:
        super().__init__(**args)
        self.view: dict | None = None
        self._solved = None

    def bind(self, view: dict | None) -> None:
        """The turn's image and what the robot measured when it was taken (the Agent's view);
        solves the aim now, so a target out of reach is refused before anything moves."""
        if not view or view.get("frame_size") is None:
            raise ValueError("point_to: there was no camera image this turn")
        w, h = view["frame_size"]
        if not (0 <= self.x_px <= w and 0 <= self.y_px <= h):
            raise ValueError(f"point_to: ({self.x_px:g}, {self.y_px:g}) is outside the {w}x{h} image")
        self.view = view
        self._solved = None
        self._solve()

    def _frame(self):
        if self.view is not None:
            q = self.view.get("q")
            waist = (0.0, 0.0, 0.0) if q is None else tuple(float(q[j]) for j in WAIST)
            return self.view["frame_size"], waist, self.view.get("imu_quat")
        return (HEAD_CAMERA_SIZE[1], HEAD_CAMERA_SIZE[0]), (0.0, 0.0, 0.0), None

    def _solve(self):
        """(target, waist yaw, rigid shoulder pitch, down_deg, lift_deg) for this pixel and distance."""
        if self._solved is None:
            (w, h), waist, imu = self._frame()
            target = aiming.pixel_point(self.x_px, self.y_px, w, h, self.distance_m, waist, imu)
            rest = (-math.radians(90.0 - limits.get("point_down_deg_default")), POINT_ROLL, 0.0, ELBOW_STRAIGHT,
                    0.0, 0.0, 0.0)
            yaw, pitch = aiming.solve_point(target, rest, waist[1], waist[2], imu, yaw0=waist[0])
            reach = limits.get("point_to_yaw_max_deg")
            if abs(math.degrees(yaw)) > reach + 1e-6:
                raise ValueError(f"point_to: the target needs the waist turned {math.degrees(yaw):.0f} degrees, beyond "
                                 f"{reach:g}: turn the feet first (move)")
            down = 90.0 + math.degrees(pitch)          # point's down_deg: 0 level, positive below
            low, high = 0.0, limits.get("point_down_deg_max")
            if not (low <= down <= high):              # a clamp with a note, like a number out of range
                self.notes.append(f"the target is {down:.0f} degrees below horizontal from the shoulder; the arm "
                                  f"points {min(max(down, low), high):.0f} (its range is {low:g} to {high:g})")
                down = min(max(down, low), high)
                pitch = math.radians(down - 90.0)
            self._solved = (target, yaw, pitch, down, self._lift(down))
        return self._solved

    def _lift(self, down: float) -> float:
        """How far above its aim to command the arm, degrees: the sag a straight arm will have. Guessed
        (point_sag_deg, most when level) unless the arm is already raised, when the robot measures it:
        the shoulder pitch it commanded against the one it measured, scaled to the new aim."""
        lift = limits.get("point_sag_deg") * math.cos(math.radians(down))
        view = self.view or {}
        q, cmd = view.get("q"), view.get("cmd")
        if q is not None and cmd is not None:
            j = R_ARM[0]
            raised_down = 90.0 + math.degrees(float(q[j]))       # where the raised arm actually points
            if cmd[j] < -math.radians(45.0) and -10.0 < raised_down < 80.0:
                sag = math.degrees(float(q[j]) - float(cmd[j]))  # + : it hangs below its command
                lift = sag * math.cos(math.radians(down)) / max(math.cos(math.radians(raised_down)), 0.3)
        return lift

    def segments(self) -> tuple[Segment, ...]:
        _, yaw, pitch, down, lift = self._solve()
        goal = dict(zip(R_ARM, (pitch - math.radians(lift), POINT_ROLL, 0.0, ELBOW_STRAIGHT, 0.0, 0.0, 0.0)))
        goal[WAIST_YAW] = yaw
        return (Segment(goal, self.rise_s, label=f"pointing at ({self.x_px:.0f}, {self.y_px:.0f}) "
                                                 f"{self.distance_m:.1f} m"),
                Segment(goal, self.hold_s, label="holding the point"))

    def aim_report(self, q, imu_quat=None) -> dict:
        """After the move, from the measured joints: how far the target lies left/above the arm's
        line, and where the line crosses the target's distance in the next image."""
        target, *_ = self._solve()
        (w, h), _, _ = self._frame()
        waist = tuple(float(q[j]) for j in WAIST)
        arm = [float(q[j]) for j in R_ARM]
        left, up = aiming.miss(target, arm, waist, imu_quat)
        shoulder, d = aiming.arm_ray(arm, waist, imu_quat)
        px = aiming.project(shoulder + d * float(np.linalg.norm(target - shoulder)), w, h, waist, imu_quat)
        return {"target_left_of_arm_deg": round(left, 1), "target_above_arm_deg": round(up, 1),
                "arm_line_px": None if px is None else [round(px[0]), round(px[1])],
                "note": "measured: the arm's line from the shoulder, at the distance you gave",
                **({"clamped": list(self.notes)} if self.notes else {})}


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
