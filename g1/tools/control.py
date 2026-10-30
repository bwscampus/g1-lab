"""Tools that move nothing or bracket a run: ``hold``, the ``check`` dry run
(GPT-Policy's ``check_path``), ``locate_point`` (a pixel's direction), the two terminals ``done`` / ``give_up``, and
the ``takeover`` / ``handback`` bookends every chain and every agent run is
wrapped in (never in a menu, never chainable)."""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from g1.core import limits
from g1.core.action import Obs, Segment
from g1.core.config import CONTROL_DT, HEAD_CAMERA_FOVY, HEAD_CAMERA_PITCH, JOINT_NAMES, joint_index
from g1.tools.base import Tool, limit, num, text
from g1.vlm import IMAGE_WIDTH


WAIST_YAW, WAIST_ROLL, WAIST_PITCH = joint_index("waist_yaw"), joint_index("waist_roll"), joint_index("waist_pitch")


class Hold(Tool):
    name = "hold"
    order = 30
    prompt = ("Stand still for seconds (up to {limit:hold_seconds_max}, default {limit:hold_seconds_default}) keeping the current pose. Use it "
              "to wait for a moving obstacle or a person to clear, or to observe again without moving.")
    params = {"seconds": num(limit("hold_seconds_min"), limit("hold_seconds_max"), limit("hold_seconds_default"),
                             "how long to wait")}

    def segments(self) -> tuple[Segment, ...]:
        return (Segment({}, self.seconds, label=f"hold {self.seconds:.1f} s"),)


def dry_run(tool: Tool, cmd: np.ndarray, joints: Sequence[int]) -> dict:
    """Plan a tool from the commanded pose through a JointMonitor without
    physics — joint limits, command speed, base limits, duration and the base
    displacement it would produce. Moves nothing. Every movement the model
    picks goes through this before it executes (their IK check before
    submission); ``check`` exposes it to the model."""
    from g1.envs.monitor import MAX_PRINTED, MOVED_RAD, JointMonitor
    from g1.tools import prefixed
    player = Tool.of(prefixed(tool), joints=list(joints), name="check")
    player.reset(Obs(cmd))
    mon = JointMonitor(strict=False)
    mon.reset(hold=cmd)
    n = 0
    while (a := player.step(n * CONTROL_DT, Obs(cmd))) is not None:
        mon.observe(a.q, a)
        n += 1
    x, y, yaw = mon.base_pose()
    moved = [JOINT_NAMES[j] for j in mon.rows() if np.isfinite(mon.cmd_min[j]) and mon.cmd_max[j] - mon.cmd_min[j] > MOVED_RAD]
    return {"status": "ok" if not mon.violations else "rejected",
            "tool": tool.name, "arguments": {k: v for k, v in tool.args.items() if k != "note"},
            "duration_s": n * CONTROL_DT, "base_delta": [x, y, math.degrees(yaw)],
            "peak_command_vel_rad_s": float(mon.peak_vel.max()), "joints_moved": moved,
            "violations": [str(v) for v in mon.violations[:MAX_PRINTED]], "n_violations": len(mon.violations)}


class Check(Tool):
    """``check(tool, arguments)``: the host plans the named tool from its
    current commanded pose through a JointMonitor and reports the verdict."""

    name = "check"
    order = 35
    kind = "query"
    prompt = ("Check a tool and its arguments without moving: the host plans it from the current commanded pose and "
              "reports joint-limit, command-speed and base-velocity violations, its duration and the base "
              "displacement it would produce. It does not simulate contact or prove the path is clear; judge that "
              "from the image. Use it for an uncertain long move or an arm_path near the limits before committing; "
              "every movement is dry-run this way before it executes anyway.")
    params = {"tool": {"$template": "tool_name"},
              "arguments": {"type": "object", "description": "the arguments that tool would run with (its note may be omitted)"}}

    def target(self) -> Tool:
        """The tool this check is about, built with its arguments."""
        from g1.tools import TOOLS
        if self.tool not in TOOLS:
            raise ValueError(f"check: unknown tool {self.tool!r}")
        cls = TOOLS[self.tool]
        if cls.kind != "motion" or not cls.visible:
            raise ValueError(f"check: {self.tool} is not a movement")
        args = dict(self.arguments)
        args.setdefault("note", self.note or "checked")
        return cls(**args)

    def query(self, cmd: np.ndarray, joints: Sequence[int], *, view: Optional[dict] = None) -> dict:
        return dry_run(self.target(), cmd, joints)


class LocatePoint(Tool):
    """GPT-Policy's ``locate_point`` for one monocular head camera: the direction of a
    pixel of the latest image, from the camera's intrinsics and the robot's own commanded
    waist yaw. A ray, never a distance: one camera has no depth."""

    name = "locate_point"
    order = 36
    kind = "query"
    prompt = ("Find the direction of something you see, without moving: give its pixel in the latest image as you "
              "see it (x_px from the left edge, y_px from the top; the observation gives the image's width and "
              "height). The host answers bearing_from_torso_deg (+ left), look_yaw_deg (the waist yaw that centres "
              "it: use it with look) and below_horizontal_deg (from the camera). It is a direction only, no "
              "distance. Use it before look or point instead of guessing an angle from the image.")
    # the image the model is sent is at most IMAGE_WIDTH wide (agent.image_width_px in configs/model.json)
    params = {"x_px": num(0, IMAGE_WIDTH, None, "pixel column, from the left edge"),
              "y_px": num(0, IMAGE_WIDTH, None, "pixel row, from the top edge")}

    def query(self, cmd: np.ndarray, joints: Sequence[int], *, view: Optional[dict] = None) -> dict:
        if not view or view.get("frame_size") is None:
            raise ValueError("locate_point: there was no camera image this turn")
        w, h = view["frame_size"]
        q = view.get("q")
        q = cmd if q is None else q                                  # measured when the image was taken
        if not (0 <= self.x_px <= w and 0 <= self.y_px <= h):
            raise ValueError(f"locate_point: ({self.x_px:g}, {self.y_px:g}) is outside the {w}x{h} image")
        return {"status": "ok", "tool": self.name, "pixel": [self.x_px, self.y_px], "image_size": [w, h],
                **bearing(self.x_px, self.y_px, w, h, (float(q[WAIST_YAW]), float(q[WAIST_ROLL]), float(q[WAIST_PITCH])),
                          view.get("imu_quat")),
                "note": "a direction only, no distance (one camera). The right shoulder sits below and to the "
                        "right of the camera: for a target about 1 m away point a little less steep than "
                        "below_horizontal_deg and centre it slightly left of the image middle."}


def _rot(axis: str, a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def _tilt(quat) -> np.ndarray:
    """The pelvis orientation from the IMU (w, x, y, z) with its heading removed: how the
    feet's frame is tipped. Level when there is no IMU."""
    if quat is None:
        return np.eye(3)
    w, x, y, z = (float(v) for v in quat)
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return _rot("z", -math.atan2(r[1, 0], r[0, 0])) @ r


def bearing(x_px: float, y_px: float, width: int, height: int, waist=(0.0, 0.0, 0.0), imu_quat=None) -> dict:
    """A pixel's ray from the head camera (pinhole, square pixels, vertical field of view
    HEAD_CAMERA_FOVY, pitched HEAD_CAMERA_PITCH down on the torso), carried through what the
    robot measures: the waist joints (yaw, roll, pitch, rad) and the pelvis tilt from the IMU.
    As angles: from the torso and from the feet (+ left), and below horizontal."""
    f = (height / 2) / math.tan(math.radians(HEAD_CAMERA_FOVY) / 2)
    x, y = (x_px - width / 2) / f, (y_px - height / 2) / f       # right, down on the image plane
    ray = _rot("y", HEAD_CAMERA_PITCH) @ np.array([1.0, -x, -y])   # in the torso: forward, left, up
    wy, wr, wp = waist
    fwd, left, up = _tilt(imu_quat) @ _rot("z", wy) @ _rot("x", wr) @ _rot("y", wp) @ ray
    look = math.degrees(math.atan2(left, fwd))                       # from the feet
    reach = limits.get("look_yaw_max_deg")
    out = {"bearing_from_torso_deg": round(math.degrees(math.remainder(math.radians(look) - wy, 2 * math.pi)), 1) + 0.0,
           "look_yaw_deg": round(max(-reach, min(reach, look)), 1) + 0.0,
           "below_horizontal_deg": round(math.degrees(math.atan2(-up, math.hypot(fwd, left))), 1) + 0.0}
    if abs(look) > reach:
        out["beyond_waist"] = f"{look:.0f} degrees from the feet is beyond the waist's reach: turn the feet (move) first"
    return out


class Done(Tool):
    name = "done"
    order = 90
    kind = "terminal"
    prompt = ("End the task only when the current image establishes the goal: for a search, the goal object is "
              "clearly visible and near (large, or in the lower third of the image). done is not a success label; a "
              "human assigns that after the run. In summary, give the direct evidence and any remaining uncertainty; "
              "in hindsight, what would have made this run shorter or surer.")
    params = {"summary": text("direct current evidence of the requested outcome and any remaining uncertainty"),
              "hindsight": {"type": "string", "description": "lessons from this attempt"}}


class GiveUp(Tool):
    name = "give_up"
    order = 91
    kind = "terminal"
    prompt = ("End when evidence shows the task cannot be completed safely within the decision budget. One failed "
              "action, blocked path or unseen goal does not establish impossibility: search from another spot first. "
              "In reason, list the strategies tried and the evidence that no reasonable safe continuation remains; "
              "hindsight records lessons from this attempt.")
    params = {"reason": text("the strategies tried and the evidence that none remains"),
              "hindsight": {"type": "string", "description": "lessons from this attempt"}}


class Takeover(Tool):
    """Bookend: ramp the arm_sdk weight 0->1 while holding the pose observed
    at reset. Nothing moves; the first tool starts from wherever the arms are.
    (No move to a neutral pose: a tool that needs a starting pose makes it its
    own first segment.)"""

    name = "takeover"
    order = 98
    visible = False
    allows_start = True
    params = {"ramp_s": num(0.0, limit("bookend_ramp_max_s"), limit("bookend_ramp_s"), "weight ramp seconds")}

    def segments(self) -> tuple[Segment, ...]:
        if self.ramp_s <= 0:
            return ()
        return (Segment("start", self.ramp_s, weight=lambda a: a, label="taking control"),)


class Handback(Tool):
    """Bookend: hold the pose the last tool ended in while ramping the weight
    1->0, so the onboard controller takes the arms back smoothly from there."""

    name = "handback"
    order = 99
    visible = False
    params = {"ramp_s": num(0.0, limit("bookend_ramp_max_s"), limit("bookend_ramp_s"), "weight ramp seconds")}

    def segments(self) -> tuple[Segment, ...]:
        if self.ramp_s <= 0:
            return ()
        return (Segment({}, self.ramp_s, weight=lambda a: 1.0 - a, label="releasing control"),)
