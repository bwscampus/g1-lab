"""Tools that move nothing or bracket a run: ``hold``, the ``check`` dry run
(GPT-Policy's ``check_path``), the two terminals ``done`` / ``give_up``, and
the ``takeover`` / ``handback`` bookends every chain and every agent run is
wrapped in (never in a menu, never chainable)."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from g1.core.action import Obs, Segment
from g1.core.config import CONTROL_DT, JOINT_NAMES
from g1.tools.base import Tool, limit, num, text


class Hold(Tool):
    name = "hold"
    order = 30
    prompt = ("Stand still for seconds (up to {limit:hold_seconds_max}, default 1) keeping the current pose. Use it "
              "to wait for a moving obstacle or a person to clear, or to observe again without moving.")
    params = {"seconds": num(0.1, limit("hold_seconds_max"), 1.0, "how long to wait")}

    def segments(self) -> tuple[Segment, ...]:
        return (Segment({}, self.seconds, label=f"hold {self.seconds:.1f} s"),)


def dry_run(tool: Tool, cmd: np.ndarray, joints: Sequence[int]) -> dict:
    """Plan a tool from the commanded pose through a JointMonitor without
    physics — joint limits, command speed, base limits, duration and the base
    displacement it would produce. Moves nothing. Every movement the model
    picks goes through this before it executes (their IK check before
    submission); ``check`` exposes it to the model."""
    from g1.envs.monitor import JointMonitor
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
    moved = [JOINT_NAMES[j] for j in mon.rows() if np.isfinite(mon.cmd_min[j]) and mon.cmd_max[j] - mon.cmd_min[j] > 5e-3]
    return {"status": "ok" if not mon.violations else "rejected",
            "tool": tool.name, "arguments": {k: v for k, v in tool.args.items() if k != "note"},
            "duration_s": n * CONTROL_DT, "base_delta": [x, y, math.degrees(yaw)],
            "peak_command_vel_rad_s": float(mon.peak_vel.max()), "joints_moved": moved,
            "violations": [str(v) for v in mon.violations[:12]], "n_violations": len(mon.violations)}


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

    def query(self, cmd: np.ndarray, joints: Sequence[int]) -> dict:
        return dry_run(self.target(), cmd, joints)


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
    params = {"ramp_s": num(0.0, 10.0, limit("bookend_ramp_s"), "weight ramp seconds")}

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
    params = {"ramp_s": num(0.0, 10.0, limit("bookend_ramp_s"), "weight ramp seconds")}

    def segments(self) -> tuple[Segment, ...]:
        if self.ramp_s <= 0:
            return ()
        return (Segment({}, self.ramp_s, weight=lambda a: 1.0 - a, label="releasing control"),)
