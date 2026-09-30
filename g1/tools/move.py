"""Base motion: a segment that holds a base velocity (LocoClient.Move on the
robot, a slide of the pinned pelvis in sim) and leaves every joint where it
is. Durations round up to whole ticks so ``distance = v * t`` is exact and the
dead-reckoned end pose is the request."""
from __future__ import annotations

import math

from g1.core import limits
from g1.core.action import Segment
from g1.core.config import CONTROL_DT
from g1.tools.base import Tool, limit, num

WALK_SPEED = limits.get("walk_speed")
SIDE_SPEED = limits.get("side_speed")
TURN_RATE = limits.get("turn_rate")
MIN_SEGMENT = limits.get("min_segment_s")   # a shorter entry segment could recentre a held waist too fast


def ticks(seconds: float) -> float:
    """Round a duration up to whole control ticks (and at least MIN_SEGMENT), so a
    velocity held for it integrates to exactly the intended distance or angle."""
    return max(math.ceil(seconds / CONTROL_DT - 1e-9), round(MIN_SEGMENT / CONTROL_DT)) * CONTROL_DT


class Move(Tool):
    """One relative base displacement in the start frame, GPT-Policy's
    ``move_to`` for a base: translate at (vx, vy), then rotate, so the
    dead-reckoned end pose is exactly (dx, dy, dyaw). Zero parts are skipped."""

    name = "move"
    order = 10
    needs_base = True
    prompt = ("Move the base by dx_m forward (up to {limit:move_dx_max_m} m either way; negative walks backward), "
              "dy_m left (up to {limit:move_dy_max_m} m either way; a sidestep) and dyaw_deg (up to "
              "{limit:move_dyaw_max_deg} degrees either way, positive turns LEFT), in the current frame. The host "
              "translates at {limit:walk_speed} m/s then turns at {limit:turn_rate} rad/s, observing every "
              "{limit:record_step_s} s but not re-deciding. Only when the note states the floor is clear for the "
              "whole displacement. To face a visible target, turn by its bearing (the image spans about 77 degrees "
              "horizontally). To search, turn LEFT in steps of 45-60 degrees and observe between turns; after a full "
              "circle without seeing the goal, move 0.5 m into clear space and search again. To pass an obstacle, "
              "sidestep or turn around it in short moves rather than walking at it.")
    params = {
        "dx_m": num(-limit("move_dx_max_m"), limit("move_dx_max_m"), 0.0, "metres forward (negative: backward)"),
        "dy_m": num(-limit("move_dy_max_m"), limit("move_dy_max_m"), 0.0, "metres to the left (negative: right)"),
        "dyaw_deg": num(-limit("move_dyaw_max_deg"), limit("move_dyaw_max_deg"), 0.0, "degrees to turn, + left / - right"),
    }

    def segments(self) -> tuple[Segment, ...]:
        out = []
        if abs(self.dx_m) > 1e-6 or abs(self.dy_m) > 1e-6:
            duration = ticks(max(abs(self.dx_m) / WALK_SPEED, abs(self.dy_m) / SIDE_SPEED))
            out.append(Segment({}, duration, base=(self.dx_m / duration, self.dy_m / duration, 0.0),
                               label=f"move {self.dx_m:+.2f} m forward, {self.dy_m:+.2f} m left"))
        rad = math.radians(self.dyaw_deg)
        if abs(rad) > 1e-6:
            duration = ticks(abs(rad) / TURN_RATE)
            out.append(Segment({}, duration, base=(0.0, 0.0, rad / duration),
                               label=f"turn {self.dyaw_deg:+.0f} deg"))
        if not out:
            out.append(Segment({}, MIN_SEGMENT, label="move 0"))
        return tuple(out)


class WalkForward(Tool):
    """A preset (CLI chains and replay): ``move`` with only dx."""

    name = "walk_forward"
    order = 60
    visible = False
    needs_base = True
    prompt = ("Walk straight ahead by distance_m metres (0.1 to {limit:move_dx_max_m}) at {limit:walk_speed} m/s, "
              "observed every {limit:record_step_s} s but not re-decided.")
    params = {"distance_m": num(0.1, limit("move_dx_max_m"), None, "metres to walk")}

    def segments(self) -> tuple[Segment, ...]:
        duration = ticks(self.distance_m / WALK_SPEED)
        v = self.distance_m / duration
        return (Segment({}, duration, base=(v, 0.0, 0.0), label=f"walk {self.distance_m:.2f} m"),)


class Turn(Tool):
    """A preset: ``move`` with only dyaw."""

    name = "turn"
    order = 61
    visible = False
    needs_base = True
    prompt = ("Turn in place by angle_deg degrees (up to {limit:move_dyaw_max_deg} either way); positive turns "
              "LEFT, negative RIGHT.")
    params = {"angle_deg": num(-limit("move_dyaw_max_deg"), limit("move_dyaw_max_deg"), None, "degrees, + left / - right")}

    def segments(self) -> tuple[Segment, ...]:
        rad = math.radians(self.angle_deg)
        if abs(rad) < 1e-6:
            return (Segment({}, MIN_SEGMENT, label="turn 0"),)
        duration = ticks(abs(rad) / TURN_RATE)
        return (Segment({}, duration, base=(0.0, 0.0, rad / duration),
                        label=f"turn {self.angle_deg:+.0f} deg"),)
