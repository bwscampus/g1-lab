"""A new tool, from ``g1 new tool NAME``. Read top to bottom, then edit.

A tool is a fixed motion computed from its arguments. The model picks it by
name and fills in the arguments; the host plans it, checks it against the
joint and speed limits, runs it, waits for the joints to settle and reports
what happened. It never watches the camera while it runs — the model does
that between tools — so it always ends.

Try it, then chain it, then let the model use it:

    g1 run --env sim --tools NAME --headless             # the defaults, fully checked
    g1 run --env sim --tools NAME:20:2:1:2 --headless    # positional arguments, in params order
    g1 run --env sim --tools NAME:angle_deg=10,hold:1,NAME:angle_deg=25 --headless
    g1 tools                                             # it is in the menu now
    mjpython -m g1 run --env sim --tools NAME            # watch it in the viewer

Every joint you may command, with its limits and stand value: ``g1 tools --joints``.
"""
from __future__ import annotations

import math

from g1.core.action import Segment
from g1.core.config import joint_index
from g1.tools.base import Tool, integer, limit, num

WAIST_PITCH = joint_index("waist_pitch")     # + bends forward; limits are +/- 0.52 rad


class ClassName(Tool):
    # -- what the model reads ----------------------------------------------------------------
    name = "NAME"
    # One paragraph: what it does, when to use it, what it needs clear. {limit:...} quotes a
    # limit from configs/limits.json so the text can never disagree with the host.
    prompt = ("Bow from the waist by angle_deg degrees (up to 30) over down_s seconds, hold the bow for hold_s, "
              "then straighten up over up_s. A greeting toward a person in front of the robot; needs the space "
              "in front of the chest clear. Not a search move.")
    # Arguments. A continuous quantity is a number with a range and a default (num); a count is
    # an integer; there are no slow/fast switches — the caller picks the seconds. A property
    # without a default is required. Every motion tool also gets a `note` automatically.
    params = {
        "angle_deg": num(5.0, 30.0, 15.0, "how far to bend forward, degrees"),
        "down_s": num(0.5, 5.0, 1.5, "seconds to bend down"),
        "hold_s": num(0.0, limit("hold_seconds_max"), 1.0, "seconds to stay bowed"),
        "up_s": num(0.5, 5.0, 1.5, "seconds to straighten up"),
        "reps": integer(1, 3, 1, "how many bows"),
    }
    # visible = False       # keep it out of the model's menu (CLI chains and replay only)
    # needs_base = True     # if it drives the base (Segment(..., base=(vx, vy, vyaw)))

    # -- what the robot runs --------------------------------------------------------------------
    def segments(self) -> tuple[Segment, ...]:
        """The motion, from the arguments. Each Segment goes to a pose (joint index -> radians,
        merged onto the previous pose) over a duration; the player eases between them at 50 Hz
        from wherever the previous tool left the joints. Touch only the joints your tool is
        about and leave them where it ends; if it needs a starting pose, make that its first
        segment (nothing else will put the arms anywhere)."""
        bowed = {WAIST_PITCH: math.radians(self.angle_deg)}
        out: list[Segment] = []
        for i in range(self.reps):
            out.append(Segment(bowed, self.down_s, label=f"bowing ({i + 1}/{self.reps})"))
            if self.hold_s > 0:
                out.append(Segment(bowed, self.hold_s, label="holding the bow"))
            out.append(Segment({WAIST_PITCH: 0.0}, self.up_s, label="straightening up"))
        return tuple(out)
