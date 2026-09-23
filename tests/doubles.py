"""Test doubles. These never ship as user-facing modes: the product either asks
the real vision model or runs a preset; tests use these to run offline."""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Sequence

from camera import Frame
from perception import Detected, Perceiver, Percept
from vision import bearing, elevation, red_blob


class FakePerceiver(Perceiver):
    """Inline stand-in: labels the red blob, or cycles scripted Percepts."""

    def __init__(self, label: str = "red ball", script: Sequence[Percept] | None = None,
                 min_interval: float = 0.0) -> None:
        super().__init__(threaded=False, min_interval=min_interval)
        self.label = label
        self.script = list(script or [])
        self._i = 0

    def describe(self, frame: Frame) -> Percept:
        if self.script:
            p = self.script[self._i % len(self.script)]
            self._i += 1
            return replace(p, frame_seq=frame.seq, frame_stamp=frame.stamp, seq=0, latency=0.0)
        blob = red_blob(frame.image)
        if blob is None:
            return Percept("nothing of interest", [], True, frame.seq, frame.stamp)
        u, v, frac = blob
        size = math.sqrt(frac)
        d = Detected(self.label, (u + 1) / 2, (v + 1) / 2, size, size, None,
                     bearing(u, frame.image.shape), elevation(v, frame.image.shape))
        return Percept(f"a {self.label} at {math.degrees(d.bearing):+.0f} deg", [d], frac < 0.2,
                       frame.seq, frame.stamp)


from decider import Context, Decider, Decision   # noqa: E402
from targets import RedDot, Sighting             # noqa: E402


class RedBallDecider(Decider):
    """Rule-based decider on the red blob (the test stand-in for a goal object):
    not seen -> turn 45 (or look, without base skills); seen off-centre -> turn
    toward it; centred -> walk 0.5 m when the path is clear; reached -> done."""

    model = "red-ball-rules"

    def __init__(self) -> None:
        super().__init__(threaded=False)
        self._look_sign = 1.0
        self.calls: list[Decision] = []

    def decide(self, ctx: Context) -> Decision:
        names = {s.name for s in ctx.skills}
        image = ctx.frame.image
        if image.shape[1] > 800:
            image = image[::2, ::2]
        blob = red_blob(image)
        if blob is None:
            scene, found, clear = "no red ball in view", False, True
            if "turn" in names:
                action, args = "turn", {"angle_deg": 45.0}
            elif "look" in names:
                action, args = "look", {"yaw_deg": 40.0 * self._look_sign}
                self._look_sign = -self._look_sign
            else:
                action, args = "done", {"found": False, "note": "cannot search"}
        else:
            s = Sighting.from_blob(*blob, ctx.frame)
            b = math.degrees(s.bearing) - ctx.waist_yaw_deg      # base-relative, + right
            scene, found, clear = f"a red ball at {b:+.0f} deg", True, blob[2] < 0.2
            if RedDot().reached(s):
                action, args = "done", {"found": True, "note": "reached"}
            elif abs(b) > 8:
                turn = max(-68.0, min(68.0, -b))
                action, args = ("turn", {"angle_deg": turn}) if "turn" in names else ("look", {"yaw_deg": max(-45.0, min(45.0, turn))})
            elif "walk_forward" in names and clear:
                action, args = "walk_forward", {"distance_m": 0.5}
            elif "turn" in names:
                action, args = "turn", {"angle_deg": 45.0}
            else:
                action, args = "done", {"found": True, "note": "cannot approach"}
        d = Decision.from_json({"scene": scene, "path_clear": clear, "found": found, "action": action,
                                "args": args, "reason": "rule"}, ctx, raw="rule")
        self.calls.append(d)
        return d
