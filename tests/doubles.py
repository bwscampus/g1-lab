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
