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


import json                                      # noqa: E402

from decider import AgentTurn, Decider, Decision   # noqa: E402
from targets import RedDot, Sighting             # noqa: E402


def select(decider: Decider, turn: AgentTurn, name: str, arguments: dict) -> Decision:
    """A test decider's reply, validated exactly like a model's."""
    return Decision.parse(json.dumps({"name": name, "arguments": arguments}), turn, decider.context)


class RedBallDecider(Decider):
    """Rule-based decider on the red blob (the test stand-in for a goal object):
    not seen -> turn 45 (or look, without base skills); seen off-centre -> turn
    toward it; centred -> walk 0.5 m when the path is clear; reached -> done."""

    model = "red-ball-rules"

    def __init__(self) -> None:
        super().__init__(threaded=False)
        self._look_sign = 1.0
        self.calls: list[Decision] = []
        self.turns: list[AgentTurn] = []

    def decide(self, turn: AgentTurn) -> Decision:
        self.turns.append(turn)
        names = {s.name for s in self.context.menu}
        obs = json.loads(turn.observation)
        waist = obs["state"]["waist_yaw_deg"]
        image = turn.images.get("head")
        if image is None:
            return self._reply(turn, "hold", {"seconds": 0.5, "note": "no image; waiting"})
        frame = Frame(image, turn.frame_stamp or 0.0, turn.frame_seq or 0)
        if image.shape[1] > 800:
            image = image[::2, ::2]
        blob = red_blob(image)
        if blob is None:
            if "move" in names:
                return self._reply(turn, "move", {"dyaw_deg": 45.0, "note": "no red ball in view; searching left"})
            if "arm_path" in names:
                yaw = 0.7 * self._look_sign
                self._look_sign = -self._look_sign
                return self._reply(turn, "arm_path", {"waypoints": [{"joints": {"waist_yaw": yaw}, "seconds": 1.5}],
                                                      "note": "no red ball in view; glancing"})
            return self._reply(turn, "give_up", {"reason": "cannot search", "hindsight": ""})
        s = Sighting.from_blob(*blob, frame)
        b = math.degrees(s.bearing) - waist          # base-relative, + right
        clear = blob[2] < 0.2
        evidence = f"a red ball at {b:+.0f} deg"
        if RedDot().reached(s):
            return self._reply(turn, "done", {"summary": f"reached: {evidence}", "hindsight": ""})
        if abs(b) > 8:
            turn_deg = max(-68.0, min(68.0, -b))
            if "move" in names:
                return self._reply(turn, "move", {"dyaw_deg": turn_deg, "note": evidence + "; facing it"})
            yaw = math.radians(max(-45.0, min(45.0, turn_deg)))
            return self._reply(turn, "arm_path", {"waypoints": [{"joints": {"waist_yaw": yaw}, "seconds": 1.5}],
                                                  "note": evidence})
        if "move" in names and clear:
            return self._reply(turn, "move", {"dx_m": 0.5, "note": evidence + "; floor clear"})
        if "move" in names:
            return self._reply(turn, "move", {"dyaw_deg": 45.0, "note": evidence + "; path blocked"})
        return self._reply(turn, "done", {"summary": "cannot approach: " + evidence, "hindsight": ""})

    def _reply(self, turn: AgentTurn, name: str, arguments: dict) -> Decision:
        d = select(self, turn, name, arguments)
        self.calls.append(d)
        return d


def sim_env(*extra, cls=None):
    """A headless sim env: the fast, fully checked stage the tests run in."""
    from envs import SimEnv
    from run import build_parser
    args = build_parser().parse_args(["--env", "sim", "--policy", "x", "--headless", *extra])
    return (cls or SimEnv)(args)


def scripted_sim(camera, *extra):
    """A headless sim whose camera feed is scripted, recording every action."""
    from envs import SimEnv

    class ScriptedSim(SimEnv):
        def setup(self):
            super().setup()
            self.replay = camera           # replaces the rendered feed (see SimEnv.frame)
            self.actions = []

        def step(self, action):
            self.actions.append(action)
            return super().step(action)

    return sim_env(*extra, cls=ScriptedSim)
