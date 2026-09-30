"""Test doubles. These never ship as user-facing modes: the product either asks
the real vision model or runs a preset; tests use these to run offline."""
from __future__ import annotations

import json
import math

import numpy as np

from g1.agent.decider import AgentTurn, Decider, Decision
from g1.core.config import HEAD_CAMERA_FOVY


def red_blob(image: np.ndarray, min_fraction: float = 0.002):
    """Centre of the red pixels as ``(u, v, fraction)`` with u, v in [-1, 1]
    (u positive to the right of the image, v down), or None when fewer than
    ``min_fraction`` of the pixels are red. The stand-in for a goal detector."""
    img = image.astype(np.int16)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mask = (r > 120) & (r - g > 60) & (r - b > 60)
    n = int(mask.sum())
    if n < min_fraction * mask.size:
        return None
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    u = 2.0 * xs.mean() / (w - 1) - 1.0
    v = 2.0 * ys.mean() / (h - 1) - 1.0
    return float(u), float(v), n / mask.size


def bearing(u: float, image_shape, fovy_deg: float = HEAD_CAMERA_FOVY) -> float:
    """Horizontal angle in rad (positive to the right) of normalised image column ``u``."""
    h, w = image_shape[:2]
    half_w = math.tan(math.radians(fovy_deg) / 2) * w / h
    return math.atan(u * half_w)


def elevation(v: float, image_shape, fovy_deg: float = HEAD_CAMERA_FOVY) -> float:
    """Vertical angle in rad (positive up) of normalised image row ``v`` (+1 at the bottom)."""
    return -math.atan(v * math.tan(math.radians(fovy_deg) / 2))


def select(decider: Decider, turn: AgentTurn, name: str, arguments: dict) -> Decision:
    """A test decider's reply, validated exactly like a model's."""
    return Decision.parse(json.dumps({"name": name, "arguments": arguments}), turn, decider.context)


class RedBallDecider(Decider):
    """Rule-based decider on the red blob (the test stand-in for a goal object):
    not seen -> turn 45 (or glance, without base tools); seen off-centre ->
    turn toward it; centred -> walk 0.5 m when the path is clear; reached
    (it looms past 3 % of the image, or sinks below -0.2 rad) -> done."""

    model = "red-ball-rules"
    reach_fraction = 0.03
    reach_elevation = -0.2

    def __init__(self) -> None:
        super().__init__(threaded=False)
        self._look_sign = 1.0
        self.calls: list[Decision] = []
        self.turns: list[AgentTurn] = []

    def decide(self, turn: AgentTurn) -> Decision:
        self.turns.append(turn)
        names = {t.name for t in self.context.menu}
        obs = json.loads(turn.observation)
        waist = obs["state"]["waist_yaw_deg"]
        image = turn.images.get("head")
        if image is None:
            return self._reply(turn, "hold", {"seconds": 0.5, "note": "no image; waiting"})
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
        u, v, frac = blob
        b = math.degrees(bearing(u, image.shape)) - waist          # base-relative, + right
        clear = frac < 0.2
        evidence = f"a red ball at {b:+.0f} deg"
        if frac >= self.reach_fraction or elevation(v, image.shape) <= self.reach_elevation:
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
    from g1.cli import run_parser
    from g1.envs import SimEnv
    args = run_parser().parse_args(["--env", "sim", "--tools", "x", "--headless", *extra])
    return (cls or SimEnv)(args)


def scripted_sim(camera, *extra):
    """A headless sim whose camera feed is scripted, recording every action."""
    from g1.envs import SimEnv

    class ScriptedSim(SimEnv):
        def setup(self):
            super().setup()
            self.replay = camera           # replaces the rendered feed (see SimEnv.frame)
            self.actions = []

        def step(self, action):
            self.actions.append(action)
            return super().step(action)

    return sim_env(*extra, cls=ScriptedSim)
