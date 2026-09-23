"""The decider: given the goal, the current frame and joint state, choose the
next skill. Runs only when the agent asks (a Worker); the result lands a few
ticks later as a Decision. The scene description and the obstacle judgement
come from the same model call as the choice.

    HF_TOKEN=hf_... python -m decider step_0003.png --goal "find the mug"    # one real decision
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from camera import Frame
from hf import DEFAULT_MODEL, HFClient, extract_json, image_part
from skills import SKILLS, Skill, menu, validate_args
from worker import Worker

SYSTEM_PROMPT = """You control a Unitree G1 humanoid robot one step at a time. Each step you receive one image from
the robot's forward head camera (it points about 47 degrees down, so the floor 1-3 m ahead fills the
lower half of the image), the robot's state, a short history, and the list of skills you may run.
The robot stands still while you decide. Choose exactly ONE skill. It runs to completion (at most
3 seconds), the robot stops, and you get a fresh image.

Rules:
- First describe the scene in one sentence, then judge path_clear: could the robot walk 1 m
  straight ahead without touching anything?
- found = true only when the goal object is clearly visible in this image.
- If the goal object is visible and near (large, or in the lower third of the image), reply with
  action "done" and found = true. If it is visible but off-centre, "turn" toward it (angle_deg is
  positive to the LEFT, negative to the RIGHT). If it is centred and far, "walk_forward".
- Never choose "walk_forward" when path_clear is false; turn instead.
- To search, turn LEFT in steps of 45-60 degrees. After a full circle without seeing the goal,
  walk 0.5 m into clear space and search again. Use "look" only when "turn" is not offered.
- Reply "done" with found = false when the goal cannot be found or moving on is unsafe.
- Use only the listed skill names and argument names; keep numbers within their ranges.
Reply with ONE JSON object and nothing else, exactly:
{"scene": "...", "path_clear": true|false, "found": true|false,
 "action": "<skill>", "args": {...}, "reason": "..."}"""


@dataclass
class Context:
    goal: str
    step: int
    max_steps: int
    frame: Frame
    q: np.ndarray
    waist_yaw_deg: float
    base_delta: tuple[float, float, float]      # x m forward, y m left, yaw deg since episode start
    history: list[dict]                         # {"step", "action", "args", "outcome", "scene"}
    skills: list[Skill]
    note: str = ""                              # e.g. why the previous reply was rejected


@dataclass
class Decision:
    scene: str
    path_clear: bool
    found: bool
    action: str
    args: dict
    reason: str
    step: int
    frame_seq: int
    frame_stamp: float
    raw: str = ""
    latency: float = 0.0
    seq: int = 0
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, data: dict, ctx: Context, raw: str = "") -> "Decision":
        action = str(data.get("action") or "")
        allowed = {s.name: s for s in ctx.skills}
        if action not in allowed:
            raise ValueError(f"unknown action {action!r}; choose one of {sorted(allowed)}")
        args = data.get("args") or {}
        if not isinstance(args, dict):
            raise ValueError("args must be an object")
        args, notes = validate_args(allowed[action], args)
        return cls(str(data.get("scene") or ""), bool(data.get("path_clear", True)),
                   bool(data.get("found", False)), action, args, str(data.get("reason") or ""),
                   ctx.step, ctx.frame.seq, ctx.frame.stamp, raw, notes=notes)

    def to_json(self) -> dict:
        return {"scene": self.scene, "path_clear": self.path_clear, "found": self.found,
                "action": self.action, "args": self.args, "reason": self.reason, "raw": self.raw,
                "latency": self.latency, "notes": list(self.notes)}


class Decider(Worker[Context, Decision]):
    model: str = "decider"

    def __init__(self, *, threaded: bool = True, min_interval: float = 0.0) -> None:
        super().__init__(threaded=threaded, min_interval=min_interval, name="decider")

    def decide(self, ctx: Context) -> Decision:
        raise NotImplementedError

    def process(self, ctx: Context) -> Decision:
        return self.decide(ctx)


class HFDecider(Decider):
    """Asks the Hugging Face vision model to pick the next skill."""

    def __init__(self, client: HFClient, *, max_width: int = 640, min_interval: float = 0.0) -> None:
        super().__init__(min_interval=min_interval)
        self.client = client
        self.max_width = max_width

    @classmethod
    def from_env(cls, model: Optional[str] = None, **client_kw) -> "HFDecider":
        return cls(HFClient.from_env(model, **client_kw))

    @property
    def model(self) -> str:      # type: ignore[override]
        return self.client.model

    def build_messages(self, ctx: Context) -> list[dict]:
        x, y, yaw = ctx.base_delta
        side = "left" if ctx.waist_yaw_deg > 2 else "right" if ctx.waist_yaw_deg < -2 else "straight ahead"
        lines = [f"Goal: {ctx.goal}", f"Step {ctx.step} of {ctx.max_steps}.",
                 f"State: waist yaw {ctx.waist_yaw_deg:+.0f} deg (the camera looks {side}); moved "
                 f"{x:+.2f} m forward, {y:+.2f} m left, turned {yaw:+.0f} deg since the start."]
        if ctx.history:
            lines.append("History (most recent last):")
            lines += [f"  #{h['step']} {h['action']}({json.dumps(h['args'])}) -> {h['outcome']}: {h['scene']}"
                      for h in ctx.history]
        lines.append("Skills: " + json.dumps([s.schema() for s in ctx.skills]))
        if ctx.note:
            lines.append(f"Note: {ctx.note}")
        lines.append("Decide the next step.")
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "text", "text": "\n".join(lines)},
                                             image_part(ctx.frame.image, self.max_width)]}]

    def decide(self, ctx: Context) -> Decision:
        text = self.client.complete(self.build_messages(ctx), max_tokens=300)
        return Decision.from_json(extract_json(text), ctx, raw=text)


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m decider",
                                description="one real decision from a saved frame (needs $HF_TOKEN)")
    p.add_argument("image", help="png/jpg, e.g. a step_NNNN.png from runs/")
    p.add_argument("--goal", required=True)
    p.add_argument("--model", default=None, help=f"HF model id (default: $G1_VISION_MODEL or {DEFAULT_MODEL})")
    p.add_argument("--no-base", action="store_true", help="hide the walking skills, as on a robot without --walk")
    args = p.parse_args(argv)
    import cv2
    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        p.error(f"could not read {args.image}")
    frame = Frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), 0.0, 1)
    try:
        dec = HFDecider.from_env(args.model, on_text=lambda s: print(s, end="", flush=True))
    except RuntimeError as e:
        p.error(str(e))
    ctx = Context(args.goal, 1, 30, frame, np.zeros(29), 0.0, (0.0, 0.0, 0.0), [], menu(not args.no_base))
    print(f"model {dec.model}; goal {args.goal!r}; image {frame.image.shape[1]}x{frame.image.shape[0]}")
    t0 = time.monotonic()
    try:
        d = dec.decide(ctx)
    except ValueError as e:
        print(f"\n-- rejected after {time.monotonic() - t0:.2f}s: {e}")
        return 1
    print(f"\n-- {time.monotonic() - t0:.2f}s")
    print(f"scene:      {d.scene}\npath_clear: {d.path_clear}\nfound:      {d.found}")
    print(f"action:     {d.action}({d.args})  {' '.join(d.notes)}\nreason:     {d.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
