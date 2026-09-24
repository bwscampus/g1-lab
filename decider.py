"""The decider: the model I/O contract of the decision step, in GPT-Policy's
shape, and the session that carries it.

  AgentContext   the system prompt (conventions, safety notes, the skill
                 catalog as bullets and as JSON, the rules), the function
                 schemas and the output schema — built once per run
  AgentTurn      one observation: the JSON text (``observation()``), the
                 named images, and on turn 0 any demonstration content
  Decision       the validated ``{"name", "arguments"}`` reply

The observation is one JSON object: ``instruction``, ``images`` (name, size,
age), ``state`` (measured joint_pos / joint_vel / joint_torque, waist yaw,
the commanded and measured base pose), ``extra`` (env_step, decisions_left,
can_walk) and, after the first turn, ``previous_result`` — what the last
skill did (residuals, base error, a measured settle report) or why it was
rejected. The reply must name a skill from the menu and satisfy its parameter
schema (jsonschema, Draft 2020-12); every movement skill carries a ``note``.

``VLMDecider`` keeps one conversation per run: the system prompt once, then
each observation and the model's own reply, so the model has its history.
Only the last ``live_image_window`` observations keep their image (their
``live_image_window``); demonstration images are never dropped.

    HF_TOKEN=hf_... python -m decider step_0003.png --goal "find the mug"    # one real decision (or VLM_PROVIDER=...)
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
from jsonschema import Draft202012Validator, ValidationError

from config import HEAD_CAMERA_FOVY, NUM_JOINTS
from vlm import DEFAULT_MODEL, VLMClient, image_part
from skills import CATALOG, Catalog, Skill, menu
from worker import Worker

MAX_TOKENS = 400


class ProtocolError(ValueError):
    """The reply was not a valid skill selection. Carries the raw text."""

    def __init__(self, message: str, raw: str = "") -> None:
        super().__init__(message)
        self.raw = raw


# --------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentContext:
    instructions: str
    tools: list[dict]
    output_schema: dict
    menu: tuple[type[Skill], ...]

    def record(self) -> dict:
        return {"base_instructions": self.instructions, "tools": self.tools,
                "output_schema": self.output_schema, "menu": [s.name for s in self.menu]}


@dataclass
class AgentTurn:
    observation: str                                 # the JSON text, exactly as sent
    images: dict[str, np.ndarray] = field(default_factory=dict)   # name -> RGB frame
    content: tuple = ()                              # demonstration parts, turn 0 only
    request_id: int = 0
    step: int = 0
    frame_seq: Optional[int] = None
    frame_stamp: Optional[float] = None


@dataclass
class Decision:
    name: str
    arguments: dict
    request_id: int = 0
    step: int = 0
    frame_seq: Optional[int] = None
    frame_stamp: Optional[float] = None
    raw: str = ""
    wire: Optional[dict] = None
    notes: list[str] = field(default_factory=list)   # range clamps applied by the host
    latency: float = 0.0
    seq: int = 0

    @property
    def note(self) -> str:
        return str(self.arguments.get("note", ""))

    def to_json(self) -> dict:
        return {"name": self.name, "arguments": self.arguments, "raw": self.raw,
                "latency": self.latency, "notes": list(self.notes)}

    @classmethod
    def parse(cls, text: str, turn: AgentTurn, context: AgentContext,
              catalog: Catalog = CATALOG) -> "Decision":
        """Validate a reply the way their ``parse_decision`` does: a whole
        JSON object (a complete ```json fence is fine, fragments are not), the
        output schema, then the chosen skill's own parameter schema. Numbers
        out of range are clamped with a note rather than rejected."""
        from skills import validate_args
        try:
            wire = parse_selection(text)
            name = wire.get("name")
            arguments = wire.get("arguments")
            if isinstance(arguments, str):
                arguments = parse_selection(arguments)
            if not isinstance(name, str) or not isinstance(arguments, dict):
                raise ValueError("a selection has a name string and an arguments object")
            allowed = {s.name: s for s in context.menu}
            if name not in allowed:
                raise ValueError(f"unknown skill {name!r}; choose one of {sorted(allowed)}")
            skill = allowed[name]
            arguments, notes = validate_args(skill, arguments)
            schema = catalog.parameters(skill, context.menu)
            Draft202012Validator(schema).validate(arguments)
            json.dumps(arguments, allow_nan=False)
        except (ValueError, TypeError, ValidationError) as e:
            msg = e.message if isinstance(e, ValidationError) else str(e)
            if isinstance(e, ValidationError) and e.absolute_path:
                msg = f"{'.'.join(str(p) for p in e.absolute_path)}: {msg}"
            raise ProtocolError(f"invalid selection: {msg}", raw=text) from None
        return cls(name, arguments, turn.request_id, turn.step, turn.frame_seq, turn.frame_stamp,
                   raw=text, wire=wire, notes=notes)


def parse_selection(text: str) -> dict:
    """One JSON object. A complete enclosing fence is accepted; a fragment
    inside prose is not (their rule: never extract arbitrary fragments)."""
    body = text.strip()
    lines = body.splitlines()
    if len(lines) >= 3 and lines[0].strip() in ("```", "```json") and lines[-1].strip() == "```":
        body = "\n".join(lines[1:-1])
    try:
        data = json.loads(body, parse_constant=_no_constants)
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"not a JSON object: {e}") from None
    if not isinstance(data, dict):
        raise ValueError("the reply is not a JSON object")
    return data


def _no_constants(name: str):
    raise ValueError(f"non-finite number {name}")


# --------------------------------------------------------------------------
# The observation and the system prompt
# --------------------------------------------------------------------------

def _rounded(value: Any, digits: int = 6) -> Any:
    """1e-6 for the model; the record keeps full precision."""
    if isinstance(value, dict):
        return {k: _rounded(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(v, digits) for v in value]
    if isinstance(value, np.ndarray):
        return [_rounded(float(v), digits) for v in value.tolist()]
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return round(v, digits) if math.isfinite(v) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def observation(instruction: str, state: dict, images: list[dict], extra: dict,
                previous: Optional[dict] = None) -> str:
    """The one JSON object a turn sends (their ``protocol.observation``)."""
    payload: dict[str, Any] = {"instruction": instruction, "images": images, "state": state, "extra": extra}
    if previous is not None:
        payload["previous_result"] = previous
    return json.dumps(_rounded(payload), ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def instructions(context_menu: Sequence[type[Skill]], *, can_walk: bool, max_decisions: int,
                 safety_notes: Sequence[str] = (), catalog: Catalog = CATALOG) -> str:
    """The system prompt, once per run: conventions, hidden facts, the rules,
    the skill catalog as bullets and, appended, as JSON."""
    notes = ""
    if safety_notes:
        lines = "\n".join(f"- {n.strip()}" for n in safety_notes)
        notes = (f"\nFixed scene and hidden obstacles (persistent physical facts):\n{lines}\n"
                 "- An obstacle absent from the image is not absent from the room. Keep clear of these "
                 "throughout every walk and turn.\n")
    walk = ("Walking skills drive the base; they are offered because the base may be driven now."
            if can_walk else
            "The base cannot be driven in this run, so no walking skills are offered: search by looking.")
    fov_h = 2 * math.degrees(math.atan(math.tan(math.radians(HEAD_CAMERA_FOVY / 2)) * 4 / 3))
    joints = ""
    if any(s.name == "arm_path" for s in context_menu):
        from skills import joint_table
        rows = "\n".join(f"- {r['name']}: [{r['min']}, {r['max']}] rad, stand {r['stand']}" for r in joint_table())
        joints = f"""
Joints arm_path may command (radians; the stand pose is where every run starts and ends):
{rows}
- Conventions verified on the model: elbow 0 is a 90-degree bend with the forearm forward, about 1.57 is a straight arm, more negative bends the forearm up; shoulder_roll +1.57 (left) / -1.57 (right) with shoulder_pitch 0 and elbow 1.47 is a T-pose; wrist_roll -1.57 (left) / +1.57 (right) turns the palms up; waist_yaw positive looks LEFT.
"""
    text = f"""You control a Unitree G1 humanoid robot through a small set of skills, one decision per turn.

Robot and camera conventions:
- {NUM_JOINTS} joints in DDS order (legs 0-11, waist 12-14, left arm 15-21, right arm 22-28), radians. state.joint_pos, joint_vel and joint_torque are measured at the observation time. The host commands only the waist and arms (arm_sdk) and, for walking skills, the base velocity (LocoClient); the legs balance on their own.
- One forward head camera (RGB, no depth) on the torso, pointing 47 degrees down: the floor 1-3 m ahead fills the lower half of the image, and the image spans about {fov_h:.0f} degrees horizontally. Pixel positions are not metric distances; judge distance from apparent size and from where an object meets the floor.
- +yaw is LEFT (counter-clockwise). state.base_pose_cmd is the dead-reckoned [x forward, y left, yaw degrees] since the start, integrated from the commands sent; base_pose_env is the simulator's own measurement and null on the real robot. state.waist_yaw_deg is where the camera looks relative to the feet.
- {walk}
{joints}{notes}
Return exactly one skill selection per turn: {{"name": "...", "arguments": {{...}}}}. No Markdown or text outside this object. The host runs the whole skill, waits for the joints to settle, then supplies a fresh observation whose previous_result reports what happened.

Skills:
{catalog.prompt_catalog(context_menu)}

Every movement skill requires a note: 1-2 short sentences stating the current visual evidence and the purpose of this action. Keep failure causes when relevant; do not repeat numbers, history, skill mechanics or these rules. Use only the selected skill's arguments, within their ranges.

Observe before committing: a target that is off-centre needs a turn before a walk; a walk needs a clear floor for its whole distance; never walk toward something you cannot see the floor in front of. previous_result.execution_feedback reports target-versus-measured errors and settle after each skill; settled means the joints are still, not that the goal is reached. A rejected selection (previous_result.error) ran nothing: fix the selection, do not repeat it.

Persistence: one failed skill, blocked path or unseen goal does not establish impossibility. Diagnose from the fresh image, the state and previous_result, then try a different viewpoint, direction or step size and verify the result. Call done only when the current image establishes the goal. Do not call give_up while reasonable safe strategies remain; when you do, list what was tried and the evidence preventing progress.

Budget: {max_decisions} decisions for this run; extra.decisions_left counts down and a rejected selection costs one.

Robot skill catalog:
{json.dumps(catalog.function_schemas(context_menu), ensure_ascii=False)}"""
    return text


def build_context(context_menu: Sequence[type[Skill]], *, can_walk: bool, max_decisions: int,
                  safety_notes: Sequence[str] = (), catalog: Catalog = CATALOG) -> AgentContext:
    m = tuple(context_menu)
    return AgentContext(instructions(m, can_walk=can_walk, max_decisions=max_decisions,
                                     safety_notes=safety_notes, catalog=catalog),
                        catalog.function_schemas(m), catalog.output_schema(m), m)


# --------------------------------------------------------------------------
# Deciders
# --------------------------------------------------------------------------

class Decider(Worker[AgentTurn, Decision]):
    """``start(context)`` once per run, then ``request(turn)`` per decision; the
    result lands in ``latest()`` a few ticks later. ``last_call`` describes the
    most recent model call (status, elapsed, usage) for the usage ledger."""

    model: str = "decider"
    provider: str = "test"

    def __init__(self, *, threaded: bool = True, min_interval: float = 0.0) -> None:
        super().__init__(threaded=threaded, min_interval=min_interval, name="decider")
        self.context: Optional[AgentContext] = None
        self.last_call: Optional[dict] = None
        self.n_calls = 0

    def start(self, context: AgentContext) -> None:
        self.context = context

    def decide(self, turn: AgentTurn) -> Decision:
        raise NotImplementedError

    def process(self, turn: AgentTurn) -> Decision:
        if self.context is None:
            raise RuntimeError("decider.start(context) was not called")
        self.n_calls += 1
        t0 = time.monotonic()
        call = {"call": self.n_calls, "at_s": time.time(), "provider": self.provider, "model": self.model,
                "step": turn.step, "request_id": turn.request_id, "status": "completed", "usage": None}
        try:
            return self.decide(turn)
        except BaseException as e:
            call["status"] = "failed"
            call["error_type"] = type(e).__name__
            raise
        finally:
            call["elapsed_s"] = round(time.monotonic() - t0, 6)
            call.update(self._call_details())
            self.last_call = call

    def _call_details(self) -> dict:
        return {}


class VLMDecider(Decider):
    """Asks the vision-language model (any OpenAI-compatible endpoint, see
    ``vlm.py``), keeping the conversation."""

    def __init__(self, client: VLMClient, *, max_width: int = 640, live_image_window: Optional[int] = 8,
                 fresh_turns: bool = False, min_interval: float = 0.0) -> None:
        super().__init__(min_interval=min_interval)
        if live_image_window is not None and live_image_window < 1:
            raise ValueError("live_image_window must be >= 1 or None")
        self.client = client
        self.max_width = max_width
        self.live_image_window = live_image_window
        self.fresh_turns = fresh_turns
        self._turns: list[dict] = []          # {"user": msg, "assistant": msg | None, "image_at": int | None}

    @classmethod
    def from_env(cls, model: Optional[str] = None, *, live_image_window: Optional[int] = 8,
                 fresh_turns: bool = False, **client_kw) -> "VLMDecider":
        return cls(VLMClient.from_env(model, **client_kw), live_image_window=live_image_window,
                   fresh_turns=fresh_turns)

    @property
    def model(self) -> str:      # type: ignore[override]
        return self.client.model

    @property
    def provider(self) -> str:   # type: ignore[override]
        return self.client.provider

    def start(self, context: AgentContext) -> None:
        super().start(context)
        self._turns = []

    # -- the conversation --------------------------------------------------
    def _user_message(self, turn: AgentTurn) -> tuple[dict, Optional[int]]:
        blocks: list[dict] = []
        for part in turn.content:
            blocks.extend(part.blocks(self.max_width))
        blocks.append({"type": "text", "text": turn.observation})
        image_at = None
        for name, image in turn.images.items():
            blocks.append({"type": "text", "text": f"Camera image: {name}"})
            image_at = len(blocks)
            blocks.append(image_part(image, self.max_width))
        return {"role": "user", "content": blocks}, image_at

    def _prune(self, incoming: int = 0) -> None:
        """Keep the newest ``live_image_window`` observation images, counting
        the ``incoming`` turn about to be sent (their refresh happens before a
        turn, so what is sent never carries more than the window)."""
        if self.live_image_window is None:
            return
        live = [t for t in self._turns if t["image_at"] is not None]
        for t in live[:max(0, len(live) + incoming - self.live_image_window)]:
            i = t["image_at"]
            content = list(t["user"]["content"])
            del content[i - 1:i + 1]         # the "Camera image:" label and the image itself
            content.append({"type": "text", "text": "(camera image omitted from history)"})
            t["user"] = {"role": "user", "content": content}    # a new message: what was sent stays as sent
            t["image_at"] = None

    def messages(self, turn: Optional[AgentTurn] = None) -> list[dict]:
        assert self.context is not None
        out = [{"role": "system", "content": self.context.instructions}]
        if not self.fresh_turns:
            for t in self._turns:
                out.append(t["user"])
                if t["assistant"] is not None:
                    out.append(t["assistant"])
        if turn is not None:
            out.append(self._user_message(turn)[0])
        return out

    def transcript(self) -> list[dict]:
        """The conversation as sent, images replaced by their labels."""
        def strip(msg):
            if isinstance(msg["content"], str):
                return msg
            blocks = [b if b["type"] == "text" else {"type": "image"} for b in msg["content"]]
            return {"role": msg["role"], "content": blocks}
        return [strip(m) for m in self.messages()]

    def decide(self, turn: AgentTurn) -> Decision:
        assert self.context is not None
        user, image_at = self._user_message(turn)
        if not self.fresh_turns:
            self._prune(incoming=1 if image_at is not None else 0)
        msgs = self.messages()
        msgs.append(user)
        text = self.client.complete(msgs, max_tokens=MAX_TOKENS, schema=self.context.output_schema)
        entry = {"user": user, "assistant": {"role": "assistant", "content": text}, "image_at": image_at}
        if not self.fresh_turns:
            self._turns.append(entry)
        return Decision.parse(text, turn, self.context)

    def _call_details(self) -> dict:
        return {"usage": self.client.last_usage, "response_mode": self.client.response_mode,
                "request_elapsed_s": round(self.client.last_elapsed, 6)}


# --------------------------------------------------------------------------
# One decision from a saved frame
# --------------------------------------------------------------------------

def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m decider",
                                description="one real decision from a saved frame (needs the provider's key, "
                                            "see vlm.py: VLM_PROVIDER / VLM_MODEL / VLM_BASE_URL)")
    p.add_argument("image", help="png/jpg, e.g. a step_NNNN.png from runs/")
    p.add_argument("--goal", required=True)
    p.add_argument("--model", default=None, help="model id (default: $VLM_MODEL, or the provider's default)")
    p.add_argument("--provider", default=None, help="VLM provider (default: $VLM_PROVIDER or huggingface)")
    p.add_argument("--no-base", action="store_true", help="hide the walking skills, as on a robot without --walk")
    args = p.parse_args(argv)
    import cv2
    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        p.error(f"could not read {args.image}")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    try:
        dec = VLMDecider.from_env(args.model, provider=args.provider, on_text=lambda s: print(s, end="", flush=True))
    except RuntimeError as e:
        p.error(str(e))
    context = build_context(menu(not args.no_base), can_walk=not args.no_base, max_decisions=30)
    dec.start(context)
    h, w = image.shape[:2]
    state = {"joint_pos": [0.0] * NUM_JOINTS, "joint_vel": [0.0] * NUM_JOINTS, "joint_torque": None,
             "waist_yaw_deg": 0.0, "base_pose_cmd": [0.0, 0.0, 0.0], "base_pose_env": None}
    obs = observation(args.goal, state, [{"name": "head", "width": w, "height": h, "captured_age_s": 0.0}],
                      {"env_step": 0, "decisions_left": 30, "can_walk": not args.no_base})
    print(f"{dec.provider}: model {dec.model}; goal {args.goal!r}; image {w}x{h}")
    t0 = time.monotonic()
    try:
        d = dec.decide(AgentTurn(obs, {"head": image}))
    except ProtocolError as e:
        print(f"\n-- rejected after {time.monotonic() - t0:.2f}s: {e}")
        return 1
    print(f"\n-- {time.monotonic() - t0:.2f}s ({dec.client.response_mode}, usage {dec.client.last_usage})")
    print(f"skill:  {d.name}({ {k: v for k, v in d.arguments.items() if k != 'note'} })  {' '.join(d.notes)}")
    print(f"note:   {d.note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
