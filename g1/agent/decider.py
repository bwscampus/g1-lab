"""The decider: the model I/O contract of the decision step, in GPT-Policy's
shape, and the session that carries it.

  AgentContext   the system prompt (conventions, the task's rules, the tool
                 catalog as bullets and as JSON, the reply rules), the function
                 schemas and the output schema — built once per run
  AgentTurn      one observation: the JSON text (``observation()``) and the
                 named images
  Decision       the validated ``{"name", "arguments"}`` reply

The observation is one JSON object: ``instruction``, ``images`` (name, size,
age, as sent), ``state`` (measured joint_pos / joint_vel / joint_torque, the
commanded waist yaw and base pose), ``extra`` (env_step, decisions_left, can_walk)
and, after the first turn, ``previous_result`` — what the last tool did (residuals,
the commanded base displacement, a measured settle report) or why it was rejected.
Only what the robot knows: never the room or the twin's truth. The reply must name a tool from the menu and satisfy its parameter
schema (jsonschema, Draft 2020-12); every movement tool carries a ``note``.

``VLMDecider`` keeps one conversation per run: the system prompt once, then
the newest ``history_full_turns`` turns in full (observation and the model's own
reply) after one host-written summary line per older turn, as GPT-Policy's
Claude adapter does. Only the last ``live_image_window`` observations keep
their image.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
from jsonschema import Draft202012Validator, ValidationError

from g1.core import limits
from g1.core.config import HEAD_CAMERA_FOV_H, HEAD_CAMERA_PITCH, NUM_JOINTS
from g1.core.worker import Worker
from g1.tools import Tool, function_schemas, joint_table, menu, output_schema, parameters, prompt_catalog, validate_args
from g1.vlm import AGENT_SETTINGS, IMAGE_WIDTH, TEMPERATURE, VLMClient, image_part



class ProtocolError(ValueError):
    """The reply was not a valid tool selection. Carries the raw text."""

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
    menu: tuple[type[Tool], ...]

    def record(self) -> dict:
        """For config.json: the system prompt, the output schema and the tools on offer."""
        return {"menu": [s.name for s in self.menu], "tools": self.tools,
                "system_prompt": self.instructions, "output_schema": self.output_schema}


@dataclass
class AgentTurn:
    observation: str                                 # the JSON text, exactly as sent
    images: dict[str, np.ndarray] = field(default_factory=dict)   # name -> RGB frame
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
    def parse(cls, text: str, turn: AgentTurn, context: AgentContext) -> "Decision":
        """Validate a reply the way their ``parse_decision`` does: a whole
        JSON object (a complete ```json fence is fine, fragments are not), the
        output schema, then the chosen tool's own parameter schema. Numbers
        out of range are clamped with a note rather than rejected."""
        try:
            wire = parse_selection(text)
            name = wire.get("name")
            arguments = wire.get("arguments")
            if isinstance(arguments, str):
                arguments = parse_selection(arguments)
            if not isinstance(name, str) or not isinstance(arguments, dict):
                raise ValueError("a selection has a name string and an arguments object")
            allowed = {t.name: t for t in context.menu}
            if name not in allowed:
                raise ValueError(f"unknown tool {name!r}; choose one of {sorted(allowed)}")
            tool = allowed[name]
            arguments, notes = validate_args(tool, arguments)
            schema = parameters(tool, context.menu)
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


def instructions(context_menu: Sequence[type[Tool]], *, can_walk: bool, max_decisions: int,
                 safety_notes: Sequence[str] = ()) -> str:
    """The system prompt, once per run: the robot's conventions, the task's rules, the
    reply rules, the tool catalog as bullets and, appended, as JSON. Nothing about the world:
    the model learns that only from the camera."""
    notes = ""
    if safety_notes:
        lines = "\n".join(f"- {n.strip()}" for n in safety_notes)
        notes = f"\nRules for this task:\n{lines}\n"
    walk = ("Walking tools drive the base; they are offered because the base may be driven now."
            if can_walk else
            "The base cannot be driven in this run, so no walking tools are offered: search by looking.")
    fov_h, pitch = HEAD_CAMERA_FOV_H, math.degrees(HEAD_CAMERA_PITCH)
    joints = ""
    if any(t.name == "arm_path" for t in context_menu):
        rows = "\n".join(f"- {r['name']}: [{r['min']}, {r['max']}] rad, stand {r['stand']}" for r in joint_table())
        joints = f"""
Joints arm_path may command (radians; the stand pose is where every run starts and ends):
{rows}
- Conventions verified on the model: elbow 0 is a 90-degree bend with the forearm forward, about 1.57 is a straight arm, more negative bends the forearm up; shoulder_roll +1.57 (left) / -1.57 (right) with shoulder_pitch 0 and elbow 1.47 is a T-pose; wrist_roll -1.57 (left) / +1.57 (right) turns the palms up; shoulder_pitch -1.57 with elbow 1.57 holds a straight arm level and forward; waist_yaw positive looks LEFT.
"""
    text = f"""You control a Unitree G1 humanoid robot through a small set of tools, one decision per turn.

Robot and camera conventions:
- {NUM_JOINTS} joints in DDS order (legs 0-11, waist 12-14, left arm 15-21, right arm 22-28), radians. state.joint_pos, joint_vel and joint_torque are measured at the observation time. The host commands only the waist and arms (arm_sdk) and, for walking tools, the base velocity (LocoClient); the legs balance on their own.
- One forward head camera (RGB, no depth) on the torso, pointing {pitch:.0f} degrees down: the floor 1-3 m ahead fills the lower half of the image, and the image spans about {fov_h:.0f} degrees horizontally. Pixel positions are not metric distances; judge distance from apparent size and from where an object meets the floor.
- +yaw is LEFT (counter-clockwise). state.base_pose_cmd is the dead-reckoned [x forward, y left, yaw degrees] since the start, integrated from the commands sent (where the base actually is can differ; judge from the image). state.waist_yaw_deg is where the camera looks relative to the feet.
- {walk}
{joints}{notes}
Return exactly one tool selection per turn: {{"name": "...", "arguments": {{...}}}}. No Markdown or text outside this object. The host runs the whole tool, waits for the joints to settle, then supplies a fresh observation whose previous_result reports what happened.

Tools:
{prompt_catalog(context_menu)}

Every movement tool requires a note: 1-2 short sentences stating the current visual evidence and the purpose of this action. Keep failure causes when relevant; do not repeat numbers, history, tool mechanics or these rules. Use only the selected tool's arguments, within their ranges.

Observe before committing: a target that is off-centre needs a turn before a walk; a walk needs a clear floor for its whole distance; never walk toward something you cannot see the floor in front of. previous_result.execution_feedback reports target-versus-measured errors and settle after each tool; settled means the joints are still, not that the goal is reached. A rejected selection (previous_result.error) ran nothing: fix the selection, do not repeat it.

Persistence: one failed tool, blocked path or unseen goal does not establish impossibility. Diagnose from the fresh image, the state and previous_result, then try a different viewpoint, direction or step size and verify the result. Call done only when the current image establishes the goal. Do not call give_up while reasonable safe strategies remain; when you do, list what was tried and the evidence preventing progress.

Budget: {max_decisions} decisions for this run; extra.decisions_left counts down and a rejected selection costs one.

Robot tool catalog:
{json.dumps(function_schemas(context_menu), ensure_ascii=False)}"""
    return text


def build_context(context_menu: Sequence[type[Tool]], *, can_walk: bool, max_decisions: int,
                  safety_notes: Sequence[str] = ()) -> AgentContext:
    m = tuple(context_menu)
    return AgentContext(instructions(m, can_walk=can_walk, max_decisions=max_decisions, safety_notes=safety_notes),
                        function_schemas(m), output_schema(m), m)


# --------------------------------------------------------------------------
# Deciders
# --------------------------------------------------------------------------

class Decider(Worker[AgentTurn, Decision]):
    """``start(context)`` once per run, then ``request(turn)`` per decision; the
    result lands in ``latest()`` a few ticks later. ``last_call`` describes the
    most recent model call (status, elapsed, usage, what the server sent back);
    ``queries[request_id]`` is the query as sent, without the system prompt
    (config.json has it) and with each image as the step file that holds it."""

    model: str = "decider"
    provider: str = "test"

    def __init__(self, *, threaded: bool = True, min_interval: float = 0.0) -> None:
        super().__init__(threaded=threaded, min_interval=min_interval, name="decider")
        self.context: Optional[AgentContext] = None
        self.last_call: Optional[dict] = None
        self.queries: dict[int, list[dict]] = {}
        self.n_calls = 0

    def start(self, context: AgentContext) -> None:
        """Take the run's context and start the worker thread (when threaded)."""
        self.context = context
        Worker.start(self)

    def decide(self, turn: AgentTurn) -> Decision:
        raise NotImplementedError

    def process(self, turn: AgentTurn) -> Decision:
        if self.context is None:
            raise RuntimeError("decider.start(context) was not called")
        self.n_calls += 1
        t0 = time.monotonic()
        self.queries[turn.request_id] = [{"role": "user", "content": turn.observation}]
        call = {"call": self.n_calls, "at_s": time.time(), "request_id": turn.request_id,
                "status": "completed", "usage": None}
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

    def settings(self) -> dict:
        """For config.json: how this run asks the model."""
        return {"model": self.model}


class VLMDecider(Decider):
    """Asks the vision-language model (Hugging Face or OpenRouter, see ``g1/vlm.py``), keeping
    the conversation."""

    def __init__(self, client: VLMClient, *, max_width: int = IMAGE_WIDTH,
                 live_image_window: Optional[int] = AGENT_SETTINGS["live_image_window"],
                 fresh_turns: bool = False, min_interval: float = 0.0,
                 history_full_turns: int = AGENT_SETTINGS["history_full_turns"]) -> None:
        super().__init__(min_interval=min_interval)
        if live_image_window is not None and live_image_window < 1:
            raise ValueError("live_image_window must be >= 1 or None")
        self.client = client
        self.max_width = max_width
        self.live_image_window = live_image_window
        self.history_full_turns = max(1, int(history_full_turns))
        self.fresh_turns = fresh_turns
        self._turns: list[dict] = []          # {"user": msg, "assistant": msg | None, "image_at": int | None}
        self._image_step: dict[int, int] = {}   # id(image block) -> the request (= step file) it came from

    @classmethod
    def from_env(cls, model: Optional[str] = None, *,
                 live_image_window: Optional[int] = AGENT_SETTINGS["live_image_window"],
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
        self._image_step = {}

    # -- the conversation --------------------------------------------------
    def _user_message(self, turn: AgentTurn) -> tuple[dict, Optional[int]]:
        blocks: list[dict] = []
        blocks.append({"type": "text", "text": turn.observation})
        image_at = None
        for name, image in turn.images.items():
            blocks.append({"type": "text", "text": f"Camera image: {name}"})
            image_at = len(blocks)
            part = image_part(image, self.max_width)
            self._image_step[id(part)] = turn.request_id
            blocks.append(part)
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

    def _summary(self, older: list[dict], following: dict) -> str:
        """One host-written line per older turn: the tool chosen, its note, and what came of it
        (the next observation's previous_result)."""
        def observed(t: dict) -> dict:
            try:
                return json.loads(t["user"]["content"][0]["text"])
            except (KeyError, IndexError, TypeError, ValueError):
                return {}
        lines = []
        for t, nxt in zip(older, older[1:] + [following]):
            step = observed(t).get("extra", {}).get("env_step", "?")
            try:
                reply = json.loads(t["assistant"]["content"]) if t["assistant"] else {}
            except (TypeError, ValueError):
                reply = {}
            args = {k: v for k, v in (reply.get("arguments") or {}).items() if k not in ("note", "summary", "hindsight")}
            note = (reply.get("arguments") or {}).get("note", "")
            prev = observed(nxt).get("previous_result") or {}
            result = prev.get("result", {}).get("status") if isinstance(prev.get("result"), dict) else None
            outcome = result or prev.get("error") or "unknown"
            lines.append(f"- turn {step}: {reply.get('name', '(no valid reply)')}({json.dumps(args, separators=(',', ':'))})"
                         f"{' — ' + note if note else ''}; result: {outcome}")
        return "Earlier in this run (summarised by the host; images and full results omitted):\n" + "\n".join(lines)

    def messages(self, turn: Optional[AgentTurn] = None) -> list[dict]:
        """The system prompt, then the newest ``history_full_turns`` turns in full; older turns
        are one summary line each, at the head of the first full turn."""
        assert self.context is not None
        out = [{"role": "system", "content": self.context.instructions}]
        if not self.fresh_turns:
            older, kept = self._turns[:-self.history_full_turns], self._turns[-self.history_full_turns:]
            for i, t in enumerate(kept):
                user = t["user"]
                if i == 0 and older:
                    user = {"role": "user", "content": [{"type": "text", "text": self._summary(older, t)},
                                                        *user["content"]]}
                out.append(user)
                if t["assistant"] is not None:
                    out.append(t["assistant"])
        if turn is not None:
            out.append(self._user_message(turn)[0])
        return out

    def as_recorded(self, msgs: list[dict]) -> list[dict]:
        """Messages as sent, each image replaced by the step file holding it (lossless there;
        the model got it resized to ``max_width`` as JPEG)."""
        def strip(msg):
            if isinstance(msg["content"], str):
                return msg
            blocks = [b if b["type"] == "text" else
                      {"type": "image", "image": f"step_{self._image_step.get(id(b), 0):04d}.png"}
                      for b in msg["content"]]
            return {"role": msg["role"], "content": blocks}
        return [strip(m) for m in msgs]

    def transcript(self) -> list[dict]:
        """The conversation so far, as recorded."""
        return self.as_recorded(self.messages())

    def decide(self, turn: AgentTurn) -> Decision:
        assert self.context is not None
        user, image_at = self._user_message(turn)
        if not self.fresh_turns:
            self._prune(incoming=1 if image_at is not None else 0)
        msgs = self.messages()
        msgs.append(user)
        self.queries[turn.request_id] = self.as_recorded(msgs[1:])     # the system prompt is in config.json
        text = self.client.complete(msgs, schema=self.context.output_schema)
        entry = {"user": user, "assistant": {"role": "assistant", "content": text}, "image_at": image_at}
        if not self.fresh_turns:
            self._turns.append(entry)
        return Decision.parse(text, turn, self.context)

    def _call_details(self) -> dict:
        return {"usage": self.client.last_usage, "response_mode": self.client.response_mode,
                "reasoning_effort": self.client.reasoning_effort,
                "request_elapsed_s": round(self.client.last_elapsed, 6), "chunks": list(self.client.last_chunks)}

    def settings(self) -> dict:
        """For config.json: how this run asks the model."""
        c = self.client
        return {"model": c.model, "provider": c.provider, "stream": c.stream,     # never .env values (base_url)
                "reasoning_effort": c.reasoning_effort, "max_tokens": c.max_tokens,
                "temperature": TEMPERATURE, "image_max_width_px": self.max_width,
                "jpeg_quality": AGENT_SETTINGS["jpeg_quality"], "history_full_turns": self.history_full_turns,
                "live_image_window": self.live_image_window}
