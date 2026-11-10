"""The live voice's one model: OpenAI's realtime API, speech in and speech out over a websocket.

What goes up: the session (``session_update``: the instructions, the tools, how it hears and speaks), the
mic as 24 kHz PCM, typed lines, tool results (a ``see`` adds the camera's image), and a stop. What comes
down: the model's voice as 24 kHz PCM, what it heard and said as text, and its tool calls. ``LiveCore``
is that exchange without any socket: server events and the parent's messages in, websocket events, the
player and messages to the parent out (``g1/voice/child.py`` wires it to the real ones; the tests to
fakes). Event names and fields: developers.openai.com/api/reference/resources/realtime (2026-11-10).
"""
from __future__ import annotations

import base64
from typing import Callable, Optional

from g1.core import limits
from g1.core.config import SPEAKER_RATE_HZ
from g1.core.sounds import Resampler, dbfs

URL = "wss://api.openai.com/v1/realtime?model={model}"
KEY_VAR = "OPENAI_API_KEY"
MODEL_RATE_HZ = 24000                  # the only PCM rate the realtime API takes and gives
NOT_ERRORS = {"response_cancel_not_active"}   # a stop with nothing to cancel: not worth a line


def url(settings: dict) -> str:
    return URL.format(model=settings["realtime_model"])


def realtime_tools(chat_tools: list[dict]) -> list[dict]:
    """The session's tools (OpenAI chat format: ``{"type": "function", "function": {...}}``) in the
    realtime API's flat format."""
    out = []
    for t in chat_tools:
        f = t["function"]
        out.append({"type": "function", "name": f["name"], "description": f.get("description", ""),
                    "parameters": f.get("parameters", {"type": "object", "properties": {}})})
    return out


def session_update(settings: dict, instructions: str, tools: list[dict]) -> dict:
    """The ``session.update`` event: the model's instructions and tools, and how it hears (far-field noise
    reduction, turn detection that answers and lets itself be interrupted) and speaks."""
    s = settings
    if s["turn_detection"] == "semantic_vad":
        turns = {"type": "semantic_vad", "eagerness": s["eagerness"]}
    else:
        turns = {"type": "server_vad"}
    turns.update(create_response=True, interrupt_response=True)
    session = {
        "type": "realtime",
        "instructions": instructions,
        "tools": realtime_tools(tools),
        "tool_choice": "auto",
        "output_modalities": ["audio"],
        "max_output_tokens": s["max_output_tokens"],
        "audio": {
            "input": {"format": {"type": "audio/pcm", "rate": MODEL_RATE_HZ},
                      "noise_reduction": {"type": s["noise_reduction"]} if s["noise_reduction"] else None,
                      "transcription": {"model": s["transcription_model"], "language": s["language"]},
                      "turn_detection": turns},
            "output": {"format": {"type": "audio/pcm", "rate": MODEL_RATE_HZ}, "voice": s["voice"]}},
    }
    if s["reasoning_effort"] != "default":
        session["reasoning"] = {"effort": s["reasoning_effort"]}
    return {"type": "session.update", "session": session}


def session_refresh(instructions: str, tools: list[dict]) -> dict:
    """A later ``session.update``: new instructions and tools only (the mode changed, a memory was kept).
    The model and voice stay as they were (the voice cannot change once it has spoken)."""
    return {"type": "session.update", "session": {"type": "realtime", "instructions": instructions,
                                                  "tools": realtime_tools(tools)}}


def say_exactly(text: str) -> dict:
    """A response that says ``text`` as written (the host's own lines: ``/say``, a fallback apology)."""
    return {"type": "response.create", "response": {
        "instructions": f"Say exactly this, word for word, and nothing else: {text}", "tool_choice": "none"}}


def user_text(text: str) -> dict:
    return {"type": "conversation.item.create", "item": {
        "type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}}


def tool_output(call_id: str, output: str) -> dict:
    return {"type": "conversation.item.create", "item": {
        "type": "function_call_output", "call_id": call_id, "output": output}}


def user_image(text: str, data_url: str) -> dict:
    return {"type": "conversation.item.create", "item": {
        "type": "message", "role": "user", "content": [{"type": "input_text", "text": text},
                                                       {"type": "input_image", "image_url": data_url}]}}


class LiveCore:
    """The voice process's logic: the model's events and the parent's messages in; websocket events
    (``ws_send``), our voice (``player``) and messages to the parent (``emit``) out.

    ``player`` has ``begin()``, ``feed(pcm16k)``, ``end()``, ``stop() -> played_ms``, ``playing`` (our voice is
    still coming out) and ``speaking`` (playing, or just stopped: its echo may still be in the mic). The
    mic starts muted (g1 start's ``/mic`` starts off). While our voice plays, ``barge_in`` decides what the
    model hears: ``open`` everything, ``gate`` only pieces louder than ``-gate_db`` dBFS (the rest as silence,
    so its turn detection keeps time)."""

    def __init__(self, ws_send: Callable[[dict], None], player, emit: Callable[[tuple], None], *,
                 barge_in: str = limits.get("voice_barge_in"), gate_db: float = limits.get("voice_gate_db"),
                 muted: bool = True, speaker_on: bool = True) -> None:
        self.ws_send, self.player, self.emit = ws_send, player, emit
        self.barge_in, self.gate_dbfs = barge_in, -abs(gate_db)
        self.muted, self.speaker_on = muted, speaker_on
        self._up = Resampler(SPEAKER_RATE_HZ, MODEL_RATE_HZ)
        self._down: Optional[Resampler] = None
        self._item: Optional[str] = None             # the assistant item whose audio plays now
        self.counts = {"mic_pieces": 0, "gated": 0, "interrupted": 0, "replies": 0}

    # -- the mic --------------------------------------------------------------------------------------
    def on_mic(self, pcm: bytes) -> None:
        """One piece of the robot's mic (16 kHz s16le)."""
        if self.muted:
            return
        if self.player.speaking and self.barge_in == "gate" and dbfs(pcm) < self.gate_dbfs:
            pcm = bytes(len(pcm))                    # the robot's own voice, most likely: silence instead
            self.counts["gated"] += 1
        up = self._up.push(pcm)
        if up:
            self.counts["mic_pieces"] += 1
            self.ws_send({"type": "input_audio_buffer.append", "audio": base64.b64encode(up).decode("ascii")})

    # -- the model ------------------------------------------------------------------------------------
    def on_server(self, ev: dict) -> None:
        t = ev.get("type", "")
        if t == "response.output_audio.delta":
            if ev.get("item_id") != self._item:
                self._start_reply(ev.get("item_id"))
            pcm = self._down.push(base64.b64decode(ev.get("delta", "")))
            if pcm and self.speaker_on:
                self.player.feed(pcm)
        elif t == "response.output_audio.done":
            if self._down is not None and ev.get("item_id") == self._item:
                tail = self._down.flush()
                if tail and self.speaker_on:
                    self.player.feed(tail)
                self._down = None
                self.player.end()
        elif t == "input_audio_buffer.speech_started":
            if self.player.playing:
                self.interrupt()                     # someone talks over the robot: it stops
            self.emit(("speech_started",))
        elif t == "conversation.item.input_audio_transcription.completed":
            text = " ".join(str(ev.get("transcript", "")).split())
            if text:
                self.emit(("user_said", text))
        elif t == "response.output_audio_transcript.done":
            text = " ".join(str(ev.get("transcript", "")).split())
            if text:
                self.emit(("robot_said", text))
        elif t == "response.function_call_arguments.done":
            self.emit(("tool_call", ev.get("call_id", ""), ev.get("name", ""), ev.get("arguments", "") or "{}"))
        elif t == "response.done":
            r = ev.get("response") or {}
            self.counts["replies"] += 1
            self.emit(("response_done", r.get("status", ""), r.get("usage") or {}))
        elif t == "session.updated":
            self.emit(("ready",))
        elif t == "error":
            err = ev.get("error") or {}
            if err.get("code") not in NOT_ERRORS:
                self.emit(("error", f"{err.get('type', 'error')}: {err.get('message', '')}".strip()))

    def _start_reply(self, item_id) -> None:
        self._item = item_id
        self._down = Resampler(MODEL_RATE_HZ, SPEAKER_RATE_HZ)
        if self.speaker_on:
            self.player.begin()

    def interrupt(self) -> None:
        """Stop our voice now: the speaker stops, the model is told how much was heard (so its memory of
        the reply ends where the person heard it end), and any reply in progress is cancelled."""
        played_ms = self.player.stop()
        self.counts["interrupted"] += 1
        if self._item:
            self.ws_send({"type": "conversation.item.truncate", "item_id": self._item, "content_index": 0,
                          "audio_end_ms": int(played_ms)})
        self.ws_send({"type": "response.cancel"})
        self._item, self._down = None, None

    # -- the parent -----------------------------------------------------------------------------------
    def on_parent(self, msg: tuple) -> None:
        kind = msg[0]
        if kind == "mute":
            self.muted = bool(msg[1])
            if self.muted:
                self.ws_send({"type": "input_audio_buffer.clear"})
        elif kind == "speaker":
            self.speaker_on = bool(msg[1])
            if not self.speaker_on and self.player.playing:
                self.player.stop()
        elif kind == "text":
            self.ws_send(user_text(msg[1]))
            self.ws_send({"type": "response.create"})
        elif kind == "say":
            self.ws_send(say_exactly(msg[1]))
        elif kind == "tool_result":
            _, call_id, output, image, respond = msg
            self.ws_send(tool_output(call_id, output))
            if image is not None:
                self.ws_send(user_image(image[0], image[1]))
            if respond:
                self.ws_send({"type": "response.create"})
        elif kind == "update":
            self.ws_send(session_refresh(msg[1], msg[2]))
        elif kind == "stop":
            if self.player.playing:
                self.interrupt()
