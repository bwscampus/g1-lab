"""The live voice as two models: OpenAI's GPT-Live talks, our backend (Sol, through the session) decides.

GPT-Live is full duplex: one voice model hears and speaks at once and decides itself when to talk, stop,
or let someone finish. It runs no tools. When something needs doing or knowing it **delegates**: with
``client`` delegation it sends ``session.delegation.created`` (an id only: the request is rebuilt from the
transcript) and our code answers. g1 start answers with its chat engine on the backend model (the menu,
the y/N confirmations, the run, ``Session._delegated``); what it would have said goes back as
``session.commentary.append`` (the voice says it in its own words) and what happened as
``session.thinking.append`` (it knows, but does not say it on arrival), each at most 500 tokens.

``DuplexCore`` is that exchange without the socket, like ``openai_realtime.LiveCore``. The mic goes up as
it comes (16 kHz PCM, which GPT-Live takes and gives: no resampling); output audio has no reply
boundaries or done event, so the player starts on the first piece after a quiet and ends when it has
played out. Event names: developers.openai.com/api/docs/guides/voice-websockets?api=live,
live-conversations and live-delegation (2026-11-10).
"""
from __future__ import annotations

import base64
import itertools
import time
from typing import Callable

from g1.core import limits
from g1.core.config import SPEAKER_RATE_HZ
from g1.core.sounds import dbfs

URL = "wss://api.openai.com/v1/live/sessions"
KEY_VAR = "OPENAI_API_KEY"
CONTENT_MAX_CHARS = 1500           # an append's content: 500 tokens at most, so about this many characters
_ids = itertools.count(1)


def event_id() -> str:
    return f"g1_{next(_ids)}"


def session_start(settings: dict, instructions: str) -> dict:
    """The first message: the voice model, its instructions and voice, 16 kHz PCM both ways (the robot's
    mic and speaker rate), and client delegation (our code is the backend)."""
    return {"type": "session.start", "event_id": event_id(), "session": {
        "model": settings["live_model"],
        "instructions": instructions,
        "audio": {"format": {"type": "audio/pcm", "rate": SPEAKER_RATE_HZ}, "output": {"voice": settings["voice"]}},
        "delegation": {"type": "client"},
    }}


def append(kind: str, delegation_id, content: str) -> dict:
    """``session.{kind}.append`` (instructions, thinking, commentary); ``delegation_id`` None: the whole session
    (instructions only)."""
    content = " ".join(str(content).split())
    if len(content) > CONTENT_MAX_CHARS:
        content = content[: CONTENT_MAX_CHARS - 3] + "..."
    return {"type": f"session.{kind}.append", "event_id": event_id(), "delegation_id": delegation_id,
            "content": content}


class DuplexCore:
    """The voice process's logic for GPT-Live: its events and the parent's messages in; websocket events,
    our voice (``player``) and messages to the parent out. Same player and gate as ``LiveCore``."""

    def __init__(self, ws_send: Callable[[dict], None], player, emit: Callable[[tuple], None], *,
                 barge_in: str = limits.get("voice_barge_in"), gate_db: float = limits.get("voice_gate_db"),
                 hold_s: float = limits.get("voice_stop_hold_s"), clock: Callable[[], float] = time.monotonic,
                 muted: bool = True, speaker_on: bool = True) -> None:
        self.ws_send, self.player, self.emit, self.clock = ws_send, player, emit, clock
        self.barge_in, self.gate_dbfs, self.hold_s = barge_in, -abs(gate_db), hold_s
        self.muted, self.speaker_on = muted, speaker_on
        self._hold_until = -1e9                      # after Esc: what it still says is dropped until then
        self.counts = {"mic_pieces": 0, "gated": 0, "delegations": 0, "dropped": 0}

    # -- the mic --------------------------------------------------------------------------------------
    def on_mic(self, pcm: bytes) -> None:
        if self.muted or not pcm:
            return
        if self.player.speaking and self.barge_in == "gate" and dbfs(pcm) < self.gate_dbfs:
            pcm = bytes(len(pcm))                    # the robot's own voice, most likely: silence instead
            self.counts["gated"] += 1
        self.counts["mic_pieces"] += 1
        self.ws_send({"type": "session.input_audio.append", "audio": base64.b64encode(pcm).decode("ascii")})

    # -- the model ------------------------------------------------------------------------------------
    def on_server(self, ev: dict) -> None:
        t = ev.get("type", "")
        if t == "session.output_audio.delta":
            if not self.speaker_on or self.clock() < self._hold_until:
                self.counts["dropped"] += 1
                return
            if not self.player.playing:
                self.player.begin()
            self.player.feed(base64.b64decode(ev.get("delta", "")))
            self.player.end()                        # no done event: it ends once what came has played
        elif t == "session.input_transcript.delta":
            if ev.get("delta"):
                self.emit(("user_delta", ev["delta"]))
        elif t == "session.output_transcript.delta":
            if ev.get("delta"):
                self.emit(("robot_delta", ev["delta"]))
        elif t == "session.delegation.created":
            self.counts["delegations"] += 1
            self.emit(("delegation", (ev.get("delegation") or {}).get("id", "")))
        elif t == "session.started":
            self.emit(("ready",))
        elif t == "session.closed":
            self.emit(("closed", str(ev.get("reason") or "the session ended")))
        elif t == "error":
            err = ev.get("error") if isinstance(ev.get("error"), dict) else ev
            self.emit(("error", f"{err.get('code', 'error')}: {err.get('message', '')}".strip()))

    # -- the parent -----------------------------------------------------------------------------------
    def on_parent(self, msg: tuple) -> None:
        kind = msg[0]
        if kind == "mute":
            self.muted = bool(msg[1])
            self.ws_send({"type": "session.input_audio.mute" if self.muted else "session.input_audio.unmute",
                          "event_id": event_id()})
        elif kind == "speaker":
            self.speaker_on = bool(msg[1])
            if not self.speaker_on and self.player.playing:
                self.player.stop()
        elif kind == "text":
            self.ws_send(append("instructions", None, f"Typed at your keyboard, answer it aloud as if it was said to "
                                                      f"you: {msg[1]}"))
        elif kind == "say":
            self.ws_send(append("instructions", None, f"Say this now, in your own words: {msg[1]}"))
        elif kind in ("commentary", "thinking"):
            self.ws_send(append(kind, msg[1], msg[2]))
        elif kind == "stop":
            self.player.stop()
            self._hold_until = self.clock() + self.hold_s
            self.ws_send(append("instructions", None, "The person at your keyboard asked you to stop talking: stop "
                                                      "now and wait until someone speaks to you."))
        elif kind in ("tool_result", "update"):
            pass                                     # the realtime engine's: GPT-Live has no tools of its own
