"""The microphone: what people say to the robot, as the robot's own speech recognition
hears it.

With voice interaction on in the Unitree app, the G1 transcribes what it hears and
publishes each utterance on DDS ``rt/audio_msg`` (``std_msgs String``, JSON;
measured 2026-10-30, docs/running-on-the-robot.md):

    {"text": "Hello, robot.", "language": "<|en|>", "index": 28, "timestamp": ..., ...}
    {"play_state": 1}        the robot's speaker started playing; {"play_state": 0} it stopped

``Microphone`` is the subscriber's callback and keeps an **utterance log**: unlike the
camera's latest-only slot, every utterance is kept (``since(seq)`` is what is new). It
drops what the model must not take as a person speaking:

  - a language not in ``heard_languages`` (short or noisy English comes back tagged zh/ja)
  - anything heard while the robot's speaker plays, or within ``mic_echo_guard_s`` after:
    the robot hearing itself (our ``say``, or its own assistant)

The robot and ``--sim`` share it: ``RobotEnv`` subscribes through its link, and the twin
publishes a task's scripted ``voices`` on the same topic.
"""
from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from typing import Callable, Optional

from g1.core import limits

TOPIC = "rt/audio_msg"


def languages(text: str = limits.get("heard_languages")) -> frozenset[str]:
    return frozenset(w.strip().lower() for w in text.split(",") if w.strip())


@dataclass(frozen=True)
class Heard:
    text: str
    language: str       # "en", or "" when the message had none
    stamp: float        # seconds on the env's clock when it arrived
    seq: int            # 1, 2, ...: since(seq) is everything after it


class Microphone:
    def __init__(self, clock: Callable[[], float], *, keep: Optional[frozenset[str]] = None,
                 echo_guard_s: float = limits.get("mic_echo_guard_s")) -> None:
        self.clock = clock
        self.keep = languages() if keep is None else keep
        self.echo_guard_s = echo_guard_s
        self._lock = threading.Lock()
        self._log: list[Heard] = []
        self.speaking = False                  # the robot's speaker is playing (play_state)
        self._quiet_at = -math.inf             # when it last stopped
        self.counts = {"messages": 0, "heard": 0, "other_language": 0, "while_speaking": 0, "unreadable": 0}

    def __call__(self, msg) -> None:
        """The subscriber callback: a ``std_msgs String`` (its ``data``) or the JSON text."""
        self.on_message(getattr(msg, "data", msg))

    def on_message(self, data: str) -> None:
        now = self.clock()
        with self._lock:
            self.counts["messages"] += 1
            try:
                m = json.loads(data)
                if not isinstance(m, dict):
                    raise ValueError
            except (TypeError, ValueError):
                self.counts["unreadable"] += 1
                return
            if "play_state" in m:
                playing = bool(m["play_state"])
                if self.speaking and not playing:
                    self._quiet_at = now
                self.speaking = playing
                return
            text = str(m.get("text") or "").strip()
            if not text:
                return
            language = str(m.get("language") or "").strip("<|>").lower()
            if language and language not in self.keep:
                self.counts["other_language"] += 1
                return
            if self.speaking or now - self._quiet_at < self.echo_guard_s:
                self.counts["while_speaking"] += 1
                return
            self.counts["heard"] += 1
            self._log.append(Heard(text, language, now, len(self._log) + 1))

    @property
    def seq(self) -> int:
        """The newest utterance's seq (0: nothing heard yet)."""
        with self._lock:
            return len(self._log)

    def since(self, seq: int) -> list[Heard]:
        with self._lock:
            return self._log[max(0, seq):]


def heard_view(heard: list[Heard], now: float, *, max_chars: int = limits.get("heard_max_chars"),
               max_age_s: float = limits.get("heard_max_age_s")) -> list[dict]:
    """What an observation carries: the newest utterances, oldest first, as text and how many
    seconds ago, at most ``max_chars`` of text and none older than ``max_age_s``."""
    out: list[dict] = []
    room = max_chars
    for h in reversed(heard):
        age = now - h.stamp
        if age > max_age_s or room <= 0:
            break
        text = h.text if len(h.text) <= room else h.text[-room:]
        room -= len(text)
        out.append({"text": text, "s_ago": round(age, 1)})
    return out[::-1]


def subscribe(link, clock: Callable[[], float]) -> Optional[Microphone]:
    """A Microphone on ``link``'s rt/audio_msg, or None (with a warning) if it cannot subscribe."""
    mic = Microphone(clock)
    try:
        mic.subscription = link.audio_msg(mic)
    except Exception as e:
        print(f"warning: microphone unavailable ({e}); the run goes on deaf")
        return None
    return mic
