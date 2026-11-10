"""g1 start's live voice: a voice model hears the robot's raw mic and talks back through its speaker.

``Voice`` is the session's handle on the voice process (``g1/voice/child.py``: the raw mic up, the model's
voice down through ``PlayStream``). Two engines (``configs/model.json`` ``voice.engine``):

* ``gpt-live`` (``openai_live.py``): GPT-Live, full duplex, talks; whatever needs doing or knowing it
  delegates to our backend: the session's chat engine on the backend model (Sol), which chooses from the
  menu and runs it the one way (``Session._delegated``). Events: ``("user_delta", text)``,
  ``("robot_delta", text)``, ``("delegation", id)``.
* ``realtime`` (``openai_realtime.py``): one realtime model hears, decides and calls the session's tools
  itself (``Session._live_call``). Events: ``("user_said", text)``, ``("robot_said", text)``,
  ``("tool_call", call_id, name, arguments)``, ``("speech_started",)``, ``("response_done", status, usage)``.

Both: ``("ready",)``, ``("speaking", on)``, ``("sdk_call", entry)``, ``("error", text)``, ``("closed", why)``,
handed to ``on_event`` on a reader thread. Either way the voice decides fast and short and anything that
takes planning goes to a task.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import threading
import time
from typing import Callable, Optional

from g1.core import limits
from g1.vlm import VOICE_SETTINGS


def available(settings: dict = VOICE_SETTINGS) -> Optional[str]:
    """Why the live voice cannot be used here (None: it can)."""
    from g1.voice.openai_realtime import KEY_VAR
    if settings["engine"] == "unitree":
        return "voice.engine is unitree in configs/model.json"
    if not os.environ.get(KEY_VAR):
        return f"no {KEY_VAR} in .env"
    if settings["engine"] == "gpt-live" and not os.environ.get("OPENROUTER_API_KEY"):
        return "no OPENROUTER_API_KEY in .env (gpt-live's backend, voice.backend_model, is on OpenRouter)"
    try:
        import websockets  # noqa: F401
    except ImportError:
        return "the websockets package is missing: pip install -e \".[voice]\""
    return None


class Voice:
    """The voice process for one connection. ``start`` blocks until the model has the session (or raises);
    the session then talks to it with the methods below, from any thread."""

    def __init__(self, iface: str, domain: int, instructions: str, tools: list[dict],
                 on_event: Callable[[tuple], None], *, settings: dict = VOICE_SETTINGS, muted: bool = True,
                 speaker_on: bool = True, target: Optional[Callable] = None,
                 timeout: float = limits.get("voice_connect_timeout_s")) -> None:
        from g1.voice import openai_live as live
        from g1.voice import openai_realtime as rt
        self.settings = settings
        self.engine = settings["engine"]
        if self.engine == "gpt-live":
            url, start = live.URL, live.session_start(settings, instructions)
        else:
            url, start = rt.url(settings), rt.session_update(settings, instructions, tools)
        self.cfg = {"engine": self.engine, "iface": iface, "domain": domain, "url": url, "start": start,
                    "muted": muted, "speaker_on": speaker_on}
        self.on_event, self.timeout = on_event, timeout
        self.target = target
        self.proc = None
        self.conn = None
        self.thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._closed: Optional[str] = None
        self.speaking = False

    @property
    def alive(self) -> bool:
        return self.proc is not None and self._closed is None

    def start(self) -> None:
        from g1.voice.child import serve
        ctx = mp.get_context("spawn")                # never fork a process that holds DDS
        self.conn, child = ctx.Pipe(duplex=True)
        self.proc = ctx.Process(target=self.target or serve, args=(child, self.cfg), daemon=True, name="g1-voice")
        self.proc.start()
        child.close()
        self.thread = threading.Thread(target=self._run, name="g1-voice", daemon=True)
        self.thread.start()
        t0 = time.monotonic()
        while not self._ready.wait(0.05):
            if self._closed is not None:
                why = self._closed
                self.close()
                raise ConnectionError(f"voice: {why}")
            if time.monotonic() - t0 > self.timeout:
                self.close()
                raise TimeoutError(f"voice: the {self.engine} model was not ready within {self.timeout:g} s")

    def _run(self) -> None:
        while True:
            try:
                msg = self.conn.recv()
            except (EOFError, OSError):
                msg = ("closed", "the voice process ended")
            if msg[0] == "ready":
                if self._ready.is_set():
                    continue                         # a later session.update (a refresh): nothing to tell
                self._ready.set()
            elif msg[0] == "speaking":
                self.speaking = bool(msg[1])
            elif msg[0] == "closed":
                if self._closed is not None:
                    return
                self._closed = msg[1]
            if self._ready.is_set() or msg[0] != "closed":
                self.on_event(msg)
            if msg[0] == "closed":
                return

    def send(self, msg: tuple) -> None:
        with self._lock:
            if self.conn is None or self._closed is not None:
                return
            try:
                self.conn.send(msg)
            except (BrokenPipeError, OSError):
                pass

    # what the session tells it
    def mute(self, on: bool) -> None:
        self.send(("mute", bool(on)))

    def speaker(self, on: bool) -> None:
        self.send(("speaker", bool(on)))

    def text(self, line: str) -> None:
        """A typed line, answered aloud like speech."""
        self.send(("text", line))

    def say(self, text: str) -> None:
        """The host's own line, said as written."""
        self.send(("say", text))

    def tool_result(self, call_id: str, output: str, *, image: Optional[tuple[str, str]] = None,
                    respond: bool = True) -> None:
        """A tool call's result; ``image``: (a line, a JPEG data URL) shown with it; ``respond``: the model
        answers it (False for ignore: it stays quiet)."""
        self.send(("tool_result", call_id, output, image, respond))

    def commentary(self, delegation_id: str, content: str) -> None:
        """gpt-live: the backend's line for the voice to say (in its own words)."""
        self.send(("commentary", delegation_id, content))

    def thinking(self, delegation_id: str, content: str) -> None:
        """gpt-live: what the backend did, for the voice to know (not said when it arrives)."""
        self.send(("thinking", delegation_id, content))

    def update(self, instructions: str, tools: list[dict]) -> None:
        self.send(("update", instructions, tools))

    def stop(self) -> None:
        """Stop talking now (Esc)."""
        self.send(("stop",))

    def close(self) -> None:
        self.send(("close",))
        if self._closed is None:
            self._closed = "closed by the session"
        if self.proc is not None:
            self.proc.join(timeout=2.0)
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(timeout=2.0)
                if self.proc.is_alive():
                    self.proc.kill()
                    self.proc.join()
        if self.conn is not None:
            self.conn.close()
            self.conn = None
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=2.0)
