"""The voice process: the robot's raw mic up to the voice model, its voice down to the robot's speaker.

One spawned child per g1 start session (``Voice`` starts it), so the audio, the websocket and the
speaker's RPCs never share a process with the session (on the Jetson, Python next to rt/lowstate starves
for the GIL; docs/running-on-the-robot.md). It holds its own DDS participant and AudioClient, like the
lidar's subscriber. Threads: the mic (``MicSocket`` -> ``LiveCore.on_mic``), the parent's messages, the
player (``Player.pump``) and the websocket (``LiveCore.on_server``, on this process's main thread).

The mic is the head unit's 4-mic array after its own echo cancelling: UDP multicast
239.168.123.161:5555, 16 kHz mono s16le, 5120-byte packets, flowing only with voice interaction on in the
Unitree app (measured 2026-10-30).
"""
from __future__ import annotations

import ipaddress
import json
import os
import socket
import struct
import sys
import threading
import time
from typing import Callable, Optional

from g1.core import limits
from g1.core.config import SPEAKER_RATE_HZ, SPEAKER_SAMPLE_BYTES
from g1.envs import sdk

MIC_GROUP, MIC_PORT = "239.168.123.161", 5555      # the head unit's mic stream (robot facts)
MIC_PACKET = 5120
APP_NAME = "g1_voice"                               # PlayStream's app_name: what PlayStop stops
BYTES_PER_S = SPEAKER_RATE_HZ * SPEAKER_SAMPLE_BYTES


class MicSocket:
    """The mic multicast, joined on ``iface`` (a name or the interface's IPv4 address). ``group=None``
    binds a plain UDP port on 127.0.0.1 instead (the tests' sender)."""

    def __init__(self, iface: str = "", group: Optional[str] = MIC_GROUP, port: int = MIC_PORT,
                 timeout: float = 1.0) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        if group is None:
            self.sock.bind(("127.0.0.1", port))
        else:
            self.sock.bind(("", port))
            self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, membership(group, iface))
        self.sock.settimeout(timeout)
        self.port = self.sock.getsockname()[1]
        self.packets = 0

    def read(self) -> Optional[bytes]:
        """The next packet, or None after the timeout."""
        try:
            data, _ = self.sock.recvfrom(65536)
        except socket.timeout:
            return None
        self.packets += 1
        return data

    def close(self) -> None:
        self.sock.close()


def membership(group: str, iface: str) -> bytes:
    """IP_ADD_MEMBERSHIP for ``group`` on ``iface``: by its address when ``iface`` is one, by its index on
    Linux (ip_mreqn), else on the default interface."""
    g = socket.inet_aton(group)
    try:
        ipaddress.IPv4Address(iface)
        return g + socket.inet_aton(iface)
    except ValueError:
        pass
    if iface and sys.platform.startswith("linux"):
        return struct.pack("4s4si", g, socket.inet_aton("0.0.0.0"), socket.if_nametoindex(iface))
    return g + socket.inet_aton("0.0.0.0")


class Player:
    """Our voice on the robot's speaker. PCM pieces in (``feed``, 16 kHz s16le), ``PlayStream`` calls out in
    pieces of ``chunk_s``, paced so at most ``ahead_s`` is queued on the robot beyond what has played (a stop
    is quick). One ``stream_id`` per reply (``begin``). ``pump`` sends what is due; the player's thread
    calls it every ``voice_poll_s``. ``play(stream_id, pcm) -> code`` and ``halt() -> code`` are the speaker."""

    def __init__(self, play: Callable[[str, bytes], int], halt: Callable[[], int], *,
                 clock: Callable[[], float] = time.monotonic, emit: Callable[[tuple], None] = lambda m: None,
                 chunk_s: float = limits.get("voice_play_chunk_s"), ahead_s: float = limits.get("voice_play_ahead_s"),
                 echo_s: float = limits.get("mic_echo_guard_s")) -> None:
        self.play, self.halt, self.clock, self.emit = play, halt, clock, emit
        self.chunk = int(chunk_s * SPEAKER_RATE_HZ) * SPEAKER_SAMPLE_BYTES
        self.ahead_s, self.echo_s = ahead_s, echo_s
        self._lock = threading.Lock()
        self._buf = bytearray()
        self._stream = ""
        self._ended = True
        self._t0: Optional[float] = None             # when the reply's first piece was sent
        self._sent_s = 0.0                           # how much of the reply has been sent
        self._quiet_at = -1e9                        # when our voice last stopped coming out
        self._playing = False
        self._n = 0
        self.counts = {"replies": 0, "pieces": 0, "failed": 0, "stopped": 0}

    @property
    def playing(self) -> bool:
        with self._lock:
            return self._playing

    @property
    def speaking(self) -> bool:
        """Playing, or stopped less than ``echo_s`` ago (its echo may still be in the mic)."""
        with self._lock:
            return self._playing or self.clock() - self._quiet_at < self.echo_s

    def begin(self) -> None:
        with self._lock:
            self._n += 1
            self._stream = f"{int(time.time() * 1000)}-{self._n}"
            self._buf.clear()
            self._ended, self._t0, self._sent_s = False, None, 0.0
            started = not self._playing
            self._playing = True
            self.counts["replies"] += 1
        if started:
            self.emit(("speaking", True))

    def feed(self, pcm: bytes) -> None:
        with self._lock:
            if self._playing:
                self._buf += pcm

    def end(self) -> None:
        """The reply's last piece has been fed: it ends once everything has played."""
        with self._lock:
            self._ended = True

    def played_ms(self) -> int:
        with self._lock:
            return self._played_ms()

    def _played_ms(self) -> int:
        if self._t0 is None:
            return 0
        return int(1000 * min(self._sent_s, self.clock() - self._t0))

    def stop(self) -> int:
        """Stop now (``PlayStop``); how much of the reply had played, in ms."""
        with self._lock:
            was = self._playing
            played = self._played_ms()
            self._buf.clear()
            self._playing, self._ended = False, True
            if was:
                self._quiet_at = self.clock()
                self.counts["stopped"] += 1
        if was:
            self.halt()
            self.emit(("speaking", False))
        return played

    def pump(self) -> None:
        """Send the next piece when it is due; end the reply when all of it has played."""
        piece, stream, finished = None, "", False
        with self._lock:
            if not self._playing:
                return
            now = self.clock()
            queued = self._sent_s - (now - self._t0 if self._t0 is not None else 0.0)
            full = len(self._buf) >= self.chunk
            if self._buf and queued < self.ahead_s and (full or self._ended):
                piece = bytes(self._buf[: self.chunk])
                del self._buf[: self.chunk]
                if self._t0 is None:
                    self._t0 = now
                self._sent_s += len(piece) / BYTES_PER_S
                stream = self._stream
            elif self._ended and not self._buf and (self._t0 is None or now - self._t0 >= self._sent_s):
                self._playing, self._quiet_at, finished = False, now, True
        if piece is not None:
            self.counts["pieces"] += 1
            if self.play(stream, piece) != 0:
                self.counts["failed"] += 1
        if finished:
            self.emit(("speaking", False))


# -- the process ----------------------------------------------------------------------------------

def serve(conn, cfg: dict) -> None:
    """The child: connect DDS (its own AudioClient) and the model's websocket, send the first message, then
    run until the parent says close or the connection drops. ``cfg``: engine (gpt-live | realtime), iface,
    domain, url, start (the first message: session.start or session.update), muted, speaker_on."""
    from websockets.sync.client import connect

    from g1.voice.openai_live import DuplexCore
    from g1.voice.openai_realtime import KEY_VAR, LiveCore

    send_lock = threading.Lock()

    def emit(msg: tuple) -> None:
        with send_lock:
            try:
                conn.send(msg)
            except (BrokenPipeError, OSError):
                os._exit(0)                          # the parent is gone

    try:
        audio = _speaker(cfg["iface"], cfg["domain"])
        ws = connect(cfg["url"], additional_headers={"Authorization": f"Bearer {os.environ[KEY_VAR]}"},
                     open_timeout=limits.get("voice_connect_timeout_s"), max_size=None)
        mic = MicSocket(cfg["iface"])
    except Exception as e:                           # no key, no network, no robot: the session falls back
        emit(("closed", f"{type(e).__name__}: {e}"))
        return
    ws_lock = threading.Lock()

    def ws_send(ev: dict) -> None:
        with ws_lock:
            ws.send(json.dumps(ev))

    def call(name: str, *args) -> int:
        log: list = []
        code, _ = sdk.call(audio, name, *args, log=log, label="AudioClient")
        for entry in log:                            # to the session's sdk log, without the PCM
            entry["args"] = [a if not isinstance(a, (bytes, bytearray)) else f"<{len(a)} bytes>" for a in entry["args"]]
            emit(("sdk_call", entry))
        return code

    player = Player(lambda stream, pcm: call("PlayStream", APP_NAME, stream, pcm), lambda: call("PlayStop", APP_NAME),
                    emit=emit)
    core = (DuplexCore if cfg["engine"] == "gpt-live" else LiveCore)(ws_send, player, emit, muted=cfg["muted"],
                                                                      speaker_on=cfg["speaker_on"])
    done = threading.Event()

    def mic_loop() -> None:
        while not done.is_set():
            pcm = mic.read()
            if pcm:
                core.on_mic(pcm)

    def player_loop() -> None:
        while not done.is_set():
            player.pump()
            time.sleep(limits.get("voice_poll_s"))

    def parent_loop() -> None:
        while not done.is_set():
            try:
                msg = conn.recv()
            except (EOFError, OSError):
                msg = ("close",)
            if msg[0] == "close":
                done.set()
                player.stop()
                ws.close()
                return
            core.on_parent(msg)

    for target in (mic_loop, player_loop, parent_loop):
        threading.Thread(target=target, name=f"voice-{target.__name__}", daemon=True).start()
    ws_send(cfg["start"])
    why = "closed"
    try:
        for raw in ws:
            core.on_server(json.loads(raw))
    except Exception as e:                           # the connection dropped
        why = f"{type(e).__name__}: {e}"
    done.set()
    player.stop()
    mic.close()
    emit(("closed", why))


def _speaker(iface: str, domain: int):
    """This process's AudioClient (its own DDS participant)."""
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
    ChannelFactoryInitialize(domain, iface)
    audio = AudioClient()
    audio.SetTimeout(limits.get("rpc_timeout_s"))
    audio.Init()
    return audio
