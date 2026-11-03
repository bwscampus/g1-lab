"""Camera frame sources.

Every env can hand the program a camera frame alongside the joint state (see
``core.action.Obs``). All sources share one rule: a **latest-only slot**. A
producer overwrites the slot whenever a new frame exists and a consumer reads
whatever is there, so nothing queues up behind a slow 20 ms tick, and
``Frame.stamp`` lets the agent tell a fresh frame from a stale one.

  WebRTCCamera  the G1 head camera over unitree_webrtc_connect (robot env, from another machine)
  DirectCamera  the G1 head camera on the robot's own Jetson: the video hub's multicast,
                decoded in hardware by a GStreamer child process (robot env, on the Jetson)
  ShmCamera     the head camera g1 twin renders, from shared memory (robot env on lo0)
  DirCamera     replays image files from a directory on the env's clock (tests)

  Viewer        ``--view``: the live head camera in a browser (MJPEG on localhost)
"""
from __future__ import annotations

import asyncio
import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from g1.core import limits

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")

# aiortc reassembles a video frame in a 128-packet jitter buffer; a longer frame comes
# out as its tail only. The G1's keyframes (SPS + PPS + IDR) run ~150 packets in a
# detailed scene, so with 128 every keyframe is cut and nothing ever decodes.
VIDEO_JITTER_PACKETS = 1024     # a power of 2 (aiortc's requirement)

# The head camera (a RealSense D435i) is plugged into the robot's Jetson. Unitree's video hub
# there (VIDEO_HUB/videohub_pc4, a GStreamer pipeline) encodes its colour stream in hardware,
# 1280x720 H.264 at 15 fps with a keyframe a second, and multicasts it as RTP on the robot's
# network; the main board (.161) relays that same stream over WebRTC. On the Jetson we take
# the multicast directly: no relay, no jitter buffer, no single-client limit, and the
# Jetson's hardware decoder in its own process (a Python decoder next to rt/lowstate's
# ~900 messages/s starved for the GIL there and fell to ~0 fps).
VIDEO_HUB = Path("/unitree/module/video_hub_pc4")
VIDEO_HUB_GROUP = "230.1.1.1"
VIDEO_HUB_PORT = 1720
VIDEO_HUB_SIZE = (720, 1280)    # (H, W) of the stream the hub multicasts


@dataclass
class Frame:
    image: np.ndarray   # (H, W, 3) uint8, RGB
    stamp: float        # seconds on the env's clock (Env.clock) when captured
    seq: int = 0        # increments per published frame; key per-frame work on it


class Camera:
    """Base class: a thread-safe latest-only frame slot."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Frame | None = None
        self._seq = 0
        self._subscribers: list[Callable[[Frame], None]] = []

    def start(self) -> None: ...
    def stop(self) -> None: ...

    def publish(self, image: np.ndarray, stamp: float) -> Frame:
        image = np.ascontiguousarray(image)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"frame must be (H, W, 3) uint8 RGB, got {image.shape} {image.dtype}")
        with self._lock:
            self._seq += 1
            self._frame = frame = Frame(image, float(stamp), self._seq)
            subscribers = list(self._subscribers)
        for fn in subscribers:
            try:
                fn(frame)                    # on the producer's thread: a subscriber only enqueues
            except Exception as e:
                print(f"camera: subscriber {getattr(fn, '__name__', fn)} failed and was dropped: {e}")
                self.unsubscribe(fn)
        return frame

    def subscribe(self, fn: Callable[[Frame], None]) -> None:
        """Call ``fn(frame)`` on every publish, on the producer's thread. It must
        return at once (the taps only enqueue); one that raises is dropped."""
        with self._lock:
            if fn not in self._subscribers:
                self._subscribers.append(fn)

    def unsubscribe(self, fn: Callable[[Frame], None]) -> None:
        with self._lock:
            if fn in self._subscribers:
                self._subscribers.remove(fn)

    def latest(self) -> Frame | None:
        with self._lock:
            return self._frame

    @property
    def count(self) -> int:
        return self._seq


class ClockedCamera(Camera):
    """A source the env drives from its own clock: call ``poll(now)`` every tick.
    Frame ``k`` is due at ``k / fps`` seconds after the first poll; if several
    frames fell due since the last poll only the newest is published."""

    def __init__(self, fps: float = limits.get("twin_camera_fps")) -> None:
        super().__init__()
        self.fps = fps
        self._t0: float | None = None
        self._issued = 0

    def image(self, index: int) -> np.ndarray | None:
        """The image for frame ``index``, or None to publish nothing."""
        raise NotImplementedError

    def poll(self, now: float) -> Frame | None:
        if self._t0 is None:
            self._t0 = now
        due = int(math.floor((now - self._t0) * self.fps + 1e-9)) + 1
        if due > self._issued:
            self._issued = due
            img = self.image(due - 1)
            if img is not None:
                self.publish(img, now)
        return self.latest()


class DirCamera(ClockedCamera):
    """Replay the image files in a directory, sorted by name."""

    def __init__(self, path: str | Path, fps: float = limits.get("twin_camera_fps"), loop: bool = False) -> None:
        super().__init__(fps)
        self.path = Path(path)
        self.loop = loop
        self.files: list[Path] = []
        self._last = -1

    def start(self) -> None:
        self.files = sorted(p for p in self.path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not self.files:
            raise FileNotFoundError(f"no image files in {self.path}")

    def image(self, index: int) -> np.ndarray | None:
        n = len(self.files)
        i = index % n if self.loop else min(index, n - 1)
        if i == self._last:
            return None                     # ran out (or no new frame yet): keep the last one
        self._last = i
        from g1.core import images
        return images.read_rgb(self.files[i])


class WebRTCCamera(Camera):
    """The G1 head camera via unitree_webrtc_connect, decoded on a background
    asyncio thread and stamped with ``time.monotonic()``. ``start`` blocks until
    the first frame arrives or raises. The robot accepts a single WebRTC client,
    so disconnect the Unitree app first."""

    def __init__(self, ip: str, aes_key: str | None = None, timeout: float = limits.get("camera_timeout_s")) -> None:
        super().__init__()
        self.ip = ip
        self.aes_key = aes_key
        self.timeout = timeout
        self.conn = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.thread: threading.Thread | None = None
        self.error: BaseException | None = None
        self._stopping = False

    def start(self) -> None:
        import logging
        self._stopping = False
        from unitree_webrtc_connect.webrtc_driver import (UnitreeWebRTCConnection,
                                                          WebRTCConnectionMethod)
        # The packets that arrive before the first keyframe cannot be decoded; aiortc
        # logs each as "H264Decoder() failed to decode, skipping package". That is
        # normal at stream start (the G1 sends 1280x720 H.264 at ~15 fps), not a fault.
        logging.getLogger("aiortc.codecs.h264").setLevel(logging.ERROR)
        roomy_video_jitter_buffer()
        self.conn =UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=self.ip,
                                            aes_128_key=self.aes_key)
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run, name="webrtc-camera", daemon=True)
        self.thread.start()
        t0 = time.monotonic()
        while self.latest() is None:
            if self.error is not None:
                self.stop()
                raise ConnectionError(f"camera at {self.ip}: {self.error}") from self.error
            if time.monotonic() - t0 > self.timeout:
                self.stop()
                raise TimeoutError(f"no camera frame from {self.ip} within {self.timeout:.0f}s "
                                   "(is the Unitree app still connected to the robot?)")
            time.sleep(0.05)

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)

        async def setup():
            await self.conn.connect()
            self.conn.video.switchVideoChannel(True)
            self.conn.video.add_track_callback(self._recv)

        try:
            self.loop.run_until_complete(setup())
        except BaseException as e:      # surfaced to start() on the main thread
            self.error = e
            return
        self.loop.run_forever()

    async def _recv(self, track) -> None:
        while True:
            try:
                frame = await track.recv()
            except Exception as e:
                if self._stopping:          # the track ends when we disconnect: not an error
                    return
                self.error = e              # the stream died mid-run: say so, once
                print(f"camera: stream ended: {type(e).__name__}: {e}")
                return
            self.publish(frame.to_ndarray(format="rgb24"), time.monotonic())

    def stop(self) -> None:
        self._stopping = True
        loop, thread = self.loop, self.thread
        if thread is None:
            return
        if loop.is_running():
            if self.conn is not None:
                try:
                    asyncio.run_coroutine_threadsafe(self.conn.disconnect(), loop).result(timeout=5.0)
                except Exception as e:
                    print(f"camera: disconnect failed: {e}")
            loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5.0)
        if not loop.is_running():
            loop.close()
        self.thread = None


def roomy_video_jitter_buffer(packets: int = VIDEO_JITTER_PACKETS) -> None:
    """Give video ``packets`` of jitter buffer in every aiortc receiver made from now on
    (audio keeps its own). aiortc has no setting for it, so its receiver module's
    JitterBuffer is replaced by a subclass of the original; calling it again is harmless."""
    from aiortc import rtcrtpreceiver
    from aiortc.jitterbuffer import JitterBuffer

    class RoomyJitterBuffer(JitterBuffer):
        def __init__(self, capacity: int, prefetch: int = 0, is_video: bool = False) -> None:
            super().__init__(max(capacity, packets) if is_video else capacity, prefetch, is_video)

    rtcrtpreceiver.JitterBuffer = RoomyJitterBuffer


def on_the_jetson() -> bool:
    """True on the robot's own Jetson: Unitree's video hub is installed and GStreamer runs."""
    return VIDEO_HUB.is_dir() and shutil.which("gst-launch-1.0") is not None


def video_hub_pipeline(iface: str, fd: int) -> list[str]:
    """gst-launch-1.0 arguments: the video hub's RTP multicast on ``iface``, decoded by the
    Jetson's hardware decoder (nvv4l2decoder), as raw RGBA frames written to ``fd``.

    A slow reader must cost whole frames, never packets: a lost packet smears the picture
    until the next keyframe (up to a second). So the socket has its own thread (the queue
    after udpsrc; the kernel's 212 KB buffer holds about one keyframe) and the decoded frames
    wait in a leaky queue that drops the oldest when the reader falls behind."""
    h, w = VIDEO_HUB_SIZE
    return ["gst-launch-1.0", "-q",
            "udpsrc", f"address={VIDEO_HUB_GROUP}", f"port={VIDEO_HUB_PORT}", f"multicast-iface={iface}",
            "caps=application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000",
            "!", "queue",
            "!", "rtph264depay", "!", "h264parse", "!", "nvv4l2decoder",
            "!", "nvvidconv", "!", f"video/x-raw,format=RGBA,width={w},height={h}",
            "!", "queue", "leaky=downstream", "max-size-buffers=1", "max-size-bytes=0", "max-size-time=0",
            "!", "fdsink", f"fd={fd}", "sync=false"]


class DirectCamera(Camera):
    """The G1 head camera on the robot's own Jetson: the video hub's multicast, decoded in
    hardware by a ``gst-launch-1.0`` child process that writes raw RGBA frames to a pipe
    (its own fd: the NVIDIA decoder prints to stdout). A thread reads whole frames and
    publishes them stamped with ``time.monotonic()``. ``start`` blocks until the first frame
    or raises, like WebRTCCamera. ``command(fd)`` builds the child's argv (tests swap it)."""

    def __init__(self, iface: str, timeout: float = limits.get("camera_timeout_s"),
                 size: tuple[int, int] = VIDEO_HUB_SIZE,
                 command: Optional[Callable[[int], list[str]]] = None) -> None:
        super().__init__()
        self.iface = iface
        self.timeout = timeout
        self.size = size
        self.command = command or (lambda fd: video_hub_pipeline(iface, fd))
        self.proc: subprocess.Popen | None = None
        self.pipe = None
        self.log = None
        self.thread: threading.Thread | None = None
        self.error: BaseException | None = None
        self._stopping = False

    def start(self) -> None:
        self._stopping = False
        self.error = None
        read_fd, write_fd = os.pipe()
        self.log = tempfile.TemporaryFile()
        try:
            self.proc = subprocess.Popen(self.command(write_fd), pass_fds=(write_fd,), stdin=subprocess.DEVNULL,
                                         stdout=subprocess.DEVNULL, stderr=self.log)
        except OSError as e:
            os.close(read_fd)
            raise ConnectionError(f"camera: cannot start the decoder: {e}") from e
        finally:
            os.close(write_fd)
        self.pipe = os.fdopen(read_fd, "rb", buffering=0)
        self.thread = threading.Thread(target=self._run, name="direct-camera", daemon=True)
        self.thread.start()
        t0 = time.monotonic()
        while self.latest() is None:
            if self.error is not None or (self.thread is not None and not self.thread.is_alive()):
                why = self.stderr_tail()
                self.stop()
                raise ConnectionError(f"camera: the video hub decoder stopped before a frame ({why or 'no output'}); "
                                      f"is videohub_pc4 running, and is UNITREE_IFACE the Jetson's robot-network "
                                      f"interface (eth0, not {self.iface!r})?")
            if time.monotonic() - t0 > self.timeout:
                self.stop()
                raise TimeoutError(f"no frame from the video hub's multicast {VIDEO_HUB_GROUP}:{VIDEO_HUB_PORT} "
                                   f"on {self.iface} within {self.timeout:.0f}s (is videohub_pc4 running?)")
            time.sleep(0.02)

    def _run(self) -> None:
        h, w = self.size
        size = h * w * 4
        buf = bytearray(size)
        view = memoryview(buf)
        while True:
            got = 0
            while got < size:
                try:
                    n = self.pipe.readinto(view[got:])
                except (OSError, ValueError) as e:     # the pipe closed under us by stop()
                    n, err = 0, e
                else:
                    err = None
                if not n:
                    if not self._stopping:
                        self.error = err or EOFError("the decoder exited")
                        if self.latest() is not None:   # the stream died mid-run: say so, once
                            print(f"camera: stream ended: {self.error} {self.stderr_tail()}".rstrip())
                    return
                got += n
            rgb = np.frombuffer(buf, np.uint8).reshape(h, w, 4)[:, :, :3]
            self.publish(rgb.copy(), time.monotonic())

    def stderr_tail(self, lines: int = 3) -> str:
        if self.log is None:
            return ""
        try:
            self.log.seek(0)
            text = self.log.read().decode(errors="replace").strip().splitlines()
        except (OSError, ValueError):
            return ""
        return " | ".join(text[-lines:])

    def stop(self) -> None:
        self._stopping = True
        if self.proc is not None:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait()
            self.proc = None
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None
        if self.pipe is not None:
            self.pipe.close()
            self.pipe = None
        if self.log is not None:
            self.log.close()
            self.log = None


class ShmCamera(Camera):
    """The head camera ``g1 twin`` renders, read from shared memory
    (``g1/twin/shm.py``) on a background thread and stamped with the twin's
    ``time.monotonic()``. ``start`` blocks until the first frame arrives or raises,
    like WebRTCCamera."""

    def __init__(self, domain: int, timeout: float = limits.get("camera_timeout_s"),
                 poll_s: float = 1 / limits.get("camera_poll_hz")) -> None:
        super().__init__()
        self.domain = domain
        self.timeout = timeout
        self.poll_s = poll_s
        self.reader = None
        self.thread: threading.Thread | None = None
        self._stopping = False

    def start(self) -> None:
        from g1.twin.shm import FrameReader, segment_name
        t0 = time.monotonic()
        while self.reader is None:
            try:
                self.reader = FrameReader(self.domain)
            except FileNotFoundError:
                if time.monotonic() - t0 > self.timeout:
                    raise TimeoutError(f"no g1 twin camera ({segment_name(self.domain)}) within "
                                       f"{self.timeout:.0f}s: is g1 twin running on domain {self.domain}?") from None
                time.sleep(0.1)
        self._stopping = False
        self.thread = threading.Thread(target=self._run, name="shm-camera", daemon=True)
        self.thread.start()
        while self.latest() is None:
            if time.monotonic() - t0 > self.timeout:
                self.stop()
                raise TimeoutError(f"g1 twin's camera sent no frame within {self.timeout:.0f}s")
            time.sleep(0.02)

    def _run(self) -> None:
        while not self._stopping:
            got = self.reader.read()
            if got is not None:
                self.publish(*got)
            time.sleep(self.poll_s)

    def stop(self) -> None:
        self._stopping = True
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None
        if self.reader is not None:
            self.reader.close()
            self.reader = None


class Viewer:
    """``g1 run TASK --mode ... --view``: the robot's head camera in a browser, an MJPEG
    server on ``127.0.0.1:port`` (0 = any free port), reading the same latest-only slot as
    the agent on its own threads, so it never touches the control tick. The robot takes
    one WebRTC client, so the live picture has to come out of this process.
    ``/`` is a page showing the stream, ``/stream`` the multipart stream (a
    new JPEG whenever the slot's seq changes; a slow browser skips frames),
    ``/frame`` one JPEG. Encoding happens on the request's thread."""

    def __init__(self, camera: Camera, port: int = limits.get("view_port"), *,
                 width: int = limits.get("view_width_px"), quality: int = limits.get("view_jpeg_quality"),
                 title: str = "g1", host: str = "127.0.0.1") -> None:
        self.camera = camera
        self.width = width
        self.quality = quality
        self.title = title
        self.host = host
        self.port = port
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def jpeg(self) -> tuple[Optional[bytes], int]:
        f = self.camera.latest()
        if f is None:
            return None, 0
        from g1.core import images
        return images.encode_jpeg(f.image, self.width, self.quality), f.seq

    def start(self) -> None:
        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:      # quiet
                pass

            def do_GET(self) -> None:
                if self.path.startswith("/stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    last = 0
                    try:
                        while viewer.server is not None:
                            data, seq = viewer.jpeg()
                            if data is None or seq == last:
                                time.sleep(1 / limits.get("view_fps"))
                                continue
                            last = seq
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                             + f"Content-Length: {len(data)}\r\n\r\n".encode() + data + b"\r\n")
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return
                elif self.path.startswith("/frame"):
                    data, _ = viewer.jpeg()
                    if data is None:
                        self.send_error(503, "no frame yet")
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    page = (f"<!doctype html><title>{viewer.title}</title><body style='margin:0;background:#111'>"
                            f"<img src='/stream' style='max-width:100vw;max-height:100vh'></body>").encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(page)))
                    self.end_headers()
                    self.wfile.write(page)

        self.server = ThreadingHTTPServer((self.host, self.port), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, name="viewer", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        server, self.server = self.server, None
        if server is None:
            return
        server.shutdown()
        server.server_close()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None
