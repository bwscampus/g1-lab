"""Camera frame sources.

Every env can hand the program a camera frame alongside the joint state (see
``core.action.Obs``). All sources share one rule: a **latest-only slot**. A
producer overwrites the slot whenever a new frame exists and a consumer reads
whatever is there, so nothing queues up behind a slow 20 ms tick, and
``Frame.stamp`` lets the agent tell a fresh frame from a stale one.

  WebRTCCamera  the G1 head camera over unitree_webrtc_connect (robot env)
  DirCamera     replays image files from a directory on the env's clock (sim, tests)

Two **taps** hang off any slot without touching the control loop, each on its
own thread, dropping frames when it falls behind:

  Viewer        ``--view``: an MJPEG server on localhost; open it in a browser
  Recorder      ``--record``: every frame to an .mp4 with PyAV, timestamped on the
                env's clock, so a sim run plays back at sim speed

Smoke-test the robot stream without touching any joint:
    g1 camera --ip 192.168.123.164
"""
from __future__ import annotations

import argparse
import asyncio
import math
import os
import queue
import threading
import time
from dataclasses import dataclass
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Optional

import numpy as np

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")


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

    def __init__(self, fps: float = 15.0) -> None:
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

    def __init__(self, path: str | Path, fps: float = 15.0, loop: bool = False) -> None:
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

    def __init__(self, ip: str, aes_key: str | None = None, timeout: float = 15.0) -> None:
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
        self.conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=self.ip,
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


# --------------------------------------------------------------------------
# Taps: a live view and a recording, off the slot, never on the tick
# --------------------------------------------------------------------------

class Viewer:
    """``--view``: an MJPEG server on ``127.0.0.1:port`` (0 = any free port).
    ``/`` is a page showing the stream, ``/stream`` the multipart stream (a
    new JPEG whenever the slot's seq changes; a slow browser skips frames),
    ``/frame`` one JPEG. Encoding happens on the request's thread."""

    def __init__(self, camera: Camera, port: int = 8765, *, width: int = 960, quality: int = 80,
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
                                time.sleep(1 / 30)
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


class Recorder:
    """``--record``: every published frame to an .mp4 (PyAV, libx264, yuv420p),
    variable frame rate with each frame's own stamp as its time, so a headless
    sim run plays back at sim speed and a robot run at real time. Frames are
    queued from the producer's thread (bounded; the oldest is dropped when the
    encoder falls behind) and encoded on this thread."""

    def __init__(self, camera: Camera, path: Path | str, *, codec: Optional[str] = None, queue_size: int = 8) -> None:
        try:
            import av  # noqa: F401
        except ImportError:
            raise RuntimeError("--record needs PyAV: pip install av") from None
        self.camera = camera
        self.path = Path(path)
        self.codec = codec
        self._q: "queue.Queue[Optional[Frame]]" = queue.Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        self.frames = 0
        self.dropped = 0
        self.duration_s = 0.0
        self.error: Optional[BaseException] = None

    def _on_frame(self, frame: Frame) -> None:
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            try:
                self._q.get_nowait()          # drop the oldest, keep the newest
                self.dropped += 1
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(frame)
            except queue.Full:
                self.dropped += 1

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="recorder", daemon=True)
        self._thread.start()
        self.camera.subscribe(self._on_frame)

    def stop(self, join: float = 30.0) -> None:
        self.camera.unsubscribe(self._on_frame)
        if self._thread is None:
            return
        self._q.put(None)                     # blocks until there is room: nothing is lost at the end
        self._thread.join(timeout=join)
        self._thread = None

    def summary(self) -> str:
        if self.error is not None:
            return f"recording failed: {self.error}"
        return (f"recorded {self.frames} frames, {self.duration_s:.1f} s, {self.dropped} dropped -> {self.path}")

    def _run(self) -> None:
        import av
        container = None
        stream = None
        t0: Optional[float] = None
        last_pts = -1
        try:
            while True:
                frame = self._q.get()
                if frame is None:
                    break
                h, w = frame.image.shape[:2]
                image = frame.image[:h - h % 2, :w - w % 2]      # yuv420p needs even sides
                if container is None:
                    container = av.open(str(self.path), mode="w")
                    stream = _add_stream(container, self.codec, image.shape[1], image.shape[0])
                    t0 = frame.stamp
                pts = max(last_pts + 1, round((frame.stamp - t0) * 1000))
                last_pts = pts
                vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24")
                vf.pts = pts
                vf.time_base = Fraction(1, 1000)
                for packet in stream.encode(vf):
                    container.mux(packet)
                self.frames += 1
                self.duration_s = pts / 1000
        except Exception as e:
            self.error = e
            print(f"recorder: {e}")
        finally:
            if container is not None:
                try:
                    for packet in stream.encode(None):
                        container.mux(packet)
                finally:
                    container.close()


def _add_stream(container, codec: Optional[str], width: int, height: int):
    """libx264 when PyAV has it, else mpeg4; ``codec`` forces one."""
    import av
    names = [codec] if codec else ["libx264", "mpeg4"]
    for name in names:
        try:
            stream = container.add_stream(name, rate=30)       # nominal; frames carry their own pts
        except Exception:
            continue
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1000)
        if name == "libx264":
            stream.options = {"preset": "veryfast", "crf": "23"}
        return stream
    raise RuntimeError(f"no video encoder available among {names} (av {av.__version__})")


def main(argv: list[str] | None = None) -> int:
    from g1.vlm import load_dotenv
    load_dotenv()              # $UNITREE_ROBOT_IP / $UNITREE_AES_128_KEY may live in .env
    p = argparse.ArgumentParser(prog="g1 camera",
                                description="print the head camera frame rate; no robot control")
    p.add_argument("--ip", default=os.environ.get("UNITREE_ROBOT_IP"),
                   help="robot IP (default: $UNITREE_ROBOT_IP)")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--save", default=None, metavar="PNG",
                   help="also save the last frame, to look at it or to try the vision model on it")
    args = p.parse_args(argv)
    if not args.ip:
        p.error("--ip is required (or set $UNITREE_ROBOT_IP)")
    cam = WebRTCCamera(args.ip, os.environ.get("UNITREE_AES_128_KEY"))
    cam.start()
    try:
        t0 = time.monotonic()
        last = cam.latest()
        n0 = last.seq
        gaps: list[float] = []
        while time.monotonic() - t0 < args.seconds:
            f = cam.latest()
            if f.seq != last.seq:
                gaps.append(f.stamp - last.stamp)
                last = f
            time.sleep(0.005)
        n = cam.count - n0
        print(f"{n} frames in {args.seconds:.1f}s = {n / args.seconds:.1f} fps; "
              f"shape {last.image.shape}; frame gap mean {np.mean(gaps) * 1e3:.1f} ms, "
              f"max {np.max(gaps) * 1e3:.1f} ms")
        if args.save:
            from g1.core import images
            images.write_png(args.save, last.image)
            print(f"saved {args.save}")
    finally:
        cam.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
