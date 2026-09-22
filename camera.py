"""Camera frame sources.

Every env can hand the policy a camera frame alongside the joint state (see
``policy.Obs``). All sources share one rule: a **latest-only slot**. A producer
overwrites the slot whenever a new frame exists and a consumer reads whatever is
there, so nothing queues up behind a slow 20 ms tick, and ``Frame.stamp`` lets
the policy tell a fresh frame from a stale one.

  WebRTCCamera  the G1 head camera over unitree_webrtc_connect (robot env)
  DirCamera     replays image files from a directory on the env's clock (check, tests)
  NoiseCamera   random frames, for fuzzing a camera policy in check

Smoke-test the robot stream without touching any joint:
    python -m camera --ip 192.168.123.164
"""
from __future__ import annotations

import argparse
import asyncio
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

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

    def start(self) -> None: ...
    def stop(self) -> None: ...

    def publish(self, image: np.ndarray, stamp: float) -> Frame:
        image = np.ascontiguousarray(image)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"frame must be (H, W, 3) uint8 RGB, got {image.shape} {image.dtype}")
        with self._lock:
            self._seq += 1
            self._frame = Frame(image, float(stamp), self._seq)
            return self._frame

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
        import cv2
        bgr = cv2.imread(str(self.files[i]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"could not decode {self.files[i]}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class NoiseCamera(ClockedCamera):
    """Random frames: pixel noise plus one solid random-colour block, so colour
    detectors see blobs at random places and sizes."""

    def __init__(self, fps: float = 15.0, size: tuple[int, int] = (480, 640), seed: int = 0) -> None:
        super().__init__(fps)
        self.size = size
        self.rng = np.random.default_rng(seed)

    def image(self, index: int) -> np.ndarray:
        h, w = self.size
        img = self.rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        y0, x0 = int(self.rng.integers(0, h // 2)), int(self.rng.integers(0, w // 2))
        img[y0:y0 + h // 2, x0:x0 + w // 2] = self.rng.integers(0, 256, 3, dtype=np.uint8)
        return img


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

    def start(self) -> None:
        from unitree_webrtc_connect.webrtc_driver import (UnitreeWebRTCConnection,
                                                          WebRTCConnectionMethod)
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
            frame = await track.recv()
            self.publish(frame.to_ndarray(format="rgb24"), time.monotonic())

    def stop(self) -> None:
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


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m camera",
                                description="print the head camera frame rate; no robot control")
    p.add_argument("--ip", default=os.environ.get("UNITREE_ROBOT_IP"),
                   help="robot IP (default: $UNITREE_ROBOT_IP)")
    p.add_argument("--seconds", type=float, default=5.0)
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
    finally:
        cam.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
