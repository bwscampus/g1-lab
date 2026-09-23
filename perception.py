"""Scene perception: describe camera frames with a vision-language model.

A Perceiver turns frames into ``Percept``s (a one-sentence summary plus the
objects in view with their image position, a rough distance and a locally
computed bearing) on a background thread and publishes latest-only results,
the same way ``camera.Camera`` publishes frames. ``policy.step`` reads
``obs.percept`` and never waits: a model round trip is 1-5 s, the control tick
is 20 ms.

  HFPerceiver    Hugging Face Inference Providers: OpenAI-compatible chat
                 completions with an image, stdlib urllib, streamed SSE
  FakePerceiver  offline and inline: labels the red blob from vision.red_blob,
                 so check, sim and tests run without a token

Smoke-test one image before any sim/robot use:
    HF_TOKEN=hf_... python -m perception head.png
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import statistics
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from typing import Callable, Iterator, Sequence

import numpy as np

from camera import Frame
from vision import bearing, elevation, red_blob

HF_BASE_URL = "https://router.huggingface.co/v1"
# Verified live on the HF router (2026-09): image input, structured output on
# its providers, small active-parameter MoE so it answers fast. Override with
# --vision-model / $G1_VISION_MODEL; ":deepinfra" etc. pins a provider.
DEFAULT_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"

SYSTEM_PROMPT = """You are the vision system of a humanoid robot. The image is from the robot's forward-facing
head camera, which points slightly downward. Describe what is in front of the robot for a
navigation planner. Reply with ONE JSON object and nothing else, exactly this schema:
{"summary": "<one sentence: what is in front of the robot>",
 "objects": [{"label": "<short noun: chair, person, door, ...>",
              "x": <centre column, 0.0 = left edge, 1.0 = right edge>,
              "y": <centre row, 0.0 = top, 1.0 = bottom>,
              "width": <fraction of image width>, "height": <fraction of image height>,
              "distance_m": <estimated metres from the camera, or null>}],
 "path_clear": <true if the robot could walk 2 metres straight ahead without hitting anything>}
List at most 6 objects, nearest first. Use null for unknown distances. No extra keys, no markdown fences."""


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

@dataclass
class Detected:
    label: str
    x: float                    # centre, 0 = left edge .. 1 = right edge
    y: float                    # centre, 0 = top .. 1 = bottom
    width: float                # fraction of the image
    height: float
    distance_m: float | None    # the model's rough estimate
    bearing: float              # rad, positive to the right; computed locally from x
    elevation: float = 0.0      # rad, positive up; computed locally from y

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class Percept:
    summary: str
    objects: list[Detected]
    path_clear: bool
    frame_seq: int              # the frame this describes
    frame_stamp: float          # ... and its stamp on the env clock
    raw: str = ""
    latency: float = 0.0        # wall seconds from request to result
    seq: int = 0                # perceiver counter; key per-percept work on it

    def salient(self) -> Detected | None:
        """Largest object; ties go to the first listed (the prompt asks nearest first)."""
        return max(self.objects, key=lambda d: d.area, default=None)

    def find(self, label: str) -> Detected | None:
        needle = label.lower()
        return next((d for d in self.objects if needle in d.label.lower()), None)

    @classmethod
    def from_json(cls, data: dict, frame: Frame, raw: str = "") -> "Percept":
        shape = frame.image.shape
        objects = []
        for o in data.get("objects") or []:
            if not isinstance(o, dict):
                continue
            x, y = _unit(o.get("x"), 0.5), _unit(o.get("y"), 0.5)
            w, h = _unit(o.get("width"), 0.0), _unit(o.get("height"), 0.0)
            objects.append(Detected(str(o.get("label") or "object"), x, y, w, h,
                                    _distance(o.get("distance_m")), bearing(2 * x - 1, shape),
                                    elevation(2 * y - 1, shape)))
        return cls(str(data.get("summary") or ""), objects, bool(data.get("path_clear", True)),
                   frame.seq, frame.stamp, raw)


def _unit(v, default: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(f) else min(1.0, max(0.0, f))


def _distance(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f >= 0 else None


def extract_json(text: str) -> dict:
    """The first JSON object in ``text``, tolerating prose and ``` fences around it."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in model output: {text[:120]!r}")
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"bad JSON in model output: {e}") from None
    if not isinstance(data, dict):
        raise ValueError("model output is not a JSON object")
    return data


def encode_jpeg(image_rgb: np.ndarray, max_width: int = 640, quality: int = 80) -> bytes:
    import cv2
    h, w = image_rgb.shape[:2]
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if w > max_width:
        bgr = cv2.resize(bgr, (max_width, int(round(h * max_width / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


# --------------------------------------------------------------------------
# Perceivers
# --------------------------------------------------------------------------

class Perceiver:
    """Base class: ``offer(frame)`` never blocks, ``latest()`` is the newest
    Percept. A daemon worker runs ``describe`` on the newest offered frame, one
    request at a time, at most once per ``min_interval`` seconds. Errors are
    counted and printed sparingly, never raised into the control loop."""

    def __init__(self, *, min_interval: float = 2.0, threaded: bool = True) -> None:
        self.min_interval = min_interval
        self.threaded = threaded
        self._cond = threading.Condition()
        self._pending: Frame | None = None
        self._lock = threading.Lock()
        self._latest: Percept | None = None
        self._seq = 0
        self._done_seq: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.requests = 0
        self.errors = 0
        self.last_error: BaseException | None = None
        self._latencies: list[float] = []

    def describe(self, frame: Frame) -> Percept:
        """Blocking: turn one frame into a Percept. May raise."""
        raise NotImplementedError

    def offer(self, frame: Frame) -> None:
        if not self.threaded:
            if frame.seq != self._done_seq:
                self._process(frame)
            return
        with self._cond:
            self._pending = frame
            self._cond.notify()

    def latest(self) -> Percept | None:
        with self._lock:
            return self._latest

    @property
    def count(self) -> int:
        return self._seq

    def start(self) -> None:
        if not self.threaded or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="perceiver", daemon=True)
        self._thread.start()

    def stop(self, join: float = 1.0) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        t = self._thread
        if t is not None:
            t.join(timeout=join)      # a hung request must never block teardown
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._cond:
                while self._pending is None and not self._stop.is_set():
                    self._cond.wait(0.5)
                frame, self._pending = self._pending, None
            if frame is None or frame.seq == self._done_seq:
                continue
            t0 = time.monotonic()
            self._process(frame)
            self._stop.wait(max(0.0, self.min_interval - (time.monotonic() - t0)))

    def _process(self, frame: Frame) -> None:
        self._done_seq = frame.seq
        self.requests += 1
        t0 = time.monotonic()
        try:
            p = self.describe(frame)
        except Exception as e:
            self.errors += 1
            self.last_error = e
            if self.errors <= 3 or self.errors % 50 == 0:
                print(f"perception: error {self.errors}: {e}")
            return
        p.latency = time.monotonic() - t0
        self._latencies.append(p.latency)
        with self._lock:
            self._seq += 1
            p.seq = self._seq
            self._latest = p

    def summary(self) -> str:
        if self._latencies:
            lat = (f"latency mean {statistics.mean(self._latencies):.2f}s, "
                   f"max {max(self._latencies):.2f}s")
        else:
            lat = "no results"
        s = (f"perception: {self.requests} request(s), {self._seq} percept(s), "
             f"{self.errors} error(s); {lat}")
        if self.last_error is not None:
            s += f"; last error: {self.last_error}"
        return s


class FakePerceiver(Perceiver):
    """Offline stand-in, run inline in ``offer`` so check (which runs far
    faster than realtime) stays deterministic. Labels the red blob, or cycles
    a scripted list of Percepts."""

    def __init__(self, label: str = "red ball", script: Sequence[Percept] | None = None,
                 min_interval: float = 0.0) -> None:
        super().__init__(min_interval=min_interval, threaded=False)
        self.label = label
        self.script = list(script or [])
        self._i = 0

    def describe(self, frame: Frame) -> Percept:
        if self.script:
            p = self.script[self._i % len(self.script)]
            self._i += 1
            return replace(p, frame_seq=frame.seq, frame_stamp=frame.stamp, seq=0, latency=0.0)
        blob = red_blob(frame.image)
        if blob is None:
            return Percept("nothing of interest", [], True, frame.seq, frame.stamp)
        u, v, frac = blob
        size = math.sqrt(frac)
        d = Detected(self.label, (u + 1) / 2, (v + 1) / 2, size, size, None,
                     bearing(u, frame.image.shape), elevation(v, frame.image.shape))
        return Percept(f"a {self.label} at {math.degrees(d.bearing):+.0f} deg", [d], frac < 0.2,
                       frame.seq, frame.stamp)


class RequestError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


def _sse_post(url: str, headers: dict, body: dict, timeout: float) -> Iterator[str]:
    """POST JSON; yield each SSE ``data:`` payload, or the whole body when the
    response is not a stream."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise RequestError(e.code, e.read().decode("utf-8", "replace")[:200]) from None
    with resp:
        if not body.get("stream"):
            yield resp.read().decode("utf-8", "replace")
            return
        seen = False
        plain: list[str] = []
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("data:"):
                seen = True
                yield line[5:].strip()
            elif line:
                plain.append(line)
        if not seen:                     # a provider answered without streaming
            yield "\n".join(plain)


class HFPerceiver(Perceiver):
    """Hugging Face Inference Providers over the OpenAI-compatible endpoint."""

    def __init__(self, model: str = DEFAULT_MODEL, token: str | None = None, *,
                 base_url: str = HF_BASE_URL, min_interval: float = 2.0, stream: bool = True,
                 json_mode: bool = True, max_width: int = 640, timeout: float = 30.0,
                 on_text: Callable[[str], None] | None = None,
                 transport: Callable[[str, dict, dict, float], Iterator[str]] | None = None) -> None:
        super().__init__(min_interval=min_interval)
        self.model = model
        self.token = token or ""
        self.base_url = base_url.rstrip("/")
        self.stream = stream
        self.json_mode = json_mode
        self.max_width = max_width
        self.timeout = timeout
        self.on_text = on_text
        self.transport = transport or _sse_post

    @classmethod
    def from_env(cls, model: str | None = None, **kw) -> "HFPerceiver":
        token = os.environ.get("HF_TOKEN") or os.environ.get("G1_VISION_API_KEY")
        if not token:
            raise RuntimeError("no Hugging Face token: set HF_TOKEN (or G1_VISION_API_KEY)")
        return cls(model or os.environ.get("G1_VISION_MODEL") or DEFAULT_MODEL, token, **kw)

    def build_request(self, frame: Frame) -> dict:
        data_url = "data:image/jpeg;base64," + base64.b64encode(
            encode_jpeg(frame.image, self.max_width)).decode("ascii")
        body = {
            "model": self.model, "stream": self.stream, "temperature": 0, "max_tokens": 400,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "Describe this frame."},
                ]},
            ],
        }
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    def describe(self, frame: Frame) -> Percept:
        body = self.build_request(frame)
        try:
            text = self._complete(body)
        except RequestError as e:
            # JSON mode is provider-dependent on the HF router: drop it and rely on the prompt.
            if self.json_mode and e.status == 400 and "response_format" in e.body:
                self.json_mode = False
                body.pop("response_format", None)
                text = self._complete(body)
            else:
                raise
        return Percept.from_json(extract_json(text), frame, raw=text)

    def _complete(self, body: dict) -> str:
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        parts: list[str] = []
        for payload in self.transport(self.base_url + "/chat/completions", headers, body, self.timeout):
            if payload.strip() == "[DONE]":
                break
            chunk = json.loads(payload)
            choices = chunk.get("choices") or []
            if not choices:
                continue
            c = choices[0]
            delta = (c.get("delta") or {}).get("content") or (c.get("message") or {}).get("content") or ""
            if delta:
                parts.append(delta)
                if self.on_text is not None:
                    self.on_text(delta)
        return "".join(parts)


# --------------------------------------------------------------------------
# CLI plumbing
# --------------------------------------------------------------------------

VISION_MODES = ("auto", "off", "fake", "api")


def build_perceiver(args: argparse.Namespace, policy) -> Perceiver | None:
    """Resolve --vision: auto = fake iff the policy uses vision; api is always explicit."""
    mode = getattr(args, "vision", "auto")
    if mode == "off" or (mode == "auto" and not getattr(policy, "uses_vision", False)):
        return None
    if mode in ("auto", "fake"):
        return FakePerceiver(label=getattr(args, "vision_fake_label", "red ball"))
    echo = (lambda s: print(s, end="", flush=True)) if getattr(args, "vision_echo", False) else None
    return HFPerceiver.from_env(getattr(args, "vision_model", None),
                                min_interval=getattr(args, "vision_interval", 2.0), on_text=echo)


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m perception",
                                description="describe one image with the vision model (needs $HF_TOKEN)")
    p.add_argument("image", help="image file (png/jpg)")
    p.add_argument("--model", default=None, help=f"HF model id (default: $G1_VISION_MODEL or {DEFAULT_MODEL})")
    p.add_argument("--no-stream", action="store_true")
    args = p.parse_args(argv)
    import cv2
    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        p.error(f"could not read {args.image}")
    frame = Frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), 0.0, 1)
    try:
        per = HFPerceiver.from_env(args.model, stream=not args.no_stream,
                                   on_text=lambda s: print(s, end="", flush=True))
    except RuntimeError as e:
        p.error(str(e))
    print(f"model {per.model}; image {frame.image.shape[1]}x{frame.image.shape[0]}")
    t0 = time.monotonic()
    percept = per.describe(frame)
    print(f"\n-- {time.monotonic() - t0:.2f}s")
    print(f"summary:    {percept.summary}")
    print(f"path_clear: {percept.path_clear}")
    for d in percept.objects:
        dist = "?" if d.distance_m is None else f"{d.distance_m:.1f} m"
        print(f"  {d.label:<14} x={d.x:.2f} y={d.y:.2f} size={d.width:.2f}x{d.height:.2f} "
              f"bearing={math.degrees(d.bearing):+.0f} deg dist={dist}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
