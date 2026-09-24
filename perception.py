"""Scene perception: describe camera frames with a vision-language model.

A Perceiver turns frames into ``Percept``s (a one-sentence summary plus the
objects in view with their image position, a rough distance and a locally
computed bearing). It runs **only when queried**: the env offers it every new
frame (a pointer swap), a policy asks with ``request()`` when it wants a fresh
view, the worker describes the frame that was current at that moment, and the
result appears as ``obs.percept`` a few ticks later. ``policy.step`` never
waits: a model round trip is 1-5 s, the control tick is 20 ms.

Smoke-test one image before any sim/robot use:
    HF_TOKEN=hf_... python -m perception head.png        # or VLM_PROVIDER=openai VLM_MODEL=... OPENAI_API_KEY=...
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from camera import Frame
from vlm import DEFAULT_MODEL, VLMClient, RequestError, encode_jpeg, extract_json, image_part  # noqa: F401 (re-exported)
from vision import bearing, elevation
from worker import Worker

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


# --------------------------------------------------------------------------
# Perceivers
# --------------------------------------------------------------------------

class Perceiver(Worker[Frame, Percept]):
    """``offer(frame)`` records the newest frame; ``request()`` asks for it to
    be described (ignored while one is in flight, when it was already
    described, or within ``min_interval`` of the last request)."""

    def __init__(self, *, threaded: bool = True, min_interval: float = 0.0) -> None:
        super().__init__(threaded=threaded, min_interval=min_interval, name="perception")
        self._frame: Optional[Frame] = None
        self._done_seq: Optional[int] = None

    def describe(self, frame: Frame) -> Percept:
        raise NotImplementedError

    def process(self, frame: Frame) -> Percept:
        self._done_seq = frame.seq
        return self.describe(frame)

    def offer(self, frame: Frame) -> None:
        self._frame = frame

    def request(self, frame: Optional[Frame] = None) -> bool:      # type: ignore[override]
        frame = frame if frame is not None else self._frame
        if frame is None or frame.seq == self._done_seq:
            return False
        return super().request(frame)


class VisionQuery:
    """When a policy asks for a fresh view: whenever nothing is in flight, the
    latest percept does not already describe the current frame, and the last
    request is older than ``refresh`` seconds of policy time. ``poll`` every
    tick; it never blocks."""

    def __init__(self, refresh: float = 2.0) -> None:
        self.refresh = refresh
        self.reset()

    def reset(self) -> None:
        self._last = -math.inf
        self.sent = 0

    def poll(self, perceiver, obs, t: float) -> bool:
        if perceiver is None or obs.frame is None or perceiver.pending:
            return False
        if obs.percept is not None and obs.percept.frame_seq == obs.frame.seq:
            return False
        if t - self._last < self.refresh:
            return False
        if perceiver.request(obs.frame):
            self._last = t
            self.sent += 1
            return True
        return False


class VLMPerceiver(Perceiver):
    """The vision model over any OpenAI-compatible endpoint (see ``vlm.py``)."""

    def __init__(self, model: str = DEFAULT_MODEL, token: Optional[str] = None, *,
                 min_interval: float = 1.0, max_width: int = 640, client: Optional[VLMClient] = None,
                 **client_kw) -> None:
        super().__init__(min_interval=min_interval)
        self.client = client or VLMClient(model, token, **client_kw)
        self.max_width = max_width

    @classmethod
    def from_env(cls, model: Optional[str] = None, *, provider: Optional[str] = None, min_interval: float = 1.0,
                 max_width: int = 640, **client_kw) -> "VLMPerceiver":
        return cls(client=VLMClient.from_env(model, provider=provider, **client_kw), min_interval=min_interval,
                   max_width=max_width)

    @property
    def model(self) -> str:
        return self.client.model

    @property
    def json_mode(self) -> bool:
        return self.client.json_mode

    def build_request(self, frame: Frame) -> list[dict]:
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [image_part(frame.image, self.max_width),
                                             {"type": "text", "text": "Describe this frame."}]}]

    def describe(self, frame: Frame) -> Percept:
        text = self.client.complete(self.build_request(frame))
        return Percept.from_json(extract_json(text), frame, raw=text)


# --------------------------------------------------------------------------
# CLI plumbing
# --------------------------------------------------------------------------

VISION_MODES = ("auto", "off", "api")


def build_perceiver(args: argparse.Namespace, policy) -> Optional[Perceiver]:
    """Resolve --vision: api runs the model; auto = api iff the policy uses
    vision and a token is present (else a notice and no perceiver)."""
    mode = getattr(args, "vision", "auto")
    uses = getattr(policy, "uses_vision", False)
    if mode == "off" or (mode == "auto" and not uses):
        return None
    echo = (lambda s: print(s, end="", flush=True)) if getattr(args, "vision_echo", False) else None
    try:
        return VLMPerceiver.from_env(getattr(args, "vision_model", None), provider=getattr(args, "vision_provider", None),
                                     min_interval=getattr(args, "vision_interval", 1.0), on_text=echo)
    except RuntimeError as e:
        if mode == "api":
            raise
        print(f"notice: this policy uses vision but the model is not configured ({e}); running without percepts")
        return None


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m perception",
                                description="describe one image with the vision model (needs the provider's key, "
                                            "see vlm.py: VLM_PROVIDER / VLM_MODEL / VLM_BASE_URL)")
    p.add_argument("image", help="image file (png/jpg)")
    p.add_argument("--model", default=None, help="model id (default: $VLM_MODEL, or the provider's default)")
    p.add_argument("--provider", default=None, help="VLM provider (default: $VLM_PROVIDER or huggingface)")
    p.add_argument("--no-stream", action="store_true")
    args = p.parse_args(argv)
    import cv2
    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        p.error(f"could not read {args.image}")
    frame = Frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), 0.0, 1)
    try:
        per = VLMPerceiver.from_env(args.model, provider=args.provider, stream=not args.no_stream,
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
