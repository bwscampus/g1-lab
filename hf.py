"""Hugging Face Inference Providers client: OpenAI-compatible chat completions
with image input, stdlib urllib, streamed SSE, JSON mode with a fallback.
Shared by the perceiver (describe a frame) and the decider (choose a skill).
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from typing import Callable, Iterator, Optional

import numpy as np

HF_BASE_URL = "https://router.huggingface.co/v1"
# Verified live on the HF router (2026-09): image input, structured output on
# its providers, small active-parameter MoE so it answers fast. Override with
# --vision-model / $G1_VISION_MODEL; ":deepinfra" etc. pins a provider.
DEFAULT_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"


class RequestError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


class Overloaded(RequestError):
    """The provider is busy (429/503/529 or says so): worth retrying with backoff."""


class QuotaExceeded(RequestError):
    """Credits or quota are gone (402, or the body says so): retrying will not help."""


def classify(status: int, body: str) -> RequestError:
    """The right error class for an HTTP failure, judged like their overload
    detection: by status and by what the body says."""
    text = body.lower()
    if status == 402 or "quota" in text or "credits" in text or "exceeded your monthly" in text:
        return QuotaExceeded(status, body)
    if status in (429, 502, 503, 529) or "overloaded" in text or "rate limit" in text or "try again" in text:
        return Overloaded(status, body)
    return RequestError(status, body)


def encode_jpeg(image_rgb: np.ndarray, max_width: int = 640, quality: int = 80) -> bytes:
    """JPEG bytes of an RGB frame, downscaled to ``max_width``. OpenCV thinks in
    BGR, so the conversion happens here and nowhere else."""
    import cv2
    h, w = image_rgb.shape[:2]
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if w > max_width:
        bgr = cv2.resize(bgr, (max_width, int(round(h * max_width / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def image_part(image_rgb: np.ndarray, max_width: int = 640) -> dict:
    """An ``image_url`` content block carrying the frame as a data URL."""
    data = base64.b64encode(encode_jpeg(image_rgb, max_width)).decode("ascii")
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + data}}


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


def _sse_post(url: str, headers: dict, body: dict, timeout: float) -> Iterator[str]:
    """POST JSON; yield each SSE ``data:`` payload, or the whole body when the
    response is not a stream."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise classify(e.code, e.read().decode("utf-8", "replace")[:300]) from None
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


class HFClient:
    """One chat completion at a time. ``transport`` is injectable for tests.

    Structured output is asked for in three steps, each dropped for the rest
    of the session on a 400 that names it: a ``json_schema`` response format
    when the caller passes a schema, then ``json_object``, then nothing (the
    prompt alone). ``response_mode`` says which one the last call used; the
    caller validates the reply itself either way. ``last_usage`` holds the
    provider's token counts for the last call, ``last_elapsed`` its wall time."""

    def __init__(self, model: str = DEFAULT_MODEL, token: Optional[str] = None, *,
                 base_url: str = HF_BASE_URL, stream: bool = True, json_mode: bool = True,
                 json_schema: bool = True, timeout: float = 30.0,
                 on_text: Optional[Callable[[str], None]] = None,
                 transport: Optional[Callable[[str, dict, dict, float], Iterator[str]]] = None) -> None:
        self.model = model
        self.token = token or ""
        self.base_url = base_url.rstrip("/")
        self.stream = stream
        self.json_mode = json_mode
        self.json_schema = json_schema
        self.timeout = timeout
        self.on_text = on_text
        self.transport = transport or _sse_post
        self.response_mode = "none"
        self.last_usage: Optional[dict] = None
        self.last_elapsed = 0.0
        self.calls = 0

    @classmethod
    def from_env(cls, model: Optional[str] = None, **kw) -> "HFClient":
        token = os.environ.get("HF_TOKEN") or os.environ.get("G1_VISION_API_KEY")
        if not token:
            raise RuntimeError("no Hugging Face token: set HF_TOKEN (or G1_VISION_API_KEY)")
        return cls(model or os.environ.get("G1_VISION_MODEL") or DEFAULT_MODEL, token, **kw)

    def complete(self, messages: list[dict], *, max_tokens: int = 400,
                 temperature: float = 0.0, schema: Optional[dict] = None) -> str:
        body = {"model": self.model, "stream": self.stream, "temperature": temperature,
                "max_tokens": max_tokens, "messages": messages}
        if self.stream:
            body["stream_options"] = {"include_usage": True}
        while True:
            attempt = dict(body)               # a fresh body per attempt, so records see what was sent
            if schema is not None and self.json_schema:
                self.response_mode = "json_schema"
                attempt["response_format"] = {"type": "json_schema",
                                              "json_schema": {"name": "selection", "schema": schema, "strict": True}}
            elif self.json_mode:
                self.response_mode = "json_object"
                attempt["response_format"] = {"type": "json_object"}
            else:
                self.response_mode = "none"
            try:
                return self._send(attempt)
            except RequestError as e:
                # Structured output is provider-dependent on the HF router: step down and retry.
                mentions = e.status == 400 and any(k in e.body for k in ("response_format", "json_schema", "schema"))
                if mentions and self.response_mode == "json_schema":
                    self.json_schema = False
                    continue
                if mentions and self.response_mode == "json_object":
                    self.json_mode = False
                    continue
                raise

    def _send(self, body: dict) -> str:
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        parts: list[str] = []
        self.last_usage = None
        self.calls += 1
        t0 = time.monotonic()
        try:
            for payload in self.transport(self.base_url + "/chat/completions", headers, body, self.timeout):
                if payload.strip() == "[DONE]":
                    break
                chunk = json.loads(payload)
                if isinstance(chunk.get("usage"), dict):
                    self.last_usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                c = choices[0]
                delta = (c.get("delta") or {}).get("content") or (c.get("message") or {}).get("content") or ""
                if delta:
                    parts.append(delta)
                    if self.on_text is not None:
                        self.on_text(delta)
        finally:
            self.last_elapsed = time.monotonic() - t0
        return "".join(parts)
