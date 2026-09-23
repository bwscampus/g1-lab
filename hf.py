"""Hugging Face Inference Providers client: OpenAI-compatible chat completions
with image input, stdlib urllib, streamed SSE, JSON mode with a fallback.
Shared by the perceiver (describe a frame) and the decider (choose a skill).
"""
from __future__ import annotations

import base64
import json
import os
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


class HFClient:
    """One chat completion at a time. ``transport`` is injectable for tests."""

    def __init__(self, model: str = DEFAULT_MODEL, token: Optional[str] = None, *,
                 base_url: str = HF_BASE_URL, stream: bool = True, json_mode: bool = True,
                 timeout: float = 30.0, on_text: Optional[Callable[[str], None]] = None,
                 transport: Optional[Callable[[str, dict, dict, float], Iterator[str]]] = None) -> None:
        self.model = model
        self.token = token or ""
        self.base_url = base_url.rstrip("/")
        self.stream = stream
        self.json_mode = json_mode
        self.timeout = timeout
        self.on_text = on_text
        self.transport = transport or _sse_post

    @classmethod
    def from_env(cls, model: Optional[str] = None, **kw) -> "HFClient":
        token = os.environ.get("HF_TOKEN") or os.environ.get("G1_VISION_API_KEY")
        if not token:
            raise RuntimeError("no Hugging Face token: set HF_TOKEN (or G1_VISION_API_KEY)")
        return cls(model or os.environ.get("G1_VISION_MODEL") or DEFAULT_MODEL, token, **kw)

    def complete(self, messages: list[dict], *, max_tokens: int = 400,
                 temperature: float = 0.0) -> str:
        body = {"model": self.model, "stream": self.stream, "temperature": temperature,
                "max_tokens": max_tokens, "messages": messages}
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            return self._send(body)
        except RequestError as e:
            # JSON mode is provider-dependent on the HF router: drop it and rely on the prompt.
            if self.json_mode and e.status == 400 and "response_format" in e.body:
                self.json_mode = False
                body.pop("response_format", None)
                return self._send(body)
            raise

    def _send(self, body: dict) -> str:
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
