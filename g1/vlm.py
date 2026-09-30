"""The vision-language model client: Hugging Face Inference Providers over
their OpenAI-compatible chat-completions endpoint, with image input, stdlib
urllib, streamed SSE and structured output with a step-down. Shared by the
decider (choose a tool) and the demo keyframe selector.

Configured by environment:

    HF_TOKEN       the key (a fine-grained token with "Make calls to Inference Providers")
    VLM_MODEL      the model id to query (default: DEFAULT_MODEL); a suffix such as
                   ":deepinfra" pins one of the router's providers
    VLM_BASE_URL   another Hugging Face endpoint, e.g. a dedicated Inference Endpoint
                   (default: the router)
    VLM_MAX_TOKENS the reply budget, reasoning included (default: MAX_TOKENS)

A ``.env`` file in the repo root (copy ``.env.example``) is read on first use;
exported variables win over it.

    HF_TOKEN=hf_... g1 decide frame.png --instruction "find the mug"
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np

from g1.core import limits

PROVIDER = "huggingface"
HF_BASE_URL = "https://router.huggingface.co/v1"
# Verified live on the HF router (2026-09): image input, structured output on its
# providers, a small active-parameter MoE that answers fast; ":deepinfra" etc. pins a provider.
DEFAULT_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
# The reply budget, reasoning included. A reasoning model thinks before it answers and the
# thinking counts: at 400 it never reached the answer. The numbers are in configs/limits.json.
MAX_TOKENS = int(limits.get("max_tokens"))
IMAGE_WIDTH = int(limits.get("image_width_px"))
JPEG_QUALITY = int(limits.get("jpeg_quality"))


DOTENV = Path(__file__).resolve().parents[1] / ".env"       # the repo root


def load_dotenv(path: Optional[Path] = None, env: Optional[dict] = None) -> dict:
    """Read ``KEY=value`` lines from ``.env`` (see ``.env.example``) into the
    environment, never overriding a variable that is already set."""
    env = os.environ if env is None else env
    loaded = {}
    try:
        lines = Path(DOTENV if path is None else path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return loaded
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip("\"'")     # quoted or not; a stray quote is a typo, not data
        if key and key not in env:
            env[key] = value
            loaded[key] = value
    return loaded


def resolve(model: Optional[str] = None, env: Optional[dict] = None) -> tuple[str, str, str]:
    """(base_url, model, key) from the environment (``.env`` in the repo root
    is read first)."""
    if env is None:
        load_dotenv()
    env = os.environ if env is None else env
    key = env.get("HF_TOKEN")
    if not key:
        raise RuntimeError("no Hugging Face token: set HF_TOKEN (in .env or the environment)")
    base_url = (env.get("VLM_BASE_URL") or HF_BASE_URL).rstrip("/")
    return base_url, model or env.get("VLM_MODEL") or DEFAULT_MODEL, key


class RequestError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


class Overloaded(RequestError):
    """The provider is busy (429/503/529 or says so): worth retrying with backoff."""


class QuotaExceeded(RequestError):
    """Credits or quota are gone (402, or the body says so): retrying will not help."""


class TruncatedReply(RequestError):
    """The reply hit ``max_tokens`` before it finished (``finish_reason: length``)."""

    def __init__(self, max_tokens, reasoning_tokens, text: str = "") -> None:
        spent = f", {reasoning_tokens} of them on reasoning" if reasoning_tokens else ""
        super().__init__(200, f"the reply was cut off at max_tokens={max_tokens}{spent}; "
                              f"raise VLM_MAX_TOKENS (now {max_tokens})")
        self.text = text


def classify(status: int, body: str) -> RequestError:
    """The right error class for an HTTP failure, judged like their overload
    detection: by status and by what the body says."""
    text = body.lower()
    if status == 402 or "quota" in text or "credits" in text or "exceeded your monthly" in text:
        return QuotaExceeded(status, body)
    if status in (429, 502, 503, 529) or "overloaded" in text or "rate limit" in text or "try again" in text:
        return Overloaded(status, body)
    return RequestError(status, body)


def encode_jpeg(image_rgb: np.ndarray, max_width: int = IMAGE_WIDTH, quality: int = JPEG_QUALITY) -> bytes:
    """JPEG bytes of an RGB frame, downscaled to ``max_width`` (``images.py``)."""
    from g1.core import images
    return images.encode_jpeg(image_rgb, max_width, quality)


def image_part(image_rgb: np.ndarray, max_width: int = IMAGE_WIDTH) -> dict:
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


class VLMClient:
    """One chat completion at a time against the Hugging Face endpoint.
    ``transport`` is injectable for tests.

    Structured output is asked for in three steps, each dropped for the rest
    of the session on a 400 that names it: a ``json_schema`` response format
    when the caller passes a schema, then ``json_object``, then nothing (the
    prompt alone); ``stream_options`` is dropped the same way for servers that
    reject it. ``response_mode`` says which one the last call used; the caller
    validates the reply itself either way. ``last_usage`` holds the provider's
    token counts for the last call, ``last_elapsed`` its wall time."""

    def __init__(self, model: str = DEFAULT_MODEL, token: Optional[str] = None, *,
                 base_url: str = HF_BASE_URL, stream: bool = True,
                 json_mode: bool = True, json_schema: bool = True,
                 timeout: float = limits.get("request_timeout_s"),
                 on_text: Optional[Callable[[str], None]] = None,
                 transport: Optional[Callable[[str, dict, dict, float], Iterator[str]]] = None) -> None:
        self.model = model
        self.token = token or ""
        self.base_url = base_url.rstrip("/")
        self.provider = PROVIDER
        self.stream = stream
        self.json_mode = json_mode
        self.json_schema = json_schema
        self.stream_options = True
        self.timeout = timeout
        self.on_text = on_text
        self.transport = transport or _sse_post
        self.response_mode = "none"
        self.last_usage: Optional[dict] = None
        self.last_elapsed = 0.0
        self.calls = 0
        self.max_tokens = MAX_TOKENS

    @classmethod
    def from_env(cls, model: Optional[str] = None, **kw) -> "VLMClient":
        base_url, model, key = resolve(model)
        client = cls(model, key, base_url=base_url, **kw)
        raw = os.environ.get("VLM_MAX_TOKENS")
        if raw:
            try:
                client.max_tokens = max(1, int(raw))
            except ValueError:
                raise RuntimeError(f"VLM_MAX_TOKENS must be a whole number, got {raw!r}") from None
        return client

    def complete(self, messages: list[dict], *, max_tokens: Optional[int] = None,
                 temperature: float = 0.0, schema: Optional[dict] = None) -> str:
        body = {"model": self.model, "stream": self.stream, "temperature": temperature,
                "max_tokens": max_tokens or self.max_tokens, "messages": messages}
        while True:
            attempt = dict(body)               # a fresh body per attempt, so records see what was sent
            if self.stream and self.stream_options:
                attempt["stream_options"] = {"include_usage": True}
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
                # Structured output and stream_options are server-dependent: step down and retry.
                if e.status == 400 and self.stream_options and "stream_options" in e.body:
                    self.stream_options = False
                    continue
                mentions = e.status == 400 and any(k in e.body for k in ("response_format", "json_schema", "schema"))
                if mentions and self.response_mode == "json_schema":
                    self.json_schema = False
                    continue
                if mentions and self.response_mode == "json_object":
                    self.json_mode = False
                    continue
                raise

    def _send(self, body: dict) -> str:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        parts: list[str] = []
        finish = None
        self.last_usage = None
        self.calls += 1
        t0 = time.monotonic()
        try:
            for payload in self.transport(self.base_url + "/chat/completions", headers, body, self.timeout):
                if payload.strip() == "[DONE]":
                    break
                chunk = json.loads(payload)
                if isinstance(chunk, dict) and chunk.get("error"):
                    # some servers answer HTTP 200 and put the failure in the stream
                    err = chunk["error"] if isinstance(chunk["error"], dict) else {"message": str(chunk["error"])}
                    code = err.get("code")
                    status = code if isinstance(code, int) and 400 <= code < 600 else 500
                    raise classify(status, f"{err.get('message', '')} ({err.get('type') or code})")
                if isinstance(chunk.get("usage"), dict):
                    self.last_usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                c = choices[0]
                finish = c.get("finish_reason") or finish
                delta = (c.get("delta") or {}).get("content") or (c.get("message") or {}).get("content") or ""
                if delta:
                    parts.append(delta)
                    if self.on_text is not None:
                        self.on_text(delta)
        finally:
            self.last_elapsed = time.monotonic() - t0
        if finish == "length":
            # A reasoning model thinks first and the thinking counts against max_tokens:
            # a small budget is spent before the answer starts, and the reply comes back empty.
            details = (self.last_usage or {}).get("completion_tokens_details") or {}
            raise TruncatedReply(body.get("max_tokens"), details.get("reasoning_tokens"), "".join(parts))
        return "".join(parts)
