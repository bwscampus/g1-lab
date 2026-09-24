"""The vision-language model client: any OpenAI-compatible chat-completions
API, with image input, stdlib urllib, streamed SSE and structured output with
a step-down. Shared by the perceiver (describe a frame), the decider (choose a
skill) and the demo keyframe selector.

The endpoint is chosen by environment:

    VLM_PROVIDER   one of PROVIDERS (default huggingface); sets the base URL and
                   which key variable is read
    VLM_MODEL      the model id to query (required unless the provider has a default)
    VLM_BASE_URL   overrides the provider's base URL (required for VLM_PROVIDER=custom)
    VLM_API_KEY    overrides the provider's key variable (HF_TOKEN, OPENAI_API_KEY, ...)

A ``.env`` file in the repo root (copy ``.env.example``) is read on first use;
exported variables win over it.

    VLM_PROVIDER=openai VLM_MODEL=gpt-4o-mini OPENAI_API_KEY=... python -m decider frame.png --goal ...
    VLM_PROVIDER=ollama VLM_MODEL=qwen2.5vl python -m perception head.png
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: Optional[str]           # None: VLM_BASE_URL is required
    key_var: Optional[str]            # None: no key needed (a local server)
    default_model: Optional[str] = None


PROVIDERS = {p.name: p for p in (
    # Verified live on the HF router (2026-09): image input, structured output on its
    # providers, a small active-parameter MoE that answers fast; ":deepinfra" etc. pins a provider.
    Provider("huggingface", "https://router.huggingface.co/v1", "HF_TOKEN", "Qwen/Qwen3-VL-30B-A3B-Instruct"),
    Provider("openai", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    Provider("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    Provider("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    Provider("together", "https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    Provider("deepinfra", "https://api.deepinfra.com/v1/openai", "DEEPINFRA_API_KEY"),
    Provider("mistral", "https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    Provider("xai", "https://api.x.ai/v1", "XAI_API_KEY"),
    Provider("gemini", "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    Provider("ollama", "http://localhost:11434/v1", None),
    Provider("custom", None, None),
)}
DEFAULT_PROVIDER = "huggingface"
DEFAULT_MODEL = PROVIDERS[DEFAULT_PROVIDER].default_model
HF_BASE_URL = PROVIDERS[DEFAULT_PROVIDER].base_url


DOTENV = Path(__file__).parent / ".env"


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
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in env:
            env[key] = value
            loaded[key] = value
    return loaded


def resolve(model: Optional[str] = None, provider: Optional[str] = None,
            env: Optional[dict] = None) -> tuple[Provider, str, str, Optional[str]]:
    """(provider, base_url, model, key) from the environment (``.env`` in the
    repo root is read first); every failure names the variable to set."""
    if env is None:
        load_dotenv()
    env = os.environ if env is None else env
    name = (provider or env.get("VLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    if name not in PROVIDERS:
        raise RuntimeError(f"unknown VLM_PROVIDER {name!r}; one of {', '.join(PROVIDERS)}")
    p = PROVIDERS[name]
    base_url = env.get("VLM_BASE_URL") or p.base_url
    if not base_url:
        raise RuntimeError(f"VLM_PROVIDER={name} needs VLM_BASE_URL (an OpenAI-compatible /v1 endpoint)")
    model = model or env.get("VLM_MODEL") or p.default_model
    if not model:
        raise RuntimeError(f"no model for provider {name}: set VLM_MODEL (or --vision-model)")
    key = env.get("VLM_API_KEY")
    if not key and p.key_var:
        key = env.get(p.key_var)
        if not key:
            raise RuntimeError(f"no API key for provider {name}: set {p.key_var} (or VLM_API_KEY)")
    return p, base_url.rstrip("/"), model, key or None


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


class VLMClient:
    """One chat completion at a time against an OpenAI-compatible endpoint.
    ``transport`` is injectable for tests.

    Structured output is asked for in three steps, each dropped for the rest
    of the session on a 400 that names it: a ``json_schema`` response format
    when the caller passes a schema, then ``json_object``, then nothing (the
    prompt alone); ``stream_options`` is dropped the same way for servers that
    reject it. ``response_mode`` says which one the last call used; the caller
    validates the reply itself either way. ``last_usage`` holds the provider's
    token counts for the last call, ``last_elapsed`` its wall time."""

    def __init__(self, model: str = DEFAULT_MODEL, token: Optional[str] = None, *,
                 base_url: str = HF_BASE_URL, provider: str = DEFAULT_PROVIDER, stream: bool = True,
                 json_mode: bool = True, json_schema: bool = True, timeout: float = 30.0,
                 on_text: Optional[Callable[[str], None]] = None,
                 transport: Optional[Callable[[str, dict, dict, float], Iterator[str]]] = None) -> None:
        self.model = model
        self.token = token or ""
        self.base_url = base_url.rstrip("/")
        self.provider = provider
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

    @classmethod
    def from_env(cls, model: Optional[str] = None, *, provider: Optional[str] = None, **kw) -> "VLMClient":
        p, base_url, model, key = resolve(model, provider)
        return cls(model, key, base_url=base_url, provider=p.name, **kw)

    def complete(self, messages: list[dict], *, max_tokens: int = 400,
                 temperature: float = 0.0, schema: Optional[dict] = None) -> str:
        body = {"model": self.model, "stream": self.stream, "temperature": temperature,
                "max_tokens": max_tokens, "messages": messages}
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
