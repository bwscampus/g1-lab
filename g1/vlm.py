"""The vision-language model client: Hugging Face Inference Providers or
OpenRouter over their OpenAI-compatible chat-completions endpoint, with image
input, stdlib urllib and streamed SSE; the model answers with one tool call.
Shared by the decider (choose a tool) and the demo keyframe selector.

Which model is asked, and how, is configs/model.json: ``provider``, a model id
per provider (``models``), ``reasoning_effort``, ``max_tokens``, ``temperature``,
and the agent loop's settings (``agent``: the decision budget, timeouts, retries,
image size, history windows). These may be tuned; configs/limits.json holds the
robot's and the run's hard limits.
The environment holds only what is private to this machine:

    HF_TOKEN            the Hugging Face key (a fine-grained token with "Make calls to Inference Providers")
    VLM_BASE_URL        another Hugging Face endpoint, e.g. a dedicated Inference Endpoint
                        (default: the router)
    OPENROUTER_API_KEY  the OpenRouter key

A ``.env`` file in the repo root (copy ``.env.example``) is read on first use;
exported variables win over it.
"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import numpy as np

from g1.core import limits

HF_BASE_URL = "https://router.huggingface.co/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# provider -> (endpoint, the variable holding its key, where a key comes from)
PROVIDERS = {
    "huggingface": (HF_BASE_URL, "HF_TOKEN", "Hugging Face token"),
    "openrouter": (OPENROUTER_BASE_URL, "OPENROUTER_API_KEY", "OpenRouter key"),
}
EFFORTS = ("none", "minimal", "low", "medium", "high", "default")
MODEL_PATH = limits.ROOT / "configs" / "model.json"
KEYS = ("version", "provider", "models", "routing", "reasoning_effort", "max_tokens", "temperature", "agent", "notes")
# the agent loop's settings (g1/agent/): whole numbers are counts, the rest seconds or sizes; all > 0 but jitter
AGENT = {"max_decisions": int, "step_timeout_s": float, "request_timeout_s": float, "retry_max": int,
         "retry_delay_base_s": float, "retry_delay_cap_s": float, "retry_jitter_s": float, "recovery_timeout_s": float,
         "image_width_px": int, "jpeg_quality": int, "live_image_window": int, "history_full_turns": int}


def load_model(path: Path | str | None = None) -> dict:
    """configs/model.json, checked: a typo or a missing key fails here, naming it."""
    p = Path(path) if path else MODEL_PATH
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"model file not found: {p}") from None
    except json.JSONDecodeError as e:
        raise ValueError(f"model file {p}: {e}") from None
    if not isinstance(m, dict) or m.get("version") != 1:
        raise ValueError(f"model file {p}: needs version 1")
    if sorted(m) != sorted(KEYS):
        raise ValueError(f"model file {p}: needs exactly {list(KEYS)}, has {sorted(m)}")
    if m["provider"] not in PROVIDERS:
        raise ValueError(f"model file {p}: provider must be one of {list(PROVIDERS)}, got {m['provider']!r}")
    models = m["models"]
    if not isinstance(models, dict) or sorted(models) != sorted(PROVIDERS) or not all(
            isinstance(v, str) and v.strip() for v in models.values()):
        raise ValueError(f"model file {p}: models needs a model id for each of {list(PROVIDERS)}")
    if not isinstance(m["routing"], dict):
        raise ValueError(f"model file {p}: routing must be an object (OpenRouter's provider preferences, {{}} for none)")
    if m["reasoning_effort"] not in EFFORTS:
        raise ValueError(f"model file {p}: reasoning_effort must be one of {list(EFFORTS)}, got {m['reasoning_effort']!r}")
    mt = m["max_tokens"]
    if mt is not None and (isinstance(mt, bool) or not isinstance(mt, int) or mt <= 0):
        raise ValueError(f"model file {p}: max_tokens must be a positive whole number, or null for no limit")
    t = m["temperature"]
    if isinstance(t, bool) or not isinstance(t, (int, float)) or t < 0:
        raise ValueError(f"model file {p}: temperature must be a number >= 0")
    a = m["agent"]
    if not isinstance(a, dict) or sorted(a) != sorted(AGENT):
        have = sorted(a) if isinstance(a, dict) else a
        raise ValueError(f"model file {p}: agent needs exactly {list(AGENT)}, has {have}")
    for name, kind in AGENT.items():
        v = a[name]
        ok = not isinstance(v, bool) and (isinstance(v, int) if kind is int else isinstance(v, (int, float)))
        if not ok or v < 0 or (v == 0 and name != "retry_jitter_s"):
            raise ValueError(f"model file {p}: agent.{name} must be a positive {'whole ' if kind is int else ''}number, "
                             f"got {v!r}")
    if a["retry_delay_base_s"] > a["retry_delay_cap_s"]:
        raise ValueError(f"model file {p}: agent.retry_delay_base_s must not exceed agent.retry_delay_cap_s")
    if a["jpeg_quality"] > 100:
        raise ValueError(f"model file {p}: agent.jpeg_quality must be 1-100")
    return m


MODEL = load_model()
PROVIDER = MODEL["provider"]
DEFAULT_MODEL = MODEL["models"][PROVIDER]
# The reply budget, reasoning included. A reasoning model thinks before it answers and the
# thinking counts: at 400 it never reached the answer.
MAX_TOKENS = MODEL["max_tokens"]          # None: not sent, the provider's own limit applies
# Sent as ``reasoning_effort`` ("default": not sent). A server that rejects it gets the request without it.
REASONING_EFFORT = MODEL["reasoning_effort"]
TEMPERATURE = MODEL["temperature"]
# OpenRouter only: which of the model's hosts serve it, sent as the request's ``provider`` object
# (e.g. {"order": ["baseten"], "allow_fallbacks": false}); without it OpenRouter picks a host per request
ROUTING = MODEL["routing"]
AGENT_SETTINGS = MODEL["agent"]           # the agent loop's budgets, timeouts, retries and image sizes
# Settings that moved from .env to configs/model.json: refused if still set, so none is silently ignored.
MOVED = {"VLM_MODEL": "models", "VLM_MAX_TOKENS": "max_tokens", "VLM_REASONING_EFFORT": "reasoning_effort"}
IMAGE_WIDTH = AGENT_SETTINGS["image_width_px"]
JPEG_QUALITY = AGENT_SETTINGS["jpeg_quality"]


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


def resolve(model: Optional[str] = None, env: Optional[dict] = None,
            provider: str = PROVIDER) -> tuple[str, str, str]:
    """(base_url, model, key) for ``provider`` from the environment (``.env`` in
    the repo root is read first)."""
    if env is None:
        load_dotenv()
    env = os.environ if env is None else env
    moved = [f"{var} (now {name})" for var, name in MOVED.items() if env.get(var)]
    if moved:
        raise RuntimeError(f"set in configs/model.json, not .env: {', '.join(moved)}; remove them from .env "
                           "or the environment")
    base_url, key_var, what = PROVIDERS[provider]
    key = env.get(key_var)
    if not key:
        raise RuntimeError(f"no {what}: set {key_var} (in .env or the environment), "
                           f"or change provider in configs/model.json (now {provider})")
    if provider == "huggingface":           # VLM_BASE_URL is another Hugging Face endpoint
        base_url = env.get("VLM_BASE_URL") or base_url
    return base_url.rstrip("/"), model or MODEL["models"][provider], key


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
        where = (f"max_tokens={max_tokens}" if max_tokens else "the provider's own limit (max_tokens not sent)")
        super().__init__(200, f"the reply was cut off at {where}{spent}; "
                              f"raise max_tokens in configs/model.json (now {max_tokens})")
        self.text = text


class NoToolCall(RequestError):
    """The model answered without exactly one tool call (text only, or several calls). The agent
    treats it as an invalid selection: it costs a decision and the model is told."""

    def __init__(self, text: str, n: int, names: list) -> None:
        what = "no tool call" if n == 0 else f"{n} tool calls ({', '.join(names)})"
        super().__init__(200, f"{what}; call exactly one tool" + (f" (text: {text[:200]!r})" if text else ""))
        self.text = text


# what a host says when it takes no tool calls (OpenRouter: no endpoint left after "Tool Compatibility")
TOOLS_REFUSED = ("tool_choice", "Tool Compatibility", "tools are not supported", "does not support tools",
                 "tool calling", "function calling")


class ToolsUnsupported(RequestError):
    """The model or host takes no tool calls: retrying will not help, choose another."""

    def __init__(self, status: int, body: str, provider: str = "", model: str = "") -> None:
        super().__init__(status, f"{model} on {provider} takes no tool calls, which the agent needs: choose a model "
                                 f"and host that support them in configs/model.json ({body[:300]})")


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
        raise classify(e.code, e.read().decode("utf-8", "replace")[:2000]) from None
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


@dataclass
class ToolCall:
    """The model's one tool call: its id (to answer it in the history), the tool's name and its
    arguments as the model wrote them (a JSON string), and any text it sent alongside."""
    id: str
    name: str
    arguments: str
    text: str = ""


class VLMClient:
    """One chat completion at a time against the provider's endpoint, answered with exactly
    one tool call (``tools``, ``tool_choice: required``, one call at a time): the model picks a
    tool and fills in its arguments, the form models are trained for. Only models and hosts
    that support tool calls are used (configs/model.json). ``transport`` is injectable for tests.

    ``stream_options`` and ``reasoning_effort`` are dropped for the rest of the session on a
    400 that names them. ``last_usage`` holds the provider's token counts for the last call,
    ``last_elapsed`` its wall time, ``last_chunks`` everything the server sent back for it, as
    sent: each streamed payload, and the status and body of any attempt it refused."""

    response_mode = "tool_calls"

    def __init__(self, model: str = DEFAULT_MODEL, token: Optional[str] = None, *,
                 base_url: str = HF_BASE_URL, provider: str = "huggingface", routing: Optional[dict] = None,
                 stream: bool = True, timeout: float = AGENT_SETTINGS["request_timeout_s"],
                 on_text: Optional[Callable[[str], None]] = None,
                 transport: Optional[Callable[[str, dict, dict, float], Iterator[str]]] = None) -> None:
        self.model = model
        self.token = token or ""
        self.base_url = base_url.rstrip("/")
        self.provider = provider
        self.routing = dict(routing) if routing else None
        self.stream = stream
        self.stream_options = True
        self.timeout = timeout
        self.on_text = on_text
        self.transport = transport or _sse_post
        self.last_usage: Optional[dict] = None
        self.last_elapsed = 0.0
        self.last_chunks: list = []
        self.calls = 0
        self.max_tokens = MAX_TOKENS
        self.reasoning_effort: Optional[str] = None if REASONING_EFFORT == "default" else REASONING_EFFORT

    @classmethod
    def from_env(cls, model: Optional[str] = None, provider: str = PROVIDER, **kw) -> "VLMClient":
        base_url, model, key = resolve(model, provider=provider)
        kw.setdefault("routing", ROUTING if provider == "openrouter" else None)
        return cls(model, key, base_url=base_url, provider=provider, **kw)

    def complete(self, messages: list[dict], *, tools: list[dict], max_tokens: Optional[int] = None,
                 temperature: float = TEMPERATURE) -> ToolCall:
        """The model's one tool call for ``messages``, choosing among ``tools`` (OpenAI function
        definitions). ``NoToolCall`` if it answered without exactly one; ``ToolsUnsupported`` if
        the host takes no tool calls."""
        body = {"model": self.model, "stream": self.stream, "temperature": temperature, "messages": messages,
                "tools": tools, "tool_choice": "required", "parallel_tool_calls": False}
        if (max_tokens or self.max_tokens) is not None:          # None: no limit of ours
            body["max_tokens"] = max_tokens or self.max_tokens
        if self.routing:
            body["provider"] = self.routing
        self.last_chunks = []
        while True:
            attempt = dict(body)               # a fresh body per attempt, so records see what was sent
            if self.stream and self.stream_options:
                attempt["stream_options"] = {"include_usage": True}
            if self.reasoning_effort is not None:
                attempt["reasoning_effort"] = self.reasoning_effort
            try:
                return self._send(attempt)
            except NoToolCall:
                raise
            except RequestError as e:
                self.last_chunks.append({"http_status": e.status, "body": e.body})
                if e.status in (400, 404) and any(k in e.body for k in TOOLS_REFUSED):
                    raise ToolsUnsupported(e.status, e.body, self.provider, self.model) from None
                if e.status == 400 and self.stream_options and "stream_options" in e.body:
                    self.stream_options = False
                    continue
                if e.status == 400 and self.reasoning_effort is not None and "reasoning_effort" in e.body:
                    self.reasoning_effort = None
                    continue
                # Some providers refuse reasoning_effort without naming it ("invalid request error"):
                # a 400 that names nothing above drops it once.
                if e.status == 400 and self.reasoning_effort is not None:
                    self.reasoning_effort = None
                    continue
                raise

    def _send(self, body: dict) -> ToolCall:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        parts: list[str] = []
        calls: dict[int, dict] = {}            # streamed tool-call fragments, joined by index
        finish = None
        self.last_usage = None
        self.calls += 1
        t0 = time.monotonic()
        try:
            for payload in self.transport(self.base_url + "/chat/completions", headers, body, self.timeout):
                if payload.strip() == "[DONE]":
                    break
                chunk = json.loads(payload)
                self.last_chunks.append(chunk)
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
                message = c.get("delta") or c.get("message") or {}
                text = message.get("content") or ""
                if text:
                    parts.append(text)
                    if self.on_text is not None:
                        self.on_text(text)
                for k, tc in enumerate(message.get("tool_calls") or []):
                    entry = calls.setdefault(tc.get("index", k), {"id": "", "name": "", "arguments": ""})
                    entry["id"] = entry["id"] or tc.get("id") or ""
                    fn = tc.get("function") or {}
                    entry["name"] += fn.get("name") or ""
                    args = fn.get("arguments")
                    entry["arguments"] += args if isinstance(args, str) else (json.dumps(args) if args else "")
        finally:
            self.last_elapsed = time.monotonic() - t0
        text = "".join(parts)
        if finish == "length":
            # A reasoning model thinks first and the thinking counts against max_tokens:
            # a small budget is spent before the answer starts, and the reply comes back empty.
            details = (self.last_usage or {}).get("completion_tokens_details") or {}
            raise TruncatedReply(body.get("max_tokens"), details.get("reasoning_tokens"), text)
        if len(calls) != 1:
            raise NoToolCall(text, len(calls), [c["name"] for c in calls.values()])
        call = next(iter(calls.values()))
        return ToolCall(call["id"] or f"call_{self.calls}", call["name"], call["arguments"] or "{}", text)
