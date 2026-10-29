import json

import pytest

from pathlib import Path

from g1.core import limits

from g1 import vlm
from g1.vlm import MODEL, RequestError, VLMClient, load_dotenv, load_model, resolve

HF_MODEL = MODEL["models"]["huggingface"]


def test_resolve_defaults_overrides_and_errors():
    hf = dict(provider="huggingface")
    assert resolve(env={"HF_TOKEN": "hf_t"}, **hf) == ("https://router.huggingface.co/v1", HF_MODEL, "hf_t")
    assert vlm.DEFAULT_MODEL == MODEL["models"][vlm.PROVIDER]           # the model is in model.json
    env = {"HF_TOKEN": "hf_t", "VLM_BASE_URL": "https://x.endpoints.huggingface.cloud/v1/"}
    assert resolve(env=env, **hf) == ("https://x.endpoints.huggingface.cloud/v1", HF_MODEL, "hf_t")
    assert resolve("c/d", env=env, **hf)[1] == "c/d"                    # the argument beats the file
    # the settings that moved to model.json are refused in .env, never silently used
    for var, name in (("VLM_MODEL", "models"), ("VLM_MAX_TOKENS", "max_tokens"),
                      ("VLM_REASONING_EFFORT", "reasoning_effort")):
        with pytest.raises(RuntimeError, match=f"configs/model.json, not .env: {var} \\(now {name}\\)"):
            resolve(env={"HF_TOKEN": "hf_t", var: "x"}, **hf)
    with pytest.raises(RuntimeError, match="set HF_TOKEN"):
        resolve(env={}, **hf)
    # the provider is in model.json, never a variable; other providers' keys are not read
    with pytest.raises(RuntimeError, match="set HF_TOKEN"):
        resolve(env={"VLM_PROVIDER": "openai", "OPENAI_API_KEY": "k", "VLM_API_KEY": "k"}, provider="huggingface")


def test_resolve_openrouter():
    env = {"OPENROUTER_API_KEY": "sk-or-t", "HF_TOKEN": "hf_t", "VLM_BASE_URL": "https://x.endpoints.huggingface.cloud/v1"}
    # its own endpoint, key and model; VLM_BASE_URL is a Hugging Face endpoint, not read here
    assert resolve(env=env, provider="openrouter") == ("https://openrouter.ai/api/v1",
                                                       MODEL["models"]["openrouter"], "sk-or-t")
    with pytest.raises(RuntimeError, match="set OPENROUTER_API_KEY.*provider in configs/model.json"):
        resolve(env={"HF_TOKEN": "hf_t"}, provider="openrouter")


def sse(text):
    return iter([json.dumps({"choices": [{"delta": {"content": text}}]}), "[DONE]"])


@pytest.mark.parametrize("provider, var, key, url", [
    ("huggingface", "HF_TOKEN", "hf_t", "https://router.huggingface.co/v1/chat/completions"),
    ("openrouter", "OPENROUTER_API_KEY", "sk-or-t", "https://openrouter.ai/api/v1/chat/completions"),
])
def test_client_from_env_sends_to_the_provider(monkeypatch, provider, var, key, url):
    for v in ("HF_TOKEN", "VLM_BASE_URL", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv(var, key)
    calls = []

    def transport(url, headers, body, timeout):
        calls.append((url, headers, body))
        return sse("{}")

    c = VLMClient.from_env("a/b", provider=provider, transport=transport)
    assert c.provider == provider and c.model == "a/b"
    assert c.complete([]) == "{}"
    sent_url, headers, body = calls[0]
    assert sent_url == url and headers["Authorization"] == f"Bearer {key}"
    assert body["model"] == "a/b"


def test_stream_options_step_down():
    replies = [RequestError(400, "unknown field stream_options"), "ok", "ok"]
    calls = []

    def transport(url, headers, body, timeout):
        calls.append(body)
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return sse(r)

    c = VLMClient("m", "k", transport=transport)
    assert c.complete([]) == "ok" and c.stream_options is False
    assert "stream_options" in calls[0] and "stream_options" not in calls[1]
    assert c.complete([]) == "ok" and "stream_options" not in calls[2]


def test_an_error_inside_the_stream_is_raised_not_swallowed():
    from g1.vlm import Overloaded, QuotaExceeded

    def failing(error):
        return VLMClient("m", "k", transport=lambda *a: iter([json.dumps({"error": error})]))

    with pytest.raises(QuotaExceeded, match="insufficient credits"):
        failing({"message": "insufficient credits", "type": "insufficient_quota", "code": 402}).complete([])
    with pytest.raises(Overloaded):
        failing({"message": "model is overloaded", "code": "overloaded"}).complete([])
    with pytest.raises(RequestError, match="boom"):
        failing("boom").complete([])


def test_dotenv_fills_gaps_only(tmp_path):
    p = tmp_path / ".env"
    p.write_text('# comment\nHF_TOKEN=hf_1\nexport VLM_MODEL="a/b"\nUNITREE_AES_128_KEY=\'k\'\n\nbroken line\n')
    env = {"VLM_MODEL": "already"}
    loaded = load_dotenv(p, env)
    assert loaded == {"HF_TOKEN": "hf_1", "UNITREE_AES_128_KEY": "k"} and env["VLM_MODEL"] == "already"
    assert load_dotenv(tmp_path / "missing", {}) == {}
    example = Path(__file__).resolve().parents[1] / ".env.example"
    env = {}
    load_dotenv(example, env)
    assert env == {"HF_TOKEN": "hf_..."}                                  # the example is loadable
    assert resolve(env=env, provider="huggingface")[1] == HF_MODEL


def test_a_reply_cut_off_by_the_token_budget_says_so(monkeypatch):
    from g1.vlm import MAX_TOKENS, TruncatedReply
    chunks = [json.dumps({"choices": [{"delta": {"content": ""}, "finish_reason": "length"}]}),
              json.dumps({"choices": [], "usage": {"completion_tokens": 400,
                                                   "completion_tokens_details": {"reasoning_tokens": 400}}}), "[DONE]"]
    calls = []

    def transport(url, headers, body, timeout):
        calls.append(body)
        return iter(chunks)

    c = VLMClient("m", "k", transport=transport)
    with pytest.raises(TruncatedReply, match=f"cut off at max_tokens={MAX_TOKENS}, 400 of them on reasoning; "
                                             "raise max_tokens in configs/model.json"):
        c.complete([])
    assert calls[0]["max_tokens"] == MAX_TOKENS == MODEL["max_tokens"]          # whatever the file says


def test_dotenv_forgives_a_stray_quote(tmp_path):
    p = tmp_path / ".env"
    p.write_text('VLM_BASE_URL=https://router.huggingface.co/v1"\nHF_TOKEN="hf_q"\n')
    env = {}
    load_dotenv(p, env)
    assert resolve(env=env, provider="huggingface") == ("https://router.huggingface.co/v1", HF_MODEL, "hf_q")


def test_reasoning_effort_is_sent_and_dropped_when_refused():
    effort = MODEL["reasoning_effort"]
    replies = [RequestError(400, "unknown field reasoning_effort"), "ok", "ok"]
    calls = []

    def transport(url, headers, body, timeout):
        calls.append(body)
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return sse(r)

    c = VLMClient("m", "k", transport=transport)
    assert effort != "default"                      # the test needs the field sent; "default" never sends it
    assert c.complete([]) == "ok" and c.reasoning_effort is None
    assert calls[0]["reasoning_effort"] == effort and "reasoning_effort" not in calls[1]
    assert c.complete([]) == "ok" and "reasoning_effort" not in calls[2]
    # a provider that refuses it without naming it (HF router, novita, 2026-10-21)
    calls.clear()
    refusal = '{"message":"invalid request error","type":"invalid_request_error"}'
    replies[:] = [RequestError(400, refusal), "ok"]
    c = VLMClient("m", "k", transport=transport)
    assert c.complete([]) == "ok" and c.reasoning_effort is None
    # last_chunks keeps everything the server sent back for the call: the refusal, then the stream
    assert c.last_chunks == [{"http_status": 400, "body": refusal}, {"choices": [{"delta": {"content": "ok"}}]}]
    assert calls[0]["reasoning_effort"] == effort and "reasoning_effort" not in calls[1]
    # a 400 with nothing left to drop is raised
    replies[:] = [RequestError(400, "bad")]
    with pytest.raises(RequestError):
        c.json_mode = False
        c.complete([])


def test_a_bad_model_file_is_refused_with_the_name(tmp_path):
    base = json.loads(vlm.MODEL_PATH.read_text())
    assert load_model() == base and base["provider"] in vlm.PROVIDERS

    def broken(mutate, match):
        d = json.loads(json.dumps(base))
        mutate(d)
        p = tmp_path / "m.json"
        p.write_text(json.dumps(d))
        with pytest.raises(ValueError, match=match):
            load_model(p)

    broken(lambda d: d.update(provider="openai"), "provider must be one of")
    broken(lambda d: d["models"].pop("openrouter"), "a model id for each of")
    broken(lambda d: d["models"].update(huggingface=""), "a model id for each of")
    broken(lambda d: d.update(reasoning_effort="a little"), "reasoning_effort must be one of")
    broken(lambda d: d.update(max_tokens=0), "max_tokens must be a positive")
    broken(lambda d: d.update(temperature=-1), "temperature must be")
    broken(lambda d: d.update(max_tokns=1), "needs exactly")
    broken(lambda d: d.update(routing=["baseten"]), "routing must be an object")
    broken(lambda d: d["agent"].pop("step_timeout_s"), "agent needs exactly")
    broken(lambda d: d["agent"].update(max_decisions=0), "agent.max_decisions must be a positive whole number")
    broken(lambda d: d["agent"].update(max_decisions=2.5), "agent.max_decisions must be a positive whole number")
    broken(lambda d: d["agent"].update(retry_delay_base_s=10.0), "must not exceed agent.retry_delay_cap_s")
    broken(lambda d: d.update(version=2), "version 1")
    with pytest.raises(ValueError, match="not found"):
        load_model(tmp_path / "none.json")


def test_limits_hold_no_model_setting():
    # the robot's and the run's limits only; which model, and how it is asked, is model.json
    names = set(limits.NAMES)
    assert not names & {"vlm_provider", "vlm_model", "openrouter_model", "reasoning_effort", "max_tokens",
                        "vlm_temperature", *vlm.AGENT}
    assert sorted(vlm.AGENT_SETTINGS) == sorted(vlm.AGENT)


def test_routing_is_sent_to_openrouter_only(monkeypatch):
    for v in ("HF_TOKEN", "VLM_BASE_URL", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-t")
    monkeypatch.setenv("HF_TOKEN", "hf_t")
    monkeypatch.setattr(vlm, "ROUTING", {"order": ["baseten"], "allow_fallbacks": False})
    calls = []

    def transport(url, headers, body, timeout):
        calls.append(body)
        return sse("{}")
    VLMClient.from_env("a/b", provider="openrouter", transport=transport).complete([])
    VLMClient.from_env("a/b", provider="huggingface", transport=transport).complete([])
    assert calls[0]["provider"] == {"order": ["baseten"], "allow_fallbacks": False} and "provider" not in calls[1]
