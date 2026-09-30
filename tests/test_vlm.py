import json

import pytest

from pathlib import Path

from g1.core import limits

from g1.vlm import DEFAULT_MODEL, RequestError, VLMClient, load_dotenv, resolve


def test_resolve_defaults_overrides_and_errors():
    assert resolve(env={"HF_TOKEN": "hf_t"}) == ("https://router.huggingface.co/v1", DEFAULT_MODEL, "hf_t")
    env = {"HF_TOKEN": "hf_t", "VLM_MODEL": "a/b:deepinfra", "VLM_BASE_URL": "https://x.endpoints.huggingface.cloud/v1/"}
    assert resolve(env=env) == ("https://x.endpoints.huggingface.cloud/v1", "a/b:deepinfra", "hf_t")
    assert resolve("c/d", env=env)[1] == "c/d"                          # the argument beats VLM_MODEL
    with pytest.raises(RuntimeError, match="set HF_TOKEN"):
        resolve(env={})
    # only Hugging Face: other providers' variables are not read
    with pytest.raises(RuntimeError, match="set HF_TOKEN"):
        resolve(env={"VLM_PROVIDER": "openai", "OPENAI_API_KEY": "k", "VLM_API_KEY": "k"})


def sse(text):
    return iter([json.dumps({"choices": [{"delta": {"content": text}}]}), "[DONE]"])


def test_client_from_env_sends_to_hugging_face(monkeypatch):
    for var in ("HF_TOKEN", "VLM_BASE_URL", "VLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_t")
    monkeypatch.setenv("VLM_MODEL", "a/b")
    calls = []

    def transport(url, headers, body, timeout):
        calls.append((url, headers, body))
        return sse("{}")

    c = VLMClient.from_env(transport=transport)
    assert c.provider == "huggingface" and c.model == "a/b"
    assert c.complete([]) == "{}"
    url, headers, body = calls[0]
    assert url == "https://router.huggingface.co/v1/chat/completions" and headers["Authorization"] == "Bearer hf_t"
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
    assert resolve(env=env)[1] == DEFAULT_MODEL


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
    with pytest.raises(TruncatedReply, match=f"cut off at max_tokens={MAX_TOKENS}, 400 of them on reasoning; raise VLM_MAX_TOKENS"):
        c.complete([])
    assert calls[0]["max_tokens"] == MAX_TOKENS == limits.get("max_tokens")      # whatever the file says
    monkeypatch.setenv("HF_TOKEN", "hf_t")
    monkeypatch.setenv("VLM_MAX_TOKENS", "9000")
    assert VLMClient.from_env().max_tokens == 9000
    monkeypatch.setenv("VLM_MAX_TOKENS", "lots")
    with pytest.raises(RuntimeError, match="VLM_MAX_TOKENS"):
        VLMClient.from_env()


def test_dotenv_forgives_a_stray_quote(tmp_path):
    p = tmp_path / ".env"
    p.write_text('VLM_BASE_URL=https://router.huggingface.co/v1"\nHF_TOKEN="hf_q"\n')
    env = {}
    load_dotenv(p, env)
    assert resolve(env=env) == ("https://router.huggingface.co/v1", DEFAULT_MODEL, "hf_q")
