import json

import pytest

from pathlib import Path

from vlm import DEFAULT_MODEL, PROVIDERS, RequestError, VLMClient, load_dotenv, resolve


def test_every_provider_resolves_its_endpoint_and_key():
    for name, p in PROVIDERS.items():
        env = {"VLM_PROVIDER": name, "VLM_MODEL": "m"}
        if p.key_var:
            env[p.key_var] = "secret-" + name
        if p.base_url is None:
            env["VLM_BASE_URL"] = "http://box:8000/v1/"
        prov, base, model, key = resolve(env=env)
        assert prov.name == name and model == "m" and not base.endswith("/")
        assert base == (p.base_url or "http://box:8000/v1")
        assert key == ("secret-" + name if p.key_var else None)


def test_defaults_overrides_and_errors():
    prov, base, model, key = resolve(env={"HF_TOKEN": "hf_t"})           # today's setup still works
    assert prov.name == "huggingface" and model == DEFAULT_MODEL and key == "hf_t"
    assert base == "https://router.huggingface.co/v1"
    _, base, model, key = resolve("c/d", env={"VLM_PROVIDER": "OpenAI", "VLM_BASE_URL": "https://proxy/v1",
                                              "VLM_API_KEY": "override", "OPENAI_API_KEY": "real"})
    assert base == "https://proxy/v1" and key == "override" and model == "c/d"          # arg beats VLM_MODEL
    assert resolve(env={"VLM_PROVIDER": "openai", "VLM_MODEL": "gpt-x", "OPENAI_API_KEY": "k"})[2] == "gpt-x"
    assert resolve(provider="ollama", env={"VLM_MODEL": "qwen2.5vl"})[3] is None        # keyless local server
    with pytest.raises(RuntimeError, match="set OPENAI_API_KEY"):
        resolve(env={"VLM_PROVIDER": "openai", "VLM_MODEL": "g"})
    with pytest.raises(RuntimeError, match="set VLM_MODEL"):
        resolve(env={"VLM_PROVIDER": "openai", "OPENAI_API_KEY": "k"})
    with pytest.raises(RuntimeError, match="needs VLM_BASE_URL"):
        resolve(env={"VLM_PROVIDER": "custom", "VLM_MODEL": "m"})
    with pytest.raises(RuntimeError, match="unknown VLM_PROVIDER"):
        resolve(env={"VLM_PROVIDER": "nope"})
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        resolve(env={})


def sse(text):
    return iter([json.dumps({"choices": [{"delta": {"content": text}}]}), "[DONE]"])


def test_client_from_env_sends_to_the_provider(monkeypatch):
    for var in ("HF_TOKEN", "VLM_API_KEY", "VLM_BASE_URL", "VLM_MODEL", "VLM_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("VLM_PROVIDER", "groq")
    monkeypatch.setenv("VLM_MODEL", "llama-vision")
    monkeypatch.setenv("GROQ_API_KEY", "gsk")
    calls = []

    def transport(url, headers, body, timeout):
        calls.append((url, headers, body))
        return sse("{}")

    c = VLMClient.from_env(transport=transport)
    assert c.provider == "groq" and c.model == "llama-vision"
    assert c.complete([]) == "{}"
    url, headers, body = calls[0]
    assert url == "https://api.groq.com/openai/v1/chat/completions" and headers["Authorization"] == "Bearer gsk"
    assert body["model"] == "llama-vision"
    monkeypatch.setenv("VLM_PROVIDER", "ollama")
    c = VLMClient.from_env("qwen", transport=transport)
    c.complete([])
    assert "Authorization" not in calls[1][1] and calls[1][0].startswith("http://localhost:11434/v1")


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


def test_dotenv_fills_gaps_only(tmp_path):
    p = tmp_path / ".env"
    p.write_text('# comment\nVLM_PROVIDER=openai\nexport VLM_MODEL="gpt-4o-mini"\nOPENAI_API_KEY=sk-1\n\nbroken line\n')
    env = {"VLM_MODEL": "already"}
    loaded = load_dotenv(p, env)
    assert loaded == {"VLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-1"} and env["VLM_MODEL"] == "already"
    assert load_dotenv(tmp_path / "missing", {}) == {}
    example = Path(__file__).resolve().parents[1] / ".env.example"
    env = {}
    load_dotenv(example, env)
    assert env["VLM_PROVIDER"] == "huggingface" and env["HF_TOKEN"] == "hf_..."      # the example is loadable
    assert resolve(env=env)[0].name == "huggingface"
