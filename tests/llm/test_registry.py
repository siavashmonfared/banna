"""Registry smoke tests: factory dispatch, config env resolution, toggling."""
from __future__ import annotations

import pytest

from banna_agent.llm.base import LLMClient
from banna_agent.llm.config import (
    AnthropicConfig,
    BedrockConfig,
    GeminiConfig,
    OllamaConfig,
    OpenAIConfig,
    resolve_provider_config,
)
from banna_agent.llm.registry import (
    list_providers,
    make_client,
    register_client,
)


# ---------------------------------------------------------------------------
# list_providers
# ---------------------------------------------------------------------------


def test_list_providers_has_all_five() -> None:
    names = list_providers()
    assert set(names) >= {"anthropic", "bedrock", "openai", "gemini", "ollama"}


# ---------------------------------------------------------------------------
# make_client dispatch — no network, just constructs
# ---------------------------------------------------------------------------


def test_make_anthropic_client_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    client = make_client("anthropic", model="claude-haiku-4-5-20251001")
    assert client.provider == "anthropic"
    assert isinstance(client, LLMClient)


def test_make_bedrock_client_defaults_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    client = make_client("bedrock")
    assert client.provider == "anthropic"  # adapter class identifier
    # transport is bedrock
    assert getattr(client, "transport", None) == "bedrock"


def test_make_openai_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    client = make_client("openai", model="gpt-5-mini")
    assert client.provider == "openai"


def test_make_gemini_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    client = make_client("gemini", model="gemini-2.5-pro")
    assert client.provider == "gemini"


def test_make_ollama_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:9999")
    client = make_client("ollama", model="qwen2.5:7b")
    assert client.provider == "ollama"
    assert client.base_url == "http://localhost:9999"


def test_make_client_unknown_provider_raises() -> None:
    with pytest.raises(ValueError, match="unknown provider"):
        make_client("martian")


def test_make_client_env_override_bypasses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    override = AnthropicConfig(api_key="sk-from-override", base_url=None)
    client = make_client("anthropic", env_override=override)
    assert client.api_key == "sk-from-override"


# ---------------------------------------------------------------------------
# register_client lets users add a custom provider
# ---------------------------------------------------------------------------


class _DummyClient:
    provider = "dummy"

    def __init__(self, model: str = "d1") -> None:
        self.model = model

    def chat(self, **_: object):
        raise NotImplementedError


def test_register_client_roundtrip() -> None:
    register_client("dummy", lambda cfg, **kw: _DummyClient(model=kw.get("model", "d1")))
    # The config resolver doesn't know about "dummy", so pass env_override=None
    client = make_client("dummy", env_override=object(), model="d2")
    assert client.provider == "dummy"
    assert client.model == "d2"


# ---------------------------------------------------------------------------
# resolve_provider_config
# ---------------------------------------------------------------------------


def test_resolve_anthropic_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-key")
    cfg = resolve_provider_config("anthropic")
    assert isinstance(cfg, AnthropicConfig)
    assert cfg.api_key == "a-key"


def test_resolve_bedrock_config_prefers_aws_region(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    cfg = resolve_provider_config("bedrock")
    assert isinstance(cfg, BedrockConfig)
    assert cfg.aws_region == "eu-west-1"


def test_resolve_gemini_accepts_search_key_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_SEARCH_API_KEY", "legacy-key")
    cfg = resolve_provider_config("gemini")
    assert isinstance(cfg, GeminiConfig)
    assert cfg.api_key == "legacy-key"


def test_resolve_ollama_default_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    cfg = resolve_provider_config("ollama")
    assert isinstance(cfg, OllamaConfig)
    assert cfg.base_url == "http://localhost:11434"


def test_resolve_openai_org_and_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.setenv("OPENAI_ORG", "org_xyz")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy/")
    cfg = resolve_provider_config("openai")
    assert isinstance(cfg, OpenAIConfig)
    assert cfg.organization == "org_xyz"
    assert cfg.base_url == "https://proxy/"


def test_resolve_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown provider"):
        resolve_provider_config("xenoform")
