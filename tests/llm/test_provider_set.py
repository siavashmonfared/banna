"""banna ships exactly these providers: anthropic, bedrock, openai, gemini (Google), openrouter, ollama."""
from __future__ import annotations

from banna_agent.cli.model_catalog import CURATED, KEY_VARS, PROVIDER_LABELS, PROVIDER_ORDER
from banna_agent.llm.config import OpenAIConfig, resolve_provider_config
from banna_agent.llm.openai import OpenAIClient
from banna_agent.llm.registry import list_providers, make_client

PROVIDERS = {"anthropic", "bedrock", "openai", "gemini", "openrouter", "ollama"}


def test_registry_and_catalog_list_the_same_providers():
    assert set(list_providers()) == PROVIDERS
    assert set(PROVIDER_ORDER) == PROVIDERS
    assert set(PROVIDER_LABELS) == PROVIDERS
    assert set(CURATED) == PROVIDERS


def test_openrouter_resolves_to_the_openai_adapter_with_its_own_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    cfg = resolve_provider_config("openrouter")
    assert isinstance(cfg, OpenAIConfig)
    assert (cfg.api_key, cfg.base_url) == ("or-key", "https://openrouter.ai/api/v1")
    assert KEY_VARS["openrouter"] == ("OPENROUTER_API_KEY",)
    client = make_client("openrouter", context_guard=False)
    assert isinstance(client, OpenAIClient)
    assert client.model == "anthropic/claude-haiku-4.5"
    assert client.base_url == "https://openrouter.ai/api/v1"


def test_openrouter_base_url_is_overridable(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "http://proxy.local/v1")
    assert resolve_provider_config("openrouter").base_url == "http://proxy.local/v1"


def test_removed_providers_are_gone():
    import pytest
    for name in ("kimi", "glm", "huggingface"):
        with pytest.raises(ValueError):
            make_client(name)


def test_providers_command_lists_every_provider(capsys, monkeypatch):
    from banna_agent.cli import model_catalog, subcommands
    from banna_agent.cli.model_catalog import ProviderStatus

    monkeypatch.setattr(model_catalog, "scan_providers",
                        lambda **kw: {p: ProviderStatus(p, False, "x") for p in PROVIDER_ORDER})
    assert subcommands._cmd_providers([]) == 0
    out = capsys.readouterr().out
    assert all(f"  {p}" in out for p in PROVIDERS)


def test_ollama_up_without_models_is_not_reported_as_down(monkeypatch):
    from banna_agent.cli import model_catalog

    monkeypatch.setattr(model_catalog, "detect_ollama", lambda timeout_s=1.0: [])
    assert "no models pulled" in model_catalog._ollama_status().detail
    monkeypatch.setattr(model_catalog, "detect_ollama", lambda timeout_s=1.0: None)
    assert "not running" in model_catalog._ollama_status().detail
