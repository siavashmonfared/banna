"""cmd_provider resets a stale model when switching providers.

Regression: switching huggingface (model=thinkingmachines/Inkling) to
anthropic used to keep the Inkling id, which 400s against the Anthropic
API. The switch must drop a model the new provider doesn't serve.
"""
from __future__ import annotations

import pytest

from banna_agent.cli import commands as C


class _App:
    def __init__(self, provider: str, model: str) -> None:
        from rich.console import Console
        self.console = Console()
        self.provider = provider
        self.model = model
        self.rebuilt = 0

    def rebuild_llm(self) -> None:
        self.rebuilt += 1  # avoid constructing real SDK clients

    def rebuild_policy(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _keys(monkeypatch):
    # _ensure_api_key must pass for the cloud providers under test.
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "MOONSHOT_API_KEY",
                "HF_TOKEN"):
        monkeypatch.setenv(var, "test-key")


def test_switch_drops_foreign_model():
    app = _App("huggingface", "thinkingmachines/Inkling")
    C.cmd_provider(app, ["anthropic"])
    assert app.provider == "anthropic"
    assert app.model != "thinkingmachines/Inkling"
    assert app.model in C.KNOWN_MODELS["anthropic"]
    assert app.rebuilt == 1


def test_switch_to_openai_compat_provider_resets_model():
    app = _App("anthropic", "claude-opus-4-8")
    C.cmd_provider(app, ["kimi"])
    assert app.provider == "kimi"
    assert app.model in C.KNOWN_MODELS["kimi"]


def test_switch_keeps_valid_model():
    # Re-selecting the same provider with a model it serves keeps it.
    app = _App("anthropic", "claude-sonnet-5")
    C.cmd_provider(app, ["anthropic"])
    assert app.model == "claude-sonnet-5"
