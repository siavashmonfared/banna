"""Tests for ProviderError(retryable=False) raised by each provider."""
from __future__ import annotations

import pytest

from banna_agent.llm.base import ContentBlock, Message, ProviderError
from banna_agent.llm.ollama import OllamaClient


def _msg(text: str) -> Message:
    return Message(role="user", content=[ContentBlock(kind="text", text=text)])


# ---------------------------------------------------------------------------
# Missing API key — non-retryable
# ---------------------------------------------------------------------------


def test_openai_missing_key_raises_non_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from banna_agent.llm.openai import OpenAIClient

    client = OpenAIClient(api_key=None)
    with pytest.raises(ProviderError) as exc:
        client.chat(messages=[_msg("hi")])
    assert exc.value.retryable is False
    assert "OPENAI_API_KEY" in str(exc.value)


def test_anthropic_missing_key_raises_non_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from banna_agent.llm.anthropic import AnthropicClient

    client = AnthropicClient(api_key=None)
    with pytest.raises(ProviderError) as exc:
        client.chat(messages=[_msg("hi")])
    assert exc.value.retryable is False
    assert "ANTHROPIC_API_KEY" in str(exc.value)


def test_gemini_missing_key_raises_non_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_SEARCH_API_KEY", raising=False)
    from banna_agent.llm.gemini import GeminiClient

    client = GeminiClient(api_key=None)
    with pytest.raises(ProviderError) as exc:
        client.chat(messages=[_msg("hi")])
    assert exc.value.retryable is False
    assert "GOOGLE_API_KEY" in str(exc.value)


# ---------------------------------------------------------------------------
# Ollama "does not support tools" — non-retryable
# ---------------------------------------------------------------------------


class _FakeResp:
    def __init__(self, status: int, body: dict) -> None:
        self.status_code = status
        self._body = body
        self.text = ""

    def json(self) -> dict:
        return self._body


def test_ollama_no_tool_support_is_non_retryable() -> None:
    def fake_post(url, json=None, timeout=None):
        return _FakeResp(400, {
            "error": "registry.ollama.ai/library/deepseek-r1:8b does not support tools"
        })

    client = OllamaClient(model="deepseek-r1:8b", http_post=fake_post)
    with pytest.raises(ProviderError) as exc:
        client.chat(messages=[_msg("hi")])
    assert exc.value.retryable is False
    assert "does not support tools" in str(exc.value)


def test_ollama_generic_500_is_retryable() -> None:
    """Transient server-side errors stay retryable (rate limits, blips)."""
    def fake_post(url, json=None, timeout=None):
        return _FakeResp(500, {"error": "internal server error"})

    client = OllamaClient(model="qwen3:8b", http_post=fake_post)
    with pytest.raises(ProviderError) as exc:
        client.chat(messages=[_msg("hi")])
    assert exc.value.retryable is True
