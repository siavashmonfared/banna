"""Tests for the setup-wizard's Ollama tool-support probe."""
from __future__ import annotations

import json
from typing import Any

import pytest

from banna_agent.cli.setup_wizard import _probe_ollama_tool_support


class _FakeResp:
    def __init__(self, status: int, body: dict | None = None, text: str = "") -> None:
        self.status_code = status
        self._body = body or {}
        self.text = text
        self.headers = {"content-type": "application/json"}

    def json(self) -> dict:
        return self._body


def test_probe_passes_for_tool_capable_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, json: Any = None, timeout: float = 0) -> _FakeResp:
        return _FakeResp(200, {"model": "qwen3:8b", "message": {"role": "assistant"}})
    monkeypatch.setattr("banna_agent.cli.setup_wizard.requests.post", fake_post)

    ok, reason = _probe_ollama_tool_support("qwen3:8b")
    assert ok is True
    assert reason == ""


def test_probe_fails_for_deepseek_r1(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, json: Any = None, timeout: float = 0) -> _FakeResp:
        return _FakeResp(400, {
            "error": "registry.ollama.ai/library/deepseek-r1:8b does not support tools"
        })
    monkeypatch.setattr("banna_agent.cli.setup_wizard.requests.post", fake_post)

    ok, reason = _probe_ollama_tool_support("deepseek-r1:8b")
    assert ok is False
    assert "does not support tools" in reason


def test_probe_soft_fails_on_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Network errors shouldn't block the wizard — return ok=True so the
    user can still pick the model; the real agent loop will surface any
    issue cleanly via ProviderError."""
    def fake_post(url: str, json: Any = None, timeout: float = 0) -> _FakeResp:
        raise ConnectionError("ollama not reachable")
    monkeypatch.setattr("banna_agent.cli.setup_wizard.requests.post", fake_post)

    ok, reason = _probe_ollama_tool_support("any-model")
    assert ok is True
