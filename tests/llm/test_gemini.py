"""Unit tests for the Gemini adapter (offline, mocked HTTP)."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from banna_agent.llm.base import ContentBlock, Message, ToolSpec
from banna_agent.llm.gemini import GeminiClient, _messages_to_gemini


@dataclass
class _FakeResp:
    status_code: int
    _json: dict

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._json


def _make_post(payload: dict, captured: dict):
    def fake_post(url, params=None, json=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        captured["body"] = json
        captured["timeout"] = timeout
        return _FakeResp(200, payload)
    return fake_post


# ---------------------------------------------------------------------------
# _messages_to_gemini
# ---------------------------------------------------------------------------


def test_messages_to_gemini_user_text() -> None:
    contents, sys = _messages_to_gemini(
        [Message(role="user", content=[ContentBlock(kind="text", text="hi")])]
    )
    assert sys is None
    assert contents == [{"role": "user", "parts": [{"text": "hi"}]}]


def test_messages_to_gemini_system_pulled_out() -> None:
    contents, sys = _messages_to_gemini(
        [
            Message(role="system", content=[ContentBlock(kind="text", text="be helpful")]),
            Message(role="user", content=[ContentBlock(kind="text", text="hi")]),
        ]
    )
    assert sys == "be helpful"
    assert len(contents) == 1  # system stripped from contents


def test_messages_to_gemini_tool_use_and_result_round_trip() -> None:
    contents, _ = _messages_to_gemini(
        [
            Message(
                role="assistant",
                content=[
                    ContentBlock(kind="tool_use", id="", name="search", arguments={"q": "x"}),
                ],
            ),
            Message(
                role="user",
                content=[
                    ContentBlock(kind="tool_result", name="search", result={"hits": ["a"]}),
                ],
            ),
        ]
    )
    assert contents[0]["role"] == "model"  # assistant → model
    assert contents[0]["parts"][0]["functionCall"]["name"] == "search"
    assert contents[1]["role"] == "user"
    assert contents[1]["parts"][0]["functionResponse"]["response"] == {"hits": ["a"]}


# ---------------------------------------------------------------------------
# End-to-end chat
# ---------------------------------------------------------------------------


def _basic_payload(text: str = "hello") -> dict:
    return {
        "candidates": [
            {
                "content": {"parts": [{"text": text}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 12,
            "candidatesTokenCount": 3,
            "totalTokenCount": 15,
            "thoughtsTokenCount": 0,
        },
    }


def _function_call_payload() -> dict:
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"functionCall": {"name": "search", "args": {"q": "netflix"}}},
                    ]
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 20, "candidatesTokenCount": 5,
            "totalTokenCount": 25, "thoughtsTokenCount": 3,
            "toolUsePromptTokenCount": 2,
        },
    }


def test_chat_text_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    monkeypatch.setenv("GOOGLE_API_KEY", "key")
    client = GeminiClient(http_post=_make_post(_basic_payload("hi"), captured))
    reply = client.chat(
        messages=[Message(role="user", content=[ContentBlock(kind="text", text="hello")])]
    )
    assert reply.provider == "gemini"
    assert reply.text == "hi"
    assert reply.stop_reason == "end_turn"
    assert reply.usage.tokens_in == 12
    assert "gemini-2.5-pro" in captured["url"]


def test_chat_function_call_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "key")
    client = GeminiClient(http_post=_make_post(_function_call_payload(), {}))
    reply = client.chat(
        messages=[Message(role="user", content=[ContentBlock(kind="text", text="search netflix")])]
    )
    assert reply.stop_reason == "tool_use"
    assert reply.has_tool_calls
    call = reply.tool_calls[0]
    assert call.name == "search"
    assert call.arguments == {"q": "netflix"}
    assert reply.usage.thoughts_tokens == 3
    assert reply.usage.tool_prompt_tokens == 2


def test_chat_with_tools_sends_function_declarations(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    monkeypatch.setenv("GOOGLE_API_KEY", "key")
    client = GeminiClient(http_post=_make_post(_basic_payload(), captured))
    client.chat(
        messages=[Message(role="user", content=[ContentBlock(kind="text", text="x")])],
        tools=[ToolSpec(name="calc", description="calculator",
                        input_schema={"type": "object", "properties": {}})],
    )
    body = captured["body"]
    decl = body["tools"][0]["function_declarations"][0]
    assert decl["name"] == "calc"


def test_chat_without_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_SEARCH_API_KEY", raising=False)
    client = GeminiClient()
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        client.chat(
            messages=[Message(role="user", content=[ContentBlock(kind="text", text="x")])]
        )
