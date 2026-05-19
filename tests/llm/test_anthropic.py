"""Tests for the Anthropic adapter.

Strategy: inject a fake SDK whose `messages.create(**kwargs)` captures the
kwargs and returns a dict-shaped response. This lets us verify:

- Request serialization: normalized Message -> Anthropic wire format,
  including tool_use / tool_result round-trips and system prompt handling.
- Response deserialization: dict response -> LLMReply with correct
  ContentBlocks, stop_reason, and Usage (including cache tokens).

No network, no real SDK required.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from banna_agent.llm.anthropic import AnthropicClient
from banna_agent.llm.base import (
    ContentBlock,
    Message,
    ToolSpec,
)


# ---------------------------------------------------------------------------
# Fake SDK
# ---------------------------------------------------------------------------


class _Messages:
    def __init__(self, fake_response: dict[str, Any]) -> None:
        self.fake_response = fake_response
        self.last_kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> dict[str, Any]:
        self.last_kwargs = kwargs
        return self.fake_response


@dataclass
class _FakeSDK:
    fake_response: dict[str, Any] = field(default_factory=dict)
    messages: _Messages = field(init=False)

    def __post_init__(self) -> None:
        self.messages = _Messages(self.fake_response)


def _basic_response(text: str = "hello") -> dict[str, Any]:
    return {
        "id": "msg_1",
        "model": "claude-opus-4-5-20251101",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 5},
    }


def _tool_use_response() -> dict[str, Any]:
    return {
        "id": "msg_2",
        "model": "claude-opus-4-5-20251101",
        "role": "assistant",
        "content": [
            {"type": "text", "text": "I'll search."},
            {
                "type": "tool_use",
                "id": "toolu_abc",
                "name": "search",
                "input": {"q": "netflix arpu"},
            },
        ],
        "stop_reason": "tool_use",
        "usage": {
            "input_tokens": 40,
            "output_tokens": 20,
            "cache_creation_input_tokens": 100,
            "cache_read_input_tokens": 200,
        },
    }


# ---------------------------------------------------------------------------
# Request serialization
# ---------------------------------------------------------------------------


def test_request_plain_text_message() -> None:
    sdk = _FakeSDK(fake_response=_basic_response())
    client = AnthropicClient(sdk=sdk)
    client.chat(
        messages=[Message(role="user", content=[ContentBlock(kind="text", text="hi")])],
        max_tokens=50,
        temperature=0.0,
    )
    assert sdk.messages.last_kwargs is not None
    k = sdk.messages.last_kwargs
    assert k["model"] == "claude-opus-4-5-20251101"
    assert k["max_tokens"] == 50
    assert k["temperature"] == 0.0
    assert k["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    ]
    assert "system" not in k
    assert "tools" not in k


def test_request_system_prompt_flows_through() -> None:
    sdk = _FakeSDK(fake_response=_basic_response())
    client = AnthropicClient(sdk=sdk)
    client.chat(
        messages=[Message(role="user", content=[ContentBlock(kind="text", text="hi")])],
        system="You are a helpful assistant.",
    )
    assert sdk.messages.last_kwargs["system"] == "You are a helpful assistant."


def test_request_tool_spec_flows_through() -> None:
    sdk = _FakeSDK(fake_response=_basic_response())
    client = AnthropicClient(sdk=sdk)
    client.chat(
        messages=[Message(role="user", content=[ContentBlock(kind="text", text="hi")])],
        tools=[
            ToolSpec(
                name="search",
                description="web search",
                input_schema={
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            )
        ],
    )
    tools = sdk.messages.last_kwargs["tools"]
    assert tools == [
        {
            "name": "search",
            "description": "web search",
            "input_schema": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        }
    ]


def test_request_tool_use_and_tool_result_round_trip() -> None:
    """The hardest serialization case: a 3-turn conversation where the
    assistant called a tool, the user replied with a tool_result, and we
    send the whole history back for the next turn."""
    sdk = _FakeSDK(fake_response=_basic_response("final answer"))
    client = AnthropicClient(sdk=sdk)

    history = [
        Message(role="user", content=[ContentBlock(kind="text", text="search netflix")]),
        Message(
            role="assistant",
            content=[
                ContentBlock(kind="text", text="searching"),
                ContentBlock(
                    kind="tool_use",
                    id="toolu_abc",
                    name="search",
                    arguments={"q": "netflix arpu"},
                ),
            ],
        ),
        Message(
            role="user",
            content=[
                ContentBlock(
                    kind="tool_result",
                    id="toolu_abc",
                    result={"hits": ["$11.64"]},
                )
            ],
        ),
    ]
    client.chat(messages=history)

    serialized = sdk.messages.last_kwargs["messages"]
    assert len(serialized) == 3
    # turn 1: user text
    assert serialized[0] == {
        "role": "user",
        "content": [{"type": "text", "text": "search netflix"}],
    }
    # turn 2: assistant text + tool_use preserved in order
    assert serialized[1]["role"] == "assistant"
    assert serialized[1]["content"][0] == {"type": "text", "text": "searching"}
    assert serialized[1]["content"][1] == {
        "type": "tool_use",
        "id": "toolu_abc",
        "name": "search",
        "input": {"q": "netflix arpu"},
    }
    # turn 3: user tool_result
    assert serialized[2]["role"] == "user"
    assert serialized[2]["content"][0]["type"] == "tool_result"
    assert serialized[2]["content"][0]["tool_use_id"] == "toolu_abc"
    assert "hits" in serialized[2]["content"][0]["content"]  # JSON-encoded


def test_tool_result_string_content_passes_through() -> None:
    sdk = _FakeSDK(fake_response=_basic_response())
    client = AnthropicClient(sdk=sdk)
    client.chat(
        messages=[
            Message(
                role="user",
                content=[
                    ContentBlock(kind="tool_result", id="t1", result="raw string result")
                ],
            )
        ]
    )
    serialized = sdk.messages.last_kwargs["messages"][0]["content"][0]
    assert serialized["content"] == "raw string result"


def test_tool_result_error_flag() -> None:
    sdk = _FakeSDK(fake_response=_basic_response())
    client = AnthropicClient(sdk=sdk)
    client.chat(
        messages=[
            Message(
                role="user",
                content=[
                    ContentBlock(
                        kind="tool_result",
                        id="t1",
                        result="boom",
                        is_error=True,
                    )
                ],
            )
        ]
    )
    serialized = sdk.messages.last_kwargs["messages"][0]["content"][0]
    assert serialized["is_error"] is True


def test_block_raw_dict_preferred_over_projection() -> None:
    """Same-provider replay: if a block has `raw`, echo it verbatim so
    fields like thinking signatures survive round-trip."""
    sdk = _FakeSDK(fake_response=_basic_response())
    client = AnthropicClient(sdk=sdk)
    raw_thinking = {
        "type": "thinking",
        "thinking": "Let me think...",
        "signature": "abc123",
    }
    client.chat(
        messages=[
            Message(
                role="assistant",
                content=[ContentBlock(kind="thinking", text="Let me think...", raw=raw_thinking)],
            )
        ]
    )
    serialized = sdk.messages.last_kwargs["messages"][0]["content"][0]
    assert serialized == raw_thinking  # signature preserved


# ---------------------------------------------------------------------------
# Response deserialization
# ---------------------------------------------------------------------------


def test_response_text_reply() -> None:
    sdk = _FakeSDK(fake_response=_basic_response("hello world"))
    client = AnthropicClient(sdk=sdk)
    reply = client.chat(
        messages=[Message(role="user", content=[ContentBlock(kind="text", text="hi")])]
    )
    assert reply.provider == "anthropic"
    assert reply.model == "claude-opus-4-5-20251101"
    assert reply.stop_reason == "end_turn"
    assert reply.text == "hello world"
    assert reply.tool_calls == []
    assert reply.usage.tokens_in == 12
    assert reply.usage.tokens_out == 5
    assert reply.usage.total_tokens == 17
    assert reply.usage.cache_read_tokens == 0
    assert reply.usage.cache_write_tokens == 0


def test_response_tool_use_reply() -> None:
    sdk = _FakeSDK(fake_response=_tool_use_response())
    client = AnthropicClient(sdk=sdk)
    reply = client.chat(
        messages=[Message(role="user", content=[ContentBlock(kind="text", text="search netflix")])]
    )
    assert reply.stop_reason == "tool_use"
    assert reply.has_tool_calls
    assert reply.text == "I'll search."
    assert len(reply.tool_calls) == 1
    call = reply.tool_calls[0]
    assert call.id == "toolu_abc"
    assert call.name == "search"
    assert call.arguments == {"q": "netflix arpu"}


def test_response_cache_tokens_populated() -> None:
    sdk = _FakeSDK(fake_response=_tool_use_response())
    client = AnthropicClient(sdk=sdk)
    reply = client.chat(
        messages=[Message(role="user", content=[ContentBlock(kind="text", text="x")])]
    )
    assert reply.usage.cache_write_tokens == 100
    assert reply.usage.cache_read_tokens == 200
    assert reply.usage.tokens_in == 40
    assert reply.usage.tokens_out == 20


def test_response_preserves_content_order() -> None:
    """Thinking block then text block then tool_use — all three survive
    in order. This is the reason `content` is primary, not text+tool_calls."""
    sdk = _FakeSDK(fake_response={
        "id": "msg_x",
        "model": "claude-opus-4-5-20251101",
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "hmm", "signature": "sig1"},
            {"type": "text", "text": "let me call a tool"},
            {"type": "tool_use", "id": "t1", "name": "calc", "input": {"expr": "2+2"}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 5, "output_tokens": 3},
    })
    client = AnthropicClient(sdk=sdk)
    reply = client.chat(
        messages=[Message(role="user", content=[ContentBlock(kind="text", text="x")])]
    )
    assert [b.kind for b in reply.content] == ["thinking", "text", "tool_use"]
    # The thinking block's raw is preserved so we can round-trip with signature.
    thinking = reply.content[0]
    assert thinking.raw["signature"] == "sig1"


def test_response_with_missing_usage_returns_zero() -> None:
    sdk = _FakeSDK(fake_response={
        "id": "m", "model": "claude-opus-4-5-20251101", "role": "assistant",
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
    })
    client = AnthropicClient(sdk=sdk)
    reply = client.chat(
        messages=[Message(role="user", content=[ContentBlock(kind="text", text="x")])]
    )
    assert reply.usage.tokens_in == 0
    assert reply.usage.tokens_out == 0


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


def test_no_api_key_raises_when_sdk_not_injected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = AnthropicClient()  # no sdk, no api_key
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        client.chat(
            messages=[Message(role="user", content=[ContentBlock(kind="text", text="hi")])]
        )


def test_custom_model_override() -> None:
    sdk = _FakeSDK(fake_response=_basic_response())
    client = AnthropicClient(model="claude-haiku-4-5-20251001", sdk=sdk)
    client.chat(
        messages=[Message(role="user", content=[ContentBlock(kind="text", text="hi")])],
        model="claude-sonnet-4-6",  # per-call override
    )
    assert sdk.messages.last_kwargs["model"] == "claude-sonnet-4-6"


def test_extra_kwargs_passed_through() -> None:
    sdk = _FakeSDK(fake_response=_basic_response())
    client = AnthropicClient(sdk=sdk)
    client.chat(
        messages=[Message(role="user", content=[ContentBlock(kind="text", text="hi")])],
        extra={"metadata": {"user_id": "sina"}},
    )
    assert sdk.messages.last_kwargs["metadata"] == {"user_id": "sina"}
