"""Unit tests for the OpenAI adapter (offline)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from banna_agent.llm.base import ContentBlock, Message, ToolSpec
from banna_agent.llm.openai import (
    OpenAIClient,
    _messages_to_openai,
)


# ---------------------------------------------------------------------------
# Fake SDK
# ---------------------------------------------------------------------------


@dataclass
class _ChatCompletions:
    fake_response: dict
    last_kwargs: dict | None = field(default=None)

    def create(self, **kwargs: Any) -> dict:
        self.last_kwargs = kwargs
        return self.fake_response


@dataclass
class _Chat:
    completions: _ChatCompletions


@dataclass
class _FakeSDK:
    fake_response: dict
    chat: _Chat = field(init=False)

    def __post_init__(self) -> None:
        self.chat = _Chat(completions=_ChatCompletions(self.fake_response))


def _basic_response(text: str = "hi") -> dict:
    return {
        "id": "chatcmpl-1",
        "model": "gpt-5-mini",
        "choices": [
            {
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 4},
    }


def _tool_call_response() -> dict:
    return {
        "id": "chatcmpl-2",
        "model": "gpt-5-mini",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": '{"q": "netflix arpu"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 30, "completion_tokens": 15,
            "completion_tokens_details": {"reasoning_tokens": 7},
            "prompt_tokens_details": {"cached_tokens": 12},
        },
    }


# ---------------------------------------------------------------------------
# Message serialization
# ---------------------------------------------------------------------------


def test_messages_to_openai_plain_user() -> None:
    msgs = _messages_to_openai(
        [Message(role="user", content=[ContentBlock(kind="text", text="hi")])]
    )
    assert msgs == [{"role": "user", "content": "hi"}]


def test_messages_to_openai_assistant_with_tool_calls() -> None:
    msgs = _messages_to_openai(
        [
            Message(
                role="assistant",
                content=[
                    ContentBlock(kind="text", text="calling"),
                    ContentBlock(kind="tool_use", id="t1", name="search", arguments={"q": "x"}),
                ],
            )
        ]
    )
    assert msgs[0]["role"] == "assistant"
    assert msgs[0]["content"] == "calling"
    assert msgs[0]["tool_calls"][0]["id"] == "t1"
    assert msgs[0]["tool_calls"][0]["function"]["name"] == "search"
    import json
    assert json.loads(msgs[0]["tool_calls"][0]["function"]["arguments"]) == {"q": "x"}


def test_messages_to_openai_splits_tool_results_into_tool_role() -> None:
    msgs = _messages_to_openai(
        [
            Message(
                role="user",
                content=[
                    ContentBlock(kind="tool_result", id="t1", result={"hits": ["a"]}),
                    ContentBlock(kind="tool_result", id="t2", result="plain"),
                ],
            )
        ]
    )
    # Tool results → one role=tool message each. No user message because no text.
    assert len(msgs) == 2
    assert all(m["role"] == "tool" for m in msgs)
    assert msgs[0]["tool_call_id"] == "t1"
    assert "hits" in msgs[0]["content"]
    assert msgs[1]["tool_call_id"] == "t2"
    assert msgs[1]["content"] == "plain"


# ---------------------------------------------------------------------------
# End-to-end round trip with fake SDK
# ---------------------------------------------------------------------------


def test_chat_plain_reply() -> None:
    sdk = _FakeSDK(fake_response=_basic_response("hello"))
    client = OpenAIClient(sdk=sdk, model="gpt-5-mini")
    reply = client.chat(
        messages=[Message(role="user", content=[ContentBlock(kind="text", text="hi")])]
    )
    assert reply.provider == "openai"
    assert reply.text == "hello"
    assert reply.stop_reason == "end_turn"
    assert reply.usage.tokens_in == 10


def test_chat_tool_call_reply() -> None:
    sdk = _FakeSDK(fake_response=_tool_call_response())
    client = OpenAIClient(sdk=sdk)
    reply = client.chat(
        messages=[Message(role="user", content=[ContentBlock(kind="text", text="search netflix")])]
    )
    assert reply.stop_reason == "tool_use"
    assert reply.has_tool_calls
    call = reply.tool_calls[0]
    assert call.id == "call_1"
    assert call.arguments == {"q": "netflix arpu"}


def test_chat_usage_reasoning_and_cache() -> None:
    sdk = _FakeSDK(fake_response=_tool_call_response())
    client = OpenAIClient(sdk=sdk)
    reply = client.chat(
        messages=[Message(role="user", content=[ContentBlock(kind="text", text="x")])]
    )
    assert reply.usage.reasoning_tokens == 7
    assert reply.usage.cache_read_tokens == 12


def test_tool_spec_flows_through() -> None:
    sdk = _FakeSDK(fake_response=_basic_response())
    client = OpenAIClient(sdk=sdk)
    client.chat(
        messages=[Message(role="user", content=[ContentBlock(kind="text", text="hi")])],
        tools=[ToolSpec(name="search", description="web", input_schema={"type": "object"})],
    )
    tools = sdk.chat.completions.last_kwargs["tools"]
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "search"


def test_no_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = OpenAIClient()
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        client.chat(
            messages=[Message(role="user", content=[ContentBlock(kind="text", text="hi")])]
        )


# ---------------------------------------------------------------------------
# Reasoning-model floor (Q2)
# ---------------------------------------------------------------------------


def test_floor_unchanged_for_non_reasoning_model() -> None:
    from banna_agent.llm.openai import _floor_for_reasoning
    # Older models pass through untouched.
    assert _floor_for_reasoning("gpt-3.5-turbo", 600) == 600
    assert _floor_for_reasoning("gpt-4-turbo", 100) == 100
    assert _floor_for_reasoning("gpt-4o-mini", 50) == 50


def test_floor_raises_for_reasoning_models() -> None:
    """Reasoning models get a floor so internal CoT doesn't blow the
    visible-output budget."""
    from banna_agent.llm.openai import _floor_for_reasoning

    # Default floor is 4096. A small policy cap should be raised.
    assert _floor_for_reasoning("gpt-5-nano", 600) == 4096
    assert _floor_for_reasoning("gpt-5", 1024) == 4096
    assert _floor_for_reasoning("o4-mini", 100) == 4096
    assert _floor_for_reasoning("o1", 1) == 4096


def test_floor_does_not_lower_a_high_cap() -> None:
    """If a policy already asked for more than the floor, keep it."""
    from banna_agent.llm.openai import _floor_for_reasoning
    assert _floor_for_reasoning("gpt-5-nano", 8000) == 8000
    assert _floor_for_reasoning("o4-mini", 16000) == 16000


def test_floor_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """OPENAI_REASONING_MIN_TOKENS overrides the default floor."""
    from banna_agent.llm.openai import _floor_for_reasoning

    monkeypatch.setenv("OPENAI_REASONING_MIN_TOKENS", "8192")
    assert _floor_for_reasoning("gpt-5-nano", 600) == 8192
    monkeypatch.setenv("OPENAI_REASONING_MIN_TOKENS", "0")  # invalid, fall back
    assert _floor_for_reasoning("gpt-5-nano", 600) == 4096
    monkeypatch.setenv("OPENAI_REASONING_MIN_TOKENS", "garbage")  # invalid
    assert _floor_for_reasoning("gpt-5-nano", 600) == 4096


def test_chat_passes_floored_value_to_api() -> None:
    """End-to-end: gpt-5-nano with max_tokens=600 should send
    max_completion_tokens=4096 to the OpenAI API."""
    from banna_agent.llm.base import ContentBlock, Message
    from banna_agent.llm.openai import OpenAIClient

    captured: dict = {}

    class _FakeSDK:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    captured.update(kwargs)
                    # Fake response shape — just enough to satisfy
                    # _response_to_reply.
                    class _Choice:
                        message = type("M", (), {
                            "content": "ok", "tool_calls": None, "role": "assistant",
                        })()
                        finish_reason = "stop"
                    class _Usage:
                        prompt_tokens = 10
                        completion_tokens = 2
                        total_tokens = 12

                    class _Resp:
                        choices = [_Choice()]
                        usage = _Usage()
                        model = kwargs.get("model", "?")
                    return _Resp()

    client = OpenAIClient(model="gpt-5-nano", api_key="sk-test", sdk=_FakeSDK())
    client.chat(
        messages=[Message(role="user", content=[ContentBlock(kind="text", text="hi")])],
        max_tokens=600,
    )
    # Floored to 4096, sent under the new param name.
    assert captured.get("max_completion_tokens") == 4096
    assert "max_tokens" not in captured
    # Temperature was not set (gpt-5-nano rejects custom).
    assert "temperature" not in captured


def test_chat_legacy_model_uses_max_tokens_unchanged() -> None:
    """Old gpt-3.5 path: legacy max_tokens param, no floor."""
    from banna_agent.llm.base import ContentBlock, Message
    from banna_agent.llm.openai import OpenAIClient

    captured: dict = {}

    class _FakeSDK:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    captured.update(kwargs)
                    class _Choice:
                        message = type("M", (), {
                            "content": "ok", "tool_calls": None, "role": "assistant",
                        })()
                        finish_reason = "stop"
                    class _Usage:
                        prompt_tokens = 10
                        completion_tokens = 2
                        total_tokens = 12
                    class _Resp:
                        choices = [_Choice()]
                        usage = _Usage()
                        model = kwargs.get("model", "?")
                    return _Resp()

    client = OpenAIClient(model="gpt-3.5-turbo", api_key="sk-test", sdk=_FakeSDK())
    client.chat(
        messages=[Message(role="user", content=[ContentBlock(kind="text", text="hi")])],
        max_tokens=600, temperature=0.0,
    )
    # Legacy path: max_tokens stays, no floor, temperature passes through.
    assert captured.get("max_tokens") == 600
    assert "max_completion_tokens" not in captured
    assert captured.get("temperature") == 0.0
