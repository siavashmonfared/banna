"""Unit tests for LLMClient contract shape.

Adapters are tested separately. This file only covers the projections and
identity invariants on the normalized types.
"""
from __future__ import annotations

from banna_agent.llm.base import (
    ContentBlock,
    LLMClient,
    LLMReply,
    Message,
    ToolCallRequest,
    ToolSpec,
    Usage,
)


# ---------------------------------------------------------------------------
# LLMReply projections
# ---------------------------------------------------------------------------


def test_text_projection_concatenates_only_text_blocks() -> None:
    reply = LLMReply(
        provider="test",
        model="mock-1",
        content=[
            ContentBlock(kind="thinking", text="hidden thought"),
            ContentBlock(kind="text", text="hello "),
            ContentBlock(kind="tool_use", id="t1", name="search", arguments={"q": "x"}),
            ContentBlock(kind="text", text="world"),
        ],
        stop_reason="end_turn",
    )
    assert reply.text == "hello world"


def test_tool_calls_projection_preserves_order() -> None:
    reply = LLMReply(
        provider="test",
        model="mock-1",
        content=[
            ContentBlock(kind="tool_use", id="t1", name="search", arguments={"q": "a"}),
            ContentBlock(kind="text", text="between"),
            ContentBlock(kind="tool_use", id="t2", name="calc", arguments={"expr": "1+1"}),
        ],
        stop_reason="tool_use",
    )
    calls = reply.tool_calls
    assert len(calls) == 2
    assert calls[0].id == "t1" and calls[0].name == "search"
    assert calls[1].id == "t2" and calls[1].name == "calc"


def test_has_tool_calls_flag() -> None:
    no_tools = LLMReply(
        provider="t", model="m",
        content=[ContentBlock(kind="text", text="hi")],
        stop_reason="end_turn",
    )
    with_tools = LLMReply(
        provider="t", model="m",
        content=[ContentBlock(kind="tool_use", id="t1", name="f", arguments={})],
        stop_reason="tool_use",
    )
    assert not no_tools.has_tool_calls
    assert with_tools.has_tool_calls


def test_empty_content_projections() -> None:
    reply = LLMReply(provider="t", model="m", content=[], stop_reason="end_turn")
    assert reply.text == ""
    assert reply.tool_calls == []
    assert not reply.has_tool_calls


def test_tool_calls_dict_is_copied_not_shared() -> None:
    """A projection mustn't let callers mutate the original block's args."""
    block = ContentBlock(kind="tool_use", id="t1", name="x", arguments={"k": 1})
    reply = LLMReply(provider="t", model="m", content=[block], stop_reason="tool_use")
    call = reply.tool_calls[0]
    call.arguments["k"] = 99
    assert block.arguments == {"k": 1}


# ---------------------------------------------------------------------------
# ContentBlock / Message / Usage construction
# ---------------------------------------------------------------------------


def test_contentblock_defaults_are_empty_not_none_dict() -> None:
    b = ContentBlock(kind="text", text="hi")
    assert b.arguments == {}
    assert b.meta == {}
    assert b.result is None


def test_message_preserves_block_order() -> None:
    blocks = [
        ContentBlock(kind="thinking", text="private"),
        ContentBlock(kind="text", text="public"),
        ContentBlock(kind="tool_use", id="t1", name="f", arguments={}),
    ]
    msg = Message(role="assistant", content=blocks)
    assert [b.kind for b in msg.content] == ["thinking", "text", "tool_use"]


def test_usage_defaults_are_zero() -> None:
    u = Usage()
    assert u.tokens_in == 0
    assert u.cache_read_tokens == 0
    assert u.reasoning_tokens == 0
    assert u.thoughts_tokens == 0
    assert u.cost_usd is None


def test_toolspec_shape() -> None:
    spec = ToolSpec(
        name="search",
        description="web search",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
    )
    assert spec.name == "search"
    assert spec.input_schema["required"] == ["q"]


# ---------------------------------------------------------------------------
# Protocol conformance — can a minimal dummy class satisfy LLMClient?
# ---------------------------------------------------------------------------


class _DummyClient:
    provider = "dummy"

    def chat(
        self,
        *,
        messages,
        tools=(),
        model=None,
        max_tokens=None,
        temperature=None,
        system=None,
        extra=None,
    ) -> LLMReply:
        return LLMReply(
            provider=self.provider,
            model=model or "dummy-1",
            content=[ContentBlock(kind="text", text="ok")],
            stop_reason="end_turn",
        )


def test_dummy_client_satisfies_protocol() -> None:
    client: LLMClient = _DummyClient()
    assert isinstance(client, LLMClient)  # runtime_checkable
    reply = client.chat(messages=[Message(role="user", content=[ContentBlock(kind="text", text="hi")])])
    assert reply.text == "ok"
    assert reply.stop_reason == "end_turn"


def test_toolcallrequest_is_constructible_standalone() -> None:
    """Used when policies convert projections into intent structures."""
    r = ToolCallRequest(id="t1", name="search", arguments={"q": "foo"})
    assert r.id == "t1"
    assert r.arguments["q"] == "foo"
    assert r.raw is None
