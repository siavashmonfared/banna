"""Provider-agnostic truncation detection.

The guard's whole point is that it makes no assumption about which
provider is underneath or what its context window is — it only compares
what we sent against the usage the provider reported.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from banna_agent.llm.base import (
    ContentBlock,
    LLMReply,
    Message,
    ProviderError,
    ToolSpec,
    Usage,
)
from banna_agent.llm.context import (
    ContextGuard,
    accounted_input_tokens,
    estimate_request_tokens,
    truncation_shortfall,
)


@dataclass
class _FakeClient:
    """Stands in for any adapter; reports whatever usage the test wants."""

    provider: str = "fake"
    usage: Usage = field(default_factory=Usage)
    seen: dict = field(default_factory=dict)

    def chat(self, **kwargs: Any) -> LLMReply:
        self.seen.update(kwargs)
        return LLMReply(
            provider=self.provider,
            model="m",
            content=[ContentBlock(kind="text", text="ok")],
            stop_reason="end_turn",
            usage=self.usage,
        )


def _msg(text: str) -> Message:
    return Message(role="user", content=[ContentBlock(kind="text", text=text)])


def _tools(n: int, desc_chars: int = 800) -> list[ToolSpec]:
    return [
        ToolSpec(name=f"tool_{i}", description="d" * desc_chars,
                 input_schema={"type": "object", "properties": {}})
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------


def test_estimate_counts_tool_schemas_not_just_messages() -> None:
    """Tool declarations are billed as input tokens and are usually the
    thing that overflows the window, so they must be in the estimate."""
    bare = estimate_request_tokens(messages=[_msg("hi")])
    with_tools = estimate_request_tokens(messages=[_msg("hi")], tools=_tools(10))
    assert with_tools > bare * 10


def test_estimate_counts_system_prompt() -> None:
    assert estimate_request_tokens(system="s" * 4000) >= 900


def test_accounted_input_tokens_includes_cache_columns() -> None:
    """A cache hit means the provider had those tokens, not that it skipped
    them — counting only `tokens_in` would look like truncation."""
    usage = Usage(tokens_in=100, cache_read_tokens=5000, cache_write_tokens=200)
    assert accounted_input_tokens(usage) == 5300


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def test_shortfall_flags_clamped_prompt() -> None:
    """The real failure: ~6500 tokens sent, provider counts exactly 4096."""
    assert truncation_shortfall(estimated=6500, usage=Usage(tokens_in=4096)) > 0


def test_shortfall_tolerates_tokenizer_drift() -> None:
    """A 4-chars-per-token heuristic is routinely off; that must not be
    mistaken for truncation."""
    assert truncation_shortfall(estimated=5000, usage=Usage(tokens_in=4400)) == 0


def test_shortfall_ignores_small_prompts() -> None:
    """Short prompts are where the estimate is least reliable, and where a
    large *proportional* gap means nothing."""
    assert truncation_shortfall(estimated=300, usage=Usage(tokens_in=40)) == 0


def test_shortfall_silent_when_provider_reports_no_usage() -> None:
    """Absence of telemetry is not evidence of truncation."""
    assert truncation_shortfall(estimated=9000, usage=Usage()) == 0
    assert truncation_shortfall(estimated=9000, usage=None) == 0


# ---------------------------------------------------------------------------
# Guard behavior
# ---------------------------------------------------------------------------


def test_guard_passes_healthy_replies_through() -> None:
    inner = _FakeClient(usage=Usage(tokens_in=2000, tokens_out=10))
    guard = ContextGuard(inner=inner)
    reply = guard.chat(messages=[_msg("x" * 8000)])
    assert reply.text == "ok"


def test_guard_raises_non_retryable_on_truncation() -> None:
    """Non-retryable: resending the same oversized prompt fails identically,
    so the loop must not burn its step budget on it."""
    inner = _FakeClient(usage=Usage(tokens_in=4096, tokens_out=10))
    guard = ContextGuard(inner=inner)
    with pytest.raises(ProviderError) as exc:
        guard.chat(messages=[_msg("x")], tools=_tools(30), system="s" * 2000)
    assert exc.value.retryable is False
    assert "truncated" in str(exc.value)


def test_guard_error_names_the_tool_count() -> None:
    """The message has to point at the cause — attached tools — because the
    user's lever is removing MCP servers, not editing their question."""
    inner = _FakeClient(usage=Usage(tokens_in=4096))
    guard = ContextGuard(inner=inner)
    with pytest.raises(ProviderError) as exc:
        guard.chat(messages=[_msg("x")], tools=_tools(30))
    assert "30 tool schema(s)" in str(exc.value)


def test_guard_non_strict_warns_instead_of_raising() -> None:
    warnings: list[str] = []
    inner = _FakeClient(usage=Usage(tokens_in=4096))
    guard = ContextGuard(inner=inner, strict=False, warn=warnings.append)
    reply = guard.chat(messages=[_msg("x")], tools=_tools(30))
    assert reply.text == "ok"
    assert warnings and "truncated" in warnings[0]


def test_guard_is_transparent_to_adapter_attributes() -> None:
    """Callers reach for `.model`, `.provider`, `.base_url` on the client;
    wrapping must not break them."""
    inner = _FakeClient(provider="ollama")
    inner.base_url = "http://localhost:11434"
    guard = ContextGuard(inner=inner)
    assert guard.provider == "ollama"
    assert guard.base_url == "http://localhost:11434"


def test_guard_forwards_all_chat_kwargs() -> None:
    inner = _FakeClient(usage=Usage(tokens_in=500))
    guard = ContextGuard(inner=inner)
    guard.chat(messages=[_msg("hi")], temperature=0.3, max_tokens=99,
               system="sys", extra={"k": "v"})
    assert inner.seen["temperature"] == 0.3
    assert inner.seen["max_tokens"] == 99
    assert inner.seen["extra"] == {"k": "v"}


def test_guard_detects_plateaued_input_count_across_turns() -> None:
    """The signature of a clamped window over a running loop: the prompt
    grows every turn but the provider's input count stops moving. Caught
    even when a single turn's shortfall looks within tolerance."""
    inner = _FakeClient(usage=Usage(tokens_in=8192))
    guard = ContextGuard(inner=inner, strict=False, warn=lambda _m: None)
    guard.chat(messages=[_msg("x" * 34000)])          # ~8.5k est, looks fine
    warnings: list[str] = []
    guard.warn = warnings.append
    guard.chat(messages=[_msg("x" * 60000)])          # ~15k est, same count back
    assert warnings and "truncated" in warnings[0]


def test_guard_allows_naturally_growing_conversation() -> None:
    """Input counts that track the prompt upward are healthy and must not
    trip the plateau check."""
    inner = _FakeClient()
    guard = ContextGuard(inner=inner)
    for chars in (4000, 8000, 16000, 32000):
        inner.usage = Usage(tokens_in=chars // 4)
        guard.chat(messages=[_msg("x" * chars)])
