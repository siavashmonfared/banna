"""Provider-agnostic LLM client contract.

Design notes (see collab thread `from_scratch_agent` Q1–Q5 for rationale):

1. `ContentBlock` is the primary payload. `LLMReply.text` and
   `LLMReply.tool_calls` are *projections*, not primary fields. This is
   because OpenAI o-series `reasoning`, Anthropic `thinking` blocks with
   signatures, and Gemini `thoughtSignature` must round-trip in order — a
   flat `(text, tool_calls)` pair is dishonest about all three.

2. Every `ContentBlock` and `Message` carries `raw`. Same-provider round
   trips preserve the exact wire form (thinking signatures, grounding
   metadata, etc). Cross-provider swap (the ablation matrix) degrades
   through normalized semantics only.

3. `Usage` is rich — cache, reasoning, thoughts, and tool-prompt tokens
   are separate columns. `Budget` only reads the flat `(tokens_in,
   tokens_out)` projection. The ablation table needs the full shape for a
   fair cost comparison across providers.

4. `LLMClient` is a `Protocol`, not a base class. Adapters are free to
   use SDKs directly; they only have to satisfy `.chat(...)`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence, runtime_checkable

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

Role = Literal["system", "user", "assistant", "tool"]

# Every content block every provider can emit collapses into one of these.
# `unknown` is a deliberate escape hatch so an adapter can emit blocks it
# doesn't yet know how to classify — the `raw` field still carries detail.
BlockKind = Literal[
    "text",
    "tool_use",
    "tool_result",
    "thinking",
    "reasoning",
    "unknown",
]

# Stop reason strings are *not* an Enum because providers add new ones all
# the time. Adapters should normalize common ones ("end_turn", "tool_use",
# "max_tokens", "stop_sequence", "refusal") and pass through the rest.


# ---------------------------------------------------------------------------
# Tool declarations (what we send to the model)
# ---------------------------------------------------------------------------


@dataclass
class ToolSpec:
    """A tool declaration advertised to the model.

    Mirrors the shape of `banna/engine/agents_lib/protocol.py::JsonTool`
    minus the executable `handler` — this is the public-facing declaration
    only. The executable tool lives in `banna_agent.tools.base.JsonTool`.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallRequest:
    """A tool call requested by the model on its last turn.

    Produced by `LLMReply.tool_calls` projection. The `id` must be echoed
    back on the corresponding `tool_result` ContentBlock or the conversation
    is malformed.
    """

    id: str
    name: str
    arguments: dict[str, Any]
    raw: Any = None


# ---------------------------------------------------------------------------
# Content blocks and messages (what we round-trip)
# ---------------------------------------------------------------------------


@dataclass
class ContentBlock:
    """One ordered piece of a message.

    Most fields are optional — which fields are populated depends on `kind`:
      - "text":        text
      - "tool_use":    id, name, arguments
      - "tool_result": id, result, is_error
      - "thinking":    text (Anthropic's extended thinking)
      - "reasoning":   text (OpenAI o-series reasoning summaries)
      - "unknown":     nothing specific; detail lives in `raw`

    `raw` always carries the provider-native block for same-provider replay.
    """

    kind: BlockKind
    text: str | None = None
    id: str | None = None
    name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    is_error: bool = False
    raw: Any = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """One turn in a conversation.

    The invariant: `content` is an *ordered* list. Adapters must preserve
    order when serializing to provider wire format, otherwise thinking
    blocks fall out of place and Anthropic rejects the request.
    """

    role: Role
    content: list[ContentBlock]
    raw: Any = None
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Usage telemetry
# ---------------------------------------------------------------------------


@dataclass
class Usage:
    """Rich per-call telemetry.

    The flat `(tokens_in, tokens_out)` pair is what `AgentState.budget`
    consumes. The rest of the columns feed the cost-normalized ablation
    table and let us compare o-series reasoning vs Anthropic caching vs
    Gemini thoughts fairly.
    """

    tokens_in: int = 0
    tokens_out: int = 0
    total_tokens: int = 0

    # Anthropic prompt caching
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    # OpenAI o-series (included in tokens_out but also exposed separately)
    reasoning_tokens: int = 0

    # Gemini thought traces
    thoughts_tokens: int = 0

    # Gemini's toolUsePromptTokenCount
    tool_prompt_tokens: int = 0

    cost_usd: float | None = None
    raw: Any = None


# ---------------------------------------------------------------------------
# The reply
# ---------------------------------------------------------------------------


@dataclass
class LLMReply:
    """The normalized response from `LLMClient.chat(...)`.

    `content` is the primary payload. `text` and `tool_calls` are
    convenience projections — they are *not* the source of truth, so
    don't rebuild requests from them alone; re-echo the full content list
    on the next assistant turn.
    """

    provider: str
    model: str
    content: list[ContentBlock]
    stop_reason: str
    usage: Usage = field(default_factory=Usage)
    raw: Any = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Concatenate all text blocks in order."""
        return "".join(b.text or "" for b in self.content if b.kind == "text")

    @property
    def tool_calls(self) -> list[ToolCallRequest]:
        """Extract tool_use blocks as ToolCallRequests, preserving order."""
        return [
            ToolCallRequest(
                id=b.id or "",
                name=b.name or "",
                arguments=dict(b.arguments),
                raw=b.raw,
            )
            for b in self.content
            if b.kind == "tool_use"
        ]

    @property
    def has_tool_calls(self) -> bool:
        return any(b.kind == "tool_use" for b in self.content)


# ---------------------------------------------------------------------------
# The protocol every adapter satisfies
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMClient(Protocol):
    """Minimal contract every provider adapter must satisfy.

    Intentionally synchronous. Async is added only when a concrete
    bottleneck forces it (per the project non-goals).
    """

    provider: str

    def chat(
        self,
        *,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] = (),
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        system: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> LLMReply:
        """Send a conversation, get a reply. All kwargs are keyword-only.

        `extra` is provider-specific passthrough — e.g. Anthropic's
        `cache_control` or Gemini's `tool_config`. Adapters may ignore
        unknown keys in `extra`.
        """
        ...
