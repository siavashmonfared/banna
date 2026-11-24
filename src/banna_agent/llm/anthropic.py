"""Anthropic direct-API adapter for the normalized `LLMClient` contract.

Uses the `anthropic` SDK against the public API (not Bedrock). Bedrock is
straightforward to add as a second code path — the serialization is
identical; only the client construction differs.

Things that are easy to get wrong (from collab thread Q3):

1. `tool_result` blocks go on a **user** turn following the assistant's
   `tool_use` turn. Attach them as the *first* content blocks of that user
   turn; any free text comes after.
2. Don't drop/reorder `thinking` blocks — if the assistant produced any
   and extended thinking is enabled, they must be echoed back on the next
   request in their original order.
3. `stop_reason ∈ {end_turn, tool_use, max_tokens, stop_sequence,
   pause_turn, refusal}`. Adapter maps these through verbatim.
4. `usage` exposes `input_tokens`, `output_tokens`,
   `cache_creation_input_tokens`, `cache_read_input_tokens`. All four
   matter for the ablation table.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any, Sequence

from .base import (
    ContentBlock,
    LLMReply,
    Message,
    ToolSpec,
    Usage,
)


# ---------------------------------------------------------------------------
# Model-shape predicates
#
# Anthropic removed the sampling parameters (`temperature`, `top_p`,
# `top_k`) on its newest models: Fable 5, Sonnet 5, Opus 4.7, and Opus
# 4.8 reject any of them with a 400 ("temperature is deprecated for this
# model"). They are still accepted on Opus 4.6, Sonnet 4.6, Opus 4.5,
# Sonnet 4.5, Haiku 4.5, and older. This is a per-version split (Opus 4.6
# accepts, 4.7 rejects; Sonnet 4.6 accepts, Sonnet 5 rejects), so we match
# a known-rejecting set rather than a prefix. Anything not matched here is
# caught at runtime by the 400-retry fallback in `chat()` and remembered
# for the rest of the session, so an unlisted future model self-heals.
# ---------------------------------------------------------------------------

_REJECTS_SAMPLING_EXACT = frozenset({
    "claude-fable-5",
    "claude-mythos-5",
    "claude-sonnet-5",
})

# Prefixes for families where every published point release rejects
# sampling (opus-4-7 and opus-4-8 and anything newer in that line).
_REJECTS_SAMPLING_PREFIXES = (
    "claude-opus-4-7",
    "claude-opus-4-8",
)


def _rejects_sampling_params(model_id: str) -> bool:
    """True if the model 400s on temperature / top_p / top_k."""
    m = (model_id or "").lower()
    # Strip a bedrock `anthropic.` prefix if present.
    if m.startswith("anthropic."):
        m = m[len("anthropic."):]
    if m in _REJECTS_SAMPLING_EXACT:
        return True
    return any(m.startswith(p) for p in _REJECTS_SAMPLING_PREFIXES)


def _is_sampling_400(exc: Exception) -> bool:
    """True if a raised error is a 400 complaining about a sampling param."""
    if getattr(exc, "status_code", None) not in (400, None):
        return False
    msg = str(exc).lower()
    return (
        ("temperature" in msg or "top_p" in msg or "top_k" in msg)
        and ("deprecated" in msg or "not supported" in msg
             or "unsupported" in msg or "unexpected" in msg
             or "removed" in msg)
    )


# ---------------------------------------------------------------------------
# Serialization helpers: normalized → Anthropic wire format
# ---------------------------------------------------------------------------


def _block_to_anthropic(block: ContentBlock) -> dict[str, Any]:
    """One ContentBlock → one Anthropic content dict.

    If the block has a `raw` dict (same-provider replay), prefer it —
    that preserves thinking signatures and other round-trip-only fields.
    """
    if isinstance(block.raw, dict):
        return dict(block.raw)

    if block.kind == "text":
        return {"type": "text", "text": block.text or ""}

    if block.kind == "tool_use":
        return {
            "type": "tool_use",
            "id": block.id or "",
            "name": block.name or "",
            "input": dict(block.arguments),
        }

    if block.kind == "tool_result":
        out: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": block.id or "",
        }
        # Anthropic accepts `content` as either a string or a list of
        # content blocks (for images, etc). We default to string JSON so
        # the return dicts from sync tools round-trip cleanly.
        if isinstance(block.result, str):
            out["content"] = block.result
        elif block.result is None:
            out["content"] = ""
        else:
            import json
            out["content"] = json.dumps(block.result, default=str)
        if block.is_error:
            out["is_error"] = True
        return out

    if block.kind == "thinking":
        # Thinking blocks require a signature on round-trip; without raw we
        # can't reconstruct it, so we emit a text block as a best effort.
        # In practice `raw` is always populated for thinking blocks from a
        # prior assistant turn.
        return {"type": "text", "text": block.text or ""}

    # "reasoning", "unknown" — fall back to raw or text
    if block.text:
        return {"type": "text", "text": block.text}
    return {"type": "text", "text": ""}


def _message_to_anthropic(msg: Message) -> dict[str, Any]:
    """One normalized Message → one Anthropic messages-API dict."""
    # Anthropic uses "user" / "assistant"; "system" is a top-level field;
    # "tool" role isn't used (tool results ride on user messages).
    role = msg.role
    if role == "tool":
        role = "user"
    content = [_block_to_anthropic(b) for b in msg.content]
    return {"role": role, "content": content}


_MAX_CACHE_BREAKPOINTS = 4
_NOT_CACHEABLE = {"thinking", "redacted_thinking"}


def _count_breakpoints(value: Any) -> int:
    if isinstance(value, dict):
        return ("cache_control" in value) + sum(_count_breakpoints(v) for v in value.values())
    if isinstance(value, list):
        return sum(_count_breakpoints(v) for v in value)
    return 0


def _mark_message(message: dict[str, Any]) -> bool:
    """Put cache_control on the last cacheable block of `message` (in place)."""
    content = message.get("content")
    if isinstance(content, str):
        message["content"] = content = [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return False
    for i in range(len(content) - 1, -1, -1):
        block = content[i]
        if isinstance(block, dict) and block.get("type") not in _NOT_CACHEABLE:
            if "cache_control" not in block:
                content[i] = {**block, "cache_control": {"type": "ephemeral"}}
            return True
    return False


def _has_tool_result(message: dict[str, Any]) -> bool:
    content = message.get("content")
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content)


def _add_conversation_breakpoints(messages: list[dict[str, Any]], used: int) -> None:
    """Cache the growing conversation, not only tools+system.

    The system-only breakpoint is often below the model's minimum cacheable
    prefix, and it never covers the conversation, so every step re-processed
    the whole history uncached. Two breakpoints fix that:

    * newest: the last message. When history is replayed verbatim
      (project_history=False) the next request extends this prefix unchanged.
    * stable: the message before the newest tool result (the assistant
      tool_use turn). With history projection (the default) the newest tool
      result is shortened on the next step, so the newest breakpoint is not
      reusable; this message, however, is replayed byte-for-byte, and the next
      request's own stable breakpoint sits a few blocks later, within the
      API's 20-block lookback, so it reads this prefix from cache.

    Operates on serialized dicts only; caller Message objects are untouched.
    Thinking blocks cannot carry cache_control. At most 4 breakpoints total.
    """
    if not messages:
        return
    budget = _MAX_CACHE_BREAKPOINTS - used
    if budget <= 0:
        return
    if _mark_message(messages[-1]):
        budget -= 1
    if budget <= 0:
        return
    last_result = next((i for i in range(len(messages) - 1, -1, -1) if _has_tool_result(messages[i])), None)
    if last_result is not None and 0 <= last_result - 1 < len(messages) - 1:
        _mark_message(messages[last_result - 1])


def _toolspec_to_anthropic(spec: ToolSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "description": spec.description,
        "input_schema": spec.input_schema,
    }


# ---------------------------------------------------------------------------
# Deserialization: Anthropic response → normalized LLMReply
# ---------------------------------------------------------------------------


def _anthropic_block_to_contentblock(raw_block: Any) -> ContentBlock:
    """Duck-typed: accepts SDK objects or plain dicts (tests use dicts).

    We mirror what `engine/agents_lib/runners/_bedrock_loop.py::_partition_blocks`
    does but keep the full block content including `thinking`.
    """
    get = _getter(raw_block)
    kind = get("type")

    if kind == "text":
        return ContentBlock(kind="text", text=get("text") or "", raw=raw_block)

    if kind == "tool_use":
        return ContentBlock(
            kind="tool_use",
            id=get("id") or "",
            name=get("name") or "",
            arguments=dict(get("input") or {}),
            raw=raw_block,
        )

    if kind == "thinking":
        return ContentBlock(
            kind="thinking",
            text=get("thinking") or get("text") or "",
            raw=raw_block,
        )

    return ContentBlock(kind="unknown", raw=raw_block)


def _getter(obj: Any):
    """Return a callable that pulls a field from either a dict or an SDK object."""
    if isinstance(obj, dict):
        return lambda k, _obj=obj: _obj.get(k)
    return lambda k, _obj=obj: getattr(_obj, k, None)


def _extract_usage(resp: Any) -> Usage:
    g = _getter(resp)
    raw_usage = g("usage")
    if raw_usage is None:
        return Usage()
    gu = _getter(raw_usage)
    tokens_in = int(gu("input_tokens") or 0)
    tokens_out = int(gu("output_tokens") or 0)
    cache_write = int(gu("cache_creation_input_tokens") or 0)
    cache_read = int(gu("cache_read_input_tokens") or 0)
    return Usage(
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        total_tokens=tokens_in + tokens_out,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
        raw=raw_usage,
    )


def _anthropic_response_to_reply(resp: Any, model: str) -> LLMReply:
    g = _getter(resp)
    content_blocks_raw = g("content") or []
    blocks = [_anthropic_block_to_contentblock(b) for b in content_blocks_raw]
    stop_reason = g("stop_reason") or "end_turn"
    used_model = g("model") or model
    usage = _extract_usage(resp)
    return LLMReply(
        provider="anthropic",
        model=used_model,
        content=blocks,
        stop_reason=stop_reason,
        usage=usage,
        raw=resp,
    )


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


@dataclass
class AnthropicClient:
    """Adapter over the `anthropic` SDK implementing `LLMClient`.

    Two transports:
      transport='direct'  -> `anthropic.Anthropic(api_key=...)`  (default)
      transport='bedrock' -> `anthropic.AnthropicBedrock(aws_region=...)`

    Both transports use identical message serialization; only the SDK
    client differs. This is the key reason to mirror Anthropic's API
    semantics precisely — Bedrock is byte-compatible once authenticated.

    For tests, pass a mock SDK via `sdk=`.
    """

    model: str = "claude-opus-4-5-20251101"
    api_key: str | None = None
    system_default: str | None = None
    sdk: Any = None  # injected for testing; default: instantiate real SDK on first call
    transport: str = "direct"  # "direct" | "bedrock"
    aws_region: str | None = None
    aws_profile: str | None = None
    base_url: str | None = None
    # Prompt caching on the stable prefix. When True (default), mark the
    # system block with `cache_control: ephemeral`. Anthropic's cache order
    # is tools → system → messages, and a breakpoint caches everything up to
    # and including it, so one marker on `system` caches the whole
    # tools+system prefix — identical on every tick of an agent loop, so
    # subsequent calls read it at ~10% of input price. No-op when no system
    # prompt is set, or when the prefix is below the model's minimum
    # cacheable size (the API silently ignores it). Set False to disable
    # (ablation lever / belt-and-suspenders for unusual prefixes).
    prompt_cache: bool = True

    provider: str = field(default="anthropic", init=False)
    # Model ids discovered at runtime to reject sampling params (via a
    # 400). Seeded empty; a caught sampling-400 adds the model here so the
    # next call skips temperature instead of failing again.
    _no_sampling_seen: set = field(default_factory=set, init=False, repr=False)

    def _client(self) -> Any:
        if self.sdk is not None:
            return self.sdk
        # Lazy import so the test suite doesn't need the SDK at import time.
        import anthropic

        if self.transport == "bedrock":
            region = (
                self.aws_region
                or os.environ.get("AWS_REGION")
                or os.environ.get("AWS_DEFAULT_REGION")
            )
            if not region:
                from .base import ProviderError
                raise ProviderError(
                    "AWS_REGION not set. Either pass aws_region= when "
                    "building the client, or `export AWS_REGION=us-east-1` "
                    "(or us-west-2, eu-west-1, etc.) in your shell.\n"
                    "Bedrock model IDs prefixed `us.` work only when the "
                    "region is in the US partition; `eu.` for EU regions; "
                    "`apac.` for APAC.",
                    retryable=False,
                )
            # Surface a hint when AWS credentials are clearly absent. The
            # AWS SDK will eventually raise NoCredentialsError; surfacing
            # it up front saves a confusing stack trace from boto3.
            has_keys = bool(
                os.environ.get("AWS_ACCESS_KEY_ID")
                and os.environ.get("AWS_SECRET_ACCESS_KEY")
            )
            has_profile = bool(self.aws_profile or os.environ.get("AWS_PROFILE"))
            if not (has_keys or has_profile):
                # Not fatal — a configured ~/.aws/credentials default
                # profile, or an IMDS-attached IAM role, still works.
                # Just emit a soft hint so first-time users have a clue.
                import warnings
                warnings.warn(
                    "Bedrock transport selected but neither "
                    "AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY nor "
                    "AWS_PROFILE is set. Falling back to default "
                    "credential chain (~/.aws/credentials, IAM role, "
                    "etc.). If the call fails with NoCredentialsError, "
                    "set keys in your shell or pass aws_profile=.",
                    stacklevel=2,
                )
            kwargs: dict[str, Any] = {"aws_region": region}
            if self.aws_profile:
                kwargs["aws_profile"] = self.aws_profile
            if self.base_url:
                kwargs["base_url"] = self.base_url
            try:
                self.sdk = anthropic.AnthropicBedrock(**kwargs)
            except ModuleNotFoundError as exc:
                # The Anthropic SDK's Bedrock client lazy-imports boto3 /
                # botocore. If those aren't installed, the message we
                # normally see is `No module named 'botocore'` — clear
                # to AWS folks, confusing to everyone else. Raise a
                # message that names the install fix directly.
                missing = exc.name or "boto3"
                raise RuntimeError(
                    f"Bedrock transport requires AWS SDK ({missing} missing).\n"
                    "  Install: `pip install boto3` "
                    "(or `pip install \"anthropic[bedrock]\"`)."
                ) from exc
            return self.sdk

        if self.transport != "direct":
            raise ValueError(f"unknown transport {self.transport!r}")

        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            from .base import ProviderError
            raise ProviderError(
                "ANTHROPIC_API_KEY not set. Pass api_key= or set the env var.",
                retryable=False,
            )
        kwargs = {"api_key": key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self.sdk = anthropic.Anthropic(**kwargs)
        return self.sdk

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
        client = self._client()
        model_id = model or self.model
        sys_prompt = system if system is not None else self.system_default

        kwargs: dict[str, Any] = {
            "model": model_id,
            "max_tokens": max_tokens if max_tokens is not None else 1024,
            "messages": [_message_to_anthropic(m) for m in messages],
        }
        if sys_prompt:
            if self.prompt_cache:
                # Structured system form so we can attach a cache breakpoint.
                # Caches tools+system (tools precede system in cache order).
                kwargs["system"] = [{
                    "type": "text",
                    "text": sys_prompt,
                    "cache_control": {"type": "ephemeral"},
                }]
            else:
                kwargs["system"] = sys_prompt
        # Newer Anthropic models (Fable 5 / Sonnet 5 / Opus 4.7+) reject
        # `temperature` with a 400. Omit it for the known set and for any
        # model a prior call in this session revealed to reject it.
        send_temperature = (
            temperature is not None
            and not _rejects_sampling_params(model_id)
            and model_id not in self._no_sampling_seen
        )
        if send_temperature:
            kwargs["temperature"] = temperature
        if tools:
            kwargs["tools"] = [_toolspec_to_anthropic(t) for t in tools]
        if extra:
            kwargs.update(extra)
        if self.prompt_cache:
            kwargs["messages"] = copy.deepcopy(kwargs["messages"])
            used = _count_breakpoints([kwargs.get("system"), kwargs.get("tools"), kwargs["messages"]])
            _add_conversation_breakpoints(kwargs["messages"], used)

        try:
            resp = client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 — re-classified below
            # Self-heal: a model we didn't know about rejects sampling
            # params. Remember it, drop temperature, and retry once so the
            # user gets an answer instead of a 400 wall.
            if _is_sampling_400(exc) and "temperature" in kwargs:
                self._no_sampling_seen.add(model_id)
                kwargs.pop("temperature", None)
                resp = client.messages.create(**kwargs)
                reply = _anthropic_response_to_reply(resp, model_id)
                if self.transport == "bedrock":
                    reply.provider = "bedrock"
                return reply
            from .base import ProviderError
            # Rate limits (429) and transient server errors (5xx / 529
            # "overloaded") are worth a backoff-and-retry; the driver
            # owns that loop and pauses the wall clock while it waits, so
            # a slow (low-tier) account doesn't burn the task's time
            # budget. We surface them as a *retryable* ProviderError.
            # Deterministic failures (bad request, auth) are not retried.
            name = type(exc).__name__
            status = getattr(exc, "status_code", None)
            retryable = (
                name in {"RateLimitError", "InternalServerError",
                         "APIConnectionError", "APITimeoutError",
                         "OverloadedError"}
                or status in {408, 409, 429, 500, 502, 503, 504, 529}
            )
            if retryable:
                raise ProviderError(
                    f"{name}: {exc}", retryable=True,
                ) from exc
            raise
        reply = _anthropic_response_to_reply(resp, model_id)
        # Keep provider name in reply consistent with transport (anthropic / bedrock).
        if self.transport == "bedrock":
            reply.provider = "bedrock"
        return reply
