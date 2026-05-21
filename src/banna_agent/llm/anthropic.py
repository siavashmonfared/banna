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

    provider: str = field(default="anthropic", init=False)

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
            kwargs["system"] = sys_prompt
        if temperature is not None:
            kwargs["temperature"] = temperature
        if tools:
            kwargs["tools"] = [_toolspec_to_anthropic(t) for t in tools]
        if extra:
            kwargs.update(extra)

        resp = client.messages.create(**kwargs)
        reply = _anthropic_response_to_reply(resp, model_id)
        # Keep provider name in reply consistent with transport (anthropic / bedrock).
        if self.transport == "bedrock":
            reply.provider = "bedrock"
        return reply
