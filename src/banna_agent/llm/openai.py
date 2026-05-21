"""OpenAI adapter for the normalized LLMClient.

Targets Chat Completions in v1 (stable, broad model coverage including
o-series via `reasoning_effort`). For 2026 o-series long tool loops you
may want to migrate to the Responses API; this adapter keeps the option
open via the `extra` kwarg.

Normalization notes:
- Assistant `message.tool_calls[i].function.arguments` is a JSON string;
  we `json.loads` it into ContentBlock.arguments.
- Tool results go on a `role="tool"` message with `tool_call_id`. Our
  normalized `tool_result` blocks live in a user Message in history;
  the serializer demotes them to a separate `role="tool"` message.
- `reasoning_tokens` (o-series) come from `usage.completion_tokens_details.reasoning_tokens`.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Sequence

from .base import ContentBlock, LLMReply, Message, ToolSpec, Usage


# ---------------------------------------------------------------------------
# Request serialization
# ---------------------------------------------------------------------------


def _messages_to_openai(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Flatten our normalized messages into OpenAI Chat format.

    A user Message containing `tool_result` blocks gets split: each
    tool_result becomes its own `role="tool"` message. Pure-text user or
    assistant turns collapse to `content: str`. Assistant turns with
    tool_use blocks emit `tool_calls` with string-encoded arguments.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        tool_results = [b for b in msg.content if b.kind == "tool_result"]
        other = [b for b in msg.content if b.kind != "tool_result"]

        if msg.role == "assistant":
            text_parts = [b.text or "" for b in other if b.kind == "text"]
            tool_uses = [b for b in other if b.kind == "tool_use"]
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": "".join(text_parts) if text_parts else None,
            }
            if tool_uses:
                assistant_msg["tool_calls"] = [
                    {
                        "id": b.id or "",
                        "type": "function",
                        "function": {
                            "name": b.name or "",
                            "arguments": json.dumps(b.arguments, default=str),
                        },
                    }
                    for b in tool_uses
                ]
            out.append(assistant_msg)

        elif msg.role in ("user", "tool"):
            # User turn with text only
            text_parts = [b.text or "" for b in other if b.kind == "text"]
            if text_parts:
                out.append({"role": "user", "content": "".join(text_parts)})
            for b in tool_results:
                content = b.result if isinstance(b.result, str) else json.dumps(b.result, default=str)
                out.append({
                    "role": "tool",
                    "tool_call_id": b.id or "",
                    "content": content,
                })
        elif msg.role == "system":
            text_parts = [b.text or "" for b in msg.content if b.kind == "text"]
            out.append({"role": "system", "content": "".join(text_parts)})
    return out


def _toolspec_to_openai(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.input_schema,
        },
    }


# ---------------------------------------------------------------------------
# Response deserialization
# ---------------------------------------------------------------------------


def _getter(obj: Any):
    if isinstance(obj, dict):
        return lambda k, _obj=obj: _obj.get(k)
    return lambda k, _obj=obj: getattr(_obj, k, None)


def _response_to_reply(resp: Any, model: str) -> LLMReply:
    g = _getter(resp)
    choices = g("choices") or []
    if not choices:
        return LLMReply(provider="openai", model=model, content=[], stop_reason="end_turn", raw=resp)
    ch0 = choices[0]
    gch = _getter(ch0)
    msg = gch("message")
    gmsg = _getter(msg)
    text = gmsg("content") or ""
    tool_calls_raw = gmsg("tool_calls") or []
    finish = gch("finish_reason") or "stop"

    blocks: list[ContentBlock] = []
    if text:
        blocks.append(ContentBlock(kind="text", text=text))
    for tc in tool_calls_raw:
        gtc = _getter(tc)
        fn = gtc("function")
        gfn = _getter(fn)
        args_str = gfn("arguments") or "{}"
        try:
            args = json.loads(args_str)
        except (ValueError, TypeError):
            args = {"_raw": args_str}
        blocks.append(ContentBlock(
            kind="tool_use",
            id=gtc("id") or "",
            name=gfn("name") or "",
            arguments=args,
            raw=tc,
        ))

    # Map OpenAI finish_reason to our normalized stop_reason.
    stop_map = {
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "length": "max_tokens",
        "content_filter": "refusal",
        "function_call": "tool_use",  # legacy
    }
    stop_reason = stop_map.get(finish, finish)

    # Usage
    raw_usage = g("usage")
    if raw_usage is None:
        usage = Usage()
    else:
        gu = _getter(raw_usage)
        prompt_tok = int(gu("prompt_tokens") or 0)
        completion_tok = int(gu("completion_tokens") or 0)
        details = gu("completion_tokens_details")
        reasoning_tok = 0
        if details is not None:
            gd = _getter(details)
            reasoning_tok = int(gd("reasoning_tokens") or 0)
        cached_tok = 0
        prompt_details = gu("prompt_tokens_details")
        if prompt_details is not None:
            gpd = _getter(prompt_details)
            cached_tok = int(gpd("cached_tokens") or 0)
        usage = Usage(
            tokens_in=prompt_tok,
            tokens_out=completion_tok,
            total_tokens=prompt_tok + completion_tok,
            cache_read_tokens=cached_tok,
            reasoning_tokens=reasoning_tok,
            raw=raw_usage,
        )

    return LLMReply(
        provider="openai",
        model=g("model") or model,
        content=blocks,
        stop_reason=stop_reason,
        usage=usage,
        raw=resp,
    )


# ---------------------------------------------------------------------------
# Model-shape predicates
#
# OpenAI changed the request schema across model generations:
#   * gpt-3.5 / gpt-4 / gpt-4-turbo: `max_tokens`, `temperature` ok.
#   * gpt-4.1+, gpt-5*, o*-series:   `max_completion_tokens` only.
#   * o-series specifically:         rejects custom `temperature`.
# We detect by model id prefix to keep older models working unchanged.
# ---------------------------------------------------------------------------


def _is_o_series(model_id: str) -> bool:
    m = (model_id or "").lower()
    return m.startswith("o1") or m.startswith("o3") or m.startswith("o4")


def _uses_max_completion_tokens(model_id: str) -> bool:
    m = (model_id or "").lower()
    return (
        m.startswith("gpt-5")
        or m.startswith("gpt-4.1")
        or _is_o_series(m)
    )


# Same set today (gpt-5* / gpt-4.1 / o-series all reject custom
# temperature). Kept as a separate predicate so it's easy to tighten or
# loosen independently if OpenAI splits the constraints later.
def _rejects_custom_temperature(model_id: str) -> bool:
    return _uses_max_completion_tokens(model_id)


# Reasoning models (gpt-5*, o-series, gpt-4.1+) emit invisible chain-of-
# thought tokens that count against `max_completion_tokens`. If a policy
# asks for, say, 600 tokens and the model needs 800 for reasoning, the
# visible response is empty and the agent silently loops on
# "model returned no text and no tool_calls" for the rest of the budget.
#
# Floor the cap so reasoning has somewhere to live. 4096 is enough for
# tool-pick agent decisions plus a sentence or two of visible output;
# tune via OPENAI_REASONING_MIN_TOKENS for harder synthesis tasks.
_REASONING_MIN_TOKENS_DEFAULT = 4096


def _reasoning_min_tokens() -> int:
    raw = os.environ.get("OPENAI_REASONING_MIN_TOKENS")
    if raw:
        try:
            v = int(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    return _REASONING_MIN_TOKENS_DEFAULT


def _floor_for_reasoning(model_id: str, requested: int) -> int:
    """Apply the reasoning-model token floor.

    For non-reasoning models, returns `requested` unchanged. For
    reasoning models, returns ``max(requested, floor)``.
    """
    if not _uses_max_completion_tokens(model_id):
        return requested
    return max(requested, _reasoning_min_tokens())


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@dataclass
class OpenAIClient:
    model: str = "gpt-5-mini"
    api_key: str | None = None
    base_url: str | None = None
    organization: str | None = None
    system_default: str | None = None
    sdk: Any = None

    provider: str = field(default="openai", init=False)

    def _client(self) -> Any:
        if self.sdk is not None:
            return self.sdk
        import openai  # lazy
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            from .base import ProviderError
            raise ProviderError("OPENAI_API_KEY not set.", retryable=False)
        kwargs: dict[str, Any] = {"api_key": key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.organization:
            kwargs["organization"] = self.organization
        self.sdk = openai.OpenAI(**kwargs)
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

        openai_msgs = _messages_to_openai(messages)
        if sys_prompt and not any(m["role"] == "system" for m in openai_msgs):
            openai_msgs = [{"role": "system", "content": sys_prompt}] + openai_msgs

        kwargs: dict[str, Any] = {"model": model_id, "messages": openai_msgs}
        if max_tokens is not None:
            # gpt-5 / o-series / gpt-4.1+ require `max_completion_tokens`
            # and reject the legacy `max_tokens`. They also count
            # invisible reasoning tokens against that budget — so we
            # apply a floor (default 4096) to leave headroom for the
            # actual visible response. Detect by model prefix so older
            # gpt-4 / gpt-3.5 keep working unchanged with no floor.
            if _uses_max_completion_tokens(model_id):
                kwargs["max_completion_tokens"] = _floor_for_reasoning(
                    model_id, max_tokens,
                )
            else:
                kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            # gpt-5* (including nano), gpt-4.1+, and o-series all reject
            # custom temperature — only the default (1.0) is supported.
            # Skip the parameter rather than forcing 1.0 so behavior is
            # predictable. Older gpt-3.5 / gpt-4 still honor it.
            if not _rejects_custom_temperature(model_id):
                kwargs["temperature"] = temperature
        if tools:
            kwargs["tools"] = [_toolspec_to_openai(t) for t in tools]
        if extra:
            kwargs.update(extra)

        resp = client.chat.completions.create(**kwargs)
        return _response_to_reply(resp, model_id)
