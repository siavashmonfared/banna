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
# Responses API (reasoning models with function tools)
#
# Chat Completions rejects function tools combined with `reasoning_effort`
# for current reasoning models ("use /v1/responses or set reasoning_effort to
# 'none'"). When a reasoning control is configured and the client talks to
# OpenAI itself, use the Responses API statelessly: store=False, ask for
# encrypted reasoning, and send each prior reasoning item back with its
# function call (block.raw carries the provider-native item).
# OpenAI-compatible endpoints (OpenRouter, proxies, ...) keep
# Chat Completions: their Responses support is absent or partial.
# ---------------------------------------------------------------------------

REASONING_CONTROLS = ("reasoning", "reasoning_effort")
OPENAI_HOSTS = ("api.openai.com",)


def _is_openai_endpoint(base_url: str | None) -> bool:
    """True when requests go to OpenAI itself (no base_url, or an OpenAI host)."""
    if not base_url:
        return True
    from urllib.parse import urlparse
    host = (urlparse(base_url).hostname or "").lower()
    return host in OPENAI_HOSTS

# Deterministic request errors: the same request fails the same way again.
_NON_RETRYABLE_STATUS = {400, 401, 403, 404, 422}


def _non_retryable(exc: Exception):
    from .base import ProviderError
    status = getattr(exc, "status_code", None)
    if status in _NON_RETRYABLE_STATUS:
        return ProviderError(f"{type(exc).__name__}: {exc}", retryable=False)
    return None


def _text_of(result: Any) -> str:
    return result if isinstance(result, str) else json.dumps(result, default=str)


def _messages_to_responses(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Normalized history -> Responses `input` items, preserving block order."""
    items: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "system":
            continue  # sent as `instructions`
        if msg.role == "assistant":
            for b in msg.content:
                if b.kind == "reasoning":
                    raw = b.raw if isinstance(b.raw, dict) else {}
                    if raw.get("encrypted_content"):
                        items.append({"type": "reasoning", "summary": raw.get("summary") or [],
                                      "encrypted_content": raw["encrypted_content"]})
                elif b.kind == "text" and b.text:
                    items.append({"role": "assistant", "content": b.text})
                elif b.kind == "tool_use":
                    items.append({"type": "function_call", "call_id": b.id or "", "name": b.name or "",
                                  "arguments": json.dumps(b.arguments, default=str)})
            continue
        for b in msg.content:
            if b.kind == "text" and b.text:
                items.append({"role": "user", "content": b.text})
            elif b.kind == "tool_result":
                items.append({"type": "function_call_output", "call_id": b.id or "",
                              "output": _text_of(b.result)})
    return items


def _toolspec_to_responses(spec: ToolSpec) -> dict[str, Any]:
    # strict=False keeps Banna's existing (non-strict) tool schemas valid.
    return {"type": "function", "name": spec.name, "description": spec.description,
            "parameters": spec.input_schema, "strict": False}


def _plain(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", exclude_none=True)
    return obj


def _responses_to_reply(resp: Any, model: str) -> LLMReply:
    data = _plain(resp) or {}
    blocks: list[ContentBlock] = []
    for item in data.get("output") or []:
        kind = item.get("type")
        if kind == "reasoning":
            summary = "".join(s.get("text", "") for s in item.get("summary") or [])
            blocks.append(ContentBlock(kind="reasoning", text=summary, raw=item))
        elif kind == "message":
            for part in item.get("content") or []:
                if part.get("type") == "output_text" and part.get("text"):
                    blocks.append(ContentBlock(kind="text", text=part["text"], raw=part))
        elif kind == "function_call":
            try:
                args = json.loads(item.get("arguments") or "{}")
            except (ValueError, TypeError):
                args = {"_raw": item.get("arguments")}
            blocks.append(ContentBlock(kind="tool_use", id=item.get("call_id") or "",
                                       name=item.get("name") or "", arguments=args, raw=item))
        else:
            blocks.append(ContentBlock(kind="unknown", raw=item))
    if any(b.kind == "tool_use" for b in blocks):
        stop_reason = "tool_use"
    elif data.get("status") == "incomplete":
        reason = (data.get("incomplete_details") or {}).get("reason")
        stop_reason = "max_tokens" if reason == "max_output_tokens" else (reason or "incomplete")
    else:
        stop_reason = "end_turn"
    u = data.get("usage") or {}
    tokens_in, tokens_out = int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)
    usage = Usage(tokens_in=tokens_in, tokens_out=tokens_out, total_tokens=tokens_in + tokens_out,
                  cache_read_tokens=int((u.get("input_tokens_details") or {}).get("cached_tokens") or 0),
                  reasoning_tokens=int((u.get("output_tokens_details") or {}).get("reasoning_tokens") or 0),
                  raw=u or None)
    return LLMReply(provider="openai", model=data.get("model") or model, content=blocks,
                    stop_reason=stop_reason, usage=usage, raw=resp)


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

    def _uses_responses_api(self, extra: dict[str, Any] | None) -> bool:
        if not any(key in (extra or {}) for key in REASONING_CONTROLS):
            return False
        return _is_openai_endpoint(self.base_url or os.environ.get("OPENAI_BASE_URL"))

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
        if self._uses_responses_api(extra):
            return self._responses(client, messages=messages, tools=tools, model_id=model_id,
                                   max_tokens=max_tokens, temperature=temperature,
                                   sys_prompt=sys_prompt, extra=dict(extra or {}))

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

        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as exc:
            mapped = _non_retryable(exc)
            if mapped is not None:
                raise mapped from exc
            raise
        return _response_to_reply(resp, model_id)

    def _responses(self, client: Any, *, messages, tools, model_id, max_tokens, temperature,
                   sys_prompt, extra) -> LLMReply:
        effort = extra.pop("reasoning_effort", None)
        reasoning = dict(extra.pop("reasoning", None) or {})
        if effort is not None:
            reasoning.setdefault("effort", effort)
        include = list(extra.pop("include", None) or [])
        if "reasoning.encrypted_content" not in include:
            include.append("reasoning.encrypted_content")
        system_texts = [b.text or "" for m in messages if m.role == "system" for b in m.content if b.kind == "text"]
        instructions = sys_prompt or ("".join(system_texts) or None)
        kwargs: dict[str, Any] = {"model": model_id, "input": _messages_to_responses(messages),
                                  "reasoning": reasoning, "store": False, "include": include}
        if instructions:
            kwargs["instructions"] = instructions
        if max_tokens is not None:
            kwargs["max_output_tokens"] = _floor_for_reasoning(model_id, max_tokens)
        if temperature is not None and not _rejects_custom_temperature(model_id):
            kwargs["temperature"] = temperature
        if tools:
            kwargs["tools"] = [_toolspec_to_responses(t) for t in tools]
        kwargs.update(extra)
        try:
            resp = client.responses.create(**kwargs)
        except Exception as exc:
            mapped = _non_retryable(exc)
            if mapped is not None:
                raise mapped from exc
            raise
        return _responses_to_reply(resp, model_id)
