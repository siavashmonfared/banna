"""Ollama local-server adapter via native `/api/chat` endpoint.

We deliberately target native `/api/chat` instead of the OpenAI-compat
endpoint because:

1. Native exposes `thinking`, `done_reason`, `prompt_eval_count`, and
   `eval_count` — the fields local models actually produce.
2. Tool-call IDs are frequently missing from local models; we synthesize
   stable IDs here so downstream round-trips don't break.

Shape (native chat):
  Request:  { model, messages: [{role, content, tool_calls?}], tools?, stream: false, options: {...} }
  Response: { message: { role, content, tool_calls? }, done_reason, prompt_eval_count, eval_count, ... }
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Sequence

from .base import ContentBlock, LLMReply, Message, ToolSpec, Usage


# ---------------------------------------------------------------------------
# Request serialization
# ---------------------------------------------------------------------------


def _messages_to_ollama(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Flatten normalized messages to Ollama's chat format.

    Ollama follows the OpenAI-style convention for tool_calls on the
    assistant message and `role='tool'` for results, with `tool_call_id`.
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "system":
            texts = [b.text or "" for b in msg.content if b.kind == "text"]
            out.append({"role": "system", "content": "".join(texts)})
            continue

        if msg.role == "assistant":
            texts = [b.text or "" for b in msg.content if b.kind == "text"]
            tool_uses = [b for b in msg.content if b.kind == "tool_use"]
            am: dict[str, Any] = {"role": "assistant", "content": "".join(texts)}
            if tool_uses:
                am["tool_calls"] = [
                    {
                        "id": b.id or _synthesize_id(),
                        "type": "function",
                        "function": {
                            "name": b.name or "",
                            "arguments": dict(b.arguments),
                        },
                    }
                    for b in tool_uses
                ]
            out.append(am)
            continue

        # user / tool
        text_blocks = [b for b in msg.content if b.kind == "text"]
        tool_results = [b for b in msg.content if b.kind == "tool_result"]
        if text_blocks:
            out.append({
                "role": "user",
                "content": "".join(b.text or "" for b in text_blocks),
            })
        for b in tool_results:
            content = b.result if isinstance(b.result, str) else _json_dumps(b.result)
            out.append({
                "role": "tool",
                "tool_call_id": b.id or "",
                "content": content,
            })
    return out


def _json_dumps(v: Any) -> str:
    import json
    return json.dumps(v, default=str)


def _synthesize_id() -> str:
    return f"tc_{uuid.uuid4().hex[:8]}"


def _toolspec_to_ollama(spec: ToolSpec) -> dict[str, Any]:
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


def _response_to_reply(data: dict[str, Any], model: str) -> LLMReply:
    msg = data.get("message") or {}
    blocks: list[ContentBlock] = []
    text = msg.get("content") or ""
    if text:
        blocks.append(ContentBlock(kind="text", text=text))
    thinking = msg.get("thinking")
    if thinking:
        blocks.insert(0, ContentBlock(kind="thinking", text=thinking))
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                import json
                args = json.loads(args)
            except Exception:
                args = {"_raw": args}
        blocks.append(ContentBlock(
            kind="tool_use",
            id=tc.get("id") or _synthesize_id(),
            name=fn.get("name") or "",
            arguments=dict(args or {}),
            raw=tc,
        ))

    done_reason = (data.get("done_reason") or "stop").lower()
    stop_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "load": "end_turn",
        "unload": "end_turn",
    }
    stop_reason = stop_map.get(done_reason, done_reason)
    if any(b.kind == "tool_use" for b in blocks):
        stop_reason = "tool_use"

    tokens_in = int(data.get("prompt_eval_count") or 0)
    tokens_out = int(data.get("eval_count") or 0)
    usage = Usage(
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        total_tokens=tokens_in + tokens_out,
        raw=data,
    )

    return LLMReply(
        provider="ollama",
        model=data.get("model") or model,
        content=blocks,
        stop_reason=stop_reason,
        usage=usage,
        raw=data,
    )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@dataclass
class OllamaClient:
    model: str = "qwen2.5:7b"
    base_url: str = "http://localhost:11434"
    system_default: str | None = None
    timeout_s: float = 120.0
    http_post: Any = None

    provider: str = field(default="ollama", init=False)

    def _post(self):
        if self.http_post is not None:
            return self.http_post
        import requests
        return requests.post

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
        model_id = model or self.model
        ollama_msgs = _messages_to_ollama(messages)
        sys_prompt = system if system is not None else self.system_default
        if sys_prompt and not any(m["role"] == "system" for m in ollama_msgs):
            ollama_msgs = [{"role": "system", "content": sys_prompt}] + ollama_msgs

        body: dict[str, Any] = {
            "model": model_id,
            "messages": ollama_msgs,
            "stream": False,
        }
        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if options:
            body["options"] = options
        if tools:
            body["tools"] = [_toolspec_to_ollama(t) for t in tools]
        if extra:
            body.update(extra)

        url = f"{self.base_url.rstrip('/')}/api/chat"
        resp = self._post()(url, json=body, timeout=self.timeout_s)
        status = getattr(resp, "status_code", 200)
        if status >= 400:
            err_text = ""
            try:
                err_body = resp.json() if hasattr(resp, "json") else None
                if isinstance(err_body, dict):
                    err_text = str(err_body.get("error") or err_body)
            except Exception:
                err_text = resp.text if hasattr(resp, "text") else ""
            raise RuntimeError(
                f"ollama {status} for model={model_id!r}: {err_text or '(no body)'}"
            )
        data = resp.json() if hasattr(resp, "json") else resp
        return _response_to_reply(data, model_id)
