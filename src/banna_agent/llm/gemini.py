"""Gemini adapter via generativelanguage.googleapis.com.

Implemented with `requests` (no google-generativeai SDK dep) so the
mocking story is identical to tools/search.py — we install fakes on
`requests.post` and test offline.

Message shape in Gemini:
  - `contents: [{role, parts: [ { text } | { functionCall } | { functionResponse } ]}]`
  - Role is "user" or "model" (not "assistant"). System goes in top-level
    `systemInstruction: {parts: [{text}]}`.
  - Tools: `tools: [{function_declarations: [ {name, description, parameters} ]}]`
  - Responses carry `candidates[0].content.parts` + `usageMetadata`.
  - `thoughtSignature` on parts must be preserved on round-trip — stored in block.raw.
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


def _blocks_to_parts(blocks: list[ContentBlock]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for b in blocks:
        if isinstance(b.raw, dict):
            parts.append(dict(b.raw))
            continue
        if b.kind == "text":
            parts.append({"text": b.text or ""})
        elif b.kind == "tool_use":
            parts.append({
                "functionCall": {
                    "name": b.name or "",
                    "args": dict(b.arguments),
                }
            })
        elif b.kind == "tool_result":
            response: Any = b.result
            if not isinstance(response, dict):
                response = {"result": response}
            parts.append({
                "functionResponse": {
                    "name": b.name or "",
                    "response": response,
                }
            })
        elif b.kind in ("thinking", "reasoning"):
            # Without raw, we can't reconstruct thoughtSignature; best effort.
            parts.append({"text": b.text or ""})
        else:
            if b.text:
                parts.append({"text": b.text})
    return parts


def _messages_to_gemini(messages: Sequence[Message]) -> tuple[list[dict[str, Any]], str | None]:
    """Return (contents, systemInstruction_text_or_None)."""
    contents: list[dict[str, Any]] = []
    system_text: str | None = None
    for msg in messages:
        if msg.role == "system":
            text_parts = [b.text or "" for b in msg.content if b.kind == "text"]
            system_text = "".join(text_parts) or system_text
            continue
        role = "user" if msg.role in ("user", "tool") else "model"
        parts = _blocks_to_parts(list(msg.content))
        if parts:
            contents.append({"role": role, "parts": parts})
    return contents, system_text


# Gemini's function_declarations.parameters is an OpenAPI 3.0 schema subset,
# not full JSON Schema. These keys cause 400s if present anywhere in the tree.
_GEMINI_SCHEMA_DROP = frozenset({
    "additionalProperties", "$schema", "$id", "$ref", "$defs", "definitions",
    "exclusiveMinimum", "exclusiveMaximum", "const",
})


def _sanitize_gemini_schema(node: Any) -> Any:
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for k, v in node.items():
            if k in _GEMINI_SCHEMA_DROP:
                continue
            if k == "type" and isinstance(v, list):
                non_null = [t for t in v if t != "null"]
                if "null" in v:
                    out["nullable"] = True
                if not non_null:
                    out["type"] = "string"
                elif "array" in non_null:
                    out["type"] = "array"
                else:
                    out["type"] = non_null[0]
                continue
            out[k] = _sanitize_gemini_schema(v)
        # Only run type-specific cleanup if this dict is itself a schema
        # (has a "type"). Otherwise we'd corrupt e.g. a `properties` map
        # whose keys happen to be named "pattern", "items", "required".
        if "type" in out:
            t = out["type"]
            if t != "array":
                out.pop("items", None)
                out.pop("minItems", None)
                out.pop("maxItems", None)
                out.pop("uniqueItems", None)
            if t != "object":
                out.pop("properties", None)
                out.pop("required", None)
            if t != "string":
                out.pop("pattern", None)
                out.pop("minLength", None)
                out.pop("maxLength", None)
        return out
    if isinstance(node, list):
        return [_sanitize_gemini_schema(v) for v in node]
    return node


def _toolspec_to_gemini(specs: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    if not specs:
        return []
    return [
        {
            "function_declarations": [
                {
                    "name": s.name,
                    "description": s.description,
                    "parameters": _sanitize_gemini_schema(s.input_schema),
                }
                for s in specs
            ]
        }
    ]


# ---------------------------------------------------------------------------
# Response deserialization
# ---------------------------------------------------------------------------


def _part_to_block(part: dict[str, Any]) -> ContentBlock:
    if "text" in part:
        return ContentBlock(kind="text", text=part["text"], raw=part)
    if "functionCall" in part:
        fc = part["functionCall"]
        return ContentBlock(
            kind="tool_use",
            id=fc.get("id") or "",
            name=fc.get("name") or "",
            arguments=dict(fc.get("args") or {}),
            raw=part,
        )
    if "functionResponse" in part:
        fr = part["functionResponse"]
        return ContentBlock(
            kind="tool_result",
            name=fr.get("name") or "",
            result=fr.get("response"),
            raw=part,
        )
    if "thought" in part:
        return ContentBlock(kind="thinking", text=part.get("text", ""), raw=part)
    return ContentBlock(kind="unknown", raw=part)


def _response_to_reply(data: dict[str, Any], model: str) -> LLMReply:
    cands = data.get("candidates") or []
    blocks: list[ContentBlock] = []
    finish = "STOP"
    if cands:
        cand = cands[0]
        content = cand.get("content") or {}
        for p in content.get("parts") or []:
            blocks.append(_part_to_block(p))
        finish = cand.get("finishReason") or "STOP"

    # Normalize finish reason.
    stop_map = {
        "STOP": "end_turn",
        "MAX_TOKENS": "max_tokens",
        "SAFETY": "refusal",
        "RECITATION": "refusal",
        "OTHER": "end_turn",
    }
    # If any tool_use block present, prefer tool_use as stop reason.
    stop_reason = stop_map.get(finish, finish.lower())
    if any(b.kind == "tool_use" for b in blocks):
        stop_reason = "tool_use"

    um = data.get("usageMetadata") or {}
    usage = Usage(
        tokens_in=int(um.get("promptTokenCount") or 0),
        tokens_out=int(um.get("candidatesTokenCount") or 0),
        total_tokens=int(um.get("totalTokenCount") or 0),
        thoughts_tokens=int(um.get("thoughtsTokenCount") or 0),
        tool_prompt_tokens=int(um.get("toolUsePromptTokenCount") or 0),
        cache_read_tokens=int(um.get("cachedContentTokenCount") or 0),
        raw=um,
    )

    return LLMReply(
        provider="gemini",
        model=model,
        content=blocks,
        stop_reason=stop_reason,
        usage=usage,
        raw=data,
    )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@dataclass
class GeminiClient:
    model: str = "gemini-2.5-pro"
    api_key: str | None = None
    base_url: str = "https://generativelanguage.googleapis.com"
    system_default: str | None = None
    timeout_s: float = 60.0
    http_post: Any = None  # inject requests.post-shaped callable for tests

    provider: str = field(default="gemini", init=False)

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
        key = self.api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_SEARCH_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY not set.")
        model_id = model or self.model

        contents, sys_from_msgs = _messages_to_gemini(messages)
        sys_text = system if system is not None else (sys_from_msgs or self.system_default)

        body: dict[str, Any] = {"contents": contents}
        if sys_text:
            body["systemInstruction"] = {"parts": [{"text": sys_text}]}
        if tools:
            body["tools"] = _toolspec_to_gemini(tools)
        gen_cfg: dict[str, Any] = {}
        if temperature is not None:
            gen_cfg["temperature"] = temperature
        if max_tokens is not None:
            gen_cfg["maxOutputTokens"] = max_tokens
        if gen_cfg:
            body["generationConfig"] = gen_cfg
        if extra:
            body.update(extra)

        url = f"{self.base_url}/v1beta/models/{model_id}:generateContent"
        post = self._post()
        resp = post(url, params={"key": key}, json=body, timeout=self.timeout_s)
        # Surface the response body in errors. The default
        # `raise_for_status` discards the JSON Gemini returns with the
        # actual reason (invalid model name, missing parts, schema mismatch),
        # which makes 400s opaque.
        status = getattr(resp, "status_code", 200)
        if status >= 400:
            try:
                err_body = resp.json() if hasattr(resp, "json") else None
            except Exception:
                err_body = None
            err_text = (err_body.get("error", {}).get("message") if isinstance(err_body, dict) else "")
            err_text = err_text or (resp.text if hasattr(resp, "text") else "")
            raise RuntimeError(
                f"gemini {status} for model={model_id!r}: {err_text or '(no body)'}"
            )
        data = resp.json() if hasattr(resp, "json") else resp
        return _response_to_reply(data, model_id)
