"""Unit tests for the Ollama adapter (offline, mocked HTTP)."""
from __future__ import annotations

from dataclasses import dataclass


from banna_agent.llm.base import ContentBlock, Message, ToolSpec
from banna_agent.llm.ollama import OllamaClient, _messages_to_ollama


@dataclass
class _FakeResp:
    status_code: int
    _json: dict

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._json


def _make_post(payload: dict, captured: dict):
    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["body"] = json
        return _FakeResp(200, payload)
    return fake_post


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_messages_to_ollama_basic() -> None:
    out = _messages_to_ollama(
        [Message(role="user", content=[ContentBlock(kind="text", text="hi")])]
    )
    assert out == [{"role": "user", "content": "hi"}]


def test_messages_to_ollama_assistant_with_tool_call() -> None:
    out = _messages_to_ollama(
        [
            Message(
                role="assistant",
                content=[
                    ContentBlock(kind="text", text="trying"),
                    ContentBlock(kind="tool_use", id="tc1", name="search", arguments={"q": "x"}),
                ],
            )
        ]
    )
    assert out[0]["content"] == "trying"
    assert out[0]["tool_calls"][0]["id"] == "tc1"
    assert out[0]["tool_calls"][0]["function"]["name"] == "search"
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"q": "x"}


def test_messages_to_ollama_synthesizes_missing_tool_id() -> None:
    out = _messages_to_ollama(
        [
            Message(
                role="assistant",
                content=[
                    ContentBlock(kind="tool_use", id="", name="search", arguments={}),
                ],
            )
        ]
    )
    tc_id = out[0]["tool_calls"][0]["id"]
    assert tc_id.startswith("tc_")
    assert len(tc_id) > 3


def test_messages_to_ollama_tool_results_split() -> None:
    out = _messages_to_ollama(
        [
            Message(
                role="user",
                content=[ContentBlock(kind="tool_result", id="tc1", result={"x": 1})],
            )
        ]
    )
    assert out == [{"role": "tool", "tool_call_id": "tc1", "content": '{"x": 1}'}]


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


def test_chat_plain_reply() -> None:
    captured: dict = {}
    payload = {
        "model": "qwen2.5:7b",
        "message": {"role": "assistant", "content": "hello"},
        "done_reason": "stop",
        "prompt_eval_count": 14,
        "eval_count": 6,
    }
    client = OllamaClient(http_post=_make_post(payload, captured))
    reply = client.chat(
        messages=[Message(role="user", content=[ContentBlock(kind="text", text="hi")])]
    )
    assert reply.provider == "ollama"
    assert reply.text == "hello"
    assert reply.stop_reason == "end_turn"
    assert reply.usage.tokens_in == 14
    assert reply.usage.tokens_out == 6
    assert captured["url"].endswith("/api/chat")


def test_chat_with_tool_calls() -> None:
    payload = {
        "model": "qwen2.5:7b",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "search", "arguments": {"q": "netflix"}}}
            ],
        },
        "done_reason": "stop",
        "prompt_eval_count": 20,
        "eval_count": 2,
    }
    client = OllamaClient(http_post=_make_post(payload, {}))
    reply = client.chat(
        messages=[Message(role="user", content=[ContentBlock(kind="text", text="search")])]
    )
    assert reply.stop_reason == "tool_use"
    call = reply.tool_calls[0]
    assert call.name == "search"
    # Ollama didn't return an id; we synthesized one
    assert call.id.startswith("tc_")


def test_chat_surfaces_thinking_block() -> None:
    payload = {
        "model": "qwen2.5:7b",
        "message": {
            "role": "assistant",
            "content": "final",
            "thinking": "let me think...",
        },
        "done_reason": "stop",
    }
    client = OllamaClient(http_post=_make_post(payload, {}))
    reply = client.chat(
        messages=[Message(role="user", content=[ContentBlock(kind="text", text="q")])]
    )
    kinds = [b.kind for b in reply.content]
    assert "thinking" in kinds
    assert "text" in kinds
    # Thinking comes first in our deserialization.
    assert kinds[0] == "thinking"


# ---------------------------------------------------------------------------
# Context window
# ---------------------------------------------------------------------------


def test_chat_pins_explicit_num_ctx() -> None:
    captured: dict = {}
    payload = {"model": "m", "message": {"role": "assistant", "content": "ok"}}
    client = OllamaClient(num_ctx=8192, http_post=_make_post(payload, captured))
    client.chat(
        messages=[Message(role="user", content=[ContentBlock(kind="text", text="hi")])]
    )
    assert captured["body"]["options"]["num_ctx"] == 8192


def test_auto_num_ctx_covers_tool_schemas() -> None:
    """Ollama's default window is 4096 and it truncates silently past that.
    A fat tool payload must push the requested window above the default."""
    captured: dict = {}
    payload = {"model": "m", "message": {"role": "assistant", "content": "ok"}}
    fat = [
        ToolSpec(name=f"tool_{i}", description="d" * 1200,
                 input_schema={"type": "object", "properties": {}})
        for i in range(25)
    ]
    client = OllamaClient(http_post=_make_post(payload, captured))
    client.chat(
        messages=[Message(role="user", content=[ContentBlock(kind="text", text="hi")])],
        tools=fat,
    )
    assert captured["body"]["options"]["num_ctx"] > 4096


def test_auto_num_ctx_stays_small_for_small_requests() -> None:
    """Sizing is bucketed, so a short turn doesn't allocate a huge KV cache."""
    captured: dict = {}
    payload = {"model": "m", "message": {"role": "assistant", "content": "ok"}}
    client = OllamaClient(http_post=_make_post(payload, captured))
    client.chat(
        messages=[Message(role="user", content=[ContentBlock(kind="text", text="hi")])]
    )
    assert captured["body"]["options"]["num_ctx"] == 4096


def test_auto_num_ctx_respects_ceiling() -> None:
    captured: dict = {}
    payload = {"model": "m", "message": {"role": "assistant", "content": "ok"}}
    huge = [ToolSpec(name=f"t_{i}", description="d" * 2000,
                     input_schema={"type": "object"}) for i in range(200)]
    client = OllamaClient(num_ctx_max=16384, http_post=_make_post(payload, captured))
    client.chat(
        messages=[Message(role="user", content=[ContentBlock(kind="text", text="hi")])],
        tools=huge,
    )
    assert captured["body"]["options"]["num_ctx"] == 16384
