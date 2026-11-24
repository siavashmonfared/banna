"""Multi-turn replay against strict fake providers.

Both failures below appear only on the SECOND model request, when banna
replays its earlier tool call:
  * OpenAI: Chat Completions rejects function tools + reasoning_effort, so a
    reasoning model must go through the Responses API, with each encrypted
    reasoning item replayed before its function call (same call_id).
  * Gemini 3: a replayed functionCall without its thoughtSignature is
    rejected; thought summaries must not come back as visible model text.
The fakes reject exactly those mistakes. A deterministic 4xx must stop the
run at once instead of spending the step budget on identical retries.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from banna_agent.core.agent import run_policy
from banna_agent.core.state import AgentState
from banna_agent.core.types import Budget
from banna_agent.llm.base import ProviderError
from banna_agent.llm.gemini import GeminiClient
from banna_agent.llm.openai import OpenAIClient, _is_openai_endpoint
from banna_agent.policies.react import ReActPolicy
from banna_agent.tools.base import JsonTool, ToolRegistry

SYSTEM = "Complete the task using run_shell. Use final_answer when finished."


def _tools(ran: list[str]) -> ToolRegistry:
    return ToolRegistry([
        JsonTool("run_shell", "Run a command",
                 {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
                 lambda args: ran.append(args["command"]) or {"stdout": "ok", "return_code": 0}),
        JsonTool("final_answer", "Finish",
                 {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
                 lambda args: args),
    ])


def _run(policy: ReActPolicy, client: Any, ran: list[str]) -> AgentState:
    state = AgentState(question="Run ls, then finish.", budget=Budget(max_steps=8))
    return run_policy(state, policy, llm=client, tools=_tools(ran))


class _HTTPError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status


# ----------------------------------------------------------------- OpenAI

class _Endpoint:
    def __init__(self, handler) -> None:
        self.handler = handler

    def create(self, **kwargs: Any) -> Any:
        return self.handler(kwargs)


class _FakeOpenAI:
    """Records which endpoint each request used."""

    def __init__(self, responses=None, chat=None) -> None:
        self.calls: list[tuple[str, dict]] = []

        def wrap(name, fn):
            def handler(kwargs):
                self.calls.append((name, kwargs))
                return fn(kwargs)
            return handler

        self.responses = _Endpoint(wrap("responses", responses or self._reject_chat))

        class _Chat:
            pass
        self.chat = _Chat()
        self.chat.completions = _Endpoint(wrap("chat", chat or self._reject_chat))

    @staticmethod
    def _reject_chat(kwargs):
        raise _HTTPError(400, "Function tools with reasoning_effort are not supported in /v1/chat/completions.")


def _responses_body(model: str, output: list[dict]) -> dict:
    return {"id": "resp", "object": "response", "model": model, "status": "completed", "output": output,
            "usage": {"input_tokens": 50, "output_tokens": 10, "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 5}}}


def test_openai_reasoning_uses_responses_and_replays_reasoning() -> None:
    model = "gpt-5-mini"

    def responses(body):
        assert body["reasoning"]["effort"] == "medium" and body["store"] is False
        assert "reasoning.encrypted_content" in body["include"]
        assert all(t["strict"] is False for t in body["tools"])
        assert body["instructions"].startswith(SYSTEM)
        n = sum(1 for name, _ in sdk.calls if name == "responses")
        if n == 1:
            return _responses_body(model, [
                {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "ENC-1"},
                {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "run_shell",
                 "arguments": '{"command": "ls"}', "status": "completed"}])
        kinds = [(i.get("type") or i.get("role"), i.get("call_id") or i.get("encrypted_content"))
                 for i in body["input"]]
        expected = [("reasoning", "ENC-1"), ("function_call", "call_1"), ("function_call_output", "call_1")]
        if kinds[-3:] != expected:
            raise _HTTPError(400, f"Replay mismatch: {kinds}")
        return _responses_body(model, [
            {"type": "function_call", "id": "fc_2", "call_id": "call_2", "name": "final_answer",
             "arguments": '{"answer": "done"}', "status": "completed"}])

    sdk = _FakeOpenAI(responses=responses)
    policy = ReActPolicy(system_prompt=SYSTEM, model=model, extra={"reasoning_effort": "medium"})
    ran: list[str] = []
    state = _run(policy, OpenAIClient(model=model, sdk=sdk), ran)
    assert [name for name, _ in sdk.calls] == ["responses", "responses"]
    assert ran == ["ls"] and state.trace.final_answer == "done"
    # The action keeps the provider's own call id for tracing.
    tool_steps = [s for s in state.trace.steps if s.action.tool_name == "run_shell"]
    assert tool_steps[0].action.meta["native_tool_call_id"] == "call_1"


@pytest.mark.parametrize("base_url,expected", [
    (None, True),
    ("https://api.openai.com/v1", True),
    ("https://openrouter.ai/api/v1", False),
    ("https://api.moonshot.ai/v1", False),
    ("http://localhost:8000/v1", False),
])
def test_responses_route_only_for_openai_endpoint(base_url, expected, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert _is_openai_endpoint(base_url) is expected
    client = OpenAIClient(base_url=base_url)
    assert client._uses_responses_api({"reasoning_effort": "low"}) is expected
    assert client._uses_responses_api({}) is False


def test_openrouter_with_reasoning_keeps_chat_completions() -> None:
    chat_reply = {"id": "c1", "model": "openai/gpt-5-mini", "choices": [{"message": {
        "role": "assistant", "content": "done"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    sdk = _FakeOpenAI(chat=lambda kwargs: chat_reply)
    client = OpenAIClient(model="openai/gpt-5-mini", base_url="https://openrouter.ai/api/v1", sdk=sdk)
    client.chat(messages=[], extra={"reasoning_effort": "low"})
    assert [name for name, _ in sdk.calls] == ["chat"]
    assert sdk.calls[0][1]["reasoning_effort"] == "low"


@pytest.mark.parametrize("extra", [{}, {"reasoning_effort": "medium"}])
def test_openai_deterministic_400_is_not_retried(extra) -> None:
    def reject(kwargs):
        raise _HTTPError(400, "Unsupported parameter")

    sdk = _FakeOpenAI(responses=reject, chat=reject)
    client = OpenAIClient(model="gpt-5-mini", sdk=sdk)
    with pytest.raises(ProviderError) as info:
        client.chat(messages=[], extra=extra)
    assert info.value.retryable is False

    ran: list[str] = []
    _run(ReActPolicy(system_prompt=SYSTEM, model="gpt-5-mini", extra=extra), client, ran)
    assert len(sdk.calls) == 2  # one from the direct call, ONE from the run: no retries
    assert ran == []


def test_openai_429_stays_retryable() -> None:
    def throttle(kwargs):
        raise _HTTPError(429, "Rate limit")

    client = OpenAIClient(model="gpt-5-mini", sdk=_FakeOpenAI(chat=throttle))
    with pytest.raises(Exception) as info:
        client.chat(messages=[])
    assert not (isinstance(info.value, ProviderError) and info.value.retryable is False)


# ----------------------------------------------------------------- Gemini

class _GeminiResponse:
    def __init__(self, status: int, body: dict) -> None:
        self.status_code, self._body = status, body
        self.text = json.dumps(body)

    def json(self) -> dict:
        return self._body


def _gemini_reply(parts: list[dict]) -> _GeminiResponse:
    return _GeminiResponse(200, {
        "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 50, "candidatesTokenCount": 10}})


def _gemini_rejection(message: str) -> _GeminiResponse:
    return _GeminiResponse(400, {"error": {"code": 400, "message": message, "status": "INVALID_ARGUMENT"}})


def test_gemini_replays_thought_signatures_and_hides_thoughts() -> None:
    bodies: list[dict] = []

    def post(url, **kwargs):
        body = kwargs["json"]
        bodies.append(body)
        if len(bodies) == 1:
            return _gemini_reply([{"text": "private plan", "thought": True},
                                  {"functionCall": {"name": "run_shell", "args": {"command": "ls"}},
                                   "thoughtSignature": "SIG-1"}])
        calls = [p for c in body["contents"] for p in c["parts"] if "functionCall" in p]
        if not calls or any(not p.get("thoughtSignature") for p in calls):
            return _gemini_rejection("Function call is missing a thought_signature in functionCall parts.")
        model_texts = [p.get("text") for c in body["contents"] if c["role"] == "model" for p in c["parts"]]
        assert "private plan" not in model_texts, "thought summary replayed as visible text"
        return _gemini_reply([{"functionCall": {"name": "final_answer", "args": {"answer": "done"}},
                               "thoughtSignature": "SIG-2"}])

    ran: list[str] = []
    client = GeminiClient(model="gemini-3-flash", api_key="fixture-key", http_post=post)
    state = _run(ReActPolicy(system_prompt=SYSTEM, model="gemini-3-flash"), client, ran)
    assert len(bodies) == 2 and ran == ["ls"] and state.trace.final_answer == "done"
    assert bodies[1]["contents"][1]["parts"][-1]["thoughtSignature"] == "SIG-1"


def test_gemini_thought_part_is_thinking_not_text() -> None:
    from banna_agent.llm.gemini import _part_to_block
    block = _part_to_block({"text": "plan", "thought": True})
    assert block.kind == "thinking"
    assert _part_to_block({"text": "answer"}).kind == "text"


def test_gemini_deterministic_400_stops_after_one_request() -> None:
    bodies: list[dict] = []

    def post(url, **kwargs):
        bodies.append(kwargs["json"])
        return _gemini_rejection("Function call is missing a thought_signature in functionCall parts.")

    client = GeminiClient(model="gemini-3-flash", api_key="fixture-key", http_post=post)
    with pytest.raises(ProviderError) as info:
        client.chat(messages=[])
    assert info.value.retryable is False
    bodies.clear()
    _run(ReActPolicy(system_prompt=SYSTEM, model="gemini-3-flash"), client, [])
    assert len(bodies) == 1
