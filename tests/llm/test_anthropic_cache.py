"""Anthropic prompt caching across the steps of a ReAct run.

banna used to mark only the system block. That prefix is often below the
model's minimum cacheable size, and the growing conversation had no breakpoint
at all, so every step re-processed the whole history uncached.

The client now adds two conversation breakpoints: one on the newest message
and one on the message before the newest tool result. With history projection
on (the default), the newest tool result is shortened on the next step, so
only the second breakpoint yields a reusable prefix.

These tests drive a real ReActPolicy against a fake SDK and replay the
requests through a small model of Anthropic's prefix cache: a breakpoint
writes the prefix ending at its block; a request reads the longest cached
prefix ending at a block boundary within 20 blocks before one of its
breakpoints. cache_control markers are not part of the prefix identity.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
from typing import Any

import pytest

from banna_agent.core.agent import run_policy
from banna_agent.core.state import AgentState
from banna_agent.core.types import Budget
from banna_agent.llm.anthropic import AnthropicClient, _add_conversation_breakpoints
from banna_agent.policies.react import ReActPolicy
from banna_agent.tools.base import JsonTool, ToolRegistry

LOOKBACK = 20
TOOL_STEPS = 4


class _RecordingMessages:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> dict[str, Any]:
        self.requests.append(copy.deepcopy(kwargs))
        n = len(self.requests)
        if n <= TOOL_STEPS:
            content = [{"type": "thinking", "thinking": f"plan {n}", "signature": f"sig{n}"},
                       {"type": "text", "text": f"Step {n}."},
                       {"type": "tool_use", "id": f"toolu_{n}", "name": "run_shell",
                        "input": {"command": f"cat file{n}.txt"}}]
            stop = "tool_use"
        else:
            content = [{"type": "tool_use", "id": "toolu_final", "name": "final_answer",
                        "input": {"answer": "done"}}]
            stop = "tool_use"
        return {"id": f"msg_{n}", "model": "claude-haiku-4-5", "role": "assistant", "content": content,
                "stop_reason": stop, "usage": {"input_tokens": 10, "output_tokens": 5}}


class _FakeSDK:
    def __init__(self) -> None:
        self.messages = _RecordingMessages()


def _tools() -> ToolRegistry:
    def run_shell(args: dict[str, Any]) -> dict[str, Any]:
        # Large, distinct output so history projection really shortens it.
        return {"stdout": f"{args['command']}\n" + ("x" * 6000), "return_code": 0}

    return ToolRegistry([
        JsonTool("run_shell", "Run a command",
                 {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
                 run_shell),
        JsonTool("final_answer", "Finish",
                 {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
                 lambda args: args),
    ])


def _run(**policy_kwargs: Any) -> list[dict[str, Any]]:
    sdk = _FakeSDK()
    client = AnthropicClient(model="claude-haiku-4-5", sdk=sdk)
    policy = ReActPolicy(system_prompt="Use run_shell, then final_answer.", **policy_kwargs)
    state = AgentState(question="Read the files, then finish.", budget=Budget(max_steps=20))
    state = run_policy(state, policy, llm=client, tools=_tools())
    assert state.trace.final_answer == "done"
    assert len(sdk.messages.requests) == TOOL_STEPS + 1
    return sdk.messages.requests


# ------------------------------------------------------------- cache model

def _strip(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k != "cache_control"}
    if isinstance(value, list):
        return [_strip(v) for v in value]
    return value


def _blocks(request: dict[str, Any]) -> list[tuple[Any, bool]]:
    """Request in cache order (tools, system, messages) as (identity, has_breakpoint) blocks."""
    out: list[tuple[Any, bool]] = []
    for tool in request.get("tools") or []:
        out.append((("tool", _strip(tool)), "cache_control" in tool))
    system = request.get("system")
    if isinstance(system, list):
        for block in system:
            out.append((("system", _strip(block)), "cache_control" in block))
    elif system:
        out.append((("system", system), False))
    for i, msg in enumerate(request["messages"]):
        content = msg["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        for block in content:
            out.append((("msg", i, msg["role"], _strip(block)), "cache_control" in block))
    return out


def _prefix_keys(blocks: list[tuple[Any, bool]]) -> list[str]:
    keys, h = [], hashlib.sha256()
    for identity, _ in blocks:
        h.update(json.dumps(identity, sort_keys=True, default=str).encode())
        keys.append(h.copy().hexdigest())
    return keys


def _simulate(requests: list[dict[str, Any]]) -> list[int]:
    """Blocks read from cache per request under the prefix-cache model."""
    cache: set[str] = set()
    hits: list[int] = []
    for request in requests:
        blocks = _blocks(request)
        keys = _prefix_keys(blocks)
        breakpoints = [i for i, (_, bp) in enumerate(blocks) if bp]
        best = 0
        for b in breakpoints:
            for j in range(b, max(-1, b - LOOKBACK - 1), -1):
                if keys[j] in cache:
                    best = max(best, j + 1)
                    break
        hits.append(best)
        cache.update(keys[b] for b in breakpoints)
    return hits


def _conversation_start(request: dict[str, Any]) -> int:
    return sum(1 for identity, _ in _blocks(request) if identity[0] != "msg")


def _blocks_through_message(request: dict[str, Any], index: int) -> int:
    return sum(1 for identity, _ in _blocks(request) if identity[0] != "msg" or identity[1] <= index)


def _last_tool_use_message(request: dict[str, Any]) -> int:
    """Index of the newest assistant tool_use turn, or -1 if there is none."""
    msgs = request["messages"]
    return max((i for i, m in enumerate(msgs) if m["role"] == "assistant" and isinstance(m["content"], list)
                and any(isinstance(b, dict) and b.get("type") == "tool_use" for b in m["content"])), default=-1)


# ------------------------------------------------------------------ tests

@pytest.mark.parametrize("project_history", [True, False])
def test_conversation_prefix_is_read_from_cache_every_step(project_history: bool) -> None:
    requests = _run(project_history=project_history)
    hits = _simulate(requests)
    for n in range(2, len(requests)):  # request 0 has no tool turn to reuse
        previous, current = requests[n - 1], requests[n]
        # Everything through the previous request's newest tool_use turn is
        # unchanged, so it must be served from cache.
        expected = _blocks_through_message(current, _last_tool_use_message(previous))
        assert hits[n] >= expected, (n, hits, expected)
        assert hits[n] > _conversation_start(current)


def test_projection_breaks_the_newest_breakpoint_alone() -> None:
    """Evidence the second breakpoint matters: with projection on and only the
    newest-message breakpoint, nothing past the question is ever reused."""
    requests = _run(project_history=True)
    newest_only = copy.deepcopy(requests)
    for request in newest_only:
        for msg in request["messages"][:-1]:
            for block in msg["content"] if isinstance(msg["content"], list) else []:
                block.pop("cache_control", None)
    hits = _simulate(newest_only)
    assert all(h <= _blocks_through_message(r, 0) for h, r in zip(hits[1:], newest_only[1:], strict=True)), hits
    # ... while the two-breakpoint version keeps extending the cached prefix.
    two = _simulate(requests)
    assert all(b > a for a, b in itertools.pairwise(two[1:])), two
    # The previous newest tool result was shortened on the next step.
    assert requests[1]["messages"][2] != requests[2]["messages"][2]


def test_breakpoint_count_and_placement() -> None:
    for request in _run():
        blocks = _blocks(request)
        assert sum(bp for _, bp in blocks) <= 4
        for msg in request["messages"]:
            for block in msg["content"] if isinstance(msg["content"], list) else []:
                if block.get("type") in ("thinking", "redacted_thinking"):
                    assert "cache_control" not in block
        last = request["messages"][-1]["content"]
        assert any("cache_control" in b for b in last)


def test_step_pressure_nudge_keeps_cache_hits() -> None:
    """A trailing nudge message after the tool result does not move the stable breakpoint."""
    requests = _run(commit_pressure_threshold=0.05)
    assert any(r["messages"][-1]["role"] == "user" and r["messages"][-1]["content"][0].get("type") == "text"
               for r in requests[1:]), "nudge never appeared; test is not exercising it"
    hits = _simulate(requests)
    for n in range(2, len(requests)):
        expected = _blocks_through_message(requests[n], _last_tool_use_message(requests[n - 1]))
        assert hits[n] >= expected, (n, hits, expected)


def test_prompt_cache_false_sends_no_breakpoints() -> None:
    sdk = _FakeSDK()
    client = AnthropicClient(model="claude-haiku-4-5", sdk=sdk, prompt_cache=False)
    state = AgentState(question="Read the files, then finish.", budget=Budget(max_steps=20))
    run_policy(state, ReActPolicy(system_prompt="sys"), llm=client, tools=_tools())
    assert sdk.messages.requests
    assert all(not any(bp for _, bp in _blocks(r)) for r in sdk.messages.requests)


def test_breakpoints_respect_budget_and_skip_thinking() -> None:
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "t", "signature": "s"},
                                          {"type": "tool_use", "id": "a", "name": "x", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "r"}]},
    ]
    original = copy.deepcopy(messages)
    serialized = copy.deepcopy(messages)
    _add_conversation_breakpoints(serialized, used=1)
    assert "cache_control" in serialized[2]["content"][0]
    assert "cache_control" in serialized[1]["content"][1]
    assert "cache_control" not in serialized[1]["content"][0]
    assert messages == original

    full = copy.deepcopy(messages)
    _add_conversation_breakpoints(full, used=4)
    assert full == original
    one_left = copy.deepcopy(messages)
    _add_conversation_breakpoints(one_left, used=3)
    assert "cache_control" in one_left[2]["content"][0]
    assert "cache_control" not in one_left[1]["content"][1]


def test_caller_messages_are_not_mutated() -> None:
    from banna_agent.llm.base import ContentBlock, Message
    sdk = _FakeSDK()
    client = AnthropicClient(model="claude-haiku-4-5", sdk=sdk)
    msgs = [Message(role="user", content=[ContentBlock(kind="text", text="hi")])]
    before = copy.deepcopy(msgs)
    client.chat(messages=msgs)
    assert msgs == before
