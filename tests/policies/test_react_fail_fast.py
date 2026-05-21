"""ReAct policy + agent loop: fail-fast on non-retryable provider errors.

Regression coverage for the bug where a missing API key (or Ollama model
without tool support) caused the loop to burn its entire step budget
retrying the same deterministic error.
"""
from __future__ import annotations

import pytest

from banna_agent.core.agent import run_policy
from banna_agent.core.budget import Budget
from banna_agent.core.state import AgentState
from banna_agent.core.types import Trace
from banna_agent.llm.base import LLMReply, ProviderError
from banna_agent.policies.react import ReActPolicy
from banna_agent.tools.base import ToolRegistry


class _NonRetryableLLM:
    provider = "openai"

    def chat(self, **_kwargs) -> LLMReply:
        raise ProviderError("OPENAI_API_KEY not set.", retryable=False)


class _RetryableLLM:
    """Raises a retryable ProviderError twice, then returns a final answer."""
    provider = "openai"

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, **_kwargs) -> LLMReply:
        self.calls += 1
        if self.calls <= 2:
            raise ProviderError("transient blip", retryable=True)
        from banna_agent.llm.base import ContentBlock, ToolCallRequest
        # Tool-call final_answer so the policy commits.
        return LLMReply(
            provider="openai",
            model="gpt-5-nano",
            content=[ContentBlock(
                kind="tool_use",
                id="x",
                name="final_answer",
                arguments={"answer": "42"},
            )],
            stop_reason="tool_use",
        )


def _fresh_state(max_steps: int = 10) -> AgentState:
    return AgentState(
        question="hello",
        trace=Trace(question="hello", run_id="t"),
        budget=Budget(max_steps=max_steps, max_wall_s=60.0),
    )


def test_non_retryable_provider_error_bails_after_one_step() -> None:
    """The loop must not retry a non-retryable error."""
    state = _fresh_state(max_steps=10)
    policy = ReActPolicy(model="gpt-5-nano")
    tools = ToolRegistry()
    llm = _NonRetryableLLM()

    out = run_policy(state, policy, llm=llm, tools=tools)

    # Loop should exit on the very first failed propose — not burn 10
    # steps. We allow 0 because the propose itself raised before the
    # step was committed.
    assert len(out.trace.steps) == 0, (
        f"expected 0 steps, got {len(out.trace.steps)} "
        f"(loop kept retrying a non-retryable error)"
    )
    assert not out.is_done


def test_retryable_provider_error_does_not_bail() -> None:
    """Transient retryable errors still produce THINK steps, loop continues."""
    state = _fresh_state(max_steps=10)
    policy = ReActPolicy(model="gpt-5-nano")
    tools = ToolRegistry()
    llm = _RetryableLLM()

    out = run_policy(state, policy, llm=llm, tools=tools)

    # Two THINK steps from the retryable errors, then the final answer.
    assert llm.calls >= 3
    # Loop should reach the FINAL_ANSWER, not exit early.
    assert out.is_done, "retryable errors should not bail the loop"
