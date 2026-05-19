"""Tests for the Planner-ReAct policy."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from banna_agent.core.agent import run_policy
from banna_agent.core.state import AgentState
from banna_agent.core.types import ActionKind, Budget
from banna_agent.llm.base import ContentBlock, LLMReply, Usage
from banna_agent.policies.planner_react import PlannerReActPolicy
from banna_agent.tools.base import ToolRegistry
from banna_agent.tools.calculator import make_calculator_tool


@dataclass
class _ScriptedLLM:
    replies: list[LLMReply] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    provider: str = "scripted"

    def chat(self, **kwargs: Any) -> LLMReply:
        self.calls.append(kwargs)
        if not self.replies:
            return LLMReply(provider="scripted", model="s", content=[], stop_reason="end_turn")
        return self.replies.pop(0)


def _text(t: str, tokens=(5, 2)) -> LLMReply:
    return LLMReply(
        provider="scripted", model="s",
        content=[ContentBlock(kind="text", text=t)],
        stop_reason="end_turn",
        usage=Usage(tokens_in=tokens[0], tokens_out=tokens[1]),
    )


def _plan_reply(steps: list[str]) -> LLMReply:
    import json
    return _text(json.dumps({"plan": steps}))


def _calc_tools() -> ToolRegistry:
    return ToolRegistry([make_calculator_tool()])


# ---------------------------------------------------------------------------
# Happy path: planner → 2 subquestions → 2 react answers → final answer
# ---------------------------------------------------------------------------


def test_planner_react_executes_each_step_then_finalizes() -> None:
    llm = _ScriptedLLM([
        _plan_reply(["find A", "compute B given A"]),    # planner call
        _text("intermediate answer A"),                  # subq 1 resolution
        _text("final answer B"),                         # subq 2 resolution
    ])
    state = AgentState(question="q", budget=Budget(max_steps=10, max_wall_s=5.0))
    state = run_policy(state, PlannerReActPolicy(), llm=llm, tools=_calc_tools())
    assert state.is_done
    # Post-Phase-2: the literal model answer is submitted (no canonicalize).
    assert state.trace.final_answer == "final answer B"
    # At minimum two steps: one THINK (resolving subq1) + one FINAL_ANSWER
    assert len(state.trace.steps) >= 2
    kinds = [s.action.kind for s in state.trace.steps]
    assert kinds[-1] == ActionKind.FINAL_ANSWER


def test_planner_react_caches_plan_between_ticks() -> None:
    llm = _ScriptedLLM([
        _plan_reply(["step a", "step b"]),
        _text("result a"),
        _text("result b"),
    ])
    state = AgentState(question="q", budget=Budget(max_steps=10, max_wall_s=5.0))
    state = run_policy(state, PlannerReActPolicy(), llm=llm, tools=_calc_tools())
    # Planner should have been called exactly once.
    planner_calls = [c for c in llm.calls if c.get("system") and "research planner" in c["system"].lower()]
    assert len(planner_calls) == 1


def test_planner_react_advances_cursor() -> None:
    llm = _ScriptedLLM([
        _plan_reply(["step 1", "step 2", "step 3"]),
        _text("a"),
        _text("b"),
        _text("c"),
    ])
    state = AgentState(question="q", budget=Budget(max_steps=10, max_wall_s=5.0))
    policy = PlannerReActPolicy()
    state = run_policy(state, policy, llm=llm, tools=_calc_tools())
    assert state.metadata["_planner_react_cursor"] == 3
    plan = state.metadata["_planner_react_plan"]
    assert len(plan.step_results) == 3
    assert plan.step_results[0]["resolution"] == "a"
    assert plan.step_results[2]["resolution"] == "c"


# ---------------------------------------------------------------------------
# Degenerate planner output
# ---------------------------------------------------------------------------


def test_planner_react_handles_empty_plan_with_fallback() -> None:
    """When the planner fails to emit valid JSON, Planner-ReAct falls back
    to a single-step plan (the whole question). The executor then runs.
    If the executor also produces empty replies, the driver trips the
    step budget — which is the correct, non-crash outcome."""
    llm = _ScriptedLLM([_text("not json")])  # planner: empty plan -> fallback
    state = AgentState(question="q", budget=Budget(max_steps=5, max_wall_s=5.0))
    state = run_policy(state, PlannerReActPolicy(), llm=llm, tools=_calc_tools())
    plan = state.metadata["_planner_react_plan"]
    assert plan.meta.get("fallback") is True
    assert plan.steps == ["q"]
    # Budget trip or fallback answer — both are non-crash outcomes.
    assert len(state.trace.steps) > 0


def test_planner_react_single_step_plan_is_terminal() -> None:
    llm = _ScriptedLLM([
        _plan_reply(["only step"]),
        _text("the answer"),
    ])
    state = AgentState(question="q", budget=Budget(max_steps=5, max_wall_s=5.0))
    state = run_policy(state, PlannerReActPolicy(), llm=llm, tools=_calc_tools())
    assert state.is_done
    # Post-Phase-2: literal answer submitted.
    assert state.trace.final_answer == "the answer"


# ---------------------------------------------------------------------------
# Tool call still flows through executor
# ---------------------------------------------------------------------------


def test_planner_react_propagates_tool_calls() -> None:
    llm = _ScriptedLLM([
        _plan_reply(["compute 2+2"]),
        # subq -> calculator tool call
        LLMReply(
            provider="scripted", model="s",
            content=[ContentBlock(kind="tool_use", id="t1",
                                  name="calculator",
                                  arguments={"expression": "2 + 2"})],
            stop_reason="tool_use",
            usage=Usage(tokens_in=5, tokens_out=2),
        ),
        _text("4"),
    ])
    state = AgentState(question="q", budget=Budget(max_steps=10, max_wall_s=5.0))
    state = run_policy(state, PlannerReActPolicy(), llm=llm, tools=_calc_tools())
    assert state.is_done
    tool_steps = [s for s in state.trace.steps if s.action.kind == ActionKind.TOOL_CALL]
    assert len(tool_steps) == 1
    assert tool_steps[0].action.tool_name == "calculator"
    assert state.trace.final_answer == "4"


# ---------------------------------------------------------------------------
# Per-subquestion trace isolation (F1)
# ---------------------------------------------------------------------------


def test_clone_for_subquestion_filters_trace_to_current_step() -> None:
    """The wrapper handed to the inner ReAct must only see steps from the
    current subquestion. Without this filter, the inner LLM mimics prior
    tool calls and loops on the same query across subquestions."""
    from banna_agent.core.types import Action, ActionKind, Observation, Step
    from banna_agent.policies._planning import Plan
    from banna_agent.policies.planner_react import _clone_for_subquestion

    plan = Plan(steps=["look up population", "look up area", "compute density"])
    state = AgentState(question="What is Iceland's population density?")

    # Simulate having executed subq 0 (population): one tool call + one
    # resolution THINK, both tagged plan_step=0.
    state.trace.steps.extend([
        Step(idx=0,
             action=Action(kind=ActionKind.TOOL_CALL, tool_name="search",
                           tool_args={"query": "Iceland population"},
                           meta={"plan_step": 0}),
             observation=Observation(ok=True, data={"hits": []})),
        Step(idx=1,
             action=Action(kind=ActionKind.THINK, text="...",
                           meta={"plan_step": 0, "resolved_subquestion": True}),
             observation=Observation(ok=True)),
    ])
    plan.with_step_result(0, {"subquestion": plan.steps[0],
                              "resolution": "approximately 372,000"})

    # Now ask for a wrapper scoped to subq 1 (area).
    w = _clone_for_subquestion(state, plan, cursor=1, current=plan.steps[1])

    # The trace the inner ReAct sees has zero subq-0 steps.
    assert len(w.trace.steps) == 0, (
        f"expected subq-1 wrapper to see 0 prior steps, got "
        f"{[s.action.meta for s in w.trace.steps]}"
    )

    # Subq 0's resolution is in the composite question so subq 1 can use it.
    assert "approximately 372,000" in w.question
    assert "Current subquestion (2/3): look up area" in w.question


def test_clone_for_subquestion_keeps_current_steps() -> None:
    """Steps tagged with the current plan_step must remain visible — only
    *prior* subquestions get filtered out."""
    from banna_agent.core.types import Action, ActionKind, Observation, Step
    from banna_agent.policies._planning import Plan
    from banna_agent.policies.planner_react import _clone_for_subquestion

    plan = Plan(steps=["a", "b"])
    state = AgentState(question="q")
    state.trace.steps.extend([
        # subq 0 — should be filtered out
        Step(idx=0, action=Action(kind=ActionKind.THINK, text="x",
                                   meta={"plan_step": 0}),
             observation=Observation(ok=True)),
        # subq 1 in progress — should be kept
        Step(idx=1, action=Action(kind=ActionKind.TOOL_CALL, tool_name="search",
                                   tool_args={"q": "b"},
                                   meta={"plan_step": 1}),
             observation=Observation(ok=True, data={"hits": []})),
    ])
    w = _clone_for_subquestion(state, plan, cursor=1, current="b")
    assert len(w.trace.steps) == 1
    assert w.trace.steps[0].action.tool_name == "search"
