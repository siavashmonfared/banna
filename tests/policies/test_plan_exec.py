"""Tests for `_plan_exec.execute_plan_step` and `score_plan`."""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from banna_agent.core.state import AgentState
from banna_agent.core.types import ActionKind
from banna_agent.llm.base import ContentBlock, LLMReply, Usage
from banna_agent.policies._plan_exec import (
    execute_plan_step,
    score_plan,
)
from banna_agent.policies._planning import Plan
from banna_agent.tools.base import ToolRegistry
from banna_agent.tools.calculator import make_calculator_tool


@dataclass
class _ScriptedLLM:
    replies: list[LLMReply] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    provider: str = "s"

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.replies.pop(0) if self.replies else LLMReply(
            provider="s", model="m", content=[], stop_reason="end_turn"
        )


def _text(t: str) -> LLMReply:
    return LLMReply(
        provider="s", model="m",
        content=[ContentBlock(kind="text", text=t)],
        stop_reason="end_turn",
        usage=Usage(tokens_in=10, tokens_out=5),
    )


def _tool(name: str, args: dict) -> LLMReply:
    return LLMReply(
        provider="s", model="m",
        content=[ContentBlock(kind="tool_use", id="t1", name=name, arguments=args)],
        stop_reason="tool_use",
        usage=Usage(tokens_in=15, tokens_out=3),
    )


# ---------------------------------------------------------------------------
# execute_plan_step — text reply
# ---------------------------------------------------------------------------


def test_execute_step_text_reply() -> None:
    llm = _ScriptedLLM([_text("the answer")])
    plan = Plan(steps=["find x", "compute y"])
    state = AgentState(question="main q")
    r = execute_plan_step(plan, 0, main_question="main q",
                          llm=llm, tools=ToolRegistry(), state=state)
    assert r.ok is True
    assert r.answer == "the answer"
    assert r.tool_name is None
    # Step appended to trace.
    assert len(state.trace.steps) == 1


def test_execute_step_empty_text_is_not_ok() -> None:
    llm = _ScriptedLLM([_text("")])
    plan = Plan(steps=["find x"])
    state = AgentState(question="q")
    r = execute_plan_step(plan, 0, main_question="q",
                          llm=llm, tools=ToolRegistry(), state=state)
    assert r.ok is False


# ---------------------------------------------------------------------------
# execute_plan_step — tool call
# ---------------------------------------------------------------------------


def test_execute_step_tool_call_dispatches() -> None:
    llm = _ScriptedLLM([_tool("calculator", {"expression": "17 * 23"})])
    tools = ToolRegistry([make_calculator_tool()])
    plan = Plan(steps=["compute 17*23"])
    state = AgentState(question="q")
    r = execute_plan_step(plan, 0, main_question="q",
                          llm=llm, tools=tools, state=state)
    assert r.ok is True
    assert r.tool_name == "calculator"
    assert r.tool_args == {"expression": "17 * 23"}
    # Trace should have the tool call as a real step.
    assert state.trace.steps[0].action.kind == ActionKind.TOOL_CALL
    # Answer field summarizes the tool's 'value'.
    assert "391" in r.answer


def test_execute_step_unknown_tool_fails_gracefully() -> None:
    llm = _ScriptedLLM([_tool("nonexistent", {})])
    plan = Plan(steps=["do it"])
    state = AgentState(question="q")
    r = execute_plan_step(plan, 0, main_question="q",
                          llm=llm, tools=ToolRegistry(), state=state)
    assert r.ok is False
    assert r.error is None or "unknown" in (r.error or "")


# ---------------------------------------------------------------------------
# execute_plan_step — LLM exception
# ---------------------------------------------------------------------------


def test_execute_step_llm_exception_returns_error_result() -> None:
    class _Boom:
        provider = "boom"
        def chat(self, **_): raise RuntimeError("timeout")

    plan = Plan(steps=["q"])
    state = AgentState(question="q")
    r = execute_plan_step(plan, 0, main_question="q",
                          llm=_Boom(), tools=ToolRegistry(), state=state)
    assert r.ok is False
    assert "RuntimeError" in (r.error or "")
    # Driver's trace is NOT polluted on LLM error.
    assert len(state.trace.steps) == 0


# ---------------------------------------------------------------------------
# score_plan
# ---------------------------------------------------------------------------


def test_score_plan_coverage_and_cost() -> None:
    plan = Plan(steps=["a", "b", "c"])
    plan.with_step_result(0, {"ok": True, "tokens_in": 10, "tokens_out": 5})
    plan.with_step_result(1, {"ok": True, "tokens_in": 20, "tokens_out": 8})
    plan.with_step_result(2, {"ok": False, "tokens_in": 5, "tokens_out": 2})
    s = score_plan(plan, evidence_before=3, evidence_after=6)
    assert s.coverage == pytest.approx(2 / 3)
    assert s.evidence_count == 3
    assert s.cost_tokens == 50
    assert s.total < s.coverage  # cost penalty subtracts


def test_score_plan_verifier_bonus() -> None:
    plan = Plan(steps=["a"])
    plan.with_step_result(0, {"ok": True, "tokens_in": 10, "tokens_out": 5, "answer": "42"})
    plan.final_answer = "42"
    s_no = score_plan(plan, evidence_before=0, evidence_after=0, verifier=None)
    s_yes = score_plan(plan, evidence_before=0, evidence_after=0,
                       verifier=lambda a: a == "42")
    assert s_yes.total > s_no.total


def test_score_plan_handles_verifier_exception() -> None:
    plan = Plan(steps=["a"])
    plan.with_step_result(0, {"ok": True, "tokens_in": 1, "tokens_out": 1})
    plan.final_answer = "x"

    def _bad(_):
        raise ValueError("nope")
    s = score_plan(plan, evidence_before=0, evidence_after=0, verifier=_bad)
    assert s.total == pytest.approx(1.0 - s.penalty)


# ---------------------------------------------------------------------------
# _summarize_tool_result (F2)
# ---------------------------------------------------------------------------


def test_summarize_tool_result_skips_empty_summary() -> None:
    """A tool result with summary='' must NOT short-circuit to ''.

    This was the bug behind BFS/DFS empty-resolution loops: search
    returns {summary: '', hits: [...]} and the old code returned ''
    because it only checked `is not None`.
    """
    from banna_agent.policies._plan_exec import _summarize_tool_result

    data = {
        "query": "Iceland population",
        "summary": "",  # empty — must be skipped
        "hits": [
            {"title": "Demographics of Iceland - Wikipedia",
             "url": "https://en.wikipedia.org/wiki/Demographics_of_Iceland",
             "snippet": "Iceland's population is approximately 372,000."},
            {"title": "Iceland Population (2026) - Worldometer",
             "url": "https://worldometers.info/iceland",
             "snippet": "..."},
        ],
    }
    out = _summarize_tool_result(data)
    assert out, f"expected non-empty summary, got {out!r}"
    assert "Demographics of Iceland" in out


def test_summarize_tool_result_uses_summary_when_present() -> None:
    from banna_agent.policies._plan_exec import _summarize_tool_result
    out = _summarize_tool_result({"summary": "the answer is 42"})
    assert out == "the answer is 42"


def test_summarize_tool_result_falls_back_to_hits_when_no_keys() -> None:
    from banna_agent.policies._plan_exec import _summarize_tool_result
    data = {"hits": [{"title": "Foo", "url": "https://foo", "snippet": ""}]}
    out = _summarize_tool_result(data)
    assert "Foo" in out


def test_summarize_tool_result_empty_string_value_is_skipped() -> None:
    """`{"value": ""}` must not be treated as 'the answer is empty'."""
    from banna_agent.policies._plan_exec import _summarize_tool_result
    data = {"value": "", "hits": [{"title": "Real Answer"}]}
    assert "Real Answer" in _summarize_tool_result(data)


# ---------------------------------------------------------------------------
# synthesize_final_answer (J1)
# ---------------------------------------------------------------------------


def test_synthesize_uses_evidence_and_step_resolutions(monkeypatch) -> None:
    """Synthesizer must read evidence + plan resolutions and produce a
    coherent final answer, not the last tool's raw return value."""
    from dataclasses import dataclass

    from banna_agent.core.state import AgentState
    from banna_agent.llm.base import ContentBlock, LLMReply, Usage
    from banna_agent.policies._plan_exec import synthesize_final_answer
    from banna_agent.policies._planning import Plan

    captured = {}

    @dataclass
    class _LLM:
        provider: str = "fake"
        def chat(self, **kw):
            for m in kw.get("messages", []):
                for b in m.content:
                    if b.kind == "text":
                        captured["prompt"] = (captured.get("prompt", "")
                                              + (b.text or ""))
            return LLMReply(provider="fake", model="m",
                content=[ContentBlock(kind="text",
                                      text="Iceland has 3.7 people per km².")],
                stop_reason="end_turn", usage=Usage(tokens_in=80, tokens_out=12))

    plan = Plan(steps=["population", "area", "compute density"])
    plan.with_step_result(0, {"resolution": "approximately 372,000"})
    plan.with_step_result(1, {"resolution": "approximately 103,000 km²"})
    plan.with_step_result(2, {"resolution": "{'op':'write','id':'mem_x'}"})

    state = AgentState(question="What is Iceland's population density?")
    state.add_evidence(source="https://wikipedia.org/Iceland",
                        content="Population around 372,000; area 103,000 km².")

    text, t_in, t_out = synthesize_final_answer(
        state.question, plan, state, llm=_LLM(),
    )
    assert "3.7" in text or "people per km" in text
    assert t_in == 80 and t_out == 12

    # Verify the prompt actually contained both evidence and resolutions.
    p = captured["prompt"]
    assert "Population around 372,000" in p
    assert "approximately 103,000" in p
    # The memory-write garbage *is* in the prompt, but the synthesizer's
    # response (which we control above) is what becomes the answer.
    assert "Main question:" in p


def test_synthesize_handles_missing_evidence_gracefully() -> None:
    from dataclasses import dataclass

    from banna_agent.core.state import AgentState
    from banna_agent.llm.base import ContentBlock, LLMReply, Usage
    from banna_agent.policies._plan_exec import synthesize_final_answer
    from banna_agent.policies._planning import Plan

    @dataclass
    class _LLM:
        provider: str = "fake"
        def chat(self, **kw):
            return LLMReply(provider="fake", model="m",
                content=[ContentBlock(kind="text", text="42")],
                stop_reason="end_turn", usage=Usage(tokens_in=5, tokens_out=1))

    plan = Plan(steps=["a"])
    plan.with_step_result(0, {"resolution": "the answer"})
    state = AgentState(question="q")  # no evidence
    text, _, _ = synthesize_final_answer(state.question, plan, state, llm=_LLM())
    assert text == "42"


def test_synthesize_catches_llm_exceptions() -> None:
    from dataclasses import dataclass

    from banna_agent.core.state import AgentState
    from banna_agent.policies._plan_exec import synthesize_final_answer
    from banna_agent.policies._planning import Plan

    @dataclass
    class _Boom:
        provider: str = "boom"
        def chat(self, **kw):
            raise RuntimeError("nope")

    plan = Plan(steps=["a"])
    plan.with_step_result(0, {"resolution": "x"})
    state = AgentState(question="q")
    text, t_in, t_out = synthesize_final_answer(
        state.question, plan, state, llm=_Boom(),
    )
    assert "synthesizer failed" in text
    assert t_in == 0 and t_out == 0


# ---------------------------------------------------------------------------
# Pending-token drain (J3)
# ---------------------------------------------------------------------------


def test_pending_tokens_round_trip() -> None:
    from banna_agent.core.state import AgentState
    from banna_agent.policies._plan_exec import (
        drain_pending_tokens,
        stash_pending_tokens,
    )

    state = AgentState(question="?")
    stash_pending_tokens(state, 50, 12)
    stash_pending_tokens(state, 30, 4)
    meta: dict = {"existing": "k"}
    drain_pending_tokens(state, meta)
    assert meta == {"existing": "k", "tokens_in": 80, "tokens_out": 16}
    # State cleared.
    assert "_pending_tokens_in" not in state.metadata
    assert "_pending_tokens_out" not in state.metadata


def test_pending_tokens_zero_no_op() -> None:
    from banna_agent.core.state import AgentState
    from banna_agent.policies._plan_exec import (
        drain_pending_tokens,
        stash_pending_tokens,
    )

    state = AgentState(question="?")
    stash_pending_tokens(state, 0, 0)
    meta: dict = {}
    drain_pending_tokens(state, meta)
    assert meta == {}


def test_drain_preserves_existing_action_tokens() -> None:
    from banna_agent.core.state import AgentState
    from banna_agent.policies._plan_exec import (
        drain_pending_tokens,
        stash_pending_tokens,
    )

    state = AgentState(question="?")
    stash_pending_tokens(state, 5, 3)
    meta: dict = {"tokens_in": 100, "tokens_out": 10}
    drain_pending_tokens(state, meta)
    # Pending tokens add to existing, not overwrite.
    assert meta["tokens_in"] == 105
    assert meta["tokens_out"] == 13


# ---------------------------------------------------------------------------
# Multi-tool-call plan steps (L1)
# ---------------------------------------------------------------------------


def test_execute_step_multi_iteration_search_then_text() -> None:
    """Plan step makes a tool call, sees the result, then commits a text answer.

    This is the key new behavior — pre-L1 the executor returned after
    the first tool call with a stringified search result as the
    "answer", forcing the synthesizer to compute from URL titles. Now
    the model gets a second LLM call with the tool result in context
    and can extract the actual value.
    """
    llm = _ScriptedLLM([
        _tool("calculator", {"expression": "17 * 23"}),  # iter 0
        _text("17 × 23 = 391"),                          # iter 1
    ])
    tools = ToolRegistry([make_calculator_tool()])
    plan = Plan(steps=["compute 17 * 23"])
    state = AgentState(question="q")

    r = execute_plan_step(plan, 0, main_question="q",
                          llm=llm, tools=tools, state=state)
    assert r.ok is True
    assert r.answer == "17 × 23 = 391"
    # Tokens accumulate across both inner LLM calls.
    assert r.tokens_in > 0
    assert r.tokens_out > 0
    # Two calls were made.
    assert len(llm.calls) == 2
    # Trace has the tool_call step + the synthesized THINK answer.
    kinds = [s.action.kind for s in state.trace.steps]
    from banna_agent.core.types import ActionKind as _AK
    assert _AK.TOOL_CALL in kinds
    assert _AK.THINK in kinds


def test_execute_step_caps_at_max_inner_steps() -> None:
    """If the model keeps making tool calls past the cap, we bail and
    use the last tool's summary as the answer."""
    llm = _ScriptedLLM([
        _tool("calculator", {"expression": "1 + 1"}),
        _tool("calculator", {"expression": "2 + 2"}),
        _tool("calculator", {"expression": "3 + 3"}),
        _tool("calculator", {"expression": "4 + 4"}),  # never reached
    ])
    tools = ToolRegistry([make_calculator_tool()])
    plan = Plan(steps=["loop"])
    state = AgentState(question="q")

    r = execute_plan_step(plan, 0, main_question="q",
                          llm=llm, tools=tools, state=state,
                          max_inner_steps=3)
    # Max iterations consumed → 3 LLM calls made, fall-through path
    # uses the last summary as the answer.
    assert len(llm.calls) == 3
    assert r.tool_name == "calculator"
    assert r.ok is True       # last tool succeeded
    # Three tool_call steps + one synthetic THINK fallback in the trace.
    from banna_agent.core.types import ActionKind as _AK
    tool_steps = [s for s in state.trace.steps
                  if s.action.kind == _AK.TOOL_CALL]
    assert len(tool_steps) == 3
    # The synthetic fall-through step has the max_inner_reached marker.
    last = state.trace.steps[-1]
    assert last.action.kind == _AK.THINK
    assert last.action.meta.get("max_inner_reached") is True


def test_execute_step_text_first_call_skips_loop() -> None:
    """When the model returns text on the first iteration, only one LLM
    call happens — same as pre-L1 behavior, no regression."""
    llm = _ScriptedLLM([_text("the answer is 42")])
    plan = Plan(steps=["q"])
    state = AgentState(question="q")
    r = execute_plan_step(plan, 0, main_question="q",
                          llm=llm, tools=ToolRegistry(), state=state)
    assert r.ok is True
    assert r.answer == "the answer is 42"
    assert len(llm.calls) == 1
    assert len(state.trace.steps) == 1
