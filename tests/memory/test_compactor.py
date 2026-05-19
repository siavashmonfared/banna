"""Unit tests for the trace compactor."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


from banna_agent.core.agent import run_policy
from banna_agent.core.events import EventKind, EventLog
from banna_agent.core.state import AgentState
from banna_agent.core.types import Action, ActionKind, Budget, Observation
from banna_agent.llm.base import ContentBlock, LLMReply, Usage
from banna_agent.memory.compactor import (
    CompactionConfig,
    TraceCompactor,
    approximate_token_count,
)
from banna_agent.tools.base import ToolRegistry


@dataclass
class _ScriptedLLM:
    replies: list[LLMReply] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    provider: str = "scripted"

    def chat(self, **kwargs: Any) -> LLMReply:
        self.calls.append(kwargs)
        if not self.replies:
            return LLMReply(provider="scripted", model="s",
                            content=[ContentBlock(kind="text", text="SUMMARY TEXT")],
                            stop_reason="end_turn",
                            usage=Usage(tokens_in=10, tokens_out=2))
        return self.replies.pop(0)


def _seed_trace(state: AgentState, n: int) -> None:
    for i in range(n):
        state.append_step(
            Action(kind=ActionKind.THINK, text=f"thought {i} " + "x" * 100),
            Observation(ok=True, text=f"thought {i}"),
        )


# ---------------------------------------------------------------------------
# approximate_token_count
# ---------------------------------------------------------------------------


def test_token_count_rough_scale() -> None:
    assert approximate_token_count("") >= 0
    assert approximate_token_count("x" * 100) >= 10
    a = approximate_token_count("x" * 1000)
    b = approximate_token_count("x" * 100)
    assert a > b


# ---------------------------------------------------------------------------
# should_compact
# ---------------------------------------------------------------------------


def test_compactor_disabled_by_default() -> None:
    state = AgentState(question="?")
    _seed_trace(state, 10)
    comp = TraceCompactor(_ScriptedLLM(), CompactionConfig())
    assert comp.should_compact(state) is False


def test_compactor_skips_when_trace_shorter_than_keep() -> None:
    state = AgentState(question="?")
    _seed_trace(state, 2)  # keep_last_n_steps default 4
    comp = TraceCompactor(_ScriptedLLM(), CompactionConfig(enabled=True, threshold_tokens=10))
    assert comp.should_compact(state) is False


def test_compactor_triggers_when_over_threshold() -> None:
    state = AgentState(question="?")
    _seed_trace(state, 20)  # each step adds >100 chars
    comp = TraceCompactor(
        _ScriptedLLM(),
        CompactionConfig(enabled=True, threshold_tokens=50, keep_last_n_steps=4),
    )
    assert comp.should_compact(state) is True


# ---------------------------------------------------------------------------
# compact
# ---------------------------------------------------------------------------


def test_compact_replaces_old_steps_with_one_summary() -> None:
    state = AgentState(question="q")
    _seed_trace(state, 10)
    llm = _ScriptedLLM()
    comp = TraceCompactor(llm, CompactionConfig(enabled=True, keep_last_n_steps=4))

    assert len(state.trace.steps) == 10
    info = comp.compact(state)

    # summary + 4 kept = 5 steps total
    assert len(state.trace.steps) == 5
    assert info["dropped_steps"] == 6
    assert info["kept_steps"] == 4

    head = state.trace.steps[0]
    assert head.action.meta.get("compaction") is True
    assert "SUMMARY TEXT" in (head.action.text or "")

    # kept-step indices reindexed
    idxs = [s.idx for s in state.trace.steps]
    assert idxs == [0, 1, 2, 3, 4]


def test_compact_noop_when_too_short() -> None:
    state = AgentState(question="q")
    _seed_trace(state, 3)
    llm = _ScriptedLLM()
    comp = TraceCompactor(llm, CompactionConfig(keep_last_n_steps=4))
    info = comp.compact(state)
    assert info["dropped_steps"] == 0
    assert len(state.trace.steps) == 3


def test_compact_llm_error_recorded_as_placeholder() -> None:
    state = AgentState(question="q")
    _seed_trace(state, 10)

    class _BrokenLLM:
        provider = "broken"
        def chat(self, **_): raise RuntimeError("boom")

    comp = TraceCompactor(_BrokenLLM(), CompactionConfig(enabled=True, keep_last_n_steps=2))
    comp.compact(state)
    # Should not raise; summary should contain the error marker.
    head_text = state.trace.steps[0].action.text or ""
    assert "summarizer failed" in head_text


# ---------------------------------------------------------------------------
# Integration with run_policy
# ---------------------------------------------------------------------------


class _ScriptedPolicy:
    name = "scripted"
    def __init__(self, actions: list[Action]) -> None:
        self.actions = list(actions)
    def propose(self, state, *, llm, tools) -> Action:
        if not self.actions:
            return Action(kind=ActionKind.FINAL_ANSWER, answer="END")
        return self.actions.pop(0)


def test_driver_emits_compact_event_and_keeps_running() -> None:
    # Pre-seed the state with 10 steps BEFORE driver runs.
    state = AgentState(question="?", budget=Budget(max_steps=15, max_wall_s=5.0))
    _seed_trace(state, 10)

    # Policy then emits a final answer.
    policy = _ScriptedPolicy([
        Action(kind=ActionKind.FINAL_ANSWER, answer="final"),
    ])
    llm = _ScriptedLLM()
    comp = TraceCompactor(llm, CompactionConfig(enabled=True, threshold_tokens=50, keep_last_n_steps=3))
    log = EventLog()

    state = run_policy(
        state, policy,
        llm=llm, tools=ToolRegistry(),
        log=log, compactor=comp,
    )

    # Compaction fired on the first tick (before propose)
    compact_events = log.filter(EventKind.COMPACT)
    assert len(compact_events) == 1
    assert compact_events[0].payload["dropped_steps"] > 0

    assert state.is_done
    assert state.trace.final_answer == "final"


def test_driver_skips_compaction_when_disabled() -> None:
    state = AgentState(question="?", budget=Budget(max_steps=5, max_wall_s=5.0))
    _seed_trace(state, 10)
    policy = _ScriptedPolicy([Action(kind=ActionKind.FINAL_ANSWER, answer="ok")])
    log = EventLog()
    comp = TraceCompactor(_ScriptedLLM(), CompactionConfig(enabled=False))

    state = run_policy(
        state, policy,
        llm=_ScriptedLLM(), tools=ToolRegistry(),
        log=log, compactor=comp,
    )
    assert log.filter(EventKind.COMPACT) == []
