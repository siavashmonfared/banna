"""Best-of-N policy tests.

Drive the policy with a scripted LLM that emits N different text replies
across N inner-policy runs. Confirm:
  - Each inner trajectory runs to completion.
  - The selector picks the right winner under both majority_vote and
    llm_judge modes.
  - Aggregate token usage is mirrored to action.meta so the outer
    driver's budget tracker sees the full cost.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


from banna_agent.core.state import AgentState
from banna_agent.core.types import ActionKind, Budget
from banna_agent.llm.base import ContentBlock, LLMReply, Usage
from banna_agent.policies.best_of_n import (
    BestOfNPolicy,
    _canonical_answer,
    _Candidate,
    _select_llm_judge,
    _select_majority,
)
from banna_agent.policies.react import ReActPolicy
from banna_agent.tools.base import ToolRegistry


@dataclass
class _ScriptedLLM:
    """Returns replies in order. `provider` lets us exercise the
    provider-specific tool_choice path in propose() if relevant."""
    replies: list[LLMReply]
    calls: list[dict[str, Any]] = field(default_factory=list)
    provider: str = "scripted"

    def chat(self, **kwargs: Any) -> LLMReply:
        self.calls.append(kwargs)
        if not self.replies:
            return LLMReply(
                provider=self.provider, model="s", content=[], stop_reason="end_turn",
            )
        return self.replies.pop(0)


def _text(t: str, tin: int = 10, tout: int = 3) -> LLMReply:
    return LLMReply(
        provider="scripted", model="s",
        content=[ContentBlock(kind="text", text=t)],
        stop_reason="end_turn",
        usage=Usage(tokens_in=tin, tokens_out=tout),
    )


# ===========================================================================
# Canonicalization
# ===========================================================================

def test_canonical_groups_equivalents() -> None:
    assert _canonical_answer("Yes.") == _canonical_answer("yes")
    assert _canonical_answer("  42  ") == _canonical_answer("42")
    assert _canonical_answer("'foo'") == "foo"


def test_canonical_keeps_distinct_apart() -> None:
    assert _canonical_answer("3") != _canonical_answer("15")
    # Conservative: 'the X' and 'X' should NOT be merged (intent might differ).
    assert _canonical_answer("the castle") != _canonical_answer("castle")


# ===========================================================================
# Selectors
# ===========================================================================

def _c(idx: int, answer: str, tin: int = 100, tout: int = 10) -> _Candidate:
    return _Candidate(
        idx=idx, answer=answer, tokens_in=tin, tokens_out=tout,
        n_steps=3, finished_reason="final",
    )


def test_majority_picks_mode() -> None:
    cands = [_c(0, "3"), _c(1, "3"), _c(2, "15")]
    assert _select_majority(cands) == 0  # or 1 — first cheapest in winning bucket


def test_majority_breaks_ties_by_cost() -> None:
    cands = [
        _c(0, "yes", tin=500, tout=10),  # expensive yes
        _c(1, "no",  tin=100, tout=5),   # cheap no
    ]
    # Tie 1-1: cheapest trajectory wins.
    assert _select_majority(cands) == 1


def test_majority_ignores_empty_answers() -> None:
    cands = [_c(0, ""), _c(1, "answer"), _c(2, "")]
    assert _select_majority(cands) == 1


def test_majority_all_empty_falls_back_to_zero() -> None:
    cands = [_c(0, ""), _c(1, ""), _c(2, "")]
    assert _select_majority(cands) == 0


def test_llm_judge_picks_index_one_from_reply() -> None:
    """Judge LLM reply '2' → pick candidate at filtered-index 1.
    Since none are empty, filtered list == original; pick.idx == 1."""
    cands = [_c(0, "3"), _c(1, "15"), _c(2, "1")]
    llm = _ScriptedLLM([_text("2")])
    assert _select_llm_judge(
        cands, question="q?", llm=llm, model=None,
    ) == 1
    assert len(llm.calls) == 1


def test_llm_judge_falls_back_to_majority_on_parse_failure() -> None:
    cands = [_c(0, "3"), _c(1, "3"), _c(2, "15")]
    llm = _ScriptedLLM([_text("not a number")])
    chosen = _select_llm_judge(cands, question="q?", llm=llm, model=None)
    # Falls back to majority → "3" wins, idx 0 or 1.
    assert chosen in (0, 1)


def test_llm_judge_skips_call_when_only_one_real_candidate() -> None:
    cands = [_c(0, ""), _c(1, "answer"), _c(2, "")]
    llm = _ScriptedLLM([])  # no replies queued — must not be called
    assert _select_llm_judge(cands, question="q?", llm=llm, model=None) == 1
    assert len(llm.calls) == 0


# ===========================================================================
# Policy.propose end-to-end
# ===========================================================================

def test_best_of_n_runs_inner_n_times_and_picks_majority() -> None:
    """3 inner ReAct runs, replies: '3', '3', '15'. Winner = '3'."""
    llm = _ScriptedLLM([_text("3"), _text("3"), _text("15")])
    state = AgentState(question="q", budget=Budget(max_steps=4, max_wall_s=5.0))
    policy = BestOfNPolicy(
        n=3, selector="majority_vote",
        inner=ReActPolicy(),  # one tick per inner run since reply is plain text
    )
    action = policy.propose(state, llm=llm, tools=ToolRegistry())
    assert action.kind == ActionKind.FINAL_ANSWER
    assert action.answer == "3"
    assert action.meta["n_candidates"] == 3
    assert action.meta["winner_idx"] in (0, 1)


def test_best_of_n_aggregates_token_meta() -> None:
    """tokens_in/out on the returned action equal the SUM across runs."""
    llm = _ScriptedLLM([
        _text("a", tin=100, tout=20),
        _text("a", tin=200, tout=30),
        _text("b", tin=300, tout=40),
    ])
    state = AgentState(question="q", budget=Budget(max_steps=4, max_wall_s=5.0))
    policy = BestOfNPolicy(n=3, inner=ReActPolicy())
    action = policy.propose(state, llm=llm, tools=ToolRegistry())
    assert action.meta["tokens_in"] == 600
    assert action.meta["tokens_out"] == 90


def test_best_of_n_records_per_candidate_telemetry() -> None:
    llm = _ScriptedLLM([_text("a"), _text("b"), _text("c")])
    state = AgentState(question="q", budget=Budget(max_steps=4, max_wall_s=5.0))
    policy = BestOfNPolicy(n=3, inner=ReActPolicy())
    policy.propose(state, llm=llm, tools=ToolRegistry())
    bn = state.metadata["best_of_n"]
    assert len(bn["candidates"]) == 3
    answers = [c["answer"] for c in bn["candidates"]]
    assert set(answers) == {"a", "b", "c"}
    assert bn["selector"] == "majority_vote"


def test_best_of_n_replays_cached_winner_on_second_call() -> None:
    """If propose() is called a second time (defensive), it returns the
    same winner without re-running the inner."""
    llm = _ScriptedLLM([_text("x"), _text("x"), _text("y")])
    state = AgentState(question="q", budget=Budget(max_steps=4, max_wall_s=5.0))
    policy = BestOfNPolicy(n=3, inner=ReActPolicy())
    a1 = policy.propose(state, llm=llm, tools=ToolRegistry())
    # All scripted replies consumed.
    a2 = policy.propose(state, llm=llm, tools=ToolRegistry())
    assert a2.kind == ActionKind.FINAL_ANSWER
    assert a2.answer == a1.answer
    assert a2.meta.get("cached") is True
