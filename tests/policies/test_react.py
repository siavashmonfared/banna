"""Unit tests for the ReAct policy.

We inject a scripted fake LLM (returns pre-canned LLMReplies in order)
so the policy's decision rule is exercised without any network.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


from banna_agent.core.agent import run_policy
from banna_agent.core.state import AgentState
from banna_agent.core.types import ActionKind, Budget
from banna_agent.llm.base import ContentBlock, LLMReply, ToolSpec, Usage
from banna_agent.policies.react import ReActPolicy
from banna_agent.tools.base import ToolRegistry
from banna_agent.tools.calculator import make_calculator_tool


@dataclass
class _ScriptedLLM:
    """Returns a fixed sequence of LLMReply objects, one per chat() call."""

    replies: list[LLMReply]
    calls: list[dict[str, Any]] = field(default_factory=list)
    provider: str = "scripted"

    def chat(self, **kwargs: Any) -> LLMReply:
        self.calls.append(kwargs)
        if not self.replies:
            return LLMReply(provider="scripted", model="s", content=[], stop_reason="end_turn")
        return self.replies.pop(0)


def _text_reply(t: str) -> LLMReply:
    return LLMReply(
        provider="scripted",
        model="s-1",
        content=[ContentBlock(kind="text", text=t)],
        stop_reason="end_turn",
        usage=Usage(tokens_in=10, tokens_out=3),
    )


def _tool_reply(name: str, args: dict) -> LLMReply:
    return LLMReply(
        provider="scripted",
        model="s-1",
        content=[ContentBlock(kind="tool_use", id="t1", name=name, arguments=args)],
        stop_reason="tool_use",
        usage=Usage(tokens_in=20, tokens_out=5),
    )


def _multi_tool_reply(calls: list[tuple[str, dict]]) -> LLMReply:
    """Reply with ≥2 tool_use blocks in one response (parallel tool use)."""
    content = [
        ContentBlock(kind="tool_use", id=f"t{i}", name=name, arguments=args)
        for i, (name, args) in enumerate(calls)
    ]
    return LLMReply(
        provider="scripted",
        model="s-1",
        content=content,
        stop_reason="tool_use",
        usage=Usage(tokens_in=20, tokens_out=5),
    )


def _calc_tools() -> ToolRegistry:
    return ToolRegistry([make_calculator_tool()])


# ---------------------------------------------------------------------------
# Policy-level tests (no driver)
# ---------------------------------------------------------------------------


def test_propose_emits_tool_call_when_llm_calls_tool() -> None:
    llm = _ScriptedLLM([_tool_reply("calculator", {"expression": "2+2"})])
    state = AgentState(question="2+2?")
    action = ReActPolicy().propose(state, llm=llm, tools=_calc_tools())
    assert action.kind == ActionKind.TOOL_CALL
    assert action.tool_name == "calculator"
    assert action.tool_args == {"expression": "2+2"}


def test_propose_emits_tool_batch_when_llm_emits_two_tool_calls() -> None:
    """≥2 tool_use blocks in one reply → ActionKind.TOOL_BATCH with both
    sub-calls in meta['batch_calls']. The driver will parallel-dispatch."""
    llm = _ScriptedLLM([
        _multi_tool_reply([
            ("calculator", {"expression": "1+1"}),
            ("calculator", {"expression": "2+2"}),
        ]),
    ])
    state = AgentState(question="?")
    action = ReActPolicy().propose(state, llm=llm, tools=_calc_tools())
    assert action.kind == ActionKind.TOOL_BATCH
    batch = action.meta["batch_calls"]
    assert len(batch) == 2
    assert {b["name"] for b in batch} == {"calculator"}
    exprs = sorted(b["args"]["expression"] for b in batch)
    assert exprs == ["1+1", "2+2"]
    assert action.meta["batch_names"] == ["calculator", "calculator"]


def test_propose_does_not_batch_when_final_answer_is_in_the_calls() -> None:
    """`final_answer` is the terminal commit; never batch it with anything else.
    Falls back to single-call path (first call wins)."""
    llm = _ScriptedLLM([
        _multi_tool_reply([
            ("calculator", {"expression": "1+1"}),
            ("final_answer", {"answer": "2"}),
        ]),
    ])
    state = AgentState(question="?")
    action = ReActPolicy().propose(state, llm=llm, tools=_calc_tools())
    assert action.kind != ActionKind.TOOL_BATCH
    # First-call path is preserved.
    assert action.kind == ActionKind.TOOL_CALL
    assert action.tool_name == "calculator"


def test_propose_dedups_identical_batch_calls_and_falls_back_to_single() -> None:
    """If the model echoes the same (name, args) twice, dedup; if it collapses
    to one call, fall through to the regular TOOL_CALL path."""
    llm = _ScriptedLLM([
        _multi_tool_reply([
            ("calculator", {"expression": "1+1"}),
            ("calculator", {"expression": "1+1"}),
        ]),
    ])
    state = AgentState(question="?")
    action = ReActPolicy().propose(state, llm=llm, tools=_calc_tools())
    assert action.kind == ActionKind.TOOL_CALL
    assert action.tool_args == {"expression": "1+1"}


def test_propose_emits_final_answer_when_llm_only_text() -> None:
    llm = _ScriptedLLM([_text_reply("42")])
    state = AgentState(question="?")
    action = ReActPolicy().propose(state, llm=llm, tools=_calc_tools())
    assert action.kind == ActionKind.FINAL_ANSWER
    assert action.answer == "42"


def test_propose_handles_empty_reply_as_think() -> None:
    empty = LLMReply(provider="scripted", model="s", content=[], stop_reason="end_turn")
    llm = _ScriptedLLM([empty])
    state = AgentState(question="?")
    action = ReActPolicy().propose(state, llm=llm, tools=_calc_tools())
    assert action.kind == ActionKind.THINK
    assert "empty_reply" in (action.text or "")


def test_propose_converts_llm_exception_to_think() -> None:
    class _Boom:
        provider = "boom"
        def chat(self, **_: object):
            raise RuntimeError("timeout")

    state = AgentState(question="?")
    action = ReActPolicy().propose(state, llm=_Boom(), tools=_calc_tools())
    assert action.kind == ActionKind.THINK
    assert "llm_error" in (action.text or "")
    assert "RuntimeError" in (action.text or "")


def test_propose_sends_history_with_tool_result_on_retry() -> None:
    """After a tool call has been executed, the next propose() should
    replay the tool_use + tool_result turn to the model."""
    llm = _ScriptedLLM([_text_reply("final")])
    state = AgentState(question="compute")
    # Simulate a prior tool call+result by appending a step.
    from banna_agent.core.types import Action, Observation
    state.append_step(
        Action(kind=ActionKind.TOOL_CALL, tool_name="calculator",
               tool_args={"expression": "2+2"}),
        Observation(ok=True, data={"value": 4}),
    )
    ReActPolicy().propose(state, llm=llm, tools=_calc_tools())
    # The scripted LLM captured the request kwargs.
    msgs = llm.calls[0]["messages"]
    kinds = [b.kind for m in msgs for b in m.content]
    assert "tool_use" in kinds
    assert "tool_result" in kinds


def test_propose_passes_tool_specs_to_llm() -> None:
    llm = _ScriptedLLM([_text_reply("done")])
    state = AgentState(question="?")
    ReActPolicy().propose(state, llm=llm, tools=_calc_tools())
    tools_kw = llm.calls[0]["tools"]
    assert any(isinstance(t, ToolSpec) and t.name == "calculator" for t in tools_kw)


def test_propose_uses_configured_model_and_tokens() -> None:
    llm = _ScriptedLLM([_text_reply("ok")])
    state = AgentState(question="?")
    policy = ReActPolicy(model="custom-1", max_tokens=50, temperature=0.3)
    policy.propose(state, llm=llm, tools=_calc_tools())
    assert llm.calls[0]["model"] == "custom-1"
    assert llm.calls[0]["max_tokens"] == 50
    assert llm.calls[0]["temperature"] == 0.3


# ---------------------------------------------------------------------------
# Driver + ReAct end-to-end
# ---------------------------------------------------------------------------


def test_driver_with_react_completes_tool_then_answer() -> None:
    llm = _ScriptedLLM([
        _tool_reply("calculator", {"expression": "17*23"}),
        _text_reply("391"),
    ])
    state = AgentState(question="17*23?", budget=Budget(max_steps=5, max_wall_s=5.0))
    state = run_policy(state, ReActPolicy(), llm=llm, tools=_calc_tools())
    assert state.is_done
    assert state.trace.final_answer == "391"
    assert len(state.trace.steps) == 2


def test_driver_with_react_respects_step_budget() -> None:
    # LLM keeps asking for tool calls; budget=2 should cap the run.
    llm = _ScriptedLLM([
        _tool_reply("calculator", {"expression": "1+1"}),
        _tool_reply("calculator", {"expression": "2+2"}),
        _tool_reply("calculator", {"expression": "3+3"}),
    ])
    state = AgentState(question="?", budget=Budget(max_steps=2, max_wall_s=5.0))
    state = run_policy(state, ReActPolicy(), llm=llm, tools=_calc_tools())
    assert not state.is_done
    assert len(state.trace.steps) == 2


# ---------------------------------------------------------------------------
# Step-pressure nudge (R2)
# ---------------------------------------------------------------------------


def test_history_no_nudge_below_threshold() -> None:
    """No commit-pressure message when budget is mostly unspent."""
    from banna_agent.core.types import Budget
    state = AgentState(question="q", budget=Budget(max_steps=10))
    state.budget.steps_used = 3   # 30%, below 0.6 default
    msgs = ReActPolicy()._history(state)
    # Just the original user message; no nudge appended.
    assert len(msgs) == 1
    assert msgs[0].role == "user"


def test_history_appends_nudge_at_threshold() -> None:
    """When ≥60% of step budget is spent, _history appends a commit
    nudge as a user message."""
    from banna_agent.core.types import Budget
    state = AgentState(question="q", budget=Budget(max_steps=10))
    state.budget.steps_used = 6   # exactly 60%
    msgs = ReActPolicy()._history(state)
    assert len(msgs) == 2
    last = msgs[-1]
    assert last.role == "user"
    body = last.content[0].text or ""
    assert "stop calling tools" in body.lower() or "commit" in body.lower()
    assert "6" in body and "10" in body  # used/max numbers leak through


def test_history_nudge_threshold_is_configurable() -> None:
    from banna_agent.core.types import Budget
    state = AgentState(question="q", budget=Budget(max_steps=10))
    state.budget.steps_used = 3
    # 0.3 threshold → should nudge
    msgs = ReActPolicy(commit_pressure_threshold=0.3)._history(state)
    assert len(msgs) == 2
    # 0.5 threshold → should NOT nudge (we're at 0.3)
    msgs = ReActPolicy(commit_pressure_threshold=0.5)._history(state)
    assert len(msgs) == 1


def test_history_nudge_disabled_when_threshold_zero() -> None:
    from banna_agent.core.types import Budget
    state = AgentState(question="q", budget=Budget(max_steps=10))
    state.budget.steps_used = 9   # 90% — would nudge by default
    msgs = ReActPolicy(commit_pressure_threshold=0.0)._history(state)
    assert len(msgs) == 1


def test_history_nudge_skipped_when_max_steps_zero() -> None:
    """If max_steps is 0 (unlimited), no nudge fires regardless of usage."""
    from banna_agent.core.types import Budget
    state = AgentState(question="q", budget=Budget(max_steps=0))
    state.budget.steps_used = 50
    msgs = ReActPolicy()._history(state)
    assert len(msgs) == 1


# ---------------------------------------------------------------------------
# B1: bounded history projection
# ---------------------------------------------------------------------------


def _append_tool_step(state: AgentState, *, text: str, evidence_id: str) -> None:
    from banna_agent.core.types import Action, Observation
    state.append_step(
        Action(kind=ActionKind.TOOL_CALL, tool_name="read_url",
               tool_args={"url": "http://x"}),
        Observation(ok=True, data={"url": "http://x", "title": "T",
                                   "text": text, "evidence_id": evidence_id}),
    )


def _tool_result_texts(msgs) -> list[str]:
    """Extract every tool_result payload's `text` field, in order."""
    out = []
    for m in msgs:
        for b in m.content:
            if b.kind == "tool_result" and isinstance(b.result, dict):
                out.append(b.result.get("text", ""))
    return out


def test_history_projects_old_observations_but_keeps_last_full() -> None:
    """Older tool results are trimmed to a snippet; the most recent one
    is replayed verbatim so the model can act on it."""
    state = AgentState(question="q")
    big_old = "A" * 5000
    big_new = "B" * 5000
    _append_tool_step(state, text=big_old, evidence_id="ev_old")
    _append_tool_step(state, text=big_new, evidence_id="ev_new")

    msgs = ReActPolicy(history_snippet_chars=1500)._history(state)
    texts = _tool_result_texts(msgs)
    assert len(texts) == 2
    old, new = texts
    # Old one is trimmed and points at recall_evidence with its id.
    assert len(old) < len(big_old)
    assert "trimmed" in old
    assert "ev_old" in old
    # Newest one is untouched.
    assert new == big_new


def test_history_projection_disabled_replays_full() -> None:
    """With project_history=False, every observation replays verbatim
    (pre-B1 behavior, the ablation/rollback lever)."""
    state = AgentState(question="q")
    big = "A" * 5000
    _append_tool_step(state, text=big, evidence_id="ev_old")
    _append_tool_step(state, text="B" * 5000, evidence_id="ev_new")

    msgs = ReActPolicy(project_history=False)._history(state)
    texts = _tool_result_texts(msgs)
    assert texts[0] == big  # old one NOT trimmed


def test_history_projection_leaves_short_fields_alone() -> None:
    """Fields below the snippet cap pass through untouched."""
    state = AgentState(question="q")
    _append_tool_step(state, text="short", evidence_id="ev_old")
    _append_tool_step(state, text="also short", evidence_id="ev_new")
    msgs = ReActPolicy(history_snippet_chars=1500)._history(state)
    texts = _tool_result_texts(msgs)
    assert texts[0] == "short"


def test_history_projection_does_not_mutate_trace() -> None:
    """Projection only affects the LLM copy; the trace keeps full text."""
    state = AgentState(question="q")
    big = "A" * 5000
    _append_tool_step(state, text=big, evidence_id="ev_old")
    _append_tool_step(state, text="B" * 5000, evidence_id="ev_new")
    ReActPolicy(history_snippet_chars=1500)._history(state)
    assert state.trace.steps[0].observation.data["text"] == big


# ---------------------------------------------------------------------------
# B6: coverage-aware commit pressure
# ---------------------------------------------------------------------------


def test_enumerable_targets_year_range() -> None:
    from banna_agent.policies.react import _enumerable_targets
    t = _enumerable_targets("What was revenue from 2018 to 2021?")
    assert t["years"] == ["2018", "2019", "2020", "2021"]
    assert t["count"] is None


def test_enumerable_targets_hyphen_range() -> None:
    from banna_agent.policies.react import _enumerable_targets
    assert _enumerable_targets("ARPU 2019-2020 trend")["years"] == ["2019", "2020"]


def test_enumerable_targets_single_year_not_enumerable() -> None:
    from banna_agent.policies.react import _enumerable_targets
    assert _enumerable_targets("what happened in 2020?")["years"] == []


def test_enumerable_targets_bare_list_of_three_years() -> None:
    from banna_agent.policies.react import _enumerable_targets
    # Three distinct bare years (no range connector) → enumerable.
    t = _enumerable_targets("population in 2000, 2010, 2020 census")
    assert t["years"] == ["2000", "2010", "2020"]


def test_enumerable_targets_two_years_need_per_year_cue() -> None:
    from banna_agent.policies.react import _enumerable_targets
    # Two years, no range connector, no cue → not enumerable.
    assert _enumerable_targets("compare 2001 census with 2011 data")["years"] == []
    # Two years plus an explicit per-year cue → enumerable.
    assert _enumerable_targets(
        "population for each year: 2001 census, 2011 data")["years"] == ["2001", "2011"]


def test_enumerable_targets_count() -> None:
    from banna_agent.policies.react import _enumerable_targets
    assert _enumerable_targets("list the three largest cities")["count"] == 3
    assert _enumerable_targets("name 5 albums")["count"] == 5
    assert _enumerable_targets("who is the president?")["count"] is None


def test_enumerable_targets_none_for_plain_question() -> None:
    from banna_agent.policies.react import _enumerable_targets
    t = _enumerable_targets("What is the capital of France?")
    assert t == {"years": [], "count": None}


def _nudge_text(policy: ReActPolicy, state: AgentState) -> str:
    msg = policy._step_pressure_message(state)
    assert msg is not None
    return msg.content[0].text


def test_coverage_nudge_fires_for_multipart_year_question() -> None:
    from banna_agent.core.types import Budget
    state = AgentState(question="Revenue from 2018 to 2021?",
                       budget=Budget(max_steps=10))
    state.budget.steps_used = 7  # 70% — between threshold and hard ceiling
    text = _nudge_text(ReActPolicy(), state)
    assert "MULTI-PART" in text
    assert "2018" in text  # nothing gathered → all years listed missing


def test_coverage_nudge_names_only_missing_years() -> None:
    from banna_agent.core.types import Action, Budget, Observation
    state = AgentState(question="Revenue from 2018 to 2020?",
                       budget=Budget(max_steps=10))
    # Gathered evidence already mentions 2018 and 2019, not 2020.
    state.append_step(
        Action(kind=ActionKind.TOOL_CALL, tool_name="read_url",
               tool_args={"url": "x"}),
        Observation(ok=True, data={"text": "rev 2018 was 5; 2019 was 6"}),
    )
    state.budget.steps_used = 7
    text = _nudge_text(ReActPolicy(), state)
    assert "2020" in text
    assert "2018, 2019" not in text  # already covered, not relisted as missing


def test_coverage_nudge_reverts_to_blunt_above_hard_ceiling() -> None:
    from banna_agent.core.types import Budget
    state = AgentState(question="Revenue from 2018 to 2021?",
                       budget=Budget(max_steps=10))
    state.budget.steps_used = 9  # 90% — at hard ceiling
    text = _nudge_text(ReActPolicy(), state)
    assert "MULTI-PART" not in text
    assert "Stop calling tools and commit" in text


def test_coverage_aware_pressure_disabled_uses_blunt_nudge() -> None:
    from banna_agent.core.types import Budget
    state = AgentState(question="Revenue from 2018 to 2021?",
                       budget=Budget(max_steps=10))
    state.budget.steps_used = 7
    text = _nudge_text(ReActPolicy(coverage_aware_pressure=False), state)
    assert "MULTI-PART" not in text
    assert "Stop calling tools and commit" in text


def test_single_answer_question_uses_blunt_nudge() -> None:
    from banna_agent.core.types import Budget
    state = AgentState(question="What is the capital of France?",
                       budget=Budget(max_steps=10))
    state.budget.steps_used = 7
    text = _nudge_text(ReActPolicy(), state)
    assert "Stop calling tools and commit" in text


# ---------------------------------------------------------------------------
# Phase 3: commit_required escalation — force tool_choice after one nudge,
# bail with preceding text after two.
# ---------------------------------------------------------------------------


def _final_answer_tools() -> ToolRegistry:
    from banna_agent.tools.final_answer import make_final_answer_tool
    return ToolRegistry([make_final_answer_tool()])


def _append_commit_required_step(state: AgentState, preceding_text: str = "42") -> None:
    """Simulate the prior tick's commit_required THINK landing in the trace."""
    from banna_agent.core.types import Action, ActionKind, Observation
    action = Action(
        kind=ActionKind.THINK,
        text="[commit_required] call final_answer",
        meta={"commit_required": True, "preceding_text": preceding_text},
    )
    state.append_step(action, Observation(ok=True, text=action.text))


def test_commit_required_one_nudge_forces_tool_choice_anthropic() -> None:
    """After one commit_required THINK, the next propose() injects
    tool_choice into the extra kwargs sent to the Anthropic provider."""
    state = AgentState(question="?")
    _append_commit_required_step(state, preceding_text="42")

    class _AnthropicLLM:
        provider = "anthropic"
        last_extra: dict | None = None
        def chat(self, **kwargs: Any) -> LLMReply:
            type(self).last_extra = kwargs.get("extra")
            return _tool_reply("final_answer", {"answer": "42"})

    llm = _AnthropicLLM()
    ReActPolicy().propose(state, llm=llm, tools=_final_answer_tools())
    assert _AnthropicLLM.last_extra is not None
    assert _AnthropicLLM.last_extra.get("tool_choice") == {
        "type": "tool", "name": "final_answer",
    }


def test_commit_required_one_nudge_forces_tool_choice_gemini() -> None:
    state = AgentState(question="?")
    _append_commit_required_step(state)

    class _GeminiLLM:
        provider = "gemini"
        last_extra: dict | None = None
        def chat(self, **kwargs: Any) -> LLMReply:
            type(self).last_extra = kwargs.get("extra")
            return _tool_reply("final_answer", {"answer": "x"})

    llm = _GeminiLLM()
    ReActPolicy().propose(state, llm=llm, tools=_final_answer_tools())
    cfg = (_GeminiLLM.last_extra or {}).get("tool_config", {})
    assert cfg.get("function_calling_config", {}).get("mode") == "ANY"
    assert "final_answer" in cfg.get("function_calling_config", {}).get(
        "allowed_function_names", []
    )


def test_commit_required_two_nudges_bails_with_preceding_text() -> None:
    """After two consecutive commit_required THINKs, the policy bails
    out without calling the LLM and emits FINAL_ANSWER using the
    preceding plain-text reply we captured."""
    state = AgentState(question="?")
    _append_commit_required_step(state, preceding_text="the answer is 42")
    _append_commit_required_step(state, preceding_text="the answer is 42")

    # No LLM reply should be consumed — guard with empty queue.
    llm = _ScriptedLLM([])
    action = ReActPolicy().propose(state, llm=llm, tools=_final_answer_tools())
    assert action.kind == ActionKind.FINAL_ANSWER
    assert action.answer == "the answer is 42"
    assert action.meta.get("commit_required_bailout") is True
    assert len(llm.calls) == 0  # short-circuit, no chat call


def test_commit_required_meta_stashes_preceding_text() -> None:
    """The commit_required THINK records the model's plain-text reply
    so the second-nudge bailout can recover it."""
    llm = _ScriptedLLM([_text_reply("17054.888")])
    state = AgentState(question="?")
    action = ReActPolicy().propose(state, llm=llm, tools=_final_answer_tools())
    assert action.kind == ActionKind.THINK
    assert action.meta.get("commit_required") is True
    assert action.meta.get("preceding_text") == "17054.888"


# ---------------------------------------------------------------------------
# Phase 7: final_answer.evidence_ids -> Claim registration
# ---------------------------------------------------------------------------


def test_final_answer_with_evidence_ids_registers_claim() -> None:
    """When the model cites evidence_ids in the final_answer tool call,
    the policy registers the answer as a Claim with those supports so
    CitationVerifier can grade grounding."""
    state = AgentState(question="?")
    # Manually register a piece of evidence so we have a real ID.
    ev = state.add_evidence(
        source="https://example.com/page",
        content="The capital of France is Paris.",
    )
    reply = LLMReply(
        provider="scripted", model="s",
        content=[ContentBlock(
            kind="tool_use", id="t1", name="final_answer",
            arguments={"answer": "Paris", "evidence_ids": [ev.evidence_id]},
        )],
        stop_reason="tool_use",
        usage=Usage(tokens_in=10, tokens_out=3),
    )
    llm = _ScriptedLLM([reply])
    action = ReActPolicy().propose(state, llm=llm, tools=_final_answer_tools())
    assert action.kind == ActionKind.FINAL_ANSWER
    assert action.answer == "Paris"
    assert action.meta.get("evidence_ids") == [ev.evidence_id]
    # Claim was registered.
    assert len(state.claims) == 1
    assert state.claims[0].text == "Paris"
    assert state.claims[0].supports == [ev.evidence_id]


def test_final_answer_without_evidence_ids_skips_claim_registration() -> None:
    """Tasks that legitimately need no external evidence (riddles,
    arithmetic) should not be force-rejected for lack of citations."""
    state = AgentState(question="?")
    reply = LLMReply(
        provider="scripted", model="s",
        content=[ContentBlock(
            kind="tool_use", id="t1", name="final_answer",
            arguments={"answer": "42"},  # no evidence_ids
        )],
        stop_reason="tool_use",
        usage=Usage(tokens_in=10, tokens_out=3),
    )
    llm = _ScriptedLLM([reply])
    action = ReActPolicy().propose(state, llm=llm, tools=_final_answer_tools())
    assert action.kind == ActionKind.FINAL_ANSWER
    assert action.answer == "42"
    assert state.claims == []


# ---------------------------------------------------------------------------
# C2: empty-reply detection upgrades
# ---------------------------------------------------------------------------


def _append_empty_reply(state: AgentState) -> None:
    from banna_agent.core.types import Action, ActionKind, Observation
    a = Action(
        kind=ActionKind.THINK,
        text="[empty_reply] model returned no text and no tool_calls",
        meta={"empty_reply": True, "repair": True},
    )
    state.append_step(a, Observation(ok=True, text=a.text))


def test_empty_reply_is_tagged_repair_and_not_billed() -> None:
    """Smoke: empty_reply produced by propose() carries repair=True so
    state.append_step routes it to the repair-step axis."""
    state = AgentState(question="?")
    llm = _ScriptedLLM([LLMReply(
        provider="scripted", model="s",
        content=[], stop_reason="end_turn", usage=Usage(tokens_in=10, tokens_out=0),
    )])
    action = ReActPolicy().propose(state, llm=llm, tools=_final_answer_tools())
    assert action.kind == ActionKind.THINK
    assert (action.meta or {}).get("repair") is True
    assert (action.meta or {}).get("empty_reply") is True


def test_two_empties_with_no_evidence_forces_required_tool_choice_openai() -> None:
    """Two empty replies in a row + no prior tool calls → next chat is
    forced with tool_choice='required' (any tool), not final_answer."""
    state = AgentState(question="?")
    _append_empty_reply(state)
    _append_empty_reply(state)

    class _OpenAILLM:
        provider = "openai"
        last_extra: dict | None = None
        def chat(self, **kwargs: Any) -> LLMReply:
            type(self).last_extra = kwargs.get("extra")
            return _tool_reply("calculator", {"expression": "1+1"})

    llm = _OpenAILLM()
    ReActPolicy().propose(state, llm=llm, tools=_calc_tools())
    assert _OpenAILLM.last_extra is not None
    assert _OpenAILLM.last_extra.get("tool_choice") == "required"


def test_two_empties_with_evidence_forces_final_answer() -> None:
    """Two empty replies + prior successful tool call → force commit
    via tool_choice=final_answer (we have material; just need to land it)."""
    from banna_agent.core.types import Action, ActionKind, Observation
    state = AgentState(question="?")
    state.append_step(
        Action(kind=ActionKind.TOOL_CALL, tool_name="search", tool_args={"q": "x"}),
        Observation(ok=True, data={"hits": [{"url": "u", "snippet": "s"}]}),
    )
    _append_empty_reply(state)
    _append_empty_reply(state)

    class _OpenAILLM:
        provider = "openai"
        last_extra: dict | None = None
        def chat(self, **kwargs: Any) -> LLMReply:
            type(self).last_extra = kwargs.get("extra")
            return _tool_reply("final_answer", {"answer": "x"})

    llm = _OpenAILLM()
    ReActPolicy().propose(state, llm=llm, tools=_final_answer_tools())
    tc = (_OpenAILLM.last_extra or {}).get("tool_choice")
    assert isinstance(tc, dict)
    assert tc.get("function", {}).get("name") == "final_answer"


# ---------------------------------------------------------------------------
# C3: budget-exhaustion synthesis
# ---------------------------------------------------------------------------


def test_synthesize_uses_last_claim_when_no_llm() -> None:
    """Cheap path: pick the most recent Claim text as the answer."""
    state = AgentState(question="?")
    state.add_claim(text="Paris")
    state.add_claim(text="Berlin")
    action = ReActPolicy().synthesize_on_exhaustion(state)
    assert action is not None
    assert action.kind == ActionKind.FINAL_ANSWER
    assert action.answer == "Berlin"
    assert (action.meta or {}).get("synthesis") == "cheap_last_claim"


def test_synthesize_uses_short_preceding_text_when_no_claims() -> None:
    from banna_agent.core.types import Action, ActionKind, Observation
    state = AgentState(question="?")
    state.append_step(
        Action(
            kind=ActionKind.THINK,
            text="[commit_required] call final_answer",
            meta={"commit_required": True, "preceding_text": "42", "repair": True},
        ),
        Observation(ok=True),
    )
    action = ReActPolicy().synthesize_on_exhaustion(state)
    assert action is not None
    assert action.answer == "42"


def test_synthesize_returns_none_when_trace_empty() -> None:
    state = AgentState(question="?")
    action = ReActPolicy().synthesize_on_exhaustion(state)
    assert action is None


def test_synthesize_calls_llm_with_forced_final_answer_when_provided() -> None:
    """With an LLM + final_answer tool registered, we issue a one-shot
    call that pins tool_choice to final_answer."""
    state = AgentState(question="What is 2+2?")
    # Make sure cheap path would not fire (no claims, no short text).
    llm = _ScriptedLLM([_tool_reply("final_answer", {"answer": "4"})])
    action = ReActPolicy().synthesize_on_exhaustion(
        state, llm=llm, tools=_final_answer_tools(),
    )
    assert action is not None
    assert action.kind == ActionKind.FINAL_ANSWER
    assert action.answer == "4"
    assert (action.meta or {}).get("synthesis") == "llm"
    assert len(llm.calls) == 1


def test_synthesize_skips_empty_reply_marker_in_cheap_fallback() -> None:
    """A trace where every tick was an empty-reply repair must not commit
    the `[empty_reply] ...` marker string as the final answer. Earlier
    versions did, and those tasks scored zero on GAIA.
    """
    from banna_agent.core.types import Action, ActionKind, Observation
    state = AgentState(question="?")
    # 3 empty-reply repair THINKs in a row, nothing else.
    for _ in range(3):
        state.append_step(
            Action(
                kind=ActionKind.THINK,
                text="[empty_reply] model returned no text and no tool_calls",
                meta={"empty_reply": True, "repair": True},
            ),
            Observation(ok=True),
        )
    action = ReActPolicy().synthesize_on_exhaustion(state)
    assert action is None  # no real candidate; refuse to fabricate one


def test_synthesize_skips_empty_reply_marker_in_claim_text() -> None:
    """Even if the marker somehow lands in claim text, it must be skipped."""
    state = AgentState(question="?")
    state.add_claim(text="[empty_reply] model returned no text and no tool_calls")
    state.add_claim(text="real-answer")
    action = ReActPolicy().synthesize_on_exhaustion(state)
    assert action is not None
    assert action.answer == "real-answer"


def test_synthesize_falls_back_to_cheap_on_llm_failure() -> None:
    """If the LLM raises, we still emit a cheap synthesis from the last claim."""
    state = AgentState(question="?")
    state.add_claim(text="fallback-answer")

    class _BoomLLM:
        provider = "openai"
        def chat(self, **kwargs: Any) -> LLMReply:
            raise RuntimeError("provider down")

    action = ReActPolicy().synthesize_on_exhaustion(
        state, llm=_BoomLLM(), tools=_final_answer_tools(),
    )
    # Threaded call swallows the exception and falls through to cheap.
    assert action is not None
    assert action.answer == "fallback-answer"
    assert (action.meta or {}).get("synthesis") == "cheap_last_claim"


def test_synthesize_llm_path_sends_real_tool_specs_and_block_content() -> None:
    """Regression for the B5 bug pair: the LLM-driven exhaustion path must
    send real ToolSpecs (via ``to_tool_specs()``, not the nonexistent
    ``tools.specs()`` which silently became ``None``) and message content as
    a list of ContentBlocks (not a bare ``str``). A real provider serializes
    both and throws on either bug — this validating mock stands in for that
    contract so the path can't silently fall back to ``_cheap()`` again.

    Pre-fix this test fails: ``tools`` arrives as ``None`` and ``content`` as
    a ``str``, the asserts raise inside the worker thread, the exception is
    swallowed, and with no claims on the trace synthesis returns ``None``.
    """
    state = AgentState(question="What is 2+2?")

    class _ValidatingLLM:
        provider = "openai"
        calls: list[dict[str, Any]] = []

        def chat(self, **kwargs: Any) -> LLMReply:
            type(self).calls.append(kwargs)
            tools = kwargs.get("tools")
            assert tools, "tools must be a non-empty list of ToolSpec"
            assert all(isinstance(t, ToolSpec) for t in tools)
            assert any(t.name == "final_answer" for t in tools)
            for m in kwargs["messages"]:
                assert isinstance(m.content, list)
                assert all(isinstance(b, ContentBlock) for b in m.content)
            return _tool_reply("final_answer", {"answer": "4"})

    action = ReActPolicy().synthesize_on_exhaustion(
        state, llm=_ValidatingLLM(), tools=_final_answer_tools(),
    )
    assert action is not None
    assert action.answer == "4"
    assert (action.meta or {}).get("synthesis") == "llm"
