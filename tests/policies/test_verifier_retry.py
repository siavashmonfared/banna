"""VerifierRetryPolicy tests — wraps an inner policy and rejects bad answers."""
from __future__ import annotations

from dataclasses import dataclass


from banna_agent.core.state import AgentState
from banna_agent.core.types import Action, ActionKind, Claim
from banna_agent.policies.verifier_retry import VerifierRetryPolicy
from banna_agent.tools.base import ToolRegistry
from banna_agent.verifiers.base import ANSWER_CLAIM_ID, ClaimCheck


@dataclass
class _ScriptedInner:
    """Inner policy that returns a fixed sequence of Actions."""

    actions: list
    name: str = "scripted"
    calls: int = 0

    def propose(self, state, *, llm, tools):
        self.calls += 1
        if not self.actions:
            return Action(kind=ActionKind.THINK, text="(no more)")
        return self.actions.pop(0)


@dataclass
class _BadVerifier:
    """Always fails the proposed answer."""
    name: str = "bad"

    def check(self, state, proposed_answer=None):
        if proposed_answer is None:
            return []
        return [ClaimCheck(
            claim_id=ANSWER_CLAIM_ID, verifier_name=self.name,
            verdict="fail", detail="bad on principle",
        )]


@dataclass
class _GoodVerifier:
    name: str = "good"

    def check(self, state, proposed_answer=None):
        return []


# ---------------------------------------------------------------------------
# Behavior
# ---------------------------------------------------------------------------


def test_passes_through_non_final_actions() -> None:
    inner_act = Action(kind=ActionKind.TOOL_CALL, tool_name="search",
                       tool_args={"q": "x"})
    inner = _ScriptedInner(actions=[inner_act])
    pol = VerifierRetryPolicy(inner=inner, verifiers=[_BadVerifier()])
    out = pol.propose(AgentState(question="?"), llm=None, tools=ToolRegistry())
    assert out.kind == ActionKind.TOOL_CALL


def test_accepts_final_when_all_verifiers_pass() -> None:
    inner = _ScriptedInner(actions=[Action(kind=ActionKind.FINAL_ANSWER, answer="42")])
    pol = VerifierRetryPolicy(inner=inner, verifiers=[_GoodVerifier()])
    out = pol.propose(AgentState(question="?"), llm=None, tools=ToolRegistry())
    assert out.kind == ActionKind.FINAL_ANSWER
    assert out.answer == "42"
    assert out.meta["verifier_passed"] is True


def test_rejects_final_when_any_verifier_fails() -> None:
    inner = _ScriptedInner(actions=[Action(kind=ActionKind.FINAL_ANSWER, answer="42")])
    pol = VerifierRetryPolicy(inner=inner, verifiers=[_BadVerifier()])
    state = AgentState(question="?")
    out = pol.propose(state, llm=None, tools=ToolRegistry())
    assert out.kind == ActionKind.THINK
    assert out.meta["verifier_retry"] is True
    assert out.meta["retry_index"] == 1
    # State carries the retry counter forward.
    assert state.metadata["verifier_retry"]["retries"] == 1


def test_max_retries_caps_loop() -> None:
    state = AgentState(question="?")
    state.metadata["verifier_retry"] = {"retries": 3}  # already exhausted
    inner = _ScriptedInner(actions=[Action(kind=ActionKind.FINAL_ANSWER, answer="42")])
    pol = VerifierRetryPolicy(inner=inner, verifiers=[_BadVerifier()],
                               max_retries=3)
    out = pol.propose(state, llm=None, tools=ToolRegistry())
    assert out.kind == ActionKind.FINAL_ANSWER  # accepted under cap
    assert out.meta["verifier_retries_exhausted"] is True


def test_retry_count_persists_across_ticks() -> None:
    state = AgentState(question="?")
    inner = _ScriptedInner(actions=[
        Action(kind=ActionKind.FINAL_ANSWER, answer="bad1"),
        Action(kind=ActionKind.FINAL_ANSWER, answer="bad2"),
        Action(kind=ActionKind.FINAL_ANSWER, answer="bad3"),
        Action(kind=ActionKind.FINAL_ANSWER, answer="bad4"),
    ])
    pol = VerifierRetryPolicy(inner=inner, verifiers=[_BadVerifier()],
                               max_retries=2)
    # Tick 1 — fail, return THINK
    a1 = pol.propose(state, llm=None, tools=ToolRegistry())
    assert a1.kind == ActionKind.THINK
    # Tick 2 — fail, return THINK
    a2 = pol.propose(state, llm=None, tools=ToolRegistry())
    assert a2.kind == ActionKind.THINK
    # Tick 3 — cap hit, accept the bad answer
    a3 = pol.propose(state, llm=None, tools=ToolRegistry())
    assert a3.kind == ActionKind.FINAL_ANSWER
    assert a3.meta["verifier_retries_exhausted"] is True


def test_failures_carried_in_meta() -> None:
    inner = _ScriptedInner(actions=[Action(kind=ActionKind.FINAL_ANSWER, answer="42")])
    pol = VerifierRetryPolicy(inner=inner, verifiers=[_BadVerifier()])
    out = pol.propose(AgentState(question="?"), llm=None, tools=ToolRegistry())
    assert "failures" in out.meta
    assert out.meta["failures"][0]["verifier_name"] == "bad"
    assert out.meta["failures"][0]["verdict"] == "fail"


def test_inner_tokens_carried_through_retry() -> None:
    inner_act = Action(kind=ActionKind.FINAL_ANSWER, answer="42",
                        meta={"tokens_in": 50, "tokens_out": 10})
    inner = _ScriptedInner(actions=[inner_act])
    pol = VerifierRetryPolicy(inner=inner, verifiers=[_BadVerifier()])
    out = pol.propose(AgentState(question="?"), llm=None, tools=ToolRegistry())
    assert out.meta["tokens_in"] == 50
    assert out.meta["tokens_out"] == 10


# ---------------------------------------------------------------------------
# C6: per-verifier actionable nudges in _format_feedback
# ---------------------------------------------------------------------------


@dataclass
class _NudgingVerifier:
    """Verifier that supplies a `nudge` string in meta on fail."""
    name: str = "nudgey"
    nudge: str = "do the thing"

    def check(self, state, proposed_answer=None):
        if proposed_answer is None:
            return []
        return [ClaimCheck(
            claim_id=ANSWER_CLAIM_ID, verifier_name=self.name,
            verdict="fail", detail="x",
            meta={"nudge": self.nudge},
        )]


def test_format_feedback_includes_per_verifier_nudge() -> None:
    """The THINK feedback text contains the verifier-supplied nudge,
    grouped by verifier name."""
    inner = _ScriptedInner(actions=[Action(kind=ActionKind.FINAL_ANSWER, answer="42")])
    pol = VerifierRetryPolicy(
        inner=inner,
        verifiers=[_NudgingVerifier(name="arith", nudge="recompute step 3")],
    )
    out = pol.propose(AgentState(question="?"), llm=None, tools=ToolRegistry())
    assert out.kind == ActionKind.THINK
    assert "arith" in out.text
    assert "recompute step 3" in out.text


def test_format_feedback_groups_by_verifier_one_per_kind() -> None:
    """Multiple failures from the same verifier collapse to one nudge."""
    @dataclass
    class _TwoFails:
        name: str = "arith"
        def check(self, state, proposed_answer=None):
            if proposed_answer is None: return []
            return [
                ClaimCheck(claim_id="c1", verifier_name=self.name,
                           verdict="fail", detail="a", meta={"nudge": "first"}),
                ClaimCheck(claim_id="c2", verifier_name=self.name,
                           verdict="fail", detail="b", meta={"nudge": "second"}),
            ]
    inner = _ScriptedInner(actions=[Action(kind=ActionKind.FINAL_ANSWER, answer="42")])
    pol = VerifierRetryPolicy(inner=inner, verifiers=[_TwoFails()])
    out = pol.propose(AgentState(question="?"), llm=None, tools=ToolRegistry())
    # Only the first nudge per verifier appears (preserves brevity).
    assert "first" in out.text
    assert "second" not in out.text


def test_format_feedback_canonical_emits_required_action() -> None:
    """If a verifier supplies a `canonical` answer string, the feedback
    ends with the strong REQUIRED ACTION directive."""
    @dataclass
    class _Canonical:
        name: str = "format"
        def check(self, state, proposed_answer=None):
            if proposed_answer is None: return []
            return [ClaimCheck(
                claim_id=ANSWER_CLAIM_ID, verifier_name=self.name,
                verdict="fail", detail="bad shape",
                meta={"canonical": "42", "nudge": "use canonical"},
            )]
    inner = _ScriptedInner(actions=[Action(kind=ActionKind.FINAL_ANSWER, answer="forty-two")])
    pol = VerifierRetryPolicy(inner=inner, verifiers=[_Canonical()])
    out = pol.propose(AgentState(question="?"), llm=None, tools=ToolRegistry())
    assert "REQUIRED ACTION" in out.text
    assert "'42'" in out.text


def test_arithmetic_verifier_emits_nudge_with_recomputed_value() -> None:
    """ArithmeticVerifier fail meta should contain a nudge naming the
    recomputed value so the model knows what to write next."""
    from banna_agent.verifiers.arithmetic import ArithmeticVerifier
    state = AgentState(question="?")
    state.claims.append(Claim(text="47 * 83 = 3801"))  # actual is 3901
    checks = ArithmeticVerifier().check(state)
    fails = [c for c in checks if c.verdict == "fail"]
    assert fails
    nudge = (fails[0].meta or {}).get("nudge", "")
    assert "3901" in nudge
    assert "3801" in nudge
    assert "final_answer" in nudge


def test_citation_verifier_emits_nudge_for_broken_id() -> None:
    """A citation pointing at a non-existent evidence_id yields a
    fail with a nudge that tells the model to use real IDs."""
    from banna_agent.verifiers.citation import CitationVerifier
    state = AgentState(question="?")
    state.claims.append(Claim(text="Paris is the capital", supports=["ev_does_not_exist"]))
    checks = CitationVerifier().check(state)
    fails = [c for c in checks if c.verdict == "fail"]
    assert fails
    nudge = (fails[0].meta or {}).get("nudge", "")
    assert "evidence_ids" in nudge
    assert "ev_does_not_exist" in nudge


def test_coverage_verifier_emits_nudge_for_unsupported_factual_claim() -> None:
    from banna_agent.verifiers.coverage import CoverageVerifier
    state = AgentState(question="?")
    state.claims.append(Claim(text="The Eiffel Tower is in Paris."))  # factual, no supports
    checks = CoverageVerifier().check(state)
    fails = [c for c in checks if c.verdict == "fail"]
    assert fails
    nudge = (fails[0].meta or {}).get("nudge", "")
    assert "search" in nudge or "read_url" in nudge
    assert "evidence_id" in nudge


def test_format_verifier_empty_answer_emits_nudge() -> None:
    from banna_agent.verifiers.format import FormatVerifier
    state = AgentState(question="?")
    checks = FormatVerifier().check(state, proposed_answer="")
    assert checks and checks[0].verdict == "fail"
    nudge = (checks[0].meta or {}).get("nudge", "")
    assert "final_answer" in nudge
