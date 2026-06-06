"""VerifierRetryPolicy — wrap any policy with a verifier feedback loop.

The policy delegates `propose` to an inner policy (ReAct by default).
When the inner policy proposes `FINAL_ANSWER`, this wrapper:

  1. Runs every configured verifier against `state` with the proposed
     answer.
  2. If at least one verifier returns `fail`, the wrapper *swaps* the
     FINAL_ANSWER for a THINK action that summarizes the failures.
     The driver appends that THINK to the trace and ticks again — the
     inner policy then sees its own bad answer + the verifier feedback
     and (hopefully) revises.
  3. Repeats until verifiers pass *or* `max_retries` is reached.

State is carried via `state.metadata["verifier_retry"]`. We do *not*
need to invent a new ActionKind: the loop is driven entirely by the
existing THINK → next-tick mechanic in `run_policy`.

Why a wrapper, not a new driver:
  * Composes with any inner policy. `verifier_retry(react)`,
    `verifier_retry(planner_react)`, etc. all work.
  * Survives budget checks naturally — each retry is one tick.
  * Failures + retry count are visible in the event log + meta, so the
    failure taxonomy (week-2 deliverable) reads them straight off.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.state import AgentState
from ..core.types import Action, ActionKind
from ..llm.base import LLMClient
from ..tools.base import ToolRegistry
from ..verifiers.base import (
    Verifier,
    apply_to_state,
    default_verifiers,
    has_failures,
    run_all,
    summarize,
)
from .react import ReActPolicy


@dataclass
class VerifierRetryPolicy:
    """Inner policy + verifier feedback loop.

    Fields:
      name        - identifier used in events / ablation tables
      inner       - the underlying policy (ReAct by default)
      verifiers   - the checks to run on FINAL_ANSWER
      max_retries - cap on retry-THINKs; once hit, the wrapper accepts
                    the inner's FINAL_ANSWER even if verifiers fail
    """

    name: str = "verifier_retry"
    inner: Any = field(default_factory=ReActPolicy)
    verifiers: list[Verifier] = field(default_factory=default_verifiers)
    max_retries: int = 3

    # ------------------------------------------------------------------
    # Policy.propose
    # ------------------------------------------------------------------

    def propose(
        self,
        state: AgentState,
        *,
        llm: LLMClient,
        tools: ToolRegistry,
    ) -> Action:
        action = self.inner.propose(state, llm=llm, tools=tools)

        # Only intercept FINAL_ANSWER. Tool calls and THINKs pass through.
        if action.kind != ActionKind.FINAL_ANSWER:
            return action

        # Stash the LLM client on state.metadata so verifiers that need
        # an LLM (ReflexionVerifier, future LLM-extraction verifiers)
        # can pick it up. Cleared after run_all to avoid leaking it
        # across ticks. Backward-compatible: existing verifiers ignore it.
        state.metadata["_verifier_llm"] = llm
        try:
            return self._propose_with_verifiers(state, action)
        finally:
            state.metadata.pop("_verifier_llm", None)

    def _propose_with_verifiers(self, state: AgentState, action: Action) -> Action:
        meta = state.metadata.setdefault("verifier_retry", {})
        retries_so_far = int(meta.get("retries", 0))

        # Budget cap: once we've spent max_retries, accept whatever the
        # inner policy proposed even if verifiers still flag it. Better
        # to ship a flagged answer than loop forever.
        if retries_so_far >= self.max_retries:
            checks = run_all(self.verifiers, state, proposed_answer=action.answer)
            apply_to_state(state, checks)
            verdicts = summarize(checks)
            action.meta = dict(action.meta)
            action.meta.update({
                "policy": self.name,
                "verifier_retries": retries_so_far,
                "verifier_verdicts": verdicts,
                "verifier_retries_exhausted": True,
            })
            return action

        # Run verifiers.
        checks = run_all(self.verifiers, state, proposed_answer=action.answer)
        verdicts = summarize(checks)

        if not has_failures(checks):
            # All clear — accept the answer.
            action.meta = dict(action.meta)
            action.meta.update({
                "policy": self.name,
                "verifier_retries": retries_so_far,
                "verifier_verdicts": verdicts,
                "verifier_passed": True,
            })
            return action

        # At least one fail — convert to feedback THINK.
        meta["retries"] = retries_so_far + 1
        feedback = self._format_feedback(action.answer or "", checks)
        return Action(
            kind=ActionKind.THINK,
            text=feedback,
            meta={
                "policy": self.name,
                "verifier_retry": True,
                "repair": True,
                "retry_index": retries_so_far + 1,
                "verdicts": verdicts,
                "failures": [
                    c.to_dict() for c in checks if c.verdict == "fail"
                ],
                # Carry the inner policy's tokens through so the budget
                # tracker still sees the LLM cost of the discarded answer.
                "tokens_in": int(action.meta.get("tokens_in") or 0),
                "tokens_out": int(action.meta.get("tokens_out") or 0),
            },
        )

    # ------------------------------------------------------------------
    # Hook delegation
    # ------------------------------------------------------------------

    def synthesize_on_exhaustion(
        self,
        state: AgentState,
        **kwargs: Any,
    ) -> Any:
        """Delegate budget-exhaustion synthesis to the inner policy.

        verifier_retry doesn't itself talk to the model — it wraps an
        inner policy that does. Best-effort commit is therefore the
        inner policy's job.
        """
        inner = getattr(self.inner, "synthesize_on_exhaustion", None)
        if not callable(inner):
            return None
        return inner(state, **kwargs)

    def on_run_end(self, state: AgentState, **kwargs: Any) -> Any:
        """Forward post-run hooks to the inner policy when present."""
        inner_hook = getattr(self.inner, "on_run_end", None)
        if inner_hook is None:
            return None
        # Default to our own verifiers when the caller didn't pass any —
        # we already know which set was used during the retry loop, and
        # the inner policy shouldn't have to redeclare them.
        kwargs.setdefault("verifiers", self.verifiers)
        return inner_hook(state, **kwargs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_feedback(answer: str, checks) -> str:
        """Build a THINK message the inner policy will see next tick.

        Groups failures by verifier and emits one actionable nudge per
        verifier — sourced from `ClaimCheck.meta["nudge"]` (populated by
        each verifier on fail). When any verifier exposes a `canonical`
        answer string, end with a strong REQUIRED ACTION directive
        naming that string. Otherwise end with a generic re-emit prompt.
        """
        failures = [c for c in checks if c.verdict == "fail"]
        n_fail = len(failures)

        # One nudge per verifier (the most informative one we saw).
        by_verifier: dict[str, str] = {}
        canonical: str | None = None
        for c in failures:
            m = c.meta or {}
            v = c.verifier_name or "verifier"
            nudge = m.get("nudge")
            if isinstance(nudge, str) and nudge.strip() and v not in by_verifier:
                by_verifier[v] = nudge.strip()
            if canonical is None:
                cand = m.get("canonical")
                if isinstance(cand, str) and cand.strip():
                    canonical = cand.strip()

        lines: list[str] = [
            f"[verifier_retry] Previous answer ({answer[:80]!r}) rejected by "
            f"{n_fail} verifier(s).",
        ]
        if by_verifier:
            lines.append("Issues:")
            for vname, nudge in by_verifier.items():
                lines.append(f"  - {vname}: {nudge}")
        else:
            # Fall back to terse `detail` strings if no verifier supplied
            # a nudge (older verifiers, edge cases).
            lines.append("Failures:")
            for c in failures[:5]:
                lines.append(f"  - {c.verifier_name}: {c.detail}")

        lines.append("")
        if canonical is not None:
            lines.append(
                f"REQUIRED ACTION: Call `final_answer` with "
                f"answer={canonical!r} — exactly that string, nothing else. "
                f"Do NOT include any reasoning, preamble, framing, or "
                f"explanation in the `answer` field."
            )
        else:
            lines.append(
                "Address the issue(s) above, then call `final_answer` "
                "again with the corrected answer in the `answer` field. "
                "Put reasoning in the `reasoning` field, not in `answer`."
            )
        return "\n".join(lines)
