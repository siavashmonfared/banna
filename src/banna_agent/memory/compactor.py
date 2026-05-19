"""Trace compactor — summarize old steps to free context.

The driver checks `compactor.should_compact(state)` before each tick.
When triggered, `compact(state)` mutates `state.trace.steps` in place:
the oldest steps (everything but the last `keep_last_n_steps`) are
replaced by a single synthetic "summary" step whose observation text
holds an LLM-generated summary.

Why this shape:
- Summary happens *at tick boundaries*, never mid-LLM-call. No
  reentrancy.
- The compacted state is still a valid AgentState — policies / tools /
  verifiers continue to work.
- The summary step is explicitly marked `action.meta["compaction"]=True`
  so replay tools and verifiers can ignore or flag it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.state import AgentState
from ..core.types import Action, ActionKind, Observation
from ..llm.base import ContentBlock, LLMClient, Message


DEFAULT_SUMMARY_PROMPT = (
    "Summarize the following agent steps into a short paragraph. "
    "Preserve: (1) concrete evidence URLs or file paths, "
    "(2) numeric values found so far, (3) unresolved subquestions, "
    "(4) the current plan or strategy. "
    "Drop: repeated reasoning, redundant tool output, exploratory dead-ends. "
    "Be terse — 4–6 sentences max."
)


def approximate_token_count(text: str) -> int:
    """Cheap approximation: ~4 chars/token. Good enough for thresholding.

    If `tiktoken` is installed, we use it for a more accurate count.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


@dataclass
class CompactionConfig:
    enabled: bool = False
    threshold_tokens: int = 30_000
    keep_last_n_steps: int = 4
    summarizer_model: str | None = None
    max_summary_tokens: int = 400


class TraceCompactor:
    """Compactor policy bound to a configured LLM client.

    Construct one per run (it's cheap). Pass it to `run_policy(..., compactor=...)`.
    """

    def __init__(
        self,
        llm: LLMClient,
        config: CompactionConfig,
    ) -> None:
        self.llm = llm
        self.config = config

    # ---- tick-time predicate ----------------------------------------------

    def should_compact(self, state: AgentState) -> bool:
        if not self.config.enabled:
            return False
        if len(state.trace.steps) <= self.config.keep_last_n_steps:
            return False
        tokens = approximate_token_count(self._trace_text(state))
        return tokens >= self.config.threshold_tokens

    # ---- mutation --------------------------------------------------------

    def compact(self, state: AgentState) -> dict[str, Any]:
        """Replace old steps with a synthetic summary step. Returns a dict
        describing what happened (for the event log)."""
        keep = self.config.keep_last_n_steps
        if len(state.trace.steps) <= keep:
            return {"dropped_steps": 0, "kept_steps": len(state.trace.steps)}

        old_steps = state.trace.steps[:-keep]
        kept_steps = state.trace.steps[-keep:]

        summary_text = self._summarize(state.question, old_steps)

        summary_action = Action(
            kind=ActionKind.THINK,
            text=f"[compaction_summary] {summary_text}",
            meta={"compaction": True},
        )
        summary_obs = Observation(
            ok=True,
            text=summary_text,
            data={"compaction": True, "replaced_steps": len(old_steps)},
        )
        from ..core.types import Step  # local import to avoid cycles
        summary_step = Step(idx=0, action=summary_action, observation=summary_obs)

        # Reindex kept_steps so indices remain contiguous from 1.
        rebuilt: list[Any] = [summary_step]
        for i, st in enumerate(kept_steps, start=1):
            st.idx = i
            rebuilt.append(st)
        state.trace.steps = rebuilt

        return {
            "dropped_steps": len(old_steps),
            "kept_steps": len(kept_steps),
            "summary_chars": len(summary_text),
        }

    # ---- helpers ---------------------------------------------------------

    def _trace_text(self, state: AgentState) -> str:
        """Rough textual projection used only for the token estimator."""
        buf: list[str] = [state.question]
        for step in state.trace.steps:
            a = step.action
            o = step.observation
            if a.kind == ActionKind.THINK and a.text:
                buf.append(a.text)
            elif a.kind == ActionKind.TOOL_CALL:
                buf.append(f"{a.tool_name}({a.tool_args})")
                if o.data:
                    buf.append(str(o.data)[:2000])
            elif a.kind == ActionKind.FINAL_ANSWER:
                buf.append(a.answer or "")
        return "\n".join(buf)

    def _summarize(self, question: str, steps: list[Any]) -> str:
        projection: list[str] = [f"QUESTION: {question}", "", "STEPS:"]
        for step in steps:
            a = step.action
            o = step.observation
            if a.kind == ActionKind.THINK and a.text:
                projection.append(f"[think] {a.text[:400]}")
            elif a.kind == ActionKind.TOOL_CALL:
                args = str(a.tool_args)[:300]
                result = str(o.data)[:800] if o.data else (o.text or "")[:400]
                status = "ok" if o.ok else f"error:{o.error}"
                projection.append(f"[{a.tool_name}] args={args} → {status} result={result}")
            elif a.kind == ActionKind.FINAL_ANSWER:
                projection.append(f"[answer] {a.answer}")
        projection_text = "\n".join(projection)

        kwargs: dict[str, Any] = {
            "messages": [Message(
                role="user",
                content=[ContentBlock(kind="text", text=projection_text)],
            )],
            "max_tokens": self.config.max_summary_tokens,
            "temperature": 0.0,
            "system": DEFAULT_SUMMARY_PROMPT,
        }
        if self.config.summarizer_model:
            kwargs["model"] = self.config.summarizer_model
        try:
            reply = self.llm.chat(**kwargs)
            return reply.text.strip() or "(summarizer returned empty)"
        except Exception as exc:
            return f"(summarizer failed: {type(exc).__name__}: {exc})"
