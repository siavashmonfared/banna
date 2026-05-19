"""Shared planning primitives.

Four policies in this module family (Planner-ReAct, BFS, DFS, best-first)
all need to *propose a plan* from the LLM, then *score* or *execute* its
steps. The proposal logic is identical; only the search over the
resulting plans differs. Extracted here once.

A `Plan` is an ordered list of natural-language subquestion strings.
Each step corresponds to one ReAct-shaped sub-goal: the executor runs
its own inner loop to resolve that subquestion into evidence and, if
the step is the *last*, a final answer.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..core.state import AgentState
from ..llm.base import ContentBlock, LLMClient, Message


DEFAULT_PLANNER_SYSTEM = (
    "You are a research planner. Decompose a question into a short, ordered "
    "list of concrete subquestions. Each subquestion should be answerable "
    "with a single tool call (search, read_url, calculator, read_file, "
    "run_python) plus one short reasoning step. The last subquestion should "
    "produce the final answer.\n\n"
    "Return ONLY a JSON object with this exact shape:\n"
    '  {"plan": ["subquestion 1", "subquestion 2", "..."]}\n\n'
    "Keep the plan to 2-6 steps. Prefer specific phrasing over vague steps."
)


DEFAULT_CANDIDATES_SYSTEM = (
    "You are a research planner. Propose N *distinct* plans to answer the "
    "question, each as an ordered list of 2-6 concrete subquestions. Plans "
    "should differ in strategy or angle — not just phrasing.\n\n"
    "Return ONLY a JSON object with this shape:\n"
    '  {"plans": [["subq a1", "subq a2", "..."], ["subq b1", "subq b2", "..."], ...]}\n\n'
    "Number of plans: exactly {n_candidates}."
)


@dataclass
class Plan:
    """An ordered list of subquestion strings."""

    steps: list[str]
    # Populated as execution proceeds.
    step_results: list[dict[str, Any]] = field(default_factory=list)
    # Final answer once the last step is resolved (or None).
    final_answer: str | None = None
    # Free-form metadata: provider, model, branch id, scores, etc.
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def depth(self) -> int:
        return len(self.steps)

    @property
    def is_done(self) -> bool:
        return self.final_answer is not None

    def with_step_result(self, idx: int, result: dict[str, Any]) -> None:
        while len(self.step_results) <= idx:
            self.step_results.append({})
        self.step_results[idx] = result


# ---------------------------------------------------------------------------
# LLM-driven plan proposal
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of `text`, tolerating code fences."""
    if not text:
        return None
    # Try a code fence first.
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = m.group(1) if m else None
    if candidate is None:
        # Fall back to the first brace-balanced span.
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    break
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except (ValueError, TypeError):
        return None


def propose_plan(
    llm: LLMClient,
    question: str,
    *,
    system: str = DEFAULT_PLANNER_SYSTEM,
    model: str | None = None,
    max_tokens: int = 600,
    temperature: float = 0.0,
) -> Plan:
    """Ask the LLM for a single plan. Returns Plan(steps=[...]) or an empty
    Plan on parse failure."""
    kwargs: dict[str, Any] = {
        "messages": [Message(role="user", content=[ContentBlock(kind="text", text=question)])],
        "system": system,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if model:
        kwargs["model"] = model
    try:
        reply = llm.chat(**kwargs)
    except Exception as exc:
        return Plan(steps=[], meta={"error": f"{type(exc).__name__}: {exc}"})
    data = _extract_json(reply.text)
    if not data or "plan" not in data:
        return Plan(steps=[], meta={"raw": reply.text[:200], "parse_fail": True})
    raw = data.get("plan") or []
    steps = [str(s).strip() for s in raw if isinstance(s, str) and str(s).strip()]
    return Plan(
        steps=steps,
        meta={
            "provider": reply.provider,
            "model": reply.model,
            "tokens_in": reply.usage.tokens_in,
            "tokens_out": reply.usage.tokens_out,
        },
    )


def propose_candidate_plans(
    llm: LLMClient,
    question: str,
    *,
    n_candidates: int = 3,
    model: str | None = None,
    max_tokens: int = 1000,
    temperature: float = 0.7,
) -> list[Plan]:
    """Ask the LLM for N distinct plans. Returns list of Plan."""
    system = DEFAULT_CANDIDATES_SYSTEM.replace("{n_candidates}", str(n_candidates))
    kwargs: dict[str, Any] = {
        "messages": [Message(role="user", content=[ContentBlock(kind="text", text=question)])],
        "system": system,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if model:
        kwargs["model"] = model
    try:
        reply = llm.chat(**kwargs)
    except Exception as exc:
        return [Plan(steps=[], meta={"error": f"{type(exc).__name__}: {exc}"})]
    data = _extract_json(reply.text)
    if not data or "plans" not in data:
        return []
    out: list[Plan] = []
    for i, plan_raw in enumerate(data.get("plans") or []):
        if not isinstance(plan_raw, list):
            continue
        steps = [str(s).strip() for s in plan_raw if isinstance(s, str) and str(s).strip()]
        if not steps:
            continue
        out.append(Plan(
            steps=steps,
            meta={
                "branch_id": i,
                "provider": reply.provider,
                "model": reply.model,
                "tokens_in": reply.usage.tokens_in,
                "tokens_out": reply.usage.tokens_out,
            },
        ))
    return out
