"""ArithmeticVerifier — re-evaluate numeric claims independently.

Scope (v1):
  * Look at every Claim and (optionally) the proposed FINAL_ANSWER.
  * If a claim asserts an arithmetic equality of the form
        "<expr> = <value>"   or   "<expr> equals <value>"   etc.,
    re-evaluate `<expr>` with a tiny safe AST evaluator and compare to
    the asserted `<value>` within a small relative tolerance.
  * Verdicts:
      ok   → re-eval matches the asserted value
      fail → re-eval mismatches by more than the tolerance
      skip → no parseable equality was found in the claim
      warn → expression parsed but evaluation crashed

The "safe" evaluator handles +, -, *, /, **, %, unary -, and bare
numbers / parentheses. Names, function calls, attribute access, etc.
are rejected — we don't run the model's code, we just check its math.
"""
from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass
from typing import Any

from ..core.state import AgentState
from .base import ANSWER_CLAIM_ID, ClaimCheck


# ---------------------------------------------------------------------------
# Safe arithmetic evaluator (allowlist-only AST walk)
# ---------------------------------------------------------------------------


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class UnsafeExpression(ValueError):
    """Raised when an expression contains an unsupported AST node."""


def safe_eval_arith(expr: str) -> float:
    """Evaluate a numeric expression. Raises if anything outside the
    arithmetic allowlist appears.

    Examples:
      safe_eval_arith("47 * 83 + 11")  # → 3912
      safe_eval_arith("2 ** 10")       # → 1024
      safe_eval_arith("__import__('os')")  # raises UnsafeExpression
    """
    tree = ast.parse(expr.strip(), mode="eval")
    return float(_eval_node(tree.body))


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise UnsafeExpression(f"non-numeric constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise UnsafeExpression(f"unsupported binary op: {type(node.op).__name__}")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise UnsafeExpression(f"unsupported unary op: {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    raise UnsafeExpression(f"unsupported AST node: {type(node).__name__}")


# ---------------------------------------------------------------------------
# Equality extraction
# ---------------------------------------------------------------------------


# `<lhs> = <rhs>`  / `<lhs> equals <rhs>` / `<lhs> is <rhs>`
# We require RHS to look numeric; LHS we'll try to safe_eval.
_EQ_RE = re.compile(
    r"""(?P<lhs>[-+(]?\s*\d[\d\s+\-*/%().eE^]*)\s*
        (?:=|==|equals|is\ equal\ to|is)\s*
        (?P<rhs>[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?)
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _try_extract_equality(text: str) -> tuple[str, float] | None:
    """Pull the FIRST arithmetic equality out of free text. Convenience
    wrapper around :func:`_extract_all_equalities`; preserved so older
    callers keep working."""
    found = _extract_all_equalities(text)
    return found[0] if found else None


def _extract_all_equalities(text: str) -> list[tuple[str, float]]:
    """Pull every arithmetic equality out of free text.

    Returns a list of (lhs_expr_str, rhs_value). The model's reasoning
    often contains multiple arithmetic steps ("17054.888 / 1000 = 17 ;
    then 17 * 2 = 34"); we verify all of them so a single bogus link
    in the chain shows up as a fail.
    """
    if not text:
        return []
    out: list[tuple[str, float]] = []
    for m in _EQ_RE.finditer(text):
        lhs = m.group("lhs").strip()
        rhs_str = m.group("rhs").replace(",", "")
        # Caret '^' often means power in everyday writing — translate to **.
        lhs_norm = lhs.replace("^", "**")
        if not any(op in lhs_norm for op in ("+", "-", "*", "/", "%", "**")):
            continue
        try:
            rhs_val = float(rhs_str)
        except ValueError:
            continue
        out.append((lhs_norm, rhs_val))
    return out


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


@dataclass
class ArithmeticVerifier:
    """Recompute numeric assertions and compare to what was claimed."""

    name: str = "arithmetic"
    rel_tol: float = 1e-3
    abs_tol: float = 1e-6

    def check(
        self,
        state: AgentState,
        proposed_answer: str | None = None,
    ) -> list[ClaimCheck]:
        out: list[ClaimCheck] = []

        # Claims
        for cl in state.claims:
            extracted = _try_extract_equality(cl.text)
            out.append(self._verdict_for(extracted, cl.claim_id, cl.text))

        # Proposed answer (only when it includes an explicit equality;
        # a bare number like "42" is verified by FormatVerifier, not here).
        if proposed_answer:
            extracted = _try_extract_equality(proposed_answer)
            if extracted is not None:
                out.append(self._verdict_for(extracted, ANSWER_CLAIM_ID, proposed_answer))

        # Phase 8: also verify arithmetic in the model's *reasoning* on
        # the most recent FINAL_ANSWER step. The model's chain-of-thought
        # often contains multiple equalities ("raw=17054.888 / 1000 = 17,
        # then rounded to 17"); a wrong link there is independently
        # gradable even when the answer string itself is just a number.
        # ALL equalities found are checked — a single bogus step trips
        # the verifier even if downstream steps would round to something
        # plausible.
        last = state.trace.steps[-1] if state.trace.steps else None
        if last is not None and last.action.kind.value == "final_answer":
            reasoning = (last.action.text or "").strip()
            # Avoid double-checking when reasoning == answer (already
            # handled via proposed_answer above).
            if reasoning and reasoning != (proposed_answer or "").strip():
                for i, eq in enumerate(_extract_all_equalities(reasoning)):
                    out.append(self._verdict_for(
                        eq,
                        f"{ANSWER_CLAIM_ID}.reasoning[{i}]",
                        reasoning[:160],
                    ))

        return out

    # ------------------------------------------------------------------

    def _verdict_for(
        self,
        extracted: tuple[str, float] | None,
        claim_id: str,
        original_text: str,
    ) -> ClaimCheck:
        if extracted is None:
            return ClaimCheck(
                claim_id=claim_id,
                verifier_name=self.name,
                verdict="skip",
                detail="no parseable equality",
            )
        lhs_expr, rhs_val = extracted
        try:
            recomputed = safe_eval_arith(lhs_expr)
        except (UnsafeExpression, SyntaxError, ZeroDivisionError, OverflowError) as exc:
            return ClaimCheck(
                claim_id=claim_id,
                verifier_name=self.name,
                verdict="warn",
                detail=f"could not re-evaluate {lhs_expr!r}: {type(exc).__name__}",
                meta={"lhs": lhs_expr, "rhs_asserted": rhs_val},
            )
        diff = abs(recomputed - rhs_val)
        tol = max(self.abs_tol, abs(rhs_val) * self.rel_tol)
        ok = diff <= tol
        meta: dict[str, Any] = {
            "lhs": lhs_expr,
            "recomputed": recomputed,
            "rhs_asserted": rhs_val,
            "diff": diff,
            "tol": tol,
        }
        if not ok:
            meta["nudge"] = (
                f"Your reasoning asserts `{lhs_expr} = {rhs_val:g}` but a "
                f"safe re-evaluation gives `{recomputed:g}` (|diff|="
                f"{diff:g}). Recompute the step, then re-emit "
                f"`final_answer` with the corrected value."
            )
        return ClaimCheck(
            claim_id=claim_id,
            verifier_name=self.name,
            verdict="ok" if ok else "fail",
            detail=(
                f"{lhs_expr} → {recomputed:g} ; asserted {rhs_val:g} ; "
                f"|diff|={diff:g} ; tol={tol:g}"
            ),
            score=1.0 if ok else 0.0,
            meta=meta,
        )
