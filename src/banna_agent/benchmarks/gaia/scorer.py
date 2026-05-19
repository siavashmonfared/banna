"""GAIA exact-match scorer.

Implements the GAIA scoring rules from Mialon et al. 2023 (with small,
documented deviations where the official rules are ambiguous):

1. **Strings** — normalize both predicted and gold:
     - lowercase
     - strip leading/trailing whitespace
     - strip surrounding quotes
     - drop leading articles (a, an, the)
     - drop punctuation (except decimals inside numbers)
     - collapse internal whitespace

2. **Numbers** — if both sides parse as numeric, compare as floats with
     a small relative tolerance. Handles commas, $, %, and signs.

3. **Lists** — if the gold answer contains ',' or ';' at the top level,
     compare as sets of normalized elements. Order-insensitive. This is
     GAIA's canonical behavior for "list" answer types.

4. **Yes/No** — mapped to {yes, no}; "true/false" aliased to yes/no.

The scorer returns a structured `GAIAScore` — `is_correct` for the
overall pass/fail, plus telemetry (`match_kind`, `normalized_pred`,
`normalized_gold`) that feeds the failure taxonomy in week 2.
"""
from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


_ARTICLES = {"a", "an", "the"}
_YES = {"yes", "true", "y", "correct", "affirmative"}
_NO = {"no", "false", "n", "incorrect", "negative"}


# Characters that we strip from the non-numeric normalized form.
# We keep periods because they appear inside numbers; we strip them only
# when they are clearly sentence terminators (end-of-string).
_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation if c != "."})


def _strip_outer_quotes(s: str) -> str:
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'`":
        return s[1:-1]
    return s


def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _drop_articles(s: str) -> str:
    return " ".join(tok for tok in s.split() if tok not in _ARTICLES)


def normalize_string(s: str) -> str:
    """Canonicalize a string answer for comparison."""
    if s is None:
        return ""
    out = str(s).strip()
    out = _strip_outer_quotes(out)
    out = out.lower()
    # Drop trailing period if the whole thing is a sentence.
    if out.endswith("."):
        out = out[:-1].rstrip()
    out = out.translate(_PUNCT_TABLE)
    out = _drop_articles(out)
    out = _collapse_ws(out)
    return out


def normalize_yes_no(s: str) -> str | None:
    """Return 'yes'/'no' if `s` is a known yes/no token; else None."""
    n = normalize_string(s)
    if n in _YES:
        return "yes"
    if n in _NO:
        return "no"
    return None


_NUMERIC_RE = re.compile(r"^[-+]?[\d,]*\.?\d+(?:[eE][-+]?\d+)?%?$")


def parse_number(s: str) -> float | None:
    """Try to interpret `s` as a number. Strips $, commas, %, whitespace.
    Returns None if the string isn't a numeric literal."""
    if s is None:
        return None
    t = str(s).strip()
    if not t:
        return None
    # Strip outer quotes first.
    t = _strip_outer_quotes(t)
    # Currency symbols at start.
    t = t.lstrip("$€£¥ ")
    # Trailing percent handled below.
    is_pct = t.endswith("%")
    if is_pct:
        t = t[:-1].strip()
    # Remove grouping commas.
    t_clean = t.replace(",", "")
    if not _NUMERIC_RE.match(t_clean + ("%" if is_pct else "")):
        # Relax: allow plain floats after our own cleanup.
        try:
            val = float(t_clean)
            return val / 100.0 if is_pct else val
        except ValueError:
            return None
    try:
        val = float(t_clean)
    except ValueError:
        return None
    return val / 100.0 if is_pct else val


def _split_list(s: str) -> list[str] | None:
    """Return the list parts if `s` is a list answer, else None.
    A list answer has ≥2 parts separated by top-level commas or semicolons."""
    if not s:
        return None
    # Prefer semicolons when present (less ambiguous).
    sep = ";" if ";" in s else ("," if s.count(",") >= 1 else None)
    if sep is None:
        return None
    parts = [p.strip() for p in s.split(sep)]
    parts = [p for p in parts if p]
    if len(parts) < 2:
        return None
    return parts


# ---------------------------------------------------------------------------
# Score dataclass
# ---------------------------------------------------------------------------


@dataclass
class GAIAScore:
    is_correct: bool
    match_kind: str          # "numeric" | "yes_no" | "list" | "string" | "empty"
    normalized_pred: Any
    normalized_gold: Any
    pred: str
    gold: str
    reason: str = ""


# ---------------------------------------------------------------------------
# The scorer
# ---------------------------------------------------------------------------


NUMERIC_TOLERANCE_RELATIVE = 1e-3   # 0.1% relative tolerance
NUMERIC_TOLERANCE_ABSOLUTE = 1e-6


def score_gaia(pred: str, gold: str) -> GAIAScore:
    """Grade a predicted answer against the gold truth."""
    pred_raw = "" if pred is None else str(pred)
    gold_raw = "" if gold is None else str(gold)

    # Empty-prediction shortcut.
    if not pred_raw.strip():
        return GAIAScore(
            is_correct=False,
            match_kind="empty",
            normalized_pred="",
            normalized_gold=normalize_string(gold_raw),
            pred=pred_raw,
            gold=gold_raw,
            reason="empty prediction",
        )

    # 1. Yes/No.
    gold_yn = normalize_yes_no(gold_raw)
    if gold_yn is not None:
        pred_yn = normalize_yes_no(pred_raw)
        return GAIAScore(
            is_correct=(pred_yn == gold_yn),
            match_kind="yes_no",
            normalized_pred=pred_yn,
            normalized_gold=gold_yn,
            pred=pred_raw,
            gold=gold_raw,
            reason="yes/no compare",
        )

    # 2. Numeric.
    gold_num = parse_number(gold_raw)
    pred_num = parse_number(pred_raw)
    if gold_num is not None and pred_num is not None:
        diff = abs(pred_num - gold_num)
        tol = max(NUMERIC_TOLERANCE_ABSOLUTE,
                  abs(gold_num) * NUMERIC_TOLERANCE_RELATIVE)
        ok = diff <= tol
        return GAIAScore(
            is_correct=ok,
            match_kind="numeric",
            normalized_pred=pred_num,
            normalized_gold=gold_num,
            pred=pred_raw,
            gold=gold_raw,
            reason=f"numeric compare; diff={diff:.6g}, tol={tol:.6g}",
        )

    # 3. List.
    gold_list = _split_list(gold_raw)
    if gold_list is not None:
        pred_list = _split_list(pred_raw) or [pred_raw]
        gold_norm = sorted({normalize_string(p) for p in gold_list})
        pred_norm = sorted({normalize_string(p) for p in pred_list})
        ok = gold_norm == pred_norm
        return GAIAScore(
            is_correct=ok,
            match_kind="list",
            normalized_pred=pred_norm,
            normalized_gold=gold_norm,
            pred=pred_raw,
            gold=gold_raw,
            reason="set compare on normalized elements",
        )

    # 4. String fallback.
    pred_norm = normalize_string(pred_raw)
    gold_norm = normalize_string(gold_raw)
    ok = pred_norm == gold_norm
    return GAIAScore(
        is_correct=ok,
        match_kind="string",
        normalized_pred=pred_norm,
        normalized_gold=gold_norm,
        pred=pred_raw,
        gold=gold_raw,
        reason="normalized string compare",
    )
