"""FormatVerifier — proposed answer matches the question's implied shape,
*and* its surface form matches what GAIA's exact-match scorer will accept.

GAIA scoring is exact-match with normalization, so format mismatches
are a major source of *correct-but-marked-wrong* failures (the model
says "Forty-two" when the gold is "42", or "yes, definitely" when the
gold is "yes"). This verifier pattern-matches the question to detect
the expected answer shape *and* re-normalizes the raw answer through
the scorer's own pipeline; if the canonicalized form differs from what
the model produced, that's a format_mismatch fail with a suggested
canonical form.

Patterns we detect:
    yes/no question  → answer must be in {yes, no, true, false}
    "how many" / "what is the number" → answer parses as int or float
    "list" / "list of" / "all the" → answer has ≥2 items (commas / newlines)
    "year" / "in what year" → answer is 4 digits in 1000–2099
    "what year was X" → year shape

Verdicts:
    ok   → answer matches the detected shape and the canonical form
    fail → answer is wrong shape OR not in canonical form
    skip → no clear shape detected (or no proposed answer to grade)

A separate top-level helper, ``finalize_answer(text, question="")``, is
the function the *policy* should call right before emitting
FINAL_ANSWER. It returns the GAIA-canonical surface form so that
"12 dollars" becomes "12", "  yes!" becomes "yes", and
"The Eiffel Tower." becomes "eiffel tower". Same primitives the scorer
uses, so the two cannot disagree.

Only inspects the proposed answer, never claims. The check fires only
at FINAL_ANSWER time.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..benchmarks.gaia.scorer import (
    normalize_string as _gaia_normalize_string,
    normalize_yes_no as _gaia_normalize_yes_no,
    parse_number as _gaia_parse_number,
    _split_list as _gaia_split_list,
)
from ..core.state import AgentState
from .base import ANSWER_CLAIM_ID, ClaimCheck


_YES = {"yes", "true", "y", "correct", "affirmative"}
_NO = {"no", "false", "n", "incorrect", "negative"}


# Detect required shape from the question text.
#
# yes/no shape: a question that opens with an auxiliary verb (is/are/was/
# were/does/do/did/can/could/should/will/would/has/have/had) followed by
# a subject and at least one more token, ending in '?'. Broad on purpose —
# false positives are filtered by checking year/list/number patterns
# *first* in `_detect_shape`.
_YES_NO_RE = re.compile(
    r"\b(?:is|are|was|were|does|do|did|"
    r"can|could|should|will|would|has|have|had)\b"
    r"\s+\w+.{0,200}\?",
    re.IGNORECASE,
)
# Match LIST intent only when "list" appears in a phrase that actually
# requests one (not when it appears in the GAIA answer-format suffix
# "a comma-separated list as appropriate"). Require an imperative
# phrasing — "list the …", "list of …", "list all …" — or one of the
# other unambiguous list cues.
_LIST_RE = re.compile(
    r"\b(?:"
    r"list\s+(?:the|of|all)\b"
    r"|enumerate\b"
    r"|name\s+all\b"
    r"|what\s+are\s+the\b"
    r"|which\s+of\s+the\s+following\b"
    r")",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(
    r"\b(?:how\s+many|how\s+much|count\s+the|number\s+of|"
    r"what(?:'s|\s+is)\s+the\s+(?:total|sum|count|number|average|mean|median))\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(
    r"\b(?:in\s+what\s+year|what\s+year|which\s+year|year\s+(?:was|is|did|of))\b",
    re.IGNORECASE,
)


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    # Strip surrounding quotes/punctuation
    while s and s[0] in "\"'`(" and len(s) >= 2:
        s = s[1:]
    while s and s[-1] in "\"'`).!?":
        s = s[:-1]
    return s.strip()


def _parse_number(s: str) -> float | None:
    s = (s or "").strip().replace(",", "").replace("$", "").rstrip("%")
    try:
        return float(s)
    except ValueError:
        return None


def _looks_like_list(s: str) -> bool:
    if "," in s or ";" in s:
        # At least 2 non-empty parts.
        parts = [p.strip() for p in re.split(r"[;,]", s) if p.strip()]
        return len(parts) >= 2
    if "\n" in s:
        parts = [p.strip() for p in s.split("\n") if p.strip()]
        return len(parts) >= 2
    return False


def _detect_shape(question: str) -> str:
    """Return one of: yes_no, list, number, year, unknown.

    Order matters — the more *content-specific* patterns (year/list/
    number) run first because the auxiliary-verb pattern in
    `_YES_NO_RE` would otherwise win on questions that contain "did",
    "does", "are", etc. as part of a how-many or what-year phrasing
    ("In what year did Iceland …", "How many feet are …").
    """
    q = question or ""
    if _YEAR_RE.search(q):
        return "year"
    if _LIST_RE.search(q):
        return "list"
    if _NUMBER_RE.search(q):
        return "number"
    if _YES_NO_RE.search(q):
        return "yes_no"
    return "unknown"


def _format_number(val: float) -> str:
    """Render a parsed number in canonical surface form.

    Integers come out without a trailing '.0'; floats use the shortest
    repr that round-trips. This is what the GAIA scorer expects to see
    after `parse_number` has stripped units/commas/percent — emitting
    it from the policy avoids "right value, wrong shape" failures.
    """
    if val != val:  # NaN
        return "nan"
    if val == int(val) and abs(val) < 1e16:
        return str(int(val))
    # Avoid scientific notation for typical GAIA-scale numbers.
    if 1e-4 <= abs(val) < 1e16:
        s = f"{val:.10g}"
        return s
    return repr(val)


def finalize_answer(text: str, question: str = "") -> str:
    """Return the GAIA-canonical surface form of `text`.

    The policy should call this right before emitting FINAL_ANSWER:

        state.trace.final_answer = finalize_answer(raw, state.question)

    Rules (matching ``benchmarks.gaia.scorer`` exactly):

      * empty / None             → ""
      * parses as a number       → plain number (no '$', no commas, no '%')
      * yes/no token             → "yes" or "no"
      * list (≥2 separated parts)→ comma-joined normalized parts, original order
      * everything else          → ``normalize_string`` output

    The `question` is currently advisory (helps choose between numeric
    and string when both are plausible); future extensions can use it
    for unit-stripping based on the question's wording.
    """
    if text is None:
        return ""
    raw = str(text).strip()
    if not raw:
        return ""

    # 1. Yes/No takes priority when the *question* invites it; otherwise
    #    fall through to numeric/string so an answer like "no" to a
    #    "how many" question doesn't get coerced.
    shape_hint = _detect_shape(question)
    if shape_hint == "yes_no":
        yn = _gaia_normalize_yes_no(raw)
        if yn is not None:
            return yn

    # 2. Numeric.
    num = _gaia_parse_number(raw)
    if num is not None and shape_hint != "list":
        return _format_number(num)

    # 3. List.
    parts = _gaia_split_list(raw)
    if parts is not None:
        return ", ".join(_gaia_normalize_string(p) for p in parts)

    # 4. Unprompted yes/no still gets canonicalized (no shape hint, but
    #    "yes." vs "yes" is a free win for GAIA).
    if shape_hint == "unknown":
        yn = _gaia_normalize_yes_no(raw)
        if yn is not None:
            return yn

    # 5. Plain string fallback.
    return _gaia_normalize_string(raw)


@dataclass
class FormatVerifier:
    """Minimal format check: flag empty answers, nothing else.

    Previously this verifier used regex pattern-matching against the
    question text to infer an expected answer shape (yes/no, number,
    list, year) and rejected answers that didn't match. That heuristic
    produced false positives that destroyed correct answers — e.g.
    classifying "What was the volume?" as yes/no, or matching the GAIA
    answer-format suffix "comma-separated list as appropriate" against
    every question. Replaced with a programmatic, GAIA-metadata-driven
    check in a later phase. For now, only empty answers fail.
    """

    name: str = "format"

    def check(
        self,
        state: AgentState,
        proposed_answer: str | None = None,
    ) -> list[ClaimCheck]:
        if proposed_answer is None:
            return []
        ans = (proposed_answer or "").strip()
        if not ans:
            return [ClaimCheck(
                claim_id=ANSWER_CLAIM_ID,
                verifier_name=self.name,
                verdict="fail",
                detail="empty answer",
                meta={
                    "error_class": "format_mismatch",
                    "nudge": (
                        "Your previous `final_answer` call had an empty "
                        "`answer` field. Call `final_answer` again with "
                        "the literal answer string in the `answer` field "
                        "— a number, a single name, or a comma-separated "
                        "list. No preamble."
                    ),
                },
            )]
        return [ClaimCheck(
            claim_id=ANSWER_CLAIM_ID,
            verifier_name=self.name,
            verdict="ok",
            detail="non-empty answer (no shape enforcement)",
            meta={"answer": ans[:120]},
        )]
