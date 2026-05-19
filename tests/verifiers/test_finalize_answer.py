"""Phase 4 gate: golden table for finalize_answer + scorer agreement.

Two contracts the verifier and scorer must jointly uphold:

  (A) Surface contract — `finalize_answer(raw, question)` returns the
      canonical surface form for each GAIA answer kind (numeric, yes/no,
      list, string). The table below pins ~40 cases.

  (B) Scorer round-trip — for every (raw, gold) pair, emitting the
      finalized form must score the same as emitting the raw form. If
      finalization ever flipped a correct answer to incorrect, that
      would be a bug in `finalize_answer`; the scorer is the
      ground-truth normalizer.

This is the load-bearing test for Phase 4: it locks the relationship
between the verifier (what we tell the policy to emit) and the scorer
(what GAIA actually accepts).
"""
from __future__ import annotations

import pytest

from banna_agent.benchmarks.gaia.scorer import score_gaia
from banna_agent.verifiers.format import finalize_answer


# ---------------------------------------------------------------------------
# (A) Surface contract — what finalize_answer should return.
# Format: (raw_answer, question_or_empty, expected_canonical_surface).
# ---------------------------------------------------------------------------


GOLDEN: list[tuple[str, str, str]] = [
    # --- numeric: commas, currency, percent, units, sign --------------
    ("12", "How many?", "12"),
    ("12.0", "How many?", "12"),
    ("12.5", "How many?", "12.5"),
    ("$12", "How much?", "12"),
    ("12,345", "How many?", "12345"),
    ("12,345.67", "How much?", "12345.67"),
    ("-7", "What's the temperature?", "-7"),
    ("+3", "What's the change?", "3"),
    ("50%", "What is the percentage?", "0.5"),
    ("100%", "Share?", "1"),
    ("1e3", "Order of magnitude?", "1000"),

    # --- yes/no -------------------------------------------------------
    ("yes", "Is X true?", "yes"),
    ("Yes", "Is X true?", "yes"),
    ("YES", "Is X true?", "yes"),
    ("yes.", "Is X true?", "yes"),
    (" Yes! ", "Is X true?", "yes"),
    ("true", "Is X true?", "yes"),
    ("no", "Is X true?", "no"),
    ("No.", "Is X true?", "no"),
    ("false", "Is X true?", "no"),

    # --- list (comma) -------------------------------------------------
    ("Nigeria, Ethiopia, Egypt", "List the three…", "nigeria, ethiopia, egypt"),
    ("apple, banana, cherry", "List the fruits", "apple, banana, cherry"),
    ("Alice; Bob; Carol", "List the names", "alice, bob, carol"),
    ("  Alice , Bob ,Carol  ", "List…", "alice, bob, carol"),

    # --- plain string -------------------------------------------------
    ("The Eiffel Tower.", "What is in Paris?", "eiffel tower"),
    ("Eiffel Tower", "What is in Paris?", "eiffel tower"),
    ("'Hello World'", "What did it print?", "hello world"),
    ('"quoted answer"', "What was the message?", "quoted answer"),
    ("Mount   Everest", "Tallest peak?", "mount everest"),
    ("A picnic basket", "What did they bring?", "picnic basket"),  # 'A' is an article

    # --- empty / whitespace -------------------------------------------
    ("", "", ""),
    ("   ", "anything", ""),

    # --- already-canonical (idempotency) ------------------------------
    ("42", "How many?", "42"),
    ("yes", "", "yes"),
    ("apple, banana", "list…", "apple, banana"),
    ("eiffel tower", "What is in Paris?", "eiffel tower"),
]


@pytest.mark.parametrize("raw, question, expected", GOLDEN)
def test_finalize_answer_golden(raw: str, question: str, expected: str) -> None:
    assert finalize_answer(raw, question) == expected


def test_finalize_answer_is_idempotent() -> None:
    """finalize(finalize(x)) == finalize(x) for every case in the table."""
    for raw, question, expected in GOLDEN:
        once = finalize_answer(raw, question)
        twice = finalize_answer(once, question)
        assert once == twice, f"non-idempotent on {raw!r}: {once!r} -> {twice!r}"


# ---------------------------------------------------------------------------
# (B) Scorer round-trip — finalization must never flip a correct answer
#     to incorrect (or vice-versa) against the official scorer.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw, question, expected", GOLDEN)
def test_finalized_form_scores_same_against_canonical_gold(
    raw: str, question: str, expected: str,
) -> None:
    """If `expected` is the canonical gold, both the raw and finalized
    answer should grade identically."""
    if expected == "":
        # Empty gold isn't a real GAIA case; skip.
        return
    final = finalize_answer(raw, question)
    raw_score = score_gaia(raw, expected)
    fin_score = score_gaia(final, expected)
    assert raw_score.is_correct == fin_score.is_correct, (
        f"finalization flipped scorer outcome on {raw!r} vs gold {expected!r}: "
        f"raw={raw_score.is_correct}, finalized={fin_score.is_correct}"
    )


# ---------------------------------------------------------------------------
# Specific high-value cases.
# ---------------------------------------------------------------------------


def test_dollar_units_get_stripped() -> None:
    assert finalize_answer("$12", "How much?") == "12"


def test_grouping_commas_get_stripped() -> None:
    assert finalize_answer("372,000", "How many?") == "372000"


def test_trailing_period_dropped_from_yes() -> None:
    assert finalize_answer("Yes.", "Is X true?") == "yes"


def test_list_preserves_original_order() -> None:
    # GAIA scoring is set-equal on lists, but for the *answer* string
    # we keep the original order so the user-visible output looks right.
    out = finalize_answer("Charlie, Alpha, Bravo", "List…")
    assert out == "charlie, alpha, bravo"


def test_yes_no_only_kicks_in_when_question_invites_it() -> None:
    # The raw "no" to a *how-many* question must NOT be coerced — it
    # falls through to number/string handling.
    out = finalize_answer("no", "How many apples?")
    # "no" isn't a number; falls through to string normalization.
    assert out == "no"


def test_number_beats_yes_no_when_question_is_numeric() -> None:
    out = finalize_answer("12", "How many apples?")
    assert out == "12"
