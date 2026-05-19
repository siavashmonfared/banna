"""Unit tests for GAIA exact-match scorer."""
from __future__ import annotations

import pytest

from banna_agent.benchmarks.gaia.scorer import (
    normalize_string,
    parse_number,
    score_gaia,
)


# ---------------------------------------------------------------------------
# normalize_string
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "s, expected",
    [
        ("The Answer", "answer"),
        ("  an Apple  ", "apple"),
        ("'quoted'", "quoted"),
        ('"also quoted"', "also quoted"),
        ("A: B-C", "b c"),
        ("Hello, World!", "hello world"),
        ("The quick brown fox.", "quick brown fox"),
        ("A.N.S.W.E.R.", "a.n.s.w.e.r"),  # periods inside tokens survive until final
    ],
)
def test_normalize_string(s: str, expected: str) -> None:
    assert normalize_string(s) == expected


# ---------------------------------------------------------------------------
# parse_number
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "s, expected",
    [
        ("42", 42.0),
        ("42.5", 42.5),
        ("-17", -17.0),
        ("1,234", 1234.0),
        ("1,234.5", 1234.5),
        ("$1,234.50", 1234.5),
        ("25%", 0.25),
        ("  12  ", 12.0),
        ("1e3", 1000.0),
    ],
)
def test_parse_number(s: str, expected: float) -> None:
    got = parse_number(s)
    assert got == pytest.approx(expected)


@pytest.mark.parametrize("s", ["", "nope", "abc", "forty-two"])
def test_parse_number_rejects_non_numeric(s: str) -> None:
    assert parse_number(s) is None


# ---------------------------------------------------------------------------
# Scoring: numeric
# ---------------------------------------------------------------------------


def test_score_numeric_exact() -> None:
    r = score_gaia("42", "42")
    assert r.is_correct
    assert r.match_kind == "numeric"


def test_score_numeric_within_tolerance() -> None:
    r = score_gaia("1000.5", "1000")
    assert r.is_correct  # 0.05% ≤ 0.1% relative tolerance


def test_score_numeric_outside_tolerance() -> None:
    r = score_gaia("1002", "1000")
    assert not r.is_correct


def test_score_numeric_with_formatting() -> None:
    r = score_gaia("$1,234.50", "1234.5")
    assert r.is_correct


def test_score_percent_matches_decimal() -> None:
    r = score_gaia("25%", "0.25")
    assert r.is_correct


# ---------------------------------------------------------------------------
# Scoring: yes/no
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pred, gold, expected",
    [
        ("Yes", "yes", True),
        ("true", "yes", True),
        ("No", "no", True),
        ("NO.", "no", True),
        ("yes", "no", False),
        ("maybe", "yes", False),
    ],
)
def test_score_yes_no(pred: str, gold: str, expected: bool) -> None:
    r = score_gaia(pred, gold)
    assert r.match_kind == "yes_no"
    assert r.is_correct is expected


# ---------------------------------------------------------------------------
# Scoring: list
# ---------------------------------------------------------------------------


def test_score_list_exact() -> None:
    r = score_gaia("apple, banana, cherry", "apple, banana, cherry")
    assert r.is_correct
    assert r.match_kind == "list"


def test_score_list_order_insensitive() -> None:
    r = score_gaia("cherry, apple, banana", "apple, banana, cherry")
    assert r.is_correct


def test_score_list_handles_articles() -> None:
    r = score_gaia("the cat, the dog", "cat, dog")
    assert r.is_correct


def test_score_list_missing_element_fails() -> None:
    r = score_gaia("apple, banana", "apple, banana, cherry")
    assert not r.is_correct


def test_score_list_extra_element_fails() -> None:
    r = score_gaia("apple, banana, cherry, date", "apple, banana, cherry")
    assert not r.is_correct


def test_score_semicolon_list() -> None:
    r = score_gaia("a; b; c", "c; a; b")
    assert r.is_correct


# ---------------------------------------------------------------------------
# Scoring: string
# ---------------------------------------------------------------------------


def test_score_string_exact() -> None:
    r = score_gaia("Paris", "Paris")
    assert r.is_correct
    assert r.match_kind == "string"


def test_score_string_punctuation_ignored() -> None:
    r = score_gaia("Paris.", "Paris")
    assert r.is_correct


def test_score_string_article_ignored() -> None:
    r = score_gaia("The Eiffel Tower", "Eiffel Tower")
    assert r.is_correct


def test_score_string_case_insensitive() -> None:
    r = score_gaia("eiffel tower", "Eiffel Tower")
    assert r.is_correct


def test_score_string_wrong() -> None:
    r = score_gaia("London", "Paris")
    assert not r.is_correct


# ---------------------------------------------------------------------------
# Empty / missing
# ---------------------------------------------------------------------------


def test_score_empty_prediction() -> None:
    r = score_gaia("", "Paris")
    assert not r.is_correct
    assert r.match_kind == "empty"


def test_score_whitespace_only_prediction() -> None:
    r = score_gaia("   \n  ", "Paris")
    assert r.match_kind == "empty"


def test_score_populates_normalized_fields() -> None:
    r = score_gaia("The Eiffel Tower!", "Eiffel Tower")
    assert r.normalized_pred == "eiffel tower"
    assert r.normalized_gold == "eiffel tower"
