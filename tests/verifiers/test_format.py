"""FormatVerifier tests.

After Phase 2: the verifier is minimal — it flags only empty answers
and passes everything else. Regex-based shape detection and canonical-
form rejection were removed because they produced false positives that
destroyed correct answers (the GAIA scorer applies its own
normalization on its own side for comparison). Programmatic, metadata-
driven shape checks are reintroduced in Phase 8.
"""
from __future__ import annotations

from banna_agent.core.state import AgentState
from banna_agent.verifiers.format import FormatVerifier


def _state(q: str) -> AgentState:
    return AgentState(question=q)


def test_non_empty_answer_passes() -> None:
    s = _state("Is Iceland in Europe?")
    checks = FormatVerifier().check(s, proposed_answer="yes")
    assert checks[0].verdict == "ok"


def test_non_canonical_form_no_longer_rejected() -> None:
    # Pre-Phase-2 this was rejected for not being in GAIA canonical form
    # ("yes" instead of "Yes."). We now submit the model's literal
    # string and let the scorer normalize for comparison.
    s = _state("Is Iceland in Europe?")
    checks = FormatVerifier().check(s, proposed_answer="Yes.")
    assert checks[0].verdict == "ok"


def test_shape_mismatch_no_longer_rejected() -> None:
    # Pre-Phase-2 this was rejected as "expected yes/no, got 'It is in
    # Europe.'". The verifier no longer infers shape from question text.
    s = _state("Is Iceland in Europe?")
    checks = FormatVerifier().check(s, proposed_answer="It is in Europe.")
    assert checks[0].verdict == "ok"


def test_number_with_comma_grouping_no_longer_rejected() -> None:
    s = _state("How many people live in Iceland?")
    checks = FormatVerifier().check(s, proposed_answer="372,000")
    assert checks[0].verdict == "ok"


def test_no_proposed_answer_returns_empty() -> None:
    s = _state("Is Iceland in Europe?")
    assert FormatVerifier().check(s) == []


def test_empty_proposed_answer_is_fail() -> None:
    s = _state("Is Iceland in Europe?")
    checks = FormatVerifier().check(s, proposed_answer="")
    assert checks[0].verdict == "fail"
    assert checks[0].meta.get("error_class") == "format_mismatch"


def test_whitespace_only_is_fail() -> None:
    s = _state("Is Iceland in Europe?")
    checks = FormatVerifier().check(s, proposed_answer="   ")
    assert checks[0].verdict == "fail"
