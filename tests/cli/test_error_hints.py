"""Tailored hints rendered for ERROR events in the streaming display.

Regression: a slow local model raising a ReadTimeout used to surface a
bare stack-trace line + "(no answer)". The display now appends an
actionable hint pointing at /model and /budget.
"""
from __future__ import annotations

import io

from rich.console import Console

from banna_agent.cli.display import StreamingEventLog
from banna_agent.core.events import AgentEvent, EventKind


def _render_error(**payload) -> str:
    buf = io.StringIO()
    log = StreamingEventLog(console=Console(file=buf, width=100))
    log.emit(AgentEvent(run_id="r", step=0, kind=EventKind.ERROR, payload=payload))
    # Collapse Rich's soft-wrap newlines so substring assertions don't
    # break on where the terminal happened to fold a line.
    return " ".join(buf.getvalue().split())


def test_read_timeout_emits_model_and_budget_hint() -> None:
    out = _render_error(
        error="ReadTimeout: HTTPConnectionPool(host='localhost', port=11434): "
              "Read timed out. (read timeout=120.0)",
        where="policy.propose",
    )
    assert "hint:" in out
    assert "/model" in out
    assert "/budget wall=300" in out


def test_non_timeout_error_has_no_timeout_hint() -> None:
    out = _render_error(error="ValueError: something unrelated", where="x")
    assert "read timeout" not in out.lower()
    assert "/budget wall=300" not in out


def test_missing_api_key_still_gets_its_own_hint_not_timeout() -> None:
    out = _render_error(
        error="api_key not set for openai",
        where="policy.propose",
        provider_error=True,
        retryable=False,
    )
    assert "set the API key" in out
    # The provider-error branch must win; no timeout hint bleed-through.
    assert "/budget wall=300" not in out
