"""Tests for llm.pricing — model→rate lookup + cost estimation."""
from __future__ import annotations

import json

import pytest

from banna_agent.llm.pricing import (
    PRICING_USD_PER_M,
    all_prices,
    estimate_cost,
    lookup_price,
)


# ---------------------------------------------------------------------------
# lookup_price
# ---------------------------------------------------------------------------


def test_exact_match_returns_listed_rates() -> None:
    inp, out = lookup_price("openai", "gpt-5-mini") or (None, None)
    assert inp == 0.25
    assert out == 2.00


def test_anthropic_dated_id_exact_match() -> None:
    inp, out = lookup_price("anthropic", "claude-opus-4-5-20251101") or (None, None)
    assert inp == 15.0
    assert out == 75.0


def test_ollama_wildcard_returns_zero() -> None:
    """Any ollama model returns (0, 0) via the wildcard entry."""
    assert lookup_price("ollama", "qwen3-coder:30b") == (0.0, 0.0)
    assert lookup_price("ollama", "anything-at-all") == (0.0, 0.0)


def test_prefix_fallback_finds_versioned_id() -> None:
    """A versioned model id like 'gpt-5-mini-2025-08-07' falls back to
    the base 'gpt-5-mini' rates."""
    rates = lookup_price("openai", "gpt-5-mini-2025-08-07")
    assert rates == (0.25, 2.00)


def test_prefix_fallback_picks_longest_match() -> None:
    """For 'gpt-5-nano-2025-09-01', the lookup must prefer 'gpt-5-nano'
    (longer suffix) over 'gpt-5' (shorter)."""
    rates = lookup_price("openai", "gpt-5-nano-2025-09-01")
    assert rates == PRICING_USD_PER_M["openai/gpt-5-nano"]


def test_unknown_provider_returns_none() -> None:
    assert lookup_price("not-a-provider", "anything") is None


def test_unknown_model_for_known_provider_returns_none() -> None:
    """A model that doesn't share a prefix with any listed entry."""
    assert lookup_price("openai", "claude-3") is None


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------


def test_estimate_cost_arithmetic() -> None:
    # 1M input + 1M output at $1/M each → $2 total.
    cost, known = estimate_cost("openai", "gpt-5-mini", 1_000_000, 1_000_000)
    assert known is True
    assert cost == pytest.approx(0.25 + 2.00)


def test_estimate_cost_handles_zero_tokens() -> None:
    cost, known = estimate_cost("openai", "gpt-5-nano", 0, 0)
    assert cost == 0.0
    assert known is True


def test_estimate_cost_unknown_returns_zero_not_known() -> None:
    cost, known = estimate_cost("not-a-provider", "x", 100, 100)
    assert cost == 0.0
    assert known is False


def test_estimate_cost_ollama_is_free() -> None:
    cost, known = estimate_cost("ollama", "qwen3-coder:30b", 5_000, 1_000)
    assert cost == 0.0
    assert known is True


# ---------------------------------------------------------------------------
# Override mechanism
# ---------------------------------------------------------------------------


def test_myagent_prices_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """MYAGENT_PRICES env JSON merges into the table on module load."""
    monkeypatch.setenv("MYAGENT_PRICES",
                       json.dumps({"openai/gpt-5-nano": [99.0, 199.0]}))
    # Re-import to re-run _load_overrides.
    import importlib

    from banna_agent.llm import pricing as pricing_mod
    importlib.reload(pricing_mod)
    rates = pricing_mod.lookup_price("openai", "gpt-5-nano")
    assert rates == (99.0, 199.0)
    # Reload again with no env to restore defaults for sibling tests.
    monkeypatch.delenv("MYAGENT_PRICES", raising=False)
    importlib.reload(pricing_mod)


def test_all_prices_returns_copy() -> None:
    a = all_prices()
    b = all_prices()
    assert a == b
    a["openai/fake"] = (1.0, 2.0)
    assert "openai/fake" not in b   # mutation didn't leak
