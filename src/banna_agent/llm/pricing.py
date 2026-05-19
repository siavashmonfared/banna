"""Per-model pricing in USD per 1M tokens.

This is a *best-effort estimate*, not bill-accurate. Provider list prices
move; cached-input discounts (Anthropic / OpenAI), batch-API rebates,
and per-region differences (Bedrock, Vertex) all change the real cost.
For the CLI's `/cost` command this is enough to gauge whether you spent
cents or dollars on a session.

Override at runtime by setting the `MYAGENT_PRICES` env var to a JSON
dict, e.g.:

    MYAGENT_PRICES='{"openai/gpt-5-nano": [0.05, 0.40]}'

Keys merge into PRICING_USD_PER_M; existing entries are overwritten.

Verify against the provider's pricing page when you care about exact
numbers:
  - https://platform.openai.com/docs/pricing
  - https://www.anthropic.com/pricing
  - https://ai.google.dev/pricing
  - https://ollama.com   (local, free)
"""
from __future__ import annotations

import json
import os
from typing import Optional


# Format: {"<provider>/<model>": (input_per_M_usd, output_per_M_usd)}
PRICING_USD_PER_M: dict[str, tuple[float, float]] = {
    # ----------------------------------------------------------------
    # OpenAI — gpt-5 family + o-series + gpt-4.x
    # Approximate as of 2026; check platform.openai.com/docs/pricing.
    # ----------------------------------------------------------------
    "openai/gpt-5":               (1.25,  10.00),
    "openai/gpt-5-mini":          (0.25,   2.00),
    "openai/gpt-5-nano":          (0.05,   0.40),
    "openai/gpt-4.1":             (2.00,   8.00),
    "openai/gpt-4.1-mini":        (0.40,   1.60),
    "openai/gpt-4.1-nano":        (0.10,   0.40),
    "openai/gpt-4o":              (2.50,  10.00),
    "openai/gpt-4o-mini":         (0.15,   0.60),
    "openai/gpt-4-turbo":        (10.00,  30.00),
    "openai/gpt-3.5-turbo":       (0.50,   1.50),
    "openai/o4-mini":             (1.10,   4.40),
    "openai/o3":                  (2.00,   8.00),
    "openai/o3-mini":             (1.10,   4.40),
    "openai/o1":                 (15.00,  60.00),
    "openai/o1-mini":             (1.10,   4.40),

    # ----------------------------------------------------------------
    # Anthropic — Claude 4.x family
    # https://www.anthropic.com/pricing
    # ----------------------------------------------------------------
    "anthropic/claude-opus-4-7":               (15.00,  75.00),
    "anthropic/claude-opus-4-5-20251101":      (15.00,  75.00),
    "anthropic/claude-opus-4-1":               (15.00,  75.00),
    "anthropic/claude-sonnet-4-6":              (3.00,  15.00),
    "anthropic/claude-sonnet-4-5-20250929":     (3.00,  15.00),
    "anthropic/claude-haiku-4-5-20251001":      (0.80,   4.00),
    "anthropic/claude-haiku-4-5":               (0.80,   4.00),
    # Older Claude 3.x — kept for users running older codepaths.
    "anthropic/claude-3-5-sonnet":              (3.00,  15.00),
    "anthropic/claude-3-5-haiku":               (0.80,   4.00),
    "anthropic/claude-3-opus":                 (15.00,  75.00),

    # ----------------------------------------------------------------
    # Google Gemini
    # https://ai.google.dev/pricing
    # ----------------------------------------------------------------
    "gemini/gemini-2.5-pro":      (1.25,  10.00),  # ≤200K context tier
    "gemini/gemini-2.5-flash":    (0.075,  0.30),
    "gemini/gemini-2.5-flash-lite":(0.05,  0.20),
    "gemini/gemini-2.0-flash":    (0.075,  0.30),
    "gemini/gemini-2.0-flash-lite":(0.05,  0.20),
    "gemini/gemini-1.5-pro":      (1.25,   5.00),
    "gemini/gemini-1.5-flash":    (0.075,  0.30),

    # ----------------------------------------------------------------
    # Ollama (local) — always free; explicit so it shows in /cost rates.
    # ----------------------------------------------------------------
    "ollama/*":                   (0.0,    0.0),

    # ----------------------------------------------------------------
    # Bedrock — same models as Anthropic, but billed via AWS. Region
    # and on-demand vs provisioned-throughput change the rate. List
    # the canonical Anthropic prices here as a reasonable estimate;
    # users on provisioned throughput should override.
    # ----------------------------------------------------------------
    # 4.x — US inference profiles.
    "bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0":     (15.00, 75.00),
    "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0":    (3.00, 15.00),
    "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0":     (0.80,  4.00),
    # 4.x — Global / EU / APAC inference profiles (same per-token prices;
    # AWS bills the same regardless of routing tier, only data-transfer
    # adders differ which we don't try to track here).
    "bedrock/global.anthropic.claude-opus-4-5-20251101-v1:0": (15.00, 75.00),
    "bedrock/global.anthropic.claude-sonnet-4-5-20250929-v1:0":(3.00, 15.00),
    "bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0": (0.80,  4.00),
    "bedrock/eu.anthropic.claude-sonnet-4-5-20250929-v1:0":    (3.00, 15.00),
    "bedrock/eu.anthropic.claude-haiku-4-5-20251001-v1:0":     (0.80,  4.00),
    "bedrock/apac.anthropic.claude-sonnet-4-5-20250929-v1:0":  (3.00, 15.00),
    "bedrock/apac.anthropic.claude-haiku-4-5-20251001-v1:0":   (0.80,  4.00),
    # 4.0 / 4.1 — older 4.x snapshots still on Bedrock.
    "bedrock/us.anthropic.claude-opus-4-1-20250805-v1:0":     (15.00, 75.00),
    "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0":      (3.00, 15.00),
    # 3.x — older / cheaper, broad availability.
    "bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0":    (3.00, 15.00),
    "bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0":    (3.00, 15.00),
    "bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0":     (1.00,  5.00),
    "bedrock/us.anthropic.claude-3-haiku-20240307-v1:0":       (0.25,  1.25),
    # Region-locked foundation IDs (no inference-profile prefix).
    "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0":       (3.00, 15.00),
    "bedrock/anthropic.claude-3-5-haiku-20241022-v1:0":        (1.00,  5.00),
    "bedrock/anthropic.claude-3-haiku-20240307-v1:0":          (0.25,  1.25),
}


def _load_overrides() -> None:
    """Merge MYAGENT_PRICES env var (JSON) into PRICING_USD_PER_M."""
    raw = os.environ.get("MYAGENT_PRICES")
    if not raw:
        return
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return
    if not isinstance(data, dict):
        return
    for k, v in data.items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            try:
                PRICING_USD_PER_M[str(k)] = (float(v[0]), float(v[1]))
            except (TypeError, ValueError):
                continue


_load_overrides()


def lookup_price(provider: str, model: str) -> Optional[tuple[float, float]]:
    """Return ``(input_USD_per_M, output_USD_per_M)`` for this combo, or None.

    Lookup order:
      1. exact match on ``"<provider>/<model>"``
      2. provider-wide wildcard ``"<provider>/*"`` (used for ollama)
      3. longest-prefix match within the provider — handles versioned
         model ids like ``gpt-5-mini-2025-08-07`` falling back to
         ``gpt-5-mini`` rates.

    Returns None when nothing matches; the caller should treat that as
    "unknown pricing — show $0 + warn".
    """
    key = f"{provider}/{model}"
    if key in PRICING_USD_PER_M:
        return PRICING_USD_PER_M[key]
    wildcard = f"{provider}/*"
    if wildcard in PRICING_USD_PER_M:
        return PRICING_USD_PER_M[wildcard]
    # Longest-prefix fallback — "gpt-5-mini-2025-08-07" → "gpt-5-mini".
    best: Optional[str] = None
    best_len = 0
    prefix = f"{provider}/"
    for k in PRICING_USD_PER_M:
        if not k.startswith(prefix) or k == wildcard:
            continue
        suffix = k[len(prefix):]
        if model.startswith(suffix) and len(suffix) > best_len:
            best = k
            best_len = len(suffix)
    return PRICING_USD_PER_M[best] if best else None


def estimate_cost(
    provider: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
) -> tuple[float, bool]:
    """Estimate cost in USD. Returns ``(cost_usd, is_known)``.

    `is_known=False` when no pricing entry was found for the
    provider/model combo; cost is then 0.0 and the caller should warn
    rather than report.
    """
    rates = lookup_price(provider, model)
    if rates is None:
        return (0.0, False)
    inp, out = rates
    cost = (tokens_in / 1_000_000.0) * inp + (tokens_out / 1_000_000.0) * out
    return (cost, True)


def all_prices() -> dict[str, tuple[float, float]]:
    """Read-only view of the merged pricing table (overrides applied)."""
    return dict(PRICING_USD_PER_M)
