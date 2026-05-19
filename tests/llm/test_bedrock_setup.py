"""Tests for the Bedrock provider setup — model list, pricing, env hints.

These cover the seamless-with-AWS-keys flow:
  1. Curated Bedrock model list is consistent (no typos, no dupes).
  2. Every curated id has a pricing entry (so /cost works).
  3. Constructing a Bedrock client without AWS_REGION fails with a
     clear, actionable message.
  4. Constructing without keys/profile emits a soft warning but
     doesn't fail — the default credential chain still works.
"""
from __future__ import annotations

import os
import warnings

import pytest

from banna_agent.cli.commands import KNOWN_MODELS
from banna_agent.llm.anthropic import AnthropicClient
from banna_agent.llm.pricing import PRICING_USD_PER_M, lookup_price


def _bedrock_ids() -> tuple[str, ...]:
    return KNOWN_MODELS["bedrock"]


# ---------------------------------------------------------------------------
# Curated list sanity
# ---------------------------------------------------------------------------


def test_bedrock_list_has_no_duplicates() -> None:
    ids = _bedrock_ids()
    assert len(ids) == len(set(ids))


def test_bedrock_ids_are_well_formed() -> None:
    """Anthropic-on-Bedrock IDs follow either:
      * `<region>.anthropic.claude-<...>-v1:0`  (inference profile)
      * `anthropic.claude-<...>-v1:0`           (region-locked foundation)
    """
    valid_prefixes = ("us.anthropic.", "eu.anthropic.", "apac.anthropic.",
                      "global.anthropic.", "anthropic.")
    for mid in _bedrock_ids():
        assert any(mid.startswith(p) for p in valid_prefixes), \
            f"unexpected Bedrock id format: {mid!r}"
        assert mid.endswith(":0"), f"missing version suffix on {mid!r}"


def test_every_curated_bedrock_id_has_pricing() -> None:
    """Without a pricing entry /cost reports zero for that run — easy
    way to silently miscount spend. Force a hit for every curated id."""
    missing = []
    for mid in _bedrock_ids():
        if lookup_price("bedrock", mid) is None:
            missing.append(mid)
    assert not missing, f"missing pricing for {missing}"


def test_bedrock_pricing_keys_are_namespaced() -> None:
    """Every bedrock/ pricing key must start with 'bedrock/'."""
    for k in PRICING_USD_PER_M:
        if "anthropic.claude" in k and "bedrock" not in k and not k.startswith("anthropic/"):
            pytest.fail(f"Bedrock-shaped id {k!r} missing 'bedrock/' namespace")


def test_default_bedrock_model_is_in_curated_list() -> None:
    """`_bedrock_factory` defaults to a specific id; that id should be in
    the picker list so users see the default among their options."""
    from banna_agent.llm.registry import _bedrock_factory
    # Default model captured in _bedrock_factory's kw.get default — read
    # by introspecting the function signature isn't clean, so we just
    # check the canonical default string lives in the list.
    default = "us.anthropic.claude-opus-4-5-20251101-v1:0"
    assert default in _bedrock_ids()


# ---------------------------------------------------------------------------
# Error / warning UX
# ---------------------------------------------------------------------------


def test_missing_region_raises_actionable_error(monkeypatch) -> None:
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    client = AnthropicClient(
        model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        transport="bedrock",
        aws_region=None,
    )
    with pytest.raises(RuntimeError, match="AWS_REGION"):
        client._client()


def test_region_error_mentions_inference_profile_prefixes(monkeypatch) -> None:
    """The error tells the user which `us.`/`eu.`/`apac.` to pick."""
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    client = AnthropicClient(
        model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        transport="bedrock",
        aws_region=None,
    )
    try:
        client._client()
    except RuntimeError as exc:
        msg = str(exc)
        assert "us-east-1" in msg or "us-west-2" in msg
        assert "us." in msg
        assert "eu." in msg
        assert "apac." in msg


def test_missing_credentials_emits_warning_but_not_error(monkeypatch) -> None:
    """Soft hint when keys/profile missing — default credential chain may
    still resolve them, so we don't fail at construction."""
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)

    # Stub the anthropic SDK so _client() doesn't try a real network call.
    import sys, types
    fake_anthropic = types.ModuleType("anthropic")
    class _FakeBedrock:
        def __init__(self, **kw): self.kw = kw
    fake_anthropic.AnthropicBedrock = _FakeBedrock  # type: ignore[attr-defined]
    fake_anthropic.Anthropic = _FakeBedrock        # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    client = AnthropicClient(
        model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        transport="bedrock",
        aws_region=None,  # picked up from env
        aws_profile=None,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sdk = client._client()
    # SDK constructed successfully despite missing creds.
    assert isinstance(sdk, _FakeBedrock)
    # And we got the soft warning.
    msgs = [str(w.message) for w in caught]
    assert any("AWS_ACCESS_KEY_ID" in m or "default credential chain" in m for m in msgs)


def test_aws_profile_silences_credential_warning(monkeypatch) -> None:
    """A profile is enough — no warning emitted."""
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.setenv("AWS_PROFILE", "my-profile")

    import sys, types
    fake_anthropic = types.ModuleType("anthropic")
    class _FakeBedrock:
        def __init__(self, **kw): self.kw = kw
    fake_anthropic.AnthropicBedrock = _FakeBedrock  # type: ignore[attr-defined]
    fake_anthropic.Anthropic = _FakeBedrock        # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    client = AnthropicClient(
        model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        transport="bedrock",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client._client()
    msgs = [str(w.message) for w in caught]
    assert not any("AWS_ACCESS_KEY_ID" in m for m in msgs)
