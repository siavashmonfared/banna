"""Per-provider token counting and calibration."""
from __future__ import annotations

import pytest

from banna_agent.llm.base import ContentBlock, Message, ToolSpec
from banna_agent.llm.tokenizers import (
    HeuristicCounter,
    TiktokenCounter,
    reset_counters,
    resolve_counter,
)

tiktoken = pytest.importorskip("tiktoken")


@pytest.fixture(autouse=True)
def _clean_cache():
    reset_counters()
    yield
    reset_counters()


def _msg(text: str) -> Message:
    return Message(role="user", content=[ContentBlock(kind="text", text=text)])


# ---------------------------------------------------------------------------
# Resolution — the dependency tree, exercised through the public entry point
# ---------------------------------------------------------------------------


def test_openai_gets_an_exact_local_tokenizer() -> None:
    """tiktoken *is* the OpenAI API's tokenizer, so no calibration needed."""
    c = resolve_counter("openai", "gpt-4o")
    assert c.exact is True


def test_other_providers_get_a_proxy_that_stays_calibratable() -> None:
    """No public local tokenizer exists for these, so the counter must keep
    correcting itself rather than claim exactness."""
    for provider in ("anthropic", "bedrock", "gemini", "ollama"):
        assert resolve_counter(provider, "some-model").exact is False


def test_unknown_provider_still_resolves() -> None:
    """A provider registered later must not crash the counter lookup."""
    assert resolve_counter("brand-new-provider", "m").count_text("hello") > 0


def test_counters_are_cached_so_calibration_persists() -> None:
    assert resolve_counter("ollama", "qwen") is resolve_counter("ollama", "qwen")
    assert resolve_counter("ollama", "qwen") is not resolve_counter("ollama", "llama")


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


def test_tiktoken_beats_the_character_heuristic_on_real_text() -> None:
    """The whole point of the upgrade: real tokenization, not chars/4."""
    text = "The quick brown fox jumps over the lazy dog. " * 20
    exact = len(tiktoken.get_encoding("o200k_base").encode(text))
    assert TiktokenCounter()._raw_count(text) == exact


def test_count_request_includes_tools_and_system() -> None:
    c = resolve_counter("openai", "gpt-4o")
    tools = [ToolSpec(name="t", description="d" * 500,
                      input_schema={"type": "object"})]
    bare = c.count_request(messages=[_msg("hi")])
    full = c.count_request(messages=[_msg("hi")], tools=tools, system="s" * 500)
    assert full > bare + 200


def test_count_request_counts_tool_results_in_history() -> None:
    """Tool output is most of an agent conversation's bulk; missing it would
    make every estimate useless a few turns in."""
    c = resolve_counter("openai", "gpt-4o")
    plain = Message(role="user", content=[ContentBlock(kind="text", text="hi")])
    with_result = Message(role="user", content=[
        ContentBlock(kind="tool_result", id="1", result="x" * 4000)])
    assert c.count_request(messages=[with_result]) > c.count_request(messages=[plain]) + 200


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_calibration_moves_the_estimate_toward_reported_usage() -> None:
    c = HeuristicCounter()
    before = c.count_text("x" * 4000)
    for _ in range(6):
        c.observe(estimated=c.count_text("x" * 4000), actual=2000)
    after = c.count_text("x" * 4000)
    assert abs(after - 2000) < abs(before - 2000)


def test_calibration_converges_and_does_not_oscillate() -> None:
    c = HeuristicCounter()
    for _ in range(10):
        c.observe(estimated=c.count_text("x" * 4000), actual=1500)
    assert c.count_text("x" * 4000) == pytest.approx(1500, rel=0.1)


def test_exact_counters_never_calibrate() -> None:
    """Correcting an exact tokenizer against noisy usage would make it
    worse, not better."""
    c = TiktokenCounter(exact=True)
    c.observe(estimated=1000, actual=250)
    assert c.scale == 1.0


def test_calibration_ignores_implausible_ratios() -> None:
    """A truncated reply reports far fewer tokens than we sent. Learning
    from it would shrink estimates until truncation looked normal — the bug
    would train the detector to ignore itself."""
    c = HeuristicCounter()
    c.observe(estimated=10_000, actual=4096)     # the truncation signature
    assert c.scale == 1.0


def test_calibration_ignores_empty_observations() -> None:
    c = HeuristicCounter()
    c.observe(estimated=0, actual=0)
    c.observe(estimated=500, actual=0)
    assert c.scale == 1.0
    assert c.calibrated is False


def test_calibration_floor_matches_the_truncation_threshold() -> None:
    """These two numbers have to agree. If calibration accepted a ratio the
    detector calls truncation, the counter would learn to hide the bug."""
    from banna_agent.llm import context, tokenizers
    assert tokenizers._MIN_PLAUSIBLE_RATIO == context._TRUNCATION_RATIO
