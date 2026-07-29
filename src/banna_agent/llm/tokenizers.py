"""Per-provider, per-model token counting.

The public surface is one function — `resolve_counter(provider, model)` —
so callers never branch on provider themselves. Behind it sits a small
dependency tree: each provider family maps to the best counter we can
actually run for it.

What "best" means differs by provider, and it's worth being honest about
which is which rather than calling them all exact:

  openai            tiktoken, locally, exactly the tokenizer the API uses.
  anthropic/bedrock no public local tokenizer since Claude 3 — counting is
                    a server-side API call. We use a local proxy encoding
                    and calibrate (below).
  gemini            same situation: countTokens is server-side.
  ollama            the tokenizer is whatever the local model was trained
                    with, and varies per model. No local access to it.

For everything in the second group, exactness would cost a network round
trip on every single turn — doubling latency to measure something we are
about to send anyway. So those counters *calibrate* instead: each real
reply tells us how many input tokens the provider actually counted, and
the counter folds that into a correction factor for its (provider, model)
pair. After a couple of turns the estimate tracks the provider's own
accounting to within a few percent, with no extra calls and no extra
dependencies.

Calibration only learns from healthy replies. A truncated reply reports
fewer tokens than we sent, and treating that as ground truth would teach
the counter to shrink its estimates until the truncation looked normal —
the bug would train the detector to ignore it.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from .base import Message, ToolSpec

# Per-message framing (role markers, separators) that providers add on top
# of the visible text. OpenAI documents ~3 tokens per message plus ~3 to
# prime the reply; the others are in the same range, and calibration
# absorbs the difference.
_PER_MESSAGE_OVERHEAD = 3
_REPLY_PRIMING_OVERHEAD = 3
# Tool declarations get wrapped in provider-specific JSON envelopes.
_PER_TOOL_OVERHEAD = 8

# Kept in lockstep with `context._TRUNCATION_RATIO` — see `observe`.
_MIN_PLAUSIBLE_RATIO = 0.75


class TokenCounter(Protocol):
    """Counts tokens for one (provider, model) pair."""

    name: str
    exact: bool

    def count_text(self, text: str) -> int: ...

    def count_request(
        self,
        *,
        messages: Sequence[Message] = (),
        tools: Sequence[ToolSpec] = (),
        system: str | None = None,
    ) -> int: ...

    def observe(self, *, estimated: int, actual: int) -> None:
        """Feed back a provider-reported count from a healthy reply."""
        ...


def _message_text(msg: Any) -> str:
    """Flatten one message to the text that will be tokenized."""
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or ():
        for attr in ("text", "name"):
            v = getattr(block, attr, None)
            if isinstance(v, str):
                parts.append(v)
        args = getattr(block, "arguments", None)
        if args:
            parts.append(repr(args))
        result = getattr(block, "result", None)
        if result is not None:
            parts.append(result if isinstance(result, str) else repr(result))
    return "\n".join(parts)


def _tool_text(spec: ToolSpec) -> str:
    return f"{spec.name}\n{spec.description}\n{spec.input_schema!r}"


@dataclass
class _BaseCounter:
    """Shared request-assembly and calibration; subclasses supply
    `_raw_count` for a single string."""

    name: str = "base"
    exact: bool = False
    # Multiplier applied to raw counts, learned from provider-reported
    # usage. 1.0 until we've seen a healthy reply.
    scale: float = 1.0
    _samples: int = 0
    _lock: Any = field(default_factory=threading.Lock, repr=False)

    def _raw_count(self, text: str) -> int:
        raise NotImplementedError

    def count_text(self, text: str) -> int:
        return int(self._raw_count(text) * self.scale)

    def count_request(
        self,
        *,
        messages: Sequence[Message] = (),
        tools: Sequence[ToolSpec] = (),
        system: str | None = None,
    ) -> int:
        raw = 0
        if system:
            raw += self._raw_count(system) + _PER_MESSAGE_OVERHEAD
        for spec in tools:
            raw += self._raw_count(_tool_text(spec)) + _PER_TOOL_OVERHEAD
        for msg in messages:
            raw += self._raw_count(_message_text(msg)) + _PER_MESSAGE_OVERHEAD
        raw += _REPLY_PRIMING_OVERHEAD
        return int(raw * self.scale)

    def observe(self, *, estimated: int, actual: int) -> None:
        """Nudge `scale` toward the provider's own accounting.

        An exponential moving average rather than a straight ratio: any one
        turn can be skewed by caching or by provider-side additions we
        can't see, and we want the correction to settle rather than chase
        each turn's noise.
        """
        if self.exact or estimated <= 0 or actual <= 0:
            return
        # Undo the scale already applied so the ratio is against raw counts.
        raw = estimated / self.scale if self.scale else estimated
        if raw <= 0:
            return
        observed = actual / raw
        # Ignore ratios that aren't tokenizer drift. The lower bound is
        # deliberately the same as the truncation detector's threshold in
        # `context.py`: a shortfall big enough to be *reported* as
        # truncation must never be *learned* as drift, or the counter would
        # slowly recalibrate the bug into looking normal. Belt and braces —
        # the guard already withholds truncated replies from calibration.
        if not (_MIN_PLAUSIBLE_RATIO <= observed <= 2.5):
            return
        with self._lock:
            weight = 0.5 if self._samples == 0 else 0.25
            self.scale = (1 - weight) * self.scale + weight * observed
            self._samples += 1

    @property
    def calibrated(self) -> bool:
        return self.exact or self._samples > 0


@dataclass
class HeuristicCounter(_BaseCounter):
    """Last-resort counter: characters divided by a fixed ratio.

    Only used when no local tokenizer is importable. Starts crude and
    becomes accurate through calibration.
    """

    name: str = "heuristic"
    exact: bool = False
    chars_per_token: float = 4.0

    def _raw_count(self, text: str) -> int:
        return int(len(text) / self.chars_per_token) if text else 0


@dataclass
class TiktokenCounter(_BaseCounter):
    """Byte-pair counter backed by `tiktoken`.

    Exact for OpenAI models, where it *is* the API's tokenizer. For other
    providers it's a proxy — the same family of BPE encoding, typically
    within 10–20% — and `exact` stays False so calibration keeps running.
    """

    name: str = "tiktoken"
    exact: bool = False
    encoding_name: str = "o200k_base"
    _enc: Any = field(default=None, repr=False)

    def _encoding(self) -> Any:
        if self._enc is None:
            import tiktoken
            self._enc = tiktoken.get_encoding(self.encoding_name)
        return self._enc

    def _raw_count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._encoding().encode(text, disallowed_special=()))


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _tiktoken_available() -> bool:
    try:
        import tiktoken  # noqa: F401
    except Exception:
        return False
    return True


def _openai_encoding_for(model: str) -> str:
    """Ask tiktoken which encoding this model uses, falling back to the
    current default for unrecognized (e.g. brand-new) model names."""
    try:
        import tiktoken
        return tiktoken.encoding_for_model(model).name
    except Exception:
        return "o200k_base"


def _build_counter(provider: str, model: str) -> TokenCounter:
    p = (provider or "").lower()
    if not _tiktoken_available():
        return HeuristicCounter()
    if p == "openai":
        # The one case where a local counter is the real thing.
        return TiktokenCounter(
            name="tiktoken:openai",
            exact=True,
            encoding_name=_openai_encoding_for(model),
        )
    # Everyone else: BPE proxy, corrected by calibration against the
    # provider's reported usage.
    return TiktokenCounter(name=f"tiktoken:{p or 'unknown'}-proxy", exact=False)


# Counters hold learned calibration, so they're cached per (provider,
# model) and reused for the life of the process.
_CACHE: dict[tuple[str, str], TokenCounter] = {}
_CACHE_LOCK = threading.Lock()


def resolve_counter(provider: str, model: str | None = None) -> TokenCounter:
    """The right counter for this provider/model, created once and reused.

    Callers never branch on provider — that decision lives here.
    """
    key = ((provider or "").lower(), model or "")
    with _CACHE_LOCK:
        counter = _CACHE.get(key)
        if counter is None:
            counter = _build_counter(key[0], key[1])
            _CACHE[key] = counter
        return counter


def reset_counters() -> None:
    """Drop cached counters and their calibration. For tests."""
    with _CACHE_LOCK:
        _CACHE.clear()
