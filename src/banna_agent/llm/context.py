"""Provider-agnostic context accounting.

Every provider has a context window, and every provider deals with an
over-long prompt differently: some error, some drop the oldest messages,
some (Ollama, by default) silently evaluate a truncated prefix and answer
from it. The last case is the dangerous one — the call *succeeds*, so
nothing upstream notices that the tool schemas and the user's question
were cut off before the model ever saw them.

The invariant here holds regardless of provider, model, or prompt:

    what we sent should be roughly what the provider says it counted

`estimate_request_tokens` gives the first half from the payload we build.
`accounted_input_tokens` gives the second half from the `Usage` every
adapter already returns. When they diverge sharply, the prompt was
truncated somewhere in transit and the reply is untrustworthy.

That check needs no per-model context-window table and no provider
special-casing — it reads numbers the providers themselves report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .base import (
    LLMReply,
    Message,
    ProviderError,
    ToolSpec,
    Usage,
)

# Characters per token. Deliberately crude: this is a smoke alarm, not a
# tokenizer. Real tokenizers vary per provider, which is exactly the
# coupling we're avoiding. ~4 chars/token holds within ±30% across every
# BPE tokenizer in use, and the failures worth catching are far larger
# than that.
_CHARS_PER_TOKEN = 4


def _block_chars(block: Any) -> int:
    """Rough serialized size of one ContentBlock."""
    n = 0
    for attr in ("text", "id", "name"):
        v = getattr(block, attr, None)
        if isinstance(v, str):
            n += len(v)
    args = getattr(block, "arguments", None)
    if args:
        n += len(repr(args))
    result = getattr(block, "result", None)
    if result is not None:
        n += len(result) if isinstance(result, str) else len(repr(result))
    return n


def estimate_request_tokens(
    *,
    messages: Sequence[Message] = (),
    tools: Sequence[ToolSpec] = (),
    system: str | None = None,
    counter: Any = None,
) -> int:
    """Input tokens of a chat request.

    Counts the three things that actually grow: the system prompt, the
    tool declarations (which providers bill as input tokens), and the
    conversation. Tool schemas are the usual surprise — a dozen MCP tools
    can outweigh the entire conversation.

    `counter` is a `TokenCounter` from `tokenizers.resolve_counter`; pass
    one to get real tokenization for the provider in play. Without it this
    falls back to the character heuristic, which is enough for callers
    that only need a rough size and don't know the provider.
    """
    if counter is not None:
        return counter.count_request(
            messages=messages, tools=tools, system=system)
    chars = len(system or "")
    for spec in tools:
        chars += len(spec.name) + len(spec.description)
        chars += len(repr(spec.input_schema))
    for msg in messages:
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            chars += len(content)
            continue
        for block in content or ():
            chars += _block_chars(block)
    return chars // _CHARS_PER_TOKEN


def accounted_input_tokens(usage: Usage | None) -> int:
    """Input tokens the provider says it processed.

    Cached tokens still count as "seen" — a cache hit means the provider
    had them, not that it skipped them. Summing the columns keeps the
    comparison honest for providers that report caching separately.
    """
    if usage is None:
        return 0
    return (
        int(usage.tokens_in or 0)
        + int(usage.cache_read_tokens or 0)
        + int(usage.cache_write_tokens or 0)
        + int(usage.tool_prompt_tokens or 0)
    )


# A reply is flagged only when the shortfall is both proportionally large
# and absolutely large. The ratio tolerates tokenizer drift; the floor
# stops short prompts (where the estimate is noisiest) from tripping it.
#
# The drift this has to tolerate is one-sided in our favor: JSON and code
# tokenize denser than 4 chars/token, so the estimate tends to run *under*
# the true count. An overshoot large enough to fake a 25% shortfall would
# mean the heuristic was wrong in the unlikely direction.
_TRUNCATION_RATIO = 0.75
_TRUNCATION_FLOOR_TOKENS = 1000


def truncation_shortfall(
    *,
    estimated: int,
    usage: Usage | None,
) -> int:
    """Tokens the provider appears to have dropped, or 0 if it looks fine.

    Returns 0 when the provider reports no usage at all — absence of
    telemetry is not evidence of truncation.
    """
    accounted = accounted_input_tokens(usage)
    if accounted <= 0 or estimated <= 0:
        return 0
    gap = estimated - accounted
    if gap < _TRUNCATION_FLOOR_TOKENS:
        return 0
    if accounted >= estimated * _TRUNCATION_RATIO:
        return 0
    return gap


@dataclass
class ContextGuard:
    """Wraps any `LLMClient` and turns silent prompt truncation into a loud,
    non-retryable error.

    Delegates `chat` untouched, then audits the reply's usage against what
    we estimated we sent. Attribute access falls through to the wrapped
    client, so this is transparent to callers that reach for adapter
    specifics like `.model` or `.base_url`.
    """

    inner: Any
    # Set False to downgrade to a warning — useful for benchmark runs where
    # a degraded answer beats a hard stop.
    strict: bool = True
    warn: Any = None
    # Input-token counts seen so far, paired with what we estimated at the
    # time. Feeds the plateau check below.
    _history: list[tuple[int, int]] = field(default_factory=list, repr=False)

    @property
    def provider(self) -> str:
        return getattr(self.inner, "provider", "unknown")

    def __getattr__(self, name: str) -> Any:
        # Only called when normal lookup fails, so declared fields win.
        # `inner` itself must never route here — during unpickling or any
        # path where it isn't set yet, delegating would recurse forever.
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def _plateaued(self, estimated: int, accounted: int) -> bool:
        """True when the provider's input count has stopped moving even
        though our prompt kept growing.

        This is the signature a clamped window leaves in the telemetry: an
        agent loop appends a tool result every turn, so the input count
        must rise monotonically. An identical count across two turns whose
        prompts differ by a fifth doesn't happen naturally — token counts
        that precise don't collide — but it is exactly what a capped
        window produces once the conversation outgrows it.
        """
        for prev_est, prev_acc in self._history:
            if prev_acc == accounted and estimated > prev_est * 1.2:
                return True
        return False

    def _counter(self, model: str | None) -> Any:
        from .tokenizers import resolve_counter
        return resolve_counter(
            self.provider, model or getattr(self.inner, "model", ""))

    def chat(self, **kwargs: Any) -> LLMReply:
        counter = self._counter(kwargs.get("model"))
        estimated = estimate_request_tokens(
            messages=kwargs.get("messages") or (),
            tools=kwargs.get("tools") or (),
            system=kwargs.get("system"),
            counter=counter,
        )
        reply = self.inner.chat(**kwargs)
        accounted = accounted_input_tokens(getattr(reply, "usage", None))
        shortfall = truncation_shortfall(
            estimated=estimated, usage=getattr(reply, "usage", None))
        if not shortfall and accounted > 0:
            if self._plateaued(estimated, accounted):
                shortfall = max(estimated - accounted, 1)
            else:
                # Calibrate only on healthy replies. Learning from a
                # truncated one would shrink the estimate until the
                # truncation stopped looking like truncation.
                counter.observe(estimated=estimated, actual=accounted)
                self._history.append((estimated, accounted))
                del self._history[:-8]
        if not shortfall:
            return reply

        ntools = len(kwargs.get("tools") or ())
        msg = (
            f"{self.provider}: prompt truncated — we sent ~{estimated} input "
            f"tokens but the provider counted {accounted} "
            f"(~{shortfall} dropped). The model did not see the whole "
            f"request, so this reply is unreliable. Most likely the context "
            f"window is smaller than the request: {ntools} tool schema(s) "
            f"plus the system prompt are sent on every turn. Raise the "
            f"model's context window, or reduce the number of attached "
            f"tools/MCP servers."
        )
        if self.strict:
            raise ProviderError(msg, retryable=False)
        if self.warn is not None:
            self.warn(msg)
        return reply
