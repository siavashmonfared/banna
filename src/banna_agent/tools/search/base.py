"""Search-backend protocol + unified result shape + cascade.

The agent sees one tool: `search`. Underneath, the call routes through
one or more `WebSearchBackend` implementations (YaCy, DuckDuckGo,
Google-grounded, Tavily, …). Every backend returns the same
`SearchResult` shape so the model (and downstream verifiers) don't have
to special-case the source.

Routing strategy (auto mode):
  CascadeBackend tries backends in order. After each backend, if the
  accumulated result list (deduped by URL) has at least N items, it
  stops. Otherwise it falls through. This means YaCy serves the
  curated/local cache first, DuckDuckGo serves the broad-web fallback,
  and Google-grounded only fires when both prior tiers came up short.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Shared result shape
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    """One web hit, normalized across backends.

    `source_tier` follows the architecture spec:
        "curated"   — YaCy, RSS-fed indexes
        "web"       — DuckDuckGo / generic broad-web
        "grounded"  — provider-mediated (Gemini grounding, Anthropic
                      web_search) where the *provider* asserts source
                      attribution
        "official"  — SEC, PubMed, ClinicalTrials, etc. (added in v2)
        "paid"      — Brave, Exa, Tavily (paid commercial APIs)
    """

    url: str
    title: str = ""
    snippet: str = ""
    source: str = ""              # backend name (yacy, duckduckgo, google, ...)
    source_tier: str = ""         # curated | web | grounded | official | paid
    published_at: str | None = None
    fetched_at: str | None = None
    score: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class WebSearchBackend(Protocol):
    """Every search backend implements this Protocol.

    `search(query, n, since)` returns a list of `SearchResult` ordered by
    backend-native relevance. Implementations should raise rather than
    return an empty list to signal a transport / auth failure — the
    cascade treats raised exceptions as "skip this backend, try the next".
    """

    name: str
    source_tier: str

    def search(
        self,
        query: str,
        *,
        n: int = 10,
        since: str | None = None,
    ) -> list[SearchResult]: ...


# ---------------------------------------------------------------------------
# Cascade
# ---------------------------------------------------------------------------


class CascadeBackend:
    """Try backends in order; stop when enough deduped results accumulate.

    `per_backend_min` sets the floor at which a backend is considered to
    have answered "well enough" for us to skip remaining backends. With
    `per_backend_min=3`, YaCy returning only one hit is *not* enough —
    we'll still consult DuckDuckGo. With `per_backend_min=1`, any single
    YaCy hit prevents the fallback.

    Errors from any single backend are caught and recorded in
    `last_errors` rather than raised — a bad YaCy daemon shouldn't take
    down the whole search call.
    """

    def __init__(
        self,
        backends: Iterable[WebSearchBackend],
        *,
        per_backend_min: int = 3,
    ) -> None:
        self.backends: list[WebSearchBackend] = list(backends)
        self.per_backend_min = per_backend_min
        self.last_errors: list[tuple[str, str]] = []

    @property
    def name(self) -> str:
        return "cascade(" + "→".join(b.name for b in self.backends) + ")"

    @property
    def source_tier(self) -> str:
        return "mixed"

    def search(
        self,
        query: str,
        *,
        n: int = 10,
        since: str | None = None,
    ) -> list[SearchResult]:
        self.last_errors = []
        seen_urls: set[str] = set()
        results: list[SearchResult] = []
        for be in self.backends:
            if len(results) >= n:
                break
            try:
                hits = be.search(query, n=n, since=since)
            except Exception as exc:
                self.last_errors.append((be.name, f"{type(exc).__name__}: {exc}"))
                continue
            new_in_pass = 0
            for h in hits:
                if not h.url or h.url in seen_urls:
                    continue
                seen_urls.add(h.url)
                results.append(h)
                new_in_pass += 1
                if len(results) >= n:
                    break
            # If this backend already filled the floor, we won't try the
            # next one. Otherwise fall through and accumulate.
            if len(results) >= self.per_backend_min and len(results) >= n:
                break
        return results[:n]
