"""Agent-facing search tool — single entry point over many backends.

Default backend is `auto`, which cascades:
    YaCy (curated) → DuckDuckGo (web) → Google-grounded (grounded)

Other modes:
    backend="yacy"        — YaCy only
    backend="duckduckgo"  — DDG only
    backend="google"      — Gemini-grounded only
    backend="tavily"      — paid Tavily
    backend="yacy,google" — explicit cascade

Env overrides (fall back to args):
    MYAGENT_SEARCH_BACKEND   default backend (auto/yacy/...)
    MYAGENT_SEARCH_CASCADE   comma list overriding the auto cascade
    YACY_BASE_URL            (yacy.py)
    GOOGLE_API_KEY           (google.py)
    TAVILY_API_KEY           (tavily.py)

The agent sees a normalized return shape:
    {"query": str, "backend": str, "n_results": int,
     "hits": [{"url": str, "title": str, "snippet": str,
               "source": str, "source_tier": str,
               "published_at": str|None, "score": float, "meta": {...}}],
     "errors": [{"backend": str, "error": str}]}
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..base import JsonTool
from .base import CascadeBackend, SearchResult, WebSearchBackend
# Direct submodule imports — keeps `backends/__init__.py` free to choose
# what (if anything) to re-export at the package level.
from .backends.arxiv import ArxivBackend
from .backends.biorxiv import BiorxivBackend
from .backends.duckduckgo import DuckDuckGoBackend
from .backends.github import GitHubBackend
from .backends.google import GoogleGroundedBackend
from .backends.tavily import TavilyBackend
from .backends.yacy import YaCyBackend


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------


_BACKEND_FACTORIES: dict[str, callable] = {
    "yacy": YaCyBackend,
    "duckduckgo": DuckDuckGoBackend,
    "ddg": DuckDuckGoBackend,
    "google": GoogleGroundedBackend,
    "gemini": GoogleGroundedBackend,  # legacy alias
    "tavily": TavilyBackend,
    # Phase 6.6: specialized backends for academic + code.
    "arxiv": ArxivBackend,
    "biorxiv": BiorxivBackend,
    "medrxiv": BiorxivBackend,  # same Europe PMC endpoint
    "github": GitHubBackend,
}


DEFAULT_AUTO_CASCADE: tuple[str, ...] = ("yacy", "duckduckgo", "google")


def _source_registry_paths() -> list[Path]:
    explicit = os.environ.get("MYAGENT_SEARCH_SOURCES") or os.environ.get("BANNA_SEARCH_SOURCES")
    if explicit:
        return [Path(explicit)]
    paths = [Path.cwd() / "config" / "search_sources.yml"]
    try:
        paths.append(Path(__file__).resolve().parents[4] / "config" / "search_sources.yml")
    except IndexError:
        pass
    return paths


def _normalized_domain(url: str) -> str:
    host = urlparse(url).netloc.split("@")[-1].split(":")[0].lower()
    if host.startswith("www."):
        host = host[4:]
    return host


@lru_cache(maxsize=1)
def _trusted_yacy_urlmask() -> str:
    explicit = os.environ.get("MYAGENT_YACY_URLMASK") or os.environ.get("BANNA_YACY_URLMASK")
    if explicit:
        return explicit

    for path in _source_registry_paths():
        if not path.is_file():
            continue
        try:
            import yaml

            with open(path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except Exception:
            continue

        defaults = data.get("defaults") or {}
        domains: set[str] = set()
        for raw in data.get("sources") or []:
            if not isinstance(raw, dict):
                continue
            src = {**defaults, **raw}
            if not src.get("enabled", True) or src.get("backend") != "yacy":
                continue
            for key in ("base_url", "crawl_url", "rss_url"):
                value = str(src.get(key) or "")
                if not value:
                    continue
                domain = _normalized_domain(value)
                if domain:
                    domains.add(domain)
        if domains:
            alternatives = "|".join(re.escape(domain) for domain in sorted(domains))
            return rf"https?://([^/]+\.)?({alternatives})(/|$).*"
    return ".*"


def _make_backend(name: str, *, source_filter: str | None = None) -> WebSearchBackend | None:
    key = name.strip().lower()
    factory = _BACKEND_FACTORIES.get(key)
    if factory is None:
        return None
    if factory is YaCyBackend:
        return YaCyBackend(
            extra_params={
                "verify": os.environ.get("BANNA_YACY_VERIFY", "cacheonly"),
                "contentdom": "text",
                "urlmaskfilter": source_filter or _trusted_yacy_urlmask(),
                "prefermaskfilter": "",
                "nav": "none",
            }
        )
    return factory()


def _resolve_backend(
    spec: str,
    cascade: list[str] | None,
    *,
    source_filter: str | None = None,
) -> WebSearchBackend:
    """Build a single backend or a CascadeBackend from a string spec."""
    spec = (spec or "auto").strip().lower()

    # Comma-list always builds a cascade.
    if "," in spec:
        names = [s.strip() for s in spec.split(",") if s.strip()]
        return _build_cascade(names, source_filter=source_filter)

    if spec == "auto":
        names = list(cascade) if cascade else list(
            os.environ.get("MYAGENT_SEARCH_CASCADE", "").split(",")
            if os.environ.get("MYAGENT_SEARCH_CASCADE") else DEFAULT_AUTO_CASCADE
        )
        names = [n.strip() for n in names if n.strip()]
        if not names:
            names = list(DEFAULT_AUTO_CASCADE)
        return _build_cascade(names, source_filter=source_filter)

    backend = _make_backend(spec, source_filter=source_filter)
    if backend is None:
        raise ValueError(
            f"unknown search backend: {spec!r}; available: "
            f"{sorted(_BACKEND_FACTORIES)} or 'auto'"
        )
    return backend


def _build_cascade(names: list[str], *, source_filter: str | None = None) -> CascadeBackend:
    backends: list[WebSearchBackend] = []
    for n in names:
        backend = _make_backend(n, source_filter=source_filter)
        if backend is not None:
            backends.append(backend)
    if not backends:
        raise ValueError(f"empty cascade after resolving {names!r}")
    if len(backends) == 1:
        return CascadeBackend(backends)
    return CascadeBackend(backends)


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------


def search(
    query: str,
    *,
    backend: str | None = None,
    cascade: list[str] | None = None,
    n: int = 10,
    max_hits: int | None = None,
    since: str | None = None,
    source_filter: str | None = None,
) -> dict[str, Any]:
    """Run a search and return a normalized payload."""
    if max_hits is not None:
        n = max_hits
    spec = (
        backend
        or os.environ.get("MYAGENT_SEARCH_BACKEND")
        or os.environ.get("BANNA_SEARCH_BACKEND")
        or "auto"
    )
    be = _resolve_backend(spec, cascade, source_filter=source_filter)

    fetched_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    hits: list[SearchResult] = be.search(query, n=n, since=since)

    for h in hits:
        if not h.fetched_at:
            h.fetched_at = fetched_at

    errors = []
    if isinstance(be, CascadeBackend):
        errors = [{"backend": name, "error": err} for name, err in be.last_errors]

    summary = ""
    for hit in hits:
        summary = str(hit.meta.get("summary") or hit.meta.get("answer") or "")
        if summary:
            break
    backend_name = "gemini" if spec.strip().lower() == "gemini" else be.name

    return {
        "query": query,
        "backend": backend_name,
        "n_results": len(hits),
        "hits": [h.to_dict() for h in hits],
        "summary": summary,
        "errors": errors,
    }


def _handler(args: dict[str, Any]) -> dict[str, Any]:
    query = args.get("query", "")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("'query' must be a non-empty string")
    n = int(args.get("max_hits") or args.get("n") or 10)
    backend = args.get("backend")
    since = args.get("since")
    source_filter = args.get("source_filter")
    if source_filter is not None and not isinstance(source_filter, str):
        raise ValueError("'source_filter' must be a string when provided")
    if since is not None and not isinstance(since, str):
        raise ValueError("'since' must be a string when provided")
    return search(query, backend=backend, n=n, since=since, source_filter=source_filter)


SEARCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Web search query."},
        "max_hits": {
            "type": "integer",
            "description": "Max number of results to return (default 10).",
            "default": 10,
        },
        "backend": {
            "type": "string",
            "description": (
                "Optional backend selector. 'auto' (default) cascades "
                "YaCy → DuckDuckGo → Google. 'yacy', 'duckduckgo', "
                "'google', 'tavily' force a single backend. Comma-lists "
                "like 'yacy,google' build a custom cascade."
            ),
        },
        "since": {
            "type": "string",
            "description": (
                "Optional recency filter: 'day', 'week', 'month', or "
                "'year'. YaCy also accepts ISO dates as best-effort filtering."
            ),
        },
        "source_filter": {
            "type": "string",
            "description": "Optional YaCy URL regex filter, e.g. '.*\\.gov/.*'.",
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}


def make_search_tool(
    *,
    backend: str = "auto",
    cascade: list[str] | None = None,
) -> JsonTool:
    """Build the agent-facing search tool.

    `backend` and `cascade` are tool-construction defaults; per-call
    overrides come through the schema's `backend` parameter.
    """
    # Capture defaults in the closure so the model can still override.
    def _bound(args: dict[str, Any]) -> dict[str, Any]:
        a = dict(args)
        query = a.get("query", "")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("'query' must be a non-empty string")
        if "backend" not in a or not a.get("backend"):
            a["backend"] = backend
        source_filter = a.get("source_filter")
        if source_filter is not None and not isinstance(source_filter, str):
            raise ValueError("'source_filter' must be a string when provided")
        # `cascade` is configuration, not tool input — it never appears
        # in `args` from the model. Inject only when backend resolves to
        # auto/cascade modes.
        kwargs = {
            "backend": a["backend"],
            "cascade": cascade,
            "n": int(a.get("max_hits") or a.get("n") or 10),
            "since": a.get("since"),
        }
        if source_filter is not None:
            kwargs["source_filter"] = source_filter
        return search(query, **kwargs)

    return JsonTool(
        name="search",
        description=(
            "Web search across multiple backends (YaCy → DuckDuckGo → "
            "Google by default). Returns a list of grounded hits with "
            "url, title, snippet, source, source_tier, published_at."
        ),
        input_schema=SEARCH_SCHEMA,
        handler=_bound,
        capabilities=frozenset({"network", "read", "llm"}),
    )
