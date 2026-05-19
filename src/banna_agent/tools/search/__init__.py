"""Web search package — multi-backend, agent-facing.

The agent sees a single `search` tool. Underneath, the tool routes
through a `WebSearchBackend` chain. Default mode is `auto`, which
cascades:

    YaCy (curated)  →  DuckDuckGo (web)  →  Google-grounded (grounded)

See `tool.py` for the agent-facing interface, `base.py` for the
`SearchResult` shape and `WebSearchBackend` Protocol, and
`backends/` for concrete implementations.
"""

from .base import CascadeBackend, SearchResult, WebSearchBackend
from .backends.duckduckgo import DuckDuckGoBackend
from .backends.google import GoogleGroundedBackend
from .backends.tavily import TavilyBackend
from .backends.yacy import YaCyBackend
from .tool import (
    DEFAULT_AUTO_CASCADE,
    SEARCH_SCHEMA,
    make_search_tool,
    search,
)


__all__ = [
    # Public API
    "make_search_tool",
    "search",
    "SEARCH_SCHEMA",
    "DEFAULT_AUTO_CASCADE",
    # Types
    "SearchResult",
    "WebSearchBackend",
    "CascadeBackend",
    # Backends
    "YaCyBackend",
    "DuckDuckGoBackend",
    "GoogleGroundedBackend",
    "TavilyBackend",
]
