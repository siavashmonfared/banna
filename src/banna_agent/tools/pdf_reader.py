"""Paginated PDF reader.

`file_reader.read_file` returns a single concatenated PDF text capped
at `max_chars`, which is too coarse for GAIA L2 questions that point
at a specific page of an attached PDF. This module exposes per-page
access:

  * `pdf_open(path)` — open + return page count, title, first-page preview.
  * `pdf_read_page(path, page)` — full text of one page (1-indexed).
  * `pdf_find(path, query)` — substring scan across all pages.
  * `pdf_read_tables(path, page)` — extract tabular structure (when
    pdfplumber is installed).

Extraction backend selection: `pdfplumber` is preferred when
importable — it gives better whitespace handling and is the only
backend that exposes tables. When pdfplumber isn't installed, we fall
back to `pypdf` (always available as a project dep). `pdf_read_tables`
returns a "pdfplumber required" error when the optional dep is
missing; install with `pip install ".[pdf]"`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import JsonTool


_DEFAULT_PAGE_MAX_CHARS = 8_000
_FIND_SNIPPET_PAD = 200
_FIND_MAX_HITS = 20


def _have_pdfplumber() -> bool:
    try:
        import pdfplumber  # noqa: F401
        return True
    except ImportError:
        return False


class _PypdfBackend:
    """Wraps pypdf to look like the small subset we need."""

    def __init__(self, path: Path) -> None:
        from pypdf import PdfReader
        self._reader = PdfReader(str(path))

    @property
    def pages(self):  # list-like with extract_text()
        return self._reader.pages

    @property
    def metadata(self):
        try:
            return self._reader.metadata or {}
        except Exception:
            return {}

    def close(self) -> None:
        pass


class _PdfplumberBackend:
    """Wraps pdfplumber with the same surface."""

    def __init__(self, path: Path) -> None:
        import pdfplumber
        self._doc = pdfplumber.open(str(path))

    @property
    def pages(self):
        return self._doc.pages

    @property
    def metadata(self):
        return self._doc.metadata or {}

    def close(self) -> None:
        self._doc.close()


def _open_reader(path: Path):
    """Prefer pdfplumber when available; otherwise pypdf.

    Both backends expose `.pages[i].extract_text() -> str` and
    `.metadata` as a dict-like.
    """
    if _have_pdfplumber():
        return _PdfplumberBackend(path)
    try:
        return _PypdfBackend(path)
    except ImportError as exc:  # pragma: no cover - pypdf is a project dep
        raise RuntimeError("neither pdfplumber nor pypdf is installed") from exc


def _safe_extract(page) -> str:
    try:
        return page.extract_text() or ""
    except Exception as exc:
        return f"[pdf page extraction failed: {type(exc).__name__}: {exc}]"


def pdf_read_tables(path: str | Path, page: int) -> dict:
    """Extract tables from one PDF page. Requires pdfplumber.

    Returns ``{"ok": True, "tables": [ [[cell, …], …], … ]}`` — a list
    of tables, each a list of rows. Returns a structured ok=False error
    if pdfplumber isn't installed.
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        return {"ok": False, "error": f"no such file: {p}"}
    if not _have_pdfplumber():
        return {
            "ok": False,
            "error": "pdfplumber not installed",
            "hint": "install with `pip install \".[pdf]\"`",
        }
    try:
        import pdfplumber
        with pdfplumber.open(str(p)) as doc:
            n = len(doc.pages)
            page_i = int(page)
            if page_i < 1 or page_i > n:
                return {"ok": False, "error": f"page {page_i} out of range [1, {n}]"}
            tables = doc.pages[page_i - 1].extract_tables() or []
    except Exception as exc:
        return {"ok": False, "error": f"table extraction failed: {type(exc).__name__}: {exc}"}
    # pdfplumber may return Nones inside rows; coerce to "".
    clean = [[[("" if c is None else str(c)) for c in row] for row in tbl] for tbl in tables]
    return {
        "ok": True, "path": str(p), "page": page_i,
        "n_tables": len(clean), "tables": clean,
    }


def pdf_open(path: str | Path, *, preview_chars: int = 1200) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        return {"ok": False, "error": f"no such file: {p}"}
    try:
        reader = _open_reader(p)
    except Exception as exc:
        return {"ok": False, "error": f"open failed: {exc}"}
    n = len(reader.pages)
    title = ""
    try:
        meta = reader.metadata or {}
        # pypdf uses "/Title", pdfplumber uses "Title". Try both.
        if hasattr(meta, "get"):
            title = str(meta.get("/Title") or meta.get("Title") or "")
    except Exception:
        title = ""
    first = _safe_extract(reader.pages[0]) if n else ""
    return {
        "ok": True,
        "path": str(p),
        "n_pages": n,
        "title": title,
        "first_page_preview": first[:preview_chars],
        "first_page_truncated": len(first) > preview_chars,
    }


def pdf_read_page(path: str | Path, page: int, *, max_chars: int = _DEFAULT_PAGE_MAX_CHARS) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        return {"ok": False, "error": f"no such file: {p}"}
    try:
        reader = _open_reader(p)
    except Exception as exc:
        return {"ok": False, "error": f"open failed: {exc}"}
    n = len(reader.pages)
    page_i = int(page)
    if page_i < 1 or page_i > n:
        return {"ok": False, "error": f"page {page_i} out of range [1, {n}]"}
    text = _safe_extract(reader.pages[page_i - 1])
    truncated = len(text) > max_chars
    return {
        "ok": True,
        "path": str(p),
        "page": page_i,
        "n_pages": n,
        "text": text[:max_chars],
        "truncated": truncated,
        "total_chars": len(text),
    }


def pdf_find(path: str | Path, query: str, *, max_hits: int = _FIND_MAX_HITS) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        return {"ok": False, "error": f"no such file: {p}"}
    if not isinstance(query, str) or not query.strip():
        return {"ok": False, "error": "'query' must be non-empty"}
    try:
        reader = _open_reader(p)
    except Exception as exc:
        return {"ok": False, "error": f"open failed: {exc}"}
    q = query.lower()
    hits: list[dict[str, Any]] = []
    for i, page in enumerate(reader.pages, 1):
        text = _safe_extract(page)
        lower = text.lower()
        start = 0
        while len(hits) < max_hits:
            j = lower.find(q, start)
            if j < 0:
                break
            lo = max(0, j - _FIND_SNIPPET_PAD)
            hi = min(len(text), j + len(query) + _FIND_SNIPPET_PAD)
            hits.append({"page": i, "snippet": text[lo:hi], "char_offset": j})
            start = j + max(1, len(query))
        if len(hits) >= max_hits:
            break
    return {
        "ok": True, "path": str(p), "query": query,
        "n_hits": len(hits), "hits": hits,
        "max_hits_reached": len(hits) >= max_hits,
    }


# ---------------------------------------------------------------------------
# JsonTool factories
# ---------------------------------------------------------------------------


def make_pdf_tools() -> tuple[JsonTool, ...]:
    open_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "preview_chars": {"type": "integer", "default": 1200},
        },
        "required": ["path"],
        "additionalProperties": False,
    }
    page_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "page": {"type": "integer", "description": "1-indexed page number."},
            "max_chars": {"type": "integer", "default": _DEFAULT_PAGE_MAX_CHARS},
        },
        "required": ["path", "page"],
        "additionalProperties": False,
    }
    find_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "query": {"type": "string"},
            "max_hits": {"type": "integer", "default": _FIND_MAX_HITS},
        },
        "required": ["path", "query"],
        "additionalProperties": False,
    }
    tables_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "page": {"type": "integer", "description": "1-indexed page number."},
        },
        "required": ["path", "page"],
        "additionalProperties": False,
    }
    return (
        JsonTool(
            name="pdf_open",
            description=(
                "Open a PDF and return its page count, title, and a short preview of page 1. "
                "Follow up with pdf_read_page for a specific page or pdf_find to locate a string."
            ),
            input_schema=open_schema,
            handler=lambda a: pdf_open(a["path"], preview_chars=int(a.get("preview_chars", 1200))),
            capabilities=frozenset({"read", "filesystem"}),
        ),
        JsonTool(
            name="pdf_read_page",
            description="Return the full extracted text of one PDF page (1-indexed).",
            input_schema=page_schema,
            handler=lambda a: pdf_read_page(
                a["path"], int(a["page"]), max_chars=int(a.get("max_chars", _DEFAULT_PAGE_MAX_CHARS))
            ),
            capabilities=frozenset({"read", "filesystem"}),
        ),
        JsonTool(
            name="pdf_find",
            description=(
                "Case-insensitive substring search across all pages of a PDF. "
                "Returns up to max_hits results with the page number and surrounding snippet."
            ),
            input_schema=find_schema,
            handler=lambda a: pdf_find(
                a["path"], a["query"], max_hits=int(a.get("max_hits", _FIND_MAX_HITS))
            ),
            capabilities=frozenset({"read", "filesystem"}),
        ),
        JsonTool(
            name="pdf_read_tables",
            description=(
                "Extract tabular structure from one PDF page (1-indexed). Returns a "
                "list of tables, each a list of rows. Requires the optional pdfplumber "
                "dep (install with `pip install \".[pdf]\"`); returns a structured "
                "error otherwise."
            ),
            input_schema=tables_schema,
            handler=lambda a: pdf_read_tables(a["path"], int(a["page"])),
            capabilities=frozenset({"read", "filesystem"}),
        ),
    )
