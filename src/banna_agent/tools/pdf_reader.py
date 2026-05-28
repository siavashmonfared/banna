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

import hashlib
import tempfile
from pathlib import Path
from typing import Any

from .base import JsonTool


_DEFAULT_PAGE_MAX_CHARS = 8_000
_FIND_SNIPPET_PAD = 200
_FIND_MAX_HITS = 20

# Cap on a downloaded PDF so a giant file can't blow out memory/disk.
_PDF_MAX_BYTES = 30_000_000
_PDF_FETCH_TIMEOUT_S = 30.0
# Per-process url -> local temp path, so paging through a remote PDF
# (open, then several read_page calls) downloads it only once.
_PDF_URL_CACHE: dict[str, Path] = {}


def _looks_like_url(path: str | Path) -> bool:
    return isinstance(path, str) and path.startswith(("http://", "https://"))


def _download_pdf(url: str) -> tuple[Path | None, str | None]:
    """Fetch a remote PDF to a temp file; return (local_path, error).

    Reuses the shared HTTP cache (so record/replay GAIA runs stay
    deterministic) and verifies the bytes are actually a PDF before
    handing back a path. Cached per-URL for the life of the process.
    """
    cached = _PDF_URL_CACHE.get(url)
    if cached is not None and cached.is_file():
        return cached, None
    try:
        from ._http_cache import cached_request
        resp = cached_request(
            "GET", url,
            headers={"User-Agent": "banna_agent/0.1 (+https://github.com/)",
                     "Accept": "application/pdf,*/*"},
            timeout=_PDF_FETCH_TIMEOUT_S,
        )
    except Exception as exc:
        return None, f"download failed: {type(exc).__name__}: {exc}"
    if resp.status_code >= 400:
        return None, f"download failed: HTTP {resp.status_code} for {url}"
    body = resp.content or b""
    if len(body) > _PDF_MAX_BYTES:
        return None, (f"PDF too large ({len(body)} bytes > {_PDF_MAX_BYTES} cap); "
                      f"not downloaded")
    ctype = resp.headers.get("Content-Type", "").lower()
    if not body.lstrip()[:5].startswith(b"%PDF") and "pdf" not in ctype:
        # Not a PDF — tell the model to use read_url / browser_open instead
        # of silently handing back a path to non-PDF bytes.
        return None, (f"URL did not return a PDF (content-type {ctype or 'unknown'}); "
                      f"use read_url or browser_open for non-PDF pages")
    fd, name = tempfile.mkstemp(
        prefix=f"banna_pdf_{hashlib.sha256(url.encode()).hexdigest()[:12]}_",
        suffix=".pdf",
    )
    import os
    with os.fdopen(fd, "wb") as f:
        f.write(body)
    p = Path(name)
    _PDF_URL_CACHE[url] = p
    return p, None


def _resolve_pdf_source(path: str | Path) -> tuple[Path | None, str | None]:
    """Resolve a PDF source to a local Path, downloading http(s) URLs.

    Returns (path, None) on success or (None, error_message) — callers
    surface the error as a structured ``{"ok": False, "error": ...}``.
    """
    if _looks_like_url(path):
        return _download_pdf(str(path))
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        return None, f"no such file: {p}"
    return p, None


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


# A page whose text layer strips to fewer than this many chars is treated
# as scanned (image-only) — the trigger for the vision-OCR fallback.
_SCANNED_TEXT_THRESHOLD = 24
_RASTER_DPI = 150
_OCR_PROMPT = (
    "Transcribe ALL text visible on this page verbatim, preserving reading "
    "order and structure. Render any table as text rows. Output only the "
    "transcription, no commentary."
)


def _rasterize_pdf_page(path: Path, page_i: int, *, dpi: int = _RASTER_DPI
                        ) -> tuple[bytes | None, str | None]:
    """Render one PDF page (1-indexed) to PNG bytes via PyMuPDF.

    Returns (png_bytes, None) or (None, error). PyMuPDF ships as a wheel
    (no system binaries); a clear message is returned if it's absent.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return None, ("PyMuPDF not installed — needed to render scanned/"
                      "image PDF pages. Install with `pip install pymupdf`.")
    try:
        doc = fitz.open(str(path))
        n = doc.page_count
        if page_i < 1 or page_i > n:
            doc.close()
            return None, f"page {page_i} out of range [1, {n}]"
        pix = doc[page_i - 1].get_pixmap(dpi=dpi)
        png = pix.tobytes("png")
        doc.close()
        return png, None
    except Exception as exc:
        return None, f"rasterize failed: {type(exc).__name__}: {exc}"


def pdf_read_page_visual(path: str | Path, page: int, question: str, *,
                         vision: Any = None) -> dict[str, Any]:
    """Render a PDF page to an image and ask the vision model about it.

    For pages where the answer is in a picture, not the text layer —
    scanned documents and figures/charts/plots. `vision` is an
    `_ImageExtractor`-shaped object (`.extract_bytes(bytes, media, q)`);
    when absent the tool reports that vision is unavailable.
    """
    if vision is None:
        return {"ok": False, "error": "vision model not available for this run"}
    if not isinstance(question, str) or not question.strip():
        return {"ok": False, "error": "'question' must be non-empty"}
    p, err = _resolve_pdf_source(path)
    if err:
        return {"ok": False, "error": err}
    png, rerr = _rasterize_pdf_page(p, int(page))
    if rerr:
        return {"ok": False, "error": rerr}
    out = vision.extract_bytes(png, "image/png", question)
    if not out.get("ok"):
        return out
    return {"ok": True, "path": str(p), "page": int(page),
            "question": question, "answer": out["answer"],
            "source": "vision", "cached": out.get("cached", False)}


def pdf_read_tables(path: str | Path, page: int) -> dict:
    """Extract tables from one PDF page. Requires pdfplumber.

    Returns ``{"ok": True, "tables": [ [[cell, …], …], … ]}`` — a list
    of tables, each a list of rows. Returns a structured ok=False error
    if pdfplumber isn't installed.
    """
    p, err = _resolve_pdf_source(path)
    if err:
        return {"ok": False, "error": err}
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
    p, err = _resolve_pdf_source(path)
    if err:
        return {"ok": False, "error": err}
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


def pdf_read_page(path: str | Path, page: int, *, max_chars: int = _DEFAULT_PAGE_MAX_CHARS,
                  vision: Any = None) -> dict[str, Any]:
    p, err = _resolve_pdf_source(path)
    if err:
        return {"ok": False, "error": err}
    try:
        reader = _open_reader(p)
    except Exception as exc:
        return {"ok": False, "error": f"open failed: {exc}"}
    n = len(reader.pages)
    page_i = int(page)
    if page_i < 1 or page_i > n:
        return {"ok": False, "error": f"page {page_i} out of range [1, {n}]"}
    text = _safe_extract(reader.pages[page_i - 1])

    # Scanned-page fallback: a near-empty text layer means the page is an
    # image. When a vision model is wired in, OCR it via the page-image
    # path. Born-digital pages (which extract fine) never reach here, so
    # they never pay the per-page vision cost.
    if vision is not None and len(text.strip()) < _SCANNED_TEXT_THRESHOLD:
        png, rerr = _rasterize_pdf_page(p, page_i)
        if rerr is None:
            ocr = vision.extract_bytes(png, "image/png", _OCR_PROMPT)
            if ocr.get("ok") and ocr["answer"].strip():
                ocr_text = ocr["answer"]
                return {
                    "ok": True, "path": str(p), "page": page_i, "n_pages": n,
                    "text": ocr_text[:max_chars],
                    "truncated": len(ocr_text) > max_chars,
                    "total_chars": len(ocr_text),
                    "source": "ocr",  # transcribed by the vision model
                }

    truncated = len(text) > max_chars
    return {
        "ok": True,
        "path": str(p),
        "page": page_i,
        "n_pages": n,
        "text": text[:max_chars],
        "truncated": truncated,
        "total_chars": len(text),
        "source": "text",
    }


def pdf_find(path: str | Path, query: str, *, max_hits: int = _FIND_MAX_HITS) -> dict[str, Any]:
    if not isinstance(query, str) or not query.strip():
        return {"ok": False, "error": "'query' must be non-empty"}
    p, err = _resolve_pdf_source(path)
    if err:
        return {"ok": False, "error": err}
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


def make_pdf_tools(llm: Any = None) -> tuple[JsonTool, ...]:
    """Build the PDF tools. When `llm` is a vision-capable client, scanned
    pages get an automatic OCR fallback and a `pdf_read_page_visual` tool
    (figures/charts/scanned docs) is added. Without `llm`, the four
    text-extraction tools work exactly as before."""
    vision = None
    if llm is not None:
        from .image_extract import _ImageExtractor
        vision = _ImageExtractor(llm=llm)
    open_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string",
                     "description": "Local file path OR an http(s) URL to a PDF "
                                    "(remote PDFs are downloaded and extracted)."},
            "preview_chars": {"type": "integer", "default": 1200},
        },
        "required": ["path"],
        "additionalProperties": False,
    }
    page_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string",
                     "description": "Local file path OR an http(s) URL to a PDF "
                                    "(remote PDFs are downloaded and extracted)."},
            "page": {"type": "integer", "description": "1-indexed page number."},
            "max_chars": {"type": "integer", "default": _DEFAULT_PAGE_MAX_CHARS},
        },
        "required": ["path", "page"],
        "additionalProperties": False,
    }
    find_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string",
                     "description": "Local file path OR an http(s) URL to a PDF "
                                    "(remote PDFs are downloaded and extracted)."},
            "query": {"type": "string"},
            "max_hits": {"type": "integer", "default": _FIND_MAX_HITS},
        },
        "required": ["path", "query"],
        "additionalProperties": False,
    }
    tables_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string",
                     "description": "Local file path OR an http(s) URL to a PDF "
                                    "(remote PDFs are downloaded and extracted)."},
            "page": {"type": "integer", "description": "1-indexed page number."},
        },
        "required": ["path", "page"],
        "additionalProperties": False,
    }
    visual_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string",
                     "description": "Local file path OR an http(s) URL to a PDF."},
            "page": {"type": "integer", "description": "1-indexed page number."},
            "question": {"type": "string",
                         "description": "What to read off the rendered page image, "
                                        "e.g. 'transcribe this page' or 'what value "
                                        "does the curve reach at t=0?'"},
        },
        "required": ["path", "page", "question"],
        "additionalProperties": False,
    }
    tools = [
        JsonTool(
            name="pdf_open",
            description=(
                "Open a PDF and return its page count, title, and a short preview of page 1. "
                "Accepts a local file path or an http(s) URL (a remote PDF is fetched and "
                "extracted). Follow up with pdf_read_page for a specific page or pdf_find "
                "to locate a string."
            ),
            input_schema=open_schema,
            handler=lambda a: pdf_open(a["path"], preview_chars=int(a.get("preview_chars", 1200))),
            capabilities=frozenset({"read", "filesystem"}),
        ),
        JsonTool(
            name="pdf_read_page",
            description=(
                "Return the full extracted text of one PDF page (1-indexed). If the page "
                "has no text layer (a scanned image), the page is rendered and read by a "
                "vision model automatically when one is available."
            ),
            input_schema=page_schema,
            handler=lambda a: pdf_read_page(
                a["path"], int(a["page"]),
                max_chars=int(a.get("max_chars", _DEFAULT_PAGE_MAX_CHARS)),
                vision=vision,
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
    ]
    # The vision-backed page reader is only useful with an LLM wired in;
    # add it only then, so make_pdf_tools() (no llm) keeps its 4 tools.
    if vision is not None:
        tools.append(JsonTool(
            name="pdf_read_page_visual",
            description=(
                "Render a PDF page to an image and ask a vision model about it. Use when "
                "a page's answer is in a picture rather than its text — a scanned "
                "(image-only) page, or a figure/chart/plot you need to read a value from."
            ),
            input_schema=visual_schema,
            handler=lambda a: pdf_read_page_visual(
                a["path"], int(a["page"]), a["question"], vision=vision,
            ),
            capabilities=frozenset({"read", "filesystem", "llm"}),
        ))
    return tuple(tools)
