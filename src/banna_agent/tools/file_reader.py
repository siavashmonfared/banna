"""File reader for GAIA-style attachments.

GAIA Level-2/3 questions often attach a file and ask the agent to
interpret it. Supported types (auto-detected by extension):

  - .txt / .md / .log / .json / .xml  -> raw text
  - .csv / .tsv                        -> pandas-read, returned as markdown-style
                                          preview + schema
  - .xlsx                              -> openpyxl-read, sheet list + preview
  - .pdf                               -> pypdf page-extraction
  - .png / .jpg / .jpeg / .gif / .bmp  -> dimensions + mode + optional
                                          EXIF metadata (no OCR; image reasoning
                                          is delegated to multimodal models later)
  - .mp3 / .wav / .m4a                 -> file size + duration (via stdlib
                                          `wave` when possible), no transcript

Every return value is a JSON-serializable dict with at minimum
`{path, kind, size_bytes}`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import JsonTool


DEFAULT_MAX_CHARS = 40_000
DEFAULT_CSV_ROWS = 50


def _read_text(path: Path, max_chars: int) -> dict[str, Any]:
    text = path.read_text(errors="replace")
    truncated = len(text) > max_chars
    return {"kind": "text", "text": text[:max_chars], "truncated": truncated,
            "chars": len(text)}


def _df_preview(df, max_rows: int, max_chars: int) -> str:
    """Try markdown table (if `tabulate` is installed); otherwise fall back
    to CSV-style preview. Always string-safe and truncated."""
    try:
        text = df.head(max_rows).to_markdown(index=False)
    except ImportError:
        text = df.head(max_rows).to_csv(index=False)
    if text and len(text) > max_chars:
        text = text[:max_chars]
    return text or ""


def _read_csv(path: Path, max_rows: int, max_chars: int) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError:
        return _read_text(path, max_chars)
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(path, sep=sep, nrows=max_rows + 10)
    return {
        "kind": "table",
        "columns": [str(c) for c in df.columns],
        "n_rows_shown": min(len(df), max_rows),
        "preview_markdown": _df_preview(df, max_rows, max_chars),
    }


def _read_xlsx(path: Path, max_rows: int, max_chars: int) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError:
        return {"kind": "binary", "note": "pandas not installed; cannot read xlsx"}
    sheets = pd.read_excel(path, sheet_name=None, nrows=max_rows + 10)
    out: dict[str, Any] = {"kind": "workbook", "sheets": {}}
    for name, df in sheets.items():
        out["sheets"][name] = {
            "columns": [str(c) for c in df.columns],
            "n_rows_shown": min(len(df), max_rows),
            "preview_markdown": _df_preview(df, max_rows, max_chars),
        }
    return out


def _read_pdf(path: Path, max_chars: int) -> dict[str, Any]:
    try:
        from pypdf import PdfReader
    except ImportError:
        return {"kind": "binary", "note": "pypdf not installed; cannot read pdf"}
    reader = PdfReader(str(path))
    pages: list[str] = []
    total = 0
    for p in reader.pages:
        t = p.extract_text() or ""
        pages.append(t)
        total += len(t)
        if total > max_chars:
            break
    joined = "\n\n".join(pages)
    truncated = len(joined) > max_chars
    return {
        "kind": "pdf",
        "n_pages": len(reader.pages),
        "text": joined[:max_chars],
        "truncated": truncated,
    }


def _read_image(path: Path) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError:
        return {"kind": "image", "note": "Pillow not installed"}
    with Image.open(path) as im:
        out = {
            "kind": "image",
            "format": im.format,
            "mode": im.mode,
            "size": list(im.size),  # [width, height]
        }
        try:
            exif = im.getexif()
            if exif:
                out["exif_keys"] = sorted(exif.keys())
        except Exception:
            pass
    return out


def _read_audio(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"kind": "audio"}
    if path.suffix.lower() == ".wav":
        try:
            import wave
            with wave.open(str(path)) as w:
                frames = w.getnframes()
                rate = w.getframerate()
                out["duration_s"] = frames / float(rate) if rate else None
                out["sample_rate"] = rate
                out["channels"] = w.getnchannels()
        except Exception as exc:
            out["note"] = f"wave read failed: {exc}"
    else:
        out["note"] = f"no transcript support for {path.suffix}"
    return out


_TEXT_EXTS = {".txt", ".md", ".log", ".json", ".xml", ".yml", ".yaml", ".py"}
_CSV_EXTS = {".csv", ".tsv"}
_XLSX_EXTS = {".xlsx", ".xlsm"}
_PDF_EXTS = {".pdf"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
_DOCX_EXTS = {".docx"}
_ZIP_EXTS = {".zip"}


def _sniff_kind(path: Path) -> str | None:
    """Return a synthetic extension ('.docx', '.xlsx', '.pdf', '.png',
    '.zip') based on the file's magic bytes, or None if nothing matched.

    GAIA-downloaded attachments sometimes lack a file extension entirely
    (the dataset stores them by hash). Without sniffing, read_file falls
    through to _read_text and the agent sees raw binary (the cffe0e32
    Secret Santa failure: `PK\\x03\\x04...[Content_Types].xml` returned
    as text). Sniffing recovers the right reader.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(8)
    except OSError:
        return None
    if head[:4] == b"%PDF":
        return ".pdf"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if head[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    if head[:4] == b"PK\x03\x04":
        # ZIP container — peek at the directory to distinguish
        # docx / xlsx / generic zip.
        import zipfile
        try:
            with zipfile.ZipFile(path) as zf:
                names = set(zf.namelist())
        except zipfile.BadZipFile:
            return ".zip"
        if "word/document.xml" in names:
            return ".docx"
        if "xl/workbook.xml" in names:
            return ".xlsx"
        return ".zip"
    return None


def _read_docx(path: Path, max_chars: int) -> dict[str, Any]:
    """Extract paragraphs from a .docx without python-docx.

    .docx is a zip containing word/document.xml; paragraph text lives
    in <w:t> elements. We pull them with stdlib zipfile + ElementTree
    so this works without an extra dep.
    """
    import xml.etree.ElementTree as ET
    import zipfile
    try:
        with zipfile.ZipFile(path) as zf:
            with zf.open("word/document.xml") as f:
                raw = f.read()
    except (KeyError, zipfile.BadZipFile) as exc:
        return {"kind": "binary", "note": f"not a readable docx: {exc}"}
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        return {"kind": "binary", "note": f"docx xml parse failed: {exc}"}
    paras: list[str] = []
    for p in root.iter(f"{ns}p"):
        parts: list[str] = []
        for t in p.iter(f"{ns}t"):
            if t.text:
                parts.append(t.text)
        if parts:
            paras.append("".join(parts))
    text = "\n\n".join(paras)
    truncated = len(text) > max_chars
    return {
        "kind": "docx",
        "n_paragraphs": len(paras),
        "text": text[:max_chars],
        "truncated": truncated,
    }


def _read_zip(path: Path, max_chars: int) -> dict[str, Any]:
    """List zip members + inline-preview text-like contents up to max_chars total."""
    import zipfile
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        return {"kind": "binary", "note": f"not a readable zip: {exc}"}
    members: list[dict[str, Any]] = []
    previews: list[dict[str, str]] = []
    budget = max_chars
    text_exts = _TEXT_EXTS | _CSV_EXTS  # only inline these
    with zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            members.append({"name": info.filename, "size": info.file_size})
            ext = Path(info.filename).suffix.lower()
            if ext in text_exts and budget > 0:
                try:
                    with zf.open(info) as f:
                        chunk = f.read(min(budget, 8_000)).decode("utf-8", errors="replace")
                except Exception:
                    continue
                previews.append({"name": info.filename, "text": chunk})
                budget -= len(chunk)
    return {
        "kind": "zip",
        "n_members": len(members),
        "members": members[:200],
        "previews": previews,
        "previews_truncated": budget <= 0,
    }


def read_file(path: str | Path, *, max_chars: int = DEFAULT_MAX_CHARS,
              max_rows: int = DEFAULT_CSV_ROWS) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"no such file: {p}")
    if not p.is_file():
        raise ValueError(f"not a regular file: {p}")
    ext = p.suffix.lower()
    size = p.stat().st_size

    # If the extension is missing OR doesn't match a known reader, sniff
    # the magic bytes. Recovers the right reader for GAIA attachments
    # downloaded by content hash with no extension (cffe0e32 bug).
    _all_known = (
        _TEXT_EXTS | _CSV_EXTS | _XLSX_EXTS | _PDF_EXTS |
        _IMAGE_EXTS | _AUDIO_EXTS | _DOCX_EXTS | _ZIP_EXTS
    )
    if ext not in _all_known:
        sniffed = _sniff_kind(p)
        if sniffed is not None:
            ext = sniffed

    if ext in _TEXT_EXTS:
        payload = _read_text(p, max_chars)
    elif ext in _CSV_EXTS:
        payload = _read_csv(p, max_rows=max_rows, max_chars=max_chars)
    elif ext in _XLSX_EXTS:
        payload = _read_xlsx(p, max_rows=max_rows, max_chars=max_chars)
    elif ext in _PDF_EXTS:
        payload = _read_pdf(p, max_chars=max_chars)
    elif ext in _IMAGE_EXTS:
        payload = _read_image(p)
    elif ext in _AUDIO_EXTS:
        payload = _read_audio(p)
    elif ext in _DOCX_EXTS:
        payload = _read_docx(p, max_chars=max_chars)
    elif ext in _ZIP_EXTS:
        payload = _read_zip(p, max_chars=max_chars)
    else:
        # Try text first, fall back to binary note.
        try:
            payload = _read_text(p, max_chars)
        except (UnicodeDecodeError, Exception):
            payload = {"kind": "binary", "note": f"no reader for extension {ext!r}"}

    return {"path": str(p), "ext": ext, "size_bytes": size, **payload}


def _handler(args: dict[str, Any]) -> dict[str, Any]:
    path = args.get("path", "")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("'path' must be a non-empty string")
    max_chars = int(args.get("max_chars", DEFAULT_MAX_CHARS))
    max_rows = int(args.get("max_rows", DEFAULT_CSV_ROWS))
    return read_file(path, max_chars=max_chars, max_rows=max_rows)


FILE_READER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Filesystem path to read. Supports text, csv/tsv, xlsx, pdf, images, audio.",
        },
        "max_chars": {
            "type": "integer",
            "description": f"Maximum characters to return for text-like content (default {DEFAULT_MAX_CHARS}).",
            "default": DEFAULT_MAX_CHARS,
        },
        "max_rows": {
            "type": "integer",
            "description": f"Maximum rows to preview for tabular content (default {DEFAULT_CSV_ROWS}).",
            "default": DEFAULT_CSV_ROWS,
        },
    },
    "required": ["path"],
    "additionalProperties": False,
}


def make_file_reader_tool() -> JsonTool:
    return JsonTool(
        name="read_file",
        description=(
            "Read a file from disk and return its content. Handles text, "
            "csv/tsv, xlsx, pdf, docx, zip, images (metadata only), audio (metadata only)."
        ),
        input_schema=FILE_READER_SCHEMA,
        handler=_handler,
        capabilities=frozenset({"read", "filesystem"}),
    )
