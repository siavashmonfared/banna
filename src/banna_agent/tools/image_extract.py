"""Image-to-text extraction via the vision-capable LLM.

GAIA L2/L3 attaches PNG/JPEG screenshots, photos of documents, charts.
`file_reader._read_image` only returns metadata; this module sends the
image to a vision-capable model and asks a targeted question.

Design:

  * Factory `make_image_extract_tool(llm)` so callers inject whichever
    LLM client is already plumbed for that GAIA run (typically the
    Anthropic client used by the agent loop).
  * Pillow normalizes orientation, downscales to a configurable max
    edge (default 1024 px) before base64 encoding — keeps token cost
    reasonable for big screenshots.
  * Per-content-hash cache so repeated calls on the same image are
    free. The cache key is sha256(question || image_bytes).

The image is delivered as a ContentBlock with `raw` set to the
Anthropic-shaped image dict; `anthropic._block_to_anthropic` prefers
`raw` verbatim, so the wire format is exactly right.
"""
from __future__ import annotations

import base64
import hashlib
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import JsonTool


_DEFAULT_MAX_EDGE_PX = 1024
_DEFAULT_MAX_TOKENS = 400


def _media_type_for(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    return {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png", "gif": "image/gif", "webp": "image/webp",
        "bmp": "image/bmp",
    }.get(ext, "image/png")


def _normalize_image_bytes(path: Path, max_edge_px: int) -> tuple[bytes, str]:
    """Open via Pillow, apply EXIF orientation, downscale, re-encode."""
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:  # pragma: no cover - Pillow is a dep
        raise RuntimeError("Pillow not installed") from exc
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        # Convert palettes / RGBA-with-no-alpha-use to RGB for JPEG.
        media = _media_type_for(path)
        if media == "image/jpeg" and im.mode != "RGB":
            im = im.convert("RGB")
        w, h = im.size
        long_edge = max(w, h)
        if long_edge > max_edge_px:
            scale = max_edge_px / long_edge
            im = im.resize((int(w * scale), int(h * scale)))
        buf = io.BytesIO()
        fmt = {"image/jpeg": "JPEG", "image/png": "PNG",
               "image/gif": "GIF", "image/webp": "WEBP",
               "image/bmp": "BMP"}.get(media, "PNG")
        im.save(buf, format=fmt)
        return buf.getvalue(), media


@dataclass
class _ImageExtractor:
    """Holds the LLM client + a per-process cache."""

    llm: Any
    model: str | None = None
    max_edge_px: int = _DEFAULT_MAX_EDGE_PX
    max_tokens: int = _DEFAULT_MAX_TOKENS
    cache: dict[str, str] = field(default_factory=dict)

    def extract(self, path: str | Path, question: str) -> dict[str, Any]:
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            return {"ok": False, "error": f"no such file: {p}"}
        if not isinstance(question, str) or not question.strip():
            return {"ok": False, "error": "'question' must be a non-empty string"}
        try:
            img_bytes, media_type = _normalize_image_bytes(p, self.max_edge_px)
        except Exception as exc:
            return {"ok": False, "error": f"image preprocessing failed: {exc}"}
        out = self.extract_bytes(img_bytes, media_type, question)
        if out.get("ok"):
            out["path"] = str(p)
        return out

    def extract_bytes(self, img_bytes: bytes, media_type: str,
                      question: str) -> dict[str, Any]:
        """Vision call on raw image bytes (already normalized/encoded by the
        caller). Shared by `extract` and by the PDF page-image path so a
        rasterized PDF page reuses the exact same cached vision plumbing."""
        if not isinstance(question, str) or not question.strip():
            return {"ok": False, "error": "'question' must be a non-empty string"}
        key = hashlib.sha256(question.encode() + b"\0" + img_bytes).hexdigest()
        if key in self.cache:
            return {"ok": True, "answer": self.cache[key], "cached": True,
                    "question": question}

        b64 = base64.b64encode(img_bytes).decode()
        image_block_raw = {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": b64},
        }

        # Build a one-shot user message with [text, image].
        from ..llm.base import ContentBlock, Message
        msg = Message(role="user", content=[
            ContentBlock(kind="text", text=question),
            ContentBlock(kind="unknown", raw=image_block_raw),
        ])
        try:
            reply = self.llm.chat(
                messages=[msg],
                model=self.model,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            return {"ok": False, "error": f"vision call failed: {type(exc).__name__}: {exc}"}

        text_out = " ".join(
            (b.text or "") for b in reply.content if getattr(b, "kind", "") == "text"
        ).strip()
        if not text_out:
            return {"ok": False, "error": "vision call returned no text content"}

        self.cache[key] = text_out
        return {"ok": True, "answer": text_out, "cached": False,
                "question": question,
                "media_type": media_type, "bytes_sent": len(img_bytes)}


def make_image_extract_tool(
    llm: Any,
    *,
    model: str | None = None,
    max_edge_px: int = _DEFAULT_MAX_EDGE_PX,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> JsonTool:
    """Build a vision-extraction JsonTool bound to the given LLM client."""
    extractor = _ImageExtractor(
        llm=llm, model=model,
        max_edge_px=max_edge_px, max_tokens=max_tokens,
    )

    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Filesystem path to a PNG/JPEG/WEBP/GIF/BMP image."},
            "question": {
                "type": "string",
                "description": (
                    "Specific extraction question, e.g. 'What number is on the digit "
                    "display?' or 'Transcribe the visible handwriting verbatim.'"
                ),
            },
        },
        "required": ["path", "question"],
        "additionalProperties": False,
    }

    return JsonTool(
        name="extract_image",
        description=(
            "Send an image to a vision-capable model along with a question and return "
            "the model's text response. Use for reading text inside an image, counting "
            "objects, describing a chart, transcribing handwriting, etc. The same "
            "(image, question) pair is cached, so re-asking is free."
        ),
        input_schema=schema,
        handler=lambda a: extractor.extract(a["path"], a["question"]),
        capabilities=frozenset({"read", "filesystem", "llm"}),
    )
