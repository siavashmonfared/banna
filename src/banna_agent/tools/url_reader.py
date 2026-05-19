"""Fetch a URL and return its text content.

Intentionally minimal: `requests.get` with a timeout, a user agent, and a
size cap. HTML goes through BeautifulSoup for text extraction; other
content types are returned as raw text (truncated).

GAIA questions often paste URLs; the agent calls this tool to read them.
The `max_chars` guard is important — GAIA allows file attachments up to
a few MB, and an unbounded read will blow out a single LLM context.
"""
from __future__ import annotations

from typing import Any

from .base import JsonTool

DEFAULT_TIMEOUT_S = 20.0
DEFAULT_MAX_CHARS = 50_000
USER_AGENT = "banna_agent/0.1 (+https://github.com/)"


def _extract_text(html: str) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return html
    soup = BeautifulSoup(html, "html.parser")
    # Strip noise before text extraction.
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return text


def read_url(url: str, *, max_chars: int = DEFAULT_MAX_CHARS,
             timeout_s: float = DEFAULT_TIMEOUT_S) -> dict[str, Any]:
    """Return `{url, status, content_type, title, text, truncated, from_cache}`."""
    from ._http_cache import cached_request
    resp = cached_request(
        "GET", url,
        headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
        timeout=timeout_s,
    )
    status = resp.status_code
    content_type = resp.headers.get("Content-Type", "").split(";")[0].strip()
    body = resp.text or ""

    # Detect HTML by Content-Type OR body sniff — some cached/proxied
    # responses arrive with an empty content-type but obviously-HTML body.
    is_html = "html" in content_type
    if not is_html and body:
        head = body.lstrip()[:200].lower()
        if head.startswith("<!doctype html") or head.startswith("<html") or "<html" in head[:200]:
            is_html = True

    title = ""
    if is_html:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(body, "html.parser")
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
        except ImportError:
            pass
        text = _extract_text(body)
    else:
        text = body

    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars]

    return {
        "url": url,
        "status": status,
        "content_type": content_type,
        "title": title,
        "text": text,
        "truncated": truncated,
        "from_cache": resp.from_cache,
    }


def _handler(args: dict[str, Any]) -> dict[str, Any]:
    url = args.get("url", "")
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise ValueError("'url' must be an http(s) string")
    max_chars = int(args.get("max_chars", DEFAULT_MAX_CHARS))
    return read_url(url, max_chars=max_chars)


URL_READER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": "An http:// or https:// URL to fetch and extract text from.",
        },
        "max_chars": {
            "type": "integer",
            "description": f"Maximum characters to return (default {DEFAULT_MAX_CHARS}).",
            "default": DEFAULT_MAX_CHARS,
        },
    },
    "required": ["url"],
    "additionalProperties": False,
}


def make_url_reader_tool() -> JsonTool:
    return JsonTool(
        name="read_url",
        description="Fetch a URL and return extracted text (HTML -> text via BeautifulSoup).",
        input_schema=URL_READER_SCHEMA,
        handler=_handler,
        capabilities=frozenset({"network", "read"}),
    )
