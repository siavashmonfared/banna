"""Remote-PDF support: pdf_* tools and read_url handle http(s) PDF URLs.

Tooling-gap fix from the GAIA L3 dig: an answer living in a PDF at a URL
was unreachable — read_url dumped raw %PDF bytes and pdf_open took local
paths only. Now the pdf_* tools download + extract a URL, and read_url
steers PDFs to them instead of returning bytes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

import banna_agent.tools.pdf_reader as pdfr
from banna_agent.tools.url_reader import read_url


@dataclass
class _FakeResp:
    status_code: int
    content: bytes
    headers: dict = field(default_factory=dict)
    from_cache: bool = False

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")


def _serve(monkeypatch, resp: _FakeResp, counter: list | None = None) -> None:
    def fake(method, url, *, headers=None, timeout=None, **kw):
        if counter is not None:
            counter.append(url)
        return resp
    monkeypatch.setattr("banna_agent.tools._http_cache.cached_request", fake)


class _FakeReader:
    def __init__(self, texts): self._texts = texts
    @property
    def pages(self):
        return [type("P", (), {"extract_text": (lambda self, t=t: t)})() for t in self._texts]
    @property
    def metadata(self): return {"/Title": "Remote Doc"}
    def close(self): pass


def test_pdf_open_downloads_and_extracts_url(monkeypatch) -> None:
    _serve(monkeypatch, _FakeResp(200, b"%PDF-1.5\n...binary...",
                                  {"Content-Type": "application/pdf"}))
    monkeypatch.setattr(pdfr, "_open_reader",
                        lambda p: _FakeReader(["page one text", "page two"]))
    r = pdfr.pdf_open("https://example.org/paper.pdf")
    assert r["ok"] and r["n_pages"] == 2
    assert "page one" in r["first_page_preview"]


def test_remote_pdf_downloaded_once_then_cached(monkeypatch) -> None:
    calls: list[str] = []
    _serve(monkeypatch, _FakeResp(200, b"%PDF-1.5\nx",
                                  {"Content-Type": "application/pdf"}), calls)
    monkeypatch.setattr(pdfr, "_open_reader", lambda p: _FakeReader(["t"]))
    url = "https://example.org/cached.pdf"
    pdfr.pdf_open(url)
    pdfr.pdf_read_page(url, 1)
    assert len(calls) == 1, "second pdf_* call on same URL must reuse the download"


def test_non_pdf_url_returns_helpful_error(monkeypatch) -> None:
    _serve(monkeypatch, _FakeResp(200, b"<html><body>hi</body></html>",
                                  {"Content-Type": "text/html"}))
    r = pdfr.pdf_open("https://example.org/not-a.pdf")
    assert not r["ok"]
    assert "did not return a PDF" in r["error"]


def test_http_error_surfaced(monkeypatch) -> None:
    _serve(monkeypatch, _FakeResp(404, b"", {"Content-Type": "text/html"}))
    r = pdfr.pdf_open("https://example.org/missing.pdf")
    assert not r["ok"] and "404" in r["error"]


def test_read_url_steers_pdf_to_pdf_tools(monkeypatch) -> None:
    _serve(monkeypatch, _FakeResp(200, b"%PDF-1.7\nstuff",
                                  {"Content-Type": "application/pdf"}))
    r = read_url("https://example.org/doc.pdf")
    assert r.get("is_pdf") is True
    assert "pdf_open" in r["text"]
    assert "%PDF" not in r["text"]  # not raw bytes
