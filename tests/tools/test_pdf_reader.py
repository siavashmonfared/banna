"""pdf_reader tests.

Generating a real text-bearing PDF in the test would require reportlab
(not a dep), so we monkeypatch `pypdf.PdfReader` with a stub that
yields canned page text. The tests still exercise the full pdf_reader
code path: page indexing, snippet extraction, search across pages.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from banna_agent.tools import pdf_reader as pdfr


class _FakePage:
    def __init__(self, text: str) -> None:
        self._t = text

    def extract_text(self) -> str:
        return self._t


class _FakeReader:
    def __init__(self, pages: list[str]) -> None:
        self.pages = [_FakePage(t) for t in pages]
        self.metadata = {"/Title": "Sample"}


@pytest.fixture
def fake_pdf(tmp_path: Path, monkeypatch):
    p = tmp_path / "sample.pdf"
    p.write_bytes(b"%PDF-1.4\n")  # contents irrelevant; we mock the reader
    pages = [
        "Page one. Topic Z was discovered in 1923 by Anna.",
        "Page two with no useful info.",
        "Page three mentions 1923 again, in a different context.",
    ]
    monkeypatch.setattr(pdfr, "_open_reader", lambda path: _FakeReader(pages))
    return p


def test_pdf_open_returns_metadata_and_preview(fake_pdf) -> None:
    r = pdfr.pdf_open(fake_pdf)
    assert r["ok"] is True
    assert r["n_pages"] == 3
    assert r["title"] == "Sample"
    assert "Topic Z" in r["first_page_preview"]


def test_pdf_read_page_returns_one_page(fake_pdf) -> None:
    r = pdfr.pdf_read_page(fake_pdf, 2)
    assert r["ok"] is True
    assert r["page"] == 2
    assert "Page two" in r["text"]


def test_pdf_read_page_rejects_out_of_range(fake_pdf) -> None:
    r = pdfr.pdf_read_page(fake_pdf, 99)
    assert r["ok"] is False
    assert "out of range" in r["error"]


def test_pdf_find_returns_hits_across_pages(fake_pdf) -> None:
    r = pdfr.pdf_find(fake_pdf, "1923")
    assert r["ok"] is True
    assert r["n_hits"] == 2
    pages_hit = sorted({h["page"] for h in r["hits"]})
    assert pages_hit == [1, 3]
    assert "1923" in r["hits"][0]["snippet"]


def test_pdf_find_handles_no_matches(fake_pdf) -> None:
    r = pdfr.pdf_find(fake_pdf, "nonexistent-term-xyz")
    assert r["ok"] is True
    assert r["n_hits"] == 0


def test_make_pdf_tools_registers_four_handlers() -> None:
    tools = pdfr.make_pdf_tools()
    assert sorted(t.name for t in tools) == [
        "pdf_find", "pdf_open", "pdf_read_page", "pdf_read_tables",
    ]


def test_pdf_read_tables_without_pdfplumber_returns_structured_error(monkeypatch) -> None:
    # Force the "not installed" branch even if pdfplumber happens to be present.
    monkeypatch.setattr(pdfr, "_have_pdfplumber", lambda: False)
    r = pdfr.pdf_read_tables(__file__, 1)
    assert r["ok"] is False
    assert "pdfplumber not installed" in r["error"]
    assert "pip install" in r["hint"]


def test_pdf_read_tables_missing_file_returns_error() -> None:
    r = pdfr.pdf_read_tables("/no/such/path.pdf", 1)
    assert r["ok"] is False
    assert "no such file" in r["error"]


def test_missing_file_returns_error() -> None:
    r = pdfr.pdf_open("/no/such/path.pdf")
    assert r["ok"] is False
    assert "no such file" in r["error"]
