"""PDF page-image vision path: scanned-page OCR fallback + visual tool.

Tier-3 / scanned-doc fix from the GAIA L3 dig. When a page has no text
layer (scanned image) or the answer is in a figure, render the page and
read it with the vision model. Born-digital pages never pay this cost.
"""
from __future__ import annotations

import pytest

import banna_agent.tools.pdf_reader as pdfr


class _FakeReader:
    def __init__(self, texts): self._texts = texts
    @property
    def pages(self):
        return [type("P", (), {"extract_text": (lambda self, t=t: t)})() for t in self._texts]
    @property
    def metadata(self): return {}
    def close(self): pass


class _FakeVision:
    """Stands in for _ImageExtractor: records calls, returns canned text."""
    def __init__(self, answer="OCR TEXT FROM PAGE"):
        self.answer = answer
        self.calls = []
    def extract_bytes(self, img_bytes, media_type, question):
        self.calls.append((media_type, question))
        return {"ok": True, "answer": self.answer, "cached": False}


@pytest.fixture
def local_pdf(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.5\n")  # content irrelevant; _open_reader is stubbed
    return str(p)


def test_scanned_page_triggers_ocr_fallback(local_pdf, monkeypatch) -> None:
    monkeypatch.setattr(pdfr, "_open_reader", lambda p: _FakeReader([""]))  # empty text layer
    monkeypatch.setattr(pdfr, "_rasterize_pdf_page", lambda p, i, **k: (b"PNG", None))
    vis = _FakeVision()
    r = pdfr.pdf_read_page(local_pdf, 1, vision=vis)
    assert r["ok"] and r["source"] == "ocr"
    assert r["text"] == "OCR TEXT FROM PAGE"
    assert len(vis.calls) == 1  # OCR was invoked


def test_born_digital_page_skips_ocr(local_pdf, monkeypatch) -> None:
    monkeypatch.setattr(pdfr, "_open_reader",
                        lambda p: _FakeReader(["This page has a real, full text layer."]))
    vis = _FakeVision()
    r = pdfr.pdf_read_page(local_pdf, 1, vision=vis)
    assert r["ok"] and r["source"] == "text"
    assert vis.calls == [], "a page with text must NOT pay the vision cost"


def test_no_vision_means_no_ocr(local_pdf, monkeypatch) -> None:
    monkeypatch.setattr(pdfr, "_open_reader", lambda p: _FakeReader([""]))
    r = pdfr.pdf_read_page(local_pdf, 1)  # vision=None
    assert r["ok"] and r["source"] == "text" and r["text"] == ""


def test_pdf_read_page_visual(local_pdf, monkeypatch) -> None:
    monkeypatch.setattr(pdfr, "_rasterize_pdf_page", lambda p, i, **k: (b"PNG", None))
    vis = _FakeVision(answer="the curve peaks at 0.2")
    r = pdfr.pdf_read_page_visual(local_pdf, 3, "what value does the curve reach?", vision=vis)
    assert r["ok"] and r["answer"] == "the curve peaks at 0.2"
    assert r["source"] == "vision" and vis.calls[0][1].startswith("what value")


def test_visual_without_vision_errs(local_pdf) -> None:
    r = pdfr.pdf_read_page_visual(local_pdf, 1, "q", vision=None)
    assert not r["ok"] and "vision" in r["error"]


def test_make_pdf_tools_adds_visual_only_with_llm() -> None:
    assert len(pdfr.make_pdf_tools()) == 4
    with_llm = pdfr.make_pdf_tools(llm=object())
    names = {t.name for t in with_llm}
    assert len(with_llm) == 5 and "pdf_read_page_visual" in names
