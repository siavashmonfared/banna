"""Docx + zip support in file_reader.

The docx reader is stdlib-only (zipfile + ElementTree); the test
builds a minimal valid .docx by zipping a hand-written document.xml.
The zip reader test packs two text files and asserts both members are
listed and at least one is previewed.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

from banna_agent.tools.file_reader import read_file


def _make_docx(path: Path, paragraphs: list[str]) -> None:
    """Write a minimal but valid .docx containing the given paragraphs."""
    body = "\n".join(
        f'<w:p><w:r><w:t>{p}</w:t></w:r></w:p>' for p in paragraphs
    )
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f'<w:body>{body}</w:body>'
        '</w:document>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '</Types>'
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("word/document.xml", document_xml)


def test_read_docx_extracts_paragraphs(tmp_path: Path) -> None:
    p = tmp_path / "doc.docx"
    _make_docx(p, ["First paragraph.", "The answer is 42.", "Final note."])
    r = read_file(p)
    assert r["kind"] == "docx"
    assert r["n_paragraphs"] == 3
    assert "answer is 42" in r["text"]


def test_read_docx_handles_corrupt_file(tmp_path: Path) -> None:
    p = tmp_path / "not_really.docx"
    p.write_bytes(b"this is not a zip")
    r = read_file(p)
    assert r["kind"] == "binary"
    assert "not a readable docx" in r["note"]


def test_read_zip_lists_members_and_previews_text(tmp_path: Path) -> None:
    p = tmp_path / "a.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("notes.txt", "important note: 1847")
        zf.writestr("readme.md", "# header\nbody")
        zf.writestr("data.bin", b"\x00\x01\x02\x03")
    r = read_file(p)
    assert r["kind"] == "zip"
    names = {m["name"] for m in r["members"]}
    assert names == {"notes.txt", "readme.md", "data.bin"}
    preview_names = {pv["name"] for pv in r["previews"]}
    assert "notes.txt" in preview_names
    # Binary file not previewed.
    assert "data.bin" not in preview_names
    # Preview text actually carries content.
    notes_pv = next(pv for pv in r["previews"] if pv["name"] == "notes.txt")
    assert "1847" in notes_pv["text"]


def test_read_zip_handles_corrupt_file(tmp_path: Path) -> None:
    p = tmp_path / "broken.zip"
    p.write_bytes(b"not a zip")
    r = read_file(p)
    assert r["kind"] == "binary"
    assert "not a readable zip" in r["note"]
