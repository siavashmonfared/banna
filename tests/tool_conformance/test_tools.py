"""Per-tool conformance tests.

Phase 4 deliverable: assert each registered tool actually does what its
schema/description claims, on small but realistic inputs.

Conventions:
  - Tests are grouped by tool, named test_<tool_name>_<case>.
  - Each test calls invoke_tool() and asserts ok + content presence.
  - Network-dependent tools (search, read_url, browser_*) are skipped
    when prerequisites are missing rather than failing.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from banna_agent.tools.base import invoke_tool

from .conftest import SENTINELS


# ===========================================================================
# read_file
# ===========================================================================

def _read_file(path: Path, **kwargs: Any) -> dict[str, Any]:
    from banna_agent.tools.file_reader import make_file_reader_tool
    inv = invoke_tool(make_file_reader_tool(), {"path": str(path), **kwargs})
    assert inv.ok, f"read_file failed: {inv.error}"
    assert isinstance(inv.result, dict)
    return inv.result


def test_read_file_txt(fixture_paths: dict[str, Path]) -> None:
    res = _read_file(fixture_paths["txt"])
    body = str(res.get("text", "")) + str(res.get("content", ""))
    assert SENTINELS["txt"] in body


def test_read_file_csv(fixture_paths: dict[str, Path]) -> None:
    res = _read_file(fixture_paths["csv"])
    body = " ".join(str(v) for v in res.values())
    assert SENTINELS["csv"] in body


def test_read_file_xlsx(fixture_paths: dict[str, Path]) -> None:
    res = _read_file(fixture_paths["xlsx"])
    body = " ".join(str(v) for v in res.values())
    assert SENTINELS["xlsx"] in body


def test_read_file_docx(fixture_paths: dict[str, Path]) -> None:
    """The bug we observed on GAIA cffe0e32: docx parsed as raw zip bytes.
    This test fails if _read_docx is missing or broken."""
    res = _read_file(fixture_paths["docx"])
    body = " ".join(str(v) for v in res.values())
    assert SENTINELS["docx"] in body
    # Hard fail mode: if the tool returned raw zip bytes ('PK\x03\x04')
    # in any field, that's the cffe0e32 failure.
    assert "PK\x03\x04" not in body, "read_file returned raw zip bytes for .docx"


def test_read_file_pdf(fixture_paths: dict[str, Path]) -> None:
    res = _read_file(fixture_paths["pdf"])
    body = " ".join(str(v) for v in res.values())
    assert SENTINELS["pdf"] in body


def test_read_file_missing_path(tmp_path: Path) -> None:
    """Should error cleanly, not silently succeed."""
    from banna_agent.tools.file_reader import make_file_reader_tool
    inv = invoke_tool(make_file_reader_tool(),
                      {"path": str(tmp_path / "does_not_exist.txt")})
    assert inv.ok is False
    assert "FileNotFound" in (inv.error or "") or "no such file" in (inv.error or "").lower()


def test_read_file_docx_without_extension(
    fixture_paths: dict[str, Path], tmp_path: Path,
) -> None:
    """The cffe0e32 GAIA failure: HF-downloaded attachments may lack a
    file extension. Without magic-byte sniffing, read_file falls through
    to _read_text and surfaces raw ZIP bytes. With sniffing, it should
    recognize the file as a docx and extract its text."""
    import shutil
    target = tmp_path / "attachment_with_no_ext"
    shutil.copy(fixture_paths["docx"], target)
    res = _read_file(target)
    body = " ".join(str(v) for v in res.values())
    assert SENTINELS["docx"] in body, (
        "extension-less docx fell through to text reader; sniffing not wired"
    )
    assert "PK\x03\x04" not in body


def test_read_file_pdf_without_extension(
    fixture_paths: dict[str, Path], tmp_path: Path,
) -> None:
    import shutil
    target = tmp_path / "attachment_no_ext_pdf"
    shutil.copy(fixture_paths["pdf"], target)
    res = _read_file(target)
    body = " ".join(str(v) for v in res.values())
    assert SENTINELS["pdf"] in body


def test_read_file_xlsx_without_extension(
    fixture_paths: dict[str, Path], tmp_path: Path,
) -> None:
    import shutil
    target = tmp_path / "attachment_no_ext_xlsx"
    shutil.copy(fixture_paths["xlsx"], target)
    res = _read_file(target)
    body = " ".join(str(v) for v in res.values())
    assert SENTINELS["xlsx"] in body


# ===========================================================================
# calculator
# ===========================================================================

def _call(factory, args: dict) -> Any:
    return invoke_tool(factory(), args)


def test_calculator_basic_arithmetic() -> None:
    from banna_agent.tools.calculator import make_calculator_tool
    inv = _call(make_calculator_tool, {"expression": "2 + 3 * 4"})
    assert inv.ok
    val = inv.result.get("value") if isinstance(inv.result, dict) else inv.result
    assert float(val) == 14


def test_calculator_division() -> None:
    from banna_agent.tools.calculator import make_calculator_tool
    inv = _call(make_calculator_tool, {"expression": "17054.888 / 1000"})
    assert inv.ok
    val = inv.result.get("value") if isinstance(inv.result, dict) else inv.result
    assert abs(float(val) - 17.054888) < 1e-6


def test_calculator_rejects_arbitrary_code() -> None:
    """A real calculator must not eval('import os'). If it does, we have a
    security problem masquerading as a math tool."""
    from banna_agent.tools.calculator import make_calculator_tool
    inv = _call(make_calculator_tool, {"expression": "__import__('os').getcwd()"})
    assert inv.ok is False, "calculator accepted code injection"


# ===========================================================================
# python_sandbox / run_python
# ===========================================================================

def test_run_python_basic_execution() -> None:
    from banna_agent.tools.python_sandbox import make_python_sandbox_tool
    inv = _call(make_python_sandbox_tool, {"code": "print(2 + 2)"})
    assert inv.ok
    assert "4" in (inv.result.get("stdout") or "")


def test_run_python_captures_stderr() -> None:
    from banna_agent.tools.python_sandbox import make_python_sandbox_tool
    inv = _call(make_python_sandbox_tool, {"code": "raise ValueError('boom')"})
    assert inv.ok  # subprocess error != tool error; ok=True with returncode!=0
    assert inv.result.get("returncode") != 0
    assert "ValueError" in (inv.result.get("stderr") or "")


def test_run_python_failure_surfaces_error_field() -> None:
    """The cffe0e32 failure: a script that crashed (e.g. ImportError)
    came back with empty stdout and stderr buried in the dict — the
    model glossed over it and fabricated an answer. The tool must now
    surface a clear top-level `error` summary when the script failed."""
    from banna_agent.tools.python_sandbox import make_python_sandbox_tool
    inv = _call(make_python_sandbox_tool,
                {"code": "import nonexistent_module_xyz123"})
    assert inv.ok
    payload = inv.result
    assert payload.get("ok") is False
    assert payload.get("error"), "missing top-level error summary on failed script"
    assert "code" in payload["error"].lower() or "exit" in payload["error"].lower()


def test_run_python_success_omits_error_field() -> None:
    from banna_agent.tools.python_sandbox import make_python_sandbox_tool
    inv = _call(make_python_sandbox_tool, {"code": "print('ok')"})
    assert inv.ok
    payload = inv.result
    assert payload.get("ok") is True
    assert "error" not in payload


def test_run_python_timeout_enforced() -> None:
    from banna_agent.tools.python_sandbox import make_python_sandbox_tool
    inv = _call(
        make_python_sandbox_tool,
        {"code": "import time; time.sleep(60)", "timeout_s": 1.0},
    )
    assert inv.ok
    assert inv.result.get("timeout") is True


def test_run_python_can_parse_docx(fixture_paths: dict[str, Path]) -> None:
    """End-to-end: if read_file failed for .docx, the model's fallback
    plan was 'use python to extract'. This test verifies that fallback
    actually works in our sandbox."""
    from banna_agent.tools.python_sandbox import make_python_sandbox_tool
    code = (
        f"import zipfile\n"
        f"with zipfile.ZipFile({str(fixture_paths['docx'])!r}) as z:\n"
        f"    print(z.read('word/document.xml').decode())\n"
    )
    inv = _call(make_python_sandbox_tool, {"code": code})
    assert inv.ok
    assert SENTINELS["docx"] in (inv.result.get("stdout") or "")


# ===========================================================================
# list_files
# ===========================================================================

def test_list_files_returns_filenames(fixtures_dir: Path) -> None:
    from banna_agent.tools.list_files import make_list_files_tool
    inv = invoke_tool(make_list_files_tool(), {"root": str(fixtures_dir)})
    assert inv.ok
    payload = inv.result
    flat = str(payload)
    for name in ("sample.txt", "sample.pdf", "sample.docx", "sample.xlsx"):
        assert name in flat, f"{name} missing from list_files output"


# ===========================================================================
# grep_text
# ===========================================================================

def test_grep_text_finds_sentinel(fixtures_dir: Path) -> None:
    from banna_agent.tools.grep import make_grep_tool
    inv = invoke_tool(
        make_grep_tool(),
        {"pattern": SENTINELS["txt"], "root": str(fixtures_dir)},
    )
    assert inv.ok
    flat = str(inv.result)
    assert "sample.txt" in flat


def test_grep_text_misses_absent_pattern(fixtures_dir: Path) -> None:
    from banna_agent.tools.grep import make_grep_tool
    inv = invoke_tool(
        make_grep_tool(),
        {"pattern": "THIS_PATTERN_DOES_NOT_EXIST_ANYWHERE", "root": str(fixtures_dir)},
    )
    assert inv.ok
    res = inv.result if isinstance(inv.result, dict) else {}
    hits = res.get("hits") or res.get("matches") or []
    assert len(hits) == 0


# ===========================================================================
# pdf_open / pdf_read_page / pdf_find
# ===========================================================================

def _pdf_tools():
    from banna_agent.tools.pdf_reader import make_pdf_tools
    tools = {t.name: t for t in make_pdf_tools()}
    return tools


def test_pdf_open_local_file(fixture_paths: dict[str, Path]) -> None:
    tools = _pdf_tools()
    inv = invoke_tool(tools["pdf_open"], {"path": str(fixture_paths["pdf"])})
    assert inv.ok
    assert isinstance(inv.result, dict)
    # Expect at least n_pages or pages metadata.
    payload = inv.result
    flat = str(payload)
    assert "1" in flat or payload.get("n_pages") == 1 or payload.get("pages") == 1


def test_pdf_read_page_extracts_text(fixture_paths: dict[str, Path]) -> None:
    tools = _pdf_tools()
    invoke_tool(tools["pdf_open"], {"path": str(fixture_paths["pdf"])})
    inv = invoke_tool(tools["pdf_read_page"],
                      {"path": str(fixture_paths["pdf"]), "page": 1})
    assert inv.ok
    flat = str(inv.result)
    assert SENTINELS["pdf"] in flat


def test_pdf_find_locates_sentinel(fixture_paths: dict[str, Path]) -> None:
    tools = _pdf_tools()
    invoke_tool(tools["pdf_open"], {"path": str(fixture_paths["pdf"])})
    inv = invoke_tool(tools["pdf_find"],
                      {"path": str(fixture_paths["pdf"]),
                       "query": SENTINELS["pdf"]})
    assert inv.ok
    flat = str(inv.result)
    assert SENTINELS["pdf"] in flat or "1" in flat


def test_pdf_open_with_url_fails_loudly_or_handles_gracefully(
    fixture_paths: dict[str, Path],
) -> None:
    """Known bug from GAIA 5d0080cb: pdf_open(url) silently no-ops.
    Either it should fetch+parse the URL, OR it should error loudly so
    the model knows to download first. Silent success with zero pages
    is the bad case."""
    tools = _pdf_tools()
    inv = invoke_tool(
        tools["pdf_open"],
        {"path": "https://example.invalid/no-such-file.pdf"},
    )
    if inv.ok:
        payload = inv.result if isinstance(inv.result, dict) else {}
        # Must have populated content OR raised — empty success is the bug.
        assert (payload.get("n_pages") or payload.get("pages") or 0) > 0 or \
            payload.get("error") is not None, \
            "pdf_open accepted a URL and silently returned empty (5d0080cb bug)"


# ===========================================================================
# xlsx tools
# ===========================================================================

def _xlsx_tools():
    from banna_agent.tools.xlsx_reader import make_xlsx_tools
    return {t.name: t for t in make_xlsx_tools()}


def test_xlsx_list_sheets(fixture_paths: dict[str, Path]) -> None:
    tools = _xlsx_tools()
    inv = invoke_tool(tools["xlsx_list_sheets"],
                      {"path": str(fixture_paths["xlsx"])})
    assert inv.ok
    flat = str(inv.result)
    assert "Summary" in flat
    assert "Detail" in flat


def test_xlsx_describe(fixture_paths: dict[str, Path]) -> None:
    tools = _xlsx_tools()
    inv = invoke_tool(tools["xlsx_describe"],
                      {"path": str(fixture_paths["xlsx"]), "sheet": "Summary"})
    assert inv.ok


def test_xlsx_read_range_returns_cells(fixture_paths: dict[str, Path]) -> None:
    tools = _xlsx_tools()
    inv = invoke_tool(
        tools["xlsx_read_range"],
        {"path": str(fixture_paths["xlsx"]), "sheet": "Summary",
         "range": "A1:B3"},
    )
    assert inv.ok
    flat = str(inv.result)
    assert "alice" in flat or "name" in flat


def test_xlsx_find_locates_sentinel(fixture_paths: dict[str, Path]) -> None:
    tools = _xlsx_tools()
    inv = invoke_tool(
        tools["xlsx_find"],
        {"path": str(fixture_paths["xlsx"]), "value": SENTINELS["xlsx"]},
    )
    assert inv.ok
    flat = str(inv.result)
    assert SENTINELS["xlsx"] in flat


# ===========================================================================
# final_answer
# ===========================================================================

def test_final_answer_returns_answer_field() -> None:
    from banna_agent.tools.final_answer import make_final_answer_tool
    inv = invoke_tool(
        make_final_answer_tool(),
        {"answer": "42", "reasoning": "computed it"},
    )
    assert inv.ok
    flat = str(inv.result)
    assert "42" in flat


def test_final_answer_requires_answer_field() -> None:
    from banna_agent.tools.final_answer import make_final_answer_tool
    # The handler should error or accept-with-empty if 'answer' missing.
    inv = invoke_tool(make_final_answer_tool(), {"reasoning": "x"})
    # Either flavor is acceptable; what's NOT acceptable is silently
    # producing a non-empty answer the model never wrote.
    if inv.ok:
        payload = inv.result if isinstance(inv.result, dict) else {}
        assert not payload.get("answer"), "final_answer fabricated an answer"


# ===========================================================================
# plan
# ===========================================================================

def test_plan_tool_add_and_list() -> None:
    from banna_agent.tools.plan import make_plan_tool
    tool = make_plan_tool()
    inv = invoke_tool(tool, {"op": "add", "step": "search for X"})
    assert inv.ok
    inv2 = invoke_tool(tool, {"op": "list"})
    assert inv2.ok
    assert "search for X" in str(inv2.result)


# ===========================================================================
# search (network, skipped without API key)
# ===========================================================================

@pytest.mark.skipif(
    not (os.environ.get("BRAVE_API_KEY") or
         os.environ.get("TAVILY_API_KEY") or
         os.environ.get("SERPER_API_KEY")),
    reason="search backends need an API key",
)
def test_search_returns_hits() -> None:
    from banna_agent.tools.search import make_search_tool
    inv = invoke_tool(make_search_tool(),
                      {"query": "GAIA benchmark huggingface dataset"})
    assert inv.ok
    payload = inv.result if isinstance(inv.result, dict) else {}
    hits = payload.get("hits") or payload.get("results") or []
    assert len(hits) > 0


# ===========================================================================
# read_url (network)
# ===========================================================================

@pytest.mark.skipif(
    os.environ.get("SKIP_NETWORK_TESTS"),
    reason="network tests disabled via SKIP_NETWORK_TESTS",
)
def test_read_url_extracts_html_text() -> None:
    from banna_agent.tools.url_reader import make_url_reader_tool
    inv = invoke_tool(make_url_reader_tool(),
                      {"url": "https://example.com/"})
    if not inv.ok:
        pytest.skip(f"network unavailable: {inv.error}")
    payload = inv.result if isinstance(inv.result, dict) else {}
    text = (payload.get("text") or "") + (payload.get("content") or "")
    assert "Example" in text or "example" in text.lower()


# ===========================================================================
# browser_* (heavy, only run if a browser backend is configured)
# ===========================================================================

@pytest.mark.skipif(
    not os.environ.get("BANNA_BROWSER_BACKEND"),
    reason="browser tools require a configured backend",
)
def test_browser_open_returns_page_state() -> None:
    from banna_agent.tools.browser import make_browser_tools
    tools = {t.name: t for t in make_browser_tools()}
    inv = invoke_tool(tools["browser_open"], {"url": "https://example.com/"})
    assert inv.ok
