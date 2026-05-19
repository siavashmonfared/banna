"""xlsx_reader tests.

Build a tiny in-memory workbook via openpyxl, write it to tmp_path,
and exercise list_sheets / describe / read_range / find.
"""
from __future__ import annotations

from pathlib import Path

import pytest

openpyxl = pytest.importorskip("openpyxl")

from banna_agent.tools.xlsx_reader import (
    make_xlsx_tools,
    xlsx_describe,
    xlsx_find,
    xlsx_list_sheets,
    xlsx_read_range,
)


@pytest.fixture
def workbook(tmp_path: Path) -> Path:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sales"
    ws.append(["Region", "Quarter", "Revenue"])
    ws.append(["North", "Q1", 100])
    ws.append(["North", "Q2", 150])
    ws.append(["South", "Q1", 80])
    ws.append(["South", "Q2", 120])
    ws2 = wb.create_sheet("Notes")
    ws2.append(["Comment"])
    ws2.append(["southern surge in Q2"])
    path = tmp_path / "wb.xlsx"
    wb.save(path)
    return path


def test_list_sheets(workbook: Path) -> None:
    r = xlsx_list_sheets(workbook)
    assert r["ok"] is True
    names = sorted(s["name"] for s in r["sheets"])
    assert names == ["Notes", "Sales"]


def test_describe_returns_header_row(workbook: Path) -> None:
    r = xlsx_describe(workbook, sheet="Sales")
    assert r["ok"] is True
    assert r["header_row"] == ["Region", "Quarter", "Revenue"]
    assert r["max_row"] == 5
    assert r["max_col"] == 3


def test_read_range_returns_rows_and_markdown(workbook: Path) -> None:
    r = xlsx_read_range(workbook, "Sales", "A1:C3")
    assert r["ok"] is True
    assert r["n_rows"] == 3
    assert r["n_cols"] == 3
    assert r["rows"][0] == ["Region", "Quarter", "Revenue"]
    assert r["rows"][1][2] == 100
    assert "Revenue" in r["markdown"]


def test_read_range_rejects_malformed_range(workbook: Path) -> None:
    r = xlsx_read_range(workbook, "Sales", "not-a-range")
    assert r["ok"] is False
    assert "A1:D20" in r["error"]


def test_read_range_enforces_max_cells(workbook: Path) -> None:
    r = xlsx_read_range(workbook, "Sales", "A1:C3", max_cells=2)
    assert r["ok"] is False
    assert "max_cells" in r["error"]


def test_find_finds_substring_match(workbook: Path) -> None:
    r = xlsx_find(workbook, "south")
    assert r["ok"] is True
    # Two cells in Sales ("South", "South") + one in Notes ("southern surge…")
    assert r["n_hits"] >= 3
    sheets = {h["sheet"] for h in r["hits"]}
    assert "Sales" in sheets
    assert "Notes" in sheets


def test_find_can_restrict_to_one_sheet(workbook: Path) -> None:
    r = xlsx_find(workbook, "south", sheet="Notes")
    assert r["ok"] is True
    assert r["n_hits"] == 1
    assert r["hits"][0]["sheet"] == "Notes"


def test_describe_rejects_unknown_sheet(workbook: Path) -> None:
    r = xlsx_describe(workbook, sheet="DoesNotExist")
    assert r["ok"] is False
    assert "no sheet named" in r["error"]


def test_make_xlsx_tools_registers_four_handlers() -> None:
    tools = make_xlsx_tools()
    names = sorted(t.name for t in tools)
    assert names == ["xlsx_describe", "xlsx_find", "xlsx_list_sheets", "xlsx_read_range"]
