"""Workbook layout, command line and packaging."""
import filecmp
from pathlib import Path

import openpyxl
import polars as pl
from typer.testing import CliRunner

from esa_toolbox.cli import app
from esa_toolbox.workbook import Workbook

ROOT = Path(__file__).parents[1]


def test_workbook_sheet(tmp_path):
    data = pl.DataFrame({"nuts_id": ["B", "A", "A", "B"], "name": ["Bee", "Ay", "Ay", "Bee"],
                         "year": [2000, 2000, 2024, 2024], "x": [1.0, 2.0, 3.0, None], "y": [1.0, 1, 1, 1]})
    wb = Workbook(tmp_path / "w.xlsx", "Heading", "EU27", "Total")
    wb.sheet("S", "Title", "Note", data, {"x": "X", "y": "Y"})
    wb.close()
    ws = openpyxl.load_workbook(tmp_path / "w.xlsx")["S"]
    assert ws["A1"].value == "Heading" and ws["C5"].value == "2000" and ws["D6"].value == "Y"
    assert [ws.cell(r, 1).value for r in (7, 8, 9)] == ["A", "B", "EU27"]
    assert ws["C9"].value == 3 and ws["E9"].value == 3  # totals ignore the missing value
    assert ws["E8"].value is None
    assert "C5:D5" in [str(m) for m in ws.merged_cells.ranges]


def test_cli_version_and_unknown_service(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = CliRunner().invoke(app, ["--version"])
    assert r.exit_code == 0
    r = CliRunner().invoke(app, ["run", "fishes"])
    assert r.exit_code == 3
    r = CliRunner().invoke(app, ["status", "carbon", "--json"])
    assert r.exit_code == 0 and "not run" in r.output


def test_run_without_inputs_says_what_is_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = CliRunner().invoke(app, ["run", "carbon", "--no-report", "--workers", "1"])
    assert r.exit_code == 3 and "esa fetch" in r.output


def test_site_extension_is_the_package_copy():
    a, b = ROOT / "_extensions" / "facts", ROOT / "src" / "esa_toolbox" / "reports" / "_extensions" / "facts"
    assert not filecmp.dircmp(a, b).diff_files
