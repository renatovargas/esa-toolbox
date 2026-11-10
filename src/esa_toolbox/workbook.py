"""The results workbook: one sheet per table and NUTS level, with a banner, a nested header by year and
metric, one row per region and a total row. The layout is the one of the earlier workbooks."""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import xlsxwriter

NUMBER_FORMAT = "#,##0.0"


class Workbook:
    def __init__(self, path: Path, heading: str, total_code: str = "Total",
                 total_name: str = "Total of the regions") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = self.path.with_name(self.path.name + ".part.xlsx")
        self.wb = xlsxwriter.Workbook(self.tmp, {"nan_inf_to_errors": True})
        self.heading = heading
        self.total = (total_code, total_name)
        wb = self.wb
        self.f_banner = wb.add_format({"text_wrap": True, "valign": "vcenter"})
        self.f_head = wb.add_format({"bold": True, "text_wrap": True, "align": "center", "valign": "vcenter"})
        self.f_head2 = wb.add_format({"bold": True, "text_wrap": True, "align": "center",
                                      "valign": "vcenter", "bottom": 1})
        self.f_num = wb.add_format({"num_format": NUMBER_FORMAT})
        self.f_tot_txt = wb.add_format({"bold": True, "top": 1, "bottom": 1})
        self.f_tot_num = wb.add_format({"bold": True, "top": 1, "bottom": 1, "num_format": NUMBER_FORMAT})
        self.f_wrap_top = wb.add_format({"text_wrap": True, "valign": "top"})

    def readme(self, lines: list[str]) -> None:
        ws = self.wb.add_worksheet("Readme")
        ws.set_column(0, 0, 110)
        for i, line in enumerate(lines):
            ws.write_string(i, 0, line, self.f_wrap_top)

    def sheet(self, name: str, title: str, note: str, data: pl.DataFrame, metrics: dict[str, str]) -> None:
        """`data` is long, with nuts_id, name, year and the metric columns."""
        years = sorted(data["year"].unique().to_list())
        wide = None
        for y in years:
            part = data.filter(pl.col("year") == y).select(
                "nuts_id", "name", *[pl.col(m).alias(f"{y}__{m}") for m in metrics])
            wide = part if wide is None else wide.join(part, on=["nuts_id", "name"], how="full",
                                                       coalesce=True)
        wide = wide.sort("nuts_id")
        value_cols = [c for c in wide.columns if "__" in c]
        totals = [wide[c].sum() for c in value_cols]  # sums ignore missing values, as na.rm = TRUE

        ws = self.wb.add_worksheet(name)
        n_cols = 2 + len(value_cols)
        n_met = len(metrics)
        last = min(n_cols, 10) - 1
        for i, text in enumerate([self.heading, title, note]):
            if last > 0:
                ws.merge_range(i, 0, i, last, text, self.f_banner)
            else:
                ws.write_string(i, 0, text, self.f_banner)
        for r, h in zip(range(3), (18, 18, 32)):
            ws.set_row(r, h)

        hr = 4  # header rows 5 and 6 of the sheet
        ws.merge_range(hr, 0, hr + 1, 0, "Code", self.f_head)
        ws.merge_range(hr, 1, hr + 1, 1, "Region", self.f_head)
        for k, y in enumerate(years):
            first = 2 + k * n_met
            if n_met > 1:
                ws.merge_range(hr, first, hr, first + n_met - 1, str(y), self.f_head)
            else:
                ws.write_string(hr, first, str(y), self.f_head)
            for j, label in enumerate(metrics.values()):
                ws.write_string(hr + 1, first + j, label, self.f_head2)
        ws.set_row(hr, 20)
        ws.set_row(hr + 1, 34)

        start = 6
        for i, row in enumerate(wide.select("nuts_id", "name", *value_cols).iter_rows()):
            ws.write_string(start + i, 0, row[0])
            ws.write_string(start + i, 1, row[1] or "")
            for j, v in enumerate(row[2:]):
                if v is not None and v == v:
                    ws.write_number(start + i, 2 + j, v, self.f_num)
        t = start + wide.height
        ws.write_string(t, 0, self.total[0], self.f_tot_txt)
        ws.write_string(t, 1, self.total[1], self.f_tot_txt)
        for j, v in enumerate(totals):
            ws.write_number(t, 2 + j, v or 0.0, self.f_tot_num)
        ws.set_column(0, 0, 10)
        ws.set_column(1, 1, 30)
        if n_cols > 2:
            ws.set_column(2, n_cols - 1, 15)
        ws.freeze_panes(start, 2)

    def close(self) -> Path:
        self.wb.close()
        os.replace(self.tmp, self.path)
        return self.path
