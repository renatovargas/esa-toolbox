"""Inputs of the wood accounts: the spatial key at 100 m and 1 km, regional shares, and the net annual
increment of the carbon budget model.

The pass that adds the 100 m key up by country
and by 1 km cell computes it from Corine and productivity, and keeps the resampled productivity in the shared
cache of esa_toolbox.forestmaps, which the 100 m maps stage reads back. The country sums of this pass are
kept, so the maps stage needs no second pass of sums.
"""

from __future__ import annotations

import sys

import numpy as np
import openpyxl
import polars as pl

from esa_toolbox import forestmaps, nuts, rio
from esa_toolbox.core import Context, EsaError, work_key
from esa_toolbox.landcover import clc_file
from esa_toolbox.grid import FOREST_GRID

GRID = FOREST_GRID
FOREST_CODES = [311, 312, 313]  # INCA Part II leaves out transitional woodland and shrub (324)
GDMP_PRODUCT = {2000: "1km", 2006: "1km", 2012: "1km", 2018: "300m", 2024: "300m"}
CBM_SHEETS = ["BE", "BG", "CZ", "DE", "EE", "IE", "EL", "ES", "FR", "HR", "DK", "IT", "CY", "LV", "LT", "LU",
              "HU", "NL", "AT", "PL", "PT", "RO", "SI", "SK", "FI", "SE"]


def weight_1km_file(ctx: Context, year: int):
    return ctx.path("prep", f"clc_gdmp_weight_1km_{year}.tif")


def sums_100m_file(ctx: Context, year: int):
    return ctx.path("prep", f"weight_100m_by_country_{year}.parquet")


def prepare(ctx: Context) -> None:
    regs = nuts.countries(ctx.store)
    geos = regs.table["geo"].to_list()
    mask = []
    for y in ctx.years:
        gdmp = ctx.need("copernicus", "gdmp_annual", f"gdmp_{GDMP_PRODUCT[y]}_{y}_4326.tif")
        cells = ctx.path("prep", f"forest_cells_{y}.parquet")
        key = work_key(sys.modules[__name__], forestmaps, rio, y, gdmp, clc_file(ctx.store, y),
                       ctx.store.joinpath(*nuts.NUTS_FILE))
        if ctx.year_done("prepare", y, key, weight_1km_file(ctx, y), sums_100m_file(ctx, y), cells):
            mask.append(pl.read_parquet(cells))
            continue
        ctx.log(f"  {y}: spatial key at 100 m")
        sums, km, counts = forestmaps.country_sums(
            ctx.store, y, GRID.with_res(100), FOREST_CODES, gdmp, ctx.settings.workers, want_1km=True,
            grid_1km=GRID, count_codes=[311, 312, 313, 324], progress=lambda d, n: ctx.tick(f"{y} key", d, n),
            cache=forestmaps.productivity_cache(ctx.settings.work, ctx.store, y, gdmp))
        rio.write(weight_1km_file(ctx, y), km, GRID, names=[f"clc_gdmp_weight_{y}"])
        pl.DataFrame({"geo": geos, "weight_sum": sums}).write_parquet(sums_100m_file(ctx, y))
        m = pl.DataFrame({"year": [y], "forest_ha": [sum(counts[c] for c in FOREST_CODES)],
                          "transitional_ha": [counts[324]]})
        m.write_parquet(cells)
        ctx.mark_year_done("prepare", y, key)
        mask.append(m)
    fm = pl.concat(mask)
    r18 = fm.filter(pl.col("year") == 2018)
    if r18.height:
        ctx.facts(transitional_share_2018=(100 * r18["transitional_ha"][0] / r18["forest_ha"][0], 1))
    ctx.table(fm.with_columns(pl.col("forest_ha", "transitional_ha") / 1e6).rename({"year": "Year", "forest_ha": "Forest mask, 311, 312 and 313 (Mha)",
                                      "transitional_ha": "Transitional woodland and shrub, 324 (Mha)"}),
              "forest_mask", 1)

    # Share of each region in the weight of its country. Countries take the whole national value.
    regs_all = nuts.regions(ctx.store)
    rows = []
    for y in ctx.years:
        w = rio.read(weight_1km_file(ctx, y), GRID, as_float=True)
        sums = nuts.sum_by_region(w, GRID, regs_all)
        rows.append(regs_all.table.select("level", "nuts_id", "name", "geo")
                    .with_columns(year=pl.lit(y, pl.Int64), weight_sum=pl.Series(np.nan_to_num(sums))))
    nw = pl.concat(rows)
    grp = ["year", "level", "geo"]
    tot = pl.col("weight_sum").sum().over(grp)
    nw = nw.with_columns(share=pl.when(pl.col("level") == "NUTS0").then(1.0)
                         .when(tot > 0).then(pl.col("weight_sum") / tot).otherwise(None))
    ctx.save(nw, "weight_by_nuts")
    first = nw.filter(pl.col("year") == ctx.years[0])
    ctx.facts(**{f"regions_{lv.lower()}": str(first.filter(pl.col("level") == lv).height)
                 for lv in ("NUTS0", "NUTS2", "NUTS3")})

    ctx.save(read_cbm(ctx), "cbm_nai")


def read_cbm(ctx: Context) -> pl.DataFrame:
    """Total forest increment of the carbon budget model by country and year, 2000 to 2019."""
    path = ctx.need("jrc", "EFA_FRA_CBM_Data_analysis_20210121.xlsx")
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out = []
    for sheet in CBM_SHEETS:
        rows = list(wb[sheet].iter_rows(min_row=1, max_row=27, values_only=True))
        header = [str(v).strip() if v is not None else "NA" for v in rows[2]]
        tot = header.index("Tot NAI")
        share = next(i for i, h in enumerate(header) if "FAWS/Tot" in h)
        for r in rows:
            try:
                year = int(float(r[0]))
            except (TypeError, ValueError):
                continue
            if 2000 <= year <= 2019:
                out.append({"geo": sheet, "year": year, "cbm_total_forest": float(r[tot]),
                            "share_available": float(r[share]) if r[share] is not None else None})
    df = pl.DataFrame(out)
    if df.height != 26 * 20 or df["cbm_total_forest"].min() <= 0:
        raise EsaError(f"The carbon budget workbook {path} does not have the expected 26 countries by 20 years")
    return df
