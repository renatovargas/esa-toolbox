"""Inputs of the carbon accounts that come from rasters: land type shares, productivity and regional weights.

The 1 km shares are float32 in the work folder. Each is a whole number of 100 m cells over 100.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from esa_toolbox import nuts, rio
from esa_toolbox.core import Context
from esa_toolbox.grid import FOREST_GRID
from esa_toolbox.landcover import shares_1km
from esa_toolbox.pool import run as pool_run

GRID = FOREST_GRID
LAND_TYPES = {
    "forest": [311, 312, 313, 324],
    "cropland": [211, 212, 213, 221, 222, 223, 241, 242, 243, 244],
    "grassland": [231, 321],
    "wetlands": [411, 412],
    "settlements": [111, 112, 121, 122, 123, 124, 131, 132, 133, 141, 142],
    "other_land": [322, 323, 331, 332, 333, 334, 335, 421, 422, 423],
}
GDMP_PRODUCT = {2000: "1km", 2006: "1km", 2012: "1km", 2018: "300m", 2024: "300m"}

# Social cost of carbon after Tol (2023), EUR 343.6 per tonne of carbon in 2023 euros.
SCC_EUR_PER_T_C_2023 = 343.6
PRICE_YEAR = 2023


def gdmp_annual(store: Path, product: str, year: int) -> Path:
    return store / "copernicus" / "gdmp_annual" / f"gdmp_{product}_{year}_4326.tif"


def _shares_job(store: str, year: int, out: str) -> str:
    s = shares_1km(Path(store), year, GRID, LAND_TYPES)
    rio.write(out, s, GRID, names=list(LAND_TYPES))
    return out


def _gdmp_job(src: str, product: str, out: str) -> str:
    method = "average" if product == "300m" else "bilinear"
    rio.write(out, rio.warp(src, GRID, method), GRID)
    return out


def shares_file(ctx: Context, year: int) -> Path:
    return ctx.path("prep", f"land_types_1km_{year}.tif")


def gdmp_file(ctx: Context, year: int) -> Path:
    return ctx.path("prep", f"gdmp_1km_{year}.tif")


def forest_share(ctx: Context, year: int) -> np.ndarray:
    return rio.read(shares_file(ctx, year), GRID, band=1, as_float=True)


def forest_weight(ctx: Context, year: int) -> np.ndarray:
    """Forest share times annual productivity, in double precision as terra computes in memory."""
    return forest_share(ctx, year).astype("float64") * rio.read(gdmp_file(ctx, year), GRID, as_float=True)


def prepare(ctx: Context) -> None:
    store = ctx.store
    jobs = []
    for y in ctx.years:
        if not shares_file(ctx, y).exists():
            jobs.append((_shares_job, (str(store), y, str(shares_file(ctx, y)))))
        if not gdmp_file(ctx, y).exists():
            src = ctx.need("copernicus", "gdmp_annual", f"gdmp_{GDMP_PRODUCT[y]}_{y}_4326.tif")
            jobs.append((_gdmp_job, (str(src), GDMP_PRODUCT[y], str(gdmp_file(ctx, y)))))
    comparison = ctx.path("prep", "gdmp_2018_1km_from_1km_product.tif")
    src_1km_2018 = gdmp_annual(store, "1km", 2018)
    if 2018 in ctx.years and not comparison.exists() and src_1km_2018.exists():
        jobs.append((_gdmp_job, (str(src_1km_2018), "1km", str(comparison))))
    ctx.log(f"  {len(jobs)} raster jobs on {ctx.settings.workers} workers")
    pool_run(_call, jobs, min(ctx.settings.workers, 6), on_done=lambda d, n: ctx.tick("rasters", d, n))

    regs = nuts.regions(store)
    # Hectares of each land type by region. A 1 km cell holds 100 hectares.
    area = []
    for y in ctx.years:
        sh = rio.read(shares_file(ctx, y), GRID, band=list(range(1, 7)), as_float=True)
        for k, lt in enumerate(LAND_TYPES):
            cells = nuts.sum_by_region(sh[k], GRID, regs)
            area.append(regs.table.select("level", "nuts_id", "name", "geo").with_columns(
                epoch=pl.lit(y, pl.Int64), land_type=pl.lit(lt), hectares=pl.Series(cells * 100)))
    area = pl.concat(area)
    ctx.save(area, "clc_area_by_nuts")

    # Forest weight of each region, the forest share times productivity, summed over the region.
    fw = []
    for y in ctx.years:
        sums = nuts.sum_by_region(forest_weight(ctx, y), GRID, regs)
        fw.append(regs.table.select("level", "nuts_id", "name", "geo").with_columns(
            epoch=pl.lit(y, pl.Int64), land_type=pl.lit("forest"), hectares=pl.Series(sums)))
    fw = pl.concat(fw)

    total_area = area.group_by("epoch", "level", "nuts_id", "geo").agg(total_ha=pl.col("hectares").sum())
    tw = (pl.concat([area.filter(pl.col("land_type") != "forest"), fw])
          .rename({"hectares": "weight_sum"})
          .join(total_area, on=["epoch", "level", "nuts_id", "geo"], how="left"))
    grp = ["epoch", "level", "geo", "land_type"]
    wsum = pl.col("weight_sum").sum().over(grp)
    asum = pl.col("total_ha").sum().over(grp)
    tw = tw.with_columns(
        fallback=(pl.col("level") != "NUTS0") & (wsum <= 0),
        share=pl.when(pl.col("level") == "NUTS0").then(1.0)
        .when(wsum > 0).then(pl.col("weight_sum") / wsum)
        .when(asum > 0).then(pl.col("total_ha") / asum)
        .otherwise(None),
    ).drop("total_ha").sort("epoch", "level", "nuts_id", "land_type")
    ctx.save(tw, "type_weights")

    # Carbon price, a real value held constant in 2023 euros and moved to the euros of each year by HICP.
    hicp = pl.read_parquet(ctx.need("eurostat", "shared", "hicp_eu27.parquet"))
    h23 = hicp.filter(pl.col("year") == PRICE_YEAR)["hicp_2015"][0]
    scc = SCC_EUR_PER_T_C_2023 * 12 / 44
    price = hicp.filter(pl.col("year").is_between(2000, 2024)).select(
        pl.col("year").cast(pl.Int64), pl.lit(scc).alias("price_eur_per_t_co2_2023"),
        (scc * pl.col("hicp_2015") / h23).alias("price_eur_per_t_co2"))
    ctx.save(price, "carbon_price")

    # The two productivity products of 2018 over cells that are mostly forest.
    if comparison.exists():
        on_forest = forest_share(ctx, 2018) > 0.5
        p300 = rio.read(gdmp_file(ctx, 2018), GRID, as_float=True)[on_forest]
        p1k = rio.read(comparison, GRID, as_float=True)[on_forest]
        ok = ~np.isnan(p300) & ~np.isnan(p1k)
        p300, p1k = p300[ok].astype("float64"), p1k[ok].astype("float64")
        from scipy.stats import spearmanr

        ctx.table(pl.DataFrame({
            "forest_cells": [int(ok.sum())], "pearson": [float(np.corrcoef(p300, p1k)[0, 1])],
            "spearman": [float(spearmanr(p300, p1k).statistic)], "mean_300m": [p300.mean()],
            "mean_1km": [p1k.mean()], "ratio_of_means": [p300.mean() / p1k.mean()]}), "gdmp_comparison", 3)
        ctx.facts(gdmp_forest_cells_m=(ok.sum() / 1e6, 1), gdmp_pearson=(float(np.corrcoef(p300, p1k)[0, 1]), 3),
                  gdmp_ratio_pct=(100 * p300.mean() / p1k.mean(), 1))


def _call(func, args):
    return func(*args)
