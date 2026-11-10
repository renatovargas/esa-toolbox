"""Maps of wood provision at 1 km and 100 m, with their checks against the tables.

The 100 m maps are whole numbers, in cubic decimetres of wood and euro cents at 2010 prices.
"""

from __future__ import annotations

import sys

import numpy as np
import polars as pl

from esa_toolbox import accounts, figures, forestmaps, nuts, rio
from esa_toolbox.accounts import country_weight_sums, grid_extent, per_weight_factors, spread
from esa_toolbox.core import Context, work_key
from esa_toolbox.landcover import clc_file
from esa_toolbox.services.wood.inputs import FOREST_CODES, GDMP_PRODUCT, GRID, sums_100m_file, weight_1km_file


def maps_1km(ctx: Context) -> None:
    nai = ctx.load("wood_nai_faws_country_year")
    value = ctx.load("wood_value_country_year")
    regs = nuts.countries(ctx.store)
    geos = regs.table["geo"].to_list()
    ids = regs.ids(GRID)
    out = ctx.out / "maps" / "1km"
    rows, panels_n, panels_v, lost = [], {}, {}, []
    for y in ctx.years:
        w = rio.read(weight_1km_file(ctx, y), GRID, as_float=True).astype("float64")
        sums = country_weight_sums(w, ids, len(geos))
        ny = dict(zip(*nai.filter(pl.col("year") == y).select("geo", "nai_faws_1000m3")))
        vy = dict(zip(*value.filter(pl.col("year") == y).select("geo", "value_mio_eur_constant")))
        lost += [g for g, s in zip(geos, sums) if not s > 0 and ny.get(g, 0) > 0]
        fm = spread(w, ids, per_weight_factors(geos, sums, ny)).astype("float32")
        vm = spread(w, ids, per_weight_factors(geos, sums, vy)).astype("float32")
        rio.write(out / f"wood_nai_clc_gdmp_1km_{y}.tif", fm, GRID, names=[f"nai_clc_gdmp_{y}"])
        rio.write(out / f"wood_value_constant_clc_gdmp_1km_{y}.tif", vm, GRID, names=[f"value_constant_{y}"])
        fb = rio.read(out / f"wood_nai_clc_gdmp_1km_{y}.tif", GRID, as_float=True)
        vb = rio.read(out / f"wood_value_constant_clc_gdmp_1km_{y}.tif", GRID, as_float=True)
        rows.append({"Year": y, "NAI, map (million m3)": np.nansum(fb, dtype="float64") / 1000,
                     "NAI, tables (million m3)": sum(ny.values()) / 1000,
                     "Value, map (M EUR 2010)": np.nansum(vb, dtype="float64"),
                     "Value, tables (M EUR 2010)": sum(vy.values())})
        panels_n[str(y)] = rio.block_sum(fb, 10)
        panels_v[str(y)] = rio.block_sum(vb, 10)
    if lost:
        from esa_toolbox.core import CheckFailed

        raise CheckFailed(f"Countries with a national value and no forest weight: {', '.join(sorted(set(lost)))}")
    t = pl.DataFrame(rows)
    ctx.check(t["NAI, map (million m3)"], t["NAI, tables (million m3)"],
              "1 km maps of the flow against the national tables", tolerance=1e-4)
    ctx.check(t["Value, map (M EUR 2010)"], t["Value, tables (M EUR 2010)"],
              "1 km maps of the value against the national tables", tolerance=1e-4)
    ctx.save_checks("checks_maps_1km")
    ctx.table(t, "maps_1km_totals", 1)
    ext = grid_extent(GRID)
    figures.map_panels(panels_n, ext, ctx.figure_path("wood_map_nai"), outlines=regs.geoms,
                       legend="Thousand m3 per 10 km cell")
    figures.map_panels(panels_v, ext, ctx.figure_path("wood_map_value"), outlines=regs.geoms,
                       legend="Million euro (2010 prices) per 10 km cell")


QUANTITIES = {  # map name: (column of the national table, factor to the unit of the map)
    "flow_dm3": ("nai", 1e6),
    "sustainable_dm3": ("sustainable", 1e6),
    "overuse_dm3": ("overuse_volume", 1e6),
    "flow_eurocent2010": ("nai", None),
    "sustainable_eurocent2010": ("sustainable", None),
    "overuse_eurocent2010": ("overuse_volume", None),
}


def national_map_values(overuse: pl.DataFrame, value: pl.DataFrame, year: int) -> dict[str, dict[str, float]]:
    """National total of every 100 m map, in the unit of the map (script helpers_common.R)."""
    d = (overuse.filter(pl.col("year") == year)
         .join(value.select("geo", "year", "unit_value_eur_m3", "deflator"), on=["geo", "year"], how="left")
         .with_columns(sustainable=pl.min_horizontal("removals", "nai"),
                       cents=pl.col("unit_value_eur_m3") * 1000 * pl.col("deflator") * 100))
    out = {}
    for q, (col, scale) in QUANTITIES.items():
        v = d[col] * scale if scale else d[col] * d["cents"]
        out[q] = dict(zip(d["geo"], v))
    return out


def maps_100m(ctx: Context) -> None:
    overuse = ctx.load("wood_overuse_country_year")
    value = ctx.load("wood_value_country_year")
    regs = nuts.countries(ctx.store)
    geos = regs.table["geo"].to_list()
    g100 = GRID.with_res(100)
    rows = []
    for y in ctx.years:
        names = {q: f"wood_{q}_100m_{y}.tif" for q in QUANTITIES}
        done = ctx.work / "maps_100m" / f"sums_{y}.parquet"
        gdmp = ctx.need("copernicus", "gdmp_annual", f"gdmp_{GDMP_PRODUCT[y]}_{y}_4326.tif")
        sums = pl.read_parquet(sums_100m_file(ctx, y))["weight_sum"].to_numpy()
        nat = national_map_values(overuse, value, y)
        # The maps of a year are rewritten whenever the code, the inputs, the key or the national values change.
        key = work_key(sys.modules[__name__], forestmaps, rio, accounts, y, gdmp, clc_file(ctx.store, y),
                       ctx.store.joinpath(*nuts.NUTS_FILE), sums, sorted(nat.items()))
        if ctx.year_done("maps_100m", y, key, done, *(ctx.out / "maps" / "100m" / n for n in names.values())):
            rows.append(pl.read_parquet(done))
            continue
        cache = forestmaps.productivity_cache(ctx.settings.work, ctx.store, y, gdmp)
        # A country without weight gets a factor of 0, so its cells hold 0, as in the earlier maps.
        factors = {q: per_weight_factors(geos, sums, nat[q], zero_when_no_weight=True) for q in QUANTITIES}
        ctx.log(f"  {y}: writing {len(QUANTITIES)} maps")
        written = forestmaps.write_maps(ctx.store, y, g100, FOREST_CODES, gdmp, ctx.settings.workers, factors,
                                        {q: 1.0 for q in QUANTITIES}, ctx.path("maps_100m", "tiles"),
                                        ctx.out / "maps" / "100m", names,
                                        progress=lambda d, n: ctx.tick(f"{y} maps", d, n), log=ctx.log,
                                        cache=cache)
        part = pl.DataFrame([{"geo": g, "year": y, "quantity": q, "map": float(written[q][i]),
                              "table": float(nat[q].get(g) or 0.0)} for q in QUANTITIES for i, g in enumerate(geos)])
        part.write_parquet(ctx.path("maps_100m", f"sums_{y}.parquet"))
        ctx.mark_year_done("maps_100m", y, key)
        rows.append(part)
    compared = pl.concat(rows)
    # A small absolute margin keeps a country with almost nothing from failing on the rounding of whole numbers.
    gap = compared.filter(pl.col("table") > 1e4)
    phys = gap.filter(pl.col("quantity").str.contains("dm3"))
    mon = gap.filter(pl.col("quantity").str.contains("eurocent"))
    ctx.check(phys["map"], phys["table"], "100 m physical maps against the national tables, by country", tolerance=1e-3)
    ctx.check(mon["map"], mon["table"], "100 m monetary maps against the national tables, by country", tolerance=1e-3)
    ctx.save_checks("checks_maps_100m")
    tot = compared.group_by("quantity", "year").agg(pl.col("map").sum(), pl.col("table").sum()).sort("quantity", "year")
    ctx.table(tot.rename({"quantity": "Map", "year": "Year", "map": "Map total", "table": "Table total"}),
              "maps_100m_totals", 0)
