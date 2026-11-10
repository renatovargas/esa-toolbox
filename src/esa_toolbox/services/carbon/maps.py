"""Maps of carbon sequestration at 1 km and 100 m, with their checks against the tables.

The 1 km maps are float32. The 100 m maps are whole numbers in small units (kilograms of CO2 and euro
cents at 2010 prices), which take about a third of the disk of float32, and every country keeps its total within the rounding of its cells.
"""

from __future__ import annotations

import sys

import numpy as np
import polars as pl

from esa_toolbox import accounts, figures, forestmaps, nuts, rio
from esa_toolbox.accounts import country_weight_sums, grid_extent, per_weight_factors, spread
from esa_toolbox.core import Context, work_key
from esa_toolbox.landcover import clc_file
from esa_toolbox.services.carbon.inputs import GDMP_PRODUCT, GRID, forest_share, forest_weight, gdmp_file

FOREST_CODES = [311, 312, 313, 324]


def _price_factor(value: pl.DataFrame, year: int) -> float:
    r = value.filter(pl.col("year") == year).row(0, named=True)
    return r["price_eur_per_t_co2"] * r["deflator"] / 1000


def maps_1km(ctx: Context) -> None:
    flow = ctx.load("carbon_flow_country_year")
    value = ctx.load("carbon_value_country_year")
    regs = nuts.countries(ctx.store)
    geos = regs.table["geo"].to_list()
    ids = regs.ids(GRID)
    out = ctx.out / "maps" / "1km"
    totals, by_country, lost = [], [], []
    stack = {"flow": {}, "value": {}}
    for y in ctx.years:
        w = forest_weight(ctx, y)
        sums = country_weight_sums(w, ids, len(geos))
        fy = flow.filter(pl.col("year") == y)
        col = dict(zip(fy["geo"], fy["flow_kt_co2"]))
        lost += [(y, g) for g, s in zip(geos, sums) if not s > 0 and col.get(g, 0) > 0]
        maps = {}
        for name, column in (("flow", "flow_kt_co2"), ("flow_no_offset", "forest_removals_kt_co2"),
                             ("missed_flow", "missed_flow_kt_co2")):
            f = per_weight_factors(geos, sums, dict(zip(fy["geo"], fy[column])))
            maps[name] = spread(w, ids, f).astype("float32")
        k = _price_factor(value, y)
        maps["value_constant"] = (maps["flow"].astype("float64") * k).astype("float32")
        maps["missed_value_constant"] = (maps["missed_flow"].astype("float64") * k).astype("float32")
        for name, a in maps.items():
            rio.write(out / f"carbon_{name}_gdmp_1km_{y}.tif", a, GRID, names=[f"carbon_{name}_{y}"])
        # The checks read the maps back from disk, so they test the files themselves.
        back = {n: rio.read(out / f"carbon_{n}_gdmp_1km_{y}.tif", GRID, as_float=True) for n in maps}
        vy = value.filter(pl.col("year") == y)
        totals.append({
            "year": y,
            "flow_map_mt": np.nansum(back["flow"], dtype="float64") / 1000, "flow_table_mt": fy["flow_kt_co2"].sum() / 1000,
            "value_map_meur": np.nansum(back["value_constant"], dtype="float64"),
            "value_table_meur": vy["value_constant_meur"].sum(),
            "no_offset_map_mt": np.nansum(back["flow_no_offset"], dtype="float64") / 1000,
            "forest_removals_mt": fy["forest_removals_kt_co2"].sum() / 1000,
            "missed_map_mt": np.nansum(back["missed_flow"], dtype="float64") / 1000,
            "missed_table_mt": fy["missed_flow_kt_co2"].sum() / 1000,
            "missed_value_map_meur": np.nansum(back["missed_value_constant"], dtype="float64"),
            "missed_value_table_meur": vy["missed_value_constant_meur"].sum(),
        })
        bc = country_weight_sums(np.nan_to_num(back["flow"].astype("float64")), ids, len(geos))
        by_country += [(g, y, m, col.get(g, 0.0)) for g, m in zip(geos, bc)]
        stack["flow"][str(y)] = rio.block_sum(back["flow"], 10)
        stack["value"][str(y)] = rio.block_sum(back["value_constant"], 10)
    t = pl.DataFrame(totals)
    bc = pl.DataFrame(by_country, schema=["geo", "year", "map", "table"], orient="row").filter(pl.col("table") > 0)
    for a, b, label in (("flow_map_mt", "flow_table_mt", "1 km flow maps against the tables, in every year"),
                        ("value_map_meur", "value_table_meur", "1 km value maps against the tables, in every year"),
                        ("no_offset_map_mt", "forest_removals_mt", "1 km maps of forest removals against the tables, in every year"),
                        ("missed_map_mt", "missed_table_mt", "1 km missed flow maps against the tables, in every year"),
                        ("missed_value_map_meur", "missed_value_table_meur", "1 km missed value maps against the tables, in every year")):
        ctx.check(t[a], t[b], label, tolerance=1e-5)
    ctx.check(bc["map"], bc["table"], "1 km flow maps against the tables, in every country and year", tolerance=1e-5)
    ctx.save_checks("checks_maps")
    ctx.table(t.select(Year="year", **{"Flow, map (Mt CO2)": "flow_map_mt", "Flow, tables (Mt CO2)": "flow_table_mt",
                                       "Value, map (M EUR)": "value_map_meur", "Value, tables (M EUR)": "value_table_meur"}),
              "map_totals", 1)
    ctx.facts(countries_without_forest=str(len(lost)))
    last = t.filter(pl.col("year") == max(ctx.years)).row(0, named=True)
    rem = flow.filter(pl.col("year") == max(ctx.years))["removals_kt_co2"].sum() / 1000
    ctx.facts(no_offset_map_gap_2024=(rem - last["no_offset_map_mt"], 1),
              forest_share_of_removals_2024=(100 * last["no_offset_map_mt"] / rem, 0))

    area = ctx.load("clc_area_by_nuts")
    fa = []
    for y in ctx.years:
        grid_mha = np.nansum(np.where(ids > 0, forest_share(ctx, y), 0), dtype="float64") * 100 / 1e6
        table_mha = area.filter((pl.col("level") == "NUTS0") & (pl.col("land_type") == "forest") &
                                (pl.col("epoch") == y))["hectares"].sum() / 1e6
        g = rio.read(gdmp_file(ctx, y), GRID, as_float=True)
        fs = forest_share(ctx, y)
        ctx.facts(**{f"forest_grid_mha_{y}": (grid_mha, 1)})
        fa.append({"Year": y, "Forest on the grid (Mha)": grid_mha, "Forest in the regions (Mha)": table_mha,
                   "Mean GDMP on forest (kg DM per ha)": float(np.nanmean(g[fs > 0.5])),
                   "Productivity product": GDMP_PRODUCT[y]})
    ctx.table(pl.DataFrame(fa), "forest_area", 1)

    outlines = regs.geoms
    ext = grid_extent(GRID)
    figures.map_panels(stack["flow"], ext, ctx.figure_path("carbon_flow_maps"), outlines=outlines,
                       cmap="Blues", legend="kt CO2 per 10 km cell", size=(10, 7.2))
    figures.map_panels(stack["value"], ext, ctx.figure_path("carbon_value_maps"), outlines=outlines,
                       cmap="Blues", legend="Million euro per 10 km cell", size=(10, 7.2))


QUANTITIES = {
    "flow_kgco2": ("flow_kt_co2", 1e6),
    "flow_no_offset_kgco2": ("forest_removals_kt_co2", 1e6),
    "missed_flow_kgco2": ("missed_flow_kt_co2", 1e6),
    "value_eurocent2010": ("value_constant_meur", 1e8),
    "missed_value_eurocent2010": ("missed_value_constant_meur", 1e8),
}


def maps_100m(ctx: Context) -> None:
    flow = ctx.load("carbon_flow_country_year")
    value = ctx.load("carbon_value_country_year")
    regs = nuts.countries(ctx.store)
    geos = regs.table["geo"].to_list()
    g100 = GRID.with_res(100)
    rows = []
    for y in ctx.years:
        names = {q: f"carbon_{q}_100m_{y}.tif" for q in QUANTITIES}
        done = ctx.work / "maps_100m" / f"sums_{y}.parquet"
        gdmp = ctx.need("copernicus", "gdmp_annual", f"gdmp_{GDMP_PRODUCT[y]}_{y}_4326.tif")
        # The maps of a year are rewritten whenever the code, the inputs or the national tables of the year change.
        key = work_key(sys.modules[__name__], forestmaps, rio, accounts, y, gdmp, clc_file(ctx.store, y),
                       ctx.store.joinpath(*nuts.NUTS_FILE), flow.filter(pl.col("year") == y),
                       value.filter(pl.col("year") == y))
        if ctx.year_done("maps_100m", y, key, done, *(ctx.out / "maps" / "100m" / n for n in names.values())):
            rows.append(pl.read_parquet(done))
            continue
        cache = forestmaps.productivity_cache(ctx.settings.work, ctx.store, y, gdmp)
        ctx.log(f"  {y}: weights by country")
        sums, _, _ = forestmaps.country_sums(ctx.store, y, g100, FOREST_CODES, gdmp, ctx.settings.workers,
                                          progress=lambda d, n: ctx.tick(f"{y} weights", d, n), cache=cache)
        fy = flow.filter(pl.col("year") == y)
        k = _price_factor(value, y)
        nat = {q: dict(zip(fy["geo"], fy[c])) for q, (c, _) in QUANTITIES.items() if c in fy.columns}
        nat["value_eurocent2010"] = {g: v * k for g, v in nat["flow_kgco2"].items()}
        nat["missed_value_eurocent2010"] = {g: v * k for g, v in nat["missed_flow_kgco2"].items()}
        factors = {q: per_weight_factors(geos, sums, nat[q]) for q in QUANTITIES}
        scales = {q: s for q, (_, s) in QUANTITIES.items()}
        ctx.log(f"  {y}: writing {len(QUANTITIES)} maps")
        written = forestmaps.write_maps(ctx.store, y, g100, FOREST_CODES, gdmp, ctx.settings.workers, factors,
                                        scales, ctx.path("maps_100m", "tiles"), ctx.out / "maps" / "100m", names,
                                        progress=lambda d, n: ctx.tick(f"{y} maps", d, n), log=ctx.log,
                                        cache=cache)
        vy = value.filter(pl.col("year") == y)
        table = {q: dict(zip(fy["geo"], fy[c])) for q, (c, _) in QUANTITIES.items() if c in fy.columns}
        table["value_eurocent2010"] = dict(zip(vy["geo"], vy["value_constant_meur"]))
        table["missed_value_eurocent2010"] = dict(zip(vy["geo"], vy["missed_value_constant_meur"]))
        part = pl.DataFrame([{"geo": g, "year": y, "quantity": q, "map": float(written[q][i]) / scales[q],
                              "table": float(table[q].get(g, 0.0))}
                             for q in QUANTITIES for i, g in enumerate(geos)])
        part.write_parquet(ctx.path("maps_100m", f"sums_{y}.parquet"))
        ctx.mark_year_done("maps_100m", y, key)
        rows.append(part)
    compared = pl.concat(rows)
    pos = compared.filter(pl.col("table") > 0)
    tot = (compared.group_by("quantity", "year").agg(pl.col("map").sum(), pl.col("table").sum())
           .with_columns(difference_pct=100 * (pl.col("map") / pl.col("table") - 1)).sort("quantity", "year"))
    # Whole numbers round every cell, so a country with very little flow, such as Malta, can move by
    # one part in ten thousand. The tolerance by country is therefore one part in a thousand, as for wood.
    ctx.check(pos["map"], pos["table"], "100 m maps against the tables, in every country and year", tolerance=1e-3)
    ctx.check(tot["map"], tot["table"], "100 m maps against the tables, in every year", tolerance=1e-4)
    ctx.save_checks("checks_maps_100m")
    ctx.table(tot.rename({"quantity": "Map", "year": "Year", "map": "Map total", "table": "Table total",
                          "difference_pct": "Difference (%)"}), "maps_100m_totals", {"Difference (%)": 5})
    ctx.facts(largest_100m_gap_pct=(float(tot["difference_pct"].abs().max()), 5))
