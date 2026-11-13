"""Maps of the crop pollination accounts at 1 km and 100 m, the flow by ecosystem type, the workbook and the
checks of maps against tables.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

from esa_toolbox import figures, forestmaps, nuts, rio
from esa_toolbox.core import Context, work_key
from esa_toolbox.pool import run as pool_run
from esa_toolbox.services.pollination import model, sdm
from esa_toolbox.services.pollination.inputs import (GRID, GRID100, cached_model_tile, model_cache, model_key,
                                                     regions_table, supply_bands, tiles, zones)
from esa_toolbox.services.pollination.tables import CROP_NOTE, capri, potential, to_nuts_levels
from esa_toolbox.workbook import Workbook


def m1(ctx: Context, name: str, year: int) -> Path:
    return ctx.out / "maps" / "1km" / f"pollination_{name}_1km_{year}.tif"


def write_1km(ctx: Context, a: np.ndarray, name: str, year: int, digits: int = 3) -> None:
    rio.write(m1(ctx, name, year), np.round(a, digits).astype("float32"), GRID, names=[name])


def account_maps(ctx: Context, year: int, factors: pl.DataFrame, prices: pl.DataFrame) -> dict[str, np.ndarray]:
    regions = regions_table(ctx)
    crops = model.crops(ctx.method)
    zc = zones(ctx, crop_data_only=True)
    area, yld = capri(ctx, year)
    pot = potential(ctx, year)
    met = (pot >= model.SPA_THRESHOLD).astype("float64")
    geo_of = regions.filter(pl.col("id").is_not_null()).select("id", "geo")
    n = int(zc.max())

    def national(table: pl.DataFrame, crop: str, column: str) -> np.ndarray:
        lut = np.full(n + 1, np.nan)
        t = geo_of.join(table.filter((pl.col("crop") == crop) & (pl.col("year") == year)).select("geo", column), on="geo")
        for i, v in zip(t["id"], t[column]):
            if i <= n and v is not None:
                lut[i] = v
        out = lut[zc]
        out[zc == 0] = np.nan
        return out

    sums = {k: 0.0 for k in ("demand_ha", "flow_t", "unmet_t", "flow_meur", "unmet_meur")}
    for r in crops.iter_rows(named=True):
        code, dep = r["capri_code"], r["dependency"]
        k = np.nan_to_num(national(factors, code, "factor"), nan=1.0)
        a = np.where(zc > 0, area[code], np.nan) * (k > 0)
        served = a * yld[code] / 1000 * k * dep
        flow = served * met
        unmet = served * (1 - met)
        u = np.nan_to_num(national(prices, code, "eur_per_t"), nan=0.0)
        sums["demand_ha"] = sums["demand_ha"] + a
        sums["flow_t"] = sums["flow_t"] + flow
        sums["unmet_t"] = sums["unmet_t"] + unmet
        sums["flow_meur"] = sums["flow_meur"] + flow * u / 1e6
        sums["unmet_meur"] = sums["unmet_meur"] + unmet * u / 1e6
    return sums


def maps_1km(ctx: Context) -> None:
    regions = regions_table(ctx)
    factors = ctx.load("pollination_eurostat_factors")
    combined = ctx.load("pollination_accounts_nuts3_crop_year")
    prices = ctx.load("pollination_value_country_crop_year").select("geo", "crop", "year", "eur_per_t")
    for y in ctx.years:
        pot = potential(ctx, y)
        write_1km(ctx, pot, "potential_all", y)
        write_1km(ctx, potential(ctx, y, "ebm_agro"), "potential_agro", y)
        rio.write(m1(ctx, "potential_class", y), model.potential_classes(pot), GRID, nodata=255)
        m = account_maps(ctx, y, factors, prices)
        write_1km(ctx, m["demand_ha"], "demand_ha", y)
        write_1km(ctx, m["flow_t"], "flow_t", y)
        write_1km(ctx, m["unmet_t"], "unmet_t", y)
        write_1km(ctx, m["flow_meur"], "flow_meur2010", y, 6)
        write_1km(ctx, m["unmet_meur"], "unmet_meur2010", y, 6)

    # Flow by ecosystem type, shared in proportion to the expert-based suitability of each type in the cell.
    zf = zones(ctx)
    n = int(zf.max())
    ecos = model.scores().ecosystems
    parts = []
    for y in ctx.years:
        sb = np.nan_to_num(supply_bands(ctx, y).astype("float64"), nan=0.0)
        total = sb.sum(axis=0)
        flow = rio.read(m1(ctx, "flow_t", y), GRID, as_float=True).astype("float64")
        present = np.flatnonzero(np.bincount(zf.ravel(), minlength=n + 1)[1:] > 0) + 1
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(total > 0, sb / np.where(total > 0, total, 1), 0.0)
        for i, e in enumerate(ecos):
            s = rio.zonal_sum(share[i] * flow, zf, n)[present - 1]
            parts.append(pl.DataFrame({"id": present, "ecosystem": e, "flow_t": s, "year": y}))
        # Flow in a cell where no ecosystem supports it is attributed to cropland, where the crops are.
        ns = rio.zonal_sum(np.where(total > 0, 0.0, flow), zf, n)[present - 1]
        parts.append(pl.DataFrame({"id": present, "ecosystem": "Cropland", "flow_t": ns, "year": y}))
    eco = (pl.concat(parts).with_columns(pl.col("id").cast(pl.Int64), pl.col("year").cast(pl.Int64))
           .group_by("id", "ecosystem", "year").agg(pl.col("flow_t").sum()))
    eco_nuts = to_nuts_levels(eco, regions, ["ecosystem", "year"], ["flow_t"]).pivot(
        on="ecosystem", index=["level", "nuts_id", "name", "year"], values="flow_t")
    names = sorted(eco["ecosystem"].unique().to_list())

    # The workbook, with all its sheets.
    wb = Workbook(ctx.output("tables", "pollination_results.xlsx"), "Crop pollination accounts")
    cn = CROP_NOTE + " Croatia, Malta and Cyprus are estimates."
    sheets = {
        "Potential": (ctx.load("pollination_potential_nuts_year"),
                      "Land of high and medium pollination potential, on all ecosystem types and on agroecosystems, in hectares",
                      "High potential is above 0.3 and medium potential from 0.2 to 0.3.",
                      {"land_ha_high": "High potential, all land (ha)", "land_ha_medium": "Medium potential, all land (ha)",
                       "agro_ha_high": "High potential, agroecosystems (ha)", "agro_ha_medium": "Medium potential, agroecosystems (ha)"}),
        "Demand": (ctx.load("pollination_accounts_nuts_year"), "Demand, the area of pollinator-dependent crops, in hectares", cn,
                   {"area_ha": "Demand (ha)", "area_met_ha": "Met (ha)", "area_unmet_ha": "Unmet (ha)"}),
        "Flow": (ctx.load("pollination_accounts_nuts_year"), "Actual flow and unmet demand in physical terms, in tonnes",
                 cn + " The flow has the same total on all ecosystem types and on agroecosystems, and only the 100 m maps place it differently.",
                 {"flow_t": "Actual flow (t)", "unmet_t": "Unmet demand (t)"}),
        "Value": (ctx.load("pollination_value_nuts_year"),
                  "Actual flow and unmet demand in monetary terms, in million euro at constant 2010 prices",
                  cn + " Eurostat holds no production value for Croatia and Cyprus in 2000, so those cells are blank.",
                  {"flow_meur2010": "Actual flow", "unmet_meur2010": "Unmet demand"}),
    }
    for nm, (data, title, note, metrics) in sheets.items():
        for lv in ("NUTS0", "NUTS2", "NUTS3"):
            wb.sheet(f"{nm} {lv}", f"{title}, {lv}", note, data.filter(pl.col("level") == lv), metrics)
    for lv in ("NUTS0", "NUTS2", "NUTS3"):
        wb.sheet(f"Flow by ecosystem {lv}", f"Actual flow in physical terms by ecosystem type, in tonnes, {lv}",
                 "The flow of each kilometre is shared in proportion to the expert-based suitability of each ecosystem "
                 "type. This is the mapped flow, without the estimates of the Flow sheets, which include Croatia, Malta "
                 "and Cyprus. " + CROP_NOTE, eco_nuts.filter(pl.col("level") == lv), {e: f"{e} (t)" for e in names})
    wb.close()

    sbe = (eco.join(regions.filter(pl.col("level") == "NUTS3").select("id"), on="id")
           .group_by("ecosystem", "year").agg(pl.col("flow_t").sum()))
    ctx.table(sbe.pivot(on="year", index="ecosystem", values="flow_t").sort("ecosystem").rename({"ecosystem": "Ecosystem type"}),
              "supply_by_ecosystem", 0)

    # Totals of the maps, read back from disk, against the tables.
    def tot(name, y):
        return float(np.nansum(rio.read(m1(ctx, name, y), GRID, as_float=True), dtype="float64"))
    mt = []
    for y in ctx.years:
        p = rio.read(m1(ctx, "potential_all", y), GRID, as_float=True)
        mt.append({"year": y, "demand_ha": tot("demand_ha", y), "flow_t": tot("flow_t", y), "unmet_t": tot("unmet_t", y),
                   "flow_meur2010": tot("flow_meur2010", y), "unmet_meur2010": tot("unmet_meur2010", y),
                   "potential_min": float(np.nanmin(p)), "potential_max": float(np.nanmax(p))})
    mt = pl.DataFrame(mt)
    mapped = (combined.filter(~pl.col("estimated")).join(prices, on=["geo", "crop", "year"], how="left")
              .with_columns(fm=pl.col("flow_t") * pl.col("eur_per_t").fill_null(0) / 1e6,
                            um=pl.col("unmet_t") * pl.col("eur_per_t").fill_null(0) / 1e6)
              .group_by("year").agg(demand_ha=pl.col("area_ha").sum(), flow_t=pl.col("flow_t").sum(),
                                    unmet_t=pl.col("unmet_t").sum(), flow_meur2010=pl.col("fm").sum(),
                                    unmet_meur2010=pl.col("um").sum()).sort("year"))
    ctx.require(mt["potential_min"].min() >= 0 and mt["potential_max"].max() <= 1, "Potential lies between 0 and 1")
    for q in ("demand_ha", "flow_t", "unmet_t"):
        ctx.check(mt[q], mapped[q], f"{q} map against table", tolerance=1e-4, allow_na=False)
    ctx.check(mt["flow_meur2010"], mapped["flow_meur2010"], "Value of flow, map against table", tolerance=1e-3, allow_na=False)
    ctx.check(mt["unmet_meur2010"], mapped["unmet_meur2010"], "Value of unmet demand, map against table", tolerance=1e-3,
              allow_na=False)
    ctx.save_checks("checks_maps")
    last = max(ctx.years)
    m_last = mt.filter(pl.col("year") == last)
    ctx.facts(map_flow_mt=(m_last["flow_t"][0] / 1e6, 2),
              mapped_flow_mt=(mapped.filter(pl.col("year") == last)["flow_t"][0] / 1e6, 2),
              table_flow_mt=(combined.filter(pl.col("year") == last)["flow_t"].sum() / 1e6, 2))
    allt = combined.group_by("year").agg(table_flow_t=pl.col("flow_t").sum(), table_demand_ha=pl.col("area_ha").sum())
    ctx.table(mt.select("year", "demand_ha", "flow_t", "unmet_t").join(allt, on="year").sort("year").rename(
        {"year": "Year", "demand_ha": "Demand, map (ha)", "flow_t": "Flow, map (t)", "unmet_t": "Unmet demand, map (t)",
         "table_flow_t": "Flow, tables with estimates (t)", "table_demand_ha": "Demand, tables with estimates (ha)"}),
        "maps_against_tables", 0)

    # Figures.
    cg = nuts.countries(ctx.store).geoms
    ext = (GRID.xmin, GRID.xmax, GRID.ymin, GRID.ymax)
    first, last = min(ctx.years), max(ctx.years)
    cls = {str(y): rio.read(m1(ctx, "potential_class", y), GRID).astype("float32") for y in (first, last)}
    for a in cls.values():
        a[a == 255] = np.nan
    figures.map_panels(cls, ext, ctx.figure_path("pollination_potential_class"), outlines=cg,
                       classes=[-0.5, 0.5, 1.5, 2.5, 3.5], class_colours=["#fdeadb", "#f9b98a", "#d97a3a", "#8c3f12"],
                       class_labels=["None", "Low", "Medium", "High"], ncols=2, size=(8, 4.4))
    lim = [0.001, 1, 5, 20, 100, 1e12]
    col5 = ["#fdeadb", "#f9b98a", "#d97a3a", "#a9541f", "#6e3210"]
    pan = {f"{t}, {last}": rio.read(m1(ctx, n, last), GRID, as_float=True)
           for n, t in (("demand_ha", "Demand, ha per km2"), ("flow_t", "Actual flow, t per km2"),
                        ("unmet_t", "Unmet demand, t per km2"))}
    figures.map_panels(pan, ext, ctx.figure_path("pollination_demand_flow_unmet"), outlines=cg, classes=lim,
                       class_colours=col5, class_labels=["Under 1", "1 to 5", "5 to 20", "20 to 100", "100 or more"],
                       size=(9.6, 3.9))


# ---------------------------------------------------------------------------------------------------------
# 100 m maps
# ---------------------------------------------------------------------------------------------------------

QUANTITIES = {  # name: (1 km map it allocates, weight, units per 1 km unit)
    "potential_all_x10000": (None, None, 1e4),
    "potential_agro_x10000": (None, None, 1e4),
    "demand_m2": ("demand_ha", "agro", 1e4),
    "flow_all_kg": ("flow_t", "ebm", 1e3),
    "flow_agro_kg": ("flow_t", "agro", 1e3),
    "unmet_kg": ("unmet_t", "agro", 1e3),
    "flow_agro_eur2010": ("flow_meur2010", "agro", 1e6),
    "unmet_eur2010": ("unmet_meur2010", "agro", 1e6),
}


STRIP = 200  # rows of 100 m processed at once: 20 km, a whole number of 1 km rows


def _maps_job(store: str, out_1km: str, year: int, tile, tile_dir: str, cache: str, sdm_1km: str):
    """The 100 m maps of one tile and their comparison with the 1 km maps.

    Every step works within a 1 km cell, so the tile goes through in strips of 1 km rows. Small arrays stay in
    the processor cache, which lets several workers run at full speed. The cells are the same as for the
    whole tile at once.
    """
    m = cached_model_tile(Path(store), year, tile, Path(cache))
    ebm_t, clc_t = m["ebm"], m["clc"]
    sdm_t = rio.warp(sdm_1km, tile, "bilinear")
    t1 = tile.with_res(1000)
    read1 = lambda name: rio.read(Path(out_1km) / f"pollination_{name}_1km_{year}.tif", t1,  # noqa: E731
                                  as_float=True).astype("float64")
    one = {src: np.nan_to_num(read1(src), nan=0.0) for src, _, _ in QUANTITIES.values() if src}
    ref = {q: read1(src if src else q.replace("_x10000", "")) for q, (src, _, _) in QUANTITIES.items()}
    full = {q: np.empty(ebm_t.shape, dtype="uint16" if src is None else "uint32") for q, (src, _, _) in QUANTITIES.items()}
    acc = {q: {"sum": 0.0, "n": 0, "largest_gap": 0.0, "over_limit": 0, "cells": 0, "lost": 0.0} for q in QUANTITIES}
    sc = model.scores()
    for r in range(0, ebm_t.shape[0], STRIP):
        k = slice(r // 10, (r + STRIP) // 10)
        ebm, clc, sdm = ebm_t[r:r + STRIP], clc_t[r:r + STRIP], sdm_t[r:r + STRIP]
        land = np.where(np.isnan(ebm), np.nan, 1.0)
        agro = np.where(np.isnan(ebm), np.nan, sc.agro[clc].astype("float64"))
        pot = (ebm.astype("float64") + sdm) / 2
        shares = {"agro": model.allocation_share(agro, land), "ebm": model.allocation_share(ebm.astype("float64"), land)}
        for q, (src, weight, units) in QUANTITIES.items():
            if src is None:
                v = pot if q == "potential_all_x10000" else np.where(agro == 1, pot, np.nan)
                v = v * units
                nd = 65535
            else:
                v = model.spread_1km(one[src][k], shares[weight]) * units
                nd = 4294967295
            ok = ~np.isnan(v)
            iv = full[q][r:r + STRIP]
            iv[...] = nd
            iv[ok] = np.rint(v[ok]).astype(iv.dtype)
            # Comparison with the 1 km map: totals, and the largest gap of a 1 km cell, which rounding bounds.
            f = np.where(ok, iv, np.nan).astype("float64")
            rk = ref[q][k]
            agg = (rio.block_mean(f, 10) if src is None else rio.block_sum(f, 10)) / units
            both = ~np.isnan(agg) & ~np.isnan(rk)
            gap = np.abs(agg[both] - rk[both])
            a = acc[q]
            a["sum"] += float(np.nansum(f))
            a["n"] += int(ok.sum())
            a["largest_gap"] = max(a["largest_gap"], float(gap.max()) if gap.size else 0.0)
            a["over_limit"] += int((gap > 50 / units * 1.001).sum())
            a["cells"] += int(both.sum())
            a["lost"] += float(rk[np.isnan(agg) & ~np.isnan(rk)].sum())
    stats = {}
    for q, (src, _, _) in QUANTITIES.items():
        nd = 65535 if src is None else 4294967295
        rio.write(Path(tile_dir) / q / f"{year}_{tile.name}.tif", full[q], tile, nodata=nd, opts=forestmaps.TILE_OPTS)
        a = acc[q]
        stats[q] = ({"sum": a["sum"], "n": a["n"], "largest_gap": a["largest_gap"]} if src is None else
                    {k2: a[k2] for k2 in ("sum", "largest_gap", "over_limit", "cells", "lost")})
    return stats


def maps_100m(ctx: Context) -> None:
    ts = tiles(ctx.store)
    out_1km = ctx.out / "maps" / "1km"
    rows = []
    for y in ctx.years:
        done = ctx.work / "maps_100m" / f"checks_{y}.parquet"
        names = {q: f"pollination_{q}_100m_{y}.tif" for q in QUANTITIES}
        mkey = model_key(ctx.store, y)
        # The 1 km maps count by their contents, since every run of maps_1km writes them again.
        used = sorted({m1(ctx, src, y) for src, _, _ in QUANTITIES.values() if src} |
                      {m1(ctx, q.replace("_x10000", ""), y) for q, (src, _, _) in QUANTITIES.items() if not src})
        key = work_key(sys.modules[__name__], model, forestmaps, rio, y, mkey,
                       sdm.file_1km(ctx, y), *(p.read_bytes() for p in used))
        if ctx.year_done("maps_100m", y, key, done, *(ctx.out / "maps" / "100m" / n for n in names.values())):
            rows.append(pl.read_parquet(done))
            continue
        cache = model_cache(ctx.work, y, mkey)
        ctx.log(f"  {y}: {len(QUANTITIES)} maps on {len(ts)} tiles")
        tile_dir = ctx.path("maps_100m", "tiles")
        res = pool_run(_maps_job, [(str(ctx.store), str(out_1km), y, t, str(tile_dir), str(cache), str(sdm.file_1km(ctx, y)))
                                   for t in ts],
                       ctx.settings.workers, on_done=lambda d, n: ctx.tick(f"{y} maps", d, n))
        for q, (_, _, units) in QUANTITIES.items():
            vrt = tile_dir / q / f"{y}.vrt"
            dtype = "uint16" if q.startswith("potential") else "uint32"
            rio.build_vrt(sorted((tile_dir / q).glob(f"{y}_*.tif")), GRID100, vrt, dtype=dtype,
                          nodata=65535 if dtype == "uint16" else 4294967295)
        forestmaps.cog_all([(tile_dir / q / f"{y}.vrt", ctx.out / "maps" / "100m" / names[q]) for q in QUANTITIES],
                           ctx.settings.workers)
        for q in QUANTITIES:
            for f in (tile_dir / q).glob(f"{y}_*"):
                f.unlink()
        part = []
        for q, (src, _, units) in QUANTITIES.items():
            s = [r[q] for r in res]
            if src is None:
                mean_100 = sum(x["sum"] for x in s) / units / max(1, sum(x["n"] for x in s))
                ref = rio.read(m1(ctx, q.replace("_x10000", ""), y), GRID, as_float=True)
                part.append({"quantity": q, "year": y, "statistic": "mean", "value_1km": float(np.nanmean(ref)),
                             "value_100m": mean_100, "largest_cell_gap": max(x["largest_gap"] for x in s),
                             "cells_over_limit": None})
            else:
                ref = float(np.nansum(rio.read(m1(ctx, src, y), GRID, as_float=True), dtype="float64"))
                part.append({"quantity": q, "year": y, "statistic": "sum", "value_1km": ref,
                             "value_100m": sum(x["sum"] for x in s) / units,
                             "largest_cell_gap": max(x["largest_gap"] for x in s),
                             "cells_over_limit": sum(x["over_limit"] for x in s)})
        part = pl.DataFrame(part).with_columns(relative_gap=(pl.col("value_100m") - pl.col("value_1km")) / pl.col("value_1km"))
        part.write_parquet(ctx.path("maps_100m", f"checks_{y}.parquet"))
        ctx.mark_year_done("maps_100m", y, key)
        rows.append(part)
    t = pl.concat(rows)
    s = t.filter(pl.col("statistic") == "sum")
    ctx.check(s["value_100m"], s["value_1km"], "100 m maps add up to the 1 km maps", tolerance=1e-3)
    ctx.require(s["cells_over_limit"].sum() == 0, "No 1 km cell differs from its 100 m cells by more than rounding allows")
    ctx.save_checks("checks_maps_100m")
    ctx.facts(largest_sum_gap=(100 * float(s["relative_gap"].abs().max()), 2),
              potential_cell_gap=(float(t.filter(pl.col("quantity") == "potential_all_x10000")["largest_cell_gap"].max()), 3))
    ctx.table(t.select(Map="quantity", Year="year", Statistic="statistic", **{"1 km map": "value_1km",
                                                                             "100 m map": "value_100m",
                                                                             "Relative gap": "relative_gap"}),
              "maps_100m_totals", {"1 km map": 2, "100 m map": 2, "Relative gap": 6})
