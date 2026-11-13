"""Tables of the crop pollination accounts: zone tables from the 1 km layers, physical and monetary accounts.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from esa_toolbox import figures, rio
from esa_toolbox.core import Context
from esa_toolbox.services.pollination import model, sdm
from esa_toolbox.services.pollination.inputs import GRID, NO_CROP_DATA, read_reduction, regions_table, zones

CROP_YEAR = {2000: 2000, 2006: 2006, 2012: 2012, 2018: 2018, 2024: 2018}
BASE_YEAR = 2010
CROP_NOTE = ("Values of 2024 use the crop area and yield of 2018, so changes between 2018 and 2024 reflect "
             "pollination potential and national statistics, and not crop relocation.")


# ---------------------------------------------------------------------------------------------------------
# 1 km layers
# ---------------------------------------------------------------------------------------------------------


def potential(ctx: Context, year: int, band: str = "ebm") -> np.ndarray:
    """Pollination potential, the mean of the expert-based model and the bumblebee model, in double precision."""
    e = read_reduction(ctx, year, band).astype("float64")
    return (e + sdm.read(ctx, year).astype("float64")) / 2  # NaN wherever the model has no value, as mask() did


def capri(ctx: Context, year: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Crop area (ha) and yield (kg per ha) by crop, with no data as zero."""
    cy = CROP_YEAR[year]
    out = []
    for kind in ("area_ha", "yield_kgha"):
        path = ctx.need("capri", f"capri_{kind}_{cy}.tif")
        import rasterio

        with rasterio.open(path) as s:
            names = list(s.descriptions)
        a = rio.read(path, GRID, band=list(range(1, len(names) + 1)), as_float=True)
        out.append({n: np.nan_to_num(a[i].astype("float64"), nan=0.0) for i, n in enumerate(names)})
    return out[0], out[1]


def zone_sums(values: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sums by zone over the zones present in `z` (terra zonal, na.rm = TRUE). Returns ids and sums."""
    n = int(z.max())
    present = np.flatnonzero(np.bincount(z.ravel(), minlength=n + 1)[1:] > 0) + 1
    s = rio.zonal_sum(values, z, n)
    return present, s[present - 1]


def zone_table(cols: dict[str, np.ndarray], z: np.ndarray) -> pl.DataFrame:
    ids = None
    data = {}
    for name, v in cols.items():
        ids, data[name] = zone_sums(v, z)
    return pl.DataFrame({"id": ids.astype("int64"), **data})


def capri_by_zone(ctx: Context, year: int, zc: np.ndarray) -> pl.DataFrame:
    area, yld = capri(ctx, year)
    met = np.nan_to_num((potential(ctx, year) >= model.SPA_THRESHOLD).astype("float64") *
                        np.where(np.isnan(potential(ctx, year)), np.nan, 1.0), nan=0.0)
    parts = []
    for crop in area:
        prod = area[crop] * yld[crop] / 1000
        t = zone_table({"area_ha": area[crop], "area_met_ha": area[crop] * met, "production_t": prod,
                        "production_met_t": prod * met}, zc)
        parts.append(t.with_columns(crop=pl.lit(crop), year=pl.lit(year, pl.Int64)))
    return pl.concat(parts).select("id", "crop", "year", "area_ha", "area_met_ha", "production_t", "production_met_t")


def potential_by_zone(ctx: Context, year: int, z: np.ndarray) -> pl.DataFrame:
    land = read_reduction(ctx, year, "land_cells").astype("float64")
    agro = np.nan_to_num(read_reduction(ctx, year, "agro_cells").astype("float64"), nan=0.0)
    cls = model.potential_classes(potential(ctx, year))
    parts = []
    for k, label in enumerate(model.CLASS_LABELS):
        inc = (cls == k).astype("float64")
        t = zone_table({"land_ha": land * inc, "agro_ha": agro * inc}, z)
        parts.append(t.with_columns(**{"class": pl.lit(label)}))
    return pl.concat(parts).with_columns(year=pl.lit(year, pl.Int64)).select("id", "year", "class", "land_ha", "agro_ha")


def to_nuts_levels(data: pl.DataFrame, regions: pl.DataFrame, by: list[str], values: list[str]) -> pl.DataFrame:
    """Add NUTS3 values up to NUTS2 (first four characters) and NUTS0 (first two)."""
    n3 = regions.filter(pl.col("level") == "NUTS3").select("id", nuts3="nuts_id")
    base = data.join(n3, on="id").with_columns(NUTS3=pl.col("nuts3"), NUTS2=pl.col("nuts3").str.slice(0, 4),
                                               NUTS0=pl.col("nuts3").str.slice(0, 2))
    parts = [base.group_by(pl.col(lv).alias("nuts_id"), *by, maintain_order=True)
             .agg([pl.col(v).sum() for v in values]).with_columns(level=pl.lit(lv))
             for lv in ("NUTS0", "NUTS2", "NUTS3")]
    out = pl.concat(parts).join(regions.select("nuts_id", "name"), on="nuts_id", how="left")
    return out.select("level", "nuts_id", "name", *by, *values).sort("level", "nuts_id", *by)


# ---------------------------------------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------------------------------------


def accounts(ctx: Context) -> None:
    zone_tables(ctx)
    physical(ctx)
    monetary(ctx)


def zone_tables(ctx: Context) -> None:
    zc = zones(ctx, crop_data_only=True)
    zf = zones(ctx)
    ctx.save(pl.concat([capri_by_zone(ctx, y, zc) for y in ctx.years]), "capri_by_zone")
    ctx.save(pl.concat([potential_by_zone(ctx, y, zf) for y in ctx.years]), "farmland_by_zone")

    regions = regions_table(ctx)
    ref = 2018 if 2018 in ctx.years else ctx.years[-1]
    land = read_reduction(ctx, ref, "land_cells")
    ctx.facts(countries=str(regions.filter(pl.col("level") == "NUTS0").height),
              nuts2=str(regions.filter(pl.col("level") == "NUTS2").height),
              nuts3=str(regions.filter(pl.col("level") == "NUTS3").height),
              land_mha=(np.nansum(land, dtype="float64") / 1e6, 1),
              agro_mha=(np.nansum(read_reduction(ctx, ref, "agro_cells"), dtype="float64") / 1e6, 1),
              cells_1km=f"{int((~np.isnan(read_reduction(ctx, ref, 'ebm'))).sum()):,}")
    rows = []
    for y in ctx.years:
        s = sdm.read(ctx, y)
        ebm = read_reduction(ctx, y, "ebm")
        has = ~np.isnan(ebm)
        ai = rio.read(ctx.need("catalogue", f"activity_index_{y}_1km.tif"), GRID, as_float=True)
        for name, a in (("Activity index (%)", ai), ("Expert-based model", ebm), ("Bumblebee model", s),
                        ("Potential", potential(ctx, y))):
            v = a[has & ~np.isnan(a)].astype("float64")
            rows.append({"Layer": name, "Year": y, "Mean": v.mean(), "Lowest": v.min(), "Highest": v.max()})
    ls = pl.DataFrame(rows)
    ctx.table(ls, "layer_summary", 3)
    b = ls.filter(pl.col("Layer") == "Bumblebee model").row(0, named=True)
    ctx.facts(sdm_min=(b["Lowest"], 2), sdm_max=(b["Highest"], 2), n_scores=str(model.read_param("clc_pollination_scores.csv").height))
    t = model.read_param("clc_pollination_scores.csv").select(
        Code="clc", Class="label", Floral="floral", Nesting="nesting", Ecosystem="ecosystem")
    ctx.table(t, "scores", 2)
    ctx.table(model.crops(ctx.method).select(Code="capri_code", Crop="crop", **{"Dependency ratio": "dependency",
                                   "Class": "dependency_class", "Eurostat items": pl.col("eaa_items").str.replace_all(";", ", ")}),
              "dependency", 3)
    comp = (model.crop_components().with_columns(text=pl.format("{} {}", "siopa_crop", pl.col("dependency").map_elements(lambda v: f"{v:.2f}", return_dtype=pl.String)))
            .group_by("capri_code", maintain_order=True).agg(pl.col("text").str.join("; "), mean=pl.col("dependency").mean()))
    both = model.crops("revised").join(model.crops("original").select("capri_code", earlier="dependency"), on="capri_code")
    ctx.table(both.join(comp, on="capri_code").select(
        Crop="crop", **{"Crops and ratios of Siopa et al. (2024)": "text", "Ratio": "dependency",
                        "Ratio of Klein et al. (2007)": "earlier"}), "dependency_revision", 3)

    large, by_class, uncovered = [], [], []
    for y in ctx.years:
        area, _ = capri(ctx, y)
        total_area = sum(area.values())
        demand = np.where(zc > 0, total_area, np.nan)
        tot = np.nansum(demand)
        large.append({"Year": y, "Cells over 100 ha": int(np.nansum(demand > 100)),
                      "Cells with demand": int(np.nansum(demand > 0)), "Largest (ha)": float(np.nanmax(demand)),
                      "Share of demand in those cells (%)": 100 * float(np.nansum(np.where(demand > 100, demand, 0)) / tot)})
        cls = model.potential_classes(potential(ctx, y))
        row = {"Year": y}
        acc = 0.0
        for k, label in enumerate(model.CLASS_LABELS):
            v = float(np.nansum(np.where(cls == k, demand, 0)))
            row[label] = v
            acc += v
        row["no potential"] = tot - acc
        by_class.append(row)
        t = regions.filter((pl.col("level") == "NUTS3") & pl.col("geo").is_in(NO_CROP_DATA))
        for geo in NO_CROP_DATA:
            ids = t.filter(pl.col("geo") == geo)["id"].to_numpy()
            m = np.isin(zf, ids)
            for crop, a in area.items():
                v = float(a[m].sum())
                if v > 0:
                    uncovered.append({"geo": geo, "crop": crop, "year": y, "area_ha": v})
    lg = pl.DataFrame(large)
    ctx.table(lg, "demand_large_cells", 1)
    r18 = lg.filter(pl.col("Year") == 2018)
    if r18.height:
        ctx.facts(large_cells_2018=(r18["Cells over 100 ha"][0], 0), largest_ha_2018=(r18["Largest (ha)"][0], 0),
                  large_share_2018=(r18["Share of demand in those cells (%)"][0], 1))
    ctx.table(pl.DataFrame(by_class), "demand_by_class", 0)
    ctx.save(pl.DataFrame(uncovered), "uncovered_crop_area")


def physical(ctx: Context) -> None:
    regions = regions_table(ctx)
    crops = model.crops(ctx.method).select(crop="capri_code", dependency="dependency")
    geo_of = regions.filter(pl.col("level") == "NUTS3").select("id", "geo")
    production = (pl.read_parquet(ctx.need("eurostat", "pollination", "production_crop.parquet"))
                  .with_columns(pl.col("year").cast(pl.Int64)))
    farmland = ctx.load("farmland_by_zone").join(geo_of, on="id")

    pn = (farmland.filter(pl.col("class").is_in(["medium", "high"]))
          .group_by("id", "year", "class").agg(pl.col("land_ha", "agro_ha").sum())
          .pivot(on="class", index=["id", "year"], values=["land_ha", "agro_ha"]))
    vals = ["land_ha_medium", "land_ha_high", "agro_ha_medium", "agro_ha_high"]
    pn = pn.with_columns([pl.col(v).fill_null(0.0) for v in vals])
    potential_nuts = to_nuts_levels(pn, regions, ["year"], vals)
    ctx.save(potential_nuts, "pollination_potential_nuts_year")

    capri_zone = ctx.load("capri_by_zone").join(geo_of, on="id").join(crops, on="crop")
    national = (capri_zone.group_by("geo", "crop", "year").agg(capri_t=pl.col("production_t").sum())
                .join(production.select("geo", "crop", "year", eurostat_t="production_t"),
                      on=["geo", "crop", "year"], how="left")
                .with_columns(eurostat_gap=pl.col("eurostat_t").is_null() & (pl.col("capri_t") > 0),
                              factor=pl.when((pl.col("capri_t") > 0) & pl.col("eurostat_t").is_not_null())
                              .then(pl.col("eurostat_t") / pl.col("capri_t")).otherwise(1.0)))
    all_scaled = (capri_zone.join(national.select("geo", "crop", "year", "factor", "eurostat_gap"),
                                  on=["geo", "crop", "year"], how="left")
                  .with_columns(flow_t=pl.col("production_met_t") * pl.col("factor") * pl.col("dependency"),
                                unmet_t=(pl.col("production_t") - pl.col("production_met_t")) * pl.col("factor") * pl.col("dependency"),
                                total_t=pl.col("production_t") * pl.col("factor"), estimated=pl.lit(False)))
    out_of_range = (pl.col("factor") > 10) | (pl.col("factor") < 0.1)
    acc = all_scaled.filter(~out_of_range)
    factors = (capri_zone.select("geo", "crop", "year").unique()
               .join(acc.select("geo", "crop", "year", "factor", "eurostat_gap").unique(), on=["geo", "crop", "year"], how="left")
               .with_columns(pl.col("factor").fill_null(0.0), pl.col("eurostat_gap").fill_null(False)))
    ctx.save(factors, "pollination_eurostat_factors")

    placed = (acc.group_by("geo", "crop", "year").agg(capri_t=pl.col("production_t").sum())
              .filter(pl.col("capri_t") > 0))
    missing = (production.filter(pl.col("production_t") > 0)
               .join(placed, on=["geo", "crop", "year"], how="anti").join(crops, on="crop"))
    shares = (farmland.group_by("geo", "id", "year").agg(
        agro=pl.col("agro_ha").sum(),
        agro_met=pl.col("agro_ha").filter(pl.col("class").is_in(["medium", "high"])).sum())
        .with_columns(region_share=pl.when(pl.col("agro").sum().over("geo", "year") > 0)
                      .then(pl.col("agro") / pl.col("agro").sum().over("geo", "year")).otherwise(0.0),
                      met_share=pl.when(pl.col("agro") > 0).then(pl.col("agro_met") / pl.col("agro")).otherwise(0.0)))
    est = (missing.join(shares, on=["geo", "year"])
           .select("id", "crop", "year", "geo", "dependency", total_t=pl.col("production_t") * pl.col("region_share"),
                   met_share="met_share")
           .with_columns(flow_t=pl.col("total_t") * pl.col("met_share") * pl.col("dependency"),
                         unmet_t=pl.col("total_t") * (1 - pl.col("met_share")) * pl.col("dependency"),
                         estimated=pl.lit(True))
           .drop("met_share")
           .join(missing.select("geo", "crop", "year", national_area="area_ha"), on=["geo", "crop", "year"], how="left")
           .with_columns(area_ha=pl.col("national_area").fill_null(0.0) * pl.col("total_t") /
                         pl.col("total_t").sum().over("geo", "crop", "year"))
           .with_columns(area_met_ha=pl.when(pl.col("total_t") > 0)
                         .then(pl.col("area_ha") * pl.col("flow_t") / (pl.col("total_t") * pl.col("dependency")))
                         .otherwise(0.0))
           .drop("national_area"))
    cols = ["id", "crop", "year", "geo", "dependency", "area_ha", "area_met_ha", "total_t", "flow_t", "unmet_t", "estimated"]
    combined = pl.concat([acc.select(cols), est.select(cols)]).with_columns(
        area_unmet_ha=pl.col("area_ha") - pl.col("area_met_ha"))
    ctx.save(combined, "pollination_accounts_nuts3_crop_year")
    contribution = (combined.group_by("geo", "crop", "year", "dependency").agg(
        pl.col("total_t", "flow_t", "unmet_t", "area_ha", "area_met_ha").sum(), estimated=pl.col("estimated").any())
        .with_columns(contribution=pl.when(pl.col("total_t") > 0).then(pl.col("flow_t") / pl.col("total_t"))))
    ctx.save(contribution, "pollination_contribution_country_crop_year")

    value_cols = ["area_ha", "area_met_ha", "area_unmet_ha", "total_t", "flow_t", "unmet_t"]
    nuts_total = to_nuts_levels(combined, regions, ["year"], value_cols)
    ctx.save(nuts_total, "pollination_accounts_nuts_year")

    lv = nuts_total.group_by("level", "year").agg(pl.col("flow_t").sum()).pivot(on="level", index="year", values="flow_t")
    ident = combined.select(left=(pl.col("flow_t") + pl.col("unmet_t")).sum(),
                            right=(pl.col("total_t") * pl.col("dependency")).sum())
    anchor = (combined.group_by("geo", "crop", "year").agg(pl.col("total_t").sum())
              .join(production.select("geo", "crop", "year", "production_t"), on=["geo", "crop", "year"])
              .filter(pl.col("production_t").is_not_null() & (pl.col("production_t") > 0)))
    ctx.check(lv["NUTS3"], lv["NUTS0"], "Flow by NUTS3 against NUTS0", tolerance=1e-4, allow_na=False)
    ctx.check(lv["NUTS2"], lv["NUTS0"], "Flow by NUTS2 against NUTS0", tolerance=1e-4, allow_na=False)
    ctx.check(ident["left"], ident["right"], "Flow plus unmet demand against dependent production", tolerance=1e-4,
              allow_na=False)
    ctx.check(anchor["total_t"], anchor["production_t"], "National production against Eurostat", tolerance=1e-3,
              allow_na=False)
    ctx.save_checks("checks_physical")

    eu = (contribution.group_by("year").agg(
        demand=pl.col("area_ha").sum() / 1e6, met=100 * pl.col("area_met_ha").sum() / pl.col("area_ha").sum(),
        flow=pl.col("flow_t").sum() / 1e6, unmet=pl.col("unmet_t").sum() / 1e6,
        est=100 * pl.col("flow_t").filter(pl.col("estimated")).sum() / pl.col("flow_t").sum()).sort("year"))
    ctx.table(eu.rename({"year": "Year", "demand": "Demand (Mha)", "met": "Met demand (%)", "flow": "Actual flow (Mt)",
                         "unmet": "Unmet demand (Mt)", "est": "Share of flow estimated (%)"}), "eu_series", 2)
    for r in eu.iter_rows(named=True):
        y = r["year"]
        ctx.facts(**{f"demand_mha_{y}": (r["demand"], 2), f"met_pct_{y}": (r["met"], 1),
                     f"flow_mt_{y}": (r["flow"], 2), f"unmet_mt_{y}": (r["unmet"], 2), f"estimated_pct_{y}": (r["est"], 1)})
    pc = (farmland.group_by("year", "class").agg(land=pl.col("land_ha").sum() / 1e6, agro=pl.col("agro_ha").sum() / 1e6))
    pcw = pc.pivot(on="class", index="year", values="land").sort("year").select("year", *model.CLASS_LABELS)
    ctx.table(pcw.rename({"year": "Year", **{c: f"{c.capitalize()} (Mha)" for c in model.CLASS_LABELS}}),
              "potential_by_class", 1)
    pca = pc.pivot(on="class", index="year", values="agro").sort("year").select("year", *model.CLASS_LABELS)
    ctx.table(pca.rename({"year": "Year", **{c: f"{c.capitalize()} (Mha)" for c in model.CLASS_LABELS}}),
              "potential_by_class_agro", 1)
    for r in pc.iter_rows(named=True):
        ctx.facts(**{f"land_{r['class']}_mha_{r['year']}": (r["land"], 1), f"agro_{r['class']}_mha_{r['year']}": (r["agro"], 1)})

    allnat = (all_scaled.group_by("geo", "crop", "year", "factor", "eurostat_gap")
              .agg(capri_t=pl.col("production_t").sum(), eurostat_t=pl.col("total_t").sum()).filter(pl.col("capri_t") > 0))
    oor = allnat.filter((pl.col("factor") > 10) | (pl.col("factor") < 0.1)).select("geo", "crop", "year", "factor",
                                                                                   "capri_t", "eurostat_t").sort("year", "geo", "crop")
    ctx.table(oor.rename({"geo": "Country", "crop": "Crop", "year": "Year", "factor": "Factor",
                          "capri_t": "CAPRI (t)", "eurostat_t": "Eurostat (t)"}), "factors_out_of_range",
              {"Factor": 2, "CAPRI (t)": 0, "Eurostat (t)": 0})
    prod_year = contribution.group_by("year").agg(tot=pl.col("total_t").sum())
    oy = (oor.group_by("year").agg(cases=pl.len(), capri=pl.col("capri_t").sum() / 1e3, euro=pl.col("eurostat_t").sum() / 1e3)
          .join(prod_year, on="year").with_columns(share=100 * pl.col("euro") * 1e3 / pl.col("tot")).sort("year"))
    ctx.table(oy.select(Year="year", Cases="cases", **{"CAPRI production set aside (kt)": "capri",
                                                       "Eurostat production re-estimated (kt)": "euro",
                                                       "Share of all production (%)": "share"}), "out_of_range_year", 2)
    def case(geo, crop, year, col):
        r = allnat.filter((pl.col("geo") == geo) & (pl.col("crop") == crop) & (pl.col("year") == year))
        return float(r[col][0]) if r.height else float("nan")
    ctx.facts(lu_ofru_2000_capri=(case("LU", "OFRU", 2000, "capri_t"), 0),
              lu_ofru_2000_eurostat=(case("LU", "OFRU", 2000, "eurostat_t"), 0),
              el_text_2006_capri_kt=(case("EL", "TEXT", 2006, "capri_t") / 1e3, 0),
              el_rape_2000_eurostat=(case("EL", "RAPE", 2000, "eurostat_t"), 0))
    pt = allnat.filter(pl.col("eurostat_gap") & (pl.col("geo") == "PT") & (pl.col("year") == 2006) & (pl.col("crop") == "TOMA"))
    ctx.facts(pt_toma_2006_mt=(float(pt["capri_t"].sum()) / 1e6, 1))
    capri_nat = capri_zone.group_by("geo", "crop", "year").agg(capri_ha=pl.col("area_ha").sum(), capri_t=pl.col("production_t").sum())
    ce = (capri_nat.join(production.select("geo", "crop", "year", "area_ha", "production_t"), on=["geo", "crop", "year"])
          .group_by("crop", "year").agg(capri_ha=pl.col("capri_ha").filter(pl.col("area_ha").is_not_null()).sum(),
                                        eurostat_ha=pl.col("area_ha").sum(),
                                        capri_t=pl.col("capri_t").filter(pl.col("production_t").is_not_null()).sum(),
                                        eurostat_t=pl.col("production_t").sum()))
    c18 = ce.filter(pl.col("year") == 2018).sort("crop")
    cnames = dict(zip(*model.crops(ctx.method).select("capri_code", "crop")))
    ctx.table(c18.select(Crop=pl.col("crop").replace_strict(cnames), **{
        "CAPRI (kha)": pl.col("capri_ha") / 1e3, "Eurostat (kha)": pl.col("eurostat_ha") / 1e3,
        "Area ratio": pl.col("capri_ha") / pl.col("eurostat_ha"), "CAPRI (kt)": pl.col("capri_t") / 1e3,
        "Eurostat (kt)": pl.col("eurostat_t") / 1e3, "Production ratio": pl.col("capri_t") / pl.col("eurostat_t")}),
        "capri_against_eurostat", {"Area ratio": 2, "Production ratio": 2})
    tx = c18.filter(pl.col("crop") == "TEXT")
    if tx.height:
        ctx.facts(text_area_ratio_2018=(tx["capri_ha"][0] / tx["eurostat_ha"][0], 0))
    last = max(ctx.years)
    fl = contribution.filter(pl.col("year") == last)
    ctx.facts(cy_mt_flow=(100 * fl.filter(pl.col("geo").is_in(["CY", "MT"]))["flow_t"].sum() / fl["flow_t"].sum(), 2))
    if 2018 in ctx.years:
        unc = ctx.load("uncovered_crop_area").filter(pl.col("year") == 2018).group_by("geo").agg(capri=pl.col("area_ha").sum())
        est18 = combined.filter(pl.col("estimated") & (pl.col("year") == 2018)).group_by("geo").agg(
            pl.col("area_ha").sum(), pl.col("flow_t").sum())
        names = {"HR": "Croatia", "CY": "Cyprus", "MT": "Malta"}
        ut = (pl.DataFrame({"geo": list(names), "Country": list(names.values())}).join(unc, on="geo", how="left")
              .join(est18, on="geo", how="left").select("Country", **{"CAPRI series (ha)": pl.col("capri").fill_null(0.0),
                                                                      "Eurostat (ha)": "area_ha", "Estimated flow (t)": "flow_t"}))
        ctx.table(ut, "uncovered", 0)
    ctx.facts(factors_out_of_range=str(oor.height),
              no_eurostat_figure=str(allnat.filter(pl.col("eurostat_gap")).height))
    est_sum = (combined.filter(pl.col("estimated"))
               .with_columns(group=pl.when(pl.col("geo").is_in(NO_CROP_DATA)).then(pl.col("geo")).otherwise(pl.lit("Other countries")))
               .group_by("group", "year").agg(pl.col("total_t", "flow_t", "unmet_t", "area_ha").sum()).sort("group", "year"))
    ctx.table(est_sum.rename({"group": "Countries", "year": "Year", "total_t": "Production (t)", "flow_t": "Actual flow (t)",
                              "unmet_t": "Unmet demand (t)", "area_ha": "Area (ha)"}), "estimates_summary", 0)
    ebc = (contribution.group_by("crop", "year").agg(pl.col("area_ha", "area_met_ha", "total_t", "flow_t", "unmet_t").sum())
           .with_columns(contribution=pl.col("flow_t") / pl.col("total_t")))
    last = max(ctx.years)
    names = dict(zip(*model.crops(ctx.method).select("capri_code", "crop")))
    ctx.table(ebc.filter(pl.col("year") == last).sort("flow_t", descending=True).select(
        Crop=pl.col("crop").replace_strict(names), **{"Demand (ha)": "area_ha", "Production (t)": "total_t",
                                                       "Actual flow (t)": "flow_t", "Unmet demand (t)": "unmet_t",
                                                       "Pollination contribution": "contribution"}),
              "eu_by_crop", {"Demand (ha)": 0, "Production (t)": 0, "Actual flow (t)": 0, "Unmet demand (t)": 0,
                             "Pollination contribution": 3})
    bc = (contribution.group_by("geo", "year").agg(pl.col("area_ha", "area_met_ha", "total_t", "flow_t", "unmet_t").sum(),
                                                   estimated=pl.col("estimated").any())
          .join(regions.filter(pl.col("level") == "NUTS0").select("geo", "name"), on="geo", how="left"))
    ctx.save(bc, "by_country")
    ctx.table(bc.filter(pl.col("year") == last).sort("flow_t", descending=True).select(
        Country="name", **{"Demand (ha)": "area_ha", "Met demand (ha)": "area_met_ha", "Actual flow (t)": "flow_t",
                           "Unmet demand (t)": "unmet_t", "Estimated": "estimated"}), "flow_by_country", 0)
    published = pl.DataFrame({"geo": ["DE", "ES", "FR", "IT", "PL", "RO"], "APPL_published": [0.57, 0.26, 0.29, 0.13, 0.59, 0.47],
                              "RAPE_published": [0.21, 0.03, 0.05, 0.05, 0.22, 0.05]})
    if 2006 in ctx.years:
        cp = (contribution.filter((pl.col("year") == 2006) & pl.col("crop").is_in(["APPL", "RAPE"]))
              .pivot(on="crop", index="geo", values="contribution").rename({"APPL": "APPL_here", "RAPE": "RAPE_here"})
              .join(published, on="geo").sort("geo"))
        ctx.table(cp.select(Country="geo", **{"Apples, here": "APPL_here", "Apples, published": "APPL_published",
                                              "Rapeseed, here": "RAPE_here", "Rapeseed, published": "RAPE_published"}),
                  "contribution_published", 2)


def monetary(ctx: Context) -> None:
    regions = regions_table(ctx)
    combined = ctx.load("pollination_accounts_nuts3_crop_year")
    contribution = ctx.load("pollination_contribution_country_crop_year")
    eaa = (pl.read_parquet(ctx.need("eurostat", "pollination", "eaa_value_crop.parquet"))
           .with_columns(pl.col("year").cast(pl.Int64)).filter(pl.col("value_meur").is_not_null()))
    hicp = pl.read_parquet(ctx.need("eurostat", "shared", "hicp_eu27.parquet")).with_columns(pl.col("year").cast(pl.Int64))
    h = {y: v for y, v in zip(hicp["year"], hicp["hicp_2015"]) if v is not None}
    # Each reporting year averages three years, each in the prices of the year before.
    defl = pl.DataFrame([{"year": y, "price_level": float(np.nanmean([h[k] for k in range(y - 2, y + 1) if k in h])),
                          "deflator": h[BASE_YEAR] / float(np.nanmean([h[k] for k in range(y - 2, y + 1) if k in h]))}
                         for y in ctx.years])
    nv = (contribution.join(eaa, on=["geo", "crop", "year"]).join(defl, on="year")
          .with_columns(unmet_share=pl.when(pl.col("total_t") > 0).then(pl.col("unmet_t") / pl.col("total_t")).otherwise(0.0),
                        flow_meur=pl.col("value_meur") * pl.col("contribution").fill_null(0.0))
          .with_columns(unmet_meur=pl.col("value_meur") * pl.col("unmet_share"))
          .with_columns(flow_meur2010=pl.col("flow_meur") * pl.col("deflator"),
                        unmet_meur2010=pl.col("unmet_meur") * pl.col("deflator"),
                        eur_per_t=pl.when(pl.col("total_t") > 0)
                        .then(pl.col("value_meur") * pl.col("deflator") * 1e6 / pl.col("total_t")).otherwise(0.0)))
    ctx.save(nv, "pollination_value_country_crop_year")
    unvalued = eaa.join(contribution.filter(pl.col("total_t") > 0), on=["geo", "crop", "year"], how="anti")["value_meur"].sum()
    no_value = 100 * contribution.join(eaa, on=["geo", "crop", "year"], how="anti")["flow_t"].sum() / contribution["flow_t"].sum()
    ctx.facts(unvalued_meur=(unvalued, 1), no_value_share=(no_value, 2))

    rv = (combined.join(nv.select("geo", "crop", "year", "eur_per_t"), on=["geo", "crop", "year"])
          .with_columns(flow_meur2010=pl.col("flow_t") * pl.col("eur_per_t") / 1e6,
                        unmet_meur2010=pl.col("unmet_t") * pl.col("eur_per_t") / 1e6))
    nuts_value = to_nuts_levels(rv, regions, ["year"], ["flow_meur2010", "unmet_meur2010"])
    ctx.save(nuts_value, "pollination_value_nuts_year")
    lv = nuts_value.group_by("level", "year").agg(v=pl.col("flow_meur2010").sum()).pivot(on="level", index="year", values="v")
    ctx.require(((nv["flow_meur"] - nv["value_meur"] * nv["dependency"]) <= 1e-6).all(),
                "The value of the flow never exceeds the dependent share of production value")
    ctx.check(rv["flow_meur2010"].sum(), nv["flow_meur2010"].sum(), "Value of flow, regions against countries",
              tolerance=1e-4, allow_na=False)
    ctx.check(rv["unmet_meur2010"].sum(), nv["unmet_meur2010"].sum(), "Value of unmet demand, regions against countries",
              tolerance=1e-4, allow_na=False)
    ctx.check(lv["NUTS3"], lv["NUTS0"], "Value of flow by NUTS3 against NUTS0", tolerance=1e-4, allow_na=False)
    ctx.save_checks("checks_monetary")

    ms = (nv.group_by("year").agg(value=(pl.col("value_meur") * pl.col("deflator")).sum(), flow=pl.col("flow_meur2010").sum(),
                                  unmet=pl.col("unmet_meur2010").sum(), flow_prev=pl.col("flow_meur").sum()).sort("year"))
    ctx.table(ms.rename({"year": "Year", "value": "Dependent crops, value (M EUR 2010)", "flow": "Value of flow (M EUR 2010)",
                         "unmet": "Value of unmet demand (M EUR 2010)",
                         "flow_prev": "Value of flow (M EUR, prices of previous year)"}), "value_eu_series", 1)
    for r in ms.iter_rows(named=True):
        y = r["year"]
        ctx.facts(**{f"value_flow_bn_{y}": (r["flow"] / 1000, 1), f"value_unmet_bn_{y}": (r["unmet"] / 1000, 1)})
        ctx.facts(**{f"value_flow_meur_{y}": (r["flow"], 0), f"value_unmet_meur_{y}": (r["unmet"], 0),
                     f"value_crops_meur_{y}": (r["value"], 0)})
    for r in defl.iter_rows(named=True):
        ctx.facts(**{f"deflator_{r['year']}": (r["deflator"], 2)})
    est = sorted(hicp.filter(pl.col("estimated"))["year"].to_list()) if "estimated" in hicp.columns else []
    ctx.facts(hicp_first_published=str(min(y for y in h if y not in est)),
              hicp_estimated=f"{est[0]} to {est[-1]}" if est else "none")
    ctx.table(defl.select(Year="year", **{"Price level (2015 = 100)": "price_level", "Deflator": "deflator"}), "deflators", 3)
    names = dict(zip(*model.crops(ctx.method).select("capri_code", "crop")))
    last = max(ctx.years)
    vbc = (nv.group_by("crop", "year").agg(value=(pl.col("value_meur") * pl.col("deflator")).sum(),
                                           flow=pl.col("flow_meur2010").sum(), unmet=pl.col("unmet_meur2010").sum()))
    ctx.table(vbc.filter(pl.col("year") == last).sort("flow", descending=True).select(
        Crop=pl.col("crop").replace_strict(names), **{"Production value (M EUR 2010)": "value",
                                                       "Value of flow (M EUR 2010)": "flow",
                                                       "Value of unmet demand (M EUR 2010)": "unmet"}), "value_by_crop", 1)
    by_country = (nv.group_by("geo", "year").agg(flow=pl.col("flow_meur2010").sum(), unmet=pl.col("unmet_meur2010").sum(),
                                                 value=(pl.col("value_meur") * pl.col("deflator")).sum()))
    ctx.table(by_country.filter(pl.col("year") == last).sort("flow", descending=True).select(
        Country="geo", **{"Value of flow (M EUR 2010)": "flow", "Value of unmet demand (M EUR 2010)": "unmet",
                          "Production value (M EUR 2010)": "value"}), "value_by_country", 1)
    if 2006 in ctx.years:
        pub = pl.DataFrame({"crop": ["APPL", "OFRU", "CITR", "PULS", "OOIL", "RAPE", "SOYA", "SUNF", "TEXT", "TOMA"],
                            "published_value": [2240.58, 5923.31, 3238.37, 883.62, 155.10, 4177.13, 287.19, 1584.96, 1135.03, 5928.34],
                            "published_contribution": [0.48, 0.30, 0.02, 0.03, 0.12, 0.17, 0.16, 0.12, 0.04, 0.04]})
        v6 = (nv.filter(pl.col("year") == 2006).group_by("crop")
              .agg(value=pl.col("value_meur").sum(), flow=pl.col("flow_meur").sum(),
                   contribution=pl.col("flow_t").sum() / pl.col("total_t").sum())
              .join(pub, on="crop", how="left").sort("crop"))
        ctx.table(v6.select(Crop=pl.col("crop").replace_strict(names), **{"Production value, here (M EUR)": "value",
                                                                         "Production value, published (M EUR)": "published_value",
                                                                         "Contribution, here": "contribution",
                                                                         "Contribution, published": "published_contribution"}),
                  "value_published_2006", {"Production value, here (M EUR)": 0, "Production value, published (M EUR)": 0,
                                           "Contribution, here": 2, "Contribution, published": 2})
    fd = by_country.filter(pl.col("year") == last).sort("flow")
    figures.dodged_bars(fd["geo"].to_list(), {"Actual flow": fd["flow"].to_list(), "Unmet demand": fd["unmet"].to_list()},
                        {"Actual flow": "#d97a3a", "Unmet demand": "#6b7f99"},
                        f"Million euro at constant 2010 prices, {last}", ctx.figure_path("pollination_value_country"))
