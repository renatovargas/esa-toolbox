"""Physical and monetary accounts of wood provision by country, and the results workbook by NUTS region.

Volumes are thousand m3.
"""

from __future__ import annotations

import polars as pl

from esa_toolbox import figures
from esa_toolbox.accounts import deflator
from esa_toolbox.core import Context, read_param
from esa_toolbox.nuts import eu27
from esa_toolbox.workbook import Workbook

YEARS_ALL = list(range(2000, 2025))
BASE_YEAR = 2010
COUNTRY_YEARS = list(range(2000, 2024))
EU_YEARS = list(range(2012, 2024))
TOOL_TIMBER_OUTLIERS = ["HR", "FI", "EL", "PT", "SE"]
TOOL_EXPORT_OUTLIERS = pl.DataFrame({"geo": ["DK", "EL"], "year": [2014, 2006]})
RATIO_BOUNDS = (1.0, 100.0)
EXPORT_FOLD = 3.0
SOURCE_LEVELS = ["reported", "fallback", "interpolated", "carried", "missing"]


def grid(geos: list[str], years: list[int]) -> pl.DataFrame:
    return pl.DataFrame({"geo": geos}).join(pl.DataFrame({"year": years}), how="cross")


def fill_series(primary: pl.DataFrame, fallback: pl.DataFrame, geos: list[str], years: list[int],
                value_col: str = "thousand_m3", fallback_label: str = "fallback") -> pl.DataFrame:
    """Reported values, then the fallback series, then the mean of the nearest earlier and later values of
    the country ("interpolated"), and at either end the nearest value ("carried")."""
    d = (grid(geos, years)
         .join(primary.select("geo", "year", primary=value_col), on=["geo", "year"], how="left")
         .join(fallback.select("geo", "year", fallback=value_col), on=["geo", "year"], how="left")
         .with_columns(first_pass=pl.coalesce("primary", "fallback"),
                       source=pl.when(pl.col("primary").is_not_null()).then(pl.lit("reported"))
                       .when(pl.col("fallback").is_not_null()).then(pl.lit(fallback_label)))
         .sort("geo", "year")
         .with_columns(before=pl.col("first_pass").forward_fill().over("geo"),
                       after=pl.col("first_pass").backward_fill().over("geo")))
    both = pl.col("before").is_not_null() & pl.col("after").is_not_null()
    d = d.with_columns(value=pl.when(pl.col("first_pass").is_not_null()).then(pl.col("first_pass"))
                       .when(both).then((pl.col("before") + pl.col("after")) / 2)
                       .otherwise(pl.coalesce("before", "after")))
    d = d.with_columns(source=pl.when(pl.col("source").is_not_null()).then(pl.col("source"))
                       .when(both).then(pl.lit("interpolated"))
                       .when(pl.col("value").is_not_null()).then(pl.lit("carried"))
                       .otherwise(pl.lit("missing")))
    return d.select("geo", "year", "value", "source")


def fill_with_means(x: pl.DataFrame, country_years: list[int], eu_years: list[int]) -> pl.DataFrame:
    """A country with values takes the mean of its own values, and one without the EU mean of yearly means."""
    cm = x.filter(pl.col("year").is_in(country_years)).group_by("geo").agg(country_mean=pl.col("value").mean())
    eu_mean = (x.filter(pl.col("year").is_in(eu_years)).group_by("year").agg(m=pl.col("value").mean())
               ["m"].mean())
    return (x.join(cm, on="geo", how="left")
            .with_columns(fill=pl.when(pl.col("value").is_not_null()).then(pl.lit("observed"))
                          .when(pl.col("country_mean").is_not_null()).then(pl.lit("country_mean"))
                          .otherwise(pl.lit("eu_mean")),
                          value=pl.coalesce("value", "country_mean", pl.lit(eu_mean)))
            .select("geo", "year", "value", "fill"))


def on_grid(x: pl.DataFrame, geos: list[str], years: list[int], col: str) -> pl.DataFrame:
    return grid(geos, years).join(x.select("geo", "year", value=col), on=["geo", "year"], how="left")


def accounts(ctx: Context) -> None:
    physical(ctx)
    monetary(ctx)


def physical(ctx: Context) -> None:
    members = eu27()
    rep = list(ctx.years)
    extract_all = pl.read_parquet(ctx.need("eurostat", "wood", "for_vol_efa_thsm3.parquet")).with_columns(
        pl.col("year").cast(pl.Int64))
    cbm = ctx.load("cbm_nai").with_columns(pl.col("year").cast(pl.Int64))
    annex = read_param("inca_part3_timber_annex.csv")
    vol = extract_all.filter(pl.col("geo").is_in(members))
    eu_aggregate = extract_all.filter(pl.col("geo") == "EU27_2020")

    def pick(flow: str, cls: str) -> pl.DataFrame:
        return vol.filter((pl.col("flow") == flow) & (pl.col("forest_class") == cls))

    no_supply = pl.DataFrame({"geo": ["MT"] * len(YEARS_ALL), "year": YEARS_ALL, "thousand_m3": [0.0] * len(YEARS_ALL)})
    cbm_total = pl.concat([cbm.select("geo", "year", thousand_m3="cbm_total_forest"), no_supply])
    nai = fill_series(pick("NAI", "FAWS"), cbm_total, members, YEARS_ALL, fallback_label="carbon_budget")
    nai = nai.with_columns(source=pl.when(pl.col("geo") == "MT").then(pl.lit("no_supply")).otherwise("source"))
    rem = fill_series(pick("RMOV", "FAWS"), pick("RMOV", "FOR"), members, YEARS_ALL)

    overuse = (nai.select("geo", "year", nai="value", nai_source="source")
               .join(rem.select("geo", "year", removals="value", removals_source="source"), on=["geo", "year"], how="left")
               .with_columns(balance=pl.col("nai") - pl.col("removals"),
                             removals_to_nai=pl.col("removals") / pl.col("nai"),
                             overuse=pl.col("removals") > pl.col("nai"),
                             overuse_volume=pl.max_horizontal(pl.col("removals") - pl.col("nai"), 0.0)))
    ctx.require(nai["value"].null_count() == 0 and rem["value"].null_count() == 0 and nai["value"].min() >= 0,
                "The increment and the removals are complete and the increment is never negative")
    acc = overuse.with_columns(sustainable=pl.min_horizontal("removals", "nai"))
    ctx.check(acc["sustainable"] + acc["overuse_volume"], acc["removals"],
              "Sustainable flow and overuse add up to removals")
    ctx.check(overuse.height, len(members) * len(YEARS_ALL), "One row per member state and year")
    ctx.save_checks("checks_physical")
    ctx.save(nai.rename({"value": "nai_faws_1000m3", "source": "nai_source"}), "wood_nai_faws_country_year")
    ctx.save(overuse, "wood_overuse_country_year")

    # Material of the report.
    latest = (nai.filter(pl.col("source").is_in(["reported", "carbon_budget"])).group_by("geo")
              .agg(last_year=pl.col("year").max()).group_by("last_year").len("Member States").sort("last_year"))
    ctx.table(latest.rename({"last_year": "Last year with an original value"}), "nai_latest_year", 0)
    cov = (vol.filter(pl.col("year").is_in(rep) & pl.col("forest_class").is_in(["FAWS", "FOR"]))
           .group_by("flow", "forest_class", "year").len().sort("year")
           .pivot(on="year", index=["flow", "forest_class"], values="len").fill_null(0)
           .sort("flow", "forest_class"))
    ctx.table(cov.rename({"flow": "Flow", "forest_class": "Forest class"}), "reporting_coverage", 0)
    for name, s in (("nai_sources", nai), ("removals_sources", rem)):
        t = (s.filter(pl.col("year").is_in(rep)).group_by("year", "source").len()
             .pivot(on="source", index="year", values="len").fill_null(0).sort("year"))
        ctx.table(t.rename({"year": "Year"}), name, 0)

    rules = {
        "Total forest reported to Eurostat": pick("NAI", "FOR"),
        "Carbon budget model, total forest": cbm_total,
        "Carbon budget model, scaled to available forest": pl.concat([
            cbm.select("geo", "year", thousand_m3=pl.col("cbm_total_forest") * pl.col("share_available")), no_supply]),
    }
    published = (annex.filter((pl.col("geo") != "UK") & pl.col("million_m3").is_not_null())
                 .select("geo", pl.col("year").cast(pl.Int64), published=pl.col("million_m3") * 1000))
    rc = []
    for rule, fb in rules.items():
        f = fill_series(pick("NAI", "FAWS"), fb, members, YEARS_ALL).join(published, on=["geo", "year"])
        rc.append(f.group_by("year").agg(
            eu_total=pl.col("value").sum() / 1000, published_total=pl.col("published").sum() / 1000,
            absolute_differences=(pl.col("value") - pl.col("published")).abs().sum() / 1000,
            median_difference_pct=100 * (pl.col("value") / pl.col("published") - 1).abs().median())
            .with_columns(rule=pl.lit(rule)).sort("year"))
    rc = pl.concat(rc).select("rule", "year", pl.exclude("rule", "year"))
    for key, rule in (("total_forest", "Total forest reported to Eurostat"), ("cbm_total", "Carbon budget model, total forest")):
        v = rc.filter(pl.col("rule") == rule).sort("year")["absolute_differences"].to_list()
        ctx.facts(**{f"rule_error_{key}": ", ".join(f"{x:,.1f}" for x in v)})
    fr = published.filter(pl.col("geo") == "FR")
    for y in (2000, 2012):
        v = fr.filter(pl.col("year") == y)["published"]
        if v.len():
            ctx.facts(**{f"published_fr_{y}": (v[0] / 1000, 1)})
    ctx.table(rc.rename({"rule": "Rule for the gaps", "year": "Year", "eu_total": "EU total (million m3)",
                         "published_total": "Published (million m3)",
                         "absolute_differences": "Sum of absolute differences (million m3)",
                         "median_difference_pct": "Median difference (%)"}), "gap_rule_comparison", 1)

    eu_series = (acc.filter(pl.col("year").is_in(rep)).group_by("year").agg(
        n_overuse=pl.col("overuse").sum(), overuse=pl.col("overuse_volume").sum() / 1000,
        sustainable=pl.col("sustainable").sum() / 1000, removals=pl.col("removals").sum() / 1000,
        nai=pl.col("nai").sum() / 1000).sort("year"))
    ctx.table(eu_series.select(Year="year", **{"Actual flow, NAI (million m3)": "nai",
                                               "Removals (million m3)": "removals",
                                               "Sustainable flow (million m3)": "sustainable",
                                               "Overuse (million m3)": "overuse",
                                               "Countries with overuse": "n_overuse"}), "eu_series", 1)
    for r in eu_series.iter_rows(named=True):
        y = r["year"]
        ctx.facts(**{f"nai_mm3_{y}": (r["nai"], 1), f"removals_mm3_{y}": (r["removals"], 1),
                     f"sustainable_mm3_{y}": (r["sustainable"], 1), f"overuse_mm3_{y}": (r["overuse"], 2),
                     f"n_overuse_{y}": str(r["n_overuse"])})
    agg = (eu_aggregate.filter((pl.col("flow") == "NAI") & pl.col("forest_class").is_in(["FAWS", "FOR"]) &
                               pl.col("year").is_in(rep))
           .select("year", "forest_class", m=pl.col("thousand_m3") / 1000)
           .pivot(on="forest_class", index="year", values="m"))
    agg = agg.rename({c: f"eurostat_{c}" for c in agg.columns if c != "year"})
    pt = published.group_by("year").agg(published=pl.col("published").sum() / 1000)
    ag = eu_series.select("year", accounts="nai").join(agg, on="year", how="left").join(pt, on="year", how="left").sort("year")
    ctx.table(ag.rename({"year": "Year", "accounts": "These accounts (million m3)"}), "nai_against_aggregates", 1)
    a12 = ag.filter(pl.col("year") == 2012)
    if a12.height and "eurostat_FAWS" in a12.columns:
        ctx.facts(nai_vs_eurostat_faws_2012=(100 * (a12["accounts"][0] / a12["eurostat_FAWS"][0] - 1), 1))

    nr = nai.filter(pl.col("year").is_in(rep)).sort("geo", "year").with_columns(
        previous_year=pl.col("year").shift(1).over("geo"), previous_value=pl.col("value").shift(1).over("geo"),
        previous_source=pl.col("source").shift(1).over("geo"))
    breaks = (nr.filter((pl.col("previous_source") == "carbon_budget") & (pl.col("source") == "reported"))
              .join(cbm.select("geo", "year", cbm_same_year="cbm_total_forest"), on=["geo", "year"], how="left")
              .select("geo", "previous_year", "year", "previous_value", "value", "cbm_same_year",
                      change_pct=100 * (pl.col("value") / pl.col("previous_value") - 1),
                      source_gap_pct=100 * (pl.col("value") / pl.col("cbm_same_year") - 1))
              .sort(pl.col("change_pct").abs(), descending=True))
    for g in ("FR", "SE"):
        v = breaks.filter(pl.col("geo") == g)["change_pct"]
        if v.len():
            ctx.facts(**{f"break_{g}_pct": (-v[0], 1)})
    ctx.table(breaks.rename({"geo": "Country", "previous_year": "From", "year": "To",
                             "previous_value": "NAI before (1000 m3)", "value": "NAI after (1000 m3)",
                             "cbm_same_year": "Carbon budget model, same year (1000 m3)",
                             "change_pct": "Change (%)", "source_gap_pct": "Gap between sources (%)"}),
              "source_breaks", 1)
    oc = (overuse.filter(pl.col("year").is_in(rep) & pl.col("overuse"))
          .select("year", "geo", "nai", "removals", "overuse_volume", "removals_to_nai", "nai_source", "removals_source")
          .sort(["year", "removals_to_nai"], descending=[False, True]))
    ctx.table(oc.rename({"year": "Year", "geo": "Country", "nai": "NAI (1000 m3)", "removals": "Removals (1000 m3)",
                         "overuse_volume": "Overuse (1000 m3)", "removals_to_nai": "Removals to NAI",
                         "nai_source": "Source of NAI", "removals_source": "Source of removals"}),
              "overuse_cases", {"Removals to NAI": 2, "NAI (1000 m3)": 0, "Removals (1000 m3)": 0,
                                "Overuse (1000 m3)": 0})
    ctx.facts(years_with_overuse=str(overuse.filter(pl.col("overuse"))["year"].n_unique()))

    fig = overuse.filter(pl.col("year").is_in(rep) & pl.col("removals_to_nai").is_finite())
    order = fig.group_by("geo").agg(m=pl.col("removals_to_nai").max()).sort("m")["geo"].to_list()
    panels = {}
    for y in rep:
        d = dict(zip(*fig.filter(pl.col("year") == y).select("geo", "removals_to_nai")))
        panels[str(y)] = [d.get(g, float("nan")) for g in order]
    figures.dots(order, panels, "Removals divided by net annual increment", ctx.figure_path("wood_removals_to_nai"),
                 vline=1.0)


def monetary(ctx: Context) -> None:
    members = eu27()
    rep = list(ctx.years)
    overuse = ctx.load("wood_overuse_country_year")
    nw = ctx.load("weight_by_nuts")
    annex = read_param("inca_part3_timber_annex.csv")
    hicp = pl.read_parquet(ctx.need("eurostat", "shared", "hicp_eu27.parquet"))
    exports = pl.read_parquet(ctx.need("eurostat", "wood", "for_basic_rw_exports.parquet")).with_columns(
        pl.col("year").cast(pl.Int64))
    nai = on_grid(ctx.load("wood_nai_faws_country_year"), members, YEARS_ALL, "nai_faws_1000m3")
    timber = on_grid(pl.read_parquet(ctx.need("eurostat", "wood", "for_sup_cp_p1tr.parquet"))
                     .with_columns(pl.col("year").cast(pl.Int64)), members, YEARS_ALL, "value_mio_eur")

    ex = (exports.filter(pl.col("geo").is_in(members))
          .with_columns(unit_value=pl.when(pl.col("exports_ths_m3") > 0)
                        .then(pl.col("exports_ths_eur") / pl.col("exports_ths_m3")))
          .with_columns(country_median=pl.col("unit_value").filter(pl.col("unit_value") > 0).median().over("geo"))
          .with_columns(relative=pl.col("unit_value") / pl.col("country_median"))
          .with_columns(removed=pl.col("unit_value").is_not_null() &
                        ((pl.col("relative") > EXPORT_FOLD) | (pl.col("relative") < 1 / EXPORT_FOLD)).fill_null(False)))
    removed = pl.concat([ex.filter(pl.col("removed")).select("geo", "year"), TOOL_EXPORT_OUTLIERS]).unique()
    uv_exports = on_grid(ex.join(removed, on=["geo", "year"], how="anti"), members, YEARS_ALL, "unit_value")

    def ratio_of(t: pl.DataFrame) -> pl.DataFrame:
        r = (nai.rename({"value": "nai"}).join(t.rename({"value": "timber"}), on=["geo", "year"], how="left")
             .join(uv_exports.rename({"value": "exports"}), on=["geo", "year"], how="left")
             .with_columns(nai_uv=pl.when(pl.col("nai") > 0).then(1000 * pl.col("timber") / pl.col("nai")))
             .with_columns(value=pl.col("exports") / pl.col("nai_uv")))
        return r.with_columns(value=pl.when(pl.col("value").is_finite()).then(pl.col("value")))

    def without(geos: list[str]) -> pl.DataFrame:
        return timber.with_columns(value=pl.when(pl.col("geo").is_in(geos)).then(None).otherwise(pl.col("value")))

    def screen_of(r: pl.DataFrame) -> pl.DataFrame:
        return (r.filter(pl.col("value").is_not_null()).group_by("geo")
                .agg(n_observed=pl.len(), median_ratio=pl.col("value").median(),
                     min_ratio=pl.col("value").min(), max_ratio=pl.col("value").max()))

    ratio_before = ratio_of(timber)
    screen = screen_of(ratio_of(without(TOOL_TIMBER_OUTLIERS)))
    screened_out = screen.filter((pl.col("median_ratio") < RATIO_BOUNDS[0]) |
                                 (pl.col("median_ratio") > RATIO_BOUNDS[1]))["geo"].sort().to_list()
    ratio_final = (ratio_of(without(TOOL_TIMBER_OUTLIERS + screened_out))
                   .with_columns(value=pl.when((pl.col("value") < RATIO_BOUNDS[0]) |
                                               (pl.col("value") > RATIO_BOUNDS[1])).then(None)
                                 .otherwise(pl.col("value")))
                   .select("geo", "year", "value"))
    uv_filled = fill_with_means(uv_exports, COUNTRY_YEARS, EU_YEARS)
    ratio_filled = fill_with_means(ratio_final, COUNTRY_YEARS, EU_YEARS)
    unit_value = (uv_filled.select("geo", "year", export_uv="value")
                  .join(ratio_filled.select("geo", "year", ratio="value", ratio_fill="fill"), on=["geo", "year"], how="left")
                  .with_columns(unit_value_eur_m3=pl.col("export_uv") / pl.col("ratio")))
    defl = deflator(hicp, BASE_YEAR)
    monetary = (nai.rename({"value": "nai_1000m3"}).join(unit_value, on=["geo", "year"], how="left")
                .with_columns(value_mio_eur_current=pl.col("nai_1000m3") * pl.col("unit_value_eur_m3") / 1000)
                .join(defl, on="year", how="left")
                .with_columns(value_mio_eur_constant=pl.col("value_mio_eur_current") * pl.col("deflator"))
                .select("geo", "year", "nai_1000m3", export_uv_eur_m3="export_uv", ratio_export_to_timber="ratio",
                        ratio_source="ratio_fill", unit_value_eur_m3="unit_value_eur_m3",
                        value_mio_eur_current="value_mio_eur_current", deflator="deflator",
                        value_mio_eur_constant="value_mio_eur_constant"))
    ctx.require(monetary["value_mio_eur_current"].null_count() == 0 and
                monetary["value_mio_eur_constant"].null_count() == 0 and monetary["unit_value_eur_m3"].min() > 0,
                "Every country and year has a positive unit value and a value")
    ctx.check(monetary["value_mio_eur_constant"] / monetary["deflator"], monetary["value_mio_eur_current"],
              "Constant prices return to current prices through the deflator")
    ctx.check(defl.filter(pl.col("year") == BASE_YEAR)["deflator"], 1, "The deflator equals one in the base year")
    ctx.save(monetary, "wood_value_country_year")

    # Workbook.
    national = (overuse.filter(pl.col("year").is_in(rep))
                .select("geo", "year", "nai", "removals", sustainable=pl.min_horizontal("removals", "nai"),
                        overuse="overuse_volume")
                .join(monetary.select("geo", "year", "unit_value_eur_m3", "deflator"), on=["geo", "year"], how="left")
                .with_columns(flow_nominal=pl.col("nai") * pl.col("unit_value_eur_m3") / 1000,
                              sustainable_nominal=pl.col("sustainable") * pl.col("unit_value_eur_m3") / 1000,
                              overuse_nominal=pl.col("overuse") * pl.col("unit_value_eur_m3") / 1000)
                .with_columns(flow_constant=pl.col("flow_nominal") * pl.col("deflator"),
                              sustainable_constant=pl.col("sustainable_nominal") * pl.col("deflator"),
                              overuse_constant=pl.col("overuse_nominal") * pl.col("deflator")))
    cols = ["nai", "removals", "sustainable", "overuse", "flow_nominal", "flow_constant", "sustainable_nominal",
            "sustainable_constant", "overuse_nominal", "overuse_constant"]
    regional = (nw.join(national, on=["geo", "year"])
                .with_columns([pl.col(c) * pl.col("share") for c in cols])
                .select("level", "nuts_id", "name", "geo", "year", *cols))
    ctx.save(regional, "wood_regional")
    tables = {
        "Flow_phys": ("Actual flow of wood provision in physical terms (thousand m3)", {"nai": "Net annual increment"}),
        "Overuse_phys": ("Overuse and sustainable flow of wood provision in physical terms (thousand m3)",
                         {"sustainable": "Sustainable flow", "overuse": "Overuse"}),
        "Flow_mon": ("Actual flow of wood provision in monetary terms (million euro)",
                     {"flow_nominal": "Current prices", "flow_constant": "Constant prices"}),
        "Overuse_mon": ("Overuse and sustainable flow of wood provision in monetary terms (million euro)",
                        {"sustainable_nominal": "Sustainable flow, current prices",
                         "sustainable_constant": "Sustainable flow, constant prices",
                         "overuse_nominal": "Overuse, current prices", "overuse_constant": "Overuse, constant prices"}),
    }
    tots = regional.group_by("level", "year").agg([pl.col(c).sum() for c in cols])
    n0 = tots.filter(pl.col("level") == "NUTS0")
    for lv in ("NUTS2", "NUTS3"):
        j = tots.filter(pl.col("level") == lv).join(n0, on="year", suffix="_n0")
        # The same test as the earlier workbook: an absolute gap within 1e-6 of the larger of 1 and the total.
        diff = max(abs(a - b) / max(1, abs(b)) for c in cols for a, b in zip(j[c], j[f"{c}_n0"]))
        ctx.check(diff, 0, f"{lv} regions add up to the countries in every table, metric and year",
                  tolerance=1e-6, relative=False)
    ctx.save_checks("checks_monetary")

    wb = Workbook(ctx.output("tables", "wood_provision_results.xlsx"),
                  "Wood provision ecosystem service accounts, EU27", "EU27", "Total")
    ys = ", ".join(map(str, rep))
    wb.readme([
        "Wood provision ecosystem service accounts, EU27",
        f"Reporting years {ys}. Constant prices use {BASE_YEAR} as the base year (Eurostat HICP, EU-27).",
        "One sheet per output table and NUTS level (NUTS0, NUTS2, NUTS3). Sheet names start with Flow or Overuse, then phys or mon.",
        "Actual flow is the net annual increment (NAI) of forest available for wood supply. Sustainable flow is the removals up to the NAI. Overuse is the removals above the NAI.",
        "National values come from Eurostat and the JRC carbon budget model estimates. NUTS2 and NUTS3 values share the national value in proportion to the weight of each region in the spatial key, so regions add up to the country.",
        "Monetary values are the physical values times the national unit value of NAI, in million euro.",
        "Eurostat holds few values for 2024, so most values of that year repeat earlier years. The report explains which values are reported, which are estimates and which repeat another year.",
    ])
    for nm, (title, ms) in tables.items():
        for lv in ("NUTS0", "NUTS2", "NUTS3"):
            wb.sheet(f"{nm}_{lv}", title, f"Level {lv}. Reporting years {ys}. Constant prices use {BASE_YEAR} as the base year.",
                     regional.filter(pl.col("level") == lv), ms)
    wb.close()

    # Material of the report.
    ctx.table(defl.filter(pl.col("year").is_in(rep)).select(Year="year", Deflator="deflator"), "deflators", 3)
    cov = pl.DataFrame({"Year": rep,
                        "Timber value": [timber.filter((pl.col("year") == y) & pl.col("value").is_not_null()).height for y in rep],
                        "Export unit value": [uv_exports.filter((pl.col("year") == y) & pl.col("value").is_not_null()).height for y in rep]})
    ctx.table(cov, "monetary_coverage", 0)
    rs = pl.concat([
        screen.with_columns(treatment=pl.when(pl.col("geo").is_in(screened_out)).then(pl.lit("All timber values removed"))
                            .otherwise(pl.lit("Kept"))),
        screen_of(ratio_before.filter(pl.col("geo").is_in(TOOL_TIMBER_OUTLIERS)))
        .with_columns(treatment=pl.lit("All timber values removed, as in the INCA tool")),
    ]).sort("median_ratio", descending=True)
    ctx.table(rs.rename({"geo": "Country", "n_observed": "Years observed", "median_ratio": "Median ratio",
                         "min_ratio": "Lowest", "max_ratio": "Highest", "treatment": "Treatment"}), "ratio_screen", 2)
    em = ratio_filled.filter((pl.col("year") == 2012) & (pl.col("fill") == "eu_mean")).height
    ctx.facts(ratio_eu_mean_2012=str(em), screened_count=str(rs.filter(pl.col("treatment") != "Kept").height))
    ctx.table(ratio_filled.filter(pl.col("year").is_in(rep)).group_by("year", "fill").len()
              .pivot(on="fill", index="year", values="len").fill_null(0).sort("year").rename({"year": "Year"}),
              "ratio_sources", 0)
    tool_pts = set(zip(TOOL_EXPORT_OUTLIERS["geo"], TOOL_EXPORT_OUTLIERS["year"]))
    epr = ex.filter((pl.col("geo") != "MT") & pl.col("unit_value").is_not_null()).filter(
        pl.col("removed") | pl.struct("geo", "year").map_elements(lambda s: (s["geo"], s["year"]) in tool_pts,
                                                                   return_dtype=pl.Boolean))
    pts = epr.select("geo", "year", "unit_value", "country_median")
    ctx.facts(export_points=str(pts.height), export_points_reporting=str(pts.filter(pl.col("year").is_in(rep)).height))
    for g, y in (("HR", 2012), ("RO", 2024)):
        r = pts.filter((pl.col("geo") == g) & (pl.col("year") == y))
        if r.height:
            ctx.facts(**{f"export_{g}_{y}": (r["unit_value"][0], 1), f"export_{g}_{y}_median": (r["country_median"][0], 1)})
    ctx.table(epr.select(Country="geo", Year="year", **{"Unit value (EUR per m3)": "unit_value",
                                                        "Median of the country": "country_median"}).sort("Country", "Year"),
              "export_points_removed", 1)
    eu_ratio = ratio_filled.filter(pl.col("fill") == "eu_mean")["value"]
    unclean = fill_with_means(ratio_before.select("geo", "year", "value"), COUNTRY_YEARS, EU_YEARS).filter(
        pl.col("fill") == "eu_mean")["value"]
    med = lambda g: ratio_before.filter(pl.col("geo") == g)["value"].median()  # noqa: E731
    ctx.facts(eu_ratio=(eu_ratio[0] if eu_ratio.len() else float("nan"), 2),
              median_screened=(ratio_final["value"].median(), 2),
              eu_ratio_uncleaned=(unclean[0] if unclean.len() else float("nan"), 2),
              uncleaned_hr_ratio=(med("HR"), 2), uncleaned_lt_ratio=(med("LT"), 2), uncleaned_be_ratio=(med("BE"), 2),
              screened_out=", ".join(screened_out) or "none")

    nv = (national.select("geo", "year", "unit_value_eur_m3", flow_current="flow_nominal", flow_constant="flow_constant",
                          sustainable_constant="sustainable_constant", overuse_constant="overuse_constant"))
    ctx.table(nv.filter(pl.col("year") == max(rep)).sort("flow_constant", descending=True).select(
        Country="geo", **{"Unit value (EUR per m3)": "unit_value_eur_m3", "Actual flow, current prices (M EUR)": "flow_current",
                          "Actual flow, 2010 prices (M EUR)": "flow_constant",
                          "Sustainable flow, 2010 prices (M EUR)": "sustainable_constant",
                          "Overuse, 2010 prices (M EUR)": "overuse_constant"}), "value_by_country", 1)
    app = nv.filter(pl.col("year").is_in([2000, 2012, 2024])).join(
        national.select("geo", "year", "nai"), on=["geo", "year"])
    wide = None
    for y in (2000, 2012, 2024):
        part = app.filter(pl.col("year") == y).select("geo", **{f"NAI {y} (1000 m3)": "nai", f"Value {y} (M EUR 2010)": "flow_constant"})
        wide = part if wide is None else wide.join(part, on="geo", how="full", coalesce=True)
    if wide is not None:
        ctx.table(wide.sort("geo").rename({"geo": "Country"}), "appendix_by_country", 0)
    veu = nv.group_by("year").agg(pl.col("flow_current", "flow_constant", "sustainable_constant", "overuse_constant").sum()).sort("year")
    ctx.table(veu.select(Year="year", **{"Actual flow, current prices (M EUR)": "flow_current",
                                         "Actual flow, 2010 prices (M EUR)": "flow_constant",
                                         "Sustainable flow, 2010 prices (M EUR)": "sustainable_constant",
                                         "Overuse, 2010 prices (M EUR)": "overuse_constant"}), "value_eu_series", 1)
    for r in veu.iter_rows(named=True):
        y = r["year"]
        ctx.facts(**{f"value_constant_bn_{y}": (r["flow_constant"] / 1000, 1),
                     f"value_current_bn_{y}": (r["flow_current"] / 1000, 2),
                     f"overuse_constant_meur_{y}": (r["overuse_constant"], 1)})
    uvr = (nv.filter(pl.col("geo") != "MT").group_by("year").agg(
        minimum=pl.col("unit_value_eur_m3").min(), median=pl.col("unit_value_eur_m3").median(),
        maximum=pl.col("unit_value_eur_m3").max(),
        highest=pl.col("geo").sort_by("unit_value_eur_m3").last()).sort("year"))
    ctx.table(uvr.rename({"year": "Year", "minimum": "Lowest (EUR per m3)", "median": "Median (EUR per m3)",
                          "maximum": "Highest (EUR per m3)", "highest": "Highest in"}), "unit_value_range", 1)
    pub = (annex.filter((pl.col("geo") != "UK") & pl.col("million_eur").is_not_null())
           .select("geo", pl.col("year").cast(pl.Int64), published="million_eur"))
    ap = nv.join(pub, on=["geo", "year"]).with_columns(difference_pct=100 * (pl.col("flow_current") / pl.col("published") - 1))
    ctx.table(ap.group_by("year").agg(accounts=pl.col("flow_current").sum(), published=pl.col("published").sum(),
                                      median_difference_pct=pl.col("difference_pct").abs().median()).sort("year")
              .rename({"year": "Year", "accounts": "These accounts (M EUR)", "published": "INCA Part III (M EUR)",
                       "median_difference_pct": "Median difference by country (%)"}), "value_against_published_eu", 1)

    fig = monetary.filter(pl.col("year").is_in(rep) & (pl.col("geo") != "MT"))
    order = fig.group_by("geo").agg(m=pl.col("unit_value_eur_m3").median()).sort("m")["geo"].to_list()
    col = {"observed": "#2f6b4f", "country_mean": "#d97a3a", "eu_mean": "#6b7f99"}
    panels, colours = {}, {}
    for y in rep:
        d = {g: (v, s) for g, v, s in fig.filter(pl.col("year") == y).select("geo", "unit_value_eur_m3", "ratio_source").iter_rows()}
        panels[str(y)] = [d.get(g, (float("nan"), "observed"))[0] for g in order]
        colours[str(y)] = [col[d.get(g, (0, "observed"))[1]] for g in order]
    figures.dots(order, panels, "Unit value of the net annual increment (euro per m3, current prices)",
                 ctx.figure_path("wood_unit_value"), colours=colours)
