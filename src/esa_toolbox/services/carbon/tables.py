"""Physical and monetary accounts of carbon sequestration, by country and year, and by NUTS region.

Values are thousand tonnes of CO2, with uptake positive.
"""

from __future__ import annotations

import json

import polars as pl

from esa_toolbox import figures
from esa_toolbox.accounts import CO2_TO_C, deflator
from esa_toolbox.core import YEARS, Context
from esa_toolbox.nuts import eu27
from esa_toolbox.workbook import Workbook

YEARS_ALL = list(range(2000, 2025))
BASE_YEAR = 2010
LULUCF_LAND = {"CRF4A": "Forest land", "CRF4B": "Cropland", "CRF4C": "Grassland",
               "CRF4D": "Wetlands", "CRF4E": "Settlements", "CRF4F": "Other land"}
LULUCF_MEMO = {"CRF4G": "Harvested wood products", "CRF4H": "Other"}
LAND_TYPE_OF = {"CRF4A": "forest", "CRF4B": "cropland", "CRF4C": "grassland", "CRF4D": "wetlands",
                "CRF4E": "settlements", "CRF4F": "other_land"}
SECTORS = {
    "CRF1A1": "Energy supply", "CRF1B": "Energy supply", "CRF1C": "Energy supply",
    "CRF1A2": "Manufacturing and construction", "CRF2": "Manufacturing and construction",
    "CRF1A3": "Transport", "CRF1A4C": "Agriculture, forestry and fishing",
    "CRF3": "Agriculture, forestry and fishing", "CRF1A4B": "Households",
    "CRF1A4A": "Services and other", "CRF1A5": "Services and other", "CRF6": "Services and other",
    "CRF_INDCO2": "Services and other", "CRF5": "Waste",
}
COLOURS = {"Removals": "#2a78d6", "Flow without offset": "#2a78d6", "Emissions": "#eb6834",
           "Missed flow": "#eb6834", "Actual flow": "#1baf7a"}

RESULT_TABLES = {
    "Flow_phys": ("Actual flow of carbon sequestration with offset in physical terms (thousand tonnes)",
                  {"flow": "CO2", "flow_c": "Carbon"}),
    "NoOffset_phys": ("Flow of carbon sequestration without offset in physical terms (thousand tonnes)",
                      {"flow_no_offset": "CO2", "flow_no_offset_c": "Carbon"}),
    "Missed_phys": ("Missed flow of carbon sequestration in physical terms (thousand tonnes)",
                    {"missed_flow": "CO2", "missed_flow_c": "Carbon"}),
    "Flow_mon": ("Actual flow of carbon sequestration with offset in monetary terms (million euro)",
                 {"flow_nominal": "Current prices", "flow_constant": "Constant prices"}),
    "Missed_mon": ("Missed flow of carbon sequestration in monetary terms (million euro)",
                   {"missed_nominal": "Current prices", "missed_constant": "Constant prices"}),
}


def signed_balance(env: pl.DataFrame, codes: dict[str, str], geos: list[str]) -> pl.DataFrame:
    """Balance by category, country and year with uptake positive. A missing category is zero, and flagged."""
    grid = (pl.DataFrame({"geo": geos}).join(pl.DataFrame({"year": YEARS_ALL}), how="cross")
            .join(pl.DataFrame({"src_crf": list(codes)}), how="cross"))
    return (grid.join(env.filter(pl.col("src_crf").is_in(list(codes))).select("geo", "year", "src_crf",
                                                                           "thousand_t_co2"),
                      on=["geo", "year", "src_crf"], how="left")
            .with_columns(gap=pl.col("thousand_t_co2").is_null(),
                          balance=-pl.col("thousand_t_co2").fill_null(0.0),
                          category=pl.col("src_crf").replace_strict(codes))
            .with_columns(removals=pl.max_horizontal(pl.col("balance"), 0.0),
                          emissions=pl.max_horizontal(-pl.col("balance"), 0.0))
            .drop("thousand_t_co2"))


def allocate_offset(land: pl.DataFrame) -> pl.DataFrame:
    """EU net flow, shared among countries by their removals of forest land (La Notte et al., 2021, Eq. 6.1)."""
    by = land.group_by("geo", "year").agg(
        forest_removals=pl.col("removals").filter(pl.col("src_crf") == "CRF4A").sum(),
        total_removals=pl.col("removals").sum(), total_emissions=pl.col("emissions").sum(),
        national_balance=pl.col("balance").sum())
    y = pl.col("year")
    return (by.with_columns(eu_removals=pl.col("total_removals").sum().over(y),
                            eu_emissions=pl.col("total_emissions").sum().over(y),
                            forest_share=pl.col("forest_removals") / pl.col("forest_removals").sum().over(y))
            .with_columns(eu_net=pl.max_horizontal(pl.col("eu_removals") - pl.col("eu_emissions"), 0.0))
            .with_columns(flow=pl.col("eu_net") * pl.col("forest_share"),
                          flow_no_offset=pl.col("total_removals"))
            .with_columns(missed_flow=pl.col("flow_no_offset") - pl.col("flow"))
            .sort("geo", "year"))


def accounts(ctx: Context) -> None:
    raw = pl.read_parquet(ctx.need("eurostat", "carbon", "env_air_gge_co2.parquet")).with_columns(
        pl.col("year").cast(pl.Int64))
    meta = json.loads((ctx.store / "eurostat" / "carbon" / "extracts.json").read_text()) \
        if (ctx.store / "eurostat" / "carbon" / "extracts.json").exists() else {}
    retrieved = (meta.get("env_air_gge_co2") or {}).get("retrieved") or "unknown"
    members = eu27()
    eu_aggregate = raw.filter(pl.col("geo") == "EU27_2020")
    env = raw.filter(pl.col("geo").is_in(members))

    lulucf = signed_balance(env, {**LULUCF_LAND, **LULUCF_MEMO}, members)
    land = lulucf.filter(pl.col("src_crf").is_in(list(LULUCF_LAND)))

    parts = list(LULUCF_LAND) + list(LULUCF_MEMO) + ["CRF4"]
    eu_check = (env.filter(pl.col("src_crf").is_in(parts))
                .group_by("src_crf", "year").agg(countries=pl.col("thousand_t_co2").sum())
                .join(eu_aggregate.select("src_crf", "year", aggregate="thousand_t_co2"),
                      on=["src_crf", "year"], how="left"))

    flow = allocate_offset(land).with_columns(
        flow_c=pl.col("flow") * CO2_TO_C, flow_no_offset_c=pl.col("flow_no_offset") * CO2_TO_C,
        missed_flow_c=pl.col("missed_flow") * CO2_TO_C)
    flow_eu = flow.group_by("year").agg(
        removals=pl.col("total_removals").sum(), emissions=pl.col("total_emissions").sum(),
        net_flow=pl.col("flow").sum(), missed_flow=pl.col("missed_flow").sum(),
        forest_removals=pl.col("forest_removals").sum()).sort("year")

    sec = pl.DataFrame({"src_crf": list(SECTORS), "sector": list(SECTORS.values())})
    sectors = (env.join(sec, on="src_crf").group_by("geo", "year", "sector")
               .agg(sector_emissions=pl.col("thousand_t_co2").sum())
               .with_columns(share=pl.col("sector_emissions") / pl.col("sector_emissions").sum().over("geo", "year"))
               .join(flow.select("geo", "year", "flow"), on=["geo", "year"])
               .with_columns(allocated_flow=pl.col("flow") * pl.col("share")))
    sector_totals = (sectors.group_by("year").agg(sum_of_sectors=pl.col("sector_emissions").sum())
                     .join(env.filter(pl.col("src_crf") == "TOTX4_MEMO").group_by("year")
                           .agg(totx4=pl.col("thousand_t_co2").sum()), on="year", how="left"))

    # Checks, which stop the run on failure.
    ctx.require(flow["flow"].min() >= 0 and flow["missed_flow"].min() >= -1e-9,
                "Actual flow and missed flow are never negative")
    ft = flow.group_by("year").agg(s=pl.col("forest_share").sum())
    tot = flow.group_by("year").agg(f=pl.col("flow").sum(), net=pl.col("eu_net").first())
    mt = flow.filter(pl.col("eu_net") > 0).group_by("year").agg(m=pl.col("missed_flow").sum(),
                                                                 e=pl.col("eu_emissions").first())
    st = sectors.group_by("geo", "year").agg(s=pl.col("share").sum())
    ctx.check(ft["s"], 1, "Shares of the forest removals add up to one in every year")
    ctx.check(tot["f"], tot["net"], "National flows add up to the EU net flow in every year")
    ctx.check(mt["m"], mt["e"], "Missed flow adds up to the EU emissions in every year")
    ctx.check(flow["flow_no_offset"], flow["flow"] + flow["missed_flow"],
              "Flow without offset equals actual flow plus missed flow", tolerance=1e-9, relative=False)
    ctx.check(st["s"], 1, "Sector shares add up to one in every country and year")
    ctx.check(sector_totals["sum_of_sectors"], sector_totals["totx4"],
              "Sectors add up to total CO2 emissions without land use", tolerance=5e-3)
    ctx.check(eu_check["countries"], eu_check["aggregate"].cast(pl.Float64),
              "Countries add up to the EU27 aggregate of Eurostat, in kt CO2", tolerance=0.1, relative=False)
    ctx.save_checks("checks_physical")

    lulucf_out = lulucf.select(
        "geo", "year", "src_crf", "category", in_flow=pl.col("src_crf").is_in(list(LULUCF_LAND)),
        balance_kt_co2="balance", removals_kt_co2="removals", emissions_kt_co2="emissions",
        balance_kt_c=pl.col("balance") * CO2_TO_C, missing_value="gap")
    ctx.save(lulucf_out, "carbon_lulucf_country_year")
    flow_out = flow.select(
        "geo", "year", forest_removals_kt_co2="forest_removals", removals_kt_co2="total_removals",
        emissions_kt_co2="total_emissions", national_balance_kt_co2="national_balance",
        forest_share="forest_share", eu_net_kt_co2="eu_net", flow_kt_co2="flow", flow_kt_c="flow_c",
        flow_no_offset_kt_co2="flow_no_offset", flow_no_offset_kt_c="flow_no_offset_c",
        missed_flow_kt_co2="missed_flow", missed_flow_kt_c="missed_flow_c")
    ctx.save(flow_out, "carbon_flow_country_year")
    ctx.save(sectors.select("geo", "year", "sector", sector_emissions_kt_co2="sector_emissions", share="share",
                            allocated_flow_kt_co2="allocated_flow"), "carbon_sector_allocation_country_year")

    # Material of the report.
    tw = ctx.load("type_weights")
    rep = list(ctx.years)
    ctx.facts(countries=str(len(members)),
              nuts2=str(tw.filter(pl.col("level") == "NUTS2")["nuts_id"].n_unique()),
              nuts3=str(tw.filter(pl.col("level") == "NUTS3")["nuts_id"].n_unique()),
              inventory_retrieved=retrieved)
    gaps = lulucf.filter(pl.col("gap")).group_by("src_crf", "geo").len("years_missing").sort("src_crf", "geo")
    ctx.table(gaps, "category_gaps", 0)
    dk = gaps.filter((pl.col("geo") == "DK") & (pl.col("src_crf") == "CRF4F"))
    ctx.facts(dk_other_land_gap_years=str(dk["years_missing"][0] if dk.height else 0),
              residual_reporters=str(env.filter((pl.col("src_crf") == "CRF4H") & pl.col("year").is_in(YEARS_ALL))
                                     ["geo"].n_unique()))
    ctx.table(tw.filter(pl.col("fallback")).unique(["epoch", "geo", "land_type"])
              .group_by("geo", "land_type").len("epochs").sort("geo", "land_type"), "nuts_fallback", 0)
    fb = sorted(tw.filter(pl.col("fallback"))["geo"].unique().to_list())
    ctx.facts(fallback_geos=" and ".join(fb) if len(fb) <= 2 else ", ".join(fb[:-1]) + " and " + fb[-1])

    bal = (lulucf.filter(pl.col("year").is_in(rep)).group_by("year", "src_crf", "category")
           .agg(balance_mt=pl.col("balance").sum() / 1000).sort("src_crf", "year"))
    ctx.table(bal.pivot(on="year", index=["src_crf", "category"], values="balance_mt")
              .rename({"src_crf": "Code", "category": "Category"}), "balance_by_category", 1)
    for code in ("CRF4A", "CRF4G"):
        for y in rep:
            v = bal.filter((pl.col("src_crf") == code) & (pl.col("year") == y))["balance_mt"]
            if v.len():
                ctx.facts(**{f"balance_{code}_{y}": (v[0], 1)})
    ctx.table(land.filter(pl.col("year").is_in(rep) & (pl.col("src_crf") == "CRF4A") & (pl.col("balance") < 0))
              .select(Year="year", Country="geo", **{"Balance of forest land (kt CO2)": "balance"})
              .sort("Year", "Balance of forest land (kt CO2)"), "forest_sources", 0)
    crf4 = (env.filter(pl.col("year").is_in(rep) & pl.col("src_crf").is_in(parts))
            .with_columns(part=pl.when(pl.col("src_crf") == "CRF4").then(pl.lit("total")).otherwise(pl.lit("parts")))
            .group_by("year", "part").agg(v=pl.col("thousand_t_co2").sum())
            .pivot(on="part", index="year", values="v").sort("year")
            .with_columns(difference=pl.col("total") - pl.col("parts")))
    ctx.table(crf4, "crf4_check", 1)
    ctx.facts(crf4_max_diff=(float(crf4["difference"].abs().max()), 2))

    eu_rep = flow_eu.filter(pl.col("year").is_in(rep))
    ctx.table(eu_rep.select(Year="year", **{"Removals (Mt CO2)": pl.col("removals") / 1000,
                                            "Emissions (Mt CO2)": pl.col("emissions") / 1000,
                                            "Actual flow (Mt CO2)": pl.col("net_flow") / 1000,
                                            "Missed flow (Mt CO2)": pl.col("missed_flow") / 1000}),
              "eu_series", 1)
    for r in flow_eu.iter_rows(named=True):
        y = r["year"]
        ctx.facts(**{f"removals_mt_{y}": (r["removals"] / 1000, 1), f"emissions_mt_{y}": (r["emissions"] / 1000, 1),
                     f"flow_mt_{y}": (r["net_flow"] / 1000, 1), f"missed_mt_{y}": (r["missed_flow"] / 1000, 1),
                     f"forest_removals_mt_{y}": (r["forest_removals"] / 1000, 1)})
    pub = pl.DataFrame({"year": [2000, 2006, 2012], "published_mt": [291.554, 292.213, 306.308]})
    ctx.table(flow_eu.join(pub, on="year").select(
        Year="year", **{"This report (Mt CO2)": pl.col("net_flow") / 1000,
                        "La Notte et al. (2021) (Mt CO2)": pl.col("published_mt"),
                        "Difference (%)": 100 * (pl.col("net_flow") / 1000 / pl.col("published_mt") - 1)}),
              "published_flow", 1)

    fc = flow.filter(pl.col("year").is_in(rep))
    ctx.table(fc.filter(pl.col("year") == max(rep)).sort("flow", descending=True).select(
        Country="geo", **{"Actual flow (kt CO2)": "flow", "Flow without offset (kt CO2)": "flow_no_offset",
                          "Missed flow (kt CO2)": "missed_flow", "Net balance of land (kt CO2)": "national_balance",
                          "Share of forest removals (%)": pl.col("forest_share") * 100}),
              "flow_by_country", {"Share of forest removals (%)": 1, "Actual flow (kt CO2)": 0,
                                  "Flow without offset (kt CO2)": 0, "Missed flow (kt CO2)": 0,
                                  "Net balance of land (kt CO2)": 0})
    last = fc.filter(pl.col("year") == max(rep)).sort("flow", descending=True)
    ctx.facts(top5_share=(100 * last["flow"].head(5).sum() / last["flow"].sum(), 1),
              top5_countries=", ".join(last["geo"].head(5).to_list()))
    counts = fc.group_by("year").agg(no_forest_removals=(pl.col("forest_removals") == 0).sum(),
                                     negative_balance=(pl.col("national_balance") < 0).sum()).sort("year")
    ctx.table(counts.rename({"year": "Year", "no_forest_removals": "Countries without forest removals",
                             "negative_balance": "Countries with a negative balance of land"}), "country_counts", 0)
    sy = (sectors.filter(pl.col("year").is_in(rep)).group_by("year", "sector")
          .agg(allocated_mt=pl.col("allocated_flow").sum() / 1000).sort("sector", "year"))
    ctx.table(sy.pivot(on="year", index="sector", values="allocated_mt").rename({"sector": "Sector"}),
              "sector_by_year", 1)
    top = sy.filter(pl.col("year") == max(rep)).sort("allocated_mt", descending=True)
    ctx.facts(top_sector=top["sector"][0], top_sector_mt=(top["allocated_mt"][0], 1))
    sg = (env.group_by("geo", "year").agg(
        parts=pl.col("thousand_t_co2").filter(pl.col("src_crf").is_in(list(SECTORS))).sum(),
        total=pl.col("thousand_t_co2").filter(pl.col("src_crf") == "TOTX4_MEMO").sum())
        .filter(pl.col("year").is_in(YEARS_ALL) & ((pl.col("parts") - pl.col("total")).abs() > 1)))
    ctx.facts(sector_gap_cases=str(sg.height))
    if sg.height:
        g0 = sg.sort("geo", "year").row(0, named=True)
        ctx.facts(sector_gap_geo=g0["geo"], sector_gap_year=str(g0["year"]),
                  sector_gap_mt=((g0["total"] - g0["parts"]) / 1000, 1))

    figures.lines({"Removals": (eu_rep["year"].to_list(), (eu_rep["removals"] / 1000).to_list()),
                   "Emissions": (eu_rep["year"].to_list(), (eu_rep["emissions"] / 1000).to_list()),
                   "Actual flow": (eu_rep["year"].to_list(), (eu_rep["net_flow"] / 1000).to_list())},
                  COLOURS, "Million tonnes of CO2", ctx.figure_path("carbon_eu_series"))
    c24 = fc.filter(pl.col("year") == max(rep)).sort("flow")
    figures.dodged_bars(c24["geo"].to_list(),
                        {"Actual flow": (c24["flow"] / 1000).to_list(),
                         "Flow without offset": (c24["flow_no_offset"] / 1000).to_list()},
                        {"Actual flow": COLOURS["Actual flow"], "Flow without offset": COLOURS["Removals"]},
                        f"Million tonnes of CO2, {max(rep)}", ctx.figure_path("carbon_flow_country"))
    allocation_comparison(ctx, flow, lulucf, members)
    monetary(ctx, flow_out, lulucf_out, tw)


def allocation_comparison(ctx: Context, flow: pl.DataFrame, lulucf: pl.DataFrame, members: list[str]) -> None:
    """The key of Equation 6.1 against a compared key, the net annual increment of all forest, with harvested
    wood products deducted from the removals of forest land. It is the finding behind the choice of key."""
    from esa_toolbox.services.wood.tables import fill_series

    vol = pl.read_parquet(ctx.need("eurostat", "wood", "for_vol_efa_thsm3.parquet")).with_columns(pl.col("year").cast(pl.Int64))
    nai = vol.filter((pl.col("flow") == "NAI") & (pl.col("forest_class") == "FOR") & pl.col("geo").is_in(members))
    nai = fill_series(nai, nai.head(0), members, YEARS_ALL).with_columns(
        nai_share=pl.col("value") / pl.col("value").sum().over("year")).select("geo", "year", "nai_share")
    hwp = lulucf.filter(pl.col("src_crf") == "CRF4G").select("geo", "year", hwp_sink="removals")
    c = (flow.select("geo", "year", "flow", "missed_flow", "total_removals", "total_emissions", "forest_removals")
         .join(hwp, on=["geo", "year"], how="left").join(nai, on=["geo", "year"], how="left")
         .with_columns(removals_earlier=pl.col("total_removals") - pl.col("forest_removals")
                       + pl.max_horizontal(pl.col("forest_removals") - pl.col("hwp_sink").fill_null(0.0), 0.0)))
    y = pl.col("year")
    c = (c.with_columns(net_earlier=pl.max_horizontal(pl.col("removals_earlier").sum().over(y)
                                                      - pl.col("total_emissions").sum().over(y), 0.0))
         .with_columns(flow_earlier=pl.col("net_earlier") * pl.col("nai_share"))
         .with_columns(missed_earlier=pl.col("removals_earlier") - pl.col("flow_earlier")))
    ctx.save(c, "carbon_allocation_comparison")
    rep = c.filter(pl.col("year").is_in(list(ctx.years)))
    t = rep.group_by("year").agg(
        a=pl.col("flow").sum() / 1000, b=pl.col("flow_earlier").sum() / 1000, h=pl.col("hwp_sink").sum() / 1000,
        n=(pl.col("missed_flow") < -1e-9).sum(), ne=(pl.col("missed_earlier") < -1e-9).sum()).sort("year")
    ctx.table(t.rename({"year": "Year", "a": "Actual flow, this method", "b": "Actual flow, compared key",
                        "h": "Harvested wood products sink", "n": "Negative missed flow, this method",
                        "ne": "Negative missed flow, compared key"}), "allocation_comparison", 1)
    de = rep.filter((pl.col("geo") == "DE") & (pl.col("year") == max(ctx.years)))
    ctx.facts(compared_de_flow_last=(float(de["flow_earlier"][0]) / 1000 if de.height else float("nan"), 1),
              compared_negative_min=str(int(t["ne"].min())), compared_negative_max=str(int(t["ne"].max())))


def monetary(ctx: Context, flow: pl.DataFrame, lulucf: pl.DataFrame, tw: pl.DataFrame) -> None:
    price = ctx.load("carbon_price")
    hicp = pl.read_parquet(ctx.need("eurostat", "shared", "hicp_eu27.parquet"))
    defl = deflator(hicp, BASE_YEAR)
    value = (flow.select("geo", "year", "flow_kt_co2", "flow_no_offset_kt_co2", "missed_flow_kt_co2")
             .join(price.select("year", "price_eur_per_t_co2"), on="year", how="left")
             .join(defl, on="year", how="left")
             .with_columns(value_current_meur=pl.col("flow_kt_co2") * pl.col("price_eur_per_t_co2") / 1000,
                           value_no_offset_current_meur=pl.col("flow_no_offset_kt_co2") * pl.col("price_eur_per_t_co2") / 1000,
                           missed_value_current_meur=pl.col("missed_flow_kt_co2") * pl.col("price_eur_per_t_co2") / 1000)
             .with_columns(value_constant_meur=pl.col("value_current_meur") * pl.col("deflator"),
                           value_no_offset_constant_meur=pl.col("value_no_offset_current_meur") * pl.col("deflator"),
                           missed_value_constant_meur=pl.col("missed_value_current_meur") * pl.col("deflator")))
    ctx.save(value, "carbon_value_country_year")

    rem = (lulucf.filter(pl.col("in_flow"))
           .with_columns(land_type=pl.col("src_crf").replace_strict(LAND_TYPE_OF))
           .select("geo", "year", "land_type", removals="removals_kt_co2"))
    national = (flow.filter(pl.col("year").is_in(list(ctx.years)))
                .select("geo", "year", flow="flow_kt_co2")
                .join(value.select("geo", "year", price="price_eur_per_t_co2", deflator="deflator"),
                      on=["geo", "year"]))
    # Each reporting year uses the Corine layer of the same year, so the epoch equals the year.
    no_offset = (tw.join(rem.rename({"year": "epoch"}), on=["geo", "epoch", "land_type"])
                 .group_by("level", "nuts_id", "name", "geo", "epoch")
                 .agg(flow_no_offset=(pl.col("removals") * pl.col("share")).sum()))
    actual = (tw.filter(pl.col("land_type") == "forest")
              .join(national.rename({"year": "epoch"}), on=["geo", "epoch"])
              .select("level", "nuts_id", "name", "geo", "epoch", "price", "deflator",
                      flow=pl.col("flow") * pl.col("share")))
    regional = (no_offset.join(actual, on=["level", "nuts_id", "name", "geo", "epoch"])
                .rename({"epoch": "year"})
                .with_columns(missed_flow=pl.col("flow_no_offset") - pl.col("flow"))
                .with_columns(flow_c=pl.col("flow") * CO2_TO_C, flow_no_offset_c=pl.col("flow_no_offset") * CO2_TO_C,
                              missed_flow_c=pl.col("missed_flow") * CO2_TO_C,
                              flow_nominal=pl.col("flow") * pl.col("price") / 1000,
                              missed_nominal=pl.col("missed_flow") * pl.col("price") / 1000)
                .with_columns(flow_constant=pl.col("flow_nominal") * pl.col("deflator"),
                              missed_constant=pl.col("missed_nominal") * pl.col("deflator"))
                .sort("level", "nuts_id", "year"))
    ctx.save(regional, "carbon_regional")

    metrics = [m for _, ms in RESULT_TABLES.values() for m in ms]
    tots = regional.group_by("level", "year").agg([pl.col(m).sum() for m in metrics])
    n0 = tots.filter(pl.col("level") == "NUTS0").drop("level")
    reg_t, nat_t = [], []
    for lv in ("NUTS2", "NUTS3"):
        j = tots.filter(pl.col("level") == lv).drop("level").join(n0, on="year", suffix="_n0")
        for m in metrics:
            reg_t += j[m].to_list()
            nat_t += j[f"{m}_n0"].to_list()
    ctx.require(value["value_current_meur"].min() >= 0 and price["price_eur_per_t_co2"].min() > 0,
                "Values and prices are positive")
    ctx.check(defl.filter(pl.col("year") == BASE_YEAR)["deflator"], 1,
              "The deflator equals one in the base year", tolerance=1e-12)
    ctx.check(value["value_constant_meur"], value["value_current_meur"] * value["deflator"],
              "Constant values equal current values times the deflator", tolerance=1e-9, relative=False)
    ctx.check(value["value_no_offset_current_meur"],
              value["value_current_meur"] + value["missed_value_current_meur"],
              "Actual value plus missed value equals the value without offset", tolerance=1e-9, relative=False)
    ctx.check(reg_t, nat_t, "NUTS2 and NUTS3 regions add up to the countries, in every table, metric and year")
    ctx.save_checks("checks_monetary")

    years = list(ctx.years)
    wb = Workbook(ctx.output("tables", "carbon_sequestration_results.xlsx"),
                  "Carbon sequestration accounts", "EU27", "Total")
    wb.readme([
        "Carbon sequestration ecosystem service accounts, global climate regulation",
        f"Reporting years {', '.join(map(str, years))}. Constant prices use {BASE_YEAR} as the base year (Eurostat HICP, EU-27).",
        "Sheets are named Flow, NoOffset or Missed, then phys or mon, then the NUTS level (NUTS0, NUTS2, NUTS3).",
        "Flow without offset is the total CO2 removals of forest land, cropland, grassland, wetlands, settlements and other land.",
        "Actual flow is the EU total removals minus the EU total emissions of the same land categories, attributed to countries in proportion to their removals of forest land.",
        "Missed flow is the flow without offset minus the actual flow, which is the part of the removals that compensates the emissions of land use.",
        "National values come from the greenhouse gas inventories in Eurostat dataset env_air_gge (CO2, thousand tonnes). Carbon is CO2 times 12/44.",
        "NUTS2 and NUTS3 values share the removals of each land type in proportion to the area of the land type in each region, and the actual flow in proportion to the forest weight of each region, so regions add up to the country.",
        "Monetary values are the CO2 values times the social cost of carbon in euro per tonne, in million euro.",
        "The 100 m maps of the flow without offset and of the actual flow in physical and monetary terms are distributed separately.",
    ])
    for nm, (title, ms) in RESULT_TABLES.items():
        for lv in ("NUTS0", "NUTS2", "NUTS3"):
            wb.sheet(f"{nm}_{lv}", title, f"Level {lv}. Reporting years {', '.join(map(str, years))}. "
                     f"Constant prices use {BASE_YEAR} as the base year.", regional.filter(pl.col("level") == lv), ms)
    wb.close()

    rep = years
    pr = price.filter(pl.col("year").is_in(rep))
    ctx.table(pr.select(Year="year", **{"Price, 2023 euros (EUR per t CO2)": "price_eur_per_t_co2_2023",
                                        "Price, current euros (EUR per t CO2)": "price_eur_per_t_co2"}),
              "carbon_price", 2)
    ctx.facts(price_2023=(pr["price_eur_per_t_co2_2023"][0], 1))
    for r in pr.iter_rows(named=True):
        ctx.facts(**{f"price_current_{r['year']}": (r["price_eur_per_t_co2"], 1)})
    ctx.table(defl.filter(pl.col("year").is_in(rep + [BASE_YEAR])).sort("year").select(
        Year="year", Deflator="deflator"), "deflators", 3)
    veu = value.group_by("year").agg(
        flow_mt=pl.col("flow_kt_co2").sum() / 1000, current=pl.col("value_current_meur").sum(),
        constant=pl.col("value_constant_meur").sum(), missed_constant=pl.col("missed_value_constant_meur").sum(),
        no_offset_constant=pl.col("value_no_offset_constant_meur").sum()).sort("year")
    vrep = veu.filter(pl.col("year").is_in(rep))
    ctx.table(vrep.select(Year="year", **{"Actual flow (Mt CO2)": "flow_mt", "Value, current prices (M EUR)": "current",
                                          "Value, 2010 prices (M EUR)": "constant",
                                          "Missed value, 2010 prices (M EUR)": "missed_constant",
                                          "Value without offset, 2010 prices (M EUR)": "no_offset_constant"}),
              "value_eu_series", 1)
    for r in veu.iter_rows(named=True):
        y = r["year"]
        ctx.facts(**{f"value_constant_bn_{y}": (r["constant"] / 1000, 1),
                     f"missed_constant_bn_{y}": (r["missed_constant"] / 1000, 1),
                     f"value_current_bn_{y}": (r["current"] / 1000, 1)})
    b12 = value.filter(pl.col("year") == 2012)
    ctx.facts(price_implied_2012=(b12["value_current_meur"].sum() / b12["flow_kt_co2"].sum() * 1000, 1),
              price_published_2012=(9189 / 306308 * 1000, 1))
    ctx.table(regional.filter((pl.col("year") == max(rep)) & (pl.col("level") == "NUTS2"))
              .sort("flow", descending=True).head(10)
              .select(Code="nuts_id", Region="name", **{"Actual flow (kt CO2)": "flow",
                                                       "Flow without offset (kt CO2)": "flow_no_offset",
                                                       "Missed flow (kt CO2)": "missed_flow",
                                                       "Value, 2010 prices (M EUR)": "flow_constant"}),
              "top_regions", 1)
    ctx.table(value.filter(pl.col("year") == max(rep)).sort("value_constant_meur", descending=True).select(
        Country="geo", **{"Value of actual flow, 2010 prices (M EUR)": "value_constant_meur",
                          "Missed value, 2010 prices (M EUR)": "missed_value_constant_meur",
                          "Value of actual flow, current prices (M EUR)": "value_current_meur"}),
              "value_by_country", 1)
    figures.lines({"Actual flow": (vrep["year"].to_list(), vrep["constant"].to_list()),
                   "Missed flow": (vrep["year"].to_list(), vrep["missed_constant"].to_list())},
                  COLOURS, "Million euro, constant 2010 prices", ctx.figure_path("carbon_value_series"))


__all__ = ["accounts", "YEARS"]
