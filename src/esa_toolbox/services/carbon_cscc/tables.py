"""The national flows of carbon sequestration valued at the country-level social cost of carbon (CSCC).

The CSCC of a country is the damage that one more tonne of CO2 emitted in 2020 causes in that country alone, in US
dollars of 2005 (Ricke et al., 2018). The prices follow the steps of the carbon accounts: the US consumer price index
to 2023, the exchange rate of 2023 to euros, the EU27 HICP to the euros of each year, and the deflator to 2010 prices.
"""

from __future__ import annotations

import json

import polars as pl

from esa_toolbox.accounts import deflator
from esa_toolbox.core import Context, EsaError, read_param
from esa_toolbox.workbook import Workbook

BASE_YEAR = 2010
PRICE_YEAR = 2023
CPI_2005, CPI_2023 = 195.30, 304.70  # US CPI-U, annual average (BLS)
USD_PER_EUR_2023 = 1.0813  # ECB reference rate, annual average of 2023
USD2005_TO_EUR2023 = CPI_2023 / CPI_2005 / USD_PER_EUR_2023
QUANTILES = {"low": "16.7 percent", "median": "median", "high": "83.3 percent"}


def carbon_table(ctx: Context, name: str) -> pl.DataFrame:
    p = ctx.settings.work / ctx.settings.folder("carbon") / "tables" / f"{name}.parquet"
    if not p.exists():
        raise EsaError(f"{p} is missing. Run `esa run carbon` first, since this valuation reads its national flows.")
    return pl.read_parquet(p)


def carbon_run_date(ctx: Context) -> str:
    f = ctx.settings.work / ctx.settings.folder("carbon") / "stages" / "accounts.json"
    return json.loads(f.read_text())["finished"][:10] if f.exists() else ""


def prices(ctx: Context, countries: pl.DataFrame) -> pl.DataFrame:
    """CSCC of each EU27 country in US dollars of 2005 and in euros of 2023, per tonne of CO2. Null where the database
    has no value for the country."""
    cscc = pl.read_parquet(ctx.need("cscc", "cscc_reference_case.parquet"))
    return (countries.join(cscc, on="iso3", how="left")
            .with_columns(**{f"eur2023_{q}": pl.col(f"cscc_{q}") * USD2005_TO_EUR2023 for q in QUANTILES})
            .sort("geo"))


def accounts(ctx: Context) -> None:
    countries = read_param("eu27_countries.csv")
    price = prices(ctx, countries)
    world = pl.read_parquet(ctx.need("cscc", "cscc_reference_case.parquet")).filter(pl.col("iso3") == "WLD")
    ctx.require(world.height == 1, "The database holds the global social cost of carbon (WLD)")
    hicp = pl.read_parquet(ctx.need("eurostat", "shared", "hicp_eu27.parquet"))
    h23 = hicp.filter(pl.col("year") == PRICE_YEAR)["hicp_2015"][0]
    defl = deflator(hicp, BASE_YEAR)

    flows = carbon_table(ctx, "carbon_value_country_year").filter(pl.col("year").is_in(list(ctx.years)))
    v = (flows.select("geo", "year", "flow_kt_co2", "missed_flow_kt_co2",
                      global_constant_meur="value_constant_meur", global_missed_constant_meur="missed_value_constant_meur")
         .join(price.select("geo", "country", *[f"eur2023_{q}" for q in QUANTILES]), on="geo", how="left")
         .join(hicp.select(pl.col("year").cast(pl.Int64), index=pl.col("hicp_2015") / h23), on="year", how="left")
         .join(defl, on="year", how="left"))
    for q in QUANTILES:
        p_t = pl.col(f"eur2023_{q}") * pl.col("index")  # euro of year t per tonne of CO2
        v = v.with_columns((pl.col("flow_kt_co2") * p_t / 1000).alias(f"value_current_{q}"),
                           (pl.col("missed_flow_kt_co2") * p_t / 1000).alias(f"missed_current_{q}"))
        v = v.with_columns((pl.col(f"value_current_{q}") * pl.col("deflator")).alias(f"value_constant_{q}"),
                           (pl.col(f"missed_current_{q}") * pl.col("deflator")).alias(f"missed_constant_{q}"))
    v = v.sort("geo", "year")
    ctx.save(v, "cscc_value_country_year")

    # Checks
    valued = v.filter(pl.col("eur2023_median").is_not_null())
    ctx.require(flows["geo"].n_unique() == 27, "The national flows cover the 27 Member States")
    ctx.require(((valued["eur2023_low"] <= valued["eur2023_median"]) &
                 (valued["eur2023_median"] <= valued["eur2023_high"])).all(),
                "The median CSCC lies between the low and the high quantile in every country")
    ctx.check(valued["value_constant_median"], valued["flow_kt_co2"] * valued["eur2023_median"] / 1000 *
              valued["index"] * valued["deflator"],
              "Constant values equal the flow times the price of the year times the deflator", tolerance=1e-9,
              relative=False)
    ctx.check(defl.filter(pl.col("year") == BASE_YEAR)["deflator"], 1, "The deflator equals one in the base year",
              tolerance=1e-12)
    ctx.check(valued["value_current_median"], valued["flow_kt_co2"] * valued["eur2023_median"] * valued["index"] / 1000,
              "Current values equal the flow times the price of the year", tolerance=1e-9, relative=False)
    ctx.save_checks("checks_monetary")

    workbook(ctx, v, price)
    report(ctx, v, price, world)


def workbook(ctx: Context, v: pl.DataFrame, price: pl.DataFrame) -> None:
    years = list(ctx.years)
    wb = Workbook(ctx.output("tables", "carbon_sequestration_cscc_results.xlsx"),
                  "Carbon sequestration valued at country-level social costs of carbon", "EU27",
                  "Total of the countries with a value")
    wb.readme([
        "Carbon sequestration ecosystem service accounts, valued at the country-level social cost of carbon",
        f"Reporting years {', '.join(map(str, years))}. Constant prices use {BASE_YEAR} as the base year (Eurostat HICP, EU-27).",
        "The physical flows are those of the carbon sequestration accounts, in thousand tonnes of CO2, and they are not changed here.",
        "The country-level social cost of carbon (CSCC) of a country is the damage that one more tonne of CO2 causes in that "
        "country alone. It comes from Ricke et al. (2018), Country-level social cost of carbon, Nature Climate Change 8, 895-900, "
        "database version 2, for the reference case (BHM short-run damage function, SSP2 and RCP 6.0, growth-adjusted "
        "discounting with a pure rate of time preference of 2 percent and an elasticity of marginal utility of 1.5).",
        "The CSCC is negative in many countries, where the model expects one more tonne of CO2 to raise output. The value of "
        "sequestration is then negative too.",
        "Low and high values use the 16.7 and 83.3 percent quantiles of the CSCC, and central values its median.",
        "Malta has no CSCC in the database, so it has no value. The total row adds up the countries with a value.",
        f"The CSCC in US dollars of 2005 is moved to 2023 with the US consumer price index ({CPI_2023:.2f}/{CPI_2005:.2f}), "
        f"to euros at {USD_PER_EUR_2023} US dollars per euro, and to the euros of each year with the EU27 HICP.",
        "These values are a separate valuation. The carbon sequestration accounts value the same flows at a global social cost of carbon.",
    ])
    wb.table("Prices", "Country-level social cost of carbon, per tonne of CO2",
             "US dollars of 2005 as published, and euros of 2023. Emissions of 2020.",
             price.select(Code="geo", Country="country",
                          **{"Low, USD 2005": "cscc_low", "Median, USD 2005": "cscc_median", "High, USD 2005": "cscc_high",
                             "Low, EUR 2023": "eur2023_low", "Median, EUR 2023": "eur2023_median",
                             "High, EUR 2023": "eur2023_high"}))
    long = v.rename({"geo": "nuts_id", "country": "name"})
    for nm, title, kind in (("Flow_mon", "Actual flow of carbon sequestration in monetary terms (million euro)", "value"),
                            ("Missed_mon", "Missed flow of carbon sequestration in monetary terms (million euro)", "missed")):
        for prices_of, label in (("constant", f"Constant prices of {BASE_YEAR}"), ("current", "Current prices")):
            wb.sheet(f"{nm}_{prices_of}_NUTS0", title, f"Level NUTS0. {label}. Low, central and high values of the CSCC.",
                     long, {f"{kind}_{prices_of}_low": "Low", f"{kind}_{prices_of}_median": "Central",
                            f"{kind}_{prices_of}_high": "High"})
    wb.close()


def report(ctx: Context, v: pl.DataFrame, price: pl.DataFrame, world: pl.DataFrame) -> None:
    years = list(ctx.years)
    first, last = years[0], years[-1]
    valued = price.filter(pl.col("cscc_median").is_not_null())
    missing = price.filter(pl.col("cscc_median").is_null())["country"].to_list()
    w = world.row(0, named=True)
    ctx.facts(cscc_countries=str(valued.height), cscc_missing=", ".join(missing) or "none",
              cscc_negative=str(int((valued["cscc_median"] < 0).sum())),
              cscc_positive=str(int((valued["cscc_median"] > 0).sum())),
              cscc_factor=(USD2005_TO_EUR2023, 4), cscc_world_usd=(w["cscc_median"], 1),
              cscc_world_low_usd=(w["cscc_low"], 1), cscc_world_high_usd=(w["cscc_high"], 1),
              cscc_world_eur=(w["cscc_median"] * USD2005_TO_EUR2023, 1),
              cscc_eu_sum_eur=(valued["eur2023_median"].sum(), 1),
              cscc_min_country=valued.sort("cscc_median")["country"][0],
              cscc_min_eur=(valued.sort("cscc_median")["eur2023_median"][0], 2),
              cscc_max_country=valued.sort("cscc_median")["country"][-1],
              cscc_max_eur=(valued.sort("cscc_median")["eur2023_median"][-1], 2),
              carbon_run_date=carbon_run_date(ctx), first_year=str(first), last_year=str(last))
    ctx.table(price.select(Country="country", **{"Median, USD 2005": "cscc_median", "Low, EUR 2023": "eur2023_low",
                                                 "Median, EUR 2023": "eur2023_median", "High, EUR 2023": "eur2023_high"}),
              "cscc_prices", 2)

    eu = (v.group_by("year").agg(pl.col("flow_kt_co2").sum() / 1000, pl.col("global_constant_meur").sum(),
                                 *[pl.col(f"value_constant_{q}").sum() for q in QUANTILES],
                                 *[pl.col(f"missed_constant_{q}").sum() for q in QUANTILES]).sort("year"))
    ctx.table(eu.select(Year="year", **{"Actual flow (Mt CO2)": "flow_kt_co2",
                                        "Low (M EUR)": "value_constant_low", "Central (M EUR)": "value_constant_median",
                                        "High (M EUR)": "value_constant_high",
                                        "Central, missed flow (M EUR)": "missed_constant_median",
                                        "Global SCC of the accounts (M EUR)": "global_constant_meur"}),
              "cscc_eu_series", 1)
    for r in eu.iter_rows(named=True):
        y = r["year"]
        ctx.facts(**{f"cscc_value_{q}_{y}": (r[f"value_constant_{q}"], 1) for q in QUANTILES},
                  **{f"cscc_global_bn_{y}": (r["global_constant_meur"] / 1000, 1),
                     f"cscc_missed_median_{y}": (r["missed_constant_median"], 1)})
    lastv = v.filter(pl.col("year") == last).sort("value_constant_median", nulls_last=True)
    ctx.table(lastv.select(Country="country", **{"Actual flow (kt CO2)": "flow_kt_co2",
                                                 "Low (M EUR)": "value_constant_low",
                                                 "Central (M EUR)": "value_constant_median",
                                                 "High (M EUR)": "value_constant_high",
                                                 "Global SCC of the accounts (M EUR)": "global_constant_meur"}),
              "cscc_value_by_country", 1)
    neg = lastv.filter(pl.col("value_constant_median") < 0)
    ctx.facts(cscc_negative_value_countries=str(neg.height),
              cscc_negative_value_meur=(neg["value_constant_median"].sum(), 1),
              cscc_positive_value_meur=(lastv.filter(pl.col("value_constant_median") > 0)["value_constant_median"].sum(), 1))
