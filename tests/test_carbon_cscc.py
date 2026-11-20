"""Valuation of carbon sequestration at country-level social costs of carbon."""
import openpyxl
import polars as pl
import pytest

from esa_toolbox.core import SERVICES, Context, Settings, read_param
from esa_toolbox.data import fetch
from esa_toolbox.data.catalog import Dataset
from esa_toolbox.services.carbon_cscc import tables
from esa_toolbox.workbook import Workbook


def test_service_registered_after_carbon():
    assert SERVICES.index("carbon_cscc") > SERVICES.index("carbon")


def test_every_member_state_has_an_iso3_code():
    t = read_param("eu27_countries.csv")
    assert t.height == 27 and t["iso3"].n_unique() == 27 and t["iso3"].str.len_chars().eq(3).all()
    assert dict(t.select("geo", "iso3").iter_rows())["EL"] == "GRC"


def test_conversion_factor():
    assert tables.USD2005_TO_EUR2023 == pytest.approx(304.70 / 195.30 / 1.0813)
    assert 1.44 < tables.USD2005_TO_EUR2023 < 1.45


def test_workbook_table_has_no_total_row(tmp_path):
    wb = Workbook(tmp_path / "w.xlsx", "Heading")
    wb.table("Prices", "Title", "Note", pl.DataFrame({"Code": ["AT", "MT"], "Country": ["Austria", "Malta"],
                                                      "Median": [-1.5, None]}))
    wb.close()
    ws = openpyxl.load_workbook(tmp_path / "w.xlsx")["Prices"]
    assert ws["A5"].value == "Code" and ws["C6"].value == -1.5 and ws["C7"].value is None and ws["A8"].value is None


def test_fetch_keeps_the_reference_case(tmp_path, monkeypatch):
    rows = ("run,dmgfuncpar,climate,SSP,RCP,N,ISO3,prtp,eta,dr,16.7%,50%,83.3%\n"
            "bhm_sr,bootstrap,uncertain,SSP2,rcp60,1000,DEU,2,1p5,NA,-8,-5,-1\n"
            "bhm_sr,bootstrap,uncertain,SSP2,rcp60,1000,WLD,2,1p5,NA,177,417,806\n"
            "bhm_sr,bootstrap,uncertain,SSP2,rcp60,1000,DEU,NA,NA,3,1,2,3\n"
            "bhm_lr,bootstrap,uncertain,SSP2,rcp60,1000,DEU,2,1p5,NA,9,9,9\n")

    def fake_get(url, dest, log=print, timeout=0):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(rows)
        return dest

    monkeypatch.setattr(fetch, "_get", fake_get)
    d = Dataset("cscc", "t", "p", "s", "cscc", ("cscc/cscc_reference_case.parquet",), ("carbon_cscc",))
    fetch.fetch_cscc(d, tmp_path, log=lambda *_: None)
    t = pl.read_parquet(tmp_path / "cscc" / "cscc_reference_case.parquet")
    assert t["iso3"].to_list() == ["DEU", "WLD"] and t["cscc_median"].to_list() == [-5.0, 417.0]
    assert not (tmp_path / "cscc" / "_download").exists()


def test_valuation_needs_the_carbon_accounts(tmp_path):
    ctx = Context("carbon_cscc", Settings.load(tmp_path / "none.toml", store=tmp_path, work=tmp_path, outputs=tmp_path))
    with pytest.raises(Exception, match="esa run carbon"):
        tables.carbon_table(ctx, "carbon_value_country_year")
