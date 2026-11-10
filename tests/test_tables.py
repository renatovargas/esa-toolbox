"""Table rules of wood and carbon."""
import polars as pl

from esa_toolbox.services.carbon.tables import allocate_offset
from esa_toolbox.services.wood.tables import fill_series, fill_with_means


def test_fill_series_order_of_rules():
    primary = pl.DataFrame({"geo": ["A", "A"], "year": [2000, 2003], "thousand_m3": [10.0, 20.0]})
    fallback = pl.DataFrame({"geo": ["A"], "year": [2001], "thousand_m3": [99.0]})
    out = fill_series(primary, fallback, ["A"], [2000, 2001, 2002, 2003, 2004]).sort("year")
    assert out["value"].to_list() == [10, 99, 59.5, 20, 20]
    assert out["source"].to_list() == ["reported", "fallback", "interpolated", "reported", "carried"]


def test_fill_with_means():
    x = pl.DataFrame({"geo": ["A", "A", "B", "B"], "year": [2012, 2013, 2012, 2013], "value": [1.0, None, None, None]})
    out = fill_with_means(x, [2012, 2013], [2012, 2013]).sort("geo", "year")
    assert out["value"].to_list() == [1, 1, 1, 1]
    assert out["fill"].to_list() == ["observed", "country_mean", "eu_mean", "eu_mean"]


def test_offset_shares_eu_net_by_forest_removals():
    land = pl.DataFrame({
        "geo": ["A", "A", "B", "B"], "year": [2000] * 4, "src_crf": ["CRF4A", "CRF4B", "CRF4A", "CRF4B"],
        "balance": [30.0, -10.0, 10.0, 0.0]}).with_columns(
        removals=pl.max_horizontal("balance", 0.0), emissions=pl.max_horizontal(-pl.col("balance"), 0.0))
    f = allocate_offset(land).sort("geo")
    assert f["flow"].to_list() == [22.5, 7.5]
    assert f["missed_flow"].sum() == 10
