"""Logic shared by the accounts of several services: deflators, regional shares and the spreading of national
values over a map with a weight, which keeps the national total."""

from __future__ import annotations

import numpy as np
import polars as pl

from esa_toolbox.grid import Grid
from esa_toolbox.rio import zonal_sum

CO2_TO_C = 12 / 44


def deflator(hicp: pl.DataFrame, base_year: int = 2010) -> pl.DataFrame:
    """Real value (prices of `base_year`) = nominal value * deflator."""
    base = hicp.filter(pl.col("year") == base_year)["hicp_2015"]
    if base.len() != 1:
        raise ValueError(f"The price index has no single value for {base_year}")
    return hicp.select(pl.col("year").cast(pl.Int64), (base[0] / pl.col("hicp_2015")).alias("deflator"))


def per_weight_factors(ids_geo: list[str], weight_sums: np.ndarray, totals: dict[str, float],
                       zero_when_no_weight: bool = False) -> np.ndarray:
    """Factor per country id (index 0 unused): national total divided by the national weight.

    A country without weight, or without a total, gets NaN (no data in the map), or 0 when asked, which
    is how the earlier 100 m wood maps treated it.
    """
    f = np.full(len(ids_geo) + 1, np.nan)
    for i, geo in enumerate(ids_geo, start=1):
        t = totals.get(geo)
        w = weight_sums[i - 1]
        if t is None or t != t:
            continue
        if w > 0:
            f[i] = t / w
        elif zero_when_no_weight:
            f[i] = 0.0
    return f


def country_weight_sums(weight: np.ndarray, ids: np.ndarray, n: int) -> np.ndarray:
    return zonal_sum(weight, ids, n)


def spread(weight: np.ndarray, ids: np.ndarray, factors: np.ndarray) -> np.ndarray:
    """Each cell takes weight times the factor of its country. Cells outside every country are NaN."""
    f = factors[np.clip(ids, 0, len(factors) - 1)]
    f[ids <= 0] = np.nan
    return weight * f


def grid_extent(grid: Grid) -> tuple[float, float, float, float]:
    return grid.xmin, grid.xmax, grid.ymin, grid.ymax
