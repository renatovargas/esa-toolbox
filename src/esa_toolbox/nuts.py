"""NUTS 2024 boundaries of the EU27, as polygons, region tables and rasterised country ids."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl
import pyogrio
import shapely

from esa_toolbox.core import read_param
from esa_toolbox.grid import Grid
from esa_toolbox.rio import rasterize

NUTS_FILE = ("nuts", "NUTS_RG_01M_2024_3035_LEVL0123_INCA.gpkg")
LAYER = "NUTS_RG_01M_2024_3035.gpkg"


def eu27() -> list[str]:
    return sorted(read_param("eu27_countries.csv")["geo"].to_list())


@dataclass
class Regions:
    """NUTS regions of levels 0, 2 and 3 in the EU27, with their polygons."""

    table: pl.DataFrame  # level, nuts_id, name, geo, levl
    geoms: np.ndarray  # shapely polygons, in the order of `table`

    def level(self, levl: int) -> "Regions":
        mask = (self.table["levl"] == levl).to_numpy()
        return Regions(self.table.filter(pl.Series(mask)), self.geoms[mask])

    def ids(self, grid: Grid, start: int = 1) -> np.ndarray:
        """Raster of row numbers (from `start`) of the regions, 0 outside. Cell centres decide."""
        box = shapely.box(*grid.bounds)
        keep = shapely.intersects(self.geoms, box)
        shapes = [(g, i + start) for i, (g, k) in enumerate(zip(self.geoms, keep)) if k]
        return rasterize(shapes, grid, dtype="int32")


@lru_cache(maxsize=4)
def _read(path: str) -> tuple[pl.DataFrame, np.ndarray]:
    _, _, wkb, fields = pyogrio.raw.read(path, layer=LAYER,
                                         columns=["NUTS_ID", "LEVL_CODE", "CNTR_CODE", "NAME_LATN"])
    table = pl.DataFrame({"nuts_id": fields[0], "levl": fields[1].astype("int64"),
                          "cntr": fields[2], "name": fields[3]})
    return table, shapely.from_wkb(wkb)


def regions(store: Path, levels: tuple[int, ...] = (0, 2, 3)) -> Regions:
    """Regions of the EU27 by level and code, the order the earlier routines used (`arrange(level, nuts_id)`)."""
    table, geoms = _read(str(store.joinpath(*NUTS_FILE)))
    members = set(eu27())
    t = table.with_row_index("row").filter(
        pl.col("levl").is_in(levels) & pl.col("nuts_id").str.slice(0, 2).is_in(members)
    ).with_columns(
        level=pl.format("NUTS{}", pl.col("levl")), geo=pl.col("nuts_id").str.slice(0, 2)
    ).sort("level", "nuts_id")
    rows = t["row"].to_numpy()
    return Regions(t.select("level", "nuts_id", "name", "geo", "levl"), geoms[rows])


def countries(store: Path) -> Regions:
    """The 27 Member States, sorted by code. Their position plus one is the country id of the rasters."""
    return regions(store, levels=(0,))


def sum_by_region(values: np.ndarray, grid: Grid, regs: Regions) -> np.ndarray:
    """Sum of a raster over each region, level by level, with cell centres deciding (terra `extract`)."""
    out = np.zeros(regs.table.height)
    for lv in sorted(set(regs.table["levl"].to_list())):
        mask = (regs.table["levl"] == lv).to_numpy()
        idx = np.flatnonzero(mask)
        ids = Regions(regs.table.filter(pl.Series(mask)), regs.geoms[mask]).ids(grid)
        v = np.asarray(values, dtype="float64")
        if v.ndim == 2:
            from esa_toolbox.rio import zonal_sum

            out[idx] = zonal_sum(v, ids, len(idx))
        else:
            raise ValueError("values must be 2D")
    return out
