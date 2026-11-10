"""Grids, windows and tiles on the EPSG:3035 Corine grid.

A grid is an axis-aligned box with a cell size. All grids of the tool share the Corine origin, so a 1 km cell
always holds exactly 10 by 10 cells of 100 m when both grids come from the same box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from affine import Affine
from rasterio.crs import CRS

CRS_3035 = CRS.from_epsg(3035)


@dataclass(frozen=True)
class Grid:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    res: float

    @property
    def width(self) -> int:
        return round((self.xmax - self.xmin) / self.res)

    @property
    def height(self) -> int:
        return round((self.ymax - self.ymin) / self.res)

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def transform(self) -> Affine:
        return Affine(self.res, 0, self.xmin, 0, -self.res, self.ymax)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.xmin, self.ymin, self.xmax, self.ymax

    def with_res(self, res: float) -> "Grid":
        return Grid(self.xmin, self.ymin, self.xmax, self.ymax, res)

    def sub(self, xmin: float, ymin: float, xmax: float, ymax: float) -> "Grid":
        return Grid(xmin, ymin, xmax, ymax, self.res)

    def buffered(self, distance: float, limit: "Grid | None" = None) -> "Grid":
        """The box grown by `distance` on each side, kept inside `limit` when given."""
        g = Grid(self.xmin - distance, self.ymin - distance, self.xmax + distance,
                 self.ymax + distance, self.res)
        if limit is not None:
            g = Grid(max(g.xmin, limit.xmin), max(g.ymin, limit.ymin), min(g.xmax, limit.xmax),
                     min(g.ymax, limit.ymax), self.res)
        return g

    def offset_in(self, outer: "Grid") -> tuple[int, int]:
        """Row and column of the top left cell of this grid inside `outer`, which has the same cell size."""
        return round((outer.ymax - self.ymax) / self.res), round((self.xmin - outer.xmin) / self.res)

    def tiles(self, cells: int) -> Iterator["Grid"]:
        """Tiles of `cells` by `cells`, from the lower left corner, clipped to the grid."""
        size = cells * self.res
        y = self.ymin
        while y < self.ymax - 1e-6:
            x = self.xmin
            while x < self.xmax - 1e-6:
                yield Grid(x, y, min(x + size, self.xmax), min(y + size, self.ymax), self.res)
                x += size
            y += size

    @property
    def name(self) -> str:
        return f"{int(self.xmin)}_{int(self.ymin)}"


# Analysis window of the pollination accounts, the European territory of the EU27 (Corine grid).
POLLINATION_GRID = Grid(2630000, 1380000, 7320000, 5420000, 1000)
# Grid of the wood and carbon accounts. Its 1 km cells are offset from the pollination grid, an item recorded
# as an open decision. The grids stay as they were, so the results match the earlier accounts.
FOREST_GRID = Grid(2636000, 1385900, 7317000, 5415900, 1000)
