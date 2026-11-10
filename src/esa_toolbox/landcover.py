"""Corine Land Cover accounting layers: reading, and their reduction to 1 km shares of groups of classes."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from esa_toolbox.grid import Grid
from esa_toolbox.rio import read

CLC_NODATA = 9999


def clc_file(store: Path, year: int) -> Path:
    p = store / "clc_accounting" / "v21" / f"CLC{year}ACC_V2024_21.tif"
    if not p.exists():
        from esa_toolbox.core import EsaError

        raise EsaError(f"Corine layer missing: {p}. Run `esa fetch` for clc_accounting.")
    return p


def lookup(groups: dict[str, list[int]]) -> np.ndarray:
    """Code to group index. Codes in no group, and no-data, map to len(groups)."""
    lut = np.full(65536, len(groups), dtype="uint8")
    for i, codes in enumerate(groups.values()):
        lut[codes] = i
    return lut


def shares_1km(store: Path, year: int, grid_1km: Grid, groups: dict[str, list[int]],
               strip_km: int = 200) -> np.ndarray:
    """Share of the 100 cells of each 1 km cell that fall in each group, as float32 (groups, rows, cols).

    A cell with no data counts as outside every group, so each share is a whole number of cells over 100,
    which is what `aggregate(clc %in% codes, 10, "mean")` gives in terra.
    """
    lut = lookup(groups)
    g = len(groups)
    out = np.zeros((g, *grid_1km.shape), dtype="float32")
    f100 = grid_1km.with_res(100)
    path = clc_file(store, year)
    for top in range(0, grid_1km.height, strip_km):
        rows = min(strip_km, grid_1km.height - top)
        ymax = grid_1km.ymax - top * 1000
        strip = f100.sub(grid_1km.xmin, ymax - rows * 1000, grid_1km.xmax, ymax)
        codes = read(path, strip, fill=CLC_NODATA)
        b = lut[codes].reshape(rows, 10, grid_1km.width, 10)
        for k in range(g):
            out[k, top:top + rows, :] = ((b == k).sum(axis=(1, 3), dtype="int16") / 100.0).astype("float32")
    return out
