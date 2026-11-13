"""Mode of elevation in each 10 km cell, from the Copernicus DEM GLO-90.

The package carries the result, `params/elevation_mode_10km.tif`, because elevation does not change between
the years and the build reads about 4 GB of tiles. This module is the record of how the file was made:

    python -m esa_toolbox.data.elevation <output.tif>

Each 3 arc-second cell of the DEM is rounded to a whole metre and assigned to the 10 km cell that holds its
centre. Cells of exactly 0 m are the sea of the DEM and are left out. Ties take the lowest elevation.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from esa_toolbox.grid import EU_GRID

SOURCE = "https://copernicus-dem-90m.s3.amazonaws.com"
GRID = EU_GRID.with_res(10000)
OFFSET, SPAN = 1000, 10000  # elevations from -1000 m to 8999 m


def tile_names() -> list[str]:
    from rasterio.warp import transform

    xs = np.arange(GRID.xmin, GRID.xmax + 1, 5000.0)
    ys = np.arange(GRID.ymin, GRID.ymax + 1, 5000.0)
    X, Y = np.meshgrid(xs, ys)
    lon, lat = transform("EPSG:3035", "EPSG:4326", X.ravel(), Y.ravel())
    cells = sorted(set(zip(np.floor(lat).astype(int), np.floor(lon).astype(int))))
    out = []
    for la, lo in cells:
        ns, ew = ("N", la) if la >= 0 else ("S", -la), ("E", lo) if lo >= 0 else ("W", -lo)
        out.append(f"Copernicus_DSM_COG_30_{ns[0]}{ns[1]:02d}_00_{ew[0]}{ew[1]:03d}_00_DEM")
    return out


def tile_counts(name: str) -> np.ndarray | None:
    """Distinct (10 km cell, elevation) keys of one tile and their counts, or None where no tile exists."""
    import rasterio
    from rasterio.warp import transform

    for attempt in range(4):
        try:
            with rasterio.open(f"/vsicurl/{SOURCE}/{name}/{name}.tif") as s:
                a = s.read(1)
                tr = s.transform
            break
        except rasterio.errors.RasterioIOError as e:
            if "404" in str(e) or "does not exist" in str(e):
                return None
            if attempt == 3:
                raise
    rows, cols = np.indices(a.shape)
    lon = tr.c + (cols.ravel() + 0.5) * tr.a
    lat = tr.f + (rows.ravel() + 0.5) * tr.e
    x, y = transform("EPSG:4326", "EPSG:3035", lon, lat)
    c = np.floor((np.asarray(x) - GRID.xmin) / GRID.res).astype("int64")
    r = np.floor((GRID.ymax - np.asarray(y)) / GRID.res).astype("int64")
    z = a.ravel()
    ok = (c >= 0) & (c < GRID.width) & (r >= 0) & (r < GRID.height) & (z != 0)
    key = (r[ok] * GRID.width + c[ok]) * SPAN + (np.rint(z[ok]).astype("int64") + OFFSET)
    k, n = np.unique(key, return_counts=True)
    return np.stack([k, n])


def build(out: Path, threads: int = 12, log=print) -> None:
    from esa_toolbox import rio

    names = tile_names()
    parts = []
    with ThreadPoolExecutor(threads) as ex:
        for i, p in enumerate(ex.map(tile_counts, names)):
            if p is not None:
                parts.append(p)
            if (i + 1) % 100 == 0:
                log(f"  {i + 1} of {len(names)} tiles")
    kn = np.concatenate(parts, axis=1)
    keys, inv = np.unique(kn[0], return_inverse=True)
    counts = np.bincount(inv, kn[1])
    cell, elev = keys // SPAN, keys % SPAN - OFFSET
    order = np.lexsort((elev, -counts, cell))  # by cell, then most frequent, then lowest
    first = np.concatenate([[True], np.diff(cell[order]) != 0])
    mode = np.full(GRID.width * GRID.height, -32768, dtype="int16")
    mode[cell[order][first]] = elev[order][first]
    rio.write(out, mode.reshape(GRID.shape), GRID, nodata=-32768, names=["elevation_mode_m"])
    log(f"  wrote {out} from {len(parts)} tiles")


if __name__ == "__main__":
    build(Path(sys.argv[1]))
