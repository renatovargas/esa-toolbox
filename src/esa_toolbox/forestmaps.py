"""The 100 m spatial key of the forest services, and the maps that spread national values over it.

The weight of a 100 m cell is 1 on forest inside an EU27 country and 0 elsewhere, times the annual gross dry
matter productivity resampled to 100 m (bilinear). Each map cell takes the national value times its weight
over the weight of its country, so the cells of a country add up to the national value.

Resampling the productivity is most of the cost of a tile, and wood and carbon need the same resampled values
twice each (country sums, then maps). The first pass of either service therefore keeps the productivity of
the forest cells of every tile in work/shared/forest_gdmp_100m/, under a key of the inputs and of this file,
and every later pass reads it back. The cache stays outside the outputs and is rebuilt whenever an input changes.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import shapely

from esa_toolbox import nuts, rio
from esa_toolbox.core import work_key
from esa_toolbox.grid import Grid
from esa_toolbox.landcover import CLC_NODATA, clc_file
from esa_toolbox.pool import run as pool_run

TILE_CELLS = 4000  # 400 km at 100 m
TILE_OPTS = dict(compress="ZSTD", zstd_level=1, predictor=2, tiled=True, blockxsize=512, blockysize=512)
UINT32_NODATA = 4294967295
CACHE_CODES = (311, 312, 313, 324)  # forest of wood and carbon, and transitional woodland of carbon
CACHE_OPTS = dict(compress="ZSTD", zstd_level=1, tiled=True, blockxsize=512, blockysize=512)


@dataclass(frozen=True)
class Quantity:
    """A map: its file stem, the national column it spreads and the factor to its whole-number unit."""

    name: str
    column: str
    scale: float


def tiles_with_land(store: Path, grid_100m: Grid) -> list[Grid]:
    regs = nuts.countries(store)
    union = shapely.union_all(regs.geoms)
    return [t for t in grid_100m.tiles(TILE_CELLS) if shapely.intersects(union, shapely.box(*t.bounds))]


def productivity_cache(work: Path, store: Path, year: int, gdmp: Path) -> Path:
    """The folder of resampled forest productivity for `year`. An older key of the same year is removed."""
    key = work_key(sys.modules[__name__], CACHE_CODES, year, gdmp, clc_file(store, year), store.joinpath(*nuts.NUTS_FILE))
    root = work / "shared" / "forest_gdmp_100m"
    folder = root / f"{year}_{key}"
    for old in root.glob(f"{year}_*"):
        if old != folder:
            shutil.rmtree(old, ignore_errors=True)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def weight_tile(store: Path, year: int, tile: Grid, forest_codes: list[int], gdmp: Path,
                cache: Path | None = None) -> tuple:
    assert set(forest_codes) <= set(CACHE_CODES)
    regs = nuts.countries(store)
    ids = regs.ids(tile)
    clc = rio.read(clc_file(store, year), tile, fill=CLC_NODATA)
    kept = cache / f"{tile.name}.tif" if cache else None
    if kept is not None and kept.exists():
        with rasterio.open(kept) as s:
            g = s.read(1)
    else:
        g = rio.warp(gdmp, tile, "bilinear")
        g = np.where(np.isin(clc, CACHE_CODES) & (ids > 0), np.nan_to_num(g, nan=0.0), 0.0).astype("float32")
        if kept is not None:
            rio.write(kept, g, tile, opts=CACHE_OPTS)
    # The same numbers as masking the productivity with the forest codes of the service directly.
    w = np.where(np.isin(clc, forest_codes), g, 0.0).astype("float32")
    return w, ids, len(regs.geoms)


def _sum_job(store: str, year: int, tile: Grid, codes: list[int], gdmp: str, want_1km: bool,
             count_codes: list[int], cache: str):
    w, ids, n = weight_tile(Path(store), year, tile, codes, Path(gdmp), Path(cache) if cache else None)
    sums = rio.zonal_sum(w, ids, n)
    km = None
    if want_1km:
        # Sum of the 100 m weights of each 1 km cell. The weight is 0 outside the countries, as in the earlier
        # weight files, so every 1 km cell has a value.
        km = rio.block_sum(w, 10).astype("float32")
    counts = {}
    if count_codes:
        clc = rio.read(clc_file(Path(store), year), tile, fill=CLC_NODATA)[ids > 0]
        counts = {c: int((clc == c).sum()) for c in count_codes}
    return sums, km, counts


def country_sums(store: Path, year: int, grid_100m: Grid, codes: list[int], gdmp: Path, workers: int,
                 want_1km: bool = False, grid_1km: Grid | None = None, progress=None,
                 count_codes: list[int] | None = None, cache: Path | None = None):
    """National sums of the 100 m weight, optionally the 1 km grid of summed weights, and optionally the number
    of 100 m cells of some Corine codes inside the countries."""
    tiles = tiles_with_land(store, grid_100m)
    res = pool_run(_sum_job, [(str(store), year, t, codes, str(gdmp), want_1km, count_codes or [],
                               str(cache or "")) for t in tiles], workers, on_done=progress)
    sums = np.sum([r[0] for r in res], axis=0)
    km = None
    if want_1km:
        km = np.zeros(grid_1km.shape, dtype="float32")
        for t, r in zip(tiles, res):
            k = r[1]
            r0, c0 = t.with_res(1000).offset_in(grid_1km)
            km[r0:r0 + k.shape[0], c0:c0 + k.shape[1]] = k
    counts = {}
    for r in res:
        for c, v in r[2].items():
            counts[c] = counts.get(c, 0) + v
    return sums, km, counts


def _map_job(store: str, year: int, tile: Grid, codes: list[int], gdmp: str,
             factors: dict[str, list[float]], scales: dict[str, float], out_dir: str, stem: str,
             zero_outside_forest_country: bool, cache: str):
    w, ids, n = weight_tile(Path(store), year, tile, codes, Path(gdmp), Path(cache) if cache else None)
    idx = np.clip(ids, 0, n)
    w64 = w.astype("float64")
    out = {}
    for q, f in factors.items():
        # Factor by country id, with NaN outside the countries (index 0), looked up once per cell.
        tab = np.asarray(f, dtype="float64").copy()
        if zero_outside_forest_country:
            tab[np.isnan(tab)] = 0.0
        tab[0] = np.nan
        v = w64 * tab[idx]
        v *= scales[q]
        valid = ~np.isnan(v)
        np.rint(v, out=v)
        iv = np.where(valid, v, float(UINT32_NODATA)).astype("uint32")
        sums = np.bincount(idx.ravel(), weights=np.where(valid, v, 0.0).ravel(), minlength=n + 1)[1:]
        path = Path(out_dir) / q / f"{stem}_{tile.name}.tif"
        rio.write(path, iv, tile, nodata=UINT32_NODATA, opts=TILE_OPTS)
        out[q] = sums
    return out


def write_maps(store: Path, year: int, grid_100m: Grid, codes: list[int], gdmp: Path, workers: int,
               factors: dict[str, np.ndarray], scales: dict[str, float], tile_dir: Path, maps_dir: Path,
               file_name: dict[str, str], zero_outside_forest_country: bool = False, progress=None,
               log=print, cache: Path | None = None) -> dict[str, np.ndarray]:
    """Write one whole-number COG per quantity, and return its sums by country in map units."""
    tiles = tiles_with_land(store, grid_100m)
    jobs = [(str(store), year, t, codes, str(gdmp), {q: list(f) for q, f in factors.items()}, scales,
             str(tile_dir), str(year), zero_outside_forest_country, str(cache or "")) for t in tiles]
    res = pool_run(_map_job, jobs, workers, on_done=progress)
    sums = {q: np.sum([r[q] for r in res], axis=0) for q in factors}
    assemble(tile_dir, maps_dir, list(factors), year, grid_100m, file_name, workers, log)
    return sums


def _cog_job(vrt: str, out: str, threads: int) -> str:
    rio.to_cog(Path(vrt), Path(out), predictor=2, threads=threads)
    return out


def cog_all(jobs: list[tuple[Path, Path]], workers: int) -> None:
    """Write several Cloud Optimised GeoTIFFs at once, sharing the processors among them."""
    n = max(1, min(4, workers, len(jobs)))
    threads = max(1, (os.cpu_count() or 4) // n)
    pool_run(_cog_job, [(str(v), str(o), threads) for v, o in jobs], n)


def assemble(tile_dir: Path, maps_dir: Path, quantities: list[str], year: int, grid: Grid,
             file_name: dict[str, str], workers: int, log=print) -> None:
    """Mosaic the tiles of each quantity into a Cloud Optimised GeoTIFF, several maps at a time."""
    jobs = []
    for q in quantities:
        tiles = sorted((tile_dir / q).glob(f"{year}_*.tif"))
        vrt = tile_dir / q / f"{year}.vrt"
        rio.build_vrt(tiles, grid, vrt, dtype="uint32", nodata=UINT32_NODATA)
        jobs.append((vrt, maps_dir / file_name[q]))
    cog_all(jobs, workers)
    for q in quantities:
        for t in (tile_dir / q).glob(f"{year}_*"):
            t.unlink()
