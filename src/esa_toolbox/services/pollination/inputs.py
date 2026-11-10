"""Regions and the expert-based model at 100 m, with its 1 km reduction.

Two points of the implementation:

- The 1 km reduction comes out of the same pass as the 100 m model, in float32. The pass also keeps the 100 m
  model of each tile in work/pollination/prep/ebm_100m/ (ZSTD, lossless), under a key of the inputs and the
  code, and the 100 m maps stage reads it back instead of running the model again. Writing a tile takes
  about a tenth of a second, and running the model on it about three seconds.
- The distance to semi-natural areas is computed inside each tile with an exact Euclidean transform, so no
  distance file is needed.
"""

from __future__ import annotations

from pathlib import Path

import shutil
import sys

import numpy as np
import polars as pl
import rasterio
import shapely

from esa_toolbox import nuts, rio
from esa_toolbox.core import Context, work_key
from esa_toolbox.grid import POLLINATION_GRID, Grid
from esa_toolbox.landcover import CLC_NODATA, clc_file
from esa_toolbox.pool import run as pool_run
from esa_toolbox.services.pollination import model

GRID = POLLINATION_GRID  # 1 km
GRID100 = GRID.with_res(100)
TILE = 5000  # cells of 100 m, as the earlier routine
NO_CROP_DATA = ("HR", "MT", "CY")  # Vallecillo et al. (2018), footnote 6
CACHE_OPTS = dict(compress="ZSTD", zstd_level=1, tiled=True, blockxsize=512, blockysize=512)


# ---------------------------------------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------------------------------------


def regions_table(ctx: Context) -> pl.DataFrame:
    return ctx.load("nuts_regions")


def prepare(ctx: Context) -> None:
    regs = nuts.regions(ctx.store)
    n3 = regs.level(3)  # already sorted by code, so the id is the row number
    ids = n3.ids(GRID)
    filled = fill_coast(ids)
    rio.write(ctx.path("prep", "nuts3_id_1km.tif"), filled.astype("uint16"), GRID, nodata=0)
    t = regs.table.with_columns(id=pl.lit(None, pl.Int64))
    t = t.join(n3.table.with_row_index("id0").select("nuts_id", pl.col("id0").cast(pl.Int64) + 1),
               on="nuts_id", how="left").with_columns(id=pl.col("id0")).drop("id0")
    ctx.save(t.select("level", "nuts_id", "name", "geo", "id").sort("level", "nuts_id"), "nuts_regions")


def fill_coast(ids: np.ndarray, size: int = 5) -> np.ndarray:
    """Cells outside every region take the most common region within 2 km (terra focal modal, NA only).

    Ties take the smallest id, which is the rule of terra's `modal`.
    """
    out = ids.copy()
    h = size // 2
    pad = np.pad(ids, h)
    from scipy.ndimage import maximum_filter

    near = maximum_filter((ids > 0).astype("uint8"), size=size, mode="constant") > 0
    empty = np.argwhere((ids == 0) & near)
    for r, c in empty:
        win = pad[r:r + size, c:c + size]
        v = win[win > 0]
        if v.size:
            vals, counts = np.unique(v, return_counts=True)
            out[r, c] = vals[np.argmax(counts)]
    return out


def zones(ctx: Context, crop_data_only: bool = False) -> np.ndarray:
    z = rio.read(ctx.work / "prep" / "nuts3_id_1km.tif", GRID).astype("int32")
    if crop_data_only:
        t = regions_table(ctx)
        drop = t.filter(pl.col("geo").is_in(NO_CROP_DATA) & pl.col("id").is_not_null())["id"].to_numpy()
        z[np.isin(z, drop)] = 0
    return z


# ---------------------------------------------------------------------------------------------------------
# Expert-based model, one tile
# ---------------------------------------------------------------------------------------------------------


def eu_geoms(store: Path):
    return nuts.countries(store).geoms


def tiles(store: Path) -> list[Grid]:
    union = shapely.union_all(eu_geoms(store))
    return [t for t in GRID100.tiles(TILE) if shapely.intersects(union, shapely.box(*t.bounds))]


def model_tile(store: Path, year: int, tile: Grid) -> dict[str, np.ndarray]:
    """The 100 m model on one tile, masked to the EU27, and the Corine codes of the tile."""
    box = tile.buffered(MAXD_CELLS * 100, GRID100)
    clc = rio.read(clc_file(store, year), box, fill=CLC_NODATA)
    roads = rio.read(store / "osm" / "osm_major_roads_100m.tif", box)
    dist = model.exact_distance(clc)
    ai = rio.warp(store / "catalogue" / f"activity_index_{year}_1km.tif", box, "bilinear")
    s = model.expert_model(clc, roads, dist, ai)
    r0, c0 = tile.offset_in(box)
    h, w = tile.shape
    s = s[r0:r0 + h, c0:c0 + w]
    clc = clc[r0:r0 + h, c0:c0 + w]
    box_geoms = [g for g in eu_geoms(store) if shapely.intersects(g, shapely.box(*tile.bounds))]
    eu = rio.rasterize([(shapely.clip_by_rect(g, *tile.bounds), 1) for g in box_geoms], tile,
                       all_touched=True, dtype="uint8")
    s[eu == 0] = np.nan
    return {"ebm": s, "clc": clc}


MAXD_CELLS = model.MAX_DISTANCE // 100 + 2


def reduce_tile(s: np.ndarray, clc: np.ndarray) -> np.ndarray:
    """1 km reduction: mean over land, mean over farmland, land and farmland cells, and the sum of the model
    on each ecosystem type, with NaN where a 1 km cell has no land (or no farmland)."""
    sc = model.scores()
    land = ~np.isnan(s)
    agro = sc.agro[clc] & land
    eco = sc.ecosystem[clc]
    H, W = s.shape
    h, w = H // 10, W // 10
    idx = (np.arange(H) // 10)[:, None] * w + (np.arange(W) // 10)[None, :]
    n = h * w
    sl = s[land].astype("float64")
    il = idx[land]
    ia = idx[agro]
    ln = np.bincount(il, minlength=n).astype("float64")
    an = np.bincount(ia, minlength=n).astype("float64")
    su = np.bincount(il, sl, n)
    sa = np.bincount(ia, s[agro].astype("float64"), n)
    k = len(sc.ecosystems)
    by = np.bincount(il + eco[land].astype("int64") * n, sl, n * k).reshape(k, n)
    nz = lambda a, c: np.where(c > 0, a, np.nan)  # noqa: E731
    with np.errstate(invalid="ignore", divide="ignore"):
        bands = [nz(su / ln, ln), nz(sa / an, an), nz(ln, ln), nz(an, an)] + [nz(by[i], ln) for i in range(k)]
    return np.stack(bands).reshape(-1, h, w).astype("float32")


def model_key(store: Path, year: int) -> str:
    """What the 100 m model of a year depends on: the code and the input files."""
    files = [clc_file(store, year), store / "osm" / "osm_major_roads_100m.tif",
             store / "catalogue" / f"activity_index_{year}_1km.tif", store.joinpath(*nuts.NUTS_FILE)]
    return work_key(sys.modules[__name__], model, rio, year, *files)


def model_cache(work: Path, year: int, key: str) -> Path:
    """The folder of 100 m model tiles of `year` under `key`. Older keys of the same year are removed."""
    root = work / "prep" / "ebm_100m"
    folder = root / f"{year}_{key}"
    for old in root.glob(f"{year}_*"):
        if old != folder:
            shutil.rmtree(old, ignore_errors=True)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def cached_model_tile(store: Path, year: int, tile: Grid, cache: Path | None) -> dict[str, np.ndarray]:
    """The 100 m model of a tile, read from the cache when it is there."""
    kept = cache / f"{tile.name}.tif" if cache else None
    if kept is not None and kept.exists():
        with rasterio.open(kept) as s:
            ebm = s.read(1)
        return {"ebm": ebm, "clc": rio.read(clc_file(store, year), tile, fill=CLC_NODATA)}
    m = model_tile(store, year, tile)
    if kept is not None:
        rio.write(kept, m["ebm"], tile, opts=CACHE_OPTS)
    return m


def _reduce_job(store: str, year: int, tile: Grid, cache: str) -> tuple[Grid, np.ndarray]:
    m = cached_model_tile(Path(store), year, tile, Path(cache))
    return tile, reduce_tile(m["ebm"], m["clc"])


def reduction_file(ctx: Context, year: int) -> Path:
    return ctx.path("prep", f"ebm_reduction_1km_{year}.tif")


def expert_stage(ctx: Context) -> None:
    sc = model.scores()
    names = ["ebm", "ebm_agro", "land_cells", "agro_cells"] + [f"sum_{e}" for e in sc.ecosystems]
    ts = tiles(ctx.store)
    for y in ctx.years:
        key = model_key(ctx.store, y)
        if ctx.year_done("model", y, key, reduction_file(ctx, y)):
            continue
        cache = model_cache(ctx.work, y, key)
        ctx.log(f"  {y}: expert-based model on {len(ts)} tiles")
        res = pool_run(_reduce_job, [(str(ctx.store), y, t, str(cache)) for t in ts], ctx.settings.workers,
                       on_done=lambda d, n: ctx.tick(f"{y} model", d, n))
        out = np.full((len(names), *GRID.shape), np.nan, dtype="float32")
        for t, r in res:
            r0, c0 = t.with_res(1000).offset_in(GRID)
            out[:, r0:r0 + r.shape[1], c0:c0 + r.shape[2]] = r
        rio.write(reduction_file(ctx, y), out, GRID, names=names)
        ctx.mark_year_done("model", y, key)


def read_reduction(ctx: Context, year: int, band: str) -> np.ndarray:
    sc = model.scores()
    names = ["ebm", "ebm_agro", "land_cells", "agro_cells"] + [f"sum_{e}" for e in sc.ecosystems]
    return rio.read(reduction_file(ctx, year), GRID, band=names.index(band) + 1, as_float=True)


def supply_bands(ctx: Context, year: int) -> np.ndarray:
    sc = model.scores()
    return rio.read(reduction_file(ctx, year), GRID, band=list(range(5, 5 + len(sc.ecosystems))), as_float=True)
