"""Small raster operations on NumPy arrays, read and written through rasterio.

Every function works on one block of a grid at a time, so memory stays bounded whatever the size of the map.
Writes go to a temporary name and are renamed when complete, so an interrupted run leaves no partial file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence
from xml.sax.saxutils import escape

import numpy as np
import rasterio
import rasterio.shutil
from rasterio.enums import Resampling
from rasterio.features import rasterize as _rasterize
from rasterio.warp import reproject
from rasterio.windows import Window

from esa_toolbox.grid import CRS_3035, Grid

FLOAT_OPTS = dict(compress="DEFLATE", predictor=3, tiled=True, blockxsize=512, blockysize=512)
INT_OPTS = dict(compress="DEFLATE", predictor=2, tiled=True, blockxsize=512, blockysize=512)


def read(path: Path | str, grid: Grid, band: int | Sequence[int] = 1, *, as_float: bool = False,
         fill: float | None = None) -> np.ndarray:
    """Read the cells of `grid` from a raster on the same cell alignment, padding outside the file.

    With `as_float`, no-data becomes NaN and the result is float32. Otherwise the raw values come back, and
    cells outside the file take `fill`, or the no-data value of the file.
    """
    with rasterio.open(path) as src:
        sx, sy = src.transform.a, -src.transform.e
        if abs(sx - grid.res) > 1e-6 or abs(sy - grid.res) > 1e-6:
            raise ValueError(f"{path}: cell size {sx} differs from {grid.res}")
        col = (grid.xmin - src.transform.c) / sx
        row = (src.transform.f - grid.ymax) / sy
        if abs(col - round(col)) > 1e-6 or abs(row - round(row)) > 1e-6:
            raise ValueError(f"{path}: not aligned with the grid at {grid.xmin}, {grid.ymax}")
        col, row = round(col), round(row)
        h, w = grid.shape
        bands = [band] if isinstance(band, int) else list(band)
        nodata = src.nodata
        pad = fill if fill is not None else (nodata if nodata is not None else 0)
        out = np.full((len(bands), h, w), pad, dtype=src.dtypes[bands[0] - 1])
        r0, c0 = max(row, 0), max(col, 0)
        r1, c1 = min(row + h, src.height), min(col + w, src.width)
        if r1 > r0 and c1 > c0:
            data = src.read(bands, window=Window(c0, r0, c1 - c0, r1 - r0))
            out[:, r0 - row:r1 - row, c0 - col:c1 - col] = data
    if as_float:
        f = out.astype("float32")
        if nodata is not None and not np.isnan(nodata):
            f[out == nodata] = np.nan
        if fill is None and nodata is None:
            pass
        out = f
    return out[0] if isinstance(band, int) else out


def warp(path: Path | str, grid: Grid, resampling: str = "bilinear", band: int = 1) -> np.ndarray:
    """Reproject one band of a raster onto `grid` as float32, with NaN where there is no source."""
    dst = np.full(grid.shape, np.nan, dtype="float32")
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, band), destination=dst, dst_transform=grid.transform,
            dst_crs=CRS_3035, dst_nodata=np.nan, resampling=getattr(Resampling, resampling),
            # The exact transformer, so a cell gets the same value whatever the tiling. GDAL's default
            # approximation (0.125 pixels) depends on the extent of each call.
            tolerance=0.0,
        )
    return dst


def write(path: Path | str, data: np.ndarray, grid: Grid, *, nodata: float | None = None,
          names: Sequence[str] | None = None, opts: dict | None = None) -> Path:
    """Write one or more bands as a tiled GeoTIFF, atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = data[None] if data.ndim == 2 else data
    if opts is None:
        opts = FLOAT_OPTS if arr.dtype.kind == "f" else INT_OPTS
    if nodata is None and arr.dtype.kind == "f":
        nodata = np.nan
    small = grid.width < 512 or grid.height < 512
    profile = dict(driver="GTiff", height=grid.height, width=grid.width, count=arr.shape[0],
                   dtype=arr.dtype.name, crs=CRS_3035, transform=grid.transform, nodata=nodata,
                   **{k: v for k, v in opts.items() if not (small and k.startswith("block"))})
    if arr.nbytes > 3.5e9:
        profile["BIGTIFF"] = "YES"
    tmp = path.with_name(path.name + ".part.tif")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(arr)
        if names:
            for i, n in enumerate(names, start=1):
                dst.set_band_description(i, n)
    os.replace(tmp, path)
    return path


def blocks(a: np.ndarray, f: int) -> np.ndarray:
    """View a 2D array as (rows, cols, f*f) blocks of f by f cells. Partial blocks at the right and
    bottom edges are padded with NaN, as terra `aggregate` does."""
    h, w = a.shape
    if h % f or w % f:
        p = np.full((-(-h // f) * f, -(-w // f) * f), np.nan, dtype=a.dtype if a.dtype.kind == "f" else "float64")
        p[:h, :w] = a
        a, (h, w) = p, p.shape
    return a.reshape(h // f, f, w // f, f).swapaxes(1, 2).reshape(h // f, w // f, f * f)


def block_sum(a: np.ndarray, f: int = 10) -> np.ndarray:
    """Sum of each f by f block, ignoring NaN, NaN where the whole block is NaN (terra aggregate, sum)."""
    b = blocks(a.astype("float64"), f)
    s = np.nansum(b, axis=2)
    s[np.isnan(b).all(axis=2)] = np.nan
    return s


def block_mean(a: np.ndarray, f: int = 10) -> np.ndarray:
    """Mean of each block over its cells with data (terra aggregate, mean, na.rm = TRUE)."""
    b = blocks(a.astype("float64"), f)
    n = (~np.isnan(b)).sum(axis=2)
    s = np.nansum(b, axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / n, np.nan)


def zonal_sum(values: np.ndarray, zones: np.ndarray, n: int) -> np.ndarray:
    """Sum of `values` in each zone 1..n, ignoring NaN and cells outside every zone (zone 0 or less)."""
    z = zones.ravel()
    v = values.ravel()
    ok = (z > 0) & (z <= n) & ~np.isnan(v)
    return np.bincount(z[ok].astype("int64"), weights=v[ok].astype("float64"), minlength=n + 1)[1:]


def rasterize(shapes: Iterable, grid: Grid, *, all_touched: bool = False, dtype: str = "int32",
              fill: int = 0) -> np.ndarray:
    """Burn (geometry, value) pairs on the grid. A cell takes a polygon when its centre falls inside it."""
    shapes = list(shapes)
    if not shapes:
        return np.full(grid.shape, fill, dtype=dtype)
    return _rasterize(shapes, out_shape=grid.shape, transform=grid.transform, fill=fill,
                      all_touched=all_touched, dtype=dtype)


def build_vrt(tiles: Sequence[Path], grid: Grid, path: Path, *, dtype: str, nodata: float | None) -> Path:
    """A VRT that places tiles of the same cell size on `grid`. Tiles that are missing leave no-data."""
    gdal_type = {"uint8": "Byte", "uint16": "UInt16", "int16": "Int16", "uint32": "UInt32",
                 "int32": "Int32", "float32": "Float32", "float64": "Float64"}[dtype]
    nd = "" if nodata is None else f"<NoDataValue>{'nan' if np.isnan(nodata) else nodata}</NoDataValue>"
    parts = [f'<VRTDataset rasterXSize="{grid.width}" rasterYSize="{grid.height}">',
             f"<SRS>{escape(CRS_3035.to_wkt())}</SRS>",
             "<GeoTransform>" + ", ".join(repr(v) for v in grid.transform.to_gdal()) + "</GeoTransform>",
             f'<VRTRasterBand dataType="{gdal_type}" band="1">{nd}']
    for t in tiles:
        with rasterio.open(t) as s:
            row = round((grid.ymax - s.bounds.top) / grid.res)
            col = round((s.bounds.left - grid.xmin) / grid.res)
            # Tiles do not overlap and share the no-data value of the band, so a plain copy of each tile is
            # exact. A source with its own no-data value would make GDAL test every cell, which is slower.
            parts.append(
                f'<SimpleSource><SourceFilename relativeToVRT="0">{escape(str(Path(t).resolve()))}'
                f'</SourceFilename><SourceBand>1</SourceBand>'
                f'<SrcRect xOff="0" yOff="0" xSize="{s.width}" ySize="{s.height}"/>'
                f'<DstRect xOff="{col}" yOff="{row}" xSize="{s.width}" ySize="{s.height}"/>'
                "</SimpleSource>")
    parts.append("</VRTRasterBand></VRTDataset>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def to_cog(src: Path, out: Path, *, predictor: int, resampling: str = "AVERAGE",
           threads: int | str = "ALL_CPUS", level: int = 6, overviews: bool = False) -> Path:
    """Copy a raster (usually a VRT) to a Cloud Optimised GeoTIFF, atomically.

    The 100 m maps have no overviews. They take about 60 percent more disk and three times as long to write,
    and QGIS draws the maps quickly without them. `gdaladdo` adds them to a copy when needed.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part.tif")
    with rasterio.Env(GDAL_NUM_THREADS=str(threads), GDAL_CACHEMAX=1024):
        rasterio.shutil.copy(src, tmp, driver="COG", compress="DEFLATE", predictor=str(predictor),
                             level=str(level), num_threads=str(threads), bigtiff="IF_SAFER",
                             overviews="AUTO" if overviews else "NONE", resampling=resampling)
    os.replace(tmp, out)
    return out
