"""The bumblebee model of each reporting year, projected from the fitted Maxent models of 2012.

The 47 species models of Vallecillo et al. (2018) are kept as fitted, in `params/bumblebee_lambdas_2012.zip`,
and applied to the layers of each year on the 10 km grid of the species records:

- 16 land cover layers, the percentage of the cell in each group of Corine classes (`sdm_land_classes.csv`);
- `snd_km`, the mean distance to semi-natural areas in kilometres, limited to 20 km;
- `bio04`, `bio05`, `bio08` and `bio15`, from the 22 years of E-OBS that end in the reporting year;
- `elmode`, the mode of elevation, which does not change (`elevation_mode_10km.tif`).

Each species has five models, one per cross-validation fold, and takes the mean of their complementary
log-log outputs. A projection is the mean probability of occurrence of the species present in a cell, where a
species is present at or above its threshold (`composite`). The layer of a year is the published layer of
2012 plus the change of the projection since 2012, interpolated to 1 km as the published layer is (`stage`).

In the original method the layer of every year is the published layer of 2012 in the store.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import ndimage

from esa_toolbox import rio
from esa_toolbox.core import PARAMS, Context, read_param, work_key
from esa_toolbox.grid import POLLINATION_GRID, Grid
from esa_toolbox.landcover import CLC_NODATA, clc_file
from esa_toolbox.pool import run as pool_run
from esa_toolbox.services.pollination import maxent, model

GRID = POLLINATION_GRID  # 1 km
GRID100 = GRID.with_res(100)
GRID10 = GRID.with_res(10000)
TILE = 5000  # cells of 100 m, a whole number of 10 km cells
SEA = 523
LAMBDAS = PARAMS / "bumblebee_lambdas_2012.zip"
ELEVATION = PARAMS / "elevation_mode_10km.tif"
CLIMATE = ("bio04", "bio05", "bio08", "bio15")
CLIMATE_YEARS = 22  # E-OBS years behind each layer, as the 1991 to 2012 of the fitted models
PUBLISHED = ("catalogue", "bumblebee_sdm_1km.tif")
BASE_YEAR = 2012


# ---------------------------------------------------------------------------------------------------------
# Layers of a year on the 10 km grid
# ---------------------------------------------------------------------------------------------------------


def land_classes() -> tuple[list[str], np.ndarray]:
    """Names of the land cover layers, and Corine code to layer index (255 for classes left out)."""
    t = read_param("sdm_land_classes.csv")
    names = sorted(set(t["layer"]))
    lut = np.full(65536, 255, dtype="uint8")
    for code, layer in t.iter_rows():
        lut[code] = names.index(layer)
    return names, lut


def _land_job(store: str, year: int, tile: Grid) -> tuple[Grid, np.ndarray]:
    """Land cover percentages, mean distance to semi-natural areas (km) and share of land, per 10 km cell."""
    names, lut = land_classes()
    box = tile.buffered(model.MAX_DISTANCE + 200, GRID100)
    clc = rio.read(clc_file(Path(store), year), box, fill=CLC_NODATA)
    d = model.exact_distance(clc)
    d[np.isnan(d)] = model.MAX_DISTANCE
    r0, c0 = tile.offset_in(box)
    h, w = tile.shape
    d, clc = d[r0:r0 + h, c0:c0 + w], clc[r0:r0 + h, c0:c0 + w]
    blocks = lambda a: a.reshape(h // 100, 100, w // 100, 100)  # noqa: E731
    group = blocks(lut[clc])
    out = np.empty((len(names) + 2, h // 100, w // 100), dtype="float32")
    for k in range(len(names)):
        out[k] = (group == k).sum(axis=(1, 3)) / 100.0
    out[-2] = blocks(d).mean(axis=(1, 3), dtype="float64") / 1000.0
    out[-1] = blocks((clc != CLC_NODATA) & (clc != SEA)).mean(axis=(1, 3))
    return tile, out


def land_file(ctx: Context, year: int) -> Path:
    return ctx.path("prep", f"sdm_land_10km_{year}.tif")


def land_layers(ctx: Context, year: int) -> dict[str, np.ndarray]:
    """The land layers of a year, computed once for each Corine file and kept in the work folder."""
    names = land_classes()[0] + ["snd_km", "land"]
    key = work_key(sys.modules[__name__], model, rio, year, clc_file(ctx.store, year),
                   PARAMS / "sdm_land_classes.csv", PARAMS / "clc_pollination_scores.csv")
    f = land_file(ctx, year)
    if not ctx.year_done("sdm_land", year, key, f):
        res = pool_run(_land_job, [(str(ctx.store), year, t) for t in GRID100.tiles(TILE)], ctx.settings.workers,
                       on_done=lambda d, n: ctx.tick(f"{year} land cover at 10 km", d, n))
        out = np.full((len(names), *GRID10.shape), np.nan, dtype="float32")
        for t, r in res:
            r0, c0 = t.with_res(10000).offset_in(GRID10)
            out[:, r0:r0 + r.shape[1], c0:c0 + r.shape[2]] = r
        rio.write(f, out, GRID10, names=names)
        ctx.mark_year_done("sdm_land", year, key)
    a = rio.read(f, GRID10, band=list(range(1, len(names) + 1)), as_float=True)
    return dict(zip(names, a.astype("float64")))


def climate_file(store: Path, year: int) -> Path:
    return store / "eobs" / f"bioclim_{year}_10km.tif"


def layers(ctx: Context, land_year: int, climate_year: int | None = None) -> dict[str, np.ndarray]:
    """The 22 layers of the models, rounded as the layers of the fitted models were.

    Land cover percentages, the bioclimatic layers and elevation are whole numbers there, and the distance
    keeps three decimals (metres). `land` is the share of land in the cell and is not a layer of the models.
    A cell where the elevation file has no value lies at sea level.
    """
    out = {k: (v if k in ("snd_km", "land") else np.rint(v)) for k, v in land_layers(ctx, land_year).items()}
    out["snd_km"] = np.round(out["snd_km"], 3)
    f = ctx.need("eobs", climate_file(ctx.store, climate_year or land_year).name)
    bio = rio.read(f, GRID10, band=[1, 2, 3, 4], as_float=True)
    out.update({n: np.rint(bio[i].astype("float64")) for i, n in enumerate(CLIMATE)})
    out["elmode"] = np.nan_to_num(rio.read(ELEVATION, GRID10, as_float=True).astype("float64"), nan=0.0)
    return out


# ---------------------------------------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------------------------------------


def species_probability(lay: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Probability of occurrence of each species, the mean of its cross-validation models."""
    runs: dict[str, list[np.ndarray]] = {}
    for m in maxent.read_zip(LAMBDAS):
        runs.setdefault(m.name.rsplit("_", 1)[0], []).append(maxent.cloglog(m, lay))
    return {k: np.mean(v, axis=0) for k, v in sorted(runs.items())}


def thresholds() -> dict[str, float]:
    return dict(read_param("bumblebee_thresholds.csv").select("species", "threshold").iter_rows())


def composite(prob: dict[str, np.ndarray], thr: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Mean probability of occurrence of the species present, and their number (Vallecillo et al., 2018).

    A species is present where its probability reaches its threshold. A cell with no species present is 0.
    """
    total = np.zeros(next(iter(prob.values())).shape, dtype="float64")
    richness = np.zeros(total.shape, dtype="int16")
    for k, p in prob.items():
        present = p >= thr[k]
        total += np.where(present, p, 0.0)
        richness += present
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(richness > 0, total / richness, 0.0)
    missing = np.isnan(np.stack(list(prob.values()))).any(axis=0)
    mean[missing] = np.nan
    return mean, richness


def fill_coast(a: np.ndarray, passes: int = 2, size: int = 21) -> np.ndarray:
    """Cells without a value take the mean of the cells with one in a window of `size`, in `passes` rounds."""
    for _ in range(passes):
        ok = ~np.isnan(a)
        s = ndimage.uniform_filter(np.where(ok, a, 0).astype("float64"), size, mode="constant") * size * size
        # The running sums leave noise of about 1e-17 where no neighbour has a value, so the count is rounded
        # back to whole cells. Without it, cells far out at sea took noise divided by noise.
        n = np.rint(ndimage.uniform_filter(ok.astype("float64"), size, mode="constant") * size * size)
        with np.errstate(divide="ignore", invalid="ignore"):
            fill = np.where(n > 0, s / n, np.nan)
        a = np.where(ok, a, fill).astype("float32")
    return a


def file_10km(ctx: Context, year: int | str) -> Path:
    return ctx.path("prep", f"sdm_10km_{year}.tif")


def file_1km(ctx: Context, year: int) -> Path:
    """The bumblebee layer of a year at 1 km. In the original method it is the published layer of 2012."""
    if ctx.method == "original":
        return ctx.need(*PUBLISHED)
    return ctx.path("prep", f"sdm_1km_{year}.tif")


def read(ctx: Context, year: int) -> np.ndarray:
    return rio.read(file_1km(ctx, year), GRID, as_float=True)


def project(ctx: Context, land_year: int, climate_year: int) -> tuple[np.ndarray, np.ndarray]:
    """The projected layer on the 10 km grid and its number of species, NaN where the cell has no land."""
    lay = layers(ctx, land_year, climate_year)
    land = lay.pop("land")
    mean, richness = composite(species_probability(lay), thresholds())
    mean[~(land > 0)] = np.nan
    return mean, np.where(np.isnan(mean), np.nan, richness)


def to_1km(a: np.ndarray) -> np.ndarray:
    """A 10 km layer interpolated bilinearly to 1 km, with the coastal gaps filled as in the published layer."""
    from rasterio.warp import Resampling, reproject

    out = np.full(GRID.shape, np.nan, dtype="float32")
    reproject(a.astype("float32"), out, src_transform=GRID10.transform, src_crs=rio.CRS_3035, src_nodata=np.nan,
              dst_transform=GRID.transform, dst_crs=rio.CRS_3035, dst_nodata=np.nan, resampling=Resampling.bilinear)
    return fill_coast(out)


def year_key(ctx: Context, land_year: int, climate_year: int) -> str:
    return work_key(sys.modules[__name__], maxent, model, rio, LAMBDAS, ELEVATION, PARAMS / "bumblebee_thresholds.csv",
                    PARAMS / "sdm_land_classes.csv", PARAMS / "clc_pollination_scores.csv",
                    clc_file(ctx.store, land_year), ctx.need("eobs", climate_file(ctx.store, climate_year).name))


def projected(ctx: Context, land_year: int, climate_year: int | None = None) -> np.ndarray:
    """Bands `bumblebee_sdm` and `species_present` of a projection on the 10 km grid, kept in the work folder."""
    climate_year = climate_year or land_year
    name = str(land_year) if climate_year == land_year else f"{land_year}_climate_{climate_year}"
    f = file_10km(ctx, name)
    key = year_key(ctx, land_year, climate_year)
    if not ctx.year_done("sdm_10km", name, key, f):
        ctx.log(f"  47 species on the 10 km grid: land cover of {land_year}, climate of {climate_year}")
        rio.write(f, np.stack(project(ctx, land_year, climate_year)).astype("float32"), GRID10,
                  names=["bumblebee_sdm", "species_present"])
        ctx.mark_year_done("sdm_10km", name, key)
    return rio.read(f, GRID10, band=[1, 2], as_float=True).astype("float64")


def stage(ctx: Context) -> None:
    """The layer of each year: the published layer of 2012 plus the projected change since 2012.

    Both terms of the change come from the same models and the same sources of layers, so what the rebuilt
    layers of 2012 miss of the original ones cancels, and 2012 keeps the published layer.
    """
    published = rio.read(ctx.need(*PUBLISHED), GRID, as_float=True)
    if ctx.method == "original":
        ctx.log("  original method: the published layer of 2012 serves every year")
        return
    from esa_toolbox.services.pollination.inputs import read_reduction

    eu = ~np.isnan(read_reduction(ctx, ctx.years[0], "ebm"))  # land of the EU27 at 1 km
    eu10 = rio.block_sum(np.where(eu, 1.0, np.nan), 10) > 0
    base = projected(ctx, BASE_YEAR)
    base_1km = to_1km(base[0])
    both = eu & ~np.isnan(published) & ~np.isnan(base_1km)
    gap = base_1km[both].astype("float64") - published[both]
    ctx.facts(sdm_base_r=(float(np.corrcoef(base_1km[both], published[both])[0, 1]), 2),
              sdm_base_mae=(float(np.abs(gap).mean()), 3), sdm_base_bias=(float(gap.mean()), 3),
              sdm_base_mean=(float(base_1km[both].mean()), 3), sdm_published_mean=(float(published[both].mean()), 3),
              sdm_base_cells=f"{int(both.sum()):,}", sdm_threshold=(next(iter(thresholds().values())), 2),
              sdm_species=str(len(thresholds())))
    rows = []
    for y in ctx.years:
        a = projected(ctx, y)
        key = work_key(year_key(ctx, y, y), year_key(ctx, BASE_YEAR, BASE_YEAR), ctx.need(*PUBLISHED))
        if not ctx.year_done("sdm", y, key, file_1km(ctx, y)):
            change = np.nan_to_num(to_1km(a[0] - base[0]), nan=0.0) if y != BASE_YEAR else 0.0
            rio.write(file_1km(ctx, y), np.clip(published + change, 0, 1).astype("float32"), GRID,
                      names=["bumblebee_sdm"])
            ctx.mark_year_done("sdm", y, key)
        ok = eu10 & ~np.isnan(a[0]) & ~np.isnan(base[0])
        lay = layers(ctx, y)
        part = {"land cover": projected(ctx, y, BASE_YEAR), "climate": projected(ctx, BASE_YEAR, y)} if y != BASE_YEAR else {}
        rows.append({"Year": y, "Projected layer": float(a[0][ok].mean()), "Species present": float(a[1][ok].mean()),
                     "Change from 2012": float((a[0] - base[0])[ok].mean()),
                     **{f"From {k}": float((v[0] - base[0])[ok].mean()) for k, v in part.items()},
                     "Temperature seasonality": float(lay["bio04"][ok].mean()),
                     "Warmest month (°C)": float(lay["bio05"][ok].mean()),
                     "Wettest quarter (°C)": float(lay["bio08"][ok].mean()),
                     "Precipitation seasonality": float(lay["bio15"][ok].mean()),
                     "Distance (km)": float(lay["snd_km"][ok].mean())})
    t = pl.DataFrame(rows).with_columns(pl.col("^From .*$").fill_null(0.0))
    signed = [pl.col(c).map_elements(lambda v: f"{v:+.4f}", return_dtype=pl.String)
              for c in ("Change from 2012", "From land cover", "From climate")]
    ctx.table(t.select("Year", "Projected layer", *signed, "Species present"), "sdm_projection",
              {"Projected layer": 3, "Species present": 1})
    ctx.table(t.select("Year", "Temperature seasonality", "Warmest month (°C)", "Wettest quarter (°C)",
                       "Precipitation seasonality", "Distance (km)"), "sdm_layers", 2)
    first, last = t.row(0, named=True), t.row(-1, named=True)
    ctx.facts(sdm_first_year=str(first["Year"]), sdm_last_year=str(last["Year"]),
              sdm_projected_first=(first["Projected layer"], 3), sdm_projected_last=(last["Projected layer"], 3),
              sdm_change_last=(last["Change from 2012"], 3), sdm_change_first=(first["Change from 2012"], 3),
              sdm_last_from_climate=(last["From climate"], 3), sdm_last_from_land=(last["From land cover"], 3),
              sdm_bio05_first=(first["Warmest month (°C)"], 1), sdm_bio05_last=(last["Warmest month (°C)"], 1))
