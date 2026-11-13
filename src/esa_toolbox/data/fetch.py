"""Fetchers that download the inputs into the store.

Every file is written under a temporary name and renamed when complete. The Eurostat extracts record their
date of retrieval in extracts.json beside them. Several steps follow the conventions of terra, the R package
in which the method was first implemented (rounding of halves, buffers, masks and crop windows), so that the
derived layers keep the values of that implementation.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import re
import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from esa_toolbox import nuts, rio
from esa_toolbox.core import EsaError, Settings
from esa_toolbox.data.catalog import CLC_RECORDS, Dataset
from esa_toolbox.grid import POLLINATION_GRID

UA = {"User-Agent": "esa-toolbox"}


def _get(url: str, dest: Path, log=print, timeout: int = 7200) -> Path:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with requests.get(url, stream=True, timeout=timeout, headers=UA) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    os.replace(tmp, dest)
    log(f"  downloaded {dest}")
    return dest


def fetch(d: Dataset, settings: Settings, log: Callable[[str], None] = print) -> None:
    fn = FETCHERS.get(d.method)
    if fn is None:
        raise EsaError(f"{d.id} has no automatic fetcher ({d.method}). {d.note}")
    fn(d, settings.store, log)


# ---------------------------------------------------------------------------------------------------------


def fetch_http(d: Dataset, store: Path, log) -> None:
    for p in d.missing(store):
        _get(d.source, p, log)


def fetch_clc(d: Dataset, store: Path, log) -> None:
    """Corine accounting layers v21: the record page names a public share that ends in /download."""
    import requests

    for year, uuid in CLC_RECORDS.items():
        out = store / "clc_accounting" / "v21" / f"CLC{year}ACC_V2024_21.tif"
        if out.exists():
            continue
        page = requests.get(f"https://sdi.eea.europa.eu/data/{uuid}", timeout=120, headers=UA).text
        m = re.search(r"https://sdi\.eea\.europa\.eu/datashare/s/[A-Za-z0-9]+/download", page)
        if not m:
            raise EsaError(f"No download link on the record page of Corine {year}. Download it by hand into {out}.")
        with tempfile.TemporaryDirectory(dir=store) as tmp:
            z = _get(m.group(0), Path(tmp) / "clc.zip", log)
            _unzip_all(z, Path(tmp) / "x")
            tif = next(Path(tmp, "x").rglob(f"CLC{year}ACC*.tif"))
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(tif, out)


def _unzip_all(z: Path, dest: Path) -> None:
    with zipfile.ZipFile(z) as f:
        f.extractall(dest)
    for inner in list(dest.rglob("*.zip")):
        with zipfile.ZipFile(inner) as f:
            f.extractall(inner.parent)


def fetch_gisco(d: Dataset, store: Path, log) -> None:
    """NUTS 2024 from GISCO, written with the layer name that the tool reads.

    As in the copy distributed with the INCA plugin, the outermost regions of France (FRY) and Svalbard (NO0B2)
    are left out, and so are their polygons in the regions above them. They lie outside the European grid, and
    would otherwise enter the regional tables as rows without cells. The regions of the EU27 then have the same
    geometry as that copy (measured on 2026-11-09).
    """
    import pyogrio
    import shapely

    out = store / nuts.NUTS_FILE[0] / nuts.NUTS_FILE[1]
    out.parent.mkdir(parents=True, exist_ok=True)
    url = "https://gisco-services.ec.europa.eu/distribution/v2/nuts/gpkg/NUTS_RG_01M_2024_3035.gpkg"
    with tempfile.TemporaryDirectory(dir=store) as tmp:
        src = _get(url, Path(tmp) / "nuts.gpkg", log)
        meta, _, geom, fields = pyogrio.raw.read(src)
        ids = np.asarray(fields[list(meta["fields"]).index("NUTS_ID")]).astype(str)
        drop = np.char.startswith(ids, "FRY") | (ids == "NO0B2")
        g = shapely.from_wkb(geom)
        away = shapely.union_all(g[drop])
        for i in np.flatnonzero(~drop & shapely.intersects(g, away)):
            g[i] = shapely.multipolygons([p for p in shapely.get_parts(g[i]) if not shapely.intersects(p, away)])
        # Five German regions cross themselves in the GISCO file. Repaired, they equal the INCA copy.
        for i in np.flatnonzero(~shapely.is_valid(g)):
            g[i] = shapely.multipolygons(shapely.get_parts(shapely.make_valid(g[i])))
        geom = shapely.to_wkb(g[~drop])
        fields = [np.asarray(f)[~drop] for f in fields]
        tmp_out = out.with_name(out.name + ".part.gpkg")
        pyogrio.raw.write(tmp_out, geom, fields, meta["fields"], layer=nuts.LAYER, driver="GPKG",
                          geometry_type=meta["geometry_type"], crs=meta["crs"])
        os.replace(tmp_out, out)


# ---------------------------------------------------------------------------------------------------------
# Eurostat
# ---------------------------------------------------------------------------------------------------------


def eurostat_table(code: str, filters: dict[str, str] | None = None) -> pl.DataFrame:
    """A Eurostat dataset in long form (geo, TIME_PERIOD, OBS_VALUE and its dimensions), from the SDMX API."""
    import requests

    key = ""
    if filters:
        key = "/" + ".".join(filters.values())
    # Compressed, the transfer is about 14 times smaller and the whole HICP table takes 35 s instead of over 10 minutes.
    url = f"https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/{code}{key}?format=SDMX-CSV&compressed=true"
    r = requests.get(url, timeout=1800, headers=UA)
    r.raise_for_status()
    df = pl.read_csv(io.BytesIO(gzip.decompress(r.content)), infer_schema_length=0)
    return df.with_columns(year=pl.col("TIME_PERIOD").str.slice(0, 4).cast(pl.Int64),
                           value=pl.col("OBS_VALUE").cast(pl.Float64, strict=False))


def _save(df: pl.DataFrame, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    df.write_parquet(tmp)
    os.replace(tmp, path)
    mfile = path.parent / "extracts.json"
    m = json.loads(mfile.read_text()) if mfile.exists() else {}
    m[path.stem] = meta
    mfile.write_text(json.dumps(m, indent=2), encoding="utf-8")


def _hicp() -> pl.DataFrame:
    """The EU27 index (2015 = 100), with the years before its first published year estimated.

    Eurostat publishes the EU27_2020 index from 2000, and pollination averages the three years before each
    reporting year, so 2000 needs 1998 and 1999. The aggregate is the weighted mean of the annual rates of its
    parts, the euro area and the other members, with the country weights of prc_hicp_cow. Earlier years take
    those rates with the weights of the first published year. For 2001 to 2007 the same calculation gives the
    published EU27 rates within 0.1 points.
    """
    t = eurostat_table("prc_hicp_aind").filter((pl.col("coicop") == "CP00") & (pl.col("unit") == "INX_A_AVG"))
    eu = (t.filter(pl.col("geo") == "EU27_2020").filter(pl.col("value").is_not_null())
          .select("year", hicp_2015="value").sort("year"))
    first = int(eu["year"].min())
    w = (eurostat_table("prc_hicp_cow").filter((pl.col("statinfo") == "COWEU27_2020") & (pl.col("year") == first) &
                                                (pl.col("geo") != "EU27_2020") & pl.col("value").is_not_null())
         .select("geo", w="value"))
    idx = t.select("geo", "year", h="value")
    rates = (idx.join(idx.with_columns(year=pl.col("year") + 1).rename({"h": "h0"}), on=["geo", "year"])
             .join(w, on="geo").drop_nulls().group_by("year")
             .agg(rate=((pl.col("h") / pl.col("h0") - 1) * pl.col("w")).sum() / pl.col("w").sum(), weight=pl.col("w").sum()))
    rate = {int(y): r for y, r, wt in rates.iter_rows() if wt >= 0.95 * w["w"].sum()}
    level, rows = float(eu["hicp_2015"][0]), []
    y = first
    while y in rate:  # the index of y - 1 from the index of y and the rate of y
        level /= 1 + rate[y]
        y -= 1
        rows.append({"year": y, "hicp_2015": round(level, 2)})
    est = pl.DataFrame(rows, schema={"year": pl.Int64, "hicp_2015": pl.Float64}).with_columns(estimated=pl.lit(True))
    return pl.concat([est, eu.with_columns(pl.col("year").cast(pl.Int64), estimated=pl.lit(False))]).sort("year")


def fetch_eurostat(d: Dataset, store: Path, log) -> None:
    today = date.today().isoformat()
    members = nuts.eu27()
    service = d.id.split("_")[1]
    folder = store / "eurostat" / service
    meta = lambda code: {"source": f"Eurostat {code}", "retrieved": today}  # noqa: E731
    if service == "hicp":
        _save(_hicp(), store / "eurostat" / "shared" / "hicp_eu27.parquet", meta("prc_hicp_aind, prc_hicp_cow"))
    elif service == "carbon":
        t = eurostat_table("env_air_gge")
        t = (t.filter((pl.col("airpol") == "CO2") & (pl.col("unit") == "THS_T") &
                      pl.col("geo").is_in(members + ["EU27_2020"]) & pl.col("year").is_between(1990, 2024) &
                      pl.col("value").is_not_null())
             .select("geo", "year", "src_crf", thousand_t_co2="value").sort("src_crf", "geo", "year"))
        _save(t, folder / "env_air_gge_co2.parquet", meta("env_air_gge"))
    elif service == "wood":
        v = eurostat_table("for_vol_efa")
        v = (v.filter((pl.col("unit") == "THS_M3") & pl.col("stk_flow").is_in(["NAI", "RMOV"]) &
                      pl.col("year").is_between(2000, 2024) & pl.col("value").is_not_null())
             .select("geo", "year", flow="stk_flow", forest_class="indic_fo", thousand_m3="value"))
        _save(v.sort("geo", "year", "flow", "forest_class"), folder / "for_vol_efa_thsm3.parquet", meta("for_vol_efa"))
        s = eurostat_table("for_sup_cp")
        s = (s.filter((pl.col("unit") == "MIO_EUR") & (pl.col("for_acc") == "P1_TR") & pl.col("year").is_between(2000, 2024)
                      & pl.col("value").is_not_null()).select("geo", "year", value_mio_eur="value"))
        _save(s.sort("geo", "year"), folder / "for_sup_cp_p1tr.parquet", meta("for_sup_cp"))
        b = eurostat_table("for_basic")
        b = (b.filter((pl.col("prod_wd") == "RW") & (pl.col("treespec") == "TOTAL") & (pl.col("stk_flow") == "EXP") &
                      pl.col("unit").is_in(["THS_EUR", "THS_M3"]) & pl.col("year").is_between(2000, 2024))
             .pivot(on="unit", index=["geo", "year"], values="value")
             .rename({"THS_EUR": "exports_ths_eur", "THS_M3": "exports_ths_m3"}))
        _save(b.sort("geo", "year"), folder / "for_basic_rw_exports.parquet", meta("for_basic"))
    elif service == "pollination":
        _fetch_eurostat_pollination(folder, members, meta)


PRODUCTION_CODES = [("APPL", "F1110", 1), ("APPL", "F1120", 1), ("APPL", "F1210_1220", 1), ("OFRU", "F0000", 1),
                    ("OFRU", "F1110", -1), ("OFRU", "F1120", -1), ("OFRU", "F1210_1220", -1), ("OFRU", "S0000", 1),
                    ("RAPE", "I1110", 1), ("SUNF", "I1120", 1), ("SOYA", "I1130", 1), ("OOIL", "I1140", 1),
                    ("OOIL", "I1150", 1), ("OOIL", "I1190", 1), ("PULS", "P0000", 1), ("TEXT", "I2100", 1),
                    ("TEXT", "I2200", 1), ("TOMA", "V3100", 1), ("CITR", "T0000", 1)]
OTHER_FRUIT_PARTS = ["F1190", "F1230", "F1240", "F1250", "F1290", "F2000", "F3000", "F4000", "S0000"]


def _window_mean(df: pl.DataFrame) -> pl.DataFrame:
    """Mean of the three years centred on each reporting year. A code with no figure stays missing."""
    win = pl.DataFrame([{"year": y, "source_year": s} for y in (2000, 2006, 2012, 2018, 2024) for s in (y - 1, y, y + 1)])
    return (df.join(win, on="source_year").group_by("geo", "code", "year")
            .agg(value=pl.when(pl.col("value").is_not_null().any()).then(pl.col("value").mean())))


def _fetch_eurostat_pollination(folder: Path, members: list[str], meta) -> None:
    from esa_toolbox.services.pollination.model import crops

    apro = eurostat_table("apro_cpsh1").filter(pl.col("geo").is_in(members))
    codes = pl.DataFrame(PRODUCTION_CODES, schema=["crop", "code", "sign"], orient="row")

    def national(indicator: str, mult: float, name: str) -> pl.DataFrame:
        raw = apro.filter(pl.col("strucpro") == indicator).select("geo", code="crops", source_year="year", value="value")
        parts = (raw.filter(pl.col("code").is_in(["F1210", "F1220"])).group_by("geo", "source_year")
                 .agg(parts=pl.when(pl.col("value").is_not_null().any()).then(pl.col("value").sum())))
        peach = (raw.filter(pl.col("code") == "F1210_1220").join(parts, on=["geo", "source_year"], how="full", coalesce=True)
                 .select("geo", code=pl.lit("F1210_1220"), source_year="source_year", value=pl.coalesce("value", "parts")))
        means = _window_mean(pl.concat([raw.filter(pl.col("code") != "F1210_1220"), peach])
                             .filter(pl.col("code").is_in(codes["code"].to_list() + OTHER_FRUIT_PARTS)))
        groups = (means.join(codes, on="code").group_by("geo", "crop", "year")
                  .agg(value=pl.when(pl.col("value").is_not_null().any())
                       .then(pl.max_horizontal((pl.col("sign") * pl.col("value")).sum(), 0.0))))
        fp = (means.filter(pl.col("code").is_in(OTHER_FRUIT_PARTS)).group_by("geo", "year")
              .agg(parts=pl.when(pl.col("value").is_not_null().any()).then(pl.col("value").sum()))
              .with_columns(crop=pl.lit("OFRU")))
        return (groups.join(fp, on=["geo", "crop", "year"], how="full", coalesce=True)
                .select("geo", "crop", "year", **{name: pl.max_horizontal("value", "parts") * mult}))

    prod = national("HPRD_HUMD_EU_THS_T", 1000, "production_t").join(
        national("AR_THS_HA", 1000, "area_ha"), on=["geo", "crop", "year"], how="full", coalesce=True)
    _save(prod, folder / "production_crop.parquet", meta("apro_cpsh1"))
    items = crops().select(crop="capri_code", code=pl.col("eaa_items").str.split(";")).explode("code")
    eaa = (eurostat_table("aact_eaa02")
           .filter((pl.col("unit") == "MIO_EUR") & (pl.col("indic_agr") == "PRD_BP") & pl.col("geo").is_in(members)
                   & pl.col("am_item").is_in(items["code"].to_list()))
           .select("geo", code="am_item", source_year="year", value="value"))
    eaa = (_window_mean(eaa).join(items, on="code").group_by("geo", "crop", "year")
           .agg(value_meur=pl.when(pl.col("value").is_not_null().any()).then(pl.col("value").sum())))
    _save(eaa, folder / "eaa_value_crop.parquet", meta("aact_eaa02"))


# ---------------------------------------------------------------------------------------------------------
# Rasters read from remote cloud optimised files
# ---------------------------------------------------------------------------------------------------------

STAC = "https://resources.integratedmodelling.org/aries-public-stac/jrc_data_catalog/"
CLIMATE_WINDOW = {2000: "1999-2001", 2006: "2005-2007", 2012: "2011-2013", 2018: "2017-2019", 2024: "2023-2025"}
MONTHS = ["April", "May", "June", "July", "August", "September"]
SHORT = {"April": "Apr", "May": "Mai", "June": "Jun", "July": "Jul", "August": "Aug", "September": "Sep"}


def _read_remote(url: str):
    import rasterio

    with rasterio.open(f"/vsicurl/{url}") as s:
        # Integer files: convert before filling the gaps, and apply scale and offset as terra does.
        a = s.read(1, masked=True).astype("float64") * s.scales[0] + s.offsets[0]
        return a.filled(np.nan), s.transform, s.crs


def _round_half_away(a: np.ndarray, digits: int) -> np.ndarray:
    """Round as terra does: in double precision, halves away from zero. numpy rounds halves to
    even, which moved about 2 percent of the CAPRI cells by 0.001."""
    f = 10.0 ** digits
    return np.sign(a) * np.floor(np.abs(a) * f + 0.5) / f


def fetch_activity(d: Dataset, store: Path, log) -> None:
    """Growing season activity index: each month on the climate grid, limited to 0 to 100, then averaged."""
    from rasterio.warp import Resampling, reproject

    g = POLLINATION_GRID
    regs = nuts.countries(store)
    import shapely

    # As terra does: buffer() with 10 segments per quarter circle, and mask() on every cell touched.
    near = rio.rasterize([(shapely.buffer(x, 5000, quad_segs=10), 1) for x in regs.geoms], g, all_touched=True,
                         dtype="uint8")
    for year in d.years:
        out = store / "catalogue" / f"activity_index_{year}_1km.tif"
        if out.exists():
            continue
        win = CLIMATE_WINDOW[year]
        acc, n = None, 0
        for m in MONTHS:
            lab = SHORT[m] if win == "2011-2013" else m
            t, tr, crs = _read_remote(f"{STAC}air_temp/t2m_{win}_{lab}_cog.tif")
            r, _, _ = _read_remote(f"{STAC}solar_radiation/ssrd_{win}_{m}_cog.tif")
            bg = -0.62 + 1.027 * (t * 0.1 - 273.15) + 0.006 * (r * 0.01)
            a = np.clip(-39.3 + 4.01 * bg, 0, 100)
            acc = a if acc is None else acc + a
            n += 1
        # Double precision up to the rounding, as terra computes it.
        mean = acc / n
        dst = np.full(g.shape, np.nan, dtype="float64")
        reproject(mean, dst, src_transform=tr, src_crs=crs, dst_transform=g.transform, dst_crs=rio.CRS_3035,
                  src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.bilinear, tolerance=0.0)
        dst = _round_half_away(dst, 2).astype("float32")
        dst[near == 0] = np.nan
        rio.write(out, dst, g, names=["activity_index"])
        log(f"  wrote {out}")


def fetch_sdm(d: Dataset, store: Path, log) -> None:
    """Bumblebee model, bilinear from 10 km to 1 km, coastal gaps filled in two passes of a 21 by 21 mean."""
    from esa_toolbox.services.pollination.sdm import fill_coast

    g = POLLINATION_GRID
    out = store / "catalogue" / "bumblebee_sdm_1km.tif"
    a = rio.warp(f"/vsicurl/{STAC}bumblebee_species_distribution/bumblebee_composite_richness_eu_cog.tif", g, "bilinear")
    rio.write(out, fill_coast(a), g, names=["bumblebee_sdm"])
    log(f"  wrote {out}")


# ---------------------------------------------------------------------------------------------------------
# E-OBS and the bioclimatic layers of the bumblebee model
# ---------------------------------------------------------------------------------------------------------

EOBS = "https://knmi-ecad-assets-prd.s3.amazonaws.com/ensembles/data/Grid_0.25deg_reg_ensemble/{v}_ens_mean_0.25deg_reg_v32.0e.nc"
EOBS_START = date(1950, 1, 1)


def bioclim(tx: np.ndarray, tn: np.ndarray, rr: np.ndarray) -> np.ndarray:
    """bio04, bio05, bio08 and bio15 from twelve monthly means of daily maximum and minimum temperature (°C)
    and twelve monthly totals of precipitation (mm), each of shape (12, rows, cols).

    The definitions are those of WorldClim. bio04 is the standard deviation of the monthly mean temperature
    times 100, bio05 the highest monthly maximum temperature, bio08 the mean temperature of the wettest three
    consecutive months, with December followed by January, and bio15 the coefficient of variation of monthly
    precipitation in percent.
    """
    tm = (tx + tn) / 2
    quarter = lambda a: a + np.roll(a, -1, axis=0) + np.roll(a, -2, axis=0)  # noqa: E731
    wettest = np.where(np.isnan(rr).any(axis=0), 0, np.nan_to_num(quarter(rr), nan=-1.0).argmax(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.stack([tm.std(axis=0, ddof=1) * 100, tx.max(axis=0),
                         np.take_along_axis(quarter(tm) / 3, wettest[None], axis=0)[0],
                         100 * rr.std(axis=0, ddof=1) / rr.mean(axis=0)])


def _eobs_monthly(path: Path, var: str, first: int, last: int, total: bool) -> tuple[np.ndarray, object]:
    """Monthly means (or totals) of a daily E-OBS file for each year, as (years, 12, rows, cols)."""
    import rasterio

    out = []
    with rasterio.open(f"netcdf:{path}:{var}") as s:
        for y in range(first, last + 1):
            months = []
            for m in range(1, 13):
                b0 = (date(y, m, 1) - EOBS_START).days + 1
                b1 = (date(y + (m == 12), m % 12 + 1, 1) - EOBS_START).days + 1
                a = s.read(list(range(b0, b1)), masked=True).astype("float64") * s.scales[0] + s.offsets[0]
                months.append((a.sum(axis=0) if total else a.mean(axis=0)).filled(np.nan))
            out.append(months)
        return np.array(out, dtype="float32"), s.transform


def fetch_eobs(d: Dataset, store: Path, log) -> None:
    """The four bioclimatic layers of each reporting year on the 10 km grid, from the 22 years that end in it.

    The three daily files (2.1 GB) go to a temporary folder of the store and are removed at the end.
    """
    from rasterio.warp import Resampling, reproject
    from scipy import ndimage

    from esa_toolbox.services.pollination.sdm import CLIMATE, CLIMATE_YEARS, GRID10, climate_file

    raw = store / "eobs" / "_download"
    first, last = min(d.years) - CLIMATE_YEARS + 1, max(d.years)
    monthly = {}
    for v in ("tx", "tn", "rr"):
        f = raw / Path(EOBS.format(v=v)).name
        if not f.exists():
            _get(EOBS.format(v=v), f, log)
        log(f"  monthly values of {v}, {first} to {last}")
        monthly[v], tr = _eobs_monthly(f, v, first, last, total=v == "rr")
    import warnings

    for year in d.years:
        i1 = year - first + 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # sea cells have no value in any year
            clim = [np.nanmean(monthly[v][i1 - CLIMATE_YEARS:i1].astype("float64"), axis=0) for v in ("tx", "tn", "rr")]
        src = bioclim(*clim)
        dst = np.full((len(CLIMATE), *GRID10.shape), np.nan, dtype="float64")
        reproject(src, dst, src_transform=tr, src_crs="EPSG:4326", dst_transform=GRID10.transform,
                  dst_crs=rio.CRS_3035, src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.bilinear,
                  tolerance=0.0)
        for k in range(len(CLIMATE)):  # E-OBS stops at the coast: cells beyond it take the nearest value
            idx = ndimage.distance_transform_edt(np.isnan(dst[k]), return_distances=False, return_indices=True)
            dst[k] = dst[k][tuple(idx)]
        rio.write(climate_file(store, year), dst.astype("float32"), GRID10, names=list(CLIMATE))
        log(f"  wrote {climate_file(store, year)}")
    shutil.rmtree(raw)


def fetch_capri(d: Dataset, store: Path, log) -> None:
    """Area and yield of the ten crops, one band per crop, three decimals, on the 1 km grid."""
    import rasterio

    from esa_toolbox.services.pollination.model import crops

    g = POLLINATION_GRID
    names = crops()["capri_code"].to_list()
    for kind, folder in (("area_ha", "agricultural_produce_area"), ("yield_kgha", "agricultural_yield")):
        for year in (2000, 2006, 2012, 2018):
            out = store / "capri" / f"capri_{kind}_{year}.tif"
            if out.exists():
                continue
            bands = []
            for c in names:
                with rasterio.open(f"/vsicurl/{STAC}{folder}/{c}_{year}_cog.tif") as s:
                    # The source grid has the grid origin, up to rounding noise in its header.
                    left, top = round(s.bounds.left), round(s.bounds.top)
                    a = s.read(1, masked=True).filled(np.nan).astype("float64")
                full = np.full(g.shape, np.nan, dtype="float64")
                r0, c0 = round((g.ymax - top) / 1000), round((left - g.xmin) / 1000)
                rs, cs = max(r0, 0), max(c0, 0)
                re_, ce = min(r0 + a.shape[0], g.height), min(c0 + a.shape[1], g.width)
                full[rs:re_, cs:ce] = a[rs - r0:re_ - r0, cs - c0:ce - c0]
                bands.append(_round_half_away(full, 3).astype("float32"))
            rio.write(out, np.stack(bands), g, names=names)
            log(f"  wrote {out}")


@contextmanager
def _environ(values: dict[str, str]):
    old = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _europe_window(transform, bounds=(-32.0, 27.0, 45.0, 72.0)):
    """The window over Europe that terra::crop takes.

    The edges fall half a cell into the GDMP grids, and terra settles such a tie toward the lower coordinate on
    both axes. rasterio's round_offsets() settles the top edge the other way, which moved the map one row north.
    """
    import math

    from rasterio.windows import Window

    xmin, ymin, xmax, ymax = bounds
    xres, yres = transform.a, -transform.e
    col = math.floor((xmin - transform.c) / xres + 0.5 - 1e-6)
    row = math.floor((transform.f - ymax) / yres + 0.5 + 1e-6)
    return Window(col, row, round((xmax - xmin) / xres), round((ymax - ymin) / yres))


def fetch_gdmp(d: Dataset, store: Path, log) -> None:
    """Annual Gross Dry Matter Productivity: the sum of 36 dekads, each times its days, over Europe."""
    import rasterio

    access, secret = os.environ.get("CDSE_S3_ACCESS_KEY"), os.environ.get("CDSE_S3_SECRET_KEY")
    if not (access and secret):
        raise EsaError("Set CDSE_S3_ACCESS_KEY and CDSE_S3_SECRET_KEY to fetch the annual productivity.")
    cat_url = ("https://s3.waw3-1.cloudferro.com/swift/v1/CatalogueCSV/bio-geophysical/"
               "dry-gross_dry_matter_productivity/{0}/{0}_cog.csv")
    root = "/vsis3/eodata/CLMS/bio-geophysical/dry-gross_dry_matter_productivity"
    env = dict(AWS_ACCESS_KEY_ID=access, AWS_SECRET_ACCESS_KEY=secret, AWS_S3_ENDPOINT="eodata.dataspace.copernicus.eu",
               AWS_VIRTUAL_HOSTING="FALSE", AWS_HTTPS="YES", AWS_DEFAULT_REGION="default",
               # A read that receives nothing for a minute is retried, instead of waiting forever (seen on
               # 2026-11-09). Not listing the folder of each file saves one request per file.
               GDAL_HTTP_LOW_SPEED_TIME="60", GDAL_HTTP_LOW_SPEED_LIMIT="1", GDAL_HTTP_MAX_RETRY="5",
               GDAL_HTTP_RETRY_DELAY="10", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR")
    import requests

    for p in d.missing(store):
        product, year = re.match(r"gdmp_(1km|300m)_(\d{4})_4326", p.stem).groups()
        dataset = "gdmp_global_1km_10daily_v2" if product == "1km" else "gdmp_global_300m_10daily_v2"
        cat = pl.read_csv(io.StringIO(requests.get(cat_url.format(dataset), timeout=300, headers=UA).text), separator=";")
        cat = cat.filter(pl.col("nominal_date").str.slice(0, 4) == year)
        if cat["name"].str.contains("-RT6_").any():
            cat = cat.filter(pl.col("name").str.contains("-RT6_"))
        cat = cat.with_columns(date=pl.col("nominal_date").str.slice(0, 10).str.to_date()).sort("date")
        if cat.height != 36:
            raise EsaError(f"GDMP {product} {year}: the catalogue lists {cat.height} dekads, not 36")
        prev = [date(int(year) - 1, 12, 31)] + cat["date"].to_list()[:-1]
        total = None
        # rasterio refuses AWS credentials as Env options, so GDAL reads them from the process environment.
        with _environ(env), rasterio.Env():
            for name, dt, pd in zip(cat["name"], cat["date"], prev):
                f = re.sub(r"^(c_gls_GDMP300|c_gls_GDMP)", r"\1-GDMP", name).replace("_cog", ".tiff")
                with rasterio.open(f"{root}/{dataset}/{dt:%Y/%m/%d}/{name}/{f}") as s:
                    w = _europe_window(s.transform)
                    # The files store whole numbers with a scale and offset, which terra applies.
                    raw = s.read(1, window=w, masked=True).astype("float64") * s.scales[0] + s.offsets[0]
                    if product == "300m":
                        # A 300 m dekad does not fit in terra's memory, so the first implementation held the scaled
                        # values in a temporary float32 file before multiplying by the days.
                        raw = raw.astype("float32").astype("float64")
                    # The first implementation stored each dekad as float32 before adding them up, so each is rounded here.
                    a = (raw.filled(0.0) * (dt - pd).days).astype("float32").astype("float64")
                    tr = s.window_transform(w)
                total = a if total is None else total + a
                log(f"  {product} {dt}")
        import rasterio as rio2

        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".part.tif")
        with rio2.open(tmp, "w", driver="GTiff", height=total.shape[0], width=total.shape[1], count=1, dtype="float32",
                       crs="EPSG:4326", transform=tr, nodata=np.nan, compress="DEFLATE", predictor=3, tiled=True) as dst:
            dst.write(total.astype("float32"), 1)
        os.replace(tmp, p)


def fetch_roads(d: Dataset, store: Path, log) -> None:
    """OpenStreetMap major roads from the k.LAB geoserver in pages, rasterised at 100 m where a road touches a cell."""
    import shapely
    from rasterio.warp import transform_geom

    wfs = "https://integration.integratedmodelling.org/geoserver/im-data-global-infrastructure/ows"
    layer = "im-data-global-infrastructure:osm_main_roads_laea_4326"
    major = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link"}
    page_dir = store / "osm" / "pages"
    page_dir.mkdir(parents=True, exist_ok=True)
    start, size = 0, 100_000
    while True:
        page = page_dir / f"page_{start:08d}.json"
        if not page.exists():
            url = (f"{wfs}?service=WFS&version=2.0.0&request=GetFeature&typeNames={layer}&outputFormat=application/json"
                   f"&srsName=EPSG:4326&sortBy=__gid&count={size}&startIndex={start}")
            _get(url, page, log)
        n = len(json.loads(page.read_text())["features"])
        if n < size:
            break
        start += size
    geoms = []
    for page in sorted(page_dir.glob("page_*.json")):
        for f in json.loads(page.read_text())["features"]:
            if f["properties"].get("highway") in major and f["geometry"]:
                geoms.append(shapely.geometry.shape(transform_geom("EPSG:4326", "EPSG:3035", f["geometry"])))
    from esa_toolbox.grid import Grid

    g = Grid(900000, 900000, 7400000, 5500000, 100)
    out = store / "osm" / "osm_major_roads_100m.tif"
    tree = shapely.STRtree(geoms)
    import rasterio

    tmp = out.with_name(out.name + ".part.tif")
    with rasterio.open(tmp, "w", driver="GTiff", height=g.height, width=g.width, count=1, dtype="uint8", nodata=255,
                       crs=rio.CRS_3035, transform=g.transform, compress="DEFLATE", tiled=True, BIGTIFF="YES") as dst:
        for t in g.tiles(5000):
            idx = tree.query(shapely.box(*t.bounds))
            a = rio.rasterize([(geoms[i], 1) for i in idx], t, all_touched=True, dtype="uint8")
            r0, c0 = t.offset_in(g)
            dst.write(a[None], window=rasterio.windows.Window(c0, r0, t.width, t.height))
    os.replace(tmp, out)
    # The pages (about 1.4 GB) only let an interrupted fetch resume.
    shutil.rmtree(page_dir)
    log(f"  wrote {out}")


FETCHERS = {"http": fetch_http, "eea_datashare": fetch_clc, "gisco": fetch_gisco, "eurostat": fetch_eurostat,
            "stac_activity": fetch_activity, "stac_sdm": fetch_sdm, "eobs": fetch_eobs, "stac_capri": fetch_capri,
            "copernicus_s3": fetch_gdmp, "wfs_roads": fetch_roads}
