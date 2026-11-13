"""Equations of the crop pollination accounts, as plain functions over arrays.

Equations and parameters follow Vallecillo et al. (2018), Appendix III. Each function works on one block of
the grid, and the block carries a margin wherever a neighbourhood enters (forest core, distance).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import polars as pl
from scipy import ndimage

from esa_toolbox.core import read_param

DECAY = 0.00046  # per metre (Ricketts et al., 2008)
MAX_DISTANCE = 20_000  # metres
SPA_THRESHOLD = 0.2
FOREST = (311, 312, 313)
CLASS_LIMITS = (0.1, 0.2, 0.3)
CLASS_LABELS = ("none", "low", "medium", "high")


@dataclass(frozen=True)
class Scores:
    suitability: np.ndarray  # code -> (floral + nesting) / 2, NaN for codes with no score
    agro: np.ndarray  # code -> bool
    seminatural: np.ndarray  # code -> bool
    ecosystem: np.ndarray  # code -> index into `ecosystems`, -1 for none
    ecosystems: tuple[str, ...]


@lru_cache(maxsize=1)
def scores() -> Scores:
    t = read_param("clc_pollination_scores.csv")
    ecos = tuple(t["ecosystem"].unique(maintain_order=True).to_list())
    suit = np.full(65536, np.nan, dtype="float32")
    agro = np.zeros(65536, dtype=bool)
    semi = np.zeros(65536, dtype=bool)
    eco = np.full(65536, -1, dtype="int16")
    for r in t.iter_rows(named=True):
        c = int(r["clc"])
        suit[c] = (float(r["floral"]) + float(r["nesting"])) / 2
        agro[c] = int(r["agro"]) == 1
        semi[c] = int(r["seminatural"]) == 1
        eco[c] = ecos.index(r["ecosystem"])
    return Scores(suit, agro, semi, eco, ecos)


def crops(method: str = "revised") -> pl.DataFrame:
    """Crop groups and their dependency ratio: Siopa et al. (2024), or Klein et al. (2007) in the original method."""
    t = read_param("crop_dependency.csv")
    if method == "original":
        t = t.with_columns(dependency=pl.col("dependency_klein2007"), dependency_class=pl.col("class_klein2007"))
    return t.select("capri_code", "crop", "dependency", "dependency_class", "eaa_items", "eurostat_group")


def dependency_class(ratio: float) -> str:
    """The classes of Klein et al. (2007), with the labels of the accounts."""
    return "Essential" if ratio >= 0.9 else "High" if ratio >= 0.4 else "Moderate" if ratio >= 0.1 else "Low"


def crop_components() -> pl.DataFrame:
    """The crops of Siopa et al. (2024) behind each crop group, and their dependency ratios."""
    return read_param("crop_dependency_siopa2024.csv")


def forest_core(clc: np.ndarray) -> np.ndarray:
    """Forest cells whose eight neighbours are forest. Cells beyond the block count as non-forest."""
    f = np.isin(clc, FOREST)
    e = f.copy()
    e[1:] &= f[:-1]
    e[:-1] &= f[1:]
    core = e.copy()
    core[:, 1:] &= e[:, :-1]
    core[:, :-1] &= e[:, 1:]
    core[[0, -1], :] = False
    core[:, [0, -1]] = False
    return core


def exact_distance(clc: np.ndarray, res: float = 100.0) -> np.ndarray:
    """Euclidean distance in metres from each cell centre to the nearest semi-natural cell, NaN beyond 20 km."""
    target = scores().seminatural[clc]
    if not target.any():
        return np.full(clc.shape, np.nan, dtype="float32")
    d = ndimage.distance_transform_edt(~target, sampling=res).astype("float32")
    d[d > MAX_DISTANCE] = np.nan
    return d


def expert_model(clc: np.ndarray, roads: np.ndarray, distance: np.ndarray, activity: np.ndarray) -> np.ndarray:
    """Expert-based suitability between 0 and 1. NaN where Corine has no score.

    Multiplying by zero, as the earlier routine did with `0 * s`, keeps no-data as no-data.
    """
    s = scores().suitability[clc]
    s[roads == 1] *= 0
    s[forest_core(clc)] *= 0
    far = np.isnan(distance)
    s = s * np.exp(-DECAY * np.where(far, 0, distance)).astype("float32")
    s[far] *= 0
    return s * activity / 100


def potential_classes(p: np.ndarray) -> np.ndarray:
    """0 none, 1 low, 2 medium, 3 high, and 255 where potential is missing."""
    out = np.digitize(p, CLASS_LIMITS, right=False).astype("uint8")
    out[np.isnan(p)] = 255
    return out


def allocate(value_1km: np.ndarray, weight: np.ndarray, land: np.ndarray) -> np.ndarray:
    """Spread 1 km totals over 100 m cells in proportion to a weight, keeping each 1 km total.

    A 1 km cell with no positive weight spreads its value evenly over its land cells. `land` is 1 on land
    and NaN elsewhere, and the result is NaN off land.
    """
    from esa_toolbox.rio import block_sum

    w_sum = block_sum(weight, 10)
    l_sum = block_sum(land, 10)
    rep = lambda a: np.repeat(np.repeat(a, 10, axis=0), 10, axis=1)  # noqa: E731
    with np.errstate(invalid="ignore", divide="ignore"):
        per_w = rep(value_1km / w_sum)
        per_l = rep(value_1km / l_sum)
    use_w = rep(w_sum > 0)
    out = np.where(use_w, per_w * np.nan_to_num(weight, nan=0.0), per_l * land)
    out[np.isnan(rep(w_sum))] = np.nan
    return out


def allocation_share(weight: np.ndarray, land: np.ndarray) -> np.ndarray:
    """The share of its 1 km total that each 100 m cell receives, so that `allocate` is one multiplication.

    Same rule as `allocate`: in proportion to the weight, or evenly over land where a kilometre has no positive
    weight, and NaN off land.
    """
    from esa_toolbox.rio import block_sum

    w_sum = block_sum(weight, 10)
    l_sum = block_sum(land, 10)
    rep = lambda a: np.repeat(np.repeat(a, 10, axis=0), 10, axis=1)  # noqa: E731
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(rep(w_sum > 0), np.nan_to_num(weight, nan=0.0) / rep(w_sum), land / rep(l_sum))
    share[np.isnan(rep(w_sum))] = np.nan
    return share


def spread_1km(value_1km: np.ndarray, share: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(value_1km, 10, axis=0), 10, axis=1) * share
