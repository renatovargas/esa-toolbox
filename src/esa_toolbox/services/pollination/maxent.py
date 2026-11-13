"""Maxent models read from their lambda files and applied to new layers, with no Java.

A lambda file lists the features of one fitted model and four constants. The bumblebee models of
Vallecillo et al. (2018) hold linear and hinge features only, so these are the features read here, and any
other kind stops the run. The equations follow Maxent 3.4 (Phillips et al., 2017) and Wilson (2009).

- A plain line `name, lambda, min, max` is a linear feature, (x - min) / (max - min). Its minimum and maximum
  are the range of the layer in the training data, and every layer is limited to that range before use
  (clamping, the default of Maxent).
- `'name, lambda, min, max` is a forward hinge, (x - min) / (max - min) above `min` and 0 below.
- `` `name, lambda, min, max`` is a reverse hinge, (max - x) / (max - min) below `max` and 0 above.

With S the sum of lambda times feature, raw = exp(S - linearPredictorNormalizer) / densityNormalizer, and the
complementary log-log output is 1 - exp(-exp(entropy) * raw).
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CONSTANTS = ("linearPredictorNormalizer", "densityNormalizer", "numBackgroundPoints", "entropy")


@dataclass(frozen=True)
class Model:
    name: str
    ranges: dict[str, tuple[float, float]]  # layer -> training minimum and maximum
    features: tuple[tuple[str, str, float, float, float], ...]  # kind, layer, lambda, min, max
    constants: dict[str, float]

    @property
    def layers(self) -> tuple[str, ...]:
        return tuple(self.ranges)


def parse(text: str, name: str = "") -> Model:
    ranges: dict[str, tuple[float, float]] = {}
    features = []
    constants: dict[str, float] = {}
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] in CONSTANTS:
            constants[parts[0]] = float(parts[1])
        elif len(parts) == 4:
            key, lam, lo, hi = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
            if key[0] == "'":
                features.append(("hinge", key[1:], lam, lo, hi))
            elif key[0] == "`":
                features.append(("reverse", key[1:], lam, lo, hi))
            elif key[0].isalpha() and not any(c in key for c in "*^()<"):
                ranges[key] = (lo, hi)
                features.append(("linear", key, lam, lo, hi))
            else:
                raise ValueError(f"{name}: feature {key!r} is neither linear nor hinge")
        elif line.strip():
            raise ValueError(f"{name}: cannot read the line {line!r}")
    missing = set(CONSTANTS) - set(constants)
    if missing:
        raise ValueError(f"{name}: no {', '.join(sorted(missing))}")
    return Model(name, ranges, tuple(f for f in features if f[2] != 0.0), constants)


def read_zip(path: Path) -> list[Model]:
    """Every model of a zip of lambda files, sorted by name. The name is the file name without its suffix."""
    out = []
    with zipfile.ZipFile(path) as z:
        for n in sorted(z.namelist(), key=lambda n: Path(n).stem):
            if n.endswith(".lambdas"):
                out.append(parse(io.TextIOWrapper(z.open(n), encoding="utf-8").read(), Path(n).stem))
    return out


def cloglog(model: Model, layers: dict[str, np.ndarray]) -> np.ndarray:
    """Complementary log-log output of a model, float64, NaN where a layer is missing."""
    clamped = {k: np.clip(layers[k].astype("float64"), lo, hi) for k, (lo, hi) in model.ranges.items()}
    s = np.zeros(next(iter(clamped.values())).shape, dtype="float64")
    for kind, layer, lam, lo, hi in model.features:
        x = clamped[layer]
        if kind == "reverse":
            f = np.where(x < hi, (hi - x) / (hi - lo), 0.0)
        elif kind == "hinge":
            f = np.where(x > lo, (x - lo) / (hi - lo), 0.0)
        else:
            f = (x - lo) / (hi - lo)
        s += lam * f
    c = model.constants
    raw = np.exp(s - c["linearPredictorNormalizer"]) / c["densityNormalizer"]
    out = 1.0 - np.exp(-np.exp(c["entropy"]) * raw)
    for x in clamped.values():
        out[np.isnan(x)] = np.nan
    return out
