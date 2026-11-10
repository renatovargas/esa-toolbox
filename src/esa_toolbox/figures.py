"""Figures of the reports, drawn with Matplotlib in one restrained style."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import shapely  # noqa: E402
from matplotlib.collections import PathCollection  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

GREY = "#52514e"
plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888888",
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def lines(series: dict[str, tuple[list, list]], colours: dict[str, str], ylabel: str, path: Path,
          size=(7.5, 3.8)) -> Path:
    """One line per series, with a direct label at its end."""
    fig, ax = plt.subplots(figsize=size)
    for name, (x, y) in series.items():
        ax.plot(x, y, color=colours.get(name), lw=1.6, label=name)
        ax.annotate(name, (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points", va="center",
                    color=GREY, fontsize=8)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="x", visible=False)
    xs = [v for x, _ in series.values() for v in x]
    ax.set_xlim(min(xs) - 0.5, max(xs) + (max(xs) - min(xs)) * 0.16)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(series))
    return _save(fig, path)


def dodged_bars(labels: list[str], values: dict[str, list[float]], colours: dict[str, str], xlabel: str,
                path: Path, size=(7, 6.2)) -> Path:
    """Horizontal bars, two or more per label, labels in the order given from bottom to top."""
    fig, ax = plt.subplots(figsize=size)
    y = np.arange(len(labels))
    k = len(values)
    h = 0.82 / k
    for i, (name, v) in enumerate(values.items()):
        ax.barh(y + (k - 1 - 2 * i) * h / 2, v, height=h * 0.9, color=colours.get(name), label=name)
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.margins(y=0.01)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=k)
    return _save(fig, path)


def dots(rows: list[str], panels: dict[str, list[float]], xlabel: str, path: Path, *, colours=None,
         vline: float | None = None, size=(7.5, 5.5)) -> Path:
    """One small panel per year with a dot per row (country)."""
    fig, axes = plt.subplots(1, len(panels), figsize=size, sharey=True)
    for ax, (title, vals) in zip(np.atleast_1d(axes), panels.items()):
        c = colours[title] if colours else "#2f6b4f"
        ax.scatter(vals, np.arange(len(rows)), s=8, c=c)
        if vline is not None:
            ax.axvline(vline, ls="--", color="#666666", lw=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_yticks(np.arange(len(rows)), rows, fontsize=7)
        ax.tick_params(axis="x", labelsize=7)
    fig.supxlabel(xlabel, fontsize=9)
    return _save(fig, path)


def _outline_paths(geoms, tolerance: float = 2000.0) -> PathCollection:
    paths = []
    for g in geoms:
        g = shapely.simplify(g, tolerance)
        polys = getattr(g, "geoms", [g])
        for p in polys:
            for ring in [p.exterior, *p.interiors]:
                c = np.asarray(ring.coords)[:, :2]
                paths.append(MPath(c))
    return PathCollection(paths, facecolor="none", edgecolor="#333333", linewidth=0.15)


def map_panels(panels: dict[str, np.ndarray], extent: tuple[float, float, float, float], path: Path, *,
               outlines=None, cmap="viridis", legend="", transform="sqrt", classes=None,
               class_colours=None, class_labels=None, ncols=3, size=(7.5, 5.5),
               view=(2.6e6, 6.6e6, 1.4e6, 5.4e6)) -> Path:
    """Maps on a common colour scale, one per panel. `extent` is xmin, xmax, ymin, ymax of the arrays."""
    from matplotlib.colors import BoundaryNorm, ListedColormap, PowerNorm

    n = len(panels)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, min(ncols, n), figsize=size, squeeze=False)
    finite = np.concatenate([a[np.isfinite(a) & (a > 0)] for a in panels.values()]) if classes is None else None
    norm = None
    if classes is not None:
        cm = ListedColormap(class_colours)
        norm = BoundaryNorm(classes, cm.N)
    else:
        cm = plt.get_cmap(cmap)
        vmax = float(np.nanpercentile(finite, 99.5)) if finite.size else 1.0
        norm = PowerNorm(0.5, vmin=0, vmax=vmax) if transform == "sqrt" else None
    im = None
    for ax, (title, a) in zip(axes.ravel(), panels.items()):
        shown = a.astype("float32").copy()
        if classes is None:
            shown[~(shown > 0)] = np.nan
        im = ax.imshow(shown, extent=extent, cmap=cm, norm=norm, interpolation="nearest")
        if outlines is not None:
            ax.add_collection(_outline_paths(outlines))
        ax.set_xlim(view[0], view[1])
        ax.set_ylim(view[2], view[3])
        ax.set_title(title, fontsize=9)
        ax.set_axis_off()
    for ax in axes.ravel()[n:]:
        ax.set_axis_off()
    if classes is not None and class_labels:
        from matplotlib.patches import Patch

        handles = [Patch(color=c, label=lab) for c, lab in zip(class_colours, class_labels)]
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8)
    elif im is not None:
        cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.02)
        cb.set_label(legend, fontsize=8)
        cb.ax.tick_params(labelsize=7)
    return _save(fig, path)
