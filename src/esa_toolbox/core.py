"""Settings, the run context, checks, facts and the stage runner shared by every service.

A service is a module with a `SERVICE` description and an ordered list of stages. A stage is a plain function
of the context. The runner records, for every stage, when it ran, how long it took and a fingerprint of the
code and settings it ran with, so `esa status` can tell what is up to date.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import os
import platform
import shutil
import time
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import polars as pl

from esa_toolbox import __version__

SERVICES = ("pollination", "wood", "carbon")
METHOD = "revised"  # the method of a normal run. A service lists the others it offers in `METHODS`
YEARS = (2000, 2006, 2012, 2018, 2024)
PARAMS = Path(__file__).parent / "params"


class EsaError(Exception):
    """An error the user can act on. Exit code 3 means inputs or settings that cannot be used."""

    exit_code = 3


class CheckFailed(EsaError):
    exit_code = 1


# ---------------------------------------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------------------------------------

QUARTO_CANDIDATES = (
    "/usr/share/positron/resources/app/quarto/bin/quarto",
    "/usr/lib/rstudio/resources/app/bin/quarto/bin/quarto",
    "/Applications/Positron.app/Contents/Resources/app/quarto/bin/quarto",
    "/Applications/RStudio.app/Contents/Resources/app/quarto/bin/quarto",
    r"C:\Program Files\Positron\resources\app\quarto\bin\quarto.exe",
    r"C:\Program Files\Quarto\bin\quarto.exe",
)


@dataclass
class Settings:
    """Where things live and how a run behaves. Read from esa.toml, then ESA_* variables, then options."""

    store: Path = Path("store")
    outputs: Path = Path("outputs")
    work: Path = Path("work")
    workers: int = 0
    quarto: str | None = None
    formats: tuple[str, ...] = ("html", "pdf", "docx")
    method: str = METHOD
    workers_explicit: bool = False

    @classmethod
    def load(cls, path: Path | None = None, **overrides: Any) -> "Settings":
        s = cls()
        path = path or Path("esa.toml")
        if path.exists():
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            for k, v in data.get("esa", data).items():
                if hasattr(s, k):
                    setattr(s, k, v)
        for k in ("store", "outputs", "work", "workers", "quarto"):
            env = os.environ.get(f"ESA_{k.upper()}")
            if env:
                setattr(s, k, env)
        for k, v in overrides.items():
            if v is not None:
                setattr(s, k, v)
        s.store, s.outputs, s.work = Path(s.store), Path(s.outputs), Path(s.work)
        s.workers_explicit = bool(int(s.workers))
        s.workers = int(s.workers) or default_workers()
        s.formats = tuple(s.formats)
        return s

    def folder(self, service: str) -> str:
        """The folder of a service in work/ and outputs/. Another method than the normal one has its own."""
        return service if self.method == METHOD else f"{service}_{self.method}"

    def quarto_path(self) -> str | None:
        for c in (self.quarto, os.environ.get("QUARTO_PATH"), shutil.which("quarto"), *QUARTO_CANDIDATES):
            if c and Path(c).exists():
                return str(c)
        return None


def available_memory_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    try:  # macOS and other systems without /proc: three quarters of the physical memory
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") * 0.75 / 1e9
    except (OSError, ValueError, AttributeError):
        return 8.0


def default_workers(memory_per_worker_gb: float = 2.0) -> int:
    cpus = os.cpu_count() or 2
    return max(1, min(cpus - 1, int(available_memory_gb() / memory_per_worker_gb)))


# ---------------------------------------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------------------------------------


def _fmt(value: float, digits: int) -> str:
    """Format a number like the reports did in R: rounded, with thousands separators."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{round(value, digits):,.{digits}f}"


@dataclass
class Context:
    service: str
    settings: Settings
    years: tuple[int, ...] = YEARS
    log: Callable[[str], None] = print
    progress: Callable[[str, int, int], None] | None = None
    checks: list[dict] = field(default_factory=list)

    # Folders -------------------------------------------------------------------------------------------
    @property
    def store(self) -> Path:
        return self.settings.store

    @property
    def out(self) -> Path:
        return self.settings.outputs / self.settings.folder(self.service)

    @property
    def work(self) -> Path:
        return self.settings.work / self.settings.folder(self.service)

    @property
    def method(self) -> str:
        return self.settings.method

    def path(self, *parts: str) -> Path:
        p = self.work.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def output(self, *parts: str) -> Path:
        p = self.out.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def need(self, *parts: str) -> Path:
        """A file in the store, with a clear message when it is missing."""
        p = self.store.joinpath(*parts)
        if not p.exists():
            raise EsaError(f"Input missing: {p}. Run `esa fetch {self.service}` or check the store setting.")
        return p

    # Tables --------------------------------------------------------------------------------------------
    def save(self, df: pl.DataFrame, name: str) -> pl.DataFrame:
        """A table that a later stage reads, in work/<service>/tables/."""
        df.write_parquet(self.path("tables", f"{name}.parquet"))
        return df

    def load(self, name: str) -> pl.DataFrame:
        p = self.work / "tables" / f"{name}.parquet"
        if not p.exists():
            raise EsaError(f"{p} is missing. Run the earlier stages of {self.service} first.")
        return pl.read_parquet(p)

    # Report material -----------------------------------------------------------------------------------
    def _facts_file(self) -> Path:
        return self.path("report", "facts.json")

    def facts(self, **values: tuple[float, int] | tuple[float, int, str] | str | float) -> None:
        """Named numbers for the report. Each value is (number, digits) or text."""
        f = self._facts_file()
        data = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        for k, v in values.items():
            if isinstance(v, tuple):
                num, digits = float(v[0]), int(v[1])
                data[k] = {"value": num, "text": _fmt(num, digits)}
            elif isinstance(v, str):
                data[k] = {"value": v, "text": v}
            else:
                data[k] = {"value": float(v), "text": _fmt(float(v), 1)}
        f.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")

    def table(self, df: pl.DataFrame, name: str, digits: int | dict[str, int] = 1) -> pl.DataFrame:
        """A small table for the report, as Markdown, and as Parquet for the record."""
        df.write_parquet(self.path("report", "tables", f"{name}.parquet"))
        self.path("report", "tables", f"{name}.md").write_text(markdown_table(df, digits), encoding="utf-8")
        return df

    def figure_path(self, name: str) -> Path:
        return self.path("report", "figures", f"{name}.png")

    # Checks --------------------------------------------------------------------------------------------
    def check(self, a, b, label: str, tolerance: float = 1e-6, relative: bool = True,
              allow_na: bool = True) -> dict:
        """Stop the run if two sets of numbers differ by more than the tolerance, and record the result."""
        import numpy as np

        a = np.atleast_1d(np.asarray(a, dtype="float64"))
        b = np.atleast_1d(np.asarray(b, dtype="float64"))
        if not allow_na and (np.isnan(a).any() or np.isnan(b).any()):
            raise CheckFailed(f"Check failed: {label}. Missing values.")
        gap = np.abs(a - b)
        if relative:
            gap = gap / np.maximum(np.abs(b), 1e-9)
        largest = float(np.nanmax(gap)) if gap.size and not np.isnan(gap).all() else 0.0
        if largest > tolerance:
            raise CheckFailed(f"Check failed: {label}. Largest gap {largest:.3g}, tolerance {tolerance:g}")
        row = {"check": label, "largest_gap": largest, "tolerance": tolerance, "relative": relative}
        self.checks.append(row)
        return row

    def require(self, condition: bool, label: str) -> None:
        if not bool(condition):
            raise CheckFailed(f"Check failed: {label}")
        self.checks.append({"check": label, "largest_gap": 0.0, "tolerance": 0.0, "relative": False})

    def save_checks(self, name: str) -> None:
        if self.checks:
            self.table(pl.DataFrame(self.checks), name, digits={"largest_gap": 3, "tolerance": 3})
        self.checks = []

    # Work done year by year -----------------------------------------------------------------------------
    def _key_file(self, name: str, year: int) -> Path:
        return self.path("keys", f"{name}_{year}.txt")

    def year_done(self, name: str, year: int, key: str, *files: Path) -> bool:
        """True when the work `name` of `year` was finished with this key and its files are all there."""
        f = self._key_file(name, year)
        return f.exists() and f.read_text().strip() == key and all(Path(p).exists() for p in files)

    def mark_year_done(self, name: str, year: int, key: str) -> None:
        self._key_file(name, year).write_text(key, encoding="utf-8")

    def tick(self, label: str, done: int, total: int) -> None:
        if self.progress:
            self.progress(label, done, total)


def markdown_table(df: pl.DataFrame, digits: int | dict[str, int] = 1) -> str:
    """A pipe table with formatted numbers, readable in HTML, PDF and Word."""
    cols = df.columns

    def cell(col: str, v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, bool):
            return "Yes" if v else "No"
        if isinstance(v, int) and not isinstance(v, bool):
            return str(v) if col.lower() in ("year", "years") or abs(v) < 10000 else f"{v:,}"
        if isinstance(v, float):
            if math.isnan(v):
                return ""
            d = digits.get(col, 1) if isinstance(digits, dict) else digits
            if d >= 3 and v != 0 and abs(v) < 10 ** -(d - 1):
                return f"{v:.{d - 1}e}"
            return _fmt(v, d)
        return str(v).replace("|", "\\|")

    numeric = [df.schema[c].is_numeric() for c in cols]
    head = "| " + " | ".join(cols) + " |"
    align = "|" + "|".join("---:" if n else ":---" for n in numeric) + "|"
    rows = ["| " + " | ".join(cell(c, v) for c, v in zip(cols, r)) + " |" for r in df.iter_rows()]
    return "\n".join([head, align, *rows]) + "\n"


# ---------------------------------------------------------------------------------------------------------
# Stages and the runner
# ---------------------------------------------------------------------------------------------------------


@dataclass
class Stage:
    name: str
    title: str
    func: Callable[[Context], None]
    heavy: bool = False
    note: str = ""
    memory_gb: float = 2.0  # peak memory of one worker of this stage


def service_module(name: str):
    if name not in SERVICES:
        raise EsaError(f"Unknown service {name!r}. Choose among {', '.join(SERVICES)}.")
    return importlib.import_module(f"esa_toolbox.services.{name}")


def check_method(service: str, settings: Settings) -> None:
    methods = getattr(service_module(service), "METHODS", (METHOD,))
    if settings.method not in methods:
        raise EsaError(f"{service} has no method {settings.method!r}. It offers {', '.join(methods)}.")


def work_key(*parts: Any) -> str:
    """A short digest of what a piece of work depends on.

    A module counts by its source file, a path by its name, size and modification time, an array by its bytes,
    a table by its contents, and anything else by its text. The tool version always counts.
    """
    import types

    import numpy as np

    h = hashlib.sha256(__version__.encode())
    for p in parts:
        if isinstance(p, types.ModuleType):
            h.update(Path(inspect.getsourcefile(p) or "").read_bytes())
        elif isinstance(p, Path):
            st = p.stat()
            h.update(f"{p.name}:{st.st_size}:{st.st_mtime_ns}".encode())
        elif isinstance(p, np.ndarray):
            h.update(np.ascontiguousarray(p).tobytes())
        elif isinstance(p, pl.DataFrame):
            h.update(p.write_csv().encode())
        else:
            h.update(repr(p).encode())
        h.update(b"|")
    return h.hexdigest()[:16]


def stage_fingerprint(stage: Stage, settings: Settings, years: tuple[int, ...]) -> str:
    src = inspect.getsourcefile(stage.func) or ""
    h = hashlib.sha256()
    h.update(__version__.encode())
    h.update(Path(src).read_bytes() if src else b"")
    h.update(json.dumps({"years": list(years)}).encode())
    return h.hexdigest()[:16]


def _stamp_file(ctx: Context, stage: str) -> Path:
    return ctx.path("stages", f"{stage}.json")


def stage_status(service: str, settings: Settings, years: tuple[int, ...] = YEARS) -> list[dict]:
    mod = service_module(service)
    ctx = Context(service, settings, years)
    rows = []
    for st in mod.STAGES:
        f = ctx.work / "stages" / f"{st.name}.json"
        info = json.loads(f.read_text()) if f.exists() else None
        if info is None:
            state = "not run"
        elif info.get("fingerprint") != stage_fingerprint(st, settings, years):
            state = "outdated"
        else:
            state = "done"
        rows.append({"stage": st.name, "title": st.title, "state": state,
                     "when": info and info.get("finished"), "seconds": info and info.get("seconds")})
    return rows


def run_service(service: str, settings: Settings, stages: list[str] | None = None,
                years: tuple[int, ...] = YEARS, force: bool = False, log: Callable[[str], None] = print,
                progress: Callable[[str, int, int], None] | None = None) -> list[dict]:
    """Run the stages of one service in order. With no stages named, run those not yet done."""
    mod = service_module(service)
    check_method(service, settings)
    names = [s.name for s in mod.STAGES]
    if stages:
        unknown = set(stages) - set(names)
        if unknown:
            raise EsaError(f"Unknown stage for {service}: {', '.join(sorted(unknown))}. Stages: {', '.join(names)}")
    status = {r["stage"]: r["state"] for r in stage_status(service, settings, years)}
    ran = []
    stale = False
    for st in mod.STAGES:
        wanted = st.name in stages if stages else (force or stale or status[st.name] != "done")
        if not wanted:
            continue
        stage_settings = settings
        if not settings.workers_explicit:
            # Size the pool for this stage from the memory free now, so a heavy stage never exhausts the machine.
            from dataclasses import replace

            stage_settings = replace(settings, workers=default_workers(st.memory_gb))
        ctx = Context(service, stage_settings, years, log=log, progress=progress)
        log(f"[bold]{service}[/] · {st.title} ({stage_settings.workers} workers)")
        t0 = time.perf_counter()
        st.func(ctx)
        seconds = time.perf_counter() - t0
        stamp = {"stage": st.name, "fingerprint": stage_fingerprint(st, settings, years),
                 "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "seconds": round(seconds, 1), "tool_version": __version__}
        _stamp_file(ctx, st.name).write_text(json.dumps(stamp, indent=1), encoding="utf-8")
        ran.append(stamp)
        stale = True  # later stages read what this one wrote
        write_provenance(ctx)
    return ran


def write_provenance(ctx: Context) -> None:
    """outputs/<service>/run.json: tool version, settings, stages and the inputs the store held."""
    import numpy, polars, rasterio, scipy

    stages = {}
    for f in sorted((ctx.work / "stages").glob("*.json")):
        stages[f.stem] = json.loads(f.read_text())
    inputs = {}
    manifest = ctx.store / "manifest.json"
    if manifest.exists():
        inputs = json.loads(manifest.read_text())
    doc = {
        "service": ctx.service,
        "tool_version": __version__,
        "method": ctx.method,
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": {k: str(v) for k, v in asdict(ctx.settings).items() if k != "quarto"},
        "platform": platform.platform(),
        "packages": {"numpy": numpy.__version__, "scipy": scipy.__version__,
                     "rasterio": rasterio.__version__, "gdal": rasterio.__gdal_version__,
                     "polars": polars.__version__},
        "stages": stages,
        "store_manifest": inputs.get(ctx.service, {}),
    }
    ctx.output("run.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")


def read_param(name: str, **kw) -> pl.DataFrame:
    return pl.read_csv(PARAMS / name, **kw)
