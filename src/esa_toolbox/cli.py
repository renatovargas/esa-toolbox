"""The `esa` command. Run it with no arguments for a guided menu, or with a command for scripts."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from esa_toolbox import __version__
from esa_toolbox.core import METHOD, SERVICES, YEARS, EsaError, Settings, available_memory_gb, run_service, service_module, stage_status

app = typer.Typer(name="esa", help="Ecosystem service accounts for the EU27, from inputs to maps, workbooks and reports.",
                  no_args_is_help=False, add_completion=False, pretty_exceptions_enable=False, rich_markup_mode="rich")
console = Console(stderr=True)
out = Console()

ServicesArg = Annotated[Optional[list[str]], typer.Argument(help="Services: pollination, wood, carbon. All when omitted.")]
ConfigOpt = Annotated[Optional[Path], typer.Option("--config", "-c", help="Settings file (default esa.toml).")]
JsonOpt = Annotated[bool, typer.Option("--json", help="Print results as JSON on stdout.")]
MethodOpt = Annotated[Optional[str], typer.Option(
    "--method", help="`original` runs crop pollination as before the revision of 2026, into its own folders.")]


def _settings(config: Path | None, **kw) -> Settings:
    return Settings.load(config, **kw)


def _services(names: list[str] | None, settings: Settings | None = None) -> list[str]:
    """The services named, or all of them. With another method than the normal one, all means those that offer it."""
    chosen = names or list(SERVICES)
    for n in chosen:
        if n not in SERVICES:
            raise EsaError(f"Unknown service {n!r}. Choose among {', '.join(SERVICES)}.")
    if settings is not None and not names:
        chosen = [n for n in chosen if settings.method in getattr(service_module(n), "METHODS", (METHOD,))]
    return chosen


def _fail(exc: Exception) -> None:
    code = getattr(exc, "exit_code", 1)
    console.print(f"[bold red]Error:[/] {exc}", highlight=False)
    if not isinstance(exc, EsaError):
        console.print_exception(show_locals=False)
    raise typer.Exit(code)


@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context,
                  version: Annotated[bool, typer.Option("--version", help="Show the version.")] = False) -> None:
    if version:
        out.print(__version__)
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        from esa_toolbox.menu import menu

        menu()


# ---------------------------------------------------------------------------------------------------------
# status and doctor
# ---------------------------------------------------------------------------------------------------------


def status_tables(settings: Settings, services: list[str]) -> list[Table]:
    from esa_toolbox.data.catalog import for_service

    tables = []
    for s in services:
        mod = service_module(s)
        t = Table(title=f"{mod.TITLE}", title_justify="left", header_style="bold", expand=False)
        t.add_column("Step")
        t.add_column("State")
        t.add_column("Last run")
        t.add_column("Time", justify="right")
        missing = [d for d in for_service(s) if d.missing(settings.store)]
        t.add_row("inputs", "[green]in the store[/]" if not missing else
                  f"[red]missing:[/] {', '.join(d.id for d in missing)}", "", "")
        for r in stage_status(s, settings):
            colour = {"done": "green", "outdated": "yellow", "not run": "dim"}[r["state"]]
            t.add_row(f"{r['stage']}", f"[{colour}]{r['state']}[/]", (r["when"] or "")[5:16].replace("T", " "),
                      f"{r['seconds']:.0f} s" if r["seconds"] else "")
        rep = settings.outputs / settings.folder(s) / "report"
        files = sorted(p.suffix[1:] for p in rep.glob(f"{s}*.*")) if rep.exists() else []
        t.add_row("report", f"[green]{', '.join(files)}[/]" if files else "[dim]not rendered[/]", "", "")
        tables.append(t)
    return tables


@app.command()
def status(services: ServicesArg = None, method: MethodOpt = None, config: ConfigOpt = None,
           as_json: JsonOpt = False) -> None:
    """Show which inputs are in the store and which steps are done, for each service."""
    try:
        st = _settings(config, method=method)
        names = _services(services, st)
        if as_json:
            out.print_json(json.dumps({s: stage_status(s, st) for s in names}))
            return
        for t in status_tables(st, names):
            out.print(t)
    except Exception as e:  # noqa: BLE001
        _fail(e)


@app.command()
def doctor(config: ConfigOpt = None) -> None:
    """Check the installation, folders, disk, memory and Quarto, and suggest a number of workers."""
    import numpy
    import polars
    import rasterio

    st = _settings(config)
    t = Table(show_header=False, box=None)
    t.add_row("esa-toolbox", __version__)
    t.add_row("Python", sys.version.split()[0])
    t.add_row("GDAL (in rasterio)", rasterio.__gdal_version__)
    t.add_row("NumPy / Polars", f"{numpy.__version__} / {polars.__version__}")
    for name, p in (("Store", st.store), ("Outputs", st.outputs), ("Work", st.work)):
        p.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(p).free / 1e9
        t.add_row(name, f"{p.resolve()}  ({free:,.0f} GB free)")
    t.add_row("Memory available", f"{available_memory_gb():.1f} GB")
    t.add_row("Workers", f"{st.workers} (set `workers` in esa.toml to change)")
    q = st.quarto_path()
    t.add_row("Quarto", q or "[red]not found[/]: reports need it. Install it, or set QUARTO_PATH.")
    import os

    keys = bool(os.environ.get("CDSE_S3_ACCESS_KEY") and os.environ.get("CDSE_S3_SECRET_KEY"))
    t.add_row("Copernicus keys", "set" if keys else "not set (only `esa fetch gdmp_annual` needs them)")
    out.print(Panel(t, title="esa doctor", expand=False))


# ---------------------------------------------------------------------------------------------------------
# run and report
# ---------------------------------------------------------------------------------------------------------


def run_with_progress(settings: Settings, services: list[str], stages: list[str] | None, years: tuple[int, ...],
                      force: bool, report: bool) -> None:
    with Progress(SpinnerColumn(), TextColumn("[bold]{task.description}"), BarColumn(), MofNCompleteColumn(),
                  TimeElapsedColumn(), console=console, transient=True) as progress:
        task = progress.add_task("starting", total=None)

        def tick(label: str, done: int, total: int) -> None:
            progress.update(task, description=label, completed=done, total=total)

        def log(msg: str) -> None:
            progress.console.print(msg)
            progress.update(task, description=msg.strip().split("·")[-1].strip()[:60], completed=0, total=None)

        for s in services:
            t0 = time.perf_counter()
            ran = run_service(s, settings, stages, years, force, log=log, progress=tick)
            if report and (not stages or "report" in (stages or [])):
                from esa_toolbox.reports import render

                log(f"[bold]{s}[/] · Report")
                render(s, settings, log=log)
            dt = time.perf_counter() - t0
            names = ", ".join(r["stage"] for r in ran) or "nothing to do"
            progress.console.print(f"[green]✓[/] {s}: {names} ({dt:,.0f} s). Outputs in {settings.outputs / settings.folder(s)}")


@app.command()
def run(
    services: ServicesArg = None,
    stage: Annotated[Optional[list[str]], typer.Option("--stage", "-s", help="Run only these stages (repeatable).")] = None,
    years: Annotated[Optional[str], typer.Option(help="Comma-separated reporting years, e.g. 2018,2024.")] = None,
    force: Annotated[bool, typer.Option("--force", help="Run every stage again, even those up to date.")] = False,
    workers: Annotated[Optional[int], typer.Option(help="Processes for the raster steps.")] = None,
    no_report: Annotated[bool, typer.Option("--no-report", help="Skip the report.")] = False,
    method: MethodOpt = None,
    config: ConfigOpt = None,
) -> None:
    """Calculate the accounts of one or more services, resuming where a run stopped, then render their reports."""
    try:
        st = _settings(config, workers=workers, method=method)
        ys = tuple(int(y) for y in years.split(",")) if years else YEARS
        bad = set(ys) - set(YEARS)
        if bad:
            raise EsaError(f"Reporting years are {', '.join(map(str, YEARS))}, not {', '.join(map(str, sorted(bad)))}")
        run_with_progress(st, _services(services, st), stage, ys, force, report=not no_report)
    except Exception as e:  # noqa: BLE001
        _fail(e)


@app.command()
def report(services: ServicesArg = None,
           formats: Annotated[Optional[str], typer.Option(help="Comma-separated: html, pdf, docx.")] = None,
           method: MethodOpt = None, config: ConfigOpt = None) -> None:
    """Render the results report of each service from its last run, in HTML, PDF and Word."""
    from esa_toolbox.reports import render

    try:
        st = _settings(config, formats=tuple(formats.split(",")) if formats else None, method=method)
        for s in _services(services, st):
            for f in render(s, st, log=console.print):
                out.print(f"[green]✓[/] {f}")
    except Exception as e:  # noqa: BLE001
        _fail(e)


# ---------------------------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------------------------


@app.command()
def inputs(services: ServicesArg = None, config: ConfigOpt = None) -> None:
    """List the inputs each service needs, their source and whether the store holds them."""
    from esa_toolbox.data.catalog import DATASETS

    st = _settings(config)
    names = _services(services)
    t = Table(header_style="bold")
    for c in ("Dataset", "Used by", "In store", "Source", "Note"):
        t.add_column(c, overflow="fold")
    for d in DATASETS:
        if not set(d.used_by) & set(names) and d.used_by:
            continue
        miss = d.missing(st.store)
        state = "[green]yes[/]" if not miss else f"[red]{len(miss)} missing[/]"
        t.add_row(d.id, ", ".join(d.used_by), state, d.source, d.note)
    out.print(t)


@app.command()
def fetch(
    datasets: Annotated[Optional[list[str]], typer.Argument(help="Dataset ids or services. All missing when omitted.")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show what would be fetched.")] = False,
    force: Annotated[bool, typer.Option("--force", help="Fetch again even when the files exist.")] = False,
    config: ConfigOpt = None,
) -> None:
    """Download the inputs into the store. Each fetcher resumes and writes files atomically."""
    from esa_toolbox.data import fetch as fetchers
    from esa_toolbox.data.catalog import BY_ID, DATASETS, for_service

    try:
        st = _settings(config)
        chosen = []
        for name in datasets or [d.id for d in DATASETS if d.used_by]:
            chosen += for_service(name) if name in SERVICES else [BY_ID[name]] if name in BY_ID else []
            if name not in SERVICES and name not in BY_ID:
                raise EsaError(f"Unknown dataset {name!r}. Run `esa inputs` to list them.")
        for d in dict.fromkeys(chosen):
            miss = d.paths(st.store) if force else d.missing(st.store)
            if not miss:
                console.print(f"[green]✓[/] {d.id}: in the store")
                continue
            console.print(f"[bold]{d.id}[/]: {len(miss)} file(s) to fetch by {d.method}")
            if not dry_run:
                fetchers.fetch(d, st, log=console.print)
    except Exception as e:  # noqa: BLE001
        _fail(e)


@app.command()
def clean(services: ServicesArg = None,
          outputs: Annotated[bool, typer.Option("--outputs", help="Also delete the outputs folder of the service.")] = False,
          yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask.")] = False,
          method: MethodOpt = None, config: ConfigOpt = None) -> None:
    """Delete the working files of a service, so the next run starts from the store."""
    st = _settings(config, method=method)
    for s in _services(services, st):
        targets = [st.work / st.folder(s)] + ([st.outputs / st.folder(s)] if outputs else [])
        targets = [t for t in targets if t.exists()]
        if not targets:
            continue
        if not yes and not typer.confirm(f"Delete {', '.join(map(str, targets))}?"):
            continue
        for t in targets:
            shutil.rmtree(t)
        console.print(f"[green]✓[/] {s} cleaned")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
