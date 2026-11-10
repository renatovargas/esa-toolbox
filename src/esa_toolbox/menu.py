"""The guided menu that `esa` opens when it runs with no command.

It shows where each service stands, then asks a few questions and runs the same functions as the commands.
Every choice it makes is printed as the equivalent command, so the menu also teaches the command line.
"""

from __future__ import annotations

import os
import subprocess
import sys

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from esa_toolbox import __version__
from esa_toolbox.core import SERVICES, YEARS, EsaError, Settings, service_module

console = Console()


def _pick_services() -> list[str]:
    t = Table(show_header=False, box=None)
    for i, s in enumerate(SERVICES, 1):
        t.add_row(f"[bold]{i}[/]", service_module(s).TITLE)
    console.print(t)
    raw = Prompt.ask("Which services? Numbers separated by commas, or [bold]a[/] for all", default="a")
    if raw.strip().lower() in ("a", "all", ""):
        return list(SERVICES)
    out = []
    for part in raw.replace(" ", "").split(","):
        if part.isdigit() and 1 <= int(part) <= len(SERVICES):
            out.append(SERVICES[int(part) - 1])
        elif part in SERVICES:
            out.append(part)
    return out or list(SERVICES)


def _pick_years() -> tuple[int, ...]:
    raw = Prompt.ask(f"Reporting years ({', '.join(map(str, YEARS))})", default="all")
    if raw.strip().lower() in ("all", "a", ""):
        return YEARS
    ys = tuple(int(y) for y in raw.replace(" ", "").split(",") if y.isdigit() and int(y) in YEARS)
    return ys or YEARS


def _command(services: list[str], extra: str = "") -> str:
    s = "" if services == list(SERVICES) else " " + " ".join(services)
    return f"esa run{s}{extra}"


def run_flow(settings: Settings) -> None:
    from esa_toolbox.cli import run_with_progress

    services = _pick_services()
    console.print("\n[bold]What should run?[/]")
    console.print("  [bold]1[/] Everything that is not up to date [dim](usual choice)[/]")
    console.print("  [bold]2[/] Only the tables and 1 km maps, no 100 m maps [dim](minutes)[/]")
    console.print("  [bold]3[/] Everything again from the start")
    console.print("  [bold]4[/] Only the reports, from the last run")
    choice = Prompt.ask("Choice", choices=["1", "2", "3", "4"], default="1")
    years = YEARS if choice == "4" else _pick_years()
    yarg = "" if years == YEARS else f" --years {','.join(map(str, years))}"
    if choice == "4":
        console.print(f"[dim]Same as: esa report {' '.join(services)}[/]")
        from esa_toolbox.reports import render

        for s in services:
            for f in render(s, settings, log=console.print):
                console.print(f"[green]✓[/] {f}")
        return
    stages, force, extra = None, False, yarg
    if choice == "2":
        stages = [st.name for s in services for st in service_module(s).STAGES if st.name != "maps_100m"]
        stages = list(dict.fromkeys(stages))
        extra += "".join(f" -s {x}" for x in stages)
    if choice == "3":
        force = True
        extra += " --force"
    console.print(f"[dim]Same as: {_command(services, extra)}[/]")
    if not Confirm.ask("Start?", default=True):
        return
    try:
        run_with_progress(settings, services, stages, years, force, report=True)
    except EsaError as e:
        console.print(f"[bold red]Stopped:[/] {e}")


def open_folder(path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    opener = {"darwin": ["open"], "win32": ["explorer"]}.get(sys.platform, ["xdg-open"])
    try:
        subprocess.Popen([*opener, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        console.print(f"Outputs are in {path.resolve()}")


def menu() -> None:
    from esa_toolbox.cli import doctor, inputs, status_tables

    settings = Settings.load()
    while True:
        console.print(Panel.fit(f"[bold]ESA toolbox[/] {__version__}   ·   store [cyan]{settings.store}[/]   ·   "
                                f"outputs [cyan]{settings.outputs}[/]   ·   {settings.workers} workers"))
        tables = status_tables(settings, list(SERVICES))
        if console.width >= 150:
            grid = Table.grid(padding=(0, 3))
            grid.add_row(*tables)
            console.print(grid)
        else:
            for t in tables:
                console.print(t)
        console.print("\n  [bold]1[/] Run accounts      [bold]2[/] Inputs and fetch      [bold]3[/] Open outputs folder"
                      "      [bold]4[/] Check installation      [bold]q[/] Quit")
        choice = Prompt.ask("What would you like to do?", choices=["1", "2", "3", "4", "q"], default="1")
        if choice == "q":
            return
        if choice == "1":
            run_flow(settings)
        elif choice == "2":
            inputs(services=None, config=None)
            if Confirm.ask("Fetch what is missing?", default=False):
                from esa_toolbox.cli import fetch

                console.print("[dim]Same as: esa fetch[/]")
                try:
                    fetch(datasets=None, dry_run=False, force=False, config=None)
                except SystemExit:
                    pass
        elif choice == "3":
            open_folder(settings.outputs)
        elif choice == "4":
            doctor(config=None)
        if not Confirm.ask("\nBack to the menu?", default=True):
            return
        os.system("")  # keeps ANSI colours working in the Windows console
