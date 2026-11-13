"""Results reports: the method of a service with the results of one run, in HTML, PDF and Word.

The method text is the same partial that the tool site shows. Here the metadata `results: true` hides the
references to the code, and the facts extension fills the numbers and tables that the run wrote to
work/<service>/report/. Quarto renders the report with no execution engine.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import date
from pathlib import Path
from typing import Callable

import polars as pl

from esa_toolbox import __version__
from esa_toolbox.core import EsaError, Settings, markdown_table

HERE = Path(__file__).parent
FILE_NAME = {"carbon": "carbon_sequestration_report", "wood": "wood_provision_report",
             "pollination": "crop_pollination_report"}
EXT = {"html": "html", "typst": "pdf", "pdf": "pdf", "docx": "docx"}


def _checks_table(src: Path) -> str:
    frames = []
    for name in ("checks_physical", "checks_monetary", "checks_maps", "checks_maps_1km", "checks_maps_100m"):
        p = src / "tables" / f"{name}.parquet"
        if p.exists():
            frames.append(pl.read_parquet(p))
    if not frames:
        return "| Check |\n|:---|\n| No checks recorded |\n"
    t = pl.concat(frames, how="diagonal_relaxed").select(
        Check="check",
        **{"Largest gap": pl.col("largest_gap").map_elements(lambda v: f"{v:.1e}", return_dtype=pl.String),
           "Tolerance": pl.col("tolerance").map_elements(lambda v: f"{v:.0e}", return_dtype=pl.String),
           "Type": pl.when(pl.col("relative")).then(pl.lit("Relative")).otherwise(pl.lit("Absolute"))})
    return markdown_table(t)


def _inputs_table(service: str, store: Path) -> str:
    """The inputs of a service, with the date of retrieval that the fetchers record for the Eurostat extracts, and
    for the other inputs the date on which their files were written to the store."""
    from esa_toolbox.data.catalog import for_service

    rows = []
    for d in for_service(service):
        dates = set()
        for p in d.paths(store):
            m = p.parent / "extracts.json"
            if m.exists():
                dates.add((json.loads(m.read_text()).get(p.stem) or {}).get("retrieved") or "")
            elif p.exists():
                dates.add(date.fromtimestamp(p.stat().st_mtime).isoformat())
        rows.append({"Dataset": d.title, "Publisher": d.publisher, "Source": d.source,
                     "Retrieved": max((x for x in dates if x), default="")})
    return markdown_table(pl.DataFrame(rows))


def build(service: str, settings: Settings) -> Path:
    src = settings.work / settings.folder(service) / "report"
    if not (src / "facts.json").exists():
        raise EsaError(f"No results for {settings.folder(service)} yet. Run `esa run {service}` first.")
    b = src / "build"
    if b.exists():
        shutil.rmtree(b)
    b.mkdir(parents=True)
    tpl = HERE / service
    for f in ("report.qmd", "methods.qmd", "references.bib"):
        shutil.copy(tpl / f, b / f)
    shutil.copy(HERE / "apa.csl", b / "apa.csl")
    shutil.copytree(HERE / "_extensions", b / "_extensions")
    (b / "_data" / "tables").mkdir(parents=True)
    facts = json.loads((src / "facts.json").read_text(encoding="utf-8"))
    run_file = settings.outputs / settings.folder(service) / "run.json"
    when = date.today().isoformat()
    if run_file.exists():
        stages = json.loads(run_file.read_text()).get("stages", {})
        if stages:
            when = max(s["finished"] for s in stages.values())[:10]
    facts["tool_version"] = {"value": __version__, "text": __version__}
    facts["run_date"] = {"value": when, "text": when}
    (b / "_data" / "facts.json").write_text(json.dumps(facts), encoding="utf-8")
    qmd = b / "report.qmd"
    qmd.write_text(qmd.read_text(encoding="utf-8").replace('"{{< fact run_date >}}"', f'"{when}"'), encoding="utf-8")
    for md in (src / "tables").glob("*.md"):
        shutil.copy(md, b / "_data" / "tables" / md.name)
    (b / "_data" / "tables" / "checks.md").write_text(_checks_table(src), encoding="utf-8")
    (b / "_data" / "tables" / "inputs.md").write_text(_inputs_table(service, settings.store), encoding="utf-8")
    if (src / "figures").exists():
        shutil.copytree(src / "figures", b / "figures")
    return b


def render(service: str, settings: Settings, log: Callable[[str], None] = print) -> list[Path]:
    quarto = settings.quarto_path()
    if not quarto:
        raise EsaError("Quarto is needed for the reports. Install it from https://quarto.org, or set QUARTO_PATH.")
    b = build(service, settings)
    out_dir = settings.outputs / settings.folder(service) / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in settings.formats:
        to = "typst" if fmt == "pdf" else fmt
        name = f"{FILE_NAME[service]}.{EXT[to]}"
        log(f"  rendering {name}")
        r = subprocess.run([quarto, "render", "report.qmd", "--to", to, "--output", name,
                            "-M", f"method-{settings.method}:true"], cwd=b,
                           capture_output=True, text=True)
        if r.returncode != 0:
            tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-15:])
            raise EsaError(f"Quarto could not render {name}:\n{tail}")
        shutil.copy(b / name, out_dir / name)
        written.append(out_dir / name)
    return written
