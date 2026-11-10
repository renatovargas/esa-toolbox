# ESA toolbox

A desktop tool that builds the ecosystem service accounts of the EU27 for **crop pollination**, **wood provision** and
**carbon sequestration**, for 2000, 2006, 2012, 2018 and 2024. It downloads the inputs, calculates the physical and monetary
accounts, writes the 100 m and 1 km maps and the results workbooks by NUTS level, and renders one report per service, in HTML,
PDF and Word, that explains the method and presents the results of that run.

Documentation: <https://renatovargas.github.io/esa-toolbox/>

## Install

```bash
uv tool install git+https://github.com/renatovargas/esa-toolbox
```

To work with exactly the versions of the libraries that made the published results, clone the repository and run the tool from
it, which installs the environment recorded in `uv.lock`:

```bash
git clone https://github.com/renatovargas/esa-toolbox && cd esa-toolbox
uv sync --frozen
uv run esa doctor
```

The reports need [Quarto](https://quarto.org). The copies inside Positron and RStudio are found automatically, and
`QUARTO_PATH` points to any other.

## Use

```bash
esa                 # guided menu: state of each service, then what to run
esa doctor          # installation, folders, disk, memory, Quarto
esa fetch carbon    # inputs of one service into the store
esa run carbon      # accounts, maps, workbook and report
esa run             # all services, only what is not up to date
esa status          # what is done and what is outdated
```

Each service writes to `outputs/<service>/`:

```
outputs/carbon/
├── tables/   carbon_sequestration_results.xlsx
├── maps/     100m/ and 1km/ GeoTIFF (Cloud Optimised, EPSG:3035)
├── report/   carbon_sequestration_report.html, .pdf, .docx
└── run.json  tool version, settings, stages and timings
```

## Repository

| Path | Content |
|---|---|
| `src/esa_toolbox/` | The package: `core.py` (settings, stages, checks, facts), `rio.py` and `grid.py` (rasters), `services/` (one folder per service), `data/` (catalogue and fetchers), `reports/` (method partials, report templates, the facts extension) |
| `index.qmd`, `site/` | Tool site, rendered to `docs/` |
| `tests/` | Unit tests (`uv run pytest`) |
| `store/`, `work/`, `outputs/` | Inputs, intermediate files and products. Not versioned |
