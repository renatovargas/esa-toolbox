## Rules

- **Two products, kept apart.** The tool site (`index.qmd`, `site/`) documents the code and the method as implemented. The
  results report of each run is rendered by the tool into `outputs/<service>/report/` and carries no references to code. Both
  read the same method partial, `src/esa_toolbox/reports/<service>/methods.qmd`, where `[...]{.content-visible unless-meta="results"}`
  holds text for the site only and `when-meta="results"` text for the report only. The site shows no number or table that needs
  a run, so every fact, run table and sentence that depends on them sits in `when-meta="results"`.
- **Numbers in text** come from `{{< fact name >}}` and tables from `{{< datatable name >}}`, written by the stages with
  `ctx.facts()` and `ctx.table()`. A report stops on an unknown fact. Never type a result into a report.
- **Checks stop the run.** Use `ctx.check()` and `ctx.require()`, which record each check for the report.
- **Record before change.** A change that moves a published number needs an entry in `CHANGELOG.md` naming the services, and a row
  in `_decisions.md` with its measured effect. `_decisions.md` is the record of the project and stays off the site.
- **Large files** stay in `store/`, `work/` and `outputs/`, which git ignores. Write files under a temporary name and rename them.
- **Raster steps** work tile by tile through `esa_toolbox.pool`, which uses `spawn` and one GDAL thread per worker.
- **Compiled code** comes in only when a profile shows one operation over 25 percent of a stage that exceeds its budget, and a
  kernel would save a factor of three on it.
- **Prose** is written in declarative full sentences, no long dashes, no "It is not X, it is Y", no colon to
  introduce a list after a complete sentence, and pending questions written as "Open question".
- The `_extensions/facts` folder at the root is a copy of the one in the package, and a test keeps them equal.
