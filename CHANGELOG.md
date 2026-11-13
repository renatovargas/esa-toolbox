# Changes

Every version that can change a published number names the services it affects.

## 0.2.0 (unreleased)

Results affected: pollination. Wood and carbon keep their numbers, and their stages run again with the new version.

- Pollination, revision of the method. The dependency ratios of the crop groups come from Siopa et al. (2024) in place of Klein
  et al. (2007), through a crosswalk between the crops of that review and the ten crop groups. Actual flow, unmet demand and their
  values change in every year. Potential, the service providing area and the area of demand do not depend on the ratios.
- Pollination, revision of the method. The bumblebee model is projected to the land cover and climate of each reporting year
  from the 47 fitted Maxent models of 2012, and the layer of a year is the published layer of 2012 plus the projected change.
  Potential, the service providing area and every quantity after them change in 2000, 2006, 2018 and 2024. The values of 2012
  keep the published layer.
- Pollination has a new stage, `sdm`, and a new input, `eobs_bioclim`, which `esa fetch` builds from the daily E-OBS grids. The
  fitted models, their presence thresholds and the mode of elevation at 10 km travel inside the package (0.6 MB).
- `esa run pollination --method original` calculates the accounts as before the revision, into `outputs/pollination_original/`
  and `work/pollination_original/`. A run without the option uses the revised method.

## 0.1.0 (unreleased)

Results affected: pollination, wood, carbon (first version).

- Crop pollination, wood provision and carbon sequestration accounts of the EU27 for 2000, 2006, 2012, 2018 and 2024, with a guided
  menu, the commands `run`, `status`, `report`, `inputs`, `fetch`, `doctor` and `clean`, and one report per service in HTML, PDF
  and Word.
- `esa fetch` downloads every input from its publisher into the store. The Eurostat extracts record their date of retrieval, and
  the report of each run lists its inputs with those dates. The extracts of this version were retrieved on 2026-11-10.
- The EU27 price index starts in 2000, and the fetcher estimates the years before from the euro area and the other Member States
  with the country weights of Eurostat. Pollination needs them for the three-year mean of 2000.
- All services share one 1 km grid, from x = 2,630,000 and y = 1,380,000 in EPSG:3035, so their 1 km cells coincide. Wood and
  carbon regional shares move with it (median 0.2 percent at NUTS3), and national tables are unchanged.
- Pollination: exact Euclidean distance to semi-natural areas, computed inside each tile with a margin of 20 km.
- Wood and carbon: reprojection with an error tolerance of zero, so every tile gets the same values whatever the tiling.
- All 100 m maps are whole numbers in small units and Cloud Optimised GeoTIFF files without overviews.
- Each year of a heavy stage is redone whenever its code, its input files or the national values it spreads change, and skipped
  otherwise. Intermediate files stay in `work/`, under a key of their inputs and code.
