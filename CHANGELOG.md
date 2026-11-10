# Changes

Every version that can change a published number names the services it affects.

## 0.1.0 (unreleased)

Results affected: pollination, wood, carbon (first version).

- Crop pollination, wood provision and carbon sequestration accounts of the EU27 for 2000, 2006, 2012, 2018 and 2024, with a guided
  menu, the commands `run`, `status`, `report`, `inputs`, `fetch`, `doctor` and `clean`, and one report per service in HTML, PDF
  and Word.
- `esa fetch` downloads every input from its publisher into the store. The Eurostat extracts record their date of retrieval, and
  the report of each run lists its inputs with those dates. The extracts of this version were retrieved on 2026-11-10.
- The EU27 price index starts in 2000, and the fetcher estimates the years before from the euro area and the other Member States
  with the country weights of Eurostat. Pollination needs them for the three-year mean of 2000.
- Pollination: exact Euclidean distance to semi-natural areas, computed inside each tile with a margin of 20 km.
- Wood and carbon: reprojection with an error tolerance of zero, so every tile gets the same values whatever the tiling.
- All 100 m maps are whole numbers in small units and Cloud Optimised GeoTIFF files without overviews.
- Each year of a heavy stage is redone whenever its code, its input files or the national values it spreads change, and skipped
  otherwise. Intermediate files stay in `work/`, under a key of their inputs and code.
