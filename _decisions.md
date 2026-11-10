# Decisions

The record of choices that change, or could change, a published number, with the effect measured when each was made. It stays
off the tool site. `CHANGELOG.md` names the services that each version affects.

## Implementation choices

The tool started as a port of three R routines. A few steps of those routines existed only to keep the repositories under the
size limits of GitHub, or depended on how a library split the work, and running locally removed the reason for them.

| Decision | Earlier routine | Tool | Effect |
|:--|:----|:----|:----|
| Exact distance to semi-natural areas | GDAL proximity over the whole grid, which overstates the distance in a small share of cells, by up to 27 m | Exact Euclidean transform inside each tile, with a margin of 20 km | EU27 pollination flow changes by less than 0.01 percent in every year, and fewer than 500 of 4.2 million kilometres change class |
| No storage rounding of the 1 km reduction | Model reduced to 1 km and stored as 16-bit whole numbers, which rounds it to four decimals | Reduction kept in float32 | Included in the line above. |
| Intermediate 100 m files kept outside the outputs | 6.6 GB of model tiles and 2.4 GB of wood weights stored in the repositories | Kept in `work/` under a key of their inputs and code, rebuilt whenever either changes, and shared by wood and carbon for the productivity | None on the numbers. Reading a tile back takes a tenth of a second, and computing it again three to five seconds |
| Exact transformer in every reprojection | terra `project`, whose bilinear values depend on the extent of the call | GDAL with an error tolerance of zero, so every tile gets the same value whatever the tiling | Carbon and wood national tables unchanged. The 1 km carbon maps of 2018 and 2024 are identical. Cells and regional shares of 2000 to 2012 move with the interpolation |
| Whole-number 100 m carbon maps | Float32, 14 GB | Kilograms of CO2 and euro cents per cell, as the other services | About a third of the disk. Every country keeps its total within one part in a thousand |
| Wood key computed afresh | 100 m weights built once and stored, and those files cannot be rebuilt from the productivity files of that repository | Forest times productivity computed in every run from Corine and the annual productivity | National tables unchanged. The key keeps the same pattern, with a cell correlation above 0.99, and regional shares move with it |
| One price index for all services | Three copies of the same Eurostat release, saved separately by each repository | One shared extract | None. The values are identical in every year |
| Comparison of carbon allocation keys in every run | Run once and kept as a table, with no script behind it | Recomputed in the accounts stage, with the current inventory | None. The recomputed table equals the kept one |
| Cloud Optimised GeoTIFF | Tiled GeoTIFF with no overviews | COG without overviews | None on the numbers. Overviews would add about 60 percent to each file and triple the time to write it, and QGIS draws the maps quickly without them. `gdaladdo` adds them to a copy when needed |

## Inputs and their releases

| Date | Decision | Effect |
|:--|:----|:----|
| 2026-11-09 | `esa fetch` tested against the network into an empty store and compared file by file with the store copied from the earlier repositories | Corine, the JRC workbook, the roads, CAPRI, NUTS (EU27) and the annual GDMP identical. The bumblebee model differs by float32 rounding (6e-8) and the activity index by 0.01 in about 300 cells a year |
| 2026-11-10 | The cell-by-cell comparison with the R results and the store adoption script are retired, with the `--legacy` setting and its GDAL distance files | None on the numbers of a normal run |
| 2026-11-10 | All Eurostat extracts retrieved again, on 2026-11-10, and each report lists the retrieval date of its inputs | Pollination: value of the flow in 2024 +0.48 percent (Spain +4.1 percent, France +0.13 percent) from revisions of `aact_eaa02`. Physical accounts unchanged. Wood and carbon: every table unchanged |
| 2026-11-10 | EU27 HICP before 2000 estimated from the annual rates of the euro area and the other Member States, with the country weights of the first published year (`prc_hicp_cow`). The same calculation gives the published EU27 rates of 2001 to 2007 within 0.1 points. Before, the mean of 1998 to 2000 skipped the missing years and used 2000 alone | Pollination: deflator of 2000 from 1.272 to 1.316, so every value of 2000 rises by 3.44 percent. Other years and services unchanged |
| 2026-11-10 | Wood and carbon move to the 1 km grid of pollination, from x = 2,630,000 and y = 1,380,000. Their earlier grid started at x = 2,636,000 and y = 1,385,900, so its 1 km cells sat 100 m north of those of pollination. The two windows had been chosen independently, with no technical reason recorded | National tables unchanged. Regions are assigned by the 1 km cell centre, so regional shares move. Median change at NUTS3 0.23 percent for wood and 0.20 percent for carbon, and at NUTS2 0.05 percent. About 3 percent of NUTS3 region-years move by more than 5 percent, all in regions with little forest. The largest, NL325, moves from 0.15 to 0.07 thousand cubic metres in 2006. The 100 m maps cover the slightly larger window and keep their cells |

## Open items

- **Unmet demand in tonnes, and all ecosystems against agroecosystems**, as the pollination report explains.
- **Service providing area.** The area of medium and high potential grows from 225.9 million hectares in 2000 to 239.8 million in
  2024. Holding land cover and climate in turn at the earlier year, the activity index explains 3.8, -0.1, 4.9 and 3.9 million
  hectares of the four steps, and land cover 0.5, -0.6, 0.1 and 1.5. The activity index rises with ERA5 temperature, which goes up
  by about 1 °C over the growing season of EU land between the windows of 2000 and 2024, while radiation adds little. About 41
  million hectares lie within 0.01 of the threshold of 0.2, so small changes of potential move a large area. Land cover moves the area
  mostly in Sweden, where transitional woodland and shrub falls from 5.2 to 3.6 million hectares between 2006 and 2012 and rises to
  4.7 million in 2024.
