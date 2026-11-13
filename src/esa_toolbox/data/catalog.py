"""The catalogue of inputs: what each service reads from the store, where it comes from and how to fetch it.

Each dataset lists the files it puts in the store, relative to the store root. A `{year}` in a file name
expands over the reporting years. `esa status` checks that the files exist, and `esa fetch` runs the fetcher
named in `method` (see esa_toolbox.data.fetch).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

YEARS = (2000, 2006, 2012, 2018, 2024)


@dataclass(frozen=True)
class Dataset:
    id: str
    title: str
    publisher: str
    source: str  # DOI or address
    method: str  # fetcher name
    files: tuple[str, ...]
    used_by: tuple[str, ...]
    size_gb: float = 0.0
    note: str = ""
    years: tuple[int, ...] = field(default=YEARS)
    needs_keys: bool = False

    def paths(self, store: Path) -> list[Path]:
        out = []
        for f in self.files:
            if "{year}" in f:
                out += [store / f.format(year=y) for y in self.years]
            elif "{product}" in f:
                continue
            else:
                out.append(store / f)
        return out

    def missing(self, store: Path) -> list[Path]:
        return [p for p in self.paths(store) if not p.exists()]


CLC_RECORDS = {
    2000: "97eed728-309d-47cf-9411-7171be9842a9", 2006: "bc5f6ad3-5dee-44b6-8fa2-0179ea43ddf0",
    2012: "d9533dc9-272b-4dac-9e98-83dfe8d1b32f", 2018: "0f878901-6c51-41ae-b54d-52e794870e9c",
    2024: "a65428d3-bd66-4b47-bcf9-6bfd6fa5df0c",
}

DATASETS = [
    Dataset("clc_accounting", "CORINE Land Cover accounting layers, version 21, 100 m", "European Environment Agency",
            "https://doi.org/10.2909/a65428d3-bd66-4b47-bcf9-6bfd6fa5df0c", "eea_datashare",
            ("clc_accounting/v21/CLC{year}ACC_V2024_21.tif",), ("pollination", "wood", "carbon"), 0.9),
    Dataset("nuts", "NUTS 2024 boundaries, 1:1 million, EPSG:3035",
            "Eurostat GISCO", "https://ec.europa.eu/eurostat/web/gisco/geodata/statistical-units/territorial-units-statistics",
            "gisco", ("nuts/NUTS_RG_01M_2024_3035_LEVL0123_INCA.gpkg",), ("pollination", "wood", "carbon"), 0.03,
            note="The fetcher builds the same layer from the primary GISCO file. Its EU27 regions equal the INCA copy."),
    Dataset("gdmp_annual", "Gross Dry Matter Productivity, annual sum of 36 dekads (v2, 1 km and 300 m)",
            "Copernicus Land Monitoring Service", "https://doi.org/10.2909/fd838f95-21e9-4f37-a9b8-b438d9dc4edf",
            "copernicus_s3",
            ("copernicus/gdmp_annual/gdmp_1km_2000_4326.tif", "copernicus/gdmp_annual/gdmp_1km_2006_4326.tif",
             "copernicus/gdmp_annual/gdmp_1km_2012_4326.tif", "copernicus/gdmp_annual/gdmp_1km_2018_4326.tif",
             "copernicus/gdmp_annual/gdmp_300m_2018_4326.tif", "copernicus/gdmp_annual/gdmp_300m_2024_4326.tif"),
            ("wood", "carbon"), 1.1, needs_keys=True,
            note="Needs CDSE_S3_ACCESS_KEY and CDSE_S3_SECRET_KEY. Reads 180 global files, which takes hours."),
    Dataset("roads", "OpenStreetMap main roads, rasterised at 100 m (1 on a major road)", "OpenStreetMap contributors",
            "https://integration.integratedmodelling.org/geoserver/im-data-global-infrastructure/ows", "wfs_roads",
            ("osm/osm_major_roads_100m.tif",), ("pollination",), 0.8),
    Dataset("activity_index", "Growing season activity index of insects, 1 km (ERA5 temperature, solar radiation)",
            "k.LAB public STAC catalogue (jrc_data_catalog)",
            "https://stac.integratedmodelling.org/collections/jrc_data_catalog", "stac_activity",
            ("catalogue/activity_index_{year}_1km.tif",), ("pollination",), 0.02),
    Dataset("bumblebee_sdm", "Bumblebee species distribution model (Polce et al., 2018), 1 km",
            "k.LAB public STAC catalogue (jrc_data_catalog)",
            "https://stac.integratedmodelling.org/collections/jrc_data_catalog", "stac_sdm",
            ("catalogue/bumblebee_sdm_1km.tif",), ("pollination",), 0.01),
    Dataset("eobs_bioclim", "Bioclimatic layers of the bumblebee model, 10 km, from E-OBS v32.0e at 0.25 degrees",
            "Copernicus Climate Change Service and ECA&D (KNMI)", "https://doi.org/10.24381/cds.151d3ec6", "eobs",
            ("eobs/bioclim_{year}_10km.tif",), ("pollination",), 0.006,
            note="Downloads 2.1 GB of daily grids, keeps five files of 1 MB and removes the download."),
    Dataset("capri", "Disaggregated CAPRI crop area and yield, ten pollinator-dependent crops, 1 km",
            "Koeble et al. (2024)", "https://doi.org/10.5281/zenodo.13992301", "stac_capri",
            tuple(f"capri/capri_{k}_{y}.tif" for k in ("area_ha", "yield_kgha") for y in (2000, 2006, 2012, 2018)),
            ("pollination",), 0.12, note="Spatial crop data end in 2018, which also represents 2024."),
    Dataset("eurostat_hicp", "Harmonised index of consumer prices, EU27, annual average (2015 = 100)", "Eurostat",
            "prc_hicp_aind, prc_hicp_cow", "eurostat", ("eurostat/shared/hicp_eu27.parquet",), ("pollination", "wood", "carbon"),
            note="One extract for all services. Years before 2000 are estimated from the euro area and the other members."),
    Dataset("eurostat_pollination", "Crop production and production value", "Eurostat",
            "apro_cpsh1, aact_eaa02", "eurostat",
            ("eurostat/pollination/production_crop.parquet", "eurostat/pollination/eaa_value_crop.parquet"), ("pollination",)),
    Dataset("eurostat_wood", "Forest volumes, value of forest trees and roundwood exports", "Eurostat",
            "for_vol_efa, for_sup_cp, for_basic", "eurostat",
            ("eurostat/wood/for_vol_efa_thsm3.parquet", "eurostat/wood/for_sup_cp_p1tr.parquet",
             "eurostat/wood/for_basic_rw_exports.parquet"), ("wood", "carbon"),
            note="Carbon reads the increment of all forest for the comparison of allocation keys."),
    Dataset("eurostat_carbon", "CO2 inventory by source sector", "Eurostat", "env_air_gge",
            "eurostat", ("eurostat/carbon/env_air_gge_co2.parquet",), ("carbon",)),
    Dataset("cbm", "Net annual increment of the carbon budget model (EFA FRA CBM data analysis)", "JRC",
            "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/EFA/EFA_FRA_CBM/LATEST/EFA_FRA_CBM_Data_analysis_20210121.xlsx",
            "http", ("jrc/EFA_FRA_CBM_Data_analysis_20210121.xlsx",), ("wood",), 0.002),
]

BY_ID = {d.id: d for d in DATASETS}


def for_service(service: str) -> list[Dataset]:
    return [d for d in DATASETS if service in d.used_by]
