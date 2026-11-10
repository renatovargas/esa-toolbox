"""Crop pollination accounts: pollination potential from the expert-based and bumblebee models, demand from
the pollinator-dependent crops, actual flow and unmet demand in tonnes and in euro, and their maps.

Method: Vallecillo et al. (2018).
"""

from esa_toolbox.core import Stage
from esa_toolbox.services.pollination import inputs, maps, tables

TITLE = "Crop pollination"
STAGES = [
    Stage("prepare", "Regions and the 1 km zone raster", inputs.prepare),
    Stage("model", "Expert-based model at 100 m and its 1 km reduction", inputs.expert_stage, heavy=True, memory_gb=3.0),
    Stage("accounts", "Zone tables, physical and monetary accounts", tables.accounts),
    Stage("maps_1km", "1 km maps, flow by ecosystem type and the workbook", maps.maps_1km),
    Stage("maps_100m", "100 m maps and their checks", maps.maps_100m, heavy=True, memory_gb=4.5),
]
