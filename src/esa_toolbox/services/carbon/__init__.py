"""Carbon sequestration accounts: removals and emissions of land from the national inventories, the EU offset,
the actual, offset-free and missed flows, their value at the social cost of carbon, and their maps.

Method: Vallecillo et al. (2019) and La Notte et al. (2021).
"""

from esa_toolbox.core import Stage
from esa_toolbox.services.carbon import inputs, maps, tables

TITLE = "Carbon sequestration"
STAGES = [
    Stage("prepare", "Land types, productivity and regional weights", inputs.prepare,
          note="Corine shares at 1 km, annual productivity on the 1 km grid, land type weights by region"),
    Stage("accounts", "Physical and monetary accounts and the workbook", tables.accounts),
    Stage("maps_1km", "1 km maps and their checks", maps.maps_1km),
    Stage("maps_100m", "100 m maps and their checks", maps.maps_100m, heavy=True, memory_gb=3.0),
]
