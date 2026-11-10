"""Wood provision accounts: the net annual increment of forest available for wood supply as the actual flow,
removals up to the increment as the sustainable flow and above it as overuse, their value, and their maps.

Method: INCA Part III (La Notte et al., 2021) and Vallecillo et al. (2019).
"""

from esa_toolbox.core import Stage
from esa_toolbox.services.wood import inputs, maps, tables

TITLE = "Wood provision"
STAGES = [
    Stage("prepare", "Spatial key, regional shares and carbon budget increment", inputs.prepare, heavy=True, memory_gb=2.5,
          note="Forest times productivity at 100 m, summed by country and by 1 km cell"),
    Stage("accounts", "Physical and monetary accounts and the workbook", tables.accounts),
    Stage("maps_1km", "1 km maps and their checks", maps.maps_1km),
    Stage("maps_100m", "100 m maps and their checks", maps.maps_100m, heavy=True, memory_gb=3.0),
]
