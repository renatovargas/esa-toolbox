"""Carbon sequestration valued at the country-level social cost of carbon of Ricke et al. (2018).

A separate valuation of the national flows of the carbon sequestration accounts. It reads the national tables that
`esa run carbon` writes and leaves them unchanged, and it writes its own workbook and report.
"""

from esa_toolbox.core import Stage
from esa_toolbox.services.carbon_cscc import tables

TITLE = "Carbon sequestration at country-level social costs of carbon"
STAGES = [
    Stage("accounts", "Values at the country-level social cost of carbon and the workbook", tables.accounts,
          note="Reads the national flows of carbon sequestration, so `esa run carbon` comes first"),
]
