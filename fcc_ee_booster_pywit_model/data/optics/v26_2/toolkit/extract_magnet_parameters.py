"""Script to extract the magnet parameters from the json file to use it in a python script
"""

import xtrack as xt
from pathlib import Path

WRK_DIR = Path(__file__).parent
PATH_JSON_FILE = WRK_DIR / "../lattices/z/heb_ring_z.json" # path to the json file
PATH_NEW_FILE = WRK_DIR / "heb_ring_z.csv" # path to the json file
X0 = -12.0 # IP offset betwenn the booster and the collider
BRHO_INJ = 66.71281904 # Minimum magnetic rigidity (at injection)
BRHO_TTBAR = 608.7544737 # Maximum magnetic rigidity (at ttbar extraction)

env = xt.Environment() # Create a X-Suite environment
# Import the line from the JSON-file to the environement. The line name is fcc_heb
env.import_line(xt.Line.from_json(PATH_JSON_FILE), line_name = "fcc_heb") 

elem_attributes = env.fcc_heb.get_table(attr=True)
# Adds the positions of the entrance magnet
surv = env.fcc_heb.survey(X0=X0)
elem_attributes["X"] = surv["X"]
elem_attributes["Y"] = surv["Y"]
elem_attributes["Z"] = surv["Z"]
# Adds the magnetic field at injection
length = elem_attributes["length"] + 1e-15 # To avoid divinding by zero
elem_attributes["B_inj"] = elem_attributes["k0l"]/length*BRHO_INJ
elem_attributes["B'_inj"] = elem_attributes["k1l"]/length*BRHO_INJ
elem_attributes["B''_inj"] = elem_attributes["k2l"]/length*BRHO_INJ
# Adds the magnetic field at ttbar
elem_attributes["B_ttbar"] = elem_attributes["k0l"]/length*BRHO_TTBAR
elem_attributes["B'_ttbar"] = elem_attributes["k1l"]/length*BRHO_TTBAR
elem_attributes["B''_ttbar"] = elem_attributes["k2l"]/length*BRHO_TTBAR
# Selects only the bends, quadrupoles and sextupoles
output = elem_attributes.rows.match(element_type="Quadrupole|Bend|Sextupole").cols["element_type s X Y Z length B_inj B_ttbar B'_inj B'_ttbar B''_inj B''_ttbar"]
output.to_csv(PATH_NEW_FILE)