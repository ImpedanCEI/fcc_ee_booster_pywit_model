"""Script to extract the line from the json file to use it in a python script
"""

import xtrack as xt
env = xt.Environment() # Create a X-Suite environment
# Import the line from the JSON-file to the environement. The line name is fcc_heb
env.import_line(xt.Line.from_json("heb_ring_z.json"), line_name = "fcc_heb") 

# Computes the Twiss table
tw0 = env.fcc_heb.twiss(compute_chromatic_properties=True)

# Plots the Twiss parameters
tw0.plot(lattice=False)
tw0.plot(lattice=False, yl="wx_chrom wy_chrom", yr="ddx")