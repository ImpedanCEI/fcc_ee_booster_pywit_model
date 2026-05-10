# FCC-ee-HEB

## Main features

This repository contains the booster lattice following the V106.2 of the collider, meaning the LCC optics and the layout of the collider.
The booster is located at about -20 m from the IP in the experimental insertions, with a ballistic optics of 52 m at the middle.
The booster is located at about +0.5585 m in the arcs to maintain the total circumference equal to the one of the collider.


## How to extract the optics

The lattice is now generated directly in python with X-Suite.
The format of the booster line is a 7z compressed file of a json file.
A python script is given in `lattices/z/heb_ring_z_read.py`.

To import the line in a X-Suite environment, the line can be importerd with the following lines:

```python
import xtrack as xt
env = xt.Environment() # Create a X-Suite environment
# Import the line from the JSON-file to the environement. The line name is fcc_heb
env.import_line(xt.Line.from_json("heb_ring_z.json"), line_name = "fcc_heb") 