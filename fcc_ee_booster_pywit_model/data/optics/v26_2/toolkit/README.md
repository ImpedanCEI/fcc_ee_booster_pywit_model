
# HEB FCC optics repository toolkit directory

Collection of scripts that help in common first analysis steps.
Scripts are written to work in conjunction with the example MAD-X scripts in the `examples` or `tests` folder.

## Prerequisites

Scripts ending on `.py` are stand-alone `python3.7` scripts, used to post-process data.
To run those, several packages need to be installed, which is done via

```bash
pip install -r python_requirements.txt
```

Scripts ending on `.madx` are `MAD-X` scripts, used to post-process data.

## Scripts

- *Python Scripts*
    - `plot_lattice.py` - given a `twiss`-file from one of the `optics`-scripts in the `tests` folder, plots the optics functions and saves them as `.png`.
    - `plot_aperture.py` - provided a `aperture`-file from from MAD-X, the beamstay clear, beam pipe radius, and momentum acceptance are plotted.
    - `make_plots.py` - a wrapper script for the three `python` plot script above, used in the `test_*` stage.
- *MAD-X Scripts*
    - `trackMacros.madx` - contains MAD-X macros to perform DA and MA tracking with `MAD-X-PTC`.
