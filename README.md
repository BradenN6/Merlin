# Merlin

## Nebular Line Emission Diagnostics from Cosmological Simulations of 
## Early Universe Galaxies.

Author: Braden Nowicki
Advisor: Dr. Massimo Ricotti

Interface with Cloudy Photoionization Software data and RAMSES-RT Simulation
data to create galaxy images and spectra in nebular emission lines.

The Merlin is a small species of falcon from the Northern Hemisphere.
The naming of this package is inspired by the Merlin's exceptionally sharp
eyesight; we generate observational diagnostics from simulated distant,
high-redshift galaxies. Birds are a bellwether of environmental decline;
populations are down nearly 3 billion birds since 1970. Please consider
supporting local efforts to safeguard birds and their habitats.

File Structure:
---------------

MERLIN
|-- CloudyFiles/ :
|-- runs/ :
|   |-- interp.in :
|   |-- run :
|
|-- combine_tables.py :
|-- read_nebula.py :
|
|-- merlin/ : primary package for analysis and visualizations
|   |-- __init__.py
|   |-- emission.py : Generate line emission fields from a Cloudy-generated
|                       linelist table.
|   |-- galaxy_visualization.py : Visualization and analysis routines for
|                       RAMSES-RT Simulations.
|   |-- movies.py : generate movies from sequence of images
|   |-- tools.py : Suite of functions for performing analysis on data
|                       output from many time slices.


Dependencies:
-->
-->
-->
-->
-->
-->
-->
-->
-->
-->
-->
-->