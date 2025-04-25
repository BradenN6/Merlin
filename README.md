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
supporting local efforts to safeguard birds, their migration, and their
habitats.

### File Structure:

MERLIN

merlin/ : primary package for analysis and visualizations
CloudyFiles/ : files related to Cloudy grid runs.
Reference/ : previously-developed analysis code.

#### Creating a Line List with Cloudy

The Cloudy Photoionization Code is used to generate line emission (flux)
throughout the desired parameter space. We consider gas parameters
Ionization Parameter, Hydrogen Number Density, and Temperature.
A simple grid run is performed, simulating emission for lines present in
CloudyFiles/gridrun/LineList_NebularO.dat and
CloudyFiles/gridrun/LineList_NebularCN.dat (can be adjusted)
from a 1 cm thick gas cell. The exact conditions can be adjusted (the star
SED, for instance) for a given run; a simulation for every combination of the
varying parameters is performed. Line list data is output in
'LineList_NebularCN.dat' and 'LineList_NebularO.dat', then combined into a
single 'linelist.dat' file via 'combine_tables.py'.

See CloudyFiles/gridrun. interp6.in is the input file. The run file (with
executable privilege) can be used in your installation of Cloudy.
For instance, in the terminal, './run interp6' would run the desired
simulations (note that grid runs require the use of the -r option, present
in 'run', and therefore the input file is given as interp6, ommitting .in.)

#### Running Merlin on RAMSES-RT Output Data

'main.py' outlines the process: create derived field functions, load
the simulation given by the filepath (as a command line argument) pointing to
the info_*.txt file within an output folder.


#### A Note on the Naming of this Code

The Merlin is a small species of falcon from the Northern Hemisphere.
The naming of this package is inspired by the Merlin's exceptionally sharp
eyesight; we generate observational diagnostics from simulated distant,
high-redshift galaxies. Birds are a bellwether of environmental decline;
populations are down nearly 3 billion birds since 1970. Please consider
supporting local efforts to safeguard birds, their migration, and their
habitats.