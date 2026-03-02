import merlin_spectra.galaxy_visualization
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import merlin_spectra
import itertools
import os

"""
Post-Analysis of the Fiducial Simulation

Authors: Braden Marazzo-Nowicki, Massimo Ricotti
2025-07-28
"""

# Directory containing analysis output from many time slices
path = '/Users/bnowicki/Research/Scratch/Ricotti/analysis_fid/maindir/'

# Create Simulation_Post_Analysis object and populate a csv
simpost = merlin_spectra.SimulationPostAnalysis('CC-Fiducial', path)

df = simpost.populate_table()

print(df.shape)
print(df.columns)

lines=["H1_6562.80A","O1_1304.86A","O1_6300.30A","O2_3728.80A","O2_3726.10A",
       "O3_1660.81A","O3_1666.15A","O3_4363.21A","O3_4958.91A","O3_5006.84A", 
       "He2_1640.41A","C2_1335.66A","C3_1906.68A","C3_1908.73A","C4_1549.00A",
       "Mg2_2795.53A","Mg2_2802.71A","Ne3_3868.76A","Ne3_3967.47A","N5_1238.82A",
       "N5_1242.80A","N4_1486.50A","N3_1749.67A","S2_6716.44A","S2_6730.82A"]

df_path = os.path.join(os.getcwd(), "CC-Fiducial_post_analysis/analysis_data.csv")
df = pd.read_csv(df_path)

simpost.lvz(df, lines, group_species=True)