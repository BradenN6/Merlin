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

# Directory containing analysis infofile output from many time slices
# must end in /
#path = '/Users/bnowicki/Research/Scratch/Ricotti/analysis_fid/maindir/'
path = '/Users/bnowicki/Research/Ricotti/fiducial-analysis/infofile-dir/'

lines=["H1_6562.80A","H1_4861.35A","O1_1304.86A","O1_6300.30A","O2_3728.80A",
       "O2_3726.10A","O3_1660.81A","O3_1666.15A","O3_4363.21A","O3_4958.91A",
       "O3_5006.84A","He2_1640.41A","C2_1335.66A","C3_1906.68A","C3_1908.73A",
       "C4_1549.00A","Mg2_2795.53A","Mg2_2802.71A","Ne3_3868.76A","Ne3_3967.47A",
       "N5_1238.82A","N5_1242.80A","N4_1486.50A","N3_1749.67A","S2_6716.44A","S2_6730.82A"]

# Create Simulation_Post_Analysis object and populate a csv
simpost = merlin_spectra.SimulationPostAnalysis('CC-Fiducial-03', path, lines)

df = simpost.populate_table()

print(df.shape)
#print(df.columns)

print(df.columns.tolist())

df_path = os.path.join(os.getcwd(), "CC-Fiducial-03_post_analysis/analysis_data.csv")
df = pd.read_csv(df_path)

lines_plot = lines

column_list = []

for line in lines:
    field = ('gas', f'luminosity_{line}')
    column = str(field) + '_agg'
    column_list.append(column)

print(column_list)

print("Column Check\n")
for i, column in enumerate(column_list):
       if column in df.columns:
             print(column)
    


simpost.lvz(df, lines_plot, group_species=True)
simpost.lvz(df, lines_plot, group_species=False)
# TODO change to Log(Luminosity) label

# TODO plot with star particle aggregate mass
# TODO massimo code
# TODO distribution of densities contributing to ratio/line