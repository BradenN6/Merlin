import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from merlin import post
import itertools
import os

path = '/Users/bnowicki/Documents/Scratch/analysis_fid/maindir/'

simpost = post.Simulation_Post_Analysis('CC-Fiducial', path)

df = simpost.populate_table()

print(df.shape)
print(df.columns)

lines=["H1_6562.80A","O1_1304.86A","O1_6300.30A","O2_3728.80A","O2_3726.10A",
       "O3_1660.81A","O3_1666.15A","O3_4363.21A","O3_4958.91A","O3_5006.84A", 
       "He2_1640.41A","C2_1335.66A","C3_1906.68A","C3_1908.73A","C4_1549.00A",
       "Mg2_2795.53A","Mg2_2802.71A","Ne3_3868.76A","Ne3_3967.47A","N5_1238.82A",
       "N5_1242.80A","N4_1486.50A","N3_1749.67A","S2_6716.44A","S2_6730.82A"]

simpost.lvz(df, lines, group_species=True)