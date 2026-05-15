import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Plots of fields evolving over time


# Load the CSV file
df = pd.read_csv("analysis_data.csv")

# Drop non-numeric or unhelpful columns
numeric_df = df.select_dtypes(include='number')

print(df.columns.tolist())