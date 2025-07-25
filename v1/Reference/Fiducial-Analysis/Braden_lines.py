import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Adapted from M. Ricotti
"""

# Load the CSV file
df = pd.read_csv("analysis_data.csv")

# Drop non-numeric or unhelpful columns
numeric_df = df.select_dtypes(include='number')

# Further filter for columns containing 'luminosity'
luminosity_agg_columns = [
    col for col in df.select_dtypes(include='number').columns
    if 'luminosity' in col and 'agg' in col and 'C' in col
]

plt.figure(figsize=(10, 4))
# Plot each luminosity column
for column in luminosity_agg_columns:
    plt.plot(df.index, df[column]*1e-37, label=column)
    #plt.title(column)   

plt.xlabel("Index")
plt.ylabel(column)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig(f"plot_C.png")
plt.close()

# Further filter for columns containing 'luminosity'
luminosity_agg_columns = [
    col for col in df.select_dtypes(include='number').columns
    if 'luminosity' in col and 'agg' in col and 'O' in col
]

O3_1=[col for col in df.select_dtypes(include='number').columns
    if 'luminosity' in col and 'agg' in col and 'O3_5006' in col
]
O3_2=[col for col in df.select_dtypes(include='number').columns
    if 'luminosity' in col and 'agg' in col and 'O3_4363' in col
]
ratio_O3=np.array(df[O3_1])/np.array(df[O3_2])

C3_1=[col for col in df.select_dtypes(include='number').columns
    if 'luminosity' in col and 'agg' in col and 'C3_1906' in col
]
C3_2=[col for col in df.select_dtypes(include='number').columns
    if 'luminosity' in col and 'agg' in col and 'C3_1908' in col
]
ratio_C3=np.array(df[C3_1])/np.array(df[C3_2])



plt.figure(figsize=(10, 4))
# Plot each luminosity column
for column in luminosity_agg_columns:
    plt.plot(df.index, df[column]*1e-37, label=column)
    #plt.title(column)   

plt.plot(df.index, ratio_O3, label='ratio_O3')
plt.plot(df.index, ratio_C3, label='ratio_C3')

print(ratio_O3[1], ratio_C3[1])

plt.xlabel("Index")
plt.ylabel(column)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig(f"plot_O.png")
plt.close()

# Further filter for columns containing 'luminosity'
luminosity_agg_columns = [
    col for col in df.select_dtypes(include='number').columns
    if 'luminosity' in col and 'agg' in col and 'N' in col
]

plt.figure(figsize=(10, 4))
# Plot each luminosity column
for column in luminosity_agg_columns:
    plt.plot(df.index, df[column]*1e-37, label=column)
    #plt.title(column)   

plt.xlabel("Index")
plt.ylabel(column)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig(f"plot_N.png")
plt.close()

# Further filter for columns containing 'luminosity'
luminosity_agg_columns = [
    col for col in df.select_dtypes(include='number').columns
    if 'luminosity' in col and 'agg' in col and 'Mg' in col
]

plt.figure(figsize=(10, 4))
# Plot each luminosity column
for column in luminosity_agg_columns:
    plt.plot(df.index, df[column]*1e-37, label=column)
    #plt.title(column)   

plt.xlabel("Index")
plt.ylabel(column)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig(f"plot_Mg.png")
plt.close()

# Further filter for columns containing 'luminosity'
luminosity_agg_columns = [
    col for col in df.select_dtypes(include='number').columns
    if 'luminosity' in col and 'agg' in col and 'H1' in col
]

plt.figure(figsize=(10, 4))
# Plot each luminosity column
for column in luminosity_agg_columns:
    plt.plot(df.index, df[column]*1e-37, label=column)
    #plt.title(column)   

plt.xlabel("Index")
plt.ylabel(column)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig(f"plot_H.png")
plt.close()

# Further filter for columns containing 'luminosity'
luminosity_agg_columns = [
    col for col in df.select_dtypes(include='number').columns
    if 'luminosity' in col and 'agg' in col and 'S' in col
]

plt.figure(figsize=(10, 4))
# Plot each luminosity column
for column in luminosity_agg_columns:
    plt.plot(df.index, df[column]*1e-37, label=column)
    #plt.title(column)   

plt.xlabel("Index")
plt.ylabel(column)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig(f"plot_S.png")
plt.close()

# Further filter for columns containing 'luminosity'
luminosity_agg_columns = [
    col for col in df.select_dtypes(include='number').columns
    if 'luminosity' in col and 'agg' in col and 'He' in col
]

plt.figure(figsize=(10, 4))
# Plot each luminosity column
for column in luminosity_agg_columns:
    plt.plot(df.index, df[column]*1e-37, label=column)
    #plt.title(column)   

plt.xlabel("Index")
plt.ylabel(column)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig(f"plot_He.png")
plt.close()
#print(f"Saved plots for {len(luminosity_columns)} luminosity-related columns.")



