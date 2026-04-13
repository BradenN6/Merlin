#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 29 17:05:30 2025

@author: ricotti
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Reload the CSV file
file_path = "analysis_data.csv"
df = pd.read_csv(file_path)



z=np.array(df["current_redshift"])
time=500*((1+z)/10)**(-1.5)


stars=np.array(df["('star', 'particle_mass')_agg"])*1e10

sfr=20.*(stars[1:]-stars[:-1])

#print(f'stars {stars}')

# Extract the required O III lines for the density diagnostic ratio
s2_6716 = df["('gas', 'luminosity_S2_6716.44A')_agg"]*1e-37
s2_6731 = df["('gas', 'luminosity_S2_6730.82A')_agg"]*1e-37

# Extract the required O III lines for the density diagnostic ratio
o3_5007 = df["('gas', 'luminosity_O3_5006.84A')_agg"]*1e-37
o3_4363 = df["('gas', 'luminosity_O3_4363.21A')_agg"]*1e-37
o3_4959 = df["('gas', 'luminosity_O3_4958.91A')_agg"]*1e-37

# Extract the required O III lines for the density diagnostic ratio
o2_3729 = df["('gas', 'luminosity_O2_3728.80A')_agg"]*1e-37
o2_3726 = df["('gas', 'luminosity_O2_3726.10A')_agg"]*1e-37

# Extract the required O III lines for the density diagnostic ratio
c3_1906 = df["('gas', 'luminosity_C3_1906.68A')_agg"]*1e-37
c3_1909 = df["('gas', 'luminosity_C3_1908.73A')_agg"]*1e-37

# Compute the ratio: (O III 5007) / (O III 4363)
s2_ratio = s2_6716/s2_6731
s2_ratio.replace([np.inf, -np.inf], np.nan, inplace=True)

# Compute the ratio: (O III 5007) / (O III 4363)
o3_ratio = o3_4363/o3_5007*30.0
o3_ratio.replace([np.inf, -np.inf], np.nan, inplace=True)

o3_ratio_full = (o3_5007 + o3_4959) / o3_4363

print(o3_ratio)
print(o3_ratio_full)

# Compute the ratio: (O II 3729) / (O II 3726)
o2_ratio = o2_3729 / o2_3726
o2_ratio.replace([np.inf, -np.inf], np.nan, inplace=True)
#o2_ratio=np.where(o2_3729 > 0.1, o2_ratio, 0.0)

# Compute the ratio: (O II 3729) / (O II 3726)
c3_ratio = c3_1906 / c3_1909
c3_ratio.replace([np.inf, -np.inf], np.nan, inplace=True)

# Compute the ratio: (O II 3729) / (O II 3726)
o32_ratio = o3_5007 / o2_3726
o32_ratio.replace([np.inf, -np.inf], np.nan, inplace=True)


x=df.index
# Plot the ratio
plt.figure(figsize=(10, 5))
plt.plot(z, o3_ratio, label='[O III] 4363 / 5007')

plt.scatter(z, o2_ratio, s=10*o2_3729, label='[O II] 3729 / 3726')
plt.plot(z, o2_ratio,'b-',lw=1.0)
 
plt.scatter(z, s2_ratio, s=20*s2_6716,label='[S II] 6716 / 6731')
plt.plot(z, s2_ratio, '-',lw=1.0)

plt.scatter(z, c3_ratio, s=10*c3_1906, label='[C III] 1906 / 1909')
plt.plot(z, c3_ratio, '-',lw=1.0)

plt.plot(z, o32_ratio/3., label='[O III] 5007 / [O II] 3726')

plt.plot(z[:-1], sfr, label='sfr')
plt.xlabel("Redshift")
plt.ylabel("O III Line Ratio")
plt.title("Oxygen III Line Ratio as Density Diagnostic")
plt.grid(True)
plt.gca().invert_xaxis()
#plt.xlim(505,520)
plt.ylim(0.5,1.7)
plt.legend()
plt.tight_layout()

# Save and show the plot
plot_path = "o3_density_diagnostic_ratio.png"
plt.savefig(plot_path)
plt.close()

plot_path


# Extract additional physical parameters
temperature = df["('gas', 'temperature')_mean"]
#density = df["('gas', 'density')_mean"]
edensity = df["('gas', 'electron_number_density')_mean"]

# Create an overlaid plot with a secondary y-axis
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot O III ratio on the primary y-axis
ax1.plot(df.current_redshift, o3_ratio_full, 'b-', label=r'[O III] ($\lambda$ 5006.84A + $\lambda$ 4958.91A)/$\lambda$ 4363.21A')
ax1.set_xlabel("Redshift")
ax1.set_ylabel("[O III] Line Ratio", color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Instantiate a second y-axis that shares the same x-axis
ax2 = ax1.twinx()

# Plot density and temperature on the secondary y-axis
#ax2.plot(df.current_redshift, density/1.6e-24, 'g--', label='Density (mean)')
ax2.plot(df.current_redshift, edensity, 'g--', label='Density (mean)')
ax2.plot(df.current_redshift, temperature, 'r-.', label='Temperature (mean)')
ax2.set_ylabel(r"Electro Number Density [cm$^{-3}$], Temperature [K]", color='k')
ax2.tick_params(axis='y', labelcolor='k')
ax2.set_yscale('log')

# Add grid, legend, and title
fig.suptitle("Oxygen III Line Ratio with Density and Temperature Overlay")
fig.tight_layout()
fig.subplots_adjust(top=0.9)  # Adjust title position

# Combine legends from both axes
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')
plt.gca().invert_xaxis()
plt.grid(True)

# Save the combined plot
overlay_plot_path = "o3_density_temp_overlay.png"
plt.savefig(overlay_plot_path)
plt.close()

import pyneb as pn
#import numpy as np
#import matplotlib.pyplot as plt

def R(ne):
    return (3.77e-4*ne+1.5)/(1.29e-3*ne+1.05)+0.05

def ne(R):
    return ((R-0.05)*1.05-1.5)/(3.77e-4-(R-0.05)*1.29e-3)

def R_s2(ne):
    return (3.77e-4*ne+1.4)/(1.29e-3*ne+1.05)+0.14

def ne_s2(R):
    return ((R-0.14)*1.05-1.4)/(3.77e-4-(R-0.14)*1.29e-3)

# Define O II ion
O2 = pn.Atom('O', 2)  # O+ ion
C3 = pn.Atom('C', 3)  # C++ ion 
S2 = pn.Atom('S', 2)  # C++ ion 

# Define electron temperature (typical for H II regions)
Te = 10000  # in Kelvin

# Define a range of electron densities (log-spaced)
ne_values = np.logspace(1, 5, 200)  # from 10 to 100,000 cm^-3

# Compute line intensities: OII 3726 and OII 3729
I_3726 = O2.getEmissivity(tem=Te, den=ne_values, wave=3726)
I_3729 = O2.getEmissivity(tem=Te, den=ne_values, wave=3729)

I_1906 = C3.getEmissivity(tem=Te, den=ne_values, wave=1906)
I_1909 = C3.getEmissivity(tem=Te, den=ne_values, wave=1909)

I_6716 = S2.getEmissivity(tem=Te, den=ne_values, wave=6716)
I_6731 = S2.getEmissivity(tem=Te, den=ne_values, wave=6731)
# Compute the ratio
ratio = I_3729 / I_3726

ratio_c = I_1906 / I_1909
ratio_s2 = I_6716 / I_6731

z=np.polyfit(ratio_c, ne_values,4)
ne_c3=np.poly1d(z)

#z1=np.polyfit(ratio_s2, ne_values,2)
#ne_s2=np.poly1d(z1)

mag=2.0
xx=np.linspace(0.45,1.45,50)
# Plotting
plt.figure(figsize=(8, 5))
plt.plot(ne_values, ratio, label='[O II] 3729 / 3726')
plt.plot(ne_values, ratio_c, label='[C III] 1906 / 1909')
plt.plot(ne_values, ratio_s2, label='[S II] 6716 / 6731')
plt.plot(ne(xx), xx , label='fit')
plt.plot(ne_c3(xx),xx, label='fit c3')
plt.plot(ne_s2(xx),xx, label='fit s2')

plt.xscale('log')
plt.xlabel('Electron Density $n_e$ (cm$^{-3}$)')
plt.ylabel('Line Ratio')
plt.title('Density Diagnostic Line Ratios vs Electron Density\n(Te = 10,000 K)')
plt.grid(True)
plt.axhline(1.5, color='gray', linestyle='--', label='Low-density limit')
plt.axhline(0.35, color='gray', linestyle=':', label='High-density limit')
plt.legend()
plt.tight_layout()
plt.savefig('theoretical_ratios.png')
#plt.show()

# Plot the ratio
plt.figure(figsize=(10, 5))
#plt.plot(time, o3_ratio, label='[O III] 4363 / 5007')

plt.scatter(df.current_redshift, ne(o2_ratio), color='orange', s=mag*10*o2_3729, label=r'[O II] $\lambda$ 3728.80A / $\lambda$ 3726.10A')
plt.plot(df.current_redshift, ne(o2_ratio),'-',color='orange', lw=1.0)
 
#plt.scatter(time, s2_ratio, s=20*s2_6716,label='[S II] 6716 / 6731')
#plt.plot(time, s2_ratio, '-',lw=1.0)

yy=ne_c3(c3_ratio)
yy=np.where(yy>10, yy, 10.)
plt.scatter(df.current_redshift, yy, s=mag*10*c3_1906, color='b', label=r'[C III] $\lambda$ 1906.68A / $\lambda$ 1908.73A')
plt.plot(df.current_redshift, yy, '-',color='b', lw=1.0)

yy=ne_s2(s2_ratio)
yy=np.where(yy>10, yy, 10.)
plt.scatter(df.current_redshift, yy, s=mag*20*s2_6716, color='r', label=r'[S II] $\lambda$ 6716.44A / $\lambda$ 6730.82A')
plt.plot(df.current_redshift, yy, '-',color='r', lw=1.0)

#plt.plot(df.current_redshift, df["('gas', 'electron_number_density')_max"], '-', lw=1.0, label='Max Electron Number Density')
# TODO mean with smaller region?

#plt.plot(time, o32_ratio/3., label='[O III] 5007 / [O II] 3726')

#plt.plot(time[:-1], 800*sfr, color='green', label='sfr')

z,ra,mass,n_H,met=np.loadtxt('logSFC-Fiducial', unpack=True, usecols=[2,4,7,8,9])
time=500*((1+z)/10)**(-1.5)

ti=np.linspace(325,570.9,1000)
massc=np.cumsum(mass)
massci=np.interp(ti,time,massc)

dti=(ti[1:]-ti[:-1])*1e6
sfri=(massci[1:]-massci[:-1])/dti
#axs[0].plot(ti[1:],sfri)

###
z_sfc, mass_sfc = np.loadtxt('logSFC-Fiducial', unpack=True, usecols=[2, 7])

time_sfc     = 500. * ((1. + z_sfc) / 10.) ** (-1.5)
ti           = np.linspace(time_sfc.min(), time_sfc.max(), 2000)
mass_cumul_i = np.interp(ti, time_sfc, np.cumsum(mass_sfc))

dti  = (ti[1:] - ti[:-1]) * 1e6
sfri = (mass_cumul_i[1:] - mass_cumul_i[:-1]) / dti   # Msun yr⁻¹

ti_mid = 0.5 * (ti[:-1] + ti[1:])
z_sfr  = 10. * (ti_mid / 500.) ** (-2. / 3.) - 1.

# Restrict SFR to the redshift range present in the CSV
z_min, z_max = df.current_redshift.min(), df.current_redshift.max()
sfr_mask = (z_sfr >= z_min) & (z_sfr <= z_max)
z_sfr = z_sfr[sfr_mask]
sfri  = sfri[sfr_mask]
###

#plt.plot(ti[:-1], 5e3*sfri, color='green', label='sfr')
plt.plot(z_sfr, 5e3*sfri, color='green', label='sfr')

plt.xlabel("Redshift")
plt.ylabel(r"Electron Number Density [cm$^{-3}$]")
plt.title("Line Ratio as Density Diagnostic")
plt.grid(True)
plt.yscale('log')
plt.gca().invert_xaxis()
#plt.xlim(505,520)
#plt.xlim(475,575)
plt.ylim(10.0,8000)
plt.legend()
plt.tight_layout()

# Save and show the plot
plot_path = "density_diagnostic_ratio.png"
plt.savefig(plot_path)
plt.close()

overlay_plot_path


############

# All line ratios over time normalised to 1 with SFR, and ionized/neutral gas ratio -------------


plt.figure(figsize=(10, 5))

# Plot the ratios
#plt.plot(df.current_redshift, o3_ratio_full / max(o3_ratio_full), label=r'[O III] ($\lambda$ 5006.84A + $\lambda$ 4958.91A)/$\lambda$ 4363.21A')

plt.scatter(df.current_redshift, o2_ratio, s=10*o2_3729, color='b', label=r'[O II] $\lambda$ 3728.80A / $\lambda$ 3726.10A')
plt.plot(df.current_redshift, o2_ratio,'b-',color='b', lw=1.0)
 
plt.scatter(df.current_redshift, s2_ratio, s=20*s2_6716,color='orange', label=r'[S II] $\lambda$ 6716.44A / $\lambda$ 6730.82A')
plt.plot(df.current_redshift, s2_ratio, '-',color='orange', lw=1.0)

plt.scatter(df.current_redshift, c3_ratio, s=10*c3_1906, color='teal', label=r'[C III] $\lambda$ 1906.68A / $\lambda$ 1908.73A')
plt.plot(df.current_redshift, c3_ratio, '-',color='teal', lw=1.0)

#lt.plot(z, o32_ratio / max(o2_ratio), label=r'[O III] $\lambda$ 5006.84A / [O II] $\lambda$ 3726.10A')

plt.plot(z_sfr, sfri / max(sfri) * max(s2_ratio), color='green', label='sfr', linestyle='dotted')


stellar_mass = np.array(df["Stellar_Mass_Msun"])
gas_mass     = np.array(df["Gas_Mass_Msun"])
ionised_mass = np.array(df["Ionised_Gas_Mass_Msun"])

#gas_ratio = ionised_mass / gas_mass
#plt.plot(df.current_redshift, gas_ratio, color='orange', label=r'M$_{HII}$/M$_{HI}$')

#plt.plot(z, sfr, label='sfr')
plt.xlabel("Redshift")
plt.ylabel("Density Diagnostic Line Ratio")
#plt.title("Line Ratio Evolution")
plt.grid(True)
plt.gca().invert_xaxis()
#plt.xlim(505,520)
#plt.ylim(0.0,1.02)
plt.legend()
plt.tight_layout()

# Save and show the plot
plot_path = "density_line_ratios_sfr.png"
plt.savefig(plot_path)
plt.close()

plot_path


# Density Diagnostics with Density ------------

# Temperature Diagnostics with Temperature -----------
