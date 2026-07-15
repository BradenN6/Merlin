#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 1: Stellar Mass + SFR vs. Redshift
Figure 2: Total Gas Mass (left) + Ionised Gas Mass (right) vs. Redshift
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Load main data ───────────────────────────────────────────────────────────
file_path = "analysis_data.csv"
df = pd.read_csv(file_path)

redshift     = np.array(df["current_redshift"])
stellar_mass = np.array(df["Stellar_Mass_Msun"])
gas_mass     = np.array(df["Gas_Mass_Msun"])
ionised_mass = np.array(df["Ionised_Gas_Mass_Msun"])

# ── Load SFR from logSFC file ────────────────────────────────────────────────
z_sfc, mass_sfc = np.loadtxt('logSFC-Fiducial', unpack=True, usecols=[2, 7])

time_sfc     = 500. * ((1. + z_sfc) / 10.) ** (-1.5)
ti           = np.linspace(time_sfc.min(), time_sfc.max(), 2000)
mass_cumul_i = np.interp(ti, time_sfc, np.cumsum(mass_sfc))

dti  = (ti[1:] - ti[:-1]) * 1e6
sfri = (mass_cumul_i[1:] - mass_cumul_i[:-1]) / dti   # Msun yr⁻¹

ti_mid = 0.5 * (ti[:-1] + ti[1:])
z_sfr  = 10. * (ti_mid / 500.) ** (-2. / 3.) - 1.

# Restrict SFR to the redshift range present in the CSV
z_min, z_max = redshift.min(), redshift.max()
sfr_mask = (z_sfr >= z_min) & (z_sfr <= z_max)
z_sfr = z_sfr[sfr_mask]
sfri  = sfri[sfr_mask]

# ════════════════════════════════════════════════════════════════════════════
# Figure 1 – Stellar Mass + SFR
# ════════════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(10, 6))

l1, = ax1.plot(redshift, stellar_mass, color='steelblue', lw=2.0,
               label=r'Stellar Mass')
ax1.set_xlabel(r'Redshift $z$', fontsize=13)
ax1.set_ylabel(r'Stellar Mass $[\mathrm{M_\odot}]$', fontsize=13, color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue', labelsize=11)
ax1.tick_params(axis='x', labelsize=11)
ax1.set_yscale('log')
ax1.invert_xaxis()
ax1.grid(True, which='both', alpha=0.3, linestyle=':')

ax2 = ax1.twinx()
l2, = ax2.plot(z_sfr, sfri, color='crimson', lw=1.5, alpha=0.85,
               label=r'SFR')
ax2.set_ylabel(r'SFR $[\mathrm{M_\odot \, yr^{-1}}]$', fontsize=13, color='crimson')
ax2.tick_params(axis='y', labelcolor='crimson', labelsize=11)
ax2.set_yscale('log')

lines  = [l1, l2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, fontsize=11, loc='upper left')

fig1.suptitle('Stellar Mass & Star Formation Rate vs. Redshift', fontsize=14)
fig1.tight_layout()
fig1.savefig("stellar_mass_sfr_vs_redshift.png", dpi=150)
plt.close(fig1)
print("Saved → stellar_mass_sfr_vs_redshift.png")

# ════════════════════════════════════════════════════════════════════════════
# Figure 2 – Total Gas Mass (left) + Ionised Gas Mass (right)
# ════════════════════════════════════════════════════════════════════════════
fig2, ax3 = plt.subplots(figsize=(10, 6))

l3, = ax3.plot(redshift, gas_mass, color='darkorange', lw=2.0,
               label=r'Total Gas Mass')
ax3.set_xlabel(r'Redshift $z$', fontsize=13)
ax3.set_ylabel(r'Total Gas Mass $[\mathrm{M_\odot}]$', fontsize=13, color='darkorange')
ax3.tick_params(axis='y', labelcolor='darkorange', labelsize=11)
ax3.tick_params(axis='x', labelsize=11)
ax3.set_yscale('log')
ax3.invert_xaxis()
ax3.grid(True, which='both', alpha=0.3, linestyle=':')

ax4 = ax3.twinx()
l4, = ax4.plot(redshift, ionised_mass, color='mediumseagreen', lw=2.0,
               linestyle='--', label=r'Ionised Gas Mass')
ax4.set_ylabel(r'Ionised Gas Mass $[\mathrm{M_\odot}]$', fontsize=13,
               color='mediumseagreen')
ax4.tick_params(axis='y', labelcolor='mediumseagreen', labelsize=11)
ax4.set_yscale('log')

lines2  = [l3, l4]
labels2 = [l.get_label() for l in lines2]
ax3.legend(lines2, labels2, fontsize=11, loc='upper left')

fig2.suptitle('Total & Ionised Gas Mass vs. Redshift', fontsize=14)
fig2.tight_layout()
fig2.savefig("gas_masses_vs_redshift.png", dpi=150)
plt.close(fig2)
print("Saved → gas_masses_vs_redshift.png")