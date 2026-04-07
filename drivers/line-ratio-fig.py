import merlin_spectra as merlin
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# List of lines in Cloudy Table
lines=["H1_6562.80A","H1_4861.35A","O1_1304.86A","O1_6300.30A","O2_3728.80A",
       "O2_3726.10A","O3_1660.81A","O3_1666.15A","O3_4363.21A","O3_4958.91A",
       "O3_5006.84A","He2_1640.41A","C2_1335.66A","C3_1906.68A","C3_1908.73A",
       "C4_1549.00A","Mg2_2795.53A","Mg2_2802.71A","Ne3_3868.76A","Ne3_3967.47A",
       "N5_1238.82A","N5_1242.80A","N4_1486.50A","N3_1749.67A","S2_6716.44A","S2_6730.82A"]

emission_interpolator = merlin.EmissionLineInterpolator(lines, filename=None,
                 use_import=True, linelist_name="linelist-all.dat")



# Recreate Line Ratios from Interpolator
#[O II] 3729/3726 idx 4,5
#[S II] 6717/6731 idx 24,25
#[O III] (5007 + 4959)/4363 idx 10,9,8
OII_3729_interp = emission_interpolator.get_interpolator(4, False)
OII_3726_interp = emission_interpolator.get_interpolator(5, False)
SII_6717_interp = emission_interpolator.get_interpolator(24, False)
SII_6731_interp = emission_interpolator.get_interpolator(25, False)
OIII_5007_interp = emission_interpolator.get_interpolator(10, False)
OIII_4959_interp = emission_interpolator.get_interpolator(9, False)
OIII_4363_interp = emission_interpolator.get_interpolator(8, False)

#tup = np.stack((Uadj, Nadj, Tadj), axis=-1)
#interp_val = interpolator(tup)

##### SII RATIO
# --- Grid setup ---
ne_vals = np.logspace(0, 5, num=12)        # electron density x-axis
temp_vals = np.logspace(3.5, 4.5, num=8)    # 8 temperatures for colorbar

# --- Ionisation parameter & other fixed params ---
# Adjust U and T as needed
log_U_fixed = -3.0

# --- Compute SII ratio for each temperature ---
fig, ax = plt.subplots(figsize=(6, 6))

cmap = cm.get_cmap('turbo', len(temp_vals))
colors = [cmap(i / (len(temp_vals) - 1)) for i in range(len(temp_vals))]

for i, T in enumerate(temp_vals):
    ratio_vals = np.zeros(len(ne_vals))
    for j, ne in enumerate(ne_vals):
        #tup = np.array([[log_U_fixed, np.log10(ne), np.log10(T)]])
        tup = np.stack((log_U_fixed, np.log10(ne), np.log10(T)), axis=-1)
        SII_6717 = SII_6717_interp(tup)[0]
        SII_6731 = SII_6731_interp(tup)[0]
        ratio_vals[j] = SII_6717 / SII_6731

    ax.plot(ne_vals, ratio_vals, color=colors[i], lw=1.5)

# --- Overplot stacked (mean over T) as dashed black ---
ratio_stacked = np.zeros(len(ne_vals))
for j, ne in enumerate(ne_vals):
    ratios_T = []
    for T in temp_vals:
        #tup = np.array([[log_U_fixed, np.log10(ne), np.log10(T)]])
        tup = np.stack((log_U_fixed, np.log10(ne), np.log10(T)), axis=-1)
        SII_6717 = SII_6717_interp(tup)[0]
        SII_6731 = SII_6731_interp(tup)[0]
        ratios_T.append(SII_6717 / SII_6731)
    ratio_stacked[j] = np.mean(ratios_T)

ax.plot(ne_vals, ratio_stacked, color='black', lw=2, linestyle='--', label='stacked (mean T)')

# --- Axes formatting ---
ax.set_xscale('log')
ax.set_xlabel(r'$n_e$ [cm$^{-3}$]', fontsize=13)
ax.set_ylabel(r'[S II] $\lambda6717/\lambda6731$', fontsize=13)
ax.set_xlim(1e0, 1e5)

# Optional: vertical dashed lines 
for xv in [1e2, 1e4]:
    ax.axvline(xv, color='gray', linestyle='--', lw=1)

# --- Colorbar ---
#sm = cm.ScalarMappable(cmap='turbo',
#                       norm=plt.Normalize(vmin=temp_vals.min(), vmax=temp_vals.max()))
#sm.set_array([])
#cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', location='top', pad=0.01)
#cbar.set_label(r'$T_e$ [K]', fontsize=11)
#cbar.set_ticks(np.linspace(temp_vals.min(), temp_vals.max(), 5))
#cbar.set_ticklabels([f'{t:.0f}' for t in np.linspace(temp_vals.min(), temp_vals.max(), 5)])

# --- Colorbar ---
sm = cm.ScalarMappable(cmap='turbo',
                       norm=plt.Normalize(vmin=np.log10(temp_vals.min()), 
                                          vmax=np.log10(temp_vals.max())))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', location='top', pad=0.01)
cbar.set_label(r'$\log_{10}(T_e)$ [K]', fontsize=11)

log_ticks = np.linspace(np.log10(temp_vals.min()), np.log10(temp_vals.max()), 5)
cbar.set_ticks(log_ticks)
cbar.set_ticklabels([f'{t:.2f}' for t in log_ticks])


plt.tight_layout()
plt.savefig('ne_vs_SII_ratio.png', dpi=150, bbox_inches='tight')
plt.show()


#### OII RATIO
# --- Grid setup ---
ne_vals = np.logspace(0, 5, num=12)        # electron density x-axis
temp_vals = np.logspace(3.5, 4.5, num=8)    # 8 temperatures for colorbar

# --- Ionisation parameter & other fixed params ---
# Adjust U and T as needed
log_U_fixed = -3.0

# --- Compute OII ratio for each temperature ---
fig, ax = plt.subplots(figsize=(6, 6))

cmap = cm.get_cmap('turbo', len(temp_vals))
colors = [cmap(i / (len(temp_vals) - 1)) for i in range(len(temp_vals))]

for i, T in enumerate(temp_vals):
    ratio_vals = np.zeros(len(ne_vals))
    for j, ne in enumerate(ne_vals):
        #tup = np.array([[log_U_fixed, np.log10(ne), np.log10(T)]])
        tup = np.stack((log_U_fixed, np.log10(ne), np.log10(T)), axis=-1)
        OII_3729 = OII_3729_interp(tup)[0]
        OII_3726 = OII_3726_interp(tup)[0]
        ratio_vals[j] = OII_3729 / OII_3726

    ax.plot(ne_vals, ratio_vals, color=colors[i], lw=1.5)

# --- Overplot stacked (mean over T) as dashed black ---
ratio_stacked = np.zeros(len(ne_vals))
for j, ne in enumerate(ne_vals):
    ratios_T = []
    for T in temp_vals:
        #tup = np.array([[log_U_fixed, np.log10(ne), np.log10(T)]])
        tup = np.stack((log_U_fixed, np.log10(ne), np.log10(T)), axis=-1)
        OII_3729 = OII_3729_interp(tup)[0]
        OII_3726 = OII_3726_interp(tup)[0]
        ratios_T.append(OII_3729 / OII_3726)
    ratio_stacked[j] = np.mean(ratios_T)

ax.plot(ne_vals, ratio_stacked, color='black', lw=2, linestyle='--', label='stacked (mean T)')

# --- Axes formatting ---
ax.set_xscale('log')
ax.set_xlabel(r'$n_e$ [cm$^{-3}$]', fontsize=13)
ax.set_ylabel(r'[O II] $\lambda3729/\lambda3726$', fontsize=13)
ax.set_xlim(1e0, 1e5)

# Optional: vertical dashed lines 
for xv in [1e2, 1e4]:
    ax.axvline(xv, color='gray', linestyle='--', lw=1)

# --- Colorbar ---
#sm = cm.ScalarMappable(cmap='turbo',
#                       norm=plt.Normalize(vmin=temp_vals.min(), vmax=temp_vals.max()))
#sm.set_array([])
#cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', location='top', pad=0.01)
#cbar.set_label(r'$T_e$ [K]', fontsize=11)
#cbar.set_ticks(np.linspace(temp_vals.min(), temp_vals.max(), 5))
#cbar.set_ticklabels([f'{t:.0f}' for t in np.linspace(temp_vals.min(), temp_vals.max(), 5)])

# --- Colorbar ---
sm = cm.ScalarMappable(cmap='turbo',
                       norm=plt.Normalize(vmin=np.log10(temp_vals.min()), 
                                          vmax=np.log10(temp_vals.max())))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', location='top', pad=0.01)
cbar.set_label(r'$\log_{10}(T_e)$ [K]', fontsize=11)

log_ticks = np.linspace(np.log10(temp_vals.min()), np.log10(temp_vals.max()), 5)
cbar.set_ticks(log_ticks)
cbar.set_ticklabels([f'{t:.2f}' for t in log_ticks])


plt.tight_layout()
plt.savefig('ne_vs_OII_ratio.png', dpi=150, bbox_inches='tight')
plt.show()




#### OIII RATIO
# --- Grid setup ---
ne_vals = np.logspace(0, 5, num=8)        # electron density x-axis
temp_vals = np.logspace(3.5, 4.5, num=10)    # 8 temperatures for colorbar

# --- Ionisation parameter & other fixed params ---
# Adjust U and T as needed
log_U_fixed = -3.0

# --- Compute OII ratio for each temperature ---
fig, ax = plt.subplots(figsize=(6, 6))

cmap = cm.get_cmap('turbo', len(ne_vals))
colors = [cmap(i / (len(ne_vals) - 1)) for i in range(len(ne_vals))]

for i, ne in enumerate(ne_vals):
    ratio_vals = np.zeros(len(temp_vals))
    for j, T in enumerate(temp_vals):
        #tup = np.array([[log_U_fixed, np.log10(ne), np.log10(T)]])
        tup = np.stack((log_U_fixed, np.log10(ne), np.log10(T)), axis=-1)
        OIII_5007 = OIII_5007_interp(tup)[0]
        OIII_4959 = OIII_4959_interp(tup)[0]
        OIII_4363 = OIII_4363_interp(tup)[0]
        ratio_vals[j] = (OIII_5007 + OIII_4959) / OIII_4363

    ax.plot(temp_vals, ratio_vals, color=colors[i], lw=1.5)

# --- Overplot stacked (mean over ne) as dashed black ---
ratio_stacked = np.zeros(len(temp_vals))
for j, T in enumerate(temp_vals):
    ratios_ne = []
    for ne in ne_vals:
        #tup = np.array([[log_U_fixed, np.log10(ne), np.log10(T)]])
        tup = np.stack((log_U_fixed, np.log10(ne), np.log10(T)), axis=-1)
        OIII_5007 = OIII_5007_interp(tup)[0]
        OIII_4959 = OIII_4959_interp(tup)[0]
        OIII_4363 = OIII_4363_interp(tup)[0]
        ratios_ne.append((OIII_5007 + OIII_4959) / OIII_4363)
    ratio_stacked[j] = np.mean(ratios_ne)

ax.plot(temp_vals, ratio_stacked, color='black', lw=2, linestyle='--', label='stacked (mean ne)')

# --- Axes formatting ---
ax.set_xscale('log')
ax.set_xlabel(r'$\log_{10}(T_e)$ [K]', fontsize=13)
ax.set_ylabel(r'[O III] $(\lambda5007 + \lambda 4959)/\lambda4363$', fontsize=13)
ax.set_xlim(10**3.5, 10**4.5)

# Optional: vertical dashed lines 
#for xv in [1e2, 1e4]:
#    ax.axvline(xv, color='gray', linestyle='--', lw=1)

# --- Colorbar ---
#sm = cm.ScalarMappable(cmap='turbo',
#                       norm=plt.Normalize(vmin=temp_vals.min(), vmax=temp_vals.max()))
#sm.set_array([])
#cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', location='top', pad=0.01)
#cbar.set_label(r'$T_e$ [K]', fontsize=11)
#cbar.set_ticks(np.linspace(temp_vals.min(), temp_vals.max(), 5))
#cbar.set_ticklabels([f'{t:.0f}' for t in np.linspace(temp_vals.min(), temp_vals.max(), 5)])

# --- Colorbar ---
sm = cm.ScalarMappable(cmap='turbo',
                       norm=plt.Normalize(vmin=np.log10(ne_vals.min()), 
                                          vmax=np.log10(ne_vals.max())))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', location='top', pad=0.01)
cbar.set_label(r'$n_e$ [cm$^{-3}$]', fontsize=11)

log_ticks = np.linspace(np.log10(ne_vals.min()), np.log10(ne_vals.max()), 5)
cbar.set_ticks(log_ticks)
cbar.set_ticklabels([f'{t:.2f}' for t in log_ticks])


plt.tight_layout()
plt.savefig('temp_vs_OIII_ratio.png', dpi=150, bbox_inches='tight')
plt.show()