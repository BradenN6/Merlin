"""
Braden J. Marazzo-Nowicki

plot_density_pdfs.py

Plot normalised probability distribution functions (PDFs) for:
  - Electron number density        ("gas", "electron_number_density")
  - H nuclei number density        ("gas", "my_H_nuclei_density")
  - Temperature                    ("gas", "my_temperature")

derived from a RAMSES-RT dataset loaded with yt, restricted to a sphere region.

The density curves share the bottom x-axis (cm^-3, log scale).
The temperature curve shares the same y-axis but uses a *separate* top x-axis
(K, log scale) so all three CDFs are visually comparable on one panel.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import yt
import yt_initialization
import merlin_spectra as merlin

# ---------------------------------------------------------------------------
# Load dataset and define sphere
# ---------------------------------------------------------------------------

filename = yt_initialization.filename
ramses_dir = yt_initialization.ramses_dir
logSFC_path = yt_initialization.logSFC_path
lines = yt_initialization.lines
wavelengths = yt_initialization.wavelengths
ds = yt_initialization.ds
ad = yt_initialization.ad
lims_fiducial_00319 = yt_initialization.lims_fiducial_00319
emission_interpolator = yt_initialization.emission_interpolator
output_dir = "/Users/bnowicki/Research/Github/Merlin/drivers"

# Create a visualization object
viz = merlin.VisualizationManager(filename, ramses_dir, logSFC_path, lines, wavelengths, ds, ad, output_dir=output_dir, minimal_output=True, lims_dict=lims_fiducial_00319)

# Star centre of mass; sphere objects; window width
star_ctr = viz.star_center(ad)
sp = ds.sphere(star_ctr, (3000, "pc"))
sp_lum = ds.sphere(star_ctr, (10, 'kpc'))
sp_small = ds.sphere(star_ctr, (1500, "pc"))
sp_tiny = ds.sphere(star_ctr, (500, "pc"))
width = (1500, 'pc')

print(star_ctr)


# ---------------------------------------------------------------------------
# Helper: extract a 1-D numpy array from a sphere for a derived field
# ---------------------------------------------------------------------------
def get_data(region, field):
    """Return a plain numpy array (in CGS / native units) for *field*."""
    return np.array(region[field])          # yt applies unit conversion

# ---------------------------------------------------------------------------
# Pull the three quantities
# ---------------------------------------------------------------------------
n_e   = get_data(sp_tiny, ("gas", "electron_number_density"))   # cm^-3
n_H   = get_data(sp_tiny, ("gas", "my_H_nuclei_density"))       # cm^-3
T     = get_data(sp_tiny, ("gas", "my_temperature"))            # K

# Guard against non-positive values before taking log
def positive_log_bins(arr, n_bins=100):
    """Return log-spaced bin edges covering the positive values of *arr*."""
    arr = arr[arr > 0]
    return np.logspace(np.log10(arr.min()), np.log10(arr.max()), n_bins + 1)

# ---------------------------------------------------------------------------
# Compute normalised histograms  (area = 1 → peak ≤ 1 on linear y-axis)
# ---------------------------------------------------------------------------
def normalised_hist(arr, bins):
    """
    Returns bin centres and PDF values normalised so that the maximum value
    equals 1 (i.e. probability relative to the mode, not a true density).
    This puts the y-axis in [0, 1] regardless of bin width.
    Empty bins are dropped so centres never contain NaN or Inf.
    """
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        raise ValueError("Array has no finite positive values — cannot build histogram.")
    log_bins = np.log10(bins)
    counts, edges = np.histogram(np.log10(arr), bins=log_bins)
    centres = 10 ** (0.5 * (edges[:-1] + edges[1:]))
    # Drop empty bins — they contribute nothing and can skew xlim
    mask = counts > 0
    centres = centres[mask]
    counts  = counts[mask]
    pdf = counts / counts.max()
    return centres, pdf

bins_n = 100    # number of bins for density fields
bins_T = 100    # number of bins for temperature

bins_e = positive_log_bins(n_e, bins_n)
bins_H = positive_log_bins(n_H, bins_n)
bins_T_edges = positive_log_bins(T,   bins_T)

centres_e, pdf_e = normalised_hist(n_e, bins_e)
centres_H, pdf_H = normalised_hist(n_H, bins_H)
centres_T, pdf_T = normalised_hist(T,   bins_T_edges)

print(f"n_e:  min={n_e[n_e>0].min():.3e}, max={n_e.max():.3e}, N={n_e.size}")
print(f"n_H:  min={n_H[n_H>0].min():.3e}, max={n_H.max():.3e}, N={n_H.size}")
print(f"T:    min={T[T>0].min():.3e},  max={T.max():.3e},  N={T.size}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
COLORS = {
    "electron": "#E05C3A",   # warm orange-red
    "hydrogen": "#3A7FE0",   # sky blue
    "temperature": "#2EBF91" # teal-green
}

fig, ax_bot = plt.subplots(figsize=(8, 5))

# --- density curves on the bottom x-axis ---
ax_bot.step(centres_e, pdf_e,
            where="mid",
            color=COLORS["electron"],
            linestyle="-",
            linewidth=1.8,
            label=r"$n_e$  (electron)")

ax_bot.step(centres_H, pdf_H,
            where="mid",
            color=COLORS["hydrogen"],
            linestyle="--",
            linewidth=1.8,
            label=r"$n_\mathrm{H}$  (H nuclei)")

ax_bot.set_xscale("log")
ax_bot.set_xlabel(r"Number density  $[\mathrm{cm}^{-3}]$", fontsize=12)
ax_bot.set_ylabel("Relative probability  (normalised to peak)", fontsize=11)
ax_bot.set_ylim(0, 1.05)
ax_bot.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax_bot.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

# Auto-limits: span both density fields, guarding against NaN/Inf in bin centres
all_n = np.concatenate([centres_e, centres_H])
all_n = all_n[np.isfinite(all_n) & (all_n > 0)]
if all_n.size == 0:
    raise ValueError("No finite positive bin centres found for density fields — "
                     "check that n_e and n_H contain valid positive values.")
ax_bot.set_xlim(all_n.min() * 0.5, all_n.max() * 2)

# --- temperature curve on the top x-axis ---
ax_top = ax_bot.twiny()          # shares the y-axis

ax_top.step(centres_T, pdf_T,
            where="mid",
            color=COLORS["temperature"],
            linestyle=":",
            linewidth=2.0,
            label=r"$T$  (temperature)")

ax_top.set_xscale("log")
ax_top.set_xlabel(r"Temperature  $[\mathrm{K}]$", fontsize=12,
                  color=COLORS["temperature"])
ax_top.tick_params(axis="x", colors=COLORS["temperature"])
ax_top.spines["top"].set_edgecolor(COLORS["temperature"])

T_pos = T[T > 0]
ax_top.set_xlim(T_pos.min() * 0.5, T_pos.max() * 2)

# --- unified legend ---
lines_bot, labels_bot = ax_bot.get_legend_handles_labels()
lines_top, labels_top = ax_top.get_legend_handles_labels()
ax_bot.legend(lines_bot + lines_top,
              labels_bot + labels_top,
              loc="upper left",
              framealpha=0.85,
              fontsize=10)

# --- grid and finishing touches ---
ax_bot.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.4)
ax_bot.grid(True, which="minor", linestyle=":",  linewidth=0.3, alpha=0.25)

# Add redshift
plt.text(0.95, 0.95, f'z = {ds.current_redshift:.5f}', color='black',
         fontsize=9, ha='right', va='top',
         transform=plt.gca().transAxes)

plt.title("Number Density and Temperature PDFs (1.5 kpc sphere region)", fontsize=13, pad=14)
plt.tight_layout()

outfile = "output_00389_ramses_pdfs_500pc.pdf"
plt.savefig(outfile, dpi=150, bbox_inches="tight")
print(f"Saved → {outfile}")
plt.show()