"""
Braden J. Marazzo-Nowicki

plot_density_pdfs.py

Provides `plot_density_pdfs()` — a self-contained function to plot normalised
probability distribution functions (PDFs) for:
  - Electron number density        ("gas", "electron_number_density")
  - H nuclei number density        ("gas", "my_H_nuclei_density")
  - Temperature                    ("gas", "my_temperature")

derived from a RAMSES-RT dataset loaded with yt, restricted to a sphere region.

Density curves share the bottom x-axis (cm^-3, log scale).
The temperature curve uses a separate top x-axis (K, log scale) so all three
PDFs are visually comparable on one panel.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _positive_log_bins(arr, n_bins=100):
    """Return log-spaced bin edges covering the positive values of *arr*."""
    arr = arr[arr > 0]
    return np.logspace(np.log10(arr.min()), np.log10(arr.max()), n_bins + 1)


def _normalised_hist(arr, bins):
    """
    Return bin centres and PDF values normalised so that the maximum equals 1
    (probability relative to the mode, not a true density).

    Empty bins are dropped so centres never contain NaN or Inf.

    Parameters
    ----------
    arr : array-like
        Raw data values (may contain non-positive or non-finite entries).
    bins : array-like
        Log-spaced bin edges (as returned by `_positive_log_bins`).

    Returns
    -------
    centres : ndarray
        Geometric bin centres for non-empty bins.
    pdf : ndarray
        Peak-normalised counts (values in [0, 1]).
    """
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        raise ValueError(
            "Array has no finite positive values — cannot build histogram."
        )
    log_bins = np.log10(bins)
    counts, edges = np.histogram(np.log10(arr), bins=log_bins)
    centres = 10 ** (0.5 * (edges[:-1] + edges[1:]))

    # Drop empty bins
    mask = counts > 0
    centres = centres[mask]
    counts  = counts[mask]

    return centres, counts / counts.max()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_density_pdfs(
    region,
    *,
    n_e_field=("gas", "electron_number_density"),
    n_H_field=("gas", "my_H_nuclei_density"),
    T_field=("gas", "my_temperature"),
    n_bins_density=100,
    n_bins_T=100,
    density_range=None,
    temp_range=None,
    redshift=None,
    title="Number Density and Temperature PDFs",
    outfile=None,
    dpi=150,
    colors=None,
    figsize=(8, 5),
    ax=None,
):
    """
    Plot peak-normalised PDFs for electron density, H nuclei density, and
    temperature extracted from a yt region (sphere, box, etc.).

    Density fields share the **bottom** x-axis (cm⁻³, log scale).
    The temperature field uses a **top** x-axis (K, log scale).
    All three share the same y-axis (relative probability, 0–1).

    Parameters
    ----------
    region : yt data object
        Any yt region that supports field indexing (sphere, box, …).
    n_e_field : tuple, optional
        yt field specifier for electron number density (cm⁻³).
    n_H_field : tuple, optional
        yt field specifier for H nuclei number density (cm⁻³).
    T_field : tuple, optional
        yt field specifier for temperature (K).
    n_bins_density : int, optional
        Number of histogram bins for density fields.
    n_bins_T : int, optional
        Number of histogram bins for temperature.
    density_range : tuple(float, float) or None, optional
        Fixed (min, max) limits in cm⁻³ for the density bin edges **and**
        the bottom x-axis.  Data outside this range is excluded before
        histogramming.  If ``None``, limits are derived from the data.
        Example: ``density_range=(1e-4, 1e4)``.
    temp_range : tuple(float, float) or None, optional
        Fixed (min, max) limits in K for the temperature bin edges **and**
        the top x-axis.  Data outside this range is excluded before
        histogramming.  If ``None``, limits are derived from the data.
        Example: ``temp_range=(10, 1e8)``.
    redshift : float or None, optional
        If provided, annotated in the top-right corner.
        Pass ``ds.current_redshift`` directly.
    title : str, optional
        Figure title.
    outfile : str or None, optional
        If given, the figure is saved to this path.
    dpi : int, optional
        Resolution used when saving.
    colors : dict or None, optional
        Override default colours.  Recognised keys:
        ``"electron"``, ``"hydrogen"``, ``"temperature"``.
    figsize : tuple, optional
        ``(width, height)`` in inches (ignored when *ax* is supplied).
    ax : matplotlib.axes.Axes or None, optional
        Bottom axes to draw into.  If ``None``, a new figure is created.
        Supplying an existing axes lets you embed the plot in a larger layout.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax_bot : matplotlib.axes.Axes   (bottom / density axis)
    ax_top : matplotlib.axes.Axes   (top / temperature axis)
    """

    # --- defaults --------------------------------------------------------
    _colors = {
        "electron":    "#E05C3A",   # warm orange-red
        "hydrogen":    "#3A7FE0",   # sky blue
        "temperature": "#2EBF91",   # teal-green
    }
    if colors:
        _colors.update(colors)

    # --- extract data ----------------------------------------------------
    n_e = np.array(region[n_e_field])
    n_H = np.array(region[n_H_field])
    T   = np.array(region[T_field])

    # --- apply optional range clipping -----------------------------------
    # Clipping is done before binning so that bin edges span exactly the
    # requested range and the histogram only counts in-range data.
    def _clip_positive(arr, range_tuple):
        """Keep only finite positive values within (lo, hi) if given."""
        mask = np.isfinite(arr) & (arr > 0)
        if range_tuple is not None:
            lo, hi = range_tuple
            mask &= (arr >= lo) & (arr <= hi)
        return arr[mask]

    n_e_clipped = _clip_positive(n_e, density_range)
    n_H_clipped = _clip_positive(n_H, density_range)
    T_clipped   = _clip_positive(T,   temp_range)

    # --- bin edges -------------------------------------------------------
    # If a range was supplied, use it directly for the bin edges so that
    # bins span the full requested interval even if no data sits at the edges.
    def _make_bins(arr_clipped, range_tuple, n_bins):
        if range_tuple is not None:
            lo, hi = range_tuple
            return np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)
        return _positive_log_bins(arr_clipped, n_bins)

    bins_e = _make_bins(n_e_clipped, density_range, n_bins_density)
    bins_H = _make_bins(n_H_clipped, density_range, n_bins_density)
    bins_T = _make_bins(T_clipped,   temp_range,    n_bins_T)

    # --- histograms ------------------------------------------------------
    centres_e, pdf_e = _normalised_hist(n_e_clipped, bins_e)
    centres_H, pdf_H = _normalised_hist(n_H_clipped, bins_H)
    centres_T, pdf_T = _normalised_hist(T_clipped,   bins_T)

    # Diagnostics
    print(f"n_e : min={n_e_clipped.min():.3e}, max={n_e_clipped.max():.3e}, N={n_e_clipped.size} (of {n_e.size} total)")
    print(f"n_H : min={n_H_clipped.min():.3e}, max={n_H_clipped.max():.3e}, N={n_H_clipped.size} (of {n_H.size} total)")
    print(f"T   : min={T_clipped.min():.3e},  max={T_clipped.max():.3e},  N={T_clipped.size} (of {T.size} total)")

    # --- figure / axes setup ---------------------------------------------
    if ax is None:
        fig, ax_bot = plt.subplots(figsize=figsize)
    else:
        ax_bot = ax
        fig = ax_bot.figure

    # --- density curves (bottom x-axis) ----------------------------------
    ax_bot.step(
        centres_e, pdf_e,
        where="mid",
        color=_colors["electron"],
        linestyle="-",
        linewidth=1.8,
        label=r"$n_e$  (electron)",
    )
    ax_bot.step(
        centres_H, pdf_H,
        where="mid",
        color=_colors["hydrogen"],
        linestyle="--",
        linewidth=1.8,
        label=r"$n_\mathrm{H}$  (H nuclei)",
    )

    ax_bot.set_xscale("log")
    ax_bot.set_xlabel(r"Number density  $[\mathrm{cm}^{-3}]$", fontsize=12)
    ax_bot.set_ylabel("Relative probability  (normalised to peak)", fontsize=11)
    ax_bot.set_ylim(0, 1.05)
    ax_bot.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax_bot.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

    all_n = np.concatenate([centres_e, centres_H])
    all_n = all_n[np.isfinite(all_n) & (all_n > 0)]
    if all_n.size == 0:
        raise ValueError(
            "No finite positive bin centres for density fields — "
            "check that n_e and n_H contain valid positive values."
        )
    if density_range is not None:
        ax_bot.set_xlim(density_range[0], density_range[1])
    else:
        ax_bot.set_xlim(all_n.min() * 0.5, all_n.max() * 2)

    # --- temperature curve (top x-axis) ----------------------------------
    ax_top = ax_bot.twiny()

    ax_top.step(
        centres_T, pdf_T,
        where="mid",
        color=_colors["temperature"],
        linestyle=":",
        linewidth=2.0,
        label=r"$T$  (temperature)",
    )

    ax_top.set_xscale("log")
    ax_top.set_xlabel(
        r"Temperature  $[\mathrm{K}]$",
        fontsize=12,
        color=_colors["temperature"],
    )
    ax_top.tick_params(axis="x", colors=_colors["temperature"])
    ax_top.spines["top"].set_edgecolor(_colors["temperature"])

    if temp_range is not None:
        ax_top.set_xlim(temp_range[0], temp_range[1])
    else:
        ax_top.set_xlim(T_clipped.min() * 0.5, T_clipped.max() * 2)

    # --- unified legend --------------------------------------------------
    lines_bot, labels_bot = ax_bot.get_legend_handles_labels()
    lines_top, labels_top = ax_top.get_legend_handles_labels()
    ax_bot.legend(
        lines_bot + lines_top,
        labels_bot + labels_top,
        loc="upper left",
        framealpha=0.85,
        fontsize=10,
    )

    # --- grid ------------------------------------------------------------
    ax_bot.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.4)
    ax_bot.grid(True, which="minor", linestyle=":",  linewidth=0.3, alpha=0.25)

    # --- optional redshift annotation ------------------------------------
    if redshift is not None:
        ax_bot.text(
            0.95, 0.95,
            f"z = {redshift:.5f}",
            color="black",
            fontsize=9,
            ha="right",
            va="top",
            transform=ax_bot.transAxes,
        )

    # --- title and layout ------------------------------------------------
    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()

    # --- optional save ---------------------------------------------------
    if outfile:
        fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
        print(f"Saved → {outfile}")

    return fig, ax_bot, ax_top


# ---------------------------------------------------------------------------
# Example usage (kept here as a reference; not executed on import)
# ---------------------------------------------------------------------------

def _example():
    """Reproduce the original script's behaviour using the new function."""
    import yt_initialization
    import merlin_spectra as merlin

    filename   = yt_initialization.filename
    ramses_dir = yt_initialization.ramses_dir
    logSFC_path = yt_initialization.logSFC_path
    lines      = yt_initialization.lines
    wavelengths = yt_initialization.wavelengths
    ds         = yt_initialization.ds
    ad         = yt_initialization.ad
    lims       = yt_initialization.lims_fiducial_00319

    output_dir = "/Users/bnowicki/Research/Github/Merlin/drivers"

    viz = merlin.VisualizationManager(
        filename, ramses_dir, logSFC_path, lines, wavelengths, ds, ad,
        output_dir=output_dir, minimal_output=True, lims_dict=lims,
    )

    star_ctr = viz.star_center(ad)
    sp_tiny  = ds.sphere(star_ctr, (300, "pc"))

    fig, ax_bot, ax_top = plot_density_pdfs(
        sp_tiny,
        redshift=ds.current_redshift,
        #title="Number Density and Temperature PDFs (200 pc sphere region)",
        title=None,
        outfile="output_00389_ramses_pdfs_300pc_new.pdf",
        density_range=(1e-3, 10**4.5),   # cm^-3  — omit to auto-scale
        temp_range=(10**2.5, 10**7.5),         # K      — omit to auto-scale
    )
    #plt.show()


if __name__ == "__main__":
    _example()