"""
plot_density_pdfs.py
====================
Provides plot_density_pdfs(), a self-contained function that produces a
publication-quality PDF panel from a yt sphere object containing the derived
fields:
    ("gas", "electron_number_density")   [cm^-3]
    ("gas", "my_H_nuclei_density")       [cm^-3]
    ("gas", "my_temperature")            [K]

Optional insets show the temperature PDF for cells restricted to specified
density ranges, with annotated connector lines from the main-plot highlight
box to the inset panel.

Example usage
-------------
import yt
ds  = yt.load("output_00100/info_00100.txt")
sp  = ds.sphere(ds.domain_center, (500, "pc"))

fig = plot_density_pdfs(
    sphere        = sp,
    title         = "Halo core — 500 pc sphere",
    density_range = (1e-4, 1e4),          # cm^-3
    temp_range    = (10,   1e8),          # K
    inset_density_ranges = [
        (1e-1, 1e1,  "warm neutral"),
        (1e1,  1e3,  "dense gas"),
    ],
    n_bins        = 120,
    outfile       = "pdfs.pdf",
)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from typing import Sequence, Tuple, Optional
import yt
import yt_initialization
import merlin_spectra as merlin


# ── colour / style palette ────────────────────────────────────────────────────
PALETTE = {
    "electron":    ("#E05C3A", "-",  1.8),   # orange-red,  solid
    "hydrogen":    ("#3A7FE0", "--", 1.8),   # sky-blue,    dashed
    "temperature": ("#2EBF91", ":",  2.0),   # teal,        dotted
}

# A small set of distinct colours for the inset highlight boxes / curves
INSET_COLORS = ["#A855F7", "#F59E0B", "#EC4899", "#14B8A6", "#F97316"]


# ── internal helpers ──────────────────────────────────────────────────────────

def _get_field(region, field: tuple) -> np.ndarray:
    """Extract a plain numpy array from a yt region."""
    return np.asarray(region[field], dtype=float)


def _log_bins(arr: np.ndarray, n_bins: int, lo: float, hi: float) -> np.ndarray:
    """Log-spaced bin edges clipped to [lo, hi]."""
    arr_pos = arr[np.isfinite(arr) & (arr > 0)]
    lo_eff = max(lo, arr_pos.min()) if lo is not None else arr_pos.min()
    hi_eff = min(hi, arr_pos.max()) if hi is not None else arr_pos.max()
    return np.logspace(np.log10(lo_eff), np.log10(hi_eff), n_bins + 1)


def _norm_hist(arr: np.ndarray, bins: np.ndarray):
    """
    Return (centres, pdf) where pdf is normalised to a peak of 1.
    Empty bins are omitted so axes limits are never NaN/Inf.
    """
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size == 0:
        raise ValueError("No finite positive values — cannot build histogram.")
    log_bins = np.log10(bins)
    counts, edges = np.histogram(np.log10(arr), bins=log_bins)
    centres = 10 ** (0.5 * (edges[:-1] + edges[1:]))
    mask    = counts > 0
    centres, counts = centres[mask], counts[mask]
    return centres, counts / counts.max()


def _safe_xlim(centres_list: list[np.ndarray], pad: float = 0.5):
    """Compute xlim from multiple centre arrays, ignoring NaN/Inf."""
    all_c = np.concatenate([c[np.isfinite(c) & (c > 0)] for c in centres_list])
    if all_c.size == 0:
        raise ValueError("All bin centres are NaN/Inf — check input data.")
    return all_c.min() * pad, all_c.max() / pad


def _display_to_axes_fraction(ax, x_data, y_data):
    """Convert data coordinates → axes-fraction coordinates (for connector lines)."""
    disp   = ax.transData.transform((x_data, y_data))
    ax_inv = ax.transAxes.inverted()
    return ax_inv.transform(disp)


# ── main function ─────────────────────────────────────────────────────────────

def plot_density_pdfs(
    sphere,
    title:                str  = "Sphere region PDFs",
    density_range:        Tuple[float, float] = (None, None),
    temp_range:           Tuple[float, float] = (None, None),
    inset_density_ranges: Optional[Sequence[Tuple[float, float, str]]] = None,
    n_bins:               int  = 100,
    outfile:              Optional[str] = "ramses_pdfs.pdf",
    figsize_base:         Tuple[float, float] = (8, 5),
) -> plt.Figure:
    """
    Plot normalised PDFs of electron density, H nuclei density, and temperature
    for all cells inside *sphere*, with optional inset panels.

    Parameters
    ----------
    sphere : yt.data_objects.selection_data_containers.YTSphere
        A yt sphere data object providing the three derived fields.
    title : str
        Figure title.
    density_range : (float, float)
        (lo, hi) limits for the bottom x-axis [cm^-3].  Use None for auto.
    temp_range : (float, float)
        (lo, hi) limits for the top x-axis [K].  Use None for auto.
    inset_density_ranges : list of (lo_n, hi_n, label) or (lo_n, hi_n), optional
        Each entry defines one inset showing the temperature PDF of cells
        whose *electron* number density falls in [lo_n, hi_n] (cm^-3).
        An optional third element supplies a text label.
    n_bins : int
        Number of histogram bins for every field.
    outfile : str or None
        If given, save the figure to this path.  Pass None to skip saving.
    figsize_base : (width, height)
        Base figure size in inches before insets are added.

    Returns
    -------
    matplotlib.figure.Figure
    """

    # ── 0. Parse inset specs ─────────────────────────────────────────────────
    insets: list[tuple] = []
    if inset_density_ranges:
        for spec in inset_density_ranges:
            lo_i, hi_i = spec[0], spec[1]
            label_i    = spec[2] if len(spec) > 2 else f"[{lo_i:.1e}, {hi_i:.1e}]"
            insets.append((lo_i, hi_i, label_i))

    n_insets = len(insets)

    # ── 1. Load field data ───────────────────────────────────────────────────
    n_e = _get_field(sphere, ("gas", "electron_number_density"))
    n_H = _get_field(sphere, ("gas", "my_H_nuclei_density"))
    T   = _get_field(sphere, ("gas", "my_temperature"))

    # ── 2. Build bin edges (respecting user-supplied limits) ─────────────────
    d_lo, d_hi = density_range
    t_lo, t_hi = temp_range

    bins_e = _log_bins(n_e, n_bins, d_lo, d_hi)
    bins_H = _log_bins(n_H, n_bins, d_lo, d_hi)
    bins_T = _log_bins(T,   n_bins, t_lo, t_hi)

    c_e, pdf_e = _norm_hist(n_e, bins_e)
    c_H, pdf_H = _norm_hist(n_H, bins_H)
    c_T, pdf_T = _norm_hist(T,   bins_T)

    # ── 3. Figure layout ─────────────────────────────────────────────────────
    # Widen figure to accommodate insets on the right
    inset_width_in = 2.2
    fig_w = figsize_base[0] + n_insets * inset_width_in
    fig   = plt.figure(figsize=(fig_w, figsize_base[1]))

    # Main axes occupies the left portion
    main_right = figsize_base[0] / fig_w
    ax_bot = fig.add_axes([0.08, 0.12, main_right - 0.12, 0.72])

    # ── 4. Main plot: density PDFs (bottom x-axis) ───────────────────────────
    ec, ls, lw = PALETTE["electron"]
    ax_bot.step(c_e, pdf_e, where="mid", color=ec, linestyle=ls,
                linewidth=lw, label=r"$n_e$  (electron)")

    hc, ls, lw = PALETTE["hydrogen"]
    ax_bot.step(c_H, pdf_H, where="mid", color=hc, linestyle=ls,
                linewidth=lw, label=r"$n_\mathrm{H}$  (H nuclei)")

    ax_bot.set_xscale("log")
    ax_bot.set_xlabel(r"Number density  $[\mathrm{cm}^{-3}]$", fontsize=12)
    ax_bot.set_ylabel("Relative probability  (normalised to peak)", fontsize=11)
    ax_bot.set_ylim(0, 1.05)
    ax_bot.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax_bot.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

    x_lo, x_hi = _safe_xlim([c_e, c_H])
    if d_lo is not None: x_lo = d_lo
    if d_hi is not None: x_hi = d_hi
    ax_bot.set_xlim(x_lo, x_hi)

    ax_bot.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.4)
    ax_bot.grid(True, which="minor", linestyle=":",  linewidth=0.3, alpha=0.25)

    # ── 5. Main plot: temperature PDF (top x-axis) ───────────────────────────
    ax_top = ax_bot.twiny()
    tc, ls, lw = PALETTE["temperature"]
    ax_top.step(c_T, pdf_T, where="mid", color=tc, linestyle=ls,
                linewidth=lw, label=r"$T$  (temperature)")
    ax_top.set_xscale("log")
    ax_top.set_xlabel(r"Temperature  $[\mathrm{K}]$", fontsize=12, color=tc)
    ax_top.tick_params(axis="x", colors=tc)
    ax_top.spines["top"].set_edgecolor(tc)

    t_pos = T[T > 0]
    tx_lo = t_lo if t_lo is not None else t_pos.min()
    tx_hi = t_hi if t_hi is not None else t_pos.max()
    ax_top.set_xlim(tx_lo, tx_hi)

    # ── 6. Legend ────────────────────────────────────────────────────────────
    lines_b, lbl_b = ax_bot.get_legend_handles_labels()
    lines_t, lbl_t = ax_top.get_legend_handles_labels()
    ax_bot.legend(lines_b + lines_t, lbl_b + lbl_t,
                  loc="upper left", framealpha=0.85, fontsize=9)

    ax_bot.set_title(title, fontsize=13, pad=28)

    # ── 7. Insets ────────────────────────────────────────────────────────────
    for idx, (lo_i, hi_i, lbl_i) in enumerate(insets):
        color_i = INSET_COLORS[idx % len(INSET_COLORS)]

        # ── 7a. Mask cells in density range (uses n_e as the reference density)
        mask_i = (n_e >= lo_i) & (n_e <= hi_i) & np.isfinite(T) & (T > 0)
        T_i    = T[mask_i]

        # ── 7b. Highlight box on main axes ───────────────────────────────────
        # Box spans [lo_i, hi_i] horizontally and [0, 1.02] vertically
        box_rect = mpatches.Rectangle(
            (lo_i, 0), hi_i - lo_i, 1.02,
            linewidth=1.8, edgecolor=color_i,
            facecolor=color_i, alpha=0.08, zorder=3,
            transform=ax_bot.get_xaxis_transform(),   # x=data, y=axes [0,1]
            clip_on=True,
        )
        ax_bot.add_patch(box_rect)
        # Solid edge lines (more visible than the patch border alone)
        for xv in (lo_i, hi_i):
            ax_bot.axvline(xv, color=color_i, linewidth=1.4,
                           linestyle="--", alpha=0.7, zorder=4)

        # ── 7c. Inset axes position in figure-fraction coords ─────────────────
        gap        = 0.03                         # gap between panels
        inset_w    = (inset_width_in - 0.3) / fig_w
        inset_h    = 0.72                         # same height as main
        inset_left = main_right + gap + idx * (inset_w + gap)
        inset_bot  = 0.12

        ax_ins = fig.add_axes([inset_left, inset_bot, inset_w, inset_h])

        # ── 7d. Temperature PDF for cells in this density range ───────────────
        if T_i.size > 1:
            bins_Ti = _log_bins(T_i, n_bins, t_lo, t_hi)
            c_Ti, pdf_Ti = _norm_hist(T_i, bins_Ti)
            ax_ins.step(c_Ti, pdf_Ti, where="mid",
                        color=color_i, linewidth=1.8)
        else:
            ax_ins.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax_ins.transAxes, fontsize=8, color="grey")

        ax_ins.set_xscale("log")
        ax_ins.set_xlim(tx_lo, tx_hi)
        ax_ins.set_ylim(0, 1.05)
        ax_ins.set_xlabel(r"$T$  [K]", fontsize=9, color=tc)
        ax_ins.tick_params(axis="x", labelsize=7, colors=tc)
        ax_ins.tick_params(axis="y", labelsize=7)
        ax_ins.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax_ins.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
        ax_ins.spines["bottom"].set_edgecolor(tc)
        ax_ins.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.35)

        # Label
        ax_ins.set_title(lbl_i, fontsize=8, color=color_i, pad=4)
        ax_ins.text(0.97, 0.97,
                    rf"$n_e \in [{lo_i:.0e},\,{hi_i:.0e}]$" + r" cm$^{-3}$",
                    transform=ax_ins.transAxes, fontsize=6.5, color="0.3",
                    ha="right", va="top")

        # ── 7e. Connector lines: top-right & bottom-right of box → inset ─────
        # We work in figure coordinates for both ends.
        fig_trans  = fig.transFigure.inverted()

        # Right edge of the box in data coords → figure coords
        # Top-right corner of box (y=1.02 in axes fraction → data y is irrelevant;
        # we use the axes bounding box top/bottom in figure coords)
        ax_bbox   = ax_bot.get_position()          # figure fraction
        box_right_fig_x = (
            ax_bbox.x0
            + ax_bbox.width
            * (np.log10(hi_i) - np.log10(ax_bot.get_xlim()[0]))
            / (np.log10(ax_bot.get_xlim()[1]) - np.log10(ax_bot.get_xlim()[0]))
        )
        box_top_fig_y    = ax_bbox.y0 + ax_bbox.height   # top of axes
        box_bot_fig_y    = ax_bbox.y0                     # bottom of axes

        inset_left_fig_x = inset_left                     # already in fig frac
        inset_top_fig_y  = inset_bot + inset_h
        inset_bot_fig_y  = inset_bot

        for (x0f, y0f), (x1f, y1f) in [
            ((box_right_fig_x, box_top_fig_y), (inset_left_fig_x, inset_top_fig_y)),
            ((box_right_fig_x, box_bot_fig_y), (inset_left_fig_x, inset_bot_fig_y)),
        ]:
            line = plt.Line2D(
                [x0f, x1f], [y0f, y1f],
                transform=fig.transFigure,
                color=color_i, linewidth=1.2,
                linestyle="--", alpha=0.75,
                zorder=10,
            )
            fig.add_artist(line)

    # ── 8. Save / return ─────────────────────────────────────────────────────
    if outfile:
        fig.savefig(outfile, dpi=150, bbox_inches="tight")
        print(f"Saved → {outfile}")

    return fig


# ── Example usage (edit and run directly) ────────────────────────────────────
if __name__ == "__main__":
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
    sp_tiny = ds.sphere(star_ctr, (200, "pc"))
    width = (1500, 'pc')

    print(star_ctr)

    #ds     = yt.load("path/to/output_XXXXX/info_XXXXX.txt")
    #centre = ds.domain_center
    #radius = ds.quan(500, "pc")
    #sphere = ds.sphere(centre, radius)

    fig = plot_density_pdfs(
        sphere        = sp_tiny,
        title         = "Number Density and Temperature PDFs (200 pc sphere region)",
        density_range = (1e-4, 1e4),       # cm^-3
        temp_range    = (10,   1e8),        # K
        inset_density_ranges = [
        #   (1e-1, 1e1,  "warm neutral"),
        #    (1e1, 1e2, "intermediate"),
        #    (1e2,  1e4,  "dense gas"),
        ],
        n_bins        = 120,
        outfile       = "ramses_pdfs.pdf",
    )
    plt.show()