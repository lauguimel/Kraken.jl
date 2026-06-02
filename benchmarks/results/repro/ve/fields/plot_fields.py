#!/usr/bin/env python3
r"""Polymer-stress field + streamlines for viscoelastic flow past a cylinder.

Self-contained: reads ``field_ux.csv`` / ``field_uy.csv`` and the polymeric
stress components ``field_tau_p_xx.csv`` / ``field_tau_p_yy.csv`` (Kraken 2-D
fields on their native regular lattice, one row per x-index) next to this
script and renders a publication figure -- a polymer-stress scalar background
with black streamlines of the in-plane flow overlaid, revealing the
viscoelastic wake (the elongated stress trail of stretched polymer downstream
of the cylinder).

Left panel:  the trace of the polymeric stress, tr(tau_p) = tau_xx + tau_yy
             (signed, RdBu_r).
Right panel: the normal stress tau_p,xx, which peaks in the downstream wake.

The Kraken lattice is already a regular grid, so no interpolation is needed:
the CSVs map straight onto ``streamplot`` once transposed to ``[y, x]`` order.

LaTeX is used when a system ``latex`` is present, otherwise matplotlib mathtext
with the Computer-Modern font set, so the figure renders identically anywhere.
Dependencies: numpy + matplotlib.

Usage:
    python plot_fields.py            # writes field_streamlines.png next to this script
"""
import os
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

HERE = os.path.dirname(os.path.abspath(__file__))
_USETEX = shutil.which("latex") is not None

# Domain geometry (lattice units) and cylinder placement.
NX, NY = 240.0, 80.0
CX, CY, R = 60.0, 40.0, 16.0

plt.rcParams.update({
    "text.usetex": _USETEX,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
})
if _USETEX:
    # \boldsymbol needs amsmath + bm in the LaTeX preamble.
    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{bm}"


def load(name):
    # CSV is Nx rows x Ny cols (f[i, j] at x_i, y_j). Transpose -> [y, x] for mpl.
    return np.loadtxt(os.path.join(HERE, name), delimiter=",").T


def main():
    ux = load("field_ux.csv")          # shape [Ny, Nx]
    uy = load("field_uy.csv")
    txx = load("field_tau_p_xx.csv")
    tyy = load("field_tau_p_yy.csv")
    ny, nx = ux.shape
    x = np.arange(nx, dtype=float)
    y = np.arange(ny, dtype=float)
    X, Y = np.meshgrid(x, y)

    # Mask the solid cylinder interior.
    inside = (X - CX) ** 2 + (Y - CY) ** 2 <= R ** 2
    uxm = np.ma.array(ux, mask=inside)
    uym = np.ma.array(uy, mask=inside)
    trace = np.ma.array(txx + tyy, mask=inside)
    txxm = np.ma.array(txx, mask=inside)

    panels = [
        (trace, r"$\mathrm{tr}\,\boldsymbol{\tau}_p = \tau_{xx}+\tau_{yy}$",
         "RdBu_r", True),
        (txxm, r"$\tau_{p,xx}$  (normal stress)", "RdBu_r", True),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), constrained_layout=True)

    for ax, (field, label, cmap, signed) in zip(axes, panels):
        vmax = np.nanpercentile(np.abs(field.compressed()), 99)
        vmax = vmax or 1.0
        vmin = -vmax
        pcm = ax.pcolormesh(X, Y, field, cmap=cmap, shading="auto",
                            vmin=vmin, vmax=vmax)
        ax.streamplot(x, y, uxm.filled(0.0), uym.filled(0.0), color="k",
                      linewidth=0.8, density=1.5, arrowsize=0.8)
        ax.add_patch(Circle((CX, CY), R, facecolor="0.75", edgecolor="k",
                            linewidth=0.8, zorder=5))
        ax.set(xlabel=r"$x$ (lattice units)", ylabel=r"$y$",
               title=label, aspect="equal")
        ax.set_xlim(0, nx - 1)
        ax.set_ylim(0, ny - 1)
        cbar = fig.colorbar(pcm, ax=ax, fraction=0.022, pad=0.02)
        cbar.ax.tick_params(labelsize=9)

    fig.suptitle("Viscoelastic flow past a cylinder ($Wi=0.5$, Oldroyd-B) --- "
                 + ("polymer stress \\& streamlines (Kraken)" if _USETEX
                    else "polymer stress & streamlines (Kraken)"),
                 fontsize=14, fontweight="bold")
    out = os.path.join(HERE, "field_streamlines.png")
    fig.savefig(out, dpi=200)
    print(f"wrote {out} (usetex={_USETEX})")


if __name__ == "__main__":
    main()
