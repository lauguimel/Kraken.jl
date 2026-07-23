#!/usr/bin/env python3
r"""Temperature field + streamlines for the differentially heated square cavity.

Self-contained: reads ``field_T.csv`` / ``field_ux.csv`` / ``field_uy.csv``
(the Kraken 2-D temperature and velocity fields on their native regular
lattice, one row per x-index) next to this script and renders a publication
figure -- the temperature scalar background with black streamlines of the
in-plane flow overlaid, revealing the buoyancy-driven convection roll.

The hot (west) wall is on the left, the cold (east) wall on the right; the
fluid rises along the hot wall, crosses under the lid and sinks along the cold
wall. The Kraken lattice is already a regular grid, so no interpolation is
needed: the CSVs map straight onto ``streamplot`` once transposed to ``[y, x]``.

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

HERE = os.path.dirname(os.path.abspath(__file__))
_USETEX = shutil.which("latex") is not None
DARK = "#1f2424"  # Documenter dark theme background

plt.rcParams.update({
    "text.usetex": _USETEX,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "figure.facecolor": DARK, "axes.facecolor": DARK, "savefig.facecolor": DARK,
    "text.color": "0.92", "axes.labelcolor": "0.92", "axes.titlecolor": "0.96",
    "axes.edgecolor": "0.55", "xtick.color": "0.85", "ytick.color": "0.85",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
})


def load(name):
    # CSV is Nx rows x Ny cols (f[i, j] at x_i, y_j). Transpose -> [y, x] for mpl.
    return np.loadtxt(os.path.join(HERE, name), delimiter=",").T


def main():
    T = load("field_T.csv")            # shape [Ny, Nx]
    ux = load("field_ux.csv")
    uy = load("field_uy.csv")
    ny, nx = T.shape
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y)

    speed = np.sqrt(ux ** 2 + uy ** 2)
    umax = speed.max() or 1.0
    speed_n = speed / umax

    panels = [
        (T, r"$T$  (temperature)", "inferno", 0.0, 1.0),
        (speed_n, r"$|U|/U_{\max}$", "magma", 0.0, None),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)

    for ax, (field, label, cmap, vmin, vmax) in zip(axes, panels):
        if vmax is None:
            vmax = np.nanpercentile(field, 99.5)
        pcm = ax.pcolormesh(X, Y, field, cmap=cmap, shading="auto",
                            vmin=vmin, vmax=vmax)
        ax.streamplot(x, y, ux, uy, color="white", linewidth=0.8,
                      density=1.5, arrowsize=0.8)
        ax.set(xlabel=r"$x/H$", ylabel=r"$y/H$", title=label, aspect="equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9, color="0.7")
        cbar.outline.set_edgecolor("0.55")

    fig.suptitle("Differentially heated cavity ($Ra=10^4$) --- temperature "
                 + ("\\& streamlines (Kraken)" if _USETEX
                    else "& streamlines (Kraken)"),
                 fontsize=15, fontweight="bold")
    out = os.path.join(HERE, "field_streamlines.png")
    fig.savefig(out, dpi=200)
    print(f"wrote {out} (usetex={_USETEX})")


if __name__ == "__main__":
    main()
