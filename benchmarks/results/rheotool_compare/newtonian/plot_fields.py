#!/usr/bin/env python3
r"""Velocity-magnitude field + streamlines for the lid-driven cavity.

Self-contained: reads ``field_ux.csv`` / ``field_uy.csv`` (the Kraken 2-D
velocity field on its native regular lattice, one row per x-index) next to this
script and renders a publication figure -- a scalar background (|U| and the
out-of-plane vorticity) with black streamlines of the in-plane flow overlaid.

Unlike an unstructured OpenFOAM field, the Kraken lattice is already a regular
grid, so no interpolation is needed: the CSVs map straight onto ``streamplot``.

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

plt.rcParams.update({
    "text.usetex": _USETEX,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
})


def load(name):
    # CSV is Nx rows x Ny cols (ux[i, j] at x_i, y_j). Transpose -> [y, x] for mpl.
    return np.loadtxt(os.path.join(HERE, name), delimiter=",").T


def main():
    ux = load("field_ux.csv")          # shape [Ny, Nx]
    uy = load("field_uy.csv")
    ny, nx = ux.shape
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y)

    speed = np.sqrt(ux ** 2 + uy ** 2)
    umax = speed.max() or 1.0
    speed_n = speed / umax
    # out-of-plane vorticity omega_z = d(uy)/dx - d(ux)/dy
    duy_dx = np.gradient(uy, x, axis=1)
    dux_dy = np.gradient(ux, y, axis=0)
    omega = duy_dx - dux_dy

    panels = [
        (speed_n, r"$|U|/U_{\mathrm{lid}}$", "viridis", False),
        (omega, r"$\omega_z$  (vorticity)", "RdBu_r", True),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)

    for ax, (field, label, cmap, signed) in zip(axes, panels):
        if signed:
            vmax = np.nanpercentile(np.abs(field), 99)
            vmin = -vmax
        else:
            vmin, vmax = 0.0, np.nanpercentile(field, 99.5)
        pcm = ax.pcolormesh(X, Y, field, cmap=cmap, shading="auto",
                            vmin=vmin, vmax=vmax)
        ax.streamplot(x, y, ux, uy, color="k", linewidth=0.8,
                      density=1.5, arrowsize=0.8)
        ax.set(xlabel=r"$x/L$", ylabel=r"$y/L$", title=label, aspect="equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        cbar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9)

    fig.suptitle("Lid-driven cavity --- velocity field \\& streamlines (Kraken)"
                 if _USETEX else
                 "Lid-driven cavity --- velocity field & streamlines (Kraken)",
                 fontsize=15, fontweight="bold")
    out = os.path.join(HERE, "field_streamlines.png")
    fig.savefig(out, dpi=200)
    print(f"wrote {out} (usetex={_USETEX})")


if __name__ == "__main__":
    main()
