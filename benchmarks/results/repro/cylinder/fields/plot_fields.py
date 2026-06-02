#!/usr/bin/env python3
r"""Velocity-magnitude & vorticity fields + streamlines for flow past a cylinder.

Self-contained: reads ``field_ux.csv`` / ``field_uy.csv`` (the Kraken 2-D
velocity field on its native regular lattice, one row per x-index) next to this
script and renders a publication figure -- a scalar background (the velocity
magnitude and the out-of-plane vorticity) with black streamlines of the
in-plane flow overlaid. The recirculating wake behind the cylinder is visible.

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

# Domain geometry (lattice units L x H) and cylinder placement.
LX, LY = 10.0, 2.5
CX, CY, R = 2.5, 1.25, 0.5

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
    x = np.linspace(0.0, LX, nx)
    y = np.linspace(0.0, LY, ny)
    X, Y = np.meshgrid(x, y)

    # Mask the solid cylinder interior so streamlines do not cross it.
    inside = (X - CX) ** 2 + (Y - CY) ** 2 <= R ** 2
    uxm = np.ma.array(ux, mask=inside)
    uym = np.ma.array(uy, mask=inside)

    speed = np.sqrt(ux ** 2 + uy ** 2)
    umax = speed.max() or 1.0
    speed_n = np.ma.array(speed / umax, mask=inside)
    # out-of-plane vorticity omega_z = d(uy)/dx - d(ux)/dy
    duy_dx = np.gradient(uy, x, axis=1)
    dux_dy = np.gradient(ux, y, axis=0)
    omega = np.ma.array(duy_dx - dux_dy, mask=inside)

    panels = [
        (speed_n, r"$|U|/U_{\mathrm{in}}$", "viridis", False),
        (omega, r"$\omega_z$  (vorticity)", "RdBu_r", True),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.4), constrained_layout=True)

    for ax, (field, label, cmap, signed) in zip(axes, panels):
        if signed:
            vmax = np.nanpercentile(np.abs(field.compressed()), 99)
            vmin = -vmax
        else:
            vmin, vmax = 0.0, np.nanpercentile(field.compressed(), 99.5)
        pcm = ax.pcolormesh(X, Y, field, cmap=cmap, shading="auto",
                            vmin=vmin, vmax=vmax)
        ax.streamplot(x, y, uxm.filled(0.0), uym.filled(0.0), color="k",
                      linewidth=0.8, density=1.5, arrowsize=0.8)
        ax.add_patch(Circle((CX, CY), R, facecolor="0.75", edgecolor="k",
                            linewidth=0.8, zorder=5))
        ax.set(xlabel=r"$x$", ylabel=r"$y$", title=label, aspect="equal")
        ax.set_xlim(0, LX)
        ax.set_ylim(0, LY)
        cbar = fig.colorbar(pcm, ax=ax, fraction=0.018, pad=0.02)
        cbar.ax.tick_params(labelsize=9)

    fig.suptitle("Flow past a cylinder ($Re=20$) --- velocity field "
                 + ("\\& streamlines (Kraken)" if _USETEX
                    else "& streamlines (Kraken)"),
                 fontsize=15, fontweight="bold")
    out = os.path.join(HERE, "field_streamlines.png")
    fig.savefig(out, dpi=200)
    print(f"wrote {out} (usetex={_USETEX})")


if __name__ == "__main__":
    main()
