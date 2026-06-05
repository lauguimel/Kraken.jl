#!/usr/bin/env python3
r"""Reproduce the 3D STL sphere-drag convergence plot from the shipped CSVs.

Self-contained: reads ``sphere_drag_conv.csv`` and
``sphere_drag_conv_lowblock.csv`` next to this script and regenerates
``comparison.png``.

A single scatter-plus-fit panel: drag coefficient $C_d$ against the blockage
ratio $D/W$ (per cent). The R = 16 sweep is the convergence series; a
least-squares quadratic $C_d = c_0 + c_1\beta + c_2\beta^2$ extrapolates to the
free-stream limit $\beta \to 0$, compared against the Clift, Grace & Weber
(1978) standard-drag-curve value at Re = 20 ($C_d = 2.61$). The R = 8 point is
shown as a coarser resolution probe.

Colour encodes the (ordered) blockage ratio of the R = 16 sweep via seaborn
``crest``. LaTeX is used for all text when a system ``latex`` is available,
otherwise the matplotlib mathtext engine with the Computer-Modern font set.
Dependencies: csv + numpy + matplotlib + seaborn.

Usage:
    python plot.py            # writes comparison.png next to this script
"""
import csv
import os
import shutil

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
_USETEX = shutil.which("latex") is not None
DARK = "#1b1b1f"  # Vitepress dark page background


def _load(name):
    path = os.path.join(HERE, name)
    with open(path, newline="") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
        return list(csv.DictReader(lines))


def main():
    sns.set_theme(style="ticks", context="talk", font="serif")
    plt.rcParams.update({
        "text.usetex": _USETEX, "font.family": "serif", "mathtext.fontset": "cm",
        "figure.facecolor": DARK, "axes.facecolor": DARK, "savefig.facecolor": DARK,
        "text.color": "0.92", "axes.labelcolor": "0.92", "axes.titlecolor": "0.96",
        "axes.edgecolor": "0.55", "xtick.color": "0.85", "ytick.color": "0.85",
    })

    rows = _load("sphere_drag_conv.csv") + _load("sphere_drag_conv_lowblock.csv")

    # R = 16 convergence series, ordered by blockage.
    r16 = sorted((float(r["blockage_pct"]), float(r["Cd"]))
                 for r in rows if int(r["R_LU"]) == 16)
    beta = np.array([b / 100.0 for b, _ in r16])
    cd = np.array([c for _, c in r16])
    # R = 8 resolution probe.
    r8 = sorted((float(r["blockage_pct"]), float(r["Cd"]))
                for r in rows if int(r["R_LU"]) == 8)

    # Quadratic LSQ fit; c0 is the beta -> 0 free-stream limit, plus R^2.
    A = np.vstack([np.ones_like(beta), beta, beta ** 2]).T
    coef, *_ = np.linalg.lstsq(A, cd, rcond=None)
    c0, c1, c2 = coef
    pred = A @ coef
    ss_res = float(np.sum((cd - pred) ** 2))
    ss_tot = float(np.sum((cd - cd.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    bb = np.linspace(0, beta.max() * 1.05, 200)
    fit = c0 + c1 * bb + c2 * bb ** 2

    # Clift, Grace & Weber (1978) standard drag curve at Re = 20.
    clift = 1.2 * (1 + 0.15 * 20 ** 0.687)

    palette = sns.color_palette("bright", len(beta))  # vivid; pops on dark

    fig, ax = plt.subplots(figsize=(8.4, 6.2), constrained_layout=True)
    ax.grid(True, color="0.45", alpha=0.4, lw=0.6)

    ax.plot(bb * 100, fit, "-", color="0.8", lw=2.2, zorder=1)
    for color, (b, c) in zip(palette, zip(beta, cd)):
        ax.plot(b * 100, c, "o", color=color, ms=12, mec="0.92", mew=1.0, zorder=4)
    if r8:
        ax.plot([b for b, _ in r8], [c for _, c in r8], "s", color="0.7",
                ms=11, mfc="none", mew=2.0, zorder=3)
    ax.axhline(clift, color="0.9", ls="--", lw=1.5, zorder=2)
    ax.plot(0, c0, "D", color="#ff6b6b", ms=13, mec="0.92", mew=1.0, zorder=5)

    ax.set_xlabel(r"blockage ratio $D/W$  [\%]" if _USETEX
                  else r"blockage ratio $D/W$  [%]")
    ax.set_ylabel(r"drag coefficient $C_d = 2F_x/(\rho\,u^2 A)$")
    ax.set_title(r"3D STL sphere drag --- extrapolation to free stream "
                 r"($\mathrm{Re}=20$)")
    ax.set_xlim(-1.2, max(21, beta.max() * 100 * 1.05))

    handles = [
        Line2D([0], [0], color="0.9", lw=2.2, marker="o", ms=11, mec="0.92",
               label=fr"Kraken STL, $R=16$ (CUDA F64), fit $R^2={r2:.4f}$"),
        Line2D([0], [0], color="0.7", ls="none", marker="s", mfc="none",
               ms=10, mew=2.0, label=r"Kraken STL, $R=8$ (resolution probe)"),
        Line2D([0], [0], color="#ff6b6b", ls="none", marker="D", ms=11,
               mec="0.92", label=fr"extrapolated $\beta\to 0$: $C_d={c0:.2f}$"),
        Line2D([0], [0], color="0.9", ls="--", lw=1.5,
               label=fr"Clift (1978) free stream: $C_d={clift:.2f}$"),
    ]
    leg = ax.legend(handles=handles, loc="upper left", fontsize=11,
                    facecolor=DARK, edgecolor="0.5", labelcolor="0.9",
                    framealpha=0.85)

    out = os.path.join(HERE, "comparison.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out} (usetex={_USETEX})")
    print(f"fit: Cd_inf={c0:.3f}, Clift={clift:.3f}, "
          f"rel={100 * (c0 - clift) / clift:+.1f}%, R2={r2:.4f}")


if __name__ == "__main__":
    main()
