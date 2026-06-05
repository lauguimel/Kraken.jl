#!/usr/bin/env python3
r"""Reproduce the viscoelastic-cylinder Cd-vs-Wi comparison plot from the shipped CSVs.

Self-contained: reads ``kraken_cd_vs_wi.csv`` and ``rheotool_cd_vs_wi.csv`` next to
this script and regenerates ``comparison.png`` (Cd vs Weissenberg number): Kraken as
filled markers, RheoTool (rheoFoam) as a dashed reference line through its three points.

Dark Documenter theme (``DARK = #1b1b1f``), matching the other rheotool_compare
figures. LaTeX is used for all text when a system ``latex`` is available, otherwise the
matplotlib mathtext engine with the Computer-Modern font set. Dependencies: csv +
matplotlib + seaborn.

Usage:
    python plot.py            # writes comparison.png next to this script

Run under the documentation figure environment:
    conda run -n kraken-v0-3-figures python plot.py
"""
import csv
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
_USETEX = shutil.which("latex") is not None
DARK = "#1b1b1f"  # Vitepress dark page background


def load(name, *cols):
    """Return {col: [floats]} from a CSV in this directory (skip # comment lines)."""
    path = os.path.join(HERE, name)
    out = {c: [] for c in cols}
    with open(path, newline="") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
        for r in csv.DictReader(lines):
            for c in cols:
                out[c].append(float(r[c]))
    return out


def main():
    sns.set_theme(style="ticks", context="talk", font="serif")
    plt.rcParams.update({
        "text.usetex": _USETEX, "font.family": "serif", "mathtext.fontset": "cm",
        "figure.facecolor": DARK, "axes.facecolor": DARK, "savefig.facecolor": DARK,
        "text.color": "0.92", "axes.labelcolor": "0.92", "axes.titlecolor": "0.96",
        "axes.edgecolor": "0.55", "xtick.color": "0.85", "ytick.color": "0.85",
    })
    palette = sns.color_palette("bright", 2)  # vivid; pops on dark
    c_kraken, c_rheo = palette[0], palette[3] if len(palette) > 3 else palette[1]

    krk = load("kraken_cd_vs_wi.csv", "Wi", "Cd")
    rt = load("rheotool_cd_vs_wi.csv", "Wi", "Cd")

    fig, ax = plt.subplots(figsize=(8.4, 6.0), constrained_layout=True)
    ax.grid(True, color="0.45", alpha=0.4, lw=0.6)

    # RheoTool reference line (dashed) through its three points.
    ax.plot(rt["Wi"], rt["Cd"], "--", color=c_rheo, lw=2.2, zorder=2)
    ax.plot(rt["Wi"], rt["Cd"], ls="none", marker="s", mfc="none", mec=c_rheo,
            ms=10, mew=1.8, zorder=3)
    # Kraken filled markers + connecting line.
    ax.plot(krk["Wi"], krk["Cd"], "-", color=c_kraken, lw=2.0, alpha=0.8, zorder=4)
    ax.plot(krk["Wi"], krk["Cd"], ls="none", marker="o", mfc=c_kraken, mec="0.95",
            ms=11, mew=1.2, zorder=5)

    # Annotate the per-point relative error (<1% gate).
    for wi, ck, cr in zip(krk["Wi"], krk["Cd"], rt["Cd"]):
        rel = 100.0 * (ck - cr) / cr
        ax.annotate(fr"${rel:+.2f}\,\%$", (wi, ck), textcoords="offset points",
                    xytext=(8, -16), fontsize=11, color="0.85")

    ax.set(xlabel=r"Weissenberg number  $\mathrm{Wi} = \lambda\,u_{\mathrm{mean}}/R$",
           ylabel=r"drag coefficient  $C_d$",
           title=r"Confined cylinder (Oldroyd-B), $R=50$, $\beta=0.59$")
    ax.set_xlim(0.0, 1.1)

    handles = [
        Line2D([0], [0], color=c_kraken, lw=2.0, marker="o", mfc=c_kraken,
               mec="0.95", ms=10, label="Kraken (LBM + log-FV, CUDA F64)"),
        Line2D([0], [0], color=c_rheo, lw=2.2, ls="--", marker="s", mfc="none",
               mec=c_rheo, ms=9, mew=1.8, label="RheoTool (rheoFoam, log-conf)"),
    ]
    leg = ax.legend(handles=handles, loc="upper right", fontsize=12,
                    facecolor=DARK, edgecolor="0.5", labelcolor="0.9",
                    framealpha=0.85)
    leg.get_frame().set_linewidth(0.8)

    fig.suptitle(r"Drag vs Weissenberg --- Kraken vs RheoTool ($<1\,\%$ on $C_d$)",
                 fontsize=15, fontweight="bold")
    out = os.path.join(HERE, "comparison.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out} (usetex={_USETEX})")


if __name__ == "__main__":
    main()
