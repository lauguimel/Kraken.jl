#!/usr/bin/env python3
r"""Reproduce the Newtonian cavity comparison plot from the shipped CSVs.

Self-contained: reads ``cavity_centerline_Re{100,400,1000}.csv`` next to this
script and regenerates ``comparison.png``. Classic centerline presentation --
the spatial coordinate on the ordinate, the velocity on the abscissa:

    left  -- u along the vertical centerline  (x = 0.5):  y vs u
    right -- v along the horizontal centerline (y = 0.5):  x vs v

Colour encodes the Reynolds number (seaborn ``crest``); Kraken is a solid line,
Ghia (1982) open circles. Both legends (Reynolds + Solver) appear on each panel.

LaTeX is used for all text when a system ``latex`` is available, otherwise the
matplotlib mathtext engine with the Computer-Modern font set, so the figure
renders identically anywhere. Dependencies: csv + matplotlib + seaborn.

Usage:
    python plot.py            # writes comparison.png next to this script
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
RES = [100, 400, 1000]
_USETEX = shutil.which("latex") is not None


def load(re):
    """Return {profile: {col: [floats]}} for one Reynolds number."""
    path = os.path.join(HERE, f"cavity_centerline_Re{re}.csv")
    rows = {}
    with open(path, newline="") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
        for r in csv.DictReader(lines):
            prof = r["profile"]
            d = rows.setdefault(prof, {"coord": [], "kraken": [], "ghia": []})
            for k in ("coord", "kraken", "ghia"):
                v = r.get(k, "")
                d[k].append(float(v) if v not in ("", "nan", "NaN") else float("nan"))
    return rows


def _draw(ax, series, color):
    """Kraken solid line + Ghia open circles, velocity on the abscissa."""
    ax.plot(series["kraken"], series["coord"], "-", color=color, lw=2.4, zorder=2)
    ax.plot(series["ghia"], series["coord"], ls="none", marker="o", mfc="none",
            mec=color, ms=8, mew=1.6, zorder=3)


def main():
    sns.set_theme(style="whitegrid", context="talk", font="serif")
    plt.rcParams.update({"text.usetex": _USETEX, "font.family": "serif",
                         "mathtext.fontset": "cm"})
    palette = sns.color_palette("crest", len(RES))
    fig, (axu, axv) = plt.subplots(1, 2, figsize=(14, 6.2), constrained_layout=True)

    for color, re in zip(palette, RES):
        d = load(re)
        _draw(axu, d["u_vert"], color)
        _draw(axv, d["v_horiz"], color)

    axu.set(xlabel=r"$|U|/U_{\mathrm{lid}}$", ylabel=r"$y/L$",
            title=r"$u$ along vertical centerline $u_{x=0.5}$")
    axv.set(xlabel=r"$|U|/U_{\mathrm{lid}}$", ylabel=r"$x/L$",
            title=r"$v$ along horizontal centerline $v_{y=0.5}$")

    re_handles = [Line2D([0], [0], color=c, lw=3.2, label=fr"$Re = {re}$")
                  for c, re in zip(palette, RES)]
    solver_handles = [
        Line2D([0], [0], color="0.3", lw=2.4, ls="-", label="Kraken (LBM)"),
        Line2D([0], [0], color="0.3", ls="none", marker="o", mfc="none",
               mec="0.3", ms=8, mew=1.6, label="Ghia (1982)"),
    ]
    def _legends(ax, re_loc, solver_loc):
        leg_re = ax.legend(handles=re_handles, loc=re_loc,
                           fontsize=10, title="Reynolds", framealpha=0.9)
        ax.add_artist(leg_re)
        ax.legend(handles=solver_handles, loc=solver_loc,
                  fontsize=10, title="Solver", framealpha=0.9)

    # Right (v) panel: legends in the opposite corners to the left (u) panel.
    _legends(axu, re_loc="lower right", solver_loc="upper left")
    _legends(axv, re_loc="upper left", solver_loc="lower right")

    fig.suptitle("Lid-driven cavity --- Kraken (LBM) vs Ghia (1982)",
                 fontsize=15, fontweight="bold")
    out = os.path.join(HERE, "comparison.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out} (usetex={_USETEX})")


if __name__ == "__main__":
    main()
