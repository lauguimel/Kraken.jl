#!/usr/bin/env python3
"""Reproduce the Newtonian cavity comparison plot from the shipped CSVs.

Self-contained: reads the `cavity_centerline_Re{100,400,1000}.csv` files that
sit next to this script and regenerates `comparison.png` (Kraken vs OpenFOAM
icoFoam vs Ghia 1982, u(y) on x=0.5 and v(x) on y=0.5). No external data, no
cross-worktree paths — only numpy + matplotlib.

Usage:
    python plot.py            # writes comparison.png next to this script
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = [100, 400, 1000]
PROFILES = [("u_vert", "u along vertical centerline (x = 0.5)", "y"),
            ("v_horiz", "v along horizontal centerline (y = 0.5)", "x")]


def load(re):
    """Return {profile: {col: [floats]}} for one Reynolds number."""
    path = os.path.join(HERE, f"cavity_centerline_Re{re}.csv")
    rows = {}
    with open(path, newline="") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
        for r in csv.DictReader(lines):
            prof = r["profile"]
            d = rows.setdefault(prof, {"coord": [], "kraken": [], "icofoam": [], "ghia": []})
            for k in ("coord", "kraken", "icofoam", "ghia"):
                v = r.get(k, "")
                d[k].append(float(v) if v not in ("", "nan", "NaN") else float("nan"))
    return rows


def main():
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for col, re in enumerate(RES):
        data = load(re)
        for row, (prof, title, axis) in enumerate(PROFILES):
            ax = axes[row][col]
            d = data.get(prof)
            if d is None:
                ax.set_visible(False)
                continue
            ax.plot(d["kraken"], d["coord"], "-", color="#0072B2", lw=2, label="Kraken (LBM)")
            ax.plot(d["icofoam"], d["coord"], "--", color="#D55E00", lw=1.6, label="icoFoam (FVM)")
            ax.plot(d["ghia"], d["coord"], "o", mfc="none", mec="k", ms=5, label="Ghia 1982")
            if row == 0:
                ax.set_title(f"Re = {re}")
            ax.set_xlabel("velocity (normalised)")
            ax.set_ylabel(axis)
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(loc="best", fontsize=9)
    fig.suptitle("Lid-driven cavity — Kraken vs icoFoam vs Ghia (1982) centerlines", fontsize=13)
    out = os.path.join(HERE, "comparison.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
