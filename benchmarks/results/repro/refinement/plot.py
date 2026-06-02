#!/usr/bin/env python3
r"""Reproduce the AMR refinement mass-conservation figure from the shipped CSVs.

Self-contained: reads the two ``amr_obstacle_convergence_2d_aqua_conv_*.csv``
files next to this script and regenerates ``comparison.png``.

The validated v0.1.0 claim for patch-based refinement is **stability /
conservation**: the relative mass drift stays at machine precision across grid
scales and across both the leaf-oracle and the route-native AMR paths. The plot
shows the relative mass drift $|\Delta m|/m$ versus the number of lattice cells
($N_x \cdot N_y$), grouped by flow (square / cylinder) and AMR method, on a log
ordinate, against the Float64 machine-epsilon band.

Colour encodes the (ordered) cell count via seaborn ``crest``; marker shape
encodes the AMR method. LaTeX is used for all text when a system ``latex`` is
available, otherwise the matplotlib mathtext engine with the Computer-Modern
font set. Dependencies: csv + matplotlib + seaborn.

Usage:
    python plot.py            # writes comparison.png next to this script
"""
import csv
import glob
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
_USETEX = shutil.which("latex") is not None

# AMR method -> marker. Two paths validated for conservation.
METHOD_MARKER = {"leaf_oracle": "o", "amr_route_native": "s"}
METHOD_LABEL = {"leaf_oracle": "leaf-oracle (reference)",
                "amr_route_native": "AMR route-native"}
EPS_F64 = 2.220446049250313e-16


def _load_all():
    """Merge every shipped conv CSV; dedupe (flow, method, cells) by mean drift."""
    rows = []
    for path in sorted(glob.glob(os.path.join(
            HERE, "amr_obstacle_convergence_2d_aqua_conv_*.csv"))):
        with open(path, newline="") as fh:
            lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
            rows.extend(csv.DictReader(lines))
    agg = {}
    for r in rows:
        cells = int(r["Nx"]) * int(r["Ny"])
        key = (r["flow"], r["method"], cells)
        agg.setdefault(key, []).append(abs(float(r["mass_rel_drift"])))
    out = []
    for (flow, method, cells), drifts in agg.items():
        out.append((flow, method, cells, sum(drifts) / len(drifts)))
    return out


def main():
    sns.set_theme(style="whitegrid", context="talk", font="serif")
    plt.rcParams.update({"text.usetex": _USETEX, "font.family": "serif",
                         "mathtext.fontset": "cm"})

    data = _load_all()
    all_cells = sorted({c for _, _, c, _ in data})
    palette = sns.color_palette("crest", len(all_cells))
    cell_color = dict(zip(all_cells, palette))

    flows = sorted({f for f, _, _, _ in data})
    fig, axes = plt.subplots(1, len(flows), figsize=(14, 6.2),
                             constrained_layout=True, sharey=True)
    if len(flows) == 1:
        axes = [axes]

    for ax, flow in zip(axes, flows):
        # machine-precision reference band.
        ax.axhspan(0, 10 * EPS_F64, color="0.85", zorder=0)
        ax.axhline(EPS_F64, color="k", ls="--", lw=1.3, zorder=1)
        for method, marker in METHOD_MARKER.items():
            pts = sorted((c, d) for f, m, c, d in data
                         if f == flow and m == method)
            if not pts:
                continue
            xs = [c for c, _ in pts]
            ys = [max(d, EPS_F64 * 0.3) for _, d in pts]
            ax.plot(xs, ys, "-", color="0.5", lw=1.4, zorder=2)
            for c, d in pts:
                ax.plot(c, max(d, EPS_F64 * 0.3), marker, color=cell_color[c],
                        ms=13, mec="0.2", mew=1.0, zorder=4)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel(r"lattice cells $N_x\!\cdot\!N_y$")
        ax.set_title(fr"{flow}")
    axes[0].set_ylabel(r"relative mass drift $|\Delta m|/m$")

    method_handles = [
        Line2D([0], [0], color="0.3", marker=METHOD_MARKER[m], ms=11,
               mec="0.2", ls="-", label=METHOD_LABEL[m])
        for m in METHOD_MARKER
    ]
    eps_handle = [Line2D([0], [0], color="k", ls="--", lw=1.3,
                         label=r"Float64 $\varepsilon_{\mathrm{mach}}$")]
    axes[-1].legend(handles=method_handles + eps_handle, loc="upper right",
                    fontsize=10, framealpha=0.9)

    fig.suptitle("Patch-based AMR --- mass conservation at machine precision",
                 fontsize=15, fontweight="bold")
    out = os.path.join(HERE, "comparison.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out} (usetex={_USETEX})")


if __name__ == "__main__":
    main()
