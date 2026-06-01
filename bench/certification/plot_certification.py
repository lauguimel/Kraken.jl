#!/usr/bin/env python3
"""Generate the E1 GPU efficiency certification bar chart.

Kraken best sustained F64 D2Q9 BGK throughput (3461 MLUPS, A100, N=2048)
plotted against memory-bandwidth rooflines and the Palabos single-GPU F64
reference. A dashed line marks the 0.5x-roofline pass gate against the most
conservative A100 ceiling.

Run:
    python3 bench/certification/plot_certification.py
or:
    conda run -n kraken-v0-3-figures python bench/certification/plot_certification.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "docs", "src", "users", "benchmarks", "gpu-certification.png",
)

# Measured + reference numbers (MLUPS). Do not edit without new measurements.
KRAKEN_BEST = 3461.0  # A100, CUDA F64, N=2048, 1000 steps

labels = [
    "Kraken\n(A100 F64)",
    "Palabos\n(A100 F64)",
    "A100-40GB\nroofline",
    "A100-80GB\nroofline",
    "H100\nroofline",
]
values = [KRAKEN_BEST, 4656.0, 5115.0, 6707.0, 11020.0]
colors = ["#2b6cb0", "#718096", "#a0aec0", "#a0aec0", "#cbd5e0"]

# 0.5x gate against the confirmed run GPU (A100-40GB, gpu0n009).
GATE = 0.5 * 5115.0  # = 2557.5 MLUPS

fig, ax = plt.subplots(figsize=(7.0, 4.4))

bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6, zorder=3)

# Highlight the Kraken bar.
bars[0].set_color("#2b6cb0")
bars[0].set_edgecolor("black")

for rect, val in zip(bars, values):
    ax.text(
        rect.get_x() + rect.get_width() / 2.0,
        val + 150,
        f"{val:,.0f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

# 0.5x-roofline pass gate (vs A100-80GB conservative ceiling).
ax.axhline(GATE, color="#c53030", linestyle="--", linewidth=1.3, zorder=2)
ax.text(
    len(labels) - 0.5,
    GATE + 120,
    f"0.5x gate (A100-40GB) = {GATE:,.0f}",
    ha="right",
    va="bottom",
    fontsize=8.5,
    color="#c53030",
)

ax.set_ylabel("Sustained throughput (MLUPS, F64)")
ax.set_title("Single-GPU certification — BGK D2Q9 CUDA Float64")
ax.set_ylim(0, 12000)
ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}")
