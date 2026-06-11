#!/usr/bin/env python3
"""Dark GPU-benchmark figures for the incompressible steady Navier-Stokes page.

All numbers are transcribed VERBATIM from the measured A100 results under
benchmarks/results/ (no new runs, no extrapolation):

    poisson_gpu_aqua_a100.md     cuDSS vs CHOLMOD, cut-cell Poisson (job 22299810)
    poisson_mg_gpu_aqua_a100.md  matrix-free MG GPU vs CPU + V-cycles (job 22299933)
    cavity_gpu_aqua_a100.md      end-to-end SIMPLE cavity GPU vs CPU (job 22305186)

Renders into docs/src/users/:

    incns-gpu-speedup.png    GPU/CPU speed-up vs DOF (cuDSS solve, matrix-free MG)
    incns-mg-vcycles.png     MG V-cycle count vs DOF (flat -> O(N) evidence)
    incns-gpu-endtoend.png   per-solve vs end-to-end SIMPLE speed-up

Run: conda run -n kraken-v0-3-figures python docs/plot_incns_gpu_benchmarks.py
"""
import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# --- locate krakendark: $KRAKENDARK_DIR -> repo viz/ -> kraken-doc skill assets
_here = pathlib.Path(__file__).resolve()
_cands = []
if os.environ.get("KRAKENDARK_DIR"):
    _cands.append(pathlib.Path(os.environ["KRAKENDARK_DIR"]))
_cands += [_d / "viz" for _d in (_here.parent, *_here.parents)]
_cands.append(pathlib.Path.home() / ".claude" / "skills" / "kraken-doc" / "assets")
for _c in _cands:
    if (_c / "krakendark.py").exists():
        sys.path.insert(0, str(_c))
        break
else:
    sys.exit("krakendark.py not found (set KRAKENDARK_DIR)")
import krakendark as kd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "src", "users")

# --- benchmarks/results/poisson_gpu_aqua_a100.md (cuDSS solve speed-up) -----
CUDSS_DOF = [16_384, 65_536, 262_144, 1_048_576]            # 128^2 .. 1024^2
CUDSS_SPEEDUP = [4.0, 8.6, 21.0, 30.0]

# --- benchmarks/results/poisson_mg_gpu_aqua_a100.md (MG GPU/CPU + V-cycles) -
MG_N = [128, 256, 512, 1024, 2048, 4096]
MG_DOF = [16_384, 65_536, 262_144, 1_048_576, 4_194_304, 16_777_216]
MG_SPEEDUP = [0.31, 1.07, 3.84, 12.68, 31.26, 43.31]
MG_VCYCLES = [10, 10, 11, 12, 12, 13]

# --- benchmarks/results/cavity_gpu_aqua_a100.md (end-to-end SIMPLE cavity) --
CAVITY_GRIDS = ["256²", "512²"]
CAVITY_SPEEDUP = [0.98, 4.26]
# per-solve numbers at the SAME sizes (65 536 / 262 144 DOF), for the gap bar:
CUDSS_AT_CAVITY = [8.6, 21.0]
MG_AT_CAVITY = [1.07, 3.84]


def speedup_vs_size():
    c_cudss, c_mg = kd.palette(2)
    fig, ax = plt.subplots(figsize=(8.8, 6.2), constrained_layout=True)
    kd.grid(ax)

    ax.axhline(1.0, color=kd.GRID, ls="--", lw=1.2, zorder=1)
    ax.text(2.1e4, 1.12, kd.tex("break-even"), color="0.7", fontsize=11)

    ax.plot(CUDSS_DOF, CUDSS_SPEEDUP, "-", color=kd.CONNECT, lw=2.0, zorder=2)
    ax.plot(CUDSS_DOF, CUDSS_SPEEDUP,
            **kd.kraken_marker(c_cudss, marker="s", ms=10), zorder=4)
    ax.plot(MG_DOF, MG_SPEEDUP, "-", color=kd.CONNECT, lw=2.0, zorder=2)
    ax.plot(MG_DOF, MG_SPEEDUP,
            **kd.kraken_marker(c_mg, marker="D", ms=10), zorder=4)

    ax.text(8.5e5, 38, kd.tex("30x @ 1M DOF"), color="0.85", fontsize=12,
            ha="right", va="bottom")
    ax.text(1.5e7, 25, kd.tex("43x @ 16.8M DOF"), color="0.85", fontsize=12,
            ha="right", va="top")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.22, 80)
    ax.set(xlabel=kd.tex("degrees of freedom"),
           ylabel=kd.tex("GPU/CPU speed-up"),
           title=kd.tex("Pressure-Poisson solve — A100 vs CPU (F64)"))
    handles = [
        Line2D([0], [0], color=kd.CONNECT, lw=2.0, marker="s", ms=9, mec="0.92",
               mfc=c_cudss, label=kd.tex("cuDSS solve (factorize once)")),
        Line2D([0], [0], color=kd.CONNECT, lw=2.0, marker="D", ms=9, mec="0.92",
               mfc=c_mg, label=kd.tex("matrix-free multigrid (full solve)")),
    ]
    kd.dark_legend(ax, handles=handles, loc="upper left")
    out = os.path.join(OUT, "incns-gpu-speedup.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


def vcycles():
    fig, ax = plt.subplots(figsize=(8.4, 5.6), constrained_layout=True)
    kd.grid(ax)
    xs = np.arange(len(MG_N))
    ax.bar(xs, MG_VCYCLES, width=0.62, color=kd.COOL, edgecolor="0.85", lw=0.8)
    for x, v in zip(xs, MG_VCYCLES):
        ax.text(x, v + 0.25, str(v), ha="center", color="0.92", fontsize=13)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{n}²" for n in MG_N])
    ax.set_ylim(0, 16)
    ax.set(xlabel=kd.tex("grid (DOF: 16k to 16.8M)"),
           ylabel=kd.tex("V-cycles to tol = 1e-8"),
           title=kd.tex("Multigrid V-cycle count stays flat — O(N) solve"))
    out = os.path.join(OUT, "incns-mg-vcycles.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


def end_to_end():
    c = kd.palette(3)
    fig, ax = plt.subplots(figsize=(8.8, 6.0), constrained_layout=True)
    kd.grid(ax)

    xs = np.arange(len(CAVITY_GRIDS))
    w = 0.26
    series = [
        (kd.tex("cuDSS back-substitution (per solve)"), CUDSS_AT_CAVITY, c[0]),
        (kd.tex("matrix-free MG (per solve)"), MG_AT_CAVITY, c[1]),
        (kd.tex("full SIMPLE cavity (end-to-end)"), CAVITY_SPEEDUP, c[2]),
    ]
    for k, (label, vals, color) in enumerate(series):
        bars = ax.bar(xs + (k - 1) * w, vals, width=w, color=color,
                      edgecolor="0.85", lw=0.8, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.35,
                    kd.tex(f"{v:g}x"), ha="center", color="0.92", fontsize=11)

    ax.axhline(1.0, color=kd.GRID, ls="--", lw=1.2, zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"cavity {g}" for g in CAVITY_GRIDS])
    ax.set(ylabel=kd.tex("GPU/CPU speed-up"),
           title=kd.tex("Per-solve vs end-to-end — orchestration eats the gap"))
    kd.dark_legend(ax, loc="upper left")
    out = os.path.join(OUT, "incns-gpu-endtoend.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    kd.apply()
    speedup_vs_size()
    vcycles()
    end_to_end()
