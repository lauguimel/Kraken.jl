#!/usr/bin/env python3
r"""Growth rate sigma(T) for the EHD electroconvection Tc sweep.

Late-window growth rate (least-squares slope of log(max|u|) over the trailing
40% of the amplitude history) vs the electric Rayleigh number T, at 197x321,
MRT + regularized charge + direct Poisson (see benchmarks/results/ehd/README.md).
Marks the sign change (Kraken) against the reference threshold T_c = 163.5
(Luo, Wu, Yi & Tan, Phys. Rev. E 93, 023309, 2016).

The values below are the validated, cross-checked `growth_rate_late` figures
(trailing-window slope of log(max|u|)); the `growth_rate_estimate` /
`cumulative_log_slope` column in the retained CSVs is NOT used here (it is a
known trap: positive even for decaying runs).

Env: conda run -n kraken-v0-3-figures python plot_tc_sweep.py
"""
import os
import sys
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_here = pathlib.Path(__file__).resolve()
for _d in [_here.parent, *_here.parents]:
    for _cand in (_d, _d / "viz", _d / "assets", _d.parent / "assets"):
        if (_cand / "krakendark.py").exists():
            sys.path.insert(0, str(_cand))
            break
    else:
        continue
    break
else:
    sys.path.insert(0, "/Users/guillaume/.claude/skills/kraken-doc/assets")
import krakendark as kd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# Late-window growth rate sigma (1/step), per mission brief M-EHD-6c, validated
# against the trailing 40% least-squares fit of log(max|u|) for each retained
# tc_sweep_T<T>_mrt_*.csv history.
T = [150, 160, 163.5, 165, 166, 167, 170, 190]
SIGMA = [-8.79e-6, -3.39e-6, -1.54e-6, -7.53e-7, -2.35e-7,
         2.79e-7, 1.80e-6, 1.26e-5]

T_C_REF = 163.5   # Luo, Wu, Yi & Tan (2016)
T_C_KRAKEN = 166.5  # sign change bracket at this resolution (166 -> 167)


def main():
    kd.apply()

    fig, ax = plt.subplots(figsize=(8.4, 6.2), constrained_layout=True)
    kd.grid(ax)

    ax.axhline(0.0, color="0.5", lw=1.0, ls="-", zorder=1)
    ax.axvline(T_C_REF, color=kd.COOL, lw=1.6, ls="--", zorder=2,
               label=kd.tex(f"reference T_c = {T_C_REF} (Luo et al. 2016)"))
    ax.axvline(T_C_KRAKEN, color=kd.ACCENT, lw=1.6, ls=":", zorder=2,
               label=kd.tex(f"Kraken bracket T_c ~ {T_C_KRAKEN}"))

    ax.plot(T, SIGMA, "-", color=kd.CONNECT, lw=2.0, zorder=3)
    palette = kd.palette(len(T))
    for color, Ti, si in zip(palette, T, SIGMA):
        marker_kw = kd.kraken_marker(color, marker="o", ms=11)
        ax.plot(Ti, si, **marker_kw, zorder=4)

    ax.set_yscale("symlog", linthresh=1e-6)
    ax.set(xlabel=kd.tex("Electric Rayleigh number T"),
           ylabel=kd.tex("late-window growth rate sigma (1/step, symlog)"),
           title=kd.tex("EHD electroconvection onset: sigma(T), 197x321"))
    kd.dark_legend(ax, title="Threshold", loc="lower right")

    out = os.path.join(HERE, "growth_rate_vs_T.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
