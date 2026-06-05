#!/usr/bin/env python3
r"""Reproduce the thermal natural-convection comparison plots from the shipped CSVs.

Self-contained: reads ``kraken_natconv_results.csv`` and ``of_natconv_results.csv``
next to this script and regenerates ``comparison.png``.

Two panels, classic differentially-heated-cavity presentation:

    left  -- Nusselt number vs Rayleigh number (log Ra):
             Kraken (LBM) solid line + markers, de Vahl Davis (1983) open
             circles, OpenFOAM ``buoyantBoussinesqSimpleFoam`` triangles
             (Ra = 10^4 OF point is dropped: under-converged).
    right -- Ra = 10^5 convergence ladder: Kraken Nu error (%) vs base mesh N,
             monotone descent across the 1 % gate near N ~ 384.

Colour encodes the Rayleigh number (seaborn ``crest``) on the left panel.

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
_USETEX = shutil.which("latex") is not None
DARK = "#1b1b1f"  # Vitepress dark page background

# Rayleigh sweep (ordered) and a human-readable LaTeX exponent label.
RA = [1e3, 1e4, 1e5]
RA_LABELS = {1e3: r"$10^{3}$", 1e4: r"$10^{4}$", 1e5: r"$10^{5}$"}


def _load(path):
    """Return list of dict rows, skipping comment lines (#...)."""
    with open(path, newline="") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
        return list(csv.DictReader(lines))


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def load_kraken():
    """Reference Nu(Ra) at the canonical 128^2 (Ra<=1e4) / 192^2 (Ra=1e5) grids,
    plus the full Ra=1e5 N-ladder. Use the F64 CPU rows where available, else
    the matching Metal F32 row -- they agree to < 0.01 % at the canonical grid."""
    rows = _load(os.path.join(HERE, "kraken_natconv_results.csv"))
    nu, ref, ladder = {}, {}, []
    # canonical Nu(Ra): cavity@128 for 1e3/1e4, cavity_finer@192 for 1e5.
    canon = {1e3: ("cavity", 128), 1e4: ("cavity", 128), 1e5: ("cavity_finer", 192)}
    for r in rows:
        ra = _f(r["Ra"])
        case, N = r["case"], int(r["N"])
        if ra in canon and (case, N) == canon[ra]:
            # prefer F64; only overwrite a Metal value with an F64 value.
            if ra not in nu or r["backend"] == "cpu_f64":
                nu[ra] = _f(r["Nu_kraken"])
                ref[ra] = _f(r["Nu_ref"])
        # Ra=1e5 convergence ladder (Metal F32 cavity_finer sweep).
        if ra == 1e5 and case == "cavity_finer" and r["backend"] == "metal_f32":
            ladder.append((N, abs(_f(r["Nu_err_pct"]))))
    ladder.sort()
    return nu, ref, ladder


def load_openfoam():
    """OpenFOAM buoyantBoussinesqSimpleFoam Nu(Ra), converged rows only."""
    rows = _load(os.path.join(HERE, "of_natconv_results.csv"))
    of = {}
    for r in rows:
        if "UNCONVERGED" in r["solver"]:
            continue
        of[_f(r["Ra"])] = _f(r["Nu_snGrad"])
    return of


def main():
    sns.set_theme(style="ticks", context="talk", font="serif")
    plt.rcParams.update({
        "text.usetex": _USETEX, "font.family": "serif", "mathtext.fontset": "cm",
        "figure.facecolor": DARK, "axes.facecolor": DARK, "savefig.facecolor": DARK,
        "text.color": "0.92", "axes.labelcolor": "0.92", "axes.titlecolor": "0.96",
        "axes.edgecolor": "0.55", "xtick.color": "0.85", "ytick.color": "0.85",
    })
    palette = sns.color_palette("bright", len(RA))  # vivid; pops on dark

    nu_k, nu_ref, ladder = load_kraken()
    of = load_openfoam()

    def draw_nu(axn):
        axn.grid(True, color="0.45", alpha=0.4, lw=0.6)
        ra_sorted = sorted(nu_k)
        axn.plot(ra_sorted, [nu_k[r] for r in ra_sorted], "-", color="0.8",
                 lw=2.0, zorder=1)
        for color, ra in zip(palette, ra_sorted):
            axn.plot(ra, nu_k[ra], "s", color=color, ms=11, mec="0.92", mew=1.0,
                     zorder=4)
            axn.plot(ra, nu_ref[ra], ls="none", marker="o", mfc="none",
                     mec=color, ms=13, mew=2.0, zorder=3)
            if ra in of:
                axn.plot(ra, of[ra], "^", color=color, ms=10, mfc="none",
                         mew=2.0, zorder=2)
        axn.set_xscale("log")
        axn.set(xlabel=r"Rayleigh number $\mathrm{Ra}$",
                ylabel=r"Nusselt number $\overline{\mathrm{Nu}}$",
                title=r"$\overline{\mathrm{Nu}}$ vs $\mathrm{Ra}$ "
                      r"--- heated cavity")
        solver_handles = [
            Line2D([0], [0], color="0.9", lw=2.0, marker="s", ms=10, mec="0.92",
                   label="Kraken (LBM)"),
            Line2D([0], [0], color="0.9", ls="none", marker="o", mfc="none",
                   mec="0.9", ms=11, mew=2.0, label="de Vahl Davis (1983)"),
            Line2D([0], [0], color="0.9", ls="none", marker="^", mfc="none",
                   mec="0.9", ms=10, mew=2.0,
                   label=r"OpenFOAM buoyantBoussinesq"),
        ]
        leg = axn.legend(handles=solver_handles, loc="upper left", fontsize=11,
                         title="Solver", facecolor=DARK, edgecolor="0.5",
                         labelcolor="0.9", framealpha=0.85)
        leg.get_title().set_color("0.9")

    def draw_conv(axc):
        axc.grid(True, color="0.45", alpha=0.4, lw=0.6)
        if ladder:
            Ns = [n for n, _ in ladder]
            errs = [e for _, e in ladder]
            lad_palette = sns.color_palette("bright", len(Ns))
            axc.plot(Ns, errs, "-", color="0.8", lw=2.0, zorder=1)
            for color, n, e in zip(lad_palette, Ns, errs):
                axc.plot(n, e, "o", color=color, ms=11, mec="0.92", mew=1.0,
                         zorder=3)
        axc.axhline(1.0, color="0.9", ls="--", lw=1.4, zorder=2)
        axc.text(0.97, 1.06, r"$1\%$ gate",
                 transform=axc.get_yaxis_transform(), ha="right", va="bottom",
                 fontsize=11)
        axc.set(xlabel=r"base mesh resolution $N$",
                ylabel=r"$|\,\overline{\mathrm{Nu}}$ error$\,|$  [\%]"
                       if _USETEX else r"$|\overline{\mathrm{Nu}}$ error$|$  [%]",
                title=r"$\mathrm{Ra}=10^{5}$ convergence ladder")

    # Combined two-panel reference figure.
    fig, (axn, axc) = plt.subplots(1, 2, figsize=(14, 6.2),
                                   constrained_layout=True)
    draw_nu(axn)
    draw_conv(axc)
    fig.suptitle("Differentially-heated square cavity --- Kraken (LBM) "
                 "vs de Vahl Davis (1983)", fontsize=15, fontweight="bold")
    out = os.path.join(HERE, "comparison.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out} (usetex={_USETEX})")

    # Two standalone single-panel page figures (clean, no suptitle bleed):
    # comparison_nu.png is the Nu-vs-Ra panel, comparison_convergence.png the
    # Ra=1e5 ladder. These map one-to-one onto the two docs image refs.
    for draw, name in ((draw_nu, "comparison_nu.png"),
                       (draw_conv, "comparison_convergence.png")):
        f1, a1 = plt.subplots(figsize=(7.6, 6.2), constrained_layout=True)
        draw(a1)
        sub = os.path.join(HERE, name)
        f1.savefig(sub, dpi=150)
        plt.close(f1)
        print(f"wrote {sub}")


if __name__ == "__main__":
    main()
