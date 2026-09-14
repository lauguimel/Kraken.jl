#!/usr/bin/env python3
"""Dark validation figures for the incompressible steady Navier-Stokes doc page.

Reads the CSV profiles written by docs/generate_incns_figures.jl (run that
first) and renders the krakendark figures used by
docs/src/users/incompressible-navier-stokes.md:

    incns-poiseuille.png        computed channel profile vs analytic parabola
    incns-cavity-re100.png      Re=100 centreline u(y), v(x) vs Ghia (1982)
    incns-cavity-re1000.png     Re=1000 centreline u(y), v(x) vs Ghia (1982)

Run: conda run -n kraken-v0-3-figures python docs/plot_incns_validation.py
"""
import csv
import os
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
DATA = os.path.join(HERE, "incns_figdata")
OUT = os.path.join(HERE, "src", "users")


def read_csv(name):
    with open(os.path.join(DATA, name)) as fh:
        rows = list(csv.reader(fh))
    cols = {h: [float(r[k]) for r in rows[1:]] for k, h in enumerate(rows[0])}
    return cols


def poiseuille():
    d = read_csv("poiseuille_profile.csv")
    y, uk, ua = d["y"], d["u_kraken"], d["u_analytic"]
    c = kd.palette(2)[0]

    fig, ax = plt.subplots(figsize=(8.4, 6.2), constrained_layout=True)
    kd.grid(ax)
    # Kraken: grey connector + filled vivid markers (subsampled for legibility).
    ax.plot(y, uk, "-", color=kd.CONNECT, lw=2.0, zorder=2)
    ax.plot(y[::4], uk[::4], **kd.kraken_marker(c, marker="s", ms=9), zorder=4)
    # analytic parabola: open rings in the same hue (reference convention).
    ax.plot(y[2::8], ua[2::8], **kd.reference_marker(c, marker="o", ms=12), zorder=3)
    ax.set(xlabel=r"$y/H$", ylabel=r"$u(y)$",
           title=kd.tex("Plane Poiseuille — SIMPLE (FVFD) vs analytic"))
    handles = [
        Line2D([0], [0], color=kd.CONNECT, lw=2.0, marker="s", ms=9, mec="0.92",
               mfc=c, label=kd.tex("Kraken IncNS (8x64), L2 = 0.033%")),
        Line2D([0], [0], ls="none", marker="o", mfc="none", mec=c, ms=11,
               mew=2.0, label=r"analytic $u = \frac{G}{2\mu}\,y\,(H-y)$"),
    ]
    kd.dark_legend(ax, handles=handles, loc="lower center")
    out = os.path.join(OUT, "incns-poiseuille.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


def cavity(tag, re_label, grid_label):
    du = read_csv(f"cavity_{tag}_u_centerline.csv")
    dv = read_csv(f"cavity_{tag}_v_centerline.csv")
    gu = read_csv(f"cavity_{tag}_ghia_u.csv")
    gv = read_csv(f"cavity_{tag}_ghia_v.csv")
    cu, cv = kd.palette(2)

    fig, (axu, axv) = plt.subplots(1, 2, figsize=(12.6, 6.0),
                                   constrained_layout=True)
    for ax in (axu, axv):
        kd.grid(ax)

    # u(y) on the vertical centreline x = 0.5 (y vertical, traditional layout).
    axu.plot(du["u_kraken"], du["y"], "-", color=cu, lw=2.2, zorder=2)
    axu.plot(gu["u_ghia"], gu["y"],
             **kd.reference_marker(cu, marker="o", ms=11), zorder=3)
    axu.set(xlabel=r"$u/U_{\mathrm{lid}}$", ylabel=r"$y/L$",
            title=kd.tex(f"u on x = 0.5 — {re_label}"))

    # v(x) on the horizontal centreline y = 0.5.
    axv.plot(dv["x"], dv["v_kraken"], "-", color=cv, lw=2.2, zorder=2)
    axv.plot(gv["x"], gv["v_ghia"],
             **kd.reference_marker(cv, marker="o", ms=11), zorder=3)
    axv.set(xlabel=r"$x/L$", ylabel=r"$v/U_{\mathrm{lid}}$",
            title=kd.tex(f"v on y = 0.5 — {re_label}"))

    handles = [
        Line2D([0], [0], color="0.9", lw=2.2,
               label=kd.tex(f"Kraken IncNS MG ({grid_label})")),
        Line2D([0], [0], ls="none", marker="o", mfc="none", mec="0.9", ms=11,
               mew=2.0, label="Ghia et al. (1982)"),
    ]
    kd.dark_legend(axu, handles=handles, loc="upper left")
    out = os.path.join(OUT, f"incns-cavity-{tag}.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    kd.apply()
    poiseuille()
    cavity("re100", "Re = 100", "128²")
    cavity("re1000", "Re = 1000", "256²")
