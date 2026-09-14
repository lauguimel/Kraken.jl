#!/usr/bin/env python3
r"""Per-Reynolds dark cavity centerline figures for the Vitepress docs.

One 2-panel figure per Reynolds number (Re = 100, 400, 1000), baked on the live
Vitepress dark page colour (#1b1b1f, ``kd.DARK``) so it sits seamlessly on the
docs page. Matches the page caption exactly:

    Kraken (solid), icoFoam (dashed), Ghia 1982 (markers); u(y) left, v(x) right.

Classic centerline presentation -- the spatial coordinate on the ordinate, the
velocity on the abscissa:

    left  -- u along the vertical centerline   (x = 0.5):  u vs y   (u_vert rows)
    right -- v along the horizontal centerline (y = 0.5):  v vs x   (v_horiz rows)

Three series per panel, distinguished by BOTH style and colour:
    Kraken  = solid line
    icoFoam = dashed line
    Ghia    = open-ring markers (no line)

Self-contained: bootstrap-finds ``krakendark`` (repo ``viz/`` or ``assets/``,
or ``$KRAKENDARK_DIR``), reads the three CSVs next to this script, and writes
the three PNGs straight into ``docs/src/users/benchmarks/`` (overwriting the
light versions).

    conda run -n kraken-v0-3-figures python plot_cavity_per_re.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- locate krakendark (repo viz/ or assets/, or $KRAKENDARK_DIR) ------------
import sys
import pathlib
_here = pathlib.Path(__file__).resolve()
_extra = os.environ.get("KRAKENDARK_DIR")
_extra_cands = (pathlib.Path(_extra),) if _extra else ()
for _d in [_here.parent, *_here.parents]:
    for _cand in (_d, _d / "viz", _d / "assets", _d.parent / "assets", *_extra_cands):
        if (_cand / "krakendark.py").exists():
            sys.path.insert(0, str(_cand))
            break
    else:
        continue
    break
import krakendark as kd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = [100, 400, 1000]

# CSVs live next to this script; PNGs go into the docs benchmarks dir.
_OUT_DIR = os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "..", "docs", "src", "users", "benchmarks"))


def load(re):
    """Return {profile: {col: [floats]}} for one Reynolds number."""
    path = os.path.join(HERE, f"cavity_centerline_Re{re}.csv")
    rows = {}
    with open(path, newline="") as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("#")]
        for r in csv.DictReader(lines):
            prof = r["profile"]
            d = rows.setdefault(prof,
                                {"coord": [], "kraken": [], "icofoam": [], "ghia": []})
            for k in ("coord", "kraken", "icofoam", "ghia"):
                v = r.get(k, "")
                d[k].append(float(v) if v not in (None, "", "nan", "NaN")
                            else float("nan"))
    return rows


def _draw(ax, series, c_kraken, c_ico, c_ghia):
    """Three series, velocity on the abscissa, spatial coordinate on the ordinate.

    Kraken = solid line, icoFoam = dashed line, Ghia = open-ring markers.
    """
    ax.plot(series["kraken"], series["coord"], "-", color=c_kraken, lw=2.6,
            zorder=3)
    ax.plot(series["icofoam"], series["coord"], "--", color=c_ico, lw=2.0,
            dashes=(5, 4), zorder=4)
    ax.plot(series["ghia"], series["coord"], ls="none", marker="o", mfc="none",
            mec=c_ghia, ms=9, mew=2.0, zorder=5)


def main():
    usetex = kd.apply()  # line/bar dark theme (seaborn talk sizes + #1b1b1f)

    # Distinct hues so the three solvers separate by colour as well as style.
    pal = kd.palette(3)
    c_kraken, c_ico, c_ghia = pal[2], pal[0], pal[1]  # green, blue, orange

    # Reusable legend handles identifying the three solvers (style + colour).
    solver_handles = [
        Line2D([0], [0], color=c_kraken, lw=2.4, ls="-", label="Kraken (LBM)"),
        Line2D([0], [0], color=c_ico, lw=2.2, ls="--", label="icoFoam"),
        Line2D([0], [0], color=c_ghia, ls="none", marker="o", mfc="none",
               mec=c_ghia, ms=9, mew=2.0, label=kd.tex("Ghia 1982")),
    ]

    os.makedirs(_OUT_DIR, exist_ok=True)
    for re in RES:
        d = load(re)
        fig, (axu, axv) = plt.subplots(1, 2, figsize=(13.2, 6.0),
                                       constrained_layout=True)
        _draw(axu, d["u_vert"], c_kraken, c_ico, c_ghia)
        _draw(axv, d["v_horiz"], c_kraken, c_ico, c_ghia)

        for ax in (axu, axv):
            kd.grid(ax)
        axu.set(xlabel=r"$u / U_{\mathrm{lid}}$", ylabel=r"$y / L$",
                title=r"$u$ along vertical centerline $x = 0.5$")
        axv.set(xlabel=r"$v / U_{\mathrm{lid}}$", ylabel=r"$x / L$",
                title=r"$v$ along horizontal centerline $y = 0.5$")

        kd.dark_legend(axu, handles=solver_handles, title="Solver",
                       loc="upper left")
        kd.dark_legend(axv, handles=solver_handles, title="Solver",
                       loc="lower right")

        fig.suptitle(kd.tex(f"Lid-driven cavity --- Re = {re}"),
                     fontsize=16, fontweight="bold")

        out = os.path.join(_OUT_DIR, f"cartesian-cavity-re{re}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out} (usetex={usetex})")


if __name__ == "__main__":
    main()
