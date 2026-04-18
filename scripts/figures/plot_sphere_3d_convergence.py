#!/usr/bin/env python3
"""
WP-3D-6 — Figure: sphere 3D Re=20 convergence (uniform vs stretched).

Parses the structured output of `hpc/slbm_sphere_h100.jl` and produces:
  paper/figures/sphere_3d_convergence.pdf — Cd error and MLUPS vs cells

Expected log line format (one per run):
    <label> <Nx>x<Ny>x<Nz> ( <cells> cells) Cd=<val> err=<val>% MLUPS=<val> (<elapsed>s, ...)

Usage:
    python scripts/figures/plot_sphere_3d_convergence.py \
        --log results/slbm_sphere_3d.log \
        --out paper/figures/sphere_3d_convergence.pdf
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

LINE_RE = re.compile(
    r"^(?P<label>\S.*?)\s+"
    r"(?P<Nx>\d+)x(?P<Ny>\d+)x(?P<Nz>\d+)\s+"
    r"\(\s*(?P<cells>\d+)\s*cells\)\s+"
    r"Cd=(?P<cd>[-\d.]+)\s+"
    r"err=(?P<err>[-\d.]+)%\s+"
    r"MLUPS=(?P<mlups>[-\d.]+)"
)


def parse(path: Path):
    rows = []
    for raw in path.read_text().splitlines():
        m = LINE_RE.search(raw.strip())
        if not m:
            continue
        rows.append({
            "label":  m["label"].strip(),
            "Nx":     int(m["Nx"]),
            "Ny":     int(m["Ny"]),
            "Nz":     int(m["Nz"]),
            "cells":  int(m["cells"]),
            "Cd":     float(m["cd"]),
            "err":    float(m["err"]),
            "mlups":  float(m["mlups"]),
        })
    return rows


def split_uniform_stretched(rows):
    uni  = [r for r in rows if r["label"].lower().startswith("uniform")]
    strd = [r for r in rows if r["label"].lower().startswith("stretch")]
    uni.sort(key=lambda r: r["cells"])
    strd.sort(key=lambda r: r["cells"])
    return uni, strd


def plot(uni, strd, out_path: Path):
    fig, (ax_err, ax_mlups) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    if uni:
        ax_err.plot([r["cells"] for r in uni], [r["err"] for r in uni],
                    "o-", label="Uniform Cartesian", linewidth=2, markersize=7)
    if strd:
        ax_err.plot([r["cells"] for r in strd], [r["err"] for r in strd],
                    "s--", label="Stretched (local-CFL)", linewidth=2, markersize=7)
    ax_err.set_xscale("log")
    ax_err.set_yscale("log")
    ax_err.set_xlabel("cells")
    ax_err.set_ylabel(r"$|\mathrm{Cd} - \mathrm{Cd}_\mathrm{ref}| / \mathrm{Cd}_\mathrm{ref}$ [%]")
    ax_err.grid(True, which="both", alpha=0.3)
    ax_err.legend()
    ax_err.set_title("Sphere Re=20 — Cd accuracy")

    if uni:
        ax_mlups.plot([r["cells"] for r in uni], [r["mlups"] for r in uni],
                      "o-", label="Uniform Cartesian", linewidth=2, markersize=7)
    if strd:
        ax_mlups.plot([r["cells"] for r in strd], [r["mlups"] for r in strd],
                      "s--", label="Stretched (local-CFL)", linewidth=2, markersize=7)
    ax_mlups.set_xscale("log")
    ax_mlups.set_xlabel("cells")
    ax_mlups.set_ylabel("MLUPS (H100, Float64)")
    ax_mlups.grid(True, which="both", alpha=0.3)
    ax_mlups.legend()
    ax_mlups.set_title("Throughput vs grid size")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, required=True,
                   help="path to slbm_sphere_3d.log produced on Aqua")
    p.add_argument("--out", type=Path,
                   default=Path("paper/figures/sphere_3d_convergence.pdf"))
    args = p.parse_args()

    if not args.log.exists():
        sys.exit(f"log file not found: {args.log}")

    rows = parse(args.log)
    if not rows:
        sys.exit("no benchmark lines parsed; check log format")

    uni, strd = split_uniform_stretched(rows)
    print(f"parsed {len(rows)} runs ({len(uni)} uniform, {len(strd)} stretched)")
    for r in rows:
        print(f"  {r['label']:30s}  cells={r['cells']:>10d}  "
              f"Cd={r['Cd']:.3f}  err={r['err']:.2f}%  MLUPS={r['mlups']:.0f}")

    plot(uni, strd, args.out)


if __name__ == "__main__":
    main()
