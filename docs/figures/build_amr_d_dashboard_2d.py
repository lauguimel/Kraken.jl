#!/usr/bin/env python3
"""Build the AMR-D dashboard from CSVs using matplotlib (strict bounding boxes)."""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors

RUNS_DIR = "/Users/guillaume/Documents/Recherche/Kraken.jl/tmp/M-NEXT-V5/scratch/dashboard_all10_fresh"
OUT = "/Users/guillaume/Documents/Recherche/Kraken.jl/tmp/M-NEXT-V5/grid_dashboard_matplotlib.png"

CASES = [
    ("[1/12] Couette\nxband_v_full",     "amr_d_couette_xband_v_full_nested1",     "amr"),
    ("[2/12] Couette\nyband_h_full",     "amr_d_couette_yband_h_full_nested1",     "amr"),
    ("[3/12] Couette\nH (3-patch)",      "amr_d_couette_H_nested1",                "amr"),
    ("[4/12] Couette\nxband_v_int",      "amr_d_couette_xband_v_interior_nested1", "amr"),
    ("[5/12] Couette\nyband_h_int",      "amr_d_couette_yband_h_interior_nested1", "amr"),
    ("[6/12] Couette\ncartesian ref",    "amr_d_couette_yband_h_full_nested1",     "ref"),
    ("[7/12] Poiseuille\nxband_v_full",  "amr_d_poiseuille_xband_v_full_nested1",  "amr"),
    ("[8/12] Poiseuille\nyband_h_full",  "amr_d_poiseuille_yband_h_full_nested1",  "amr"),
    ("[9/12] Poiseuille\nH (3-patch)",   "amr_d_poiseuille_H_nested1",             "amr"),
    ("[10/12] Poiseuille\nxband_v_int",  "amr_d_poiseuille_xband_v_interior_nested1", "amr"),
    ("[11/12] Poiseuille\nyband_h_int",  "amr_d_poiseuille_yband_h_interior_nested1", "amr"),
    ("[12/12] Poiseuille\ncartesian ref","amr_d_poiseuille_yband_h_full_nested1",  "ref"),
]


def ref_suffix(case_dir):
    if os.path.isfile(os.path.join(RUNS_DIR, case_dir, "fields_cartesian_classic.csv")):
        return "cartesian_classic"
    return "leaf_oracle"


def read_csv_with_header(path):
    with open(path, "r") as f:
        header = f.readline().strip().split(",")
        rows = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(line.split(","))
    return header, rows


def load_field(fields_csv):
    """Return (rho_grid, ux_grid, nx, ny) where grids are nx×ny arrays."""
    header, rows = read_csv_with_header(fields_csv)
    idx = {name: k for k, name in enumerate(header)}
    nx = max(int(r[idx["i_leaf"]]) for r in rows)
    ny = max(int(r[idx["j_leaf"]]) for r in rows)
    rho = np.full((nx, ny), np.nan)
    ux = np.full((nx, ny), np.nan)
    for r in rows:
        i = int(r[idx["i_leaf"]]) - 1
        j = int(r[idx["j_leaf"]]) - 1
        rho_str = r[idx["rho"]]
        ux_str = r[idx["ux"]]
        rho[i, j] = float(rho_str) if rho_str.lower() != "nan" else np.nan
        ux[i, j] = float(ux_str) if ux_str.lower() != "nan" else np.nan
    return rho, ux, nx, ny


def load_mesh(mesh_csv):
    """Return list of (level, imin, imax, jmin, jmax) for active cells."""
    header, rows = read_csv_with_header(mesh_csv)
    idx = {name: k for k, name in enumerate(header)}
    out = []
    for r in rows:
        active_field = r[idx["active"]]
        active = active_field.lower() in ("true", "1")
        if not active:
            continue
        out.append((
            int(r[idx["level"]]),
            float(r[idx["leaf_i_min"]]),
            float(r[idx["leaf_i_max"]]),
            float(r[idx["leaf_j_min"]]),
            float(r[idx["leaf_j_max"]]),
        ))
    return out


def load_profile(profile_csv):
    """Return (y, ux_amr, analytic_ux) for the 'mean_y' rows."""
    if not os.path.isfile(profile_csv):
        return None, None, None
    header, rows = read_csv_with_header(profile_csv)
    idx = {name: k for k, name in enumerate(header)}
    y, ux, an = [], [], []
    for r in rows:
        if r[idx["kind"]].strip().strip('"') != "mean_y":
            continue
        y.append(float(r[idx["coord"]]))
        ux.append(float(r[idx["ux"]]))
        an_str = r[idx["analytic_ux"]]
        try:
            an.append(float(an_str))
        except Exception:
            an.append(np.nan)
    return np.array(y), np.array(ux), np.array(an)


def draw_mesh(ax, mesh_cells):
    level_colors = ["#b0c4de", "#ff8c00", "#dc143c", "#800080", "#000000"]
    for (lvl, imin, imax, jmin, jmax) in mesh_cells:
        c = level_colors[min(lvl, len(level_colors) - 1)]
        ax.plot([imin - 0.5, imax + 0.5, imax + 0.5, imin - 0.5, imin - 0.5],
                [jmin - 0.5, jmin - 0.5, jmax + 0.5, jmax + 0.5, jmin - 0.5],
                color=c, linewidth=0.6)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def draw_heatmap(ax, field, cmap, vmin, vmax, title=None, cbar_label=None,
                 cbar_ticks=None, cbar_tick_labels=None):
    nx, ny = field.shape
    im = ax.imshow(field.T, origin='lower',
                   extent=[0.5, nx + 0.5, 0.5, ny + 0.5],
                   cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect='equal', interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=8, pad=2)
    # Inline colorbar : sized to the imshow vertical extent
    cax = ax.inset_axes([1.02, 0.0, 0.05, 1.0])
    cb = plt.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=6, length=2, pad=1)
    if cbar_ticks is not None:
        cb.set_ticks(cbar_ticks)
        if cbar_tick_labels is not None:
            cb.set_ticklabels(cbar_tick_labels)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=7, labelpad=2)
    return im


def draw_profile(ax, y, ux, an, label_prefix=""):
    if y is None:
        ax.text(0.5, 0.5, "(no profile)", ha='center', va='center', fontsize=8,
                transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        return
    ax.plot(ux, y, color="#dc143c", linewidth=1.4, label="AMR-D")
    if not np.all(np.isnan(an)):
        ax.scatter(an, y, color="black", s=4, label="analytic", zorder=3)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("u_x", fontsize=8, labelpad=1)
    ax.set_ylabel("y", fontsize=8, labelpad=1)
    ax.legend(fontsize=6, loc='best', frameon=False)


def build_dashboard():
    nrows = len(CASES)
    ncols = 5  # label, mesh, u_x, rho, profile
    fig_width = 16.0      # inches
    row_height = 1.5      # inches per row
    fig_height = 1.0 + nrows * row_height  # 1 inch header

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)
    # Widen the heatmap columns to make room for inline colorbars
    gs = fig.add_gridspec(
        nrows + 1, ncols,
        width_ratios=[1.2, 2.5, 3.4, 3.4, 3.5],
        height_ratios=[0.4] + [1.0] * nrows,
        hspace=0.30, wspace=0.35,
        left=0.02, right=0.99, top=0.97, bottom=0.02
    )

    # Header row
    ax = fig.add_subplot(gs[0, 0]); ax.axis('off')
    for col, txt in enumerate(["Wireframe (mesh)", "u_x field",
                                "ρ field (centered on 1)", "Profile U(y)"], start=1):
        ax = fig.add_subplot(gs[0, col])
        ax.text(0.5, 0.5, txt, ha='center', va='center',
                fontsize=12, fontweight='bold', transform=ax.transAxes)
        ax.axis('off')

    for i, (label, case_dir, mode) in enumerate(CASES):
        row = i + 1
        base = os.path.join(RUNS_DIR, case_dir)
        ref = ref_suffix(case_dir)
        if mode == "amr":
            fc = os.path.join(base, "fields_amr_d.csv")
            mc = os.path.join(base, "mesh_amr_d.csv")
        else:
            fc = os.path.join(base, f"fields_{ref}.csv")
            mc = os.path.join(base, f"mesh_{ref}.csv")
        pa = os.path.join(base, "profiles_amr_d.csv")
        pr = os.path.join(base, f"profiles_{ref}.csv")

        # Label
        ax_lab = fig.add_subplot(gs[row, 0]); ax_lab.axis('off')
        ax_lab.text(0.5, 0.5, label, ha='center', va='center',
                    fontsize=10, fontweight='bold', transform=ax_lab.transAxes,
                    wrap=True)

        # Mesh
        ax_mesh = fig.add_subplot(gs[row, 1])
        if os.path.isfile(mc):
            cells = load_mesh(mc)
            draw_mesh(ax_mesh, cells)
        else:
            ax_mesh.text(0.5, 0.5, "(no mesh)", ha='center', va='center', fontsize=8)
            ax_mesh.axis('off')

        # Fields
        if os.path.isfile(fc):
            rho, ux, nx, ny = load_field(fc)
            ux_finite = ux[np.isfinite(ux)]
            if ux_finite.size > 0:
                vmin, vmax = ux_finite.min(), ux_finite.max()
                if vmax - vmin < 1e-9:
                    vmax = vmin + 1e-6
            else:
                vmin, vmax = 0.0, 1.0
            ax_ux = fig.add_subplot(gs[row, 2])
            draw_heatmap(ax_ux, ux, "inferno", vmin, vmax,
                         cbar_label="u_x",
                         cbar_ticks=[vmin, vmax],
                         cbar_tick_labels=[f"{vmin:.3g}", f"{vmax:.3g}"])

            rho_finite = rho[np.isfinite(rho)]
            if rho_finite.size > 0:
                dev = np.max(np.abs(rho_finite - 1.0))
            else:
                dev = 0.0
            span = max(dev, 1e-4)
            ax_rho = fig.add_subplot(gs[row, 3])
            draw_heatmap(ax_rho, rho, "RdBu_r", 1.0 - span, 1.0 + span,
                         title=f"max|ρ-1|={dev:.2e}",
                         cbar_label="ρ",
                         cbar_ticks=[1.0 - span, 1.0, 1.0 + span],
                         cbar_tick_labels=[f"{1.0-span:.4f}", "1.000", f"{1.0+span:.4f}"])

        # Profile — load AMR-D for amr rows, reference for ref rows
        ax_prof = fig.add_subplot(gs[row, 4])
        profile_csv = pa if mode == "amr" else pr
        y, ux_p, an = load_profile(profile_csv)
        draw_profile(ax_prof, y, ux_p, an)

    fig.savefig(OUT, dpi=150, bbox_inches='tight', pad_inches=0.1,
                facecolor='white')
    print(f"wrote {OUT}")
    print(f"size: {os.path.getsize(OUT)} bytes")


if __name__ == "__main__":
    build_dashboard()
