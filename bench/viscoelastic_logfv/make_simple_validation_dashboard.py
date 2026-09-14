#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import csv
import html
import math
from pathlib import Path
import warnings
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


SUMMARY_COLUMNS_NUMERIC = {
    "Nx",
    "Ny",
    "steps",
    "polymer_substeps",
    "nu_s",
    "nu_p",
    "Fx_body",
    "lambda",
    "bsd_fraction",
    "min_c_eig",
    "max_speed",
    "max_abs_psi",
    "max_abs_tau",
    "rho_min",
    "rho_max",
    "max_rel_error",
    "first_nonfinite_step",
}


def latest_run_dir(root: Path) -> Path:
    candidates = [p for p in root.iterdir() if p.is_dir() and (p / "summary.csv").is_file()]
    if not candidates:
        raise FileNotFoundError(f"no validation output with summary.csv found under {root}")
    return sorted(candidates)[-1]


def read_summary(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in SUMMARY_COLUMNS_NUMERIC:
            if key in row and row[key] != "":
                try:
                    if key in {"Nx", "Ny", "steps", "polymer_substeps", "first_nonfinite_step"}:
                        row[key] = int(float(row[key]))
                    else:
                        row[key] = float(row[key])
                except ValueError:
                    pass
    return rows


def pvd_snapshot_path(pvd_path: Path) -> Path:
    tree = ET.parse(pvd_path)
    root = tree.getroot()
    dataset = root.find(".//DataSet")
    if dataset is None or "file" not in dataset.attrib:
        raise ValueError(f"no DataSet file entry found in {pvd_path}")
    return (pvd_path.parent / dataset.attrib["file"]).resolve()


def read_vtr(path: Path) -> dict[str, object]:
    reader = vtk.vtkXMLRectilinearGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = reader.GetOutput()
    dims = grid.GetDimensions()
    nx = dims[0] - 1
    ny = dims[1] - 1
    if nx <= 0 or ny <= 0:
        raise ValueError(f"expected a 2D rectilinear grid in {path}, got dimensions {dims}")

    x = vtk_to_numpy(grid.GetXCoordinates()).astype(float)
    y = vtk_to_numpy(grid.GetYCoordinates()).astype(float)
    cell_data = grid.GetCellData()
    fields: dict[str, np.ndarray] = {}
    for k in range(cell_data.GetNumberOfArrays()):
        array = cell_data.GetArray(k)
        if array is None:
            continue
        values = vtk_to_numpy(array)
        if values.ndim != 1:
            continue
        fields[array.GetName()] = values.reshape((nx, ny), order="F").astype(float)
    return {
        "path": path,
        "nx": nx,
        "ny": ny,
        "x": x,
        "y": y,
        "xc": 0.5 * (x[:-1] + x[1:]),
        "yc": 0.5 * (y[:-1] + y[1:]),
        "fields": fields,
    }


def field_range(field: np.ndarray, solid: np.ndarray | None) -> tuple[float, float]:
    if solid is not None:
        values = field[solid < 0.5]
        if values.size:
            return float(np.nanmin(values)), float(np.nanmax(values))
    return float(np.nanmin(field)), float(np.nanmax(field))


def draw_mesh(ax, x: np.ndarray, y: np.ndarray, nx: int, ny: int, color: str = "#23272e") -> None:
    max_lines = 70
    sx = max(1, math.ceil((nx + 1) / max_lines))
    sy = max(1, math.ceil((ny + 1) / max_lines))
    ax.vlines(x[::sx], y[0], y[-1], color=color, linewidth=0.22, alpha=0.35)
    ax.hlines(y[::sy], x[0], x[-1], color=color, linewidth=0.22, alpha=0.35)


def plot_case_maps(case: str, data: dict[str, object], asset_dir: Path) -> Path:
    fields: dict[str, np.ndarray] = data["fields"]  # type: ignore[assignment]
    x = data["x"]  # type: ignore[assignment]
    y = data["y"]  # type: ignore[assignment]
    nx = int(data["nx"])
    ny = int(data["ny"])
    speed = fields.get("speed")
    if speed is None:
        speed = np.hypot(fields["ux"], fields["uy"])
    rho = fields["rho"]
    solid = fields.get("is_solid")

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)
    panels = [
        ("|u|", speed, "magma"),
        ("rho", rho, "viridis"),
        ("mesh + solid", solid if solid is not None else np.zeros_like(rho), "Greys"),
    ]
    for ax, (title, field, cmap) in zip(axes, panels):
        vmin, vmax = field_range(field, solid if title != "mesh + solid" else None)
        pcm = ax.pcolormesh(x, y, field.T, shading="flat", cmap=cmap, vmin=vmin, vmax=vmax)
        draw_mesh(ax, x, y, nx, ny)
        if solid is not None and np.nanmax(solid) > 0 and title != "mesh + solid":
            xc = data["xc"]  # type: ignore[assignment]
            yc = data["yc"]  # type: ignore[assignment]
            ax.contour(xc, yc, solid.T, levels=[0.5], colors="white", linewidths=1.0)
            ax.contour(xc, yc, solid.T, levels=[0.5], colors="black", linewidths=0.35)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(pcm, ax=ax, shrink=0.84)
    out = asset_dir / f"{case}_maps.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def poiseuille_reference(row: dict[str, object], ny: int) -> np.ndarray:
    fx = float(row.get("Fx_body", 1.0e-5) or 1.0e-5)
    nu_total = float(row["nu_s"]) + float(row["nu_p"])
    j = np.arange(1, ny + 1, dtype=float)
    return fx / (2.0 * nu_total) * (j - 0.5) * (ny + 0.5 - j)


def plot_poiseuille_profile(row: dict[str, object], data: dict[str, object], asset_dir: Path) -> tuple[Path, float, float]:
    fields: dict[str, np.ndarray] = data["fields"]  # type: ignore[assignment]
    yc = data["yc"]  # type: ignore[assignment]
    ux_mean = fields["ux"].mean(axis=0)
    rho_mean = fields["rho"].mean(axis=0)
    ux_ref = poiseuille_reference(row, len(yc))
    err = ux_mean - ux_ref
    rel_l2 = float(np.linalg.norm(err) / max(np.linalg.norm(ux_ref), np.finfo(float).eps))
    linf = float(np.max(np.abs(err)))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    axes[0].plot(ux_mean, yc, "o-", label="Kraken mean ux", linewidth=1.7, markersize=4)
    axes[0].plot(ux_ref, yc, "--", label="analytical", linewidth=1.7)
    axes[0].set_xlabel("ux")
    axes[0].set_ylabel("y")
    axes[0].set_title(f"Poiseuille profile: rel L2={rel_l2:.3e}")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(rho_mean, yc, "o-", label="Kraken mean rho", linewidth=1.7, markersize=4)
    axes[1].axvline(1.0, color="black", linestyle="--", linewidth=1.3, label="analytical rho=1")
    axes[1].set_xlabel("rho")
    axes[1].set_ylabel("y")
    axes[1].set_title("Density profile")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    out = asset_dir / "poiseuille_profile.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out, rel_l2, linf


def image_data_uri(path: Path) -> str:
    raw = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{raw}"


def fluidfoam_status() -> str:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import fluidfoam  # noqa: F401

        return "available"
    except Exception as exc:  # pragma: no cover - informational only
        return f"unavailable: {exc}"


def plot_fluidfoam_case(args: argparse.Namespace, asset_dir: Path) -> Path | None:
    if args.foam_case is None:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import fluidfoam

        x, y, _ = fluidfoam.readmesh(args.foam_case, time_name=args.foam_time, verbose=False)
        u = fluidfoam.readfield(args.foam_case, time_name=args.foam_time, name=args.foam_u_field, verbose=False)
        if u.ndim == 2 and u.shape[0] >= 2:
            speed = np.hypot(u[0], u[1])
        else:
            speed = np.asarray(u, dtype=float).ravel()
        fig, ax = plt.subplots(figsize=(6.5, 4.7), constrained_layout=True)
        sc = ax.scatter(x, y, c=speed, s=4, cmap="magma")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"fluidfoam {args.foam_u_field} magnitude")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(sc, ax=ax, shrink=0.86)
        out = asset_dir / "fluidfoam_u_mesh.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        return out
    except Exception as exc:
        print(f"[dashboard] fluidfoam panel skipped: {exc}")
        return None


def html_table(rows: list[dict[str, object]]) -> str:
    cols = [
        "case",
        "status",
        "backend",
        "Nx",
        "Ny",
        "steps",
        "min_c_eig",
        "max_speed",
        "rho_min",
        "rho_max",
        "max_rel_error",
    ]
    parts = ["<table><thead><tr>"]
    for col in cols:
        parts.append(f"<th>{html.escape(col)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        status = str(row.get("status", ""))
        tr_class = "pass" if status == "pass" else "fail"
        parts.append(f'<tr class="{tr_class}">')
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                text = f"{value:.6e}"
            else:
                text = str(value)
            parts.append(f"<td>{html.escape(text)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def build_dashboard(args: argparse.Namespace) -> Path:
    run_dir = Path(args.run_dir).resolve() if args.run_dir else latest_run_dir(Path(args.root).resolve())
    summary_path = run_dir / "summary.csv"
    rows = read_summary(summary_path)
    asset_dir = run_dir / "dashboard_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)

    case_images: list[tuple[str, Path]] = []
    data_by_case: dict[str, dict[str, object]] = {}
    for row in rows:
        case = str(row["case"])
        pvd = Path(str(row["pvd"]))
        if not pvd.is_absolute():
            pvd = run_dir / pvd
        elif not pvd.exists():
            local_pvd = run_dir / case / pvd.name
            if local_pvd.exists():
                pvd = local_pvd
        vtr = pvd_snapshot_path(pvd)
        data = read_vtr(vtr)
        data_by_case[case] = data
        case_images.append((case, plot_case_maps(case, data, asset_dir)))

    profile_html = ""
    if "poiseuille_coupled" in data_by_case:
        row = next(r for r in rows if str(r["case"]) == "poiseuille_coupled")
        profile_path, rel_l2, linf = plot_poiseuille_profile(row, data_by_case["poiseuille_coupled"], asset_dir)
        profile_html = f"""
        <section>
          <h2>Poiseuille Profile Vs Analytical</h2>
          <p>relative L2 = <strong>{rel_l2:.6e}</strong>, Linf = <strong>{linf:.6e}</strong></p>
          <img src="{image_data_uri(profile_path)}" alt="Poiseuille profile vs analytical">
        </section>
        """

    foam_image = plot_fluidfoam_case(args, asset_dir)
    foam_html = ""
    if foam_image is not None:
        foam_html = f"""
        <section>
          <h2>fluidfoam/OpenFOAM Mesh And U</h2>
          <img src="{image_data_uri(foam_image)}" alt="fluidfoam OpenFOAM mesh and U magnitude">
        </section>
        """

    case_sections = []
    for case, image in case_images:
        pvd = next(str(r["pvd"]) for r in rows if str(r["case"]) == case)
        case_sections.append(
            f"""
            <section>
              <h2>{html.escape(case)}</h2>
              <p class="path">PVD: {html.escape(pvd)}</p>
              <img src="{image_data_uri(image)}" alt="{html.escape(case)} maps">
            </section>
            """
        )

    css = """
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1e2430; background: #f4f6f8; }
    main { max-width: 1280px; margin: 0 auto; padding: 28px; }
    header { margin-bottom: 20px; }
    h1 { margin: 0 0 8px; font-size: 30px; letter-spacing: 0; }
    h2 { margin: 0 0 10px; font-size: 20px; letter-spacing: 0; }
    section { background: white; border: 1px solid #d9dee7; border-radius: 8px; padding: 18px; margin: 18px 0; box-shadow: 0 1px 2px rgba(20, 30, 45, 0.05); }
    img { width: 100%; max-width: 100%; height: auto; display: block; }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td { padding: 8px 10px; border-bottom: 1px solid #e3e7ee; text-align: left; white-space: nowrap; }
    th { background: #eef2f6; }
    tr.pass td:first-child::before { content: "OK "; color: #16743a; font-weight: 700; }
    tr.fail td:first-child::before { content: "FAIL "; color: #b00020; font-weight: 700; }
    .path { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; color: #4e5969; overflow-wrap: anywhere; }
    .meta { color: #4e5969; }
    """

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Kraken Log-FV Simple Validation Dashboard</title>
  <style>{css}</style>
</head>
<body>
<main>
  <header>
    <h1>Kraken Log-FV Simple Validation Dashboard</h1>
    <p class="meta">Run directory: <span class="path">{html.escape(str(run_dir))}</span></p>
    <p class="meta">fluidfoam: {html.escape(fluidfoam_status())}</p>
  </header>
  <section>
    <h2>Summary</h2>
    {html_table(rows)}
  </section>
  {profile_html}
  {''.join(case_sections)}
  {foam_html}
</main>
</body>
</html>
"""
    out = Path(args.out).resolve() if args.out else run_dir / "dashboard.html"
    out.write_text(html_text, encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a static dashboard for Kraken log-FV validation outputs.")
    parser.add_argument("run_dir", nargs="?", help="Validation output directory containing summary.csv.")
    parser.add_argument("--root", default="tmp/logfv_simple_validation_outputs", help="Root used to auto-pick latest run.")
    parser.add_argument("--out", help="Dashboard HTML path. Defaults to <run_dir>/dashboard.html.")
    parser.add_argument("--foam-case", help="Optional OpenFOAM case directory to visualize through fluidfoam.")
    parser.add_argument("--foam-time", default="latestTime", help="OpenFOAM time for fluidfoam reads.")
    parser.add_argument("--foam-u-field", default="U", help="OpenFOAM velocity field name.")
    return parser.parse_args()


def main() -> None:
    dashboard = build_dashboard(parse_args())
    print(f"Dashboard: {dashboard}")


if __name__ == "__main__":
    main()
