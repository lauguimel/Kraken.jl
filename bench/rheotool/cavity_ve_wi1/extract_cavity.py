"""Extract the cavity comparison quantities from a solved RheoTool run.

Reads the cell-centre fields written by rheoFoam (``<t>/U[.gz]``,
``<t>/tau[.gz]``) and the ``kinEner.txt`` log of the case, and writes three
plain CSV files (one header line each):

- ``<prefix>_energy.csv``   t, mean kinetic energy, mean elastic energy
  (as logged by the coded function object every 20 steps);
- ``<prefix>_profiles.csv`` long format, one row per (t, line, coordinate):
  u_x, u_y and, for a viscoelastic run, tau_p and C = I + (lambda/eta_p) tau_p,
  along x = 0.5 (vs y) and y = 0.5, 0.75, 0.9, 0.95 (vs x);
- ``<prefix>_vortex.csv``   t, primary-vortex centre and stream-function minimum.

Line values are linear interpolations between the two cell-centre rows (or
columns) around the line, so a Kraken field sampled at the same cell centres
can be reduced with the identical rule. The mesh is the tutorial's single
hex block (N x N x 1), whose cells are ordered x fastest, then y; the cell
centres are ((i + 1/2)/N, (j + 1/2)/N).

Usage:
    python extract_cavity.py <case_dir> <N> <out_prefix> [--lambda 1 --etaP 0.5]
"""

import argparse
import gzip
import os
import re

import numpy as np

HORIZONTAL_LINES = (0.5, 0.75, 0.9, 0.95)
VERTICAL_LINES = (0.5,)


def read_internal_field(path, ncomp):
    """Return the internalField of an ascii volField as an (ncells, ncomp) array."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        text = fh.read()
    body = text[text.index("internalField"):]
    if body.startswith("internalField   uniform") or re.match(r"internalField\s+uniform", body):
        raise ValueError(f"{path}: uniform internalField, no solved data")
    m = re.match(r"internalField\s+nonuniform\s+List<\w+>\s*(\d+)\s*\(", body)
    if m is None:
        raise ValueError(f"{path}: cannot parse internalField header")
    n = int(m.group(1))
    start = m.end()
    end = body.index("\n)", start)
    values = np.array(body[start:end].replace("(", " ").replace(")", " ").split(), dtype=float)
    if values.size != n * ncomp:
        raise ValueError(f"{path}: expected {n * ncomp} values, read {values.size}")
    return values.reshape(n, ncomp)


def field_path(case, tdir, name):
    for candidate in (name + ".gz", name):
        p = os.path.join(case, tdir, candidate)
        if os.path.exists(p):
            return p
    return None


def time_dirs(case):
    out = []
    for d in os.listdir(case):
        try:
            t = float(d)
        except ValueError:
            continue
        if t > 0 and field_path(case, d, "U") is not None:
            out.append((t, d))
    return sorted(out)


def interp_rows(field, coord, n):
    """Linear interpolation across the row index at physical coordinate `coord`.

    `field` is (N_row, ...) indexed by the cell index normal to the line.
    """
    s = coord * n - 0.5
    j0 = int(np.floor(s))
    w = s - j0
    return (1.0 - w) * field[j0] + w * field[j0 + 1]


def stream_function(ux, n):
    """psi(x, y) = int_0^y u_x dy' on cell centres, midpoint rule from the bottom wall."""
    dy = 1.0 / n
    return np.cumsum(ux, axis=0) * dy - 0.5 * ux * dy


def vortex_centre(psi, n):
    """Location and value of min(psi), refined by a 1D parabola in each direction."""
    j, i = np.unravel_index(np.argmin(psi), psi.shape)
    x = (i + 0.5) / n
    y = (j + 0.5) / n
    if 0 < i < n - 1:
        a, b, c = psi[j, i - 1], psi[j, i], psi[j, i + 1]
        den = a - 2 * b + c
        if den != 0:
            x += 0.5 * (a - c) / den / n
    if 0 < j < n - 1:
        a, b, c = psi[j - 1, i], psi[j, i], psi[j + 1, i]
        den = a - 2 * b + c
        if den != 0:
            y += 0.5 * (a - c) / den / n
    return x, y, psi[j, i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("N", type=int)
    ap.add_argument("prefix")
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0)
    ap.add_argument("--etaP", type=float, default=0.5)
    args = ap.parse_args()
    n = args.N

    kin = np.loadtxt(os.path.join(args.case, "kinEner.txt"), ndmin=2)
    np.savetxt(args.prefix + "_energy.csv", kin[:, :3], delimiter=",",
               header="t,kinetic_energy_mean,elastic_energy_mean", comments="", fmt="%.12g")

    prof_rows, vort_rows = [], []
    for t, d in time_dirs(args.case):
        U = read_internal_field(field_path(args.case, d, "U"), 3).reshape(n, n, 3)  # [j, i, c]
        tau_path = field_path(args.case, d, "tau")
        tau = None
        if tau_path is not None:
            tau = read_internal_field(tau_path, 6).reshape(n, n, 6)  # xx xy xz yy yz zz
        centres = (np.arange(n) + 0.5) / n

        def emit(line, coord_vals, uvals, tvals):
            for k, c in enumerate(coord_vals):
                row = [t, line, c, uvals[k, 0], uvals[k, 1]]
                if tvals is not None:
                    txx, txy, tyy = tvals[k, 0], tvals[k, 1], tvals[k, 3]
                    f = args.lam / args.etaP
                    row += [txx, txy, tyy, 1.0 + f * txx, f * txy, 1.0 + f * tyy]
                else:
                    row += [""] * 6
                prof_rows.append(row)

        for xl in VERTICAL_LINES:
            emit(f"x={xl}", centres, interp_rows(np.swapaxes(U, 0, 1), xl, n),
                 None if tau is None else interp_rows(np.swapaxes(tau, 0, 1), xl, n))
        for yl in HORIZONTAL_LINES:
            emit(f"y={yl}", centres, interp_rows(U, yl, n),
                 None if tau is None else interp_rows(tau, yl, n))

        psi = stream_function(U[:, :, 0], n)
        xc, yc, pmin = vortex_centre(psi, n)
        top_flux = np.max(np.abs(psi[-1] + 0.5 * U[-1, :, 0] / n))
        vort_rows.append([t, xc, yc, pmin, top_flux])

    header = "t,line,coord,ux,uy,tau_xx,tau_xy,tau_yy,C_xx,C_xy,C_yy"
    with open(args.prefix + "_profiles.csv", "w") as fh:
        fh.write(header + "\n")
        for r in prof_rows:
            fh.write(",".join(r_ if isinstance(r_, str) else f"{r_:.12g}" for r_ in r) + "\n")
    np.savetxt(args.prefix + "_vortex.csv", np.array(vort_rows), delimiter=",",
               header="t,x_centre,y_centre,psi_min,column_flux_residual", comments="", fmt="%.12g")


if __name__ == "__main__":
    main()
