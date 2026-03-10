#!/usr/bin/env python3
"""Extract OpenFOAM benchmark results and output JSON to stdout.

Compatible with Python 3.6+ (no | type union syntax).
Reads U field directly from latest time directory if postProcess fails.
"""

import json
import os
import re
import sys
from pathlib import Path


def find_latest_time_dir(case_dir):
    """Find the latest time directory in an OpenFOAM case."""
    time_dirs = []
    for d in case_dir.iterdir():
        if d.is_dir():
            try:
                t = float(d.name)
                if t > 0:
                    time_dirs.append((t, d))
            except ValueError:
                continue
    if not time_dirs:
        return None
    time_dirs.sort(key=lambda x: x[0], reverse=True)
    return time_dirs[0][1]


def read_of_vector_field(filepath):
    """Parse an OpenFOAM volVectorField file and return a list of (Ux, Uy, Uz) tuples."""
    vectors = []
    with open(filepath) as f:
        content = f.read()

    # Find the internalField section — use greedy match up to closing ");'
    match = re.search(r'internalField\s+nonuniform\s+List<vector>\s+(\d+)\s*\((.*)\);',
                      content, re.DOTALL)
    if not match:
        return vectors

    n = int(match.group(1))
    data = match.group(2)

    # Parse (Ux Uy Uz) entries
    for m in re.finditer(r'\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)', data):
        vectors.append((float(m.group(1)), float(m.group(2)), float(m.group(3))))

    return vectors


def extract_centerline_from_field(case_dir, N=64):
    """Extract u(y) at x=0.5 from the U field directly.

    For a 64x64 mesh, cell centers are at (i+0.5)/N for i=0..N-1.
    The centerline x=0.5 corresponds to column i=31 (0-indexed) or i=32 (1-indexed center = 0.5078).
    """
    latest = find_latest_time_dir(case_dir)
    if latest is None:
        print("WARNING: No time directory found", file=sys.stderr)
        return [], []

    u_file = latest / "U"
    if not u_file.exists():
        print("WARNING: No U file in " + str(latest), file=sys.stderr)
        return [], []

    print("Reading U from: " + str(u_file), file=sys.stderr)
    vectors = read_of_vector_field(u_file)

    if len(vectors) != N * N:
        print("WARNING: Expected {} vectors, got {}".format(N * N, len(vectors)), file=sys.stderr)
        if len(vectors) == 0:
            return [], []
        # Try to figure out N from the data
        import math
        N = int(math.sqrt(len(vectors)))

    # OpenFOAM blockMesh with hex ordering: cell (i,j) = vectors[j*N + i]
    # Centerline: i = N//2 (x = (N//2 + 0.5) / N)
    i_mid = N // 2
    dx = 1.0 / N

    y_vals = []
    u_vals = []
    for j in range(N):
        idx = j * N + i_mid
        if idx < len(vectors):
            y_vals.append((j + 0.5) * dx)
            u_vals.append(vectors[idx][0])  # Ux component

    return y_vals, u_vals


def read_postprocess_profile(case_dir):
    """Try to read from postProcessing directory."""
    for base_name in ["sampleDict", "sets"]:
        postproc_base = case_dir / "postProcessing" / base_name
        if not postproc_base.exists():
            continue

        # Find latest time
        time_dirs = []
        for d in postproc_base.iterdir():
            if d.is_dir():
                try:
                    time_dirs.append((float(d.name), d))
                except ValueError:
                    continue
        if not time_dirs:
            continue

        time_dirs.sort(key=lambda x: x[0], reverse=True)
        latest = time_dirs[0][1]

        # Find centerline file
        for f in latest.iterdir():
            if "centerline" in f.name.lower():
                y_vals, u_vals = [], []
                with open(f) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = line.split()
                        if len(parts) >= 2:
                            y_vals.append(float(parts[0]))
                            u_vals.append(float(parts[1]))
                if y_vals:
                    return y_vals, u_vals

    return [], []


def main():
    if len(sys.argv) < 2:
        print("Usage: extract_results.py <cavity_time_secs>", file=sys.stderr)
        sys.exit(1)

    cavity_time = float(sys.argv[1])
    case_dir = Path("/benchmarks/cases/cavity")

    # Try postProcess first, then direct field reading
    y_profile, u_profile = read_postprocess_profile(case_dir)

    if not y_profile:
        print("Falling back to direct field reading...", file=sys.stderr)
        y_profile, u_profile = extract_centerline_from_field(case_dir, N=64)

    print("Extracted {} profile points".format(len(y_profile)), file=sys.stderr)

    results = {
        "solver": "openfoam",
        "version": "v10",
        "cases": {
            "cavity": {
                "time_seconds": cavity_time,
                "N": 64,
                "Re": 100,
                "y_profile": y_profile,
                "u_profile": u_profile,
            }
        },
    }

    json.dump(results, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
