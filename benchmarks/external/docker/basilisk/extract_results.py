#!/usr/bin/env python3
"""
Extract Basilisk benchmark results and output JSON to stdout.
Usage: python3 extract_results.py <cavity_time_s> <taylor_green_time_s>
"""

import json
import os
import sys


def read_profile(filepath):
    """Read a two-column dat file, skip comment lines and aberrant values."""
    col1, col2 = [], []
    if not os.path.exists(filepath):
        return col1, col2
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                v1, v2 = float(parts[0]), float(parts[1])
                # Filter out Basilisk interpolation artifacts (1e+30 = nodata)
                if abs(v2) > 1e10:
                    continue
                col1.append(v1)
                col2.append(v2)
    return col1, col2


def read_taylor_green_results(filepath):
    """Read taylor_green_results.dat (single data line: L2_error umax)."""
    if not os.path.exists(filepath):
        return None, None
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                return float(parts[0]), float(parts[1])
    return None, None


def get_basilisk_version():
    """Try to get Basilisk version info."""
    version_file = os.path.join(
        os.environ.get("BASILISK", "/opt/basilisk/src"), "..", "VERSION"
    )
    if os.path.exists(version_file):
        with open(version_file) as f:
            return f.read().strip()
    # Fallback: check git or date
    return "unknown"


def main():
    if len(sys.argv) < 3:
        print("Usage: extract_results.py <cavity_time_s> <tg_time_s>", file=sys.stderr)
        sys.exit(1)

    cavity_time = float(sys.argv[1])
    tg_time = float(sys.argv[2])

    cases_dir = "/benchmarks/cases"

    # Cavity profiles
    y_vals, u_vals = read_profile(os.path.join(cases_dir, "cavity_profile.dat"))
    x_vals, v_vals = read_profile(os.path.join(cases_dir, "cavity_profile_v.dat"))

    # Taylor-Green results
    l2_error, umax = read_taylor_green_results(
        os.path.join(cases_dir, "taylor_green_results.dat")
    )

    results = {
        "solver": "basilisk",
        "version": get_basilisk_version(),
        "cases": {
            "cavity": {
                "time_seconds": round(cavity_time, 3),
                "N": 64,
                "Re": 100,
                "level": 6,
                "domain": [0, 1],
                "y_profile": y_vals,
                "u_profile": u_vals,
                "x_profile": x_vals,
                "v_profile": v_vals,
            },
            "taylor_green": {
                "time_seconds": round(tg_time, 3),
                "N": 64,
                "level": 6,
                "nu": 0.01,
                "t_final": 1.0,
                "l2_error": l2_error,
                "umax": umax,
            },
        },
    }

    json.dump(results, sys.stdout, indent=2)
    print()  # trailing newline


if __name__ == "__main__":
    main()
