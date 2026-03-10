#!/usr/bin/env python3
"""
Extract Gerris benchmark results and output JSON to stdout.

Gerris domain is [-0.5, 0.5]^2. We shift coordinates to [0, 1]^2
for comparison with other solvers (Kraken.jl, OpenFOAM, Basilisk).

The cavity profile is extracted along the vertical centerline:
  x=0 in Gerris domain => x=0.5 in [0,1] domain
"""

import argparse
import json
import sys


def parse_gerris_location_output(filepath):
    """
    Parse Gerris OutputLocation file.

    Gerris OutputLocation format (space-separated):
      x y z U V P ...

    Columns depend on the variables defined. For NavierStokes:
      col 0: x
      col 1: y
      col 2: z
      col 3: U (x-velocity)
      col 4: V (y-velocity)
      col 5: P (pressure)

    Returns (y_coords, u_values) shifted to [0,1] domain.
    """
    y_coords = []
    u_values = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                y_gerris = float(parts[1])  # y in [-0.5, 0.5]
                u_vel = float(parts[3])  # U velocity
            except (ValueError, IndexError):
                continue

            # Shift from [-0.5, 0.5] to [0, 1]
            y_shifted = y_gerris + 0.5
            y_coords.append(round(y_shifted, 10))
            u_values.append(round(u_vel, 10))

    # Sort by y coordinate
    paired = sorted(zip(y_coords, u_values), key=lambda p: p[0])
    if paired:
        y_coords, u_values = zip(*paired)
        return list(y_coords), list(u_values)
    return [], []


def main():
    parser = argparse.ArgumentParser(
        description="Extract Gerris benchmark results as JSON"
    )
    parser.add_argument(
        "--solver-version", default="unknown", help="Gerris version string"
    )
    parser.add_argument(
        "--cavity-time", type=float, default=0.0, help="Cavity wall-clock time (s)"
    )
    parser.add_argument(
        "--cavity-profile",
        required=True,
        help="Path to cavity OutputLocation result file",
    )
    args = parser.parse_args()

    # Parse cavity profile
    y_profile, u_profile = parse_gerris_location_output(args.cavity_profile)

    if not y_profile:
        print(
            "WARNING: No data extracted from cavity profile", file=sys.stderr
        )

    results = {
        "solver": "gerris",
        "version": args.solver_version,
        "cases": {
            "cavity": {
                "time_seconds": args.cavity_time,
                "N": 64,
                "Re": 100,
                "y_profile": y_profile,
                "u_profile": u_profile,
            }
        },
    }

    # Output JSON to stdout
    json.dump(results, sys.stdout, indent=2)
    print()  # trailing newline


if __name__ == "__main__":
    main()
