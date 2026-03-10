#!/bin/bash
# Run all Gerris benchmark cases and output JSON results to stdout
# All diagnostic/progress output goes to stderr

set -e

cd /benchmarks/cases

# --- Lid-driven cavity Re=100, 64x64 ---
echo "=== Running cavity Re=100 64x64 ===" >&2

# Run with timing (time output goes to stderr)
TIMEFORMAT='%R'
START_TIME=$(date +%s.%N)
gerris2D cavity.gfs 2>cavity_stderr.log
END_TIME=$(date +%s.%N)

# Compute elapsed time
ELAPSED=$(python3 -c "print(f'{${END_TIME} - ${START_TIME}:.3f}')")
echo "Cavity completed in ${ELAPSED}s" >&2

# Extract Gerris version
GERRIS_VERSION=$(gerris2D --version 2>&1 | head -1 || echo "unknown")
echo "Gerris version: ${GERRIS_VERSION}" >&2

# Extract results and produce JSON
python3 extract_results.py \
    --solver-version "${GERRIS_VERSION}" \
    --cavity-time "${ELAPSED}" \
    --cavity-profile results/cavity_profile.dat
