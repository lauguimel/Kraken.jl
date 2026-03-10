#!/bin/bash
# =============================================================================
# Kraken.jl — Cross-solver benchmark runner
# Builds Docker images for Gerris, Basilisk, and OpenFOAM, runs benchmarks,
# and collects JSON results in benchmarks/external/results/
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCKER_DIR="$SCRIPT_DIR/docker"
RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "  Kraken.jl — Cross-Solver Benchmark Suite"
echo "============================================================"
echo ""

# Track which solvers succeeded
GERRIS_OK=false
BASILISK_OK=false
OPENFOAM_OK=false

# --- Gerris ---
echo "--- [1/3] Gerris (Ubuntu 18.04) ---"
if docker build -t kraken-bench-gerris "$DOCKER_DIR/gerris" 2>&1 | tail -5; then
    echo "  Image built. Running benchmarks..."
    if docker run --rm kraken-bench-gerris > "$RESULTS_DIR/gerris.json" 2>"$RESULTS_DIR/gerris.log"; then
        echo "  Results saved to results/gerris.json"
        GERRIS_OK=true
    else
        echo "  WARNING: Gerris benchmark failed. See results/gerris.log"
    fi
else
    echo "  WARNING: Gerris Docker build failed."
fi
echo ""

# --- Basilisk ---
echo "--- [2/3] Basilisk (Ubuntu 22.04) ---"
if docker build -t kraken-bench-basilisk "$DOCKER_DIR/basilisk" 2>&1 | tail -5; then
    echo "  Image built. Running benchmarks..."
    if docker run --rm kraken-bench-basilisk > "$RESULTS_DIR/basilisk.json" 2>"$RESULTS_DIR/basilisk.log"; then
        echo "  Results saved to results/basilisk.json"
        BASILISK_OK=true
    else
        echo "  WARNING: Basilisk benchmark failed. See results/basilisk.log"
    fi
else
    echo "  WARNING: Basilisk Docker build failed."
fi
echo ""

# --- OpenFOAM ---
echo "--- [3/3] OpenFOAM v10 (Ubuntu 22.04) ---"
if docker build -t kraken-bench-openfoam "$DOCKER_DIR/openfoam" 2>&1 | tail -5; then
    echo "  Image built. Running benchmarks..."
    if docker run --rm kraken-bench-openfoam > "$RESULTS_DIR/openfoam.json" 2>"$RESULTS_DIR/openfoam.log"; then
        echo "  Results saved to results/openfoam.json"
        OPENFOAM_OK=true
    else
        echo "  WARNING: OpenFOAM benchmark failed. See results/openfoam.log"
    fi
else
    echo "  WARNING: OpenFOAM Docker build failed."
fi
echo ""

# --- Summary ---
echo "============================================================"
echo "  Summary"
echo "============================================================"
echo "  Gerris:   $( $GERRIS_OK && echo 'OK' || echo 'FAILED' )"
echo "  Basilisk: $( $BASILISK_OK && echo 'OK' || echo 'FAILED' )"
echo "  OpenFOAM: $( $OPENFOAM_OK && echo 'OK' || echo 'FAILED' )"
echo ""
echo "  Results directory: $RESULTS_DIR/"
echo ""

# List result files
if ls "$RESULTS_DIR"/*.json 1>/dev/null 2>&1; then
    echo "  JSON result files:"
    for f in "$RESULTS_DIR"/*.json; do
        SIZE=$(wc -c < "$f" | tr -d ' ')
        echo "    $(basename "$f") ($SIZE bytes)"
    done
else
    echo "  No results collected."
fi

echo ""
echo "  Next step: julia --project=benchmarks benchmarks/compare_solvers.jl"
echo "============================================================"
