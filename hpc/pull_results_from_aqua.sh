#!/usr/bin/env bash
# Pull benchmark results from AQUA back to the local repo.
#
# After a PBS job finishes, the CSVs and any generated PNGs live in
# ~/Kraken.jl/benchmarks/results/ on AQUA. This script copies them
# back to benchmarks/results/ locally so the documentation pipeline
# (view/generate_doc_figures.jl, docs/src/benchmarks/*.md) sees them.
#
# Usage:
#     hpc/pull_results_from_aqua.sh

set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/benchmarks/results/"
mkdir -p "$DEST"

rsync -avz \
    --include='*.csv' \
    --include='*.png' \
    --include='*.svg' \
    --include='*.log' \
    --exclude='*' \
    'aqua:Kraken.jl/benchmarks/results/' "$DEST"

echo
echo "Pulled into $DEST"
ls -la "$DEST"
