#!/usr/bin/env bash
# Push the local Kraken.jl tree to AQUA via rsync.
#
# AQUA convention: ~/Kraken.jl is rsync'd from the laptop, not git-cloned.
# This script keeps the layout in sync while excluding heavy artefacts
# and machine-local files.
#
# Usage:
#     hpc/sync_to_aqua.sh           # dry-run, prints what would change
#     hpc/sync_to_aqua.sh --apply   # actually transfer the files

set -euo pipefail

REMOTE="aqua:Kraken.jl/"
SRC="$(cd "$(dirname "$0")/.." && pwd)/"

DRY_RUN="--dry-run"
if [[ "${1:-}" == "--apply" ]]; then
    DRY_RUN=""
fi

# Excludes: machine-local artefacts, build outputs, heavy generated files,
# .git is also excluded because AQUA doesn't track history (rsync workflow).
rsync -avz $DRY_RUN \
    --delete \
    --exclude '.git/' \
    --exclude '.claude/' \
    --exclude '.vscode/' \
    --exclude 'docs/' \
    --exclude 'output/' \
    --exclude 'results/' \
    --exclude 'Manifest.toml' \
    --exclude '*.jl.cov' \
    --exclude '*.jl.mem' \
    --exclude '.DS_Store' \
    --exclude 'benchmarks/results/*.csv' \
    --exclude 'benchmarks/results/*.png' \
    --exclude 'PLAN.md' \
    --exclude 'PLAN_old.md' \
    --exclude 'AUDIT.md' \
    "$SRC" "$REMOTE"

echo
if [[ -z "$DRY_RUN" ]]; then
    echo "Done. Files synced to aqua:~/Kraken.jl/"
    echo "Next step:"
    echo "    ssh aqua 'cd ~/Kraken.jl && qsub hpc/aqua_perf_h100.pbs'"
else
    echo "Dry-run only. Re-run with --apply to actually sync."
fi
