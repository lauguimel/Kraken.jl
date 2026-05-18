#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 1

HARNESS="bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl"
PBS_FILE="bench/viscoelastic_logfv/run_cyl_bigsweep_v2_a100.pbs"
RHEOTOOL_MESH="bench/rheotool/cylinder_wi0.1/system/blockMeshDict"
PLAN_FILE="$(find bench/viscoelastic_audit -maxdepth 1 -name 'CYLINDER_DOE_PLAN_*_20260518.md' -print | sort | head -n 1)"

exit_code=0
missing_env=()

check_file() {
    local path="$1"
    if [[ -f "$path" ]]; then
        echo "[PASS] file $path"
    else
        echo "[MISS] file $path"
        exit_code=1
    fi
}

check_file "$PBS_FILE"
check_file "$HARNESS"
check_file "$RHEOTOOL_MESH"

if [[ -n "$PLAN_FILE" && -s "$PLAN_FILE" ]]; then
    echo "[PASS] DoE plan is non-empty"
else
    echo "[MISS] DoE plan is non-empty"
    exit_code=1
fi

manifest_env_vars=(
    KRAKEN_BACKEND
    KRAKEN_FT
    KRAKEN_BETA_LIST
    KRAKEN_WI_LIST
    KRAKEN_RE_LIST
    KRAKEN_R_LIST
    KRAKEN_BSD_LIST
    KRAKEN_U_MEAN
    KRAKEN_MAX_STEPS_BASE
    KRAKEN_AVG_WINDOW_FRAC
    KRAKEN_L_UP_LIST
    KRAKEN_L_DOWN_LIST
    KRAKEN_EMBEDDED_GRADIENT
    KRAKEN_EMBEDDED_ADVECTION
    KRAKEN_EMBEDDED_FORCE
    KRAKEN_EMBEDDED_DRAG
    KRAKEN_EMBEDDED_GEOMETRY
    KRAKEN_OUTPUT_DIR
)

for var in "${manifest_env_vars[@]}"; do
    if grep -q "$var" "$HARNESS"; then
        echo "[PASS] $var"
    else
        echo "[MISS] $var"
        missing_env+=("$var")
    fi
done

eta_hours() {
    local total="0"
    shift
    while [[ "$#" -gt 0 ]]; do
        local r="${1%%:*}"
        local count="${1##*:}"
        total="$(awk -v acc="$total" -v radius="$r" -v n="$count" \
            'BEGIN { printf "%.6f", acc + n * 15.0 * (radius / 30.0)^2 / 60.0 }')"
        shift
    done
    awk -v h="$total" 'BEGIN { printf "%.1f", h }'
}

tier1_eta="$(eta_hours tier1 30:12)"
tier2_eta="$(eta_hours tier2 50:18)"
tier3_eta="$(eta_hours tier3 30:3 50:3 80:3 100:3)"

echo "Tier 1: 12 cases, ETA ~${tier1_eta}h"
echo "Tier 2: 18 cases, ETA ~${tier2_eta}h"
echo "Tier 3: 12 cases, ETA ~${tier3_eta}h"

echo "Files needing modification:"
echo "- $HARNESS: add env parsing, zipped domain/embedded loops, driver kwargs, summary columns"
echo "- $PBS_FILE: export new env defaults and echo them"

if [[ "${#missing_env[@]}" -gt 0 ]]; then
    printf 'Missing env vars: %s\n' "${missing_env[*]}" >&2
    exit 1
fi

exit "$exit_code"
