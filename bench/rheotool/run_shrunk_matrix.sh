#!/usr/bin/env bash
# ============================================================================
# run_shrunk_matrix.sh — drive the 3 shrunk rheoTool cases (M32 Phase 2)
# ----------------------------------------------------------------------------
# Cases:
#   - cylinder_newtonian_re1_shrunk15R   (Re=1, no polymer, gate G3)
#   - cylinder_wi0.1_shrunk15R           (Wi=0.1, Re=1, Oldroyd-B log)
#   - cylinder_wi1.0_shrunk15R           (Wi=1.0, Re=1, Oldroyd-B log, endTime=20)
#
# Geometry (matches Kraken bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl):
#   L_up = 15 R, L_down = 15 R, channel half-height = 2 R (blockage = R/2R = 0.5)
#
# Mesh delta vs original cylinder_wi1.0:
#   blockMesh cells: 12447 -> 10247 (-17.7%)
#   post-mirror:     24894 -> 20494 (-17.7%)
#
# Run mode: local OpenFOAM via Docker (image guiguitcho/openfoam9-rheotool:v1.2).
# rheoTool is NOT available as Apptainer SIF on Aqua (no SIF in
# ~/Documents/Clouds/UGA/Recherche/HPC/aqua/) — local execution only.
#
# Expected walltime per case (workstation, 1 cpu, ~10k cells):
#   - Newtonian (endTime=10, dt=2e-2 -> 500 steps):  ~5-15 min
#   - Wi=0.1   (endTime=10, dt=2e-2 -> 500 steps):   ~15-40 min
#   - Wi=1.0   (endTime=20, dt=1e-2 -> 2000 steps):  ~1.5-2.5 h
# Total matrix:                                        ~2-3 h sequential.
# Compared to the original L_down=60R case (~2-3 h alone for Wi=1.0 at endTime=10),
# this is ~25-35% wall-clock savings due to fewer cells per timestep,
# offset by the doubled endTime for Wi=1.0.
#
# Usage:
#   bash run_shrunk_matrix.sh             # run all 3 cases sequentially
#   bash run_shrunk_matrix.sh newtonian   # run only Newtonian
#   bash run_shrunk_matrix.sh wi0.1       # run only Wi=0.1
#   bash run_shrunk_matrix.sh wi1.0       # run only Wi=1.0
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CASES=(
    "cylinder_newtonian_re1_shrunk15R"
    "cylinder_wi0.1_shrunk15R"
    "cylinder_wi1.0_shrunk15R"
)

run_case() {
    local case_dir="$1"
    local case_path="${ROOT}/${case_dir}"
    if [ ! -d "${case_path}" ]; then
        echo "[ERROR] Missing case dir: ${case_path}"
        return 1
    fi
    echo "==============================================="
    echo "[$(date '+%F %T')] Running ${case_dir}"
    echo "==============================================="
    pushd "${case_path}" > /dev/null
    bash run_docker.sh
    popd > /dev/null
    echo "[$(date '+%F %T')] Finished ${case_dir}"
}

# Selector
case "${1:-all}" in
    all)
        for c in "${CASES[@]}"; do run_case "${c}"; done
        ;;
    newtonian)
        run_case "cylinder_newtonian_re1_shrunk15R"
        ;;
    wi0.1)
        run_case "cylinder_wi0.1_shrunk15R"
        ;;
    wi1.0)
        run_case "cylinder_wi1.0_shrunk15R"
        ;;
    *)
        echo "Usage: $0 [all|newtonian|wi0.1|wi1.0]"
        exit 1
        ;;
esac

echo
echo "[$(date '+%F %T')] M32 Phase 2 shrunk matrix complete."
echo "Summarise Cd via:"
for c in "${CASES[@]}"; do
    echo "  bash ${ROOT}/${c}/summarize_cd.sh"
done
