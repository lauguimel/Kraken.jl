# Shared environment for the inverse-rheology GNF jobs on Aqua (sourced by the PBS files).
# Julia 1.11.9 (not 1.12: Enzyme + weakdeps landmine), dedicated depot stacked
# on the depot of the 2026-10-12 Guo remeasure (same Manifest, read-only reuse;
# ~/.julia is avoided: a stuck Lustre lock under ~/.julia/packages hung a job).
#
# Every path can be overridden by setting the variable before sourcing (or
# with qsub -v). Defaults:
#   REPO              $HOME/Kraken.jl-inverse
#   RUNS              $HOME/inverse_rheology_runs          (outputs, never in the repo)
#   JL                $HOME/.julia/juliaup/julia-1.11.9+0.x64.linux.gnu/bin/julia
#   JULIA_DEPOT_PATH  $RUNS/depot:$HOME/kraken_guo_remeasure/depot2
#   KRAKEN_REFERENCE_DATA  (unset)  optional reference data; fixture tests are skipped without it
export REPO="${REPO:-$HOME/Kraken.jl-inverse}"
export RUNS="${RUNS:-$HOME/inverse_rheology_runs}"
export JL="${JL:-$HOME/.julia/juliaup/julia-1.11.9+0.x64.linux.gnu/bin/julia}"
export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$RUNS/depot:$HOME/kraken_guo_remeasure/depot2}"
export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_CUDA_USE_COMPAT=false
