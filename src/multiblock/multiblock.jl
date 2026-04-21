# =====================================================================
# Multi-block structured LBM (v0.3).
#
# Entry point that includes all the submodules in their dependency
# order:
#
#   topology.jl  — types (Block, Interface, MultiBlockMesh2D)
#   sanity.jl    — validation checks run before a simulation
#   exchange.jl  — per-timestep ghost-cell copy across interfaces
#                   (added in Phase A.4)
#
# Everything here is plain imperative Julia: structs, functions, no
# macro trickery. The fused-kernel DSL stays at the inner-loop level
# (src/kernels/dsl/); this module is for assembling blocks and
# steering the outer loop.
#
# Design notes for v0.3 MVP:
# - Halo strict: each block stores its own populations; a 1-row ghost
#   buffer is refreshed at the start of each step by copying from the
#   neighbour block. Simpler to reason about than overlap zones.
# - Aligned orientation only: two interface edges must run in the same
#   local direction. Flip/rotation orientations deferred.
# - SLBM/LI-BB unchanged: the existing per-block step kernels are
#   called in a loop over blocks between exchange passes.
# =====================================================================

include("topology.jl")
include("sanity.jl")
include("state.jl")
include("exchange.jl")
include("wall_ghost.jl")
include("mesh_gmsh_multiblock.jl")
include("reorient.jl")
