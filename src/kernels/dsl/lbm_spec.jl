# =====================================================================
# Kernel DSL — specification layer.
#
# A fused LBM timestep is decomposed into an ordered sequence of
# *bricks* (pull-stream, collision, BC overwrite, moment computation,
# write-back, …). A spec type encodes this sequence and the stencil;
# `build_lbm_kernel(backend, spec)` compiles the spec into a single
# KernelAbstractions `@kernel` by inlining each brick's code fragment.
#
# Shared local-variable vocabulary across bricks (a brick reads/writes
# by convention; there is no explicit dataflow graph):
#
#   fp1..fp9              pulled populations (post-stream, pre-collision)
#   ρ, ux, uy, usq        moments (pre-collision / current timestep)
#   feq1..feq9            equilibrium populations
#   fp1c..fp9c            post-collision populations
#   fp1_new..fp9_new      post-BC-overwrite populations (if applicable)
#
# Every brick is a singleton `<: LBMBrick` type. Two multimethods are
# dispatched per brick type:
#
#   required_args(::Brick) -> NTuple{N,Symbol}
#       The kernel-level arguments this brick reads. The builder takes
#       the union of these symbols across all bricks in the spec and
#       generates the minimal kernel signature (avoids GPU register
#       pressure from unused parameters).
#
#   emit_code(::Brick)     -> Expr
#       A `quote` block that assumes the shared-vocabulary locals are
#       already in scope. The builder concatenates these blocks in spec
#       order inside an `@inbounds begin … end` inside the `@kernel`.
# =====================================================================

abstract type LBMBrick end

"""
    LBMSpec(bricks...; stencil=:D2Q9) :: LBMSpec{stencil, Tuple{...}}

Encode an ordered fused-kernel recipe as a singleton type. The type
parameter carries the brick sequence so that different specs dispatch
to different compiled kernels (cached per (spec-type, backend)).
"""
struct LBMSpec{Stencil, Bricks<:Tuple} end

function LBMSpec(bricks::LBMBrick...; stencil::Symbol=:D2Q9)
    return LBMSpec{stencil, Tuple{map(typeof, bricks)...}}()
end

"""
    required_args(::LBMBrick) -> NTuple{N, Symbol}

Declare the kernel-level parameters this brick needs. Default: none.
Concrete bricks override.
"""
required_args(::LBMBrick) = ()

"""
    emit_code(::LBMBrick) -> Expr

Return the code fragment this brick contributes to the generated
kernel body. Default: empty block.
"""
emit_code(::LBMBrick) = Expr(:block)

"""
    phase(::LBMBrick) -> Symbol

Where in the generated kernel body this brick's code is placed:

- `:pre_solid` → before the `if is_solid[i, j]` check. Typically
   pull-stream bricks: both solid and fluid cells need the pulled
   populations (solid → for BB swap, fluid → for collision).
- `:solid`     → inside the `if is_solid[i, j]` branch. Handles the
   solid-cell path (swap-BB or inert).
- `:fluid`     → inside the `else` branch. Handles the fluid-cell
   path (moments, collision, BC overwrite, write).

If a spec has NO `:solid` brick, the generated kernel skips the
`if/else` entirely and emits all `:fluid` bricks flat (useful for
purely-fluid debug kernels like pull-only tests).

Default: `:fluid`.
"""
phase(::LBMBrick) = :fluid

# Canonical argument order for the generated kernel signature. Arrays
# that bricks reference must appear here; scalars likewise. Unlisted
# symbols get appended in insertion order at the tail.
const CANONICAL_ARG_ORDER = [
    # outputs (mutated)
    :f_out, :ρ_out, :ux_out, :uy_out, :uz_out,
    # input arrays (@Const-able)
    :f_in, :is_solid, :q_wall, :uw_link_x, :uw_link_y, :uw_link_z,
    # SLBM departure-point arrays (@Const-able)
    :i_dep, :j_dep,
    # integer scalars
    :Nx, :Ny, :Nz,
    # float scalars
    :ω, :s_plus, :s_minus,
    # SLBM boolean scalars
    :periodic_ξ, :periodic_η,
]

function _canonical_sort(args::AbstractVector{Symbol})
    known    = filter(a -> a in CANONICAL_ARG_ORDER, args)
    unknown  = filter(a -> !(a in CANONICAL_ARG_ORDER), args)
    sort!(known; by = a -> findfirst(==(a), CANONICAL_ARG_ORDER))
    return vcat(known, unknown)
end
