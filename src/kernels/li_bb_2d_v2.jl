# =====================================================================
# TRT + LI-BB refactored kernel — V2.
#
# Assembled via the kernel DSL from composable bricks. Fixes the
# double-BC bug of `fused_trt_libb_step!` at the ROOT CAUSE, by
# applying the halfway-BB correction ONCE, pre-collision, via an
# inline substitution on the pulled populations.
#
# Key insight (discovered while iterating on this fix): the legacy
# kernel applied BC twice — once via SolidSwapBB on solid cells, once
# via post-collision Bouzidi LI-BB on fluid cells. My first V2 fix
# replaced SolidSwapBB with SolidInert (solids = rest equilibrium) and
# kept the post-collision LI-BB, but the collision produced biased
# fp*c because pre-collision moments were polluted by solid-sourced
# w_q populations. Adding ApplyHalfwayBBPrePhase to substitute those
# junk pops fixed the moments, but running BOTH pre-phase and
# post-phase LI-BB resulted in L2 ≈ 2.2 %: a *second* double-BC.
#
# The correct spec is PRE-PHASE ONLY: substitute fp_{q̄} with
# lag-1 halfway-BB estimate before collision (pre-phase), then the
# collision's fp*c IS the correctly-bounced post-collision pop. No
# post-phase overwrite needed. Result: L2 = 0.06 % at Ny=33, profile
# ratio 0.998-1.000 across the whole gap (Ginzburg-exact to TRT
# Λ=3/16 precision).
#
# Bricks:
#   PullHalfwayBB → SolidInert | ApplyHalfwayBBPrePhase →
#                   Moments → CollideTRTDirect → WriteMoments
#
# This file must be included AFTER the DSL (`dsl/lbm_builder.jl`) and
# after `li_bb_2d.jl` (for `_libb_branch`, not used here but kept for
# the legacy kernel and for Bouzidi variants with q_w ≠ 0.5).
# =====================================================================

const _TRT_LIBB_V2_SPEC = LBMSpec(
    PullHalfwayBB(), SolidInert(),
    ApplyHalfwayBBPrePhase(),           # substitute solid-sourced pops before moments
    Moments(), CollideTRTDirect(),       # collision writes f_out directly (no fp*c intermediate)
    WriteMoments(),                      # pre-collision moments (correct after pre-phase)
)

"""
    fused_trt_libb_v2_step!(f_out, f_in, ρ, ux, uy, is_solid,
                             q_wall, uw_x, uw_y, Nx, Ny, ν; Λ=3/16)

Refactored TRT + LI-BB step that fixes the double-BC bug of
`fused_trt_libb_step!`. Same call signature — drop-in replacement.
"""
function fused_trt_libb_v2_step!(f_out, f_in, ρ, ux, uy, is_solid,
                                   q_wall, uw_link_x, uw_link_y,
                                   Nx, Ny, ν; Λ::Real=3/16)
    backend = KernelAbstractions.get_backend(f_in)
    ET = eltype(f_in)
    s_plus, s_minus = trt_rates(ν; Λ=Λ)
    kernel! = build_lbm_kernel(backend, _TRT_LIBB_V2_SPEC)
    kernel!(f_out, ρ, ux, uy, f_in, is_solid,
            q_wall, uw_link_x, uw_link_y,
            Nx, Ny, ET(s_plus), ET(s_minus);
            ndrange=(Nx, Ny))
end
