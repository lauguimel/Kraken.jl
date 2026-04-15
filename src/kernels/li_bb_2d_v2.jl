# =====================================================================
# TRT + LI-BB refactored kernel — V2.
#
# Assembled via the kernel DSL from composable bricks. Fixes the
# double-BC bug of `fused_trt_libb_step!` by making solid cells INERT
# instead of swapping populations: all wall physics goes through the
# Bouzidi overwrite on fluid-cell populations, so bounce-back is no
# longer applied twice. Moments are recomputed from the post-BC
# populations (Krüger et al. 2017, §5.3.4, pitfall #1).
#
# Bricks:
#   PullHalfwayBB → SolidInert | Moments → CollideTRT → ApplyLiBB →
#                   RecomputeMoments → WriteFLiBB → WriteMoments
#
# This file must be included AFTER the DSL (`dsl/lbm_builder.jl`) and
# after `li_bb_2d.jl` (which defines `_libb_branch`, referenced by the
# `ApplyLiBB` brick's emitted code).
# =====================================================================

const _TRT_LIBB_V2_SPEC = LBMSpec(
    PullHalfwayBB(), SolidInert(),
    Moments(), CollideTRT(), ApplyLiBB(),
    RecomputeMoments(),
    WriteFLiBB(), WriteMoments(),
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
