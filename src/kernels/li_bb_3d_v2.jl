# =====================================================================
# TRT + LI-BB 3D kernel (D3Q19), Ginzburg-exact via DSL.
#
# Same recipe as li_bb_2d_v2.jl, ported to D3Q19:
#   PullHalfwayBB_3D → SolidInert_3D | ApplyLiBBPrePhase_3D →
#                      Moments_3D → CollideTRTDirect_3D →
#                      WriteMoments_3D
#
# Reuses `_libb_branch` from li_bb_2d.jl (the Bouzidi formula is
# stencil-independent).
# =====================================================================

const _TRT_LIBB_V2_SPEC_3D = LBMSpec(
    PullHalfwayBB_3D(), SolidInert_3D(),
    ApplyLiBBPrePhase_3D(),
    Moments_3D(), CollideTRTDirect_3D(),
    WriteMoments_3D();
    stencil = :D3Q19,
)

"""
    fused_trt_libb_v2_step_3d!(f_out, f_in, ρ, ux, uy, uz, is_solid,
                                q_wall, uw_x, uw_y, uw_z,
                                Nx, Ny, Nz, ν; Λ=3/16)

Single D3Q19 step: pull-stream + SolidInert on solid cells, or
ApplyLiBBPrePhase + TRT collision on fluid cells. Ginzburg-exact for
halfway-BB + TRT Λ = 3/16 on Couette; handles arbitrary q_w ∈ (0, 1]
via the full Bouzidi formula.
"""
function fused_trt_libb_v2_step_3d!(f_out, f_in, ρ, ux, uy, uz, is_solid,
                                      q_wall, uw_link_x, uw_link_y, uw_link_z,
                                      Nx, Ny, Nz, ν; Λ::Real=3/16)
    backend = KernelAbstractions.get_backend(f_in)
    ET = eltype(f_in)
    s_plus, s_minus = trt_rates(ν; Λ=Λ)
    kernel! = build_lbm_kernel(backend, _TRT_LIBB_V2_SPEC_3D)
    kernel!(f_out, ρ, ux, uy, uz, f_in, is_solid,
            q_wall, uw_link_x, uw_link_y, uw_link_z,
            Nx, Ny, Nz, ET(s_plus), ET(s_minus);
            ndrange=(Nx, Ny, Nz))
end
