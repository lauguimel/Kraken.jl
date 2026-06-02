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
    ApplyLiBBPrePhase(),                # full Bouzidi pre-phase (any q_w ∈ (0, 1])
    Moments(), CollideTRTDirect(),       # collision writes f_out directly (no fp*c intermediate)
    WriteMoments(),                      # pre-collision moments (correct after pre-phase)
)

const _TRT_LIBB_V2_HERMITE_SPEC = LBMSpec(
    PullHalfwayBB(), SolidInert(),
    ApplyLiBBPrePhase(),
    Moments(), CollideTRTDirectHermite(),
    WriteMoments(),
)

const _TRT_LIBB_V2_GUO_FIELD_SPEC = LBMSpec(
    PullHalfwayBB(), SolidInert(),
    ApplyLiBBPrePhase(),
    Moments(), CollideTRTDirectGuoField(),
    WriteMoments(),
)

const _TRT_LIBB_V2_GUO_FIELD_BOUZIDI_FL_SPEC = LBMSpec(
    PullHalfwayBB(), SolidInert(),
    Moments(), CollideTRTDirectGuoField(),
    ApplyBouzidiFLPostCollide(),
    WriteMoments(),
)

# Two-pass Bouzidi-FL pass-1 RAW spec: PullHalfwayBB + SolidInert + Moments +
# CollideTRTDirectGuoField + WriteMoments. CRITICALLY OMITS `ApplyLiBBPrePhase`.
# Reusing `_TRT_LIBB_V2_GUO_FIELD_SPEC` for pass-1 stacks pre-phase Bouzidi-FL
# on top of pass-2's post-collision Bouzidi-FL — the V2-motivating double-BC
# bug. See bench/viscoelastic_audit/M34_DEBUG_VERDICT.md and
# M34_SPEC_AUDIT_VERDICT.md (M34-fix, 2026-05-22).
const _TRT_LIBB_V2_GUO_FIELD_RAW_SPEC = LBMSpec(
    PullHalfwayBB(), SolidInert(),
    Moments(), CollideTRTDirectGuoField(),
    WriteMoments(),
)

# Two-pass Bouzidi-FL: pass-1 uses _TRT_LIBB_V2_GUO_FIELD_RAW_SPEC (collision +
# halfwayBB writes f_out, ρ_out, NO pre-phase BC). Pass-2 launches ONLY the
# Bouzidi-FL overwrite brick — by then pass-1 has globally synchronised, so
# f_out and ρ_out are lag-0 everywhere. Closes the lag-1 x_ff defect identified
# in M30 Phase 2b.
const _TRT_LIBB_V2_GUO_FIELD_BOUZIDI_FL_TWOPASS_PASS2_SPEC = LBMSpec(
    ApplyBouzidiFLPostCollideTwoPass(),
)

# Pass-3 (M34v3): cut-link-only ρ recompute. After pass-2 overwrites
# `f_out[i, j, qbar]` on flagged cut links, the `ρ_out` left by pass-1 is the
# sum of the *pre-Bouzidi-FL* pops at those same cells — i.e. inconsistent
# with the post-pass-2 f-set. Downstream readers (the log-FV polymer pipeline
# at next step, plus the next step's Guo body-force `f → ρ` chain) need ρ
# consistent with the f's they read. This brick re-sums `f_out[i, j, 1..9]`
# and overwrites `ρ_out[i, j]` ONLY on cut-link cells, leaving non-cut-link
# cells bit-exact. See `bench/viscoelastic_audit/M34_FIX_DIAG_VERDICT.md`
# §"Candidate residual bugs" #1 (HIGH).
const _TRT_LIBB_V2_GUO_FIELD_BOUZIDI_FL_TWOPASS_PASS3_SPEC = LBMSpec(
    ApplyCutLinkRhoRecompute(),
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

"""
    fused_trt_libb_v2_hermite_step!(f_out, f_in, ρ, ux, uy, is_solid,
                                     q_wall, uw_x, uw_y,
                                     tau_p_xx, tau_p_xy, tau_p_yy,
                                     Nx, Ny, ν; Λ=3/16, source_scale=1)

Experimental LI-BB V2 step with the polymer Hermite stress source fused into
the TRT collision. `source_scale=1` gives the local Liu/Yu in-collision
amplitude; `source_scale=1/(1-s_plus/2)` gives the standalone CE-corrected
amplitude used by the post-collision source.
"""
function fused_trt_libb_v2_hermite_step!(f_out, f_in, ρ, ux, uy, is_solid,
                                           q_wall, uw_link_x, uw_link_y,
                                           tau_p_xx, tau_p_xy, tau_p_yy,
                                           Nx, Ny, ν; Λ::Real=3/16,
                                           source_scale::Real=1,
                                           source_on_cutlinks::Bool=true)
    backend = KernelAbstractions.get_backend(f_in)
    ET = eltype(f_in)
    s_plus, s_minus = trt_rates(ν; Λ=Λ)
    kernel! = build_lbm_kernel(backend, _TRT_LIBB_V2_HERMITE_SPEC)
    kernel!(f_out, ρ, ux, uy, f_in, is_solid,
            q_wall, uw_link_x, uw_link_y,
            Nx, Ny, ET(s_plus), ET(s_minus),
            tau_p_xx, tau_p_xy, tau_p_yy,
            ET(source_scale), source_on_cutlinks;
            ndrange=(Nx, Ny))
end

"""
    fused_trt_libb_v2_guo_field_step!(f_out, f_in, ρ, ux, uy, is_solid,
                                       q_wall, uw_x, uw_y, Fx, Fy,
                                       Nx, Ny, ν; Λ=3/16,
                                       wall_bc=:halfwayBB)

Modular LI-BB V2 solvent step with a per-cell Guo force field. This keeps the
same pull-stream, solid handling, and cut-link pre-phase as
`fused_trt_libb_v2_step!`, while replacing only the collision brick with a
TRT+Guo-field brick. It is intended for cell-centered log-FV polymer coupling.
"""
function fused_trt_libb_v2_guo_field_step!(f_out, f_in, ρ, ux, uy, is_solid,
                                            q_wall, uw_link_x, uw_link_y,
                                            Fx_field, Fy_field,
                                            Nx, Ny, ν; Λ::Real=3/16,
                                            wall_bc::Symbol=:halfwayBB)
    wall_bc in (:halfwayBB, :bouzidi_fl, :bouzidi_fl_twopass) ||
        throw(ArgumentError("wall_bc must be :halfwayBB, :bouzidi_fl, or :bouzidi_fl_twopass"))
    @trace_enter :lbm_step
    return _fused_trt_libb_v2_guo_field_step!(
        Val(wall_bc), f_out, f_in, ρ, ux, uy, is_solid,
        q_wall, uw_link_x, uw_link_y, Fx_field, Fy_field,
        Nx, Ny, ν; Λ=Λ,
    )
end

function _fused_trt_libb_v2_guo_field_step!(::Val{:halfwayBB},
                                             f_out, f_in, ρ, ux, uy, is_solid,
                                             q_wall, uw_link_x, uw_link_y,
                                             Fx_field, Fy_field,
                                             Nx, Ny, ν; Λ::Real=3/16)
    @trace_enter :lbm_step_halfwayBB
    backend = KernelAbstractions.get_backend(f_in)
    ET = eltype(f_in)
    s_plus, s_minus = trt_rates(ν; Λ=Λ)
    kernel! = build_lbm_kernel(backend, _TRT_LIBB_V2_GUO_FIELD_SPEC)
    kernel!(f_out, ρ, ux, uy, f_in, is_solid,
            q_wall, uw_link_x, uw_link_y,
            Nx, Ny, ET(s_plus), ET(s_minus),
            Fx_field, Fy_field;
            ndrange=(Nx, Ny))
end

function _fused_trt_libb_v2_guo_field_step!(::Val{:bouzidi_fl},
                                             f_out, f_in, ρ, ux, uy, is_solid,
                                             q_wall, uw_link_x, uw_link_y,
                                             Fx_field, Fy_field,
                                             Nx, Ny, ν; Λ::Real=3/16)
    @trace_enter :lbm_step_bouzidiFL
    backend = KernelAbstractions.get_backend(f_in)
    ET = eltype(f_in)
    s_plus, s_minus = trt_rates(ν; Λ=Λ)
    kernel! = build_lbm_kernel(backend, _TRT_LIBB_V2_GUO_FIELD_BOUZIDI_FL_SPEC)
    kernel!(f_out, ρ, ux, uy, f_in, is_solid,
            q_wall, uw_link_x, uw_link_y,
            Nx, Ny, ET(s_plus), ET(s_minus),
            Fx_field, Fy_field;
            ndrange=(Nx, Ny))
end

# Two-pass Bouzidi-FL dispatch. Pass-1 = the standard halfwayBB collide step
# (writes f_out + ρ_out everywhere). We then explicitly synchronise the backend
# so pass-2 sees fully-written f_out / ρ_out (no cross-thread race).
# Pass-2 = a minimal kernel that overwrites f_out[i, j, qbar] on flagged cut
# links, reading lag-0 f_out at both x_f and x_ff and lag-0 ρ_out for rho_w.
# Closes the M30 Phase 2b lag mismatch (see
# bench/viscoelastic_audit/M30_PHASE2B_AUDIT_VERDICT.md §"Proposed minimal fix").
function _fused_trt_libb_v2_guo_field_step!(::Val{:bouzidi_fl_twopass},
                                             f_out, f_in, ρ, ux, uy, is_solid,
                                             q_wall, uw_link_x, uw_link_y,
                                             Fx_field, Fy_field,
                                             Nx, Ny, ν; Λ::Real=3/16)
    @trace_enter :lbm_step_bouzidiFL_twopass
    backend = KernelAbstractions.get_backend(f_in)
    ET = eltype(f_in)
    s_plus, s_minus = trt_rates(ν; Λ=Λ)
    # Pass 1: RAW collide + halfwayBB + WriteMoments (NO pre-phase BC), writes
    # f_out, ρ_out. The RAW spec drops `ApplyLiBBPrePhase` from
    # `_TRT_LIBB_V2_GUO_FIELD_SPEC` to avoid stacking pre-phase + post-collision
    # Bouzidi-FL — i.e. the double-BC trap (see M34_DEBUG_VERDICT.md).
    # Canonical arg order for the RAW spec drops :q_wall, :uw_link_x,
    # :uw_link_y (no pre-phase brick referencing them).
    pass1! = build_lbm_kernel(backend, _TRT_LIBB_V2_GUO_FIELD_RAW_SPEC)
    pass1!(f_out, ρ, ux, uy, f_in, is_solid,
           Nx, Ny, ET(s_plus), ET(s_minus),
           Fx_field, Fy_field;
           ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
    # Pass 2: Bouzidi-FL overwrite, lag-0 on x_f, x_ff, ρ_w.
    # Pass-2 arg order = canonical sort of the brick's required_args:
    #   :f_out, :ρ_out, :is_solid, :q_wall, :uw_link_x, :uw_link_y, :Nx, :Ny
    pass2! = build_lbm_kernel(backend, _TRT_LIBB_V2_GUO_FIELD_BOUZIDI_FL_TWOPASS_PASS2_SPEC)
    pass2!(f_out, ρ, is_solid, q_wall, uw_link_x, uw_link_y, Nx, Ny;
           ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
    # Pass 3 (M34v3): cut-link-only ρ recompute. Re-sums f_out at cells with
    # any q_wall[i,j,q] > 0 (q ∈ 2..9), overwriting `ρ_out` so downstream
    # readers see ρ consistent with the post-pass-2 cut-link f-set. Non-cut-
    # link cells stay bit-exact (pass-1 `ρ_out`). Fixes the rho_w
    # inconsistency identified in M34_FIX_DIAG_VERDICT §"Candidate residual
    # bugs" #1 (HIGH).
    # Pass-3 arg order = canonical sort of the brick's required_args:
    #   :f_out, :ρ_out, :is_solid, :q_wall, :Nx, :Ny
    pass3! = build_lbm_kernel(backend, _TRT_LIBB_V2_GUO_FIELD_BOUZIDI_FL_TWOPASS_PASS3_SPEC)
    pass3!(f_out, ρ, is_solid, q_wall, Nx, Ny;
           ndrange=(Nx, Ny))
end
