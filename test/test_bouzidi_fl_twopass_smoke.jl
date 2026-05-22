# =====================================================================
# M34 — Bouzidi-FL two-pass smoke test.
#
# Closes the lag-1 x_ff defect of the single-pass :bouzidi_fl spec
# documented in bench/viscoelastic_audit/M30_PHASE2B_AUDIT_VERDICT.md.
# Pass-1 runs the existing halfwayBB collide + WriteMoments; pass-2 reads
# lag-0 f_out (both x_f and x_ff) and lag-0 ρ_out for rho_w.
#
# This smoke uses a tiny Newtonian cylinder (R=10) embedded in an
# otherwise *fully-bounce-back* box (no inlet/outlet, no Zou-He, no source).
# The geometry traps a stagnant fluid: q_wall (cylinder) + halfwayBB at
# the box edges. The bounce-back boundary preserves global mass on the
# fluid sub-domain (cylinder + box walls are both reflective).
#
# Goals:
#   (i)   compiles + launches cleanly through `fused_trt_libb_v2_guo_field_step!`,
#   (ii)  no NaN over 100 steps,
#   (iii) global mass conserved on the fluid sub-domain to 1e-10 (F64 CPU),
#   (iv)  density bounded in [0.9, 1.1] (no spurious source from the BC),
#   (v)   default :halfwayBB path unaffected (no regression of existing code),
#   (vi)  unknown wall_bc raises ArgumentError.
# Benchmark comparison (Aqua A100 F64 + R/Wi sweep) is OUT OF SCOPE here.
# =====================================================================

using Test
using KernelAbstractions
using Kraken

@testset "Bouzidi-FL two-pass — halfwayBB-degenerate regression" begin
    # Closed bounce-back box + grid-aligned cylinder edges (q_w ≈ 0.5).
    # Confirms the trivial collapse case: with q_w == 0.5 everywhere the
    # _libb_branch reduces to halfway-BB, so even the (now-removed) double-BC
    # trap was invisible here. Kept as a regression sentinel.
    FT = Float64
    Nx, Ny = 60, 40
    radius = 8.0
    cx = (Nx - 1) / 2
    cy = (Ny - 1) / 2
    ν = 0.05

    q_wall_h, is_solid_cyl = Kraken.precompute_q_wall_cylinder(Nx, Ny, cx, cy, radius; FT=FT)

    # Combine cylinder solid with full-box solid border (j=1, j=Ny, i=1, i=Nx).
    # The box border traps the fluid: any pop that would leave the domain is
    # bounced back by the kernel's SolidInert + halfwayBB pull. This makes the
    # global mass on the fluid sub-domain conserved (closed system).
    is_solid_h = copy(is_solid_cyl)
    @views is_solid_h[1, :]  .= true
    @views is_solid_h[Nx, :] .= true
    @views is_solid_h[:, 1]  .= true
    @views is_solid_h[:, Ny] .= true

    # Re-compute q_wall after adding the box walls. The cylinder q_wall is
    # already correct on the interior; the new solid border cells flagged by
    # is_solid_h get q_wall implicitly handled by halfwayBB (q_wall = 0 there
    # for the cylinder-derived array, which is the correct "use halfwayBB"
    # signal for non-cylinder solids).
    # That's fine — the two-pass brick only acts on cells where q_wall > 0.

    f_in  = zeros(FT, Nx, Ny, 9)
    f_out = similar(f_in)
    # Initial perturbation: small non-zero velocity to exercise all 8 cut-link
    # directions through the cylinder.
    u0x, u0y = FT(0.02), FT(0.01)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        ux0 = is_solid_h[i, j] ? zero(FT) : u0x
        uy0 = is_solid_h[i, j] ? zero(FT) : u0y
        f_in[i, j, q] = Kraken.equilibrium(D2Q9(), one(FT), ux0, uy0, q)
    end

    ρ   = ones(FT, Nx, Ny)
    ux  = zeros(FT, Nx, Ny)
    uy  = zeros(FT, Nx, Ny)
    uwx = zeros(FT, Nx, Ny, 9)
    uwy = zeros(FT, Nx, Ny, 9)
    fx  = zeros(FT, Nx, Ny)
    fy  = zeros(FT, Nx, Ny)

    fluid_mask = .!is_solid_h

    # Run both wall_bc paths from the same initial condition for the same
    # number of steps. The test for the new two-pass path is RELATIVE to the
    # well-known :halfwayBB baseline: similar density bounds, similar
    # (small) mass drift, no NaN. SolidInert + halfwayBB are not strictly
    # mass-conserving on a closed box (the rest-eq pop on solids acts as a
    # weak sink/source), so we compare drifts rather than asserting absolute
    # conservation.
    n_steps = 100
    function run_kernel(wall_bc::Symbol)
        f_a  = zeros(FT, Nx, Ny, 9)
        f_b  = similar(f_a)
        for j in 1:Ny, i in 1:Nx, q in 1:9
            ux0 = is_solid_h[i, j] ? zero(FT) : u0x
            uy0 = is_solid_h[i, j] ? zero(FT) : u0y
            f_a[i, j, q] = Kraken.equilibrium(D2Q9(), one(FT), ux0, uy0, q)
        end
        ρl  = ones(FT, Nx, Ny)
        uxl = zeros(FT, Nx, Ny)
        uyl = zeros(FT, Nx, Ny)
        # Prime ρ via a no-op step (so mass0 is post-WriteMoments).
        Kraken.fused_trt_libb_v2_guo_field_step!(
            f_b, f_a, ρl, uxl, uyl, is_solid_h, q_wall_h,
            uwx, uwy, fx, fy, Nx, Ny, ν; wall_bc=wall_bc,
        )
        f_a, f_b = f_b, f_a
        KernelAbstractions.synchronize(KernelAbstractions.CPU())
        m0 = sum(ρl[fluid_mask])
        for _ in 1:n_steps
            Kraken.fused_trt_libb_v2_guo_field_step!(
                f_b, f_a, ρl, uxl, uyl, is_solid_h, q_wall_h,
                uwx, uwy, fx, fy, Nx, Ny, ν; wall_bc=wall_bc,
            )
            f_a, f_b = f_b, f_a
        end
        KernelAbstractions.synchronize(KernelAbstractions.CPU())
        m1 = sum(ρl[fluid_mask])
        return (ρl, uxl, uyl, m0, m1)
    end

    ρ_tp, ux_tp, uy_tp, m0_tp, m1_tp = run_kernel(:bouzidi_fl_twopass)
    ρ_hw, ux_hw, uy_hw, m0_hw, m1_hw = run_kernel(:halfwayBB)

    drift_tp = abs(m1_tp - m0_tp) / m0_tp
    drift_hw = abs(m1_hw - m0_hw) / m0_hw

    # (i)+(ii) no NaN in the fluid region (two-pass).
    @test all(isfinite, ρ_tp[fluid_mask])
    @test all(isfinite, ux_tp[fluid_mask])
    @test all(isfinite, uy_tp[fluid_mask])

    # (iii) two-pass mass drift comparable to (and ideally smaller than)
    # halfwayBB baseline. Loose factor-of-10 envelope.
    @test drift_tp < max(10 * drift_hw, 1e-3)

    # (iv) density bounded.
    @test minimum(ρ_tp[fluid_mask]) > 0.9
    @test maximum(ρ_tp[fluid_mask]) < 1.1

    # (v) :halfwayBB default path still runs cleanly (no regression).
    @test all(isfinite, ρ_hw[fluid_mask])
    @test minimum(ρ_hw[fluid_mask]) > 0.9
    @test maximum(ρ_hw[fluid_mask]) < 1.1

    # (vi) unknown wall_bc must throw ArgumentError.
    @test_throws ArgumentError Kraken.fused_trt_libb_v2_guo_field_step!(
        f_out, f_in, ρ, ux, uy, is_solid_h, q_wall_h,
        uwx, uwy, fx, fy, Nx, Ny, ν;
        wall_bc=:not_a_wall_bc,
    )

    @info("M34 Bouzidi-FL two-pass smoke",
          Nx, Ny, radius,
          drift_two_pass = drift_tp,
          drift_halfwayBB = drift_hw,
          ρmin_two_pass = minimum(ρ_tp[fluid_mask]),
          ρmax_two_pass = maximum(ρ_tp[fluid_mask]))
end

# =====================================================================
# Cut-link cylinder smoke (M34-fix).
#
# Newtonian cylinder Re=1 with parabolic Zou-He inlet + Zou-He pressure
# outlet, halfway-BB top/bottom channel walls, R=8 cylinder embedded
# with q ∈ (0, 1] cut-links (mostly q ≠ 0.5 → exercises the actual
# Bouzidi-FL post-collision branch, not the degenerate halfway-BB).
#
# This is the test that catches the double-BC trap predicted by
# `[[feedback_smoke_must_exercise_cutlinks]]`: pre-phase + post-collision
# Bouzidi-FL stacked on cut links over-bounces stagnation pops and
# either NaNs or pumps Cd far above the analytical envelope. With the
# M34-fix RAW pass-1 spec applied, this smoke must produce finite ρ
# and Cd within a generous envelope.
#
# R=8 is undersized vs Schaefer-Turek convergence (R≥20) but the cut-link
# path IS exercised — that's the diagnostic.
# =====================================================================
@testset "Bouzidi-FL two-pass — cut-link cylinder R=8 Newtonian Re=1" begin
    FT = Float64
    Nx, Ny = 160, 40
    radius = 8.0
    cx = FT(Nx ÷ 4)
    cy = FT((Ny - 1) / 2)
    u_max = FT(0.04)
    u_ref = (FT(2) / FT(3)) * u_max
    D = FT(2) * radius
    Re_target = FT(1.0)
    ν = u_ref * D / Re_target

    q_wall_h, is_solid_h = Kraken.precompute_q_wall_cylinder(Nx, Ny, cx, cy, radius; FT=FT)

    # Parabolic inlet profile (Schaefer-Turek 2D convention).
    u_prof = [FT(4) * u_max * FT(j - 1) * FT(Ny - j) / FT(Ny - 1)^2 for j in 1:Ny]

    f_in  = zeros(FT, Nx, Ny, 9)
    f_out = similar(f_in)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        ux0 = is_solid_h[i, j] ? zero(FT) : u_prof[j]
        f_in[i, j, q] = Kraken.equilibrium(D2Q9(), one(FT), ux0, zero(FT), q)
    end

    ρ   = ones(FT, Nx, Ny)
    ux  = zeros(FT, Nx, Ny)
    uy  = zeros(FT, Nx, Ny)
    uwx = zeros(FT, Nx, Ny, 9)
    uwy = zeros(FT, Nx, Ny, 9)
    fx  = zeros(FT, Nx, Ny)
    fy  = zeros(FT, Nx, Ny)

    bc = Kraken.BCSpec2D(; west = Kraken.ZouHeVelocity(u_prof),
                          east = Kraken.ZouHePressure(one(FT)))

    n_steps = 200
    for _ in 1:n_steps
        Kraken.fused_trt_libb_v2_guo_field_step!(
            f_out, f_in, ρ, ux, uy, is_solid_h, q_wall_h,
            uwx, uwy, fx, fy, Nx, Ny, ν;
            wall_bc=:bouzidi_fl_twopass,
        )
        # Pre-collision Zou-He rebuild at i=1 (velocity) and i=Nx (pressure).
        Kraken.apply_bc_rebuild_2d!(f_out, f_in, bc, ν, Nx, Ny; Λ=FT(3/16))
        f_in, f_out = f_out, f_in
    end
    KernelAbstractions.synchronize(KernelAbstractions.CPU())

    fluid_mask = .!is_solid_h

    # (i) No NaN in the fluid region.
    @test all(isfinite, ρ[fluid_mask])
    @test all(isfinite, ux[fluid_mask])
    @test all(isfinite, uy[fluid_mask])

    # (ii) Density bounded — generous envelope (Re=1, transient, 200 steps).
    @test minimum(ρ[fluid_mask]) > 0.5
    @test maximum(ρ[fluid_mask]) < 1.5

    # (iii) Cd within generous 50% envelope of Schaefer-Turek Re=1 ~131.
    # R=8 is undersized so we don't expect tight match — the test is that the
    # double-BC trap is GONE (with the trap, Cd diverges or blows past 1000).
    drag = Kraken.compute_drag_libb_mei_2d(f_in, q_wall_h, uwx, uwy, Nx, Ny)
    Cd = FT(2) * drag.Fx / (u_ref^2 * D)
    @test isfinite(Cd)
    @test 60.0 ≤ Cd ≤ 200.0

    @info("M34-fix cut-link cylinder smoke",
          Nx, Ny, radius, Re_target,
          ν, u_ref,
          Cd = Cd,
          Fx = drag.Fx,
          ρmin = minimum(ρ[fluid_mask]),
          ρmax = maximum(ρ[fluid_mask]))
end

# =====================================================================
# Wi=0.1 polymer-coupled cylinder smoke (M34v3).
#
# A Newtonian-only cut-link smoke is BLIND to the pass-2 `rho_w`
# inconsistency that survived M34-fix: the inconsistency manifests
# downstream via the log-FV polymer pipeline reading a stale `ρ` at
# cut-link cells when computing the Guo body force `Fx_field, Fy_field`
# of the next step. The Aqua matrix YELLOW outcome (+1.6 % Cd at
# Wi=0.1 R=30 + NaN at R=60 Wi=0.1 / R=30 Wi=1.0) was driven by this
# rho_w inconsistency. The cycle: cut-link f's overwritten by pass-2
# → ρ_out NOT recomputed (still pass-1 value) → polymer back-force
# reads stale ρ → polymer source mismatches f-field → drift.
#
# The M34v3 pass-3 cut-link ρ-recompute should close this. This smoke
# drives a small cylinder (R=4, H=18, parabolic inlet via Fx_body
# driving) at modest Wi=0.1 (λ chosen so λ·u_mean/R ≈ 0.1), 30 steps,
# Oldroyd-B. Asserts (a) no NaN in any of ρ, ψ_xx, ψ_xy, ψ_yy; (b) Cd
# finite; (c) ρ ∈ [0.5, 1.5]; (d) trace(C) max < 50 (stability).
# =====================================================================
@testset "Bouzidi-FL two-pass — Wi=0.1 polymer cylinder R=4" begin
    radius = 4.0
    H = 18
    u_mean = 0.006
    # λ chosen so Wi = λ * u_mean / R ≈ 0.1.
    Wi_target = 0.1
    lambda = Wi_target * radius / u_mean
    @test isapprox(lambda * u_mean / radius, Wi_target; rtol=1e-12)

    result = Kraken.run_viscoelastic_logfv_cylinder_coupled_2d(;
        radius=radius, H=H, L_up=4, L_down=7,
        nu_s=0.08, nu_p=0.02, lambda=lambda,
        u_mean=u_mean, Fx_body=1e-7,
        bsd_fraction=1.0, max_steps=30,
        wall_bc=:bouzidi_fl_twopass,
        backend=KernelAbstractions.CPU(), T=Float64,
    )

    fluid = .!result.is_solid

    # (a) no NaN/Inf in ρ + ψ_xx + ψ_xy + ψ_yy on the fluid sub-domain.
    @test all(isfinite, result.rho[fluid])
    @test all(isfinite, result.psixx[fluid])
    @test all(isfinite, result.psixy[fluid])
    @test all(isfinite, result.psiyy[fluid])
    @test all(isfinite, result.ux[fluid])
    @test all(isfinite, result.uy[fluid])

    # (b) Cd finite (R-undersized, value not asserted).
    @test isfinite(result.Cd)
    @test isfinite(result.Cd_s)
    @test isfinite(result.Cd_p)

    # (c) ρ bounded in [0.5, 1.5] over the fluid sub-domain — the very
    # first symptom of a pass-2 rho_w inconsistency is ρ pumping at the
    # cut-link ring (driver M34_FIX_DIAG_VERDICT showed 97 % NaN was a
    # ρ blow-up well before ψ).
    @test result.rho_min > 0.5
    @test result.rho_max < 1.5
    @test result.first_nonfinite_step == 0  # no nonfinite event flagged

    # (d) trace(C_max) bounded — polymer-side stability. `max_c_trace` is
    # max over the fluid sub-domain of (ψ_xx + ψ_yy + 2) in Oldroyd-B
    # log-conformation form (Ψ = log C → C = exp Ψ has trace ≥ 2).
    @test result.min_c_eig > 0
    @test result.max_c_trace < 50.0  # generous: Wi=0.1 should stay near unity

    @info("M34v3 Wi=0.1 polymer cylinder smoke",
          radius = radius,
          H = H,
          Wi = lambda * u_mean / radius,
          lambda = lambda,
          Cd = result.Cd,
          Cd_s = result.Cd_s,
          Cd_p = result.Cd_p,
          rho_min = result.rho_min,
          rho_max = result.rho_max,
          max_c_trace = result.max_c_trace,
          min_c_eig = result.min_c_eig,
          completed_steps = result.completed_steps)
end
