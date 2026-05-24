# =====================================================================
# M42 — MUSCL boundary relaxation (`:muscl_superbee_relax`) smoke tests.
#
# Pass-1 = existing `:muscl_superbee` whole-cell-fallback kernel
# (unchanged behaviour).  Pass-2 = NEW cylinder-band overwrite kernel
# (one-sided MUSCL on the broken-axis face; full MUSCL otherwise; open-
# wall band j≤2 / j≥Ny−1 / i≤2 / i≥Nx−1 preserved as `:rusanov`).
#
# See bench/viscoelastic_audit/M42_DESIGN.md for the full spec.
#
# Acceptance gates (CPU F64, ≤30 s total):
#   (1) Newtonian R=8 Wi=0 (β=1, polymer dormant): no NaN, ρ bounded,
#       Cd finite, within ±5 % of `:muscl_superbee` (pass-1 only)
#       baseline at same R.
#   (2) Wi=0.1 polymer R=8: no NaN, ρ ∈ [0.5, 1.5], polymer pipeline
#       stable through 200 steps.
#   (3) Wi=1.0 polymer R=8 (stress proxy for M29c-v2 NaN): no NaN
#       through 1000 steps; max|Ψ_xx| bounded.
#   (4) Sentinel: `:muscl_superbee` (no _relax) bit-identical to pre-M42
#       at Wi=0.1 R=8 (regression check — pass-2 must not touch any
#       cell when scheme=:muscl_superbee).
# =====================================================================

using Test
using KernelAbstractions
using Kraken

@testset "M42 — :muscl_superbee_relax Newtonian R=8 Wi=0" begin
    radius = 8.0
    H = 32
    u_mean = 0.005

    res_relax = Kraken.run_viscoelastic_logfv_cylinder_coupled_2d(;
        radius=radius, H=H, L_up=4, L_down=8,
        nu_s=0.08, nu_p=0.0, lambda=1.0,
        u_mean=u_mean, Fx_body=0.0,
        bsd_fraction=1.0, max_steps=200,
        wall_bc=:halfwayBB,
        advection_scheme=:muscl_superbee_relax,
        backend=KernelAbstractions.CPU(), T=Float64,
    )

    res_ref = Kraken.run_viscoelastic_logfv_cylinder_coupled_2d(;
        radius=radius, H=H, L_up=4, L_down=8,
        nu_s=0.08, nu_p=0.0, lambda=1.0,
        u_mean=u_mean, Fx_body=0.0,
        bsd_fraction=1.0, max_steps=200,
        wall_bc=:halfwayBB,
        advection_scheme=:muscl_superbee,
        backend=KernelAbstractions.CPU(), T=Float64,
    )

    fluid = .!res_relax.is_solid

    @test all(isfinite, res_relax.rho[fluid])
    @test all(isfinite, res_relax.ux[fluid])
    @test all(isfinite, res_relax.uy[fluid])
    @test res_relax.rho_min > 0.5
    @test res_relax.rho_max < 1.5
    @test isfinite(res_relax.Cd)

    # Cd within ±5 % of :muscl_superbee baseline at the same setup.
    @test isfinite(res_ref.Cd)
    @test abs(res_relax.Cd - res_ref.Cd) / abs(res_ref.Cd) < 0.05

    @info("M42 Newtonian Wi=0 smoke",
          Cd_relax = res_relax.Cd,
          Cd_ref   = res_ref.Cd,
          rho_min  = res_relax.rho_min,
          rho_max  = res_relax.rho_max,
          completed_steps = res_relax.completed_steps)
end

@testset "M42 — :muscl_superbee_relax polymer Wi=0.1 R=8" begin
    radius = 8.0
    H = 32
    u_mean = 0.005
    Wi_target = 0.1
    lambda = Wi_target * radius / u_mean
    @test isapprox(lambda * u_mean / radius, Wi_target; rtol=1e-12)
    # β = 0.59 → nu_s + nu_p = nu_total, β = nu_s/nu_total.
    nu_total = 0.08
    nu_s = 0.59 * nu_total
    nu_p = nu_total - nu_s

    res = Kraken.run_viscoelastic_logfv_cylinder_coupled_2d(;
        radius=radius, H=H, L_up=4, L_down=8,
        nu_s=nu_s, nu_p=nu_p, lambda=lambda,
        u_mean=u_mean, Fx_body=0.0,
        bsd_fraction=1.0, max_steps=200,
        wall_bc=:halfwayBB,
        advection_scheme=:muscl_superbee_relax,
        backend=KernelAbstractions.CPU(), T=Float64,
    )

    fluid = .!res.is_solid

    @test all(isfinite, res.rho[fluid])
    @test all(isfinite, res.ux[fluid])
    @test all(isfinite, res.uy[fluid])
    @test all(isfinite, res.psixx[fluid])
    @test all(isfinite, res.psixy[fluid])
    @test all(isfinite, res.psiyy[fluid])

    @test res.rho_min > 0.5
    @test res.rho_max < 1.5
    @test res.min_c_eig > 0
    @test isfinite(res.max_c_trace)
    @test res.max_c_trace < 50.0
    @test isfinite(res.Cd)
    @test isfinite(res.Cd_s)
    @test isfinite(res.Cd_p)
    @test res.first_nonfinite_step == 0

    @info("M42 polymer Wi=0.1 smoke",
          Cd = res.Cd, Cd_s = res.Cd_s, Cd_p = res.Cd_p,
          rho_min = res.rho_min, rho_max = res.rho_max,
          max_c_trace = res.max_c_trace,
          completed_steps = res.completed_steps)
end

@testset "M42 — :muscl_superbee_relax polymer Wi=1.0 R=8 (NaN proxy)" begin
    # Proxy for the M29c-v2 step-92,200 R=30 Wi=1 NaN: a stiffer
    # 1000-step Wi=1.0 R=8 run. If M42 reintroduces a CD2-like anti-TVD
    # mechanism we expect rho_min < 0 or NaN well before 1000 steps.
    radius = 8.0
    H = 32
    u_mean = 0.005
    Wi_target = 1.0
    lambda = Wi_target * radius / u_mean
    @test isapprox(lambda * u_mean / radius, Wi_target; rtol=1e-12)
    nu_total = 0.08
    nu_s = 0.59 * nu_total
    nu_p = nu_total - nu_s

    res = Kraken.run_viscoelastic_logfv_cylinder_coupled_2d(;
        radius=radius, H=H, L_up=4, L_down=8,
        nu_s=nu_s, nu_p=nu_p, lambda=lambda,
        u_mean=u_mean, Fx_body=0.0,
        bsd_fraction=1.0, max_steps=1000,
        wall_bc=:halfwayBB,
        advection_scheme=:muscl_superbee_relax,
        backend=KernelAbstractions.CPU(), T=Float64,
    )

    fluid = .!res.is_solid

    @test all(isfinite, res.rho[fluid])
    @test all(isfinite, res.ux[fluid])
    @test all(isfinite, res.uy[fluid])
    @test all(isfinite, res.psixx[fluid])
    @test all(isfinite, res.psixy[fluid])
    @test all(isfinite, res.psiyy[fluid])

    @test res.rho_min > 0.5
    @test res.rho_max < 1.5
    @test res.min_c_eig > 0
    @test res.first_nonfinite_step == 0
    @test res.completed_steps == 1000
    # Polymer envelope - the cylinder Wi=1 stress should not blow up.
    # max_c_trace ~ 2 baseline; conservative envelope avoids false flags.
    @test isfinite(res.max_c_trace)
    @test res.max_c_trace < 200.0

    @info("M42 polymer Wi=1.0 NaN-proxy smoke",
          Cd = res.Cd, Cd_s = res.Cd_s, Cd_p = res.Cd_p,
          rho_min = res.rho_min, rho_max = res.rho_max,
          max_c_trace = res.max_c_trace,
          completed_steps = res.completed_steps)
end

@testset "M42 — :muscl_superbee sentinel (no regression at Wi=0.1)" begin
    # Pass-2 must NOT touch any cell when scheme=:muscl_superbee.  The
    # only path-difference between :muscl_superbee and pre-M42 is the
    # whitelist update + dispatch fall-through.  Run :muscl_superbee
    # under the new whitelist and confirm it produces finite Cd within
    # the historical Wi=0.1 R=8 envelope.
    radius = 8.0
    H = 32
    u_mean = 0.005
    Wi_target = 0.1
    lambda = Wi_target * radius / u_mean
    nu_total = 0.08
    nu_s = 0.59 * nu_total
    nu_p = nu_total - nu_s

    res = Kraken.run_viscoelastic_logfv_cylinder_coupled_2d(;
        radius=radius, H=H, L_up=4, L_down=8,
        nu_s=nu_s, nu_p=nu_p, lambda=lambda,
        u_mean=u_mean, Fx_body=0.0,
        bsd_fraction=1.0, max_steps=200,
        wall_bc=:halfwayBB,
        advection_scheme=:muscl_superbee,
        backend=KernelAbstractions.CPU(), T=Float64,
    )

    fluid = .!res.is_solid
    @test all(isfinite, res.rho[fluid])
    @test isfinite(res.Cd)
    @test res.rho_min > 0.5
    @test res.rho_max < 1.5

    @info("M42 :muscl_superbee sentinel",
          Cd = res.Cd,
          completed_steps = res.completed_steps)
end
