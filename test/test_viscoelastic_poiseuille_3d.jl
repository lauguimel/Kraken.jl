using Test
using Kraken
using KernelAbstractions

# ==========================================================================
# 3D viscoelastic Oldroyd-B planar Poiseuille — second canonical analytical
# canary. Channel flow in x driven by a constant Guo body force, no-slip
# half-way bounce-back walls on the y-faces, periodic x AND z. Unlike Couette
# (constant γ̇), Poiseuille has γ̇(y) varying LINEARLY across the channel, so it
# tests the conformation velocity-gradient computation across y via the C(y) and
# N1(y) PROFILES.
#
# Closed-form steady Oldroyd-B targets (OB shear viscosity is constant → the
# velocity is the Newtonian parabola). Half-way BB walls at y=0.5 and y=Ny+0.5,
# fluid row j at height (j − 0.5):
#   u_x(j) = Fx/(2 ν_total)·(j − 0.5)·(Ny + 0.5 − j)      (parabola)
#   γ̇(j) = |du_x/dy| linear,  C_xy(j) = λ·γ̇(j),  C_xx(j) = 1 + 2·(λγ̇(j))²
#   C_yy = C_zz = 1,  C_xz = C_yz = 0,  N1(j) = 2·η_p·λ·γ̇(j)²,  N2 = 0.
#
# ── FINDING (this canary, β=0.5, peak Wi≈0.5) ─────────────────────────────
# The 3D viscoelastic stack couples the polymer to the momentum equation via the
# *Hermite stress source* (apply_hermite_source_3d!) with s_plus = ω_s, the SAME
# coupling the 3D Couette and sphere drivers use. In FORCE-DRIVEN flow this
# over-resists the momentum balance: the measured velocity is ≈ 0.42× the
# analytical parabola at β=0.5 (effective viscosity inflated ~2.4×), worsening
# monotonically as β decreases. The validated 2D Poiseuille canary uses a
# DIFFERENT, stress-divergence body-force coupling (nu_lbm = nu_s + bsd·nu_p,
# bsd=0) and reproduces the parabola to 0.07% at the same β — so this is a
# coupling-scheme finding, not a bug in the streamer/force path (the pure-solvent
# limit ν_p→0 of THIS driver reproduces the parabola to 0.05%, asserted below).
# Couette masked it because its moving walls impose the velocity mechanically.
#
# Separately, the conformation TRT advection-diffusion over-diffuses C across y:
# the centre-line C_xx settles at ≈1.015 (analytical 1.0) and the near-wall C_xy
# under-predicts λ·γ̇_meas by ~25-40%. So the coupled C(y)/N1(y) profiles do NOT
# match the local-γ̇ analytical profile in the interior. Per the mission
# guardrail these mismatches are REPORTED, not masked: the profile-match
# assertions are @test_broken (tracked, not forced green); the headline 3D
# invariants and the pure-solvent velocity control are hard @test.
#
# Fast: CPU Float64, 6×32×6 box, 40 000 steps (converged — identical at 80 k).
# ==========================================================================

@testset "Viscoelastic planar Poiseuille — 3D Oldroyd-B canary" begin

    backend = CPU()
    FT = Float64

    Nx, Ny, Nz = 6, 32, 6
    ν_total = 0.1
    beta    = 0.5
    ν_s = beta * ν_total
    ν_p = (1 - beta) * ν_total
    Fx  = 1.5e-5
    # γ̇_wall = Fx/(2 ν_total)·(Ny−1); set λ for a peak Weissenberg Wi_wall ≈ 0.5.
    γ̇_wall = Fx / (2 * ν_total) * (Ny - 1)
    Wi_wall_target = 0.5
    λ = Wi_wall_target / γ̇_wall
    max_steps = 40_000

    # --- Units sanity: stability + steady-state-time estimate -------------
    plan = Kraken.Units.compile(; physics = :viscoelastic,
                                  geometry = (; type = :poiseuille, blockage = 0.0),
                                  bc = (; wall_bc = :halfwayBB),
                                  Re = (Fx * Ny^2 / (8 * ν_total)) * Ny / ν_total,
                                  Wi = Wi_wall_target, beta = beta,
                                  R_LU = Ny, u_LU = Fx * Ny^2 / (8 * ν_total),
                                  nu_s = ν_s, nu_p = ν_p, lambda = λ,
                                  strict = false)
    @test plan.units.tau_hydro > 0.55
    @test plan.units.Ma < 0.05
    est = Kraken.Units.estimate_steady_state(plan)
    @test est.exists                      # Poiseuille admits a steady state
    @test est.t_diff > 0 && isfinite(est.t_diff)
    @test isempty(Kraken.Units.blocking_issues(plan.warnings))

    res = run_conformation_poiseuille_libb_3d(; Nx = Nx, Ny = Ny, Nz = Nz,
                                                Fx = Fx, ν_s = ν_s, ν_p = ν_p,
                                                lambda = λ, tau_plus = 1.0,
                                                max_steps = max_steps,
                                                backend = backend, FT = FT)

    rel(a, b) = abs(a - b) / abs(b)

    # --- NaN-free & physical sanity --------------------------------------
    @test !any(isnan, res.ux)
    @test !any(isnan, res.C_xx)
    @test !any(isnan, res.C_xy)
    @test res.beta ≈ beta atol = 1e-12
    # Density conservation (incompressible reference ρ₀ = 1).
    @test abs(sum(res.ρ) - Nx * Ny * Nz) / (Nx * Ny * Nz) < 1e-3

    # --- (A) Uniquely-3D invariants (machine precision) — HEADLINE 3D check
    # C_yy = C_zz = 1, C_xz = C_yz = 0 and N2 = 0 are exact Oldroyd-B planar-shear
    # symmetries across the WHOLE channel. The 6-component solver reproduces them
    # to ~1e-13 at every y-station (the off-diagonal / z-coupling 2D never sees).
    @test maximum(abs.(res.Cyy_prof .- 1)) < 1e-6
    @test maximum(abs.(res.Czz_prof .- 1)) < 1e-6
    @test maximum(abs.(res.Cxz_prof)) < 1e-6
    @test maximum(abs.(res.Cyz_prof)) < 1e-6
    @test maximum(abs.(res.N2_prof)) < 1e-6 * maximum(abs.(res.N1_prof))

    # --- (B) Velocity: pure-solvent control validates streamer + force path
    # With ν_p → 0 the Hermite source vanishes and the driver must reproduce the
    # analytical parabola (this isolates the reusable periodic-xz / no-slip-y
    # streamer + Guo body force from the polymer coupling).
    res_solvent = run_conformation_poiseuille_libb_3d(; Nx = Nx, Ny = Ny, Nz = Nz,
                                                        Fx = Fx, ν_s = ν_total - 1e-6,
                                                        ν_p = 1e-6, lambda = 10.0,
                                                        tau_plus = 1.0,
                                                        max_steps = 20_000,
                                                        backend = backend, FT = FT)
    u_max_an = res_solvent.u_max
    Linf_solvent = maximum(abs.(res_solvent.profile[2:end-1] .-
                                res_solvent.u_analytical[2:end-1])) / u_max_an
    @test Linf_solvent < 0.01        # pure-solvent parabola ≤ 1 %

    # --- (C) FINDINGS (tracked via @test_broken, NOT forced green) --------
    # (C1) Coupled velocity parabola: the Hermite-source momentum over-coupling
    #      gives a ~57 % velocity deficit at β=0.5 → the parabola match FAILS.
    Linf_coupled = maximum(abs.(res.profile[2:end-1] .-
                                res.u_analytical[2:end-1])) / res.u_max
    @test_broken Linf_coupled < 0.01

    # (C2) Conformation / N1 profile vs the LOCAL measured shear rate. Even
    #      against Wi_local(y) = λ·γ̇_meas(y) the near-wall C_xy under-predicts by
    #      ~25-40 % and the centre C_xx floors at ≈1.015 (TRT over-diffusion of
    #      C across y). Track the near-wall self-consistency (j = 4, where γ̇ is
    #      large and well resolved) as broken.
    prof = res.profile
    sgrad4 = (prof[5] - prof[3]) / 2            # signed du_x/dy at j=4
    Cxy_an4 = λ * sgrad4
    @test_broken rel(res.Cxy_prof[4], Cxy_an4) < 0.02
    Cxx_an4 = 1 + 2 * (λ * sgrad4)^2
    @test_broken rel(res.Cxx_prof[4], Cxx_an4) < 0.03
    # Centre-line C_xx should relax to 1 (γ̇→0); measured ≈1.015.
    @test_broken abs(res.Cxx_prof[Ny ÷ 2 + 1] - 1.0) < 0.01

    # --- Report (measured vs analytical profile, several y-stations) ------
    @info("VE 3D planar-Poiseuille canary — setup",
          Nx, Ny, Nz, Fx, u_max_an = res.u_max, gamma_dot_wall = res.gamma_dot_wall,
          Wi_wall = res.Wi_wall, beta, λ, max_steps,
          t_diff = est.t_diff, t_poly = est.t_poly)
    @info("VE 3D Poiseuille — VELOCITY (FINDING: Hermite-source over-coupling)",
          u_meas_peak = maximum(res.profile), u_an_peak = res.u_max,
          ratio = maximum(res.profile) / res.u_max,
          Linf_coupled, Linf_solvent_control = Linf_solvent)
    jc = Ny ÷ 2 + 1
    for (label, j) in (("near-wall", 4), ("mid", 8), ("centre", jc))
        sg = j == 1 ? prof[2] - prof[1] :
             j == Ny ? prof[Ny] - prof[Ny-1] : (prof[j+1] - prof[j-1]) / 2
        cxy_an = λ * sg
        cxx_an = 1 + 2 * (λ * sg)^2
        n1_an  = 2 * res.eta_p * λ * sg^2
        @info("VE 3D Poiseuille — profile @ $label (j=$j)",
              gamma_dot_meas = sg, Wi_local = λ * sg,
              Cxy = res.Cxy_prof[j], Cxy_target_local = cxy_an,
              Cxx = res.Cxx_prof[j], Cxx_target_local = cxx_an,
              N1 = res.N1_prof[j], N1_target_local = n1_an,
              N2 = res.N2_prof[j],
              Cyy = res.Cyy_prof[j], Czz = res.Czz_prof[j],
              Cxz = res.Cxz_prof[j], Cyz = res.Cyz_prof[j])
    end
end
