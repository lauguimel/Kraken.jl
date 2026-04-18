using Test
using Kraken

@testset "Conformation sphere 3D (TRT-LBM viscoelastic, D3Q19)" begin

    # ----------------------------------------------------------------
    # 1. Smoke test: small CPU run, no NaN, sane macro fields
    # ----------------------------------------------------------------
    @testset "smoke test (Oldroyd-B, tiny CPU)" begin
        r = run_conformation_sphere_libb_3d(;
                Nx=24, Ny=12, Nz=12, radius=3,
                u_in=0.04, ν_s=0.04, ν_p=0.02, lambda=5.0,
                inlet=:parabolic_y,
                max_steps=200, avg_window=50)

        @test all(isfinite, r.ux)
        @test all(isfinite, r.uy)
        @test all(isfinite, r.uz)
        @test all(isfinite, r.tau_p_xx)
        @test all(isfinite, r.tau_p_xy)
        @test all(isfinite, r.tau_p_xz)
        @test all(isfinite, r.C_xx)
        @test all(isfinite, r.C_yz)
        @test isfinite(r.Cd) && r.Cd > 0
        @test r.beta ≈ 0.04 / (0.04 + 0.02) atol=1e-12

        # No flow inside sphere
        @test maximum(abs.(r.ux[r.is_solid])) < 1e-6
    end

    # ----------------------------------------------------------------
    # 2. Newtonian-limit consistency: at very low Wi the conformation
    # solver should give Cd ≈ Newtonian Cd (run_sphere_libb_3d) with
    # ν_total = ν_s + ν_p. This confirms the Hermite source injects the
    # polymer stress at the right magnitude.
    # ----------------------------------------------------------------
    @testset "low-Wi Newtonian consistency" begin
        # Grid kept <= 40^3-ish: `run_sphere_libb_3d` (the Newtonian
        # reference) segfaults at 60x30x30 on this machine — pre-existing
        # issue tracked separately, unrelated to the viscoelastic port.
        Nx, Ny, Nz = 40, 20, 20
        radius = 4
        u_in   = 0.04
        ν_s = 0.06
        ν_p = 0.02
        ν_total = ν_s + ν_p
        lambda  = 2.0    # Wi ≈ 5e-3 — well within Newtonian regime
        max_steps = 2_000

        ref = run_sphere_libb_3d(; Nx=Nx, Ny=Ny, Nz=Nz, radius=radius,
                                    u_in=u_in, ν=ν_total,
                                    inlet=:parabolic_y,
                                    max_steps=max_steps,
                                    avg_window=max_steps ÷ 5)

        res = run_conformation_sphere_libb_3d(; Nx=Nx, Ny=Ny, Nz=Nz, radius=radius,
                                                u_in=u_in, ν_s=ν_s, ν_p=ν_p,
                                                lambda=lambda,
                                                inlet=:parabolic_y,
                                                max_steps=max_steps,
                                                avg_window=max_steps ÷ 5)

        @info "3D Newtonian-limit sphere" Re=ref.Cd Cd_ref=ref.Cd Cd_visco=res.Cd Wi=res.Wi
        @test isfinite(res.Cd) && res.Cd > 0
        # Tolerance loose because: short run, small grid, finite Wi correction.
        # We just verify the magnitude is right (no factor-of-2 bug from
        # the Hermite source double-counting or polymer term sign flip).
        # Tighter tolerance (<5%) needs ≥80³ resolution and ≥50k steps —
        # see hpc/run_sphere_oldroyd_3d.pbs for the H100 validation run.
        rel = abs(res.Cd - ref.Cd) / ref.Cd
        @test rel < 0.50
    end
end
