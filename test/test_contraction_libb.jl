using Test
using Kraken

@testset "Contraction LI-BB driver (4:1 planar viscoelastic)" begin

    # ----------------------------------------------------------------
    # 1. Geometry helper: q_wall + is_solid
    # ----------------------------------------------------------------
    @testset "precompute_q_wall_contraction_2d geometry" begin
        H_out = 8; β_c = 4
        Nx = (4 + 4) * H_out          # short upstream + downstream for the test
        Ny = β_c * H_out              # = 32
        i_step = 4 * H_out + 1        # = 33
        j_low  = (Ny - H_out) ÷ 2 + 1          # = 13 (centred)
        j_high = j_low + H_out - 1             # = 20
        q_wall, is_solid = precompute_q_wall_contraction_2d(
            Nx, Ny; i_step=i_step, j_low=j_low, j_high=j_high)

        @test size(q_wall)   == (Nx, Ny, 9)
        @test size(is_solid) == (Nx, Ny)

        # Upstream block: no internal solid
        @test !any(is_solid[1:i_step-1, :])

        # Downstream solid blocks above + below outlet
        @test all(is_solid[i_step:Nx, 1:j_low-1])
        @test all(is_solid[i_step:Nx, j_high+1:Ny])
        # Downstream outlet remains fluid
        @test !any(is_solid[i_step:Nx, j_low:j_high])

        # q_wall = 0.5 on the cell just below the south outlet floor at i ≥ i_step
        # (cell (i_step, j_low) sees a solid neighbour at (i_step, j_low-1) along q=5)
        @test q_wall[i_step, j_low, 5] ≈ 0.5
        @test q_wall[i_step, j_high, 3] ≈ 0.5

        # Step face: cell just upstream of step (i_step-1, j_low-1) sees solid east
        # neighbour at (i_step, j_low-1) along q=2
        @test q_wall[i_step-1, j_low-1, 2] ≈ 0.5
    end

    # ----------------------------------------------------------------
    # 2. Driver smoke test: tiny CPU run, no NaN, sane macro fields
    # ----------------------------------------------------------------
    @testset "driver smoke test (Oldroyd-B, tiny CPU)" begin
        r = run_conformation_contraction_libb_2d(;
                H_out=8, β_c=4, L_up=4, L_down=8,
                u_out_mean=0.02, ν_s=0.354, ν_p=0.246, lambda=5.0,
                max_steps=400, avg_window=100)

        @test all(isfinite, r.ux)
        @test all(isfinite, r.uy)
        @test all(isfinite, r.tau_p_xx)
        @test all(isfinite, r.C_xx)
        @test r.beta ≈ 0.354 / (0.354 + 0.246) atol=1e-12

        # Mass conservation: mean upstream u ≈ u_out_mean / β_c
        u_in_mean_target = 0.02 / 4
        u_in_mean = sum(r.ux[2, :]) / r.Ny
        @test isapprox(u_in_mean, u_in_mean_target; rtol=0.30)

        # No flow inside solid blocks
        @test maximum(abs.(r.ux[r.is_solid])) < 1e-6
    end

    # ----------------------------------------------------------------
    # 3. Driver smoke test with log-conformation polymer model
    # ----------------------------------------------------------------
    @testset "driver smoke test (log-conformation)" begin
        r = run_conformation_contraction_libb_2d(;
                H_out=8, β_c=4, L_up=4, L_down=8,
                u_out_mean=0.02, ν_s=0.354,
                polymer_model=LogConfOldroydB(G=0.246/5.0, λ=5.0),
                max_steps=400, avg_window=100)

        @test all(isfinite, r.ux)
        @test all(isfinite, r.tau_p_xx)
        @test all(isfinite, r.C_xx)
        @test r.Wi ≈ 5.0 * 0.02 / (8/2) atol=1e-12
    end

    # ----------------------------------------------------------------
    # 4. Post-processing helpers don't crash on a tiny field
    # ----------------------------------------------------------------
    @testset "post-processing helpers" begin
        r = run_conformation_contraction_libb_2d(;
                H_out=8, β_c=4, L_up=4, L_down=8,
                u_out_mean=0.02, ν_s=0.354, ν_p=0.246, lambda=5.0,
                max_steps=200, avg_window=50)
        X_R, j_probe = vortex_length_contraction_2d(r.ux, r.uy, r.is_solid;
                          i_step=r.i_step, j_low=r.j_low, j_high=r.j_high,
                          side=:south)
        @test X_R ≥ 0
        @test 1 ≤ j_probe ≤ r.Ny

        N1 = outlet_centerline_N1_contraction_2d(r.tau_p_xx, r.tau_p_yy;
                  i_step=r.i_step, j_low=r.j_low, j_high=r.j_high)
        @test length(N1) == r.Nx - r.i_step + 1
        @test all(isfinite, N1)
    end
end
