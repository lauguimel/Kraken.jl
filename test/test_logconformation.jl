using Test
using Kraken, KernelAbstractions

@testset "Log-conformation TRT-LBM (Fattal-Kupferman)" begin

    @testset "Ψ ↔ C inverse (symmetric 2×2)" begin
        # Round-trip: C → Ψ → C must be identity at machine precision.
        Nx, Ny = 8, 8
        C_xx = 1.5 .+ 0.3 * randn(Nx, Ny)
        C_xy = 0.1 * randn(Nx, Ny)
        C_yy = 1.2 .+ 0.3 * randn(Nx, Ny)
        # Force positive-definiteness: C_xx·C_yy − C_xy² > 0
        C_xx .= max.(C_xx, 0.2 .+ abs.(C_xy))
        C_yy .= max.(C_yy, 0.2 .+ abs.(C_xy))

        Ψ_xx = similar(C_xx); Ψ_xy = similar(C_xy); Ψ_yy = similar(C_yy)
        C_xx2 = similar(C_xx); C_xy2 = similar(C_xy); C_yy2 = similar(C_yy)

        C_to_psi_2d!(Ψ_xx, Ψ_xy, Ψ_yy, C_xx, C_xy, C_yy)
        psi_to_C_2d!(C_xx2, C_xy2, C_yy2, Ψ_xx, Ψ_xy, Ψ_yy)

        @test maximum(abs.(C_xx2 .- C_xx)) < 1e-12
        @test maximum(abs.(C_xy2 .- C_xy)) < 1e-12
        @test maximum(abs.(C_yy2 .- C_yy)) < 1e-12
    end

    @testset "Ψ = 0 ⇒ C = I" begin
        Nx, Ny = 4, 4
        Ψ_xx = zeros(Nx, Ny); Ψ_xy = zeros(Nx, Ny); Ψ_yy = zeros(Nx, Ny)
        C_xx = similar(Ψ_xx); C_xy = similar(Ψ_xy); C_yy = similar(Ψ_yy)
        psi_to_C_2d!(C_xx, C_xy, C_yy, Ψ_xx, Ψ_xy, Ψ_yy)
        @test all(C_xx .≈ 1.0)
        @test all(C_xy .≈ 0.0)
        @test all(C_yy .≈ 1.0)
    end

    @testset "Low-Wi consistency: direct-C ≈ log-conformation" begin
        # At low Wi, the two formulations must agree to high precision
        # in the cylinder flow. Verifies the log-conformation source
        # (with Loewner derivative) against the validated direct-C source.
        common = (; Nx=240, Ny=60, radius=8, u_mean=0.02, ν_s=0.06,
                    inlet=:parabolic, ρ_out=1.0, tau_plus=1.0,
                    max_steps=6000, avg_window=1000,
                    backend=KernelAbstractions.CPU(), FT=Float64)

        m_direct  = OldroydB(G=0.04/5.0, λ=5.0)
        m_logconf = LogConfOldroydB(G=0.04/5.0, λ=5.0)

        r1 = run_conformation_cylinder_libb_2d(; common..., polymer_model=m_direct)
        r2 = run_conformation_cylinder_libb_2d(; common...,
            polymer_model=m_logconf, polymer_bc=LogFieldWallBC())

        rel_diff = abs(r1.Cd - r2.Cd) / r1.Cd
        @info "log-conformation low-Wi" Cd_direct=round(r1.Cd, digits=4) Cd_logconf=round(r2.Cd, digits=4) rel_diff
        # This compares different wall closures: direct-C uses strict CNEBB,
        # logconf uses the field-pinned log-space closure. Keep it as a
        # coarse end-to-end smoke test, not a wall-BC equivalence proof.
        @test rel_diff < 0.02   # < 2 % at Wi ≈ 0.012
        @test isfinite(r2.Cd) && r2.Cd > 0
        @test r1.drag_mode === :post_source_mea
        @test r1.hermite_source_mode === :liu_direct
        @test r1.solvent_source_mode === :post_collision
        @test r1.conformation_magic == 1e-6
        @test r1.momentum_exchange_mode === :mei_reconstruct
        @test r1.Cd == r1.Cd_mea_post_source
        @test r1.Cd_split_explicit ≈ r1.Cd_s + r1.Cd_p
    end

    @testset "source-scaled force mode requires explicit diagnostic opt-in" begin
        model = OldroydB(G=0.04/5.0, λ=5.0)
        @test_throws ErrorException run_conformation_cylinder_libb_2d(;
            Nx=24, Ny=12, radius=2, u_mean=0.01, ν_s=0.06,
            polymer_model=model, max_steps=1, avg_window=1,
            drag_mode=:source_scaled_mea)
    end

    @testset "diagnostic polymer wall BC requires explicit opt-in" begin
        model = OldroydB(G=0.04/5.0, λ=5.0)
        @test_throws ErrorException run_conformation_cylinder_libb_2d(;
            Nx=24, Ny=12, radius=2, u_mean=0.01, ν_s=0.06,
            polymer_model=model, polymer_bc=CNEBBEqGradient(),
            max_steps=1, avg_window=1)
    end

    @testset "unsupported conformation collision windows require diagnostic opt-in" begin
        model = OldroydB(G=0.04/5.0, λ=5.0)
        @test_throws ErrorException run_conformation_cylinder_libb_2d(;
            Nx=24, Ny=12, radius=2, u_mean=0.01, ν_s=0.06,
            polymer_model=model, tau_plus=0.50001,
            conformation_collision=:trt, max_steps=1, avg_window=1)
        @test_throws ErrorException run_conformation_cylinder_libb_2d(;
            Nx=24, Ny=12, radius=2, u_mean=0.01, ν_s=0.06,
            polymer_model=model, tau_plus=1.0,
            conformation_collision=:regularized, max_steps=1, avg_window=1)
    end

    @testset "LogFieldWallBC is the validated log-conformation wall closure" begin
        model = LogConfOldroydB(G=0.04/5.0, λ=5.0)
        @test_throws ErrorException run_conformation_cylinder_libb_2d(;
            Nx=24, Ny=12, radius=2, u_mean=0.01, ν_s=0.06,
            polymer_model=model, max_steps=1, avg_window=1)
        result = run_conformation_cylinder_libb_2d(;
            Nx=24, Ny=12, radius=2, u_mean=0.01, ν_s=0.06,
            polymer_model=model, polymer_bc=LogFieldWallBC(),
            max_steps=1, avg_window=1)
        @test result.first_nonfinite_step == 0
        @test isfinite(result.Cd)
    end

    @testset "LogFieldWallBC is rejected for direct-C validation" begin
        model = OldroydB(G=0.04/5.0, λ=5.0)
        @test_throws ErrorException run_conformation_cylinder_libb_2d(;
            Nx=24, Ny=12, radius=2, u_mean=0.01, ν_s=0.06,
            polymer_model=model, polymer_bc=LogFieldWallBC(),
            max_steps=1, avg_window=1)
    end

    @testset "log-conformation accepts production gradient stencils" begin
        model = LogConfOldroydB(G=0.04/5.0, λ=5.0)
        for mode in (:embedded_axis, :wallfit4)
            result = run_conformation_cylinder_libb_2d(;
                Nx=24, Ny=12, radius=2, u_mean=0.01, ν_s=0.06,
                polymer_model=model, polymer_bc=LogFieldWallBC(),
                conformation_gradient_mode=mode,
                conformation_gradient_max_terms=mode === :embedded_axis ? 4 : 64,
                max_steps=1, avg_window=1)
            @test result.conformation_gradient_mode === mode
            @test result.first_nonfinite_step == 0
            @test isfinite(result.Cd)
        end
    end
end
