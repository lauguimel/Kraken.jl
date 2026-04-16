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
                    max_steps=3000, avg_window=500,
                    backend=KernelAbstractions.CPU(), FT=Float64)

        m_direct  = OldroydB(G=0.04/5.0, λ=5.0)
        m_logconf = LogConfOldroydB(G=0.04/5.0, λ=5.0)

        r1 = run_conformation_cylinder_libb_2d(; common..., polymer_model=m_direct)
        r2 = run_conformation_cylinder_libb_2d(; common..., polymer_model=m_logconf)

        rel_diff = abs(r1.Cd - r2.Cd) / r1.Cd
        @info "log-conformation low-Wi" Cd_direct=round(r1.Cd, digits=4) Cd_logconf=round(r2.Cd, digits=4) rel_diff
        @test rel_diff < 0.01   # < 1 % at Wi ≈ 0.012
        @test isfinite(r2.Cd) && r2.Cd > 0
    end
end
