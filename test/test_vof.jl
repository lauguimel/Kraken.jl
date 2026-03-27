using Test
using Kraken

@testset "VOF PLIC 2D" begin

    @testset "Interface normal computation" begin
        # Smooth circular interface using tanh profile
        N = 32; R = 10.0; cx = N÷2; cy = N÷2; W = 2.0  # interface width
        C = zeros(Float64, N, N)
        for j in 1:N, i in 1:N
            r = sqrt(Float64((i-cx)^2 + (j-cy)^2))
            C[i,j] = 0.5 * (1.0 - tanh((r - R) / W))
        end

        nx = zeros(Float64, N, N)
        ny = zeros(Float64, N, N)
        compute_vof_normal_2d!(nx, ny, C, N, N)

        # Normal = ∇C/|∇C| → points from gas (C=0) toward liquid (C=1)
        # At (cx+R, cy): C decreases in +x → ∂C/∂x < 0 → nx < 0
        # So normal points INWARD (toward liquid center) — this is the convention
        i_check = cx + Int(round(R)); j_check = cy
        @test nx[i_check, j_check] < -0.5  # points inward (-x direction)
        @test abs(ny[i_check, j_check]) < 0.5
        @info "VOF normal at (cx+R, cy): nx=$(round(nx[i_check,j_check], digits=3))"
    end

    @testset "Height function curvature" begin
        # Smooth circular interface
        N = 64; R = 15.0; cx = N÷2; cy = N÷2; W = 2.0
        C = zeros(Float64, N, N)
        for j in 1:N, i in 1:N
            r = sqrt(Float64((i-cx)^2 + (j-cy)^2))
            C[i,j] = 0.5 * (1.0 - tanh((r - R) / W))
        end

        nx = zeros(Float64, N, N)
        ny = zeros(Float64, N, N)
        κ = zeros(Float64, N, N)
        compute_vof_normal_2d!(nx, ny, C, N, N)
        compute_hf_curvature_2d!(κ, C, nx, ny, N, N)

        # Sample curvature at interface cells (0.1 < C < 0.9)
        κ_interface = Float64[]
        for j in 1:N, i in 1:N
            if C[i,j] > 0.1 && C[i,j] < 0.9 && abs(κ[i,j]) > 1e-6
                push!(κ_interface, abs(κ[i,j]))
            end
        end

        @test !isempty(κ_interface)
        if !isempty(κ_interface)
            κ_mean = sum(κ_interface) / length(κ_interface)
            κ_analytical = 1.0 / R
            @test κ_mean > 0  # positive curvature
            @info "HF curvature: κ_mean=$(round(κ_mean, digits=4)), analytical=$(round(κ_analytical, digits=4))"
        end
    end

    @testset "Capillary bridge relaxation (2D)" begin
        # In 2D (planar), Rayleigh-Plateau instability does NOT occur:
        # perturbations are STABLE (surface tension smooths them out).
        # This validates that the VOF-LBM surface tension acts correctly.
        # r_min should INCREASE toward R0 as the perturbation decays.
        result = run_plateau_pinch_2d(; Nx=128, Ny=32, R0=10, λ_ratio=4.5, ε=0.1,
                                       σ=0.005, ν=0.05, ρ_l=1.0, ρ_g=0.1,
                                       max_steps=3000, output_interval=200)

        @test !any(isnan, result.C)
        @test !any(isnan, result.ρ)

        # In 2D: perturbation should decay → r_min increases toward R0
        if length(result.r_min) >= 3
            @test result.r_min[end] >= result.r_min[1]  # relaxation (correct 2D physics)
            @info "Capillary relaxation: r_min $(round(result.r_min[1], digits=2)) → $(round(result.r_min[end], digits=2)) (R0=$(result.R0))"
        end
    end

    @testset "Static droplet stability" begin
        # Low density ratio for stability; just verify no NaN
        result = run_static_droplet_2d(; N=64, R=15, σ=0.001, ν=0.1,
                                        ρ_l=1.0, ρ_g=0.5, max_steps=500)

        @test !any(isnan, result.ρ)
        @test !any(isnan, result.C)
        @info "Static droplet: max|u|=$(round(result.max_u_spurious, sigdigits=3)), ρ range=$(round.(extrema(result.ρ), digits=3))"
    end
end
