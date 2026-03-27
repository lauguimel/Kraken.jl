using Test
using Kraken

@testset "CLSVOF 2D" begin

    @testset "LS redistanciation" begin
        # Start with φ = R² - r² (NOT a signed distance)
        # After redistanciation, should approximate φ = R - r
        N = 64; R = 15.0; cx = N÷2; cy = N÷2
        phi = zeros(Float64, N, N)
        for j in 1:N, i in 1:N
            r = sqrt(Float64((i-cx)^2 + (j-cy)^2))
            phi[i,j] = R^2 - r^2
        end
        phi_work = similar(phi)
        phi0 = similar(phi)
        reinit_ls_2d!(phi, phi_work, phi0, N, N; n_iter=30, dtau=0.5)

        # Near the interface, φ should approximate the signed distance R - r
        errors = Float64[]
        for j in 1:N, i in 1:N
            r = sqrt(Float64((i-cx)^2 + (j-cy)^2))
            if abs(r - R) < 5.0  # within 5 cells of interface
                push!(errors, abs(phi[i,j] - (R - r)))
            end
        end
        mean_err = sum(errors) / length(errors)
        @test mean_err < 2.0  # within 2 lattice spacings
        @info "Redistanciation: mean error near interface = $(round(mean_err, digits=3))"
    end

    @testset "LS curvature (circle)" begin
        # Exact signed distance for circle: φ = R - r
        N = 64; R = 15.0; cx = N÷2; cy = N÷2
        phi = zeros(Float64, N, N)
        for j in 1:N, i in 1:N
            r = sqrt(Float64((i-cx)^2 + (j-cy)^2))
            phi[i,j] = R - r
        end
        κ = zeros(Float64, N, N)
        curvature_ls_2d!(κ, phi, N, N)

        # Near interface (|φ| < 2), κ should be close to 1/R
        κ_vals = Float64[]
        for j in 1:N, i in 1:N
            if abs(phi[i,j]) < 2.0 && abs(κ[i,j]) > 1e-6
                push!(κ_vals, κ[i,j])
            end
        end
        κ_mean = sum(κ_vals) / length(κ_vals)
        κ_exact = 1.0 / R
        rel_err = abs(κ_mean - κ_exact) / κ_exact

        @test rel_err < 0.15  # within 15%
        @info "LS curvature: κ_mean=$(round(κ_mean, digits=5)), exact=$(round(κ_exact, digits=5)), error=$(round(rel_err*100, digits=1))%"
    end

    @testset "ls_from_vof + redistanciation" begin
        # Reconstruct φ from C, redistance, check φ ≈ 0 where C ≈ 0.5
        N = 64; R = 15.0; cx = N÷2; cy = N÷2
        C = zeros(Float64, N, N)
        for j in 1:N, i in 1:N
            r = sqrt(Float64((i-cx)^2 + (j-cy)^2))
            C[i,j] = 0.5 * (1.0 - tanh((r - R) / 2.0))
        end
        phi = zeros(Float64, N, N)
        ls_from_vof_2d!(phi, C, N, N)

        phi_work = similar(phi)
        phi0 = similar(phi)
        reinit_ls_2d!(phi, phi_work, phi0, N, N; n_iter=10, dtau=0.5)

        # At interface cells (0.3 < C < 0.7), φ should be near zero
        phi_at_interface = Float64[]
        for j in 1:N, i in 1:N
            if 0.3 < C[i,j] < 0.7
                push!(phi_at_interface, abs(phi[i,j]))
            end
        end
        mean_abs_phi = sum(phi_at_interface) / length(phi_at_interface)
        @test mean_abs_phi < 3.0
        @info "φ from VOF: mean |φ| at interface = $(round(mean_abs_phi, digits=2))"
    end

    @testset "CLSVOF static droplet" begin
        result = run_static_droplet_clsvof_2d(; N=64, R=15, σ=0.001, ν=0.1,
                                                ρ_l=1.0, ρ_g=0.5, max_steps=500)

        @test !any(isnan, result.ρ)
        @test !any(isnan, result.C)
        @test !any(isnan, result.phi)

        @info "CLSVOF droplet: max|u|=$(round(result.max_u_spurious, sigdigits=3)), Δp=$(round(result.Δp, sigdigits=3)), Δp_th=$(round(result.Δp_analytical, sigdigits=3))"
    end

    @testset "RP axisym CLSVOF pinch-off" begin
        # Gerris-like setup: small perturbation, LS curvature drives instability
        result = run_rp_clsvof_2d(; Nz=128, Nr=30, R0=12, λ_ratio=7.0, ε=0.05,
                                    σ=0.01, ν=1/6, ρ_l=1.0, ρ_g=0.5,
                                    max_steps=10000, output_interval=1000)

        @test !any(isnan, result.C)
        @test !any(isnan, result.phi)

        # r_min should DECREASE (jet thins toward pinch-off)
        @test result.r_min[end] < result.r_min[1]

        thinning = (1 - result.r_min[end] / result.r_min[1]) * 100
        @info "RP CLSVOF: r_min $(round(result.r_min[1], digits=1)) → $(round(result.r_min[end], digits=1)) ($(round(thinning, digits=0))% thinning)"
    end
end
