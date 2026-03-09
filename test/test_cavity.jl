using Test
using Kraken
using LinearAlgebra

@testset "Lid-driven cavity Re=100" begin
    # Run cavity simulation
    N = 64
    u, v, p, converged = Kraken.run_cavity(N=N, Re=100.0, cfl=0.2,
                                            max_steps=20000, tol=1e-7,
                                            verbose=true)

    @test converged

    # --- Compare u(y) at x=0.5 with Ghia et al. 1982 ---
    # Ghia reference data for Re=100
    y_ghia = [1.0, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344,
              0.6172, 0.5, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703,
              0.0625, 0.0547, 0.0]
    u_ghia = [1.0, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332,
              -0.13641, -0.20581, -0.21090, -0.15662, -0.10150, -0.06434,
              -0.04775, -0.04192, -0.03717, 0.0]

    dx = 1.0 / (N - 1)

    # Extract u profile at x=0.5 (nearest grid column)
    i_mid = round(Int, 0.5 / dx) + 1  # 1-indexed
    y_grid = range(0.0, 1.0, length=N)
    u_profile = u[i_mid, :]

    # Interpolate simulation results to Ghia y-locations
    u_interp = zeros(length(y_ghia))
    for (k, yg) in enumerate(y_ghia)
        # Linear interpolation
        if yg <= 0.0
            u_interp[k] = u_profile[1]
        elseif yg >= 1.0
            u_interp[k] = u_profile[N]
        else
            j_float = yg / dx + 1.0
            j_lo = floor(Int, j_float)
            j_hi = j_lo + 1
            j_lo = clamp(j_lo, 1, N)
            j_hi = clamp(j_hi, 1, N)
            w = j_float - floor(j_float)
            u_interp[k] = (1 - w) * u_profile[j_lo] + w * u_profile[j_hi]
        end
    end

    # Compute L2 relative error
    l2_error = norm(u_interp .- u_ghia) / norm(u_ghia)
    println("L2 relative error vs Ghia: $(round(l2_error * 100, digits=2))%")
    println("Max pointwise error: $(round(maximum(abs.(u_interp .- u_ghia)), digits=4))")

    @test l2_error < 0.05  # < 5% L2 error
end

@testset "Lid-driven cavity — Metal GPU" begin
    gpu_available = false
    try
        @eval using Metal
        if Metal.functional()
            gpu_available = true
        end
    catch
    end

    if !gpu_available
        @info "Metal not available, skipping GPU cavity tests"
        @test_skip false
    else
        using KernelAbstractions

        # Smoke test: run a few steps on both CPU and Metal GPU,
        # compare results (full convergence is too slow on Metal)
        N = 16
        n_steps = 10

        u_cpu, v_cpu, p_cpu, _ = Kraken.run_cavity(
            N=N, Re=100.0, cfl=0.2, max_steps=n_steps, tol=1e-20, verbose=false,
            backend=KernelAbstractions.CPU(), float_type=Float32)

        u_gpu, v_gpu, p_gpu, _ = Kraken.run_cavity(
            N=N, Re=100.0, cfl=0.2, max_steps=n_steps, tol=1e-20, verbose=false,
            backend=Metal.MetalBackend(), float_type=Float32)

        # Compare results after same number of steps
        diff_u = maximum(abs.(Array(u_gpu) .- u_cpu))
        diff_v = maximum(abs.(Array(v_gpu) .- v_cpu))
        @test diff_u < 1e-3
        @test diff_v < 1e-3
        @info "Metal cavity smoke test: max u diff = $diff_u, max v diff = $diff_v"
    end
end
