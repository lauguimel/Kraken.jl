using Test
using Kraken
using LinearAlgebra
using Statistics
using FFTW

@testset "Poisson FFT — periodic BCs" begin
    # Analytical: φ(x,y) = sin(2πx)sin(2πy)
    # ∇²φ = -8π²sin(2πx)sin(2πy)
    # The FFT solver solves the discrete system exactly (to machine precision).
    # The discrete solution converges to the continuous one at O(h²).

    # Test 1: discrete residual should be near machine precision
    @testset "Discrete residual ≈ 0" begin
        N = 64
        dx = 1.0 / N
        xs = range(0.0, step=dx, length=N)
        ys = range(0.0, step=dx, length=N)

        f = [-8π^2 * sin(2π * x) * sin(2π * y) for x in xs, y in ys]
        phi = zeros(N, N)
        Kraken.solve_poisson_fft!(phi, f, dx)

        # Apply discrete Laplacian to phi and check it matches f
        # Periodic Laplacian: (phi[i+1,j] + phi[i-1,j] + phi[i,j+1] + phi[i,j-1] - 4*phi[i,j]) / dx²
        inv_dx2 = 1.0 / (dx * dx)
        residual = zeros(N, N)
        for j in 1:N, i in 1:N
            ip = mod1(i + 1, N)
            im = mod1(i - 1, N)
            jp = mod1(j + 1, N)
            jm = mod1(j - 1, N)
            residual[i, j] = (phi[ip, j] + phi[im, j] + phi[i, jp] + phi[i, jm] - 4 * phi[i, j]) * inv_dx2 - f[i, j]
        end
        @test norm(residual, Inf) < 1e-10
    end

    # Test 2: O(h²) convergence to analytical solution
    @testset "O(h²) convergence" begin
        errors = Float64[]
        for N in [32, 64, 128]
            dx = 1.0 / N
            xs = range(0.0, step=dx, length=N)
            ys = range(0.0, step=dx, length=N)

            f = [-8π^2 * sin(2π * x) * sin(2π * y) for x in xs, y in ys]
            phi_exact = [sin(2π * x) * sin(2π * y) for x in xs, y in ys]

            phi = zeros(N, N)
            Kraken.solve_poisson_fft!(phi, f, dx)

            phi .-= mean(phi)
            phi_exact .-= mean(phi_exact)

            push!(errors, norm(phi .- phi_exact, Inf))
        end

        # Check O(h²) convergence rate
        for k in 1:length(errors)-1
            rate = log(errors[k] / errors[k+1]) / log(2)
            @test 1.9 < rate < 2.1
        end
    end
end

@testset "Poisson CG — Dirichlet BCs" begin
    # Analytical: φ(x,y) = sin(πx)sin(πy) on [0,1]²
    # ∇²φ = -2π²sin(πx)sin(πy)
    # Dirichlet BC: φ=0 on boundary (satisfied by sin(πx)sin(πy))

    for N in [34, 66, 130]  # N = n_interior + 2
        dx = 1.0 / (N - 1)
        xs = range(0.0, 1.0, length=N)
        ys = range(0.0, 1.0, length=N)

        f = [-2π^2 * sin(π * x) * sin(π * y) for x in xs, y in ys]
        phi_exact = [sin(π * x) * sin(π * y) for x in xs, y in ys]

        phi = zeros(N, N)
        phi, niter = Kraken.solve_poisson_cg!(phi, f, dx; maxiter=1000, rtol=1e-12)

        # Error on interior points (discretization error is O(h²))
        err = norm(phi[2:N-1, 2:N-1] .- phi_exact[2:N-1, 2:N-1], Inf)
        @test err < 1e-3  # O(h²) discretization error
        @test niter < 500
    end
end

@testset "Poisson CG — convergence on 128×128" begin
    N = 130  # 128 interior points
    dx = 1.0 / (N - 1)
    xs = range(0.0, 1.0, length=N)
    ys = range(0.0, 1.0, length=N)

    f = [-2π^2 * sin(π * x) * sin(π * y) for x in xs, y in ys]
    phi_exact = [sin(π * x) * sin(π * y) for x in xs, y in ys]

    phi = zeros(N, N)
    phi, niter = Kraken.solve_poisson_cg!(phi, f, dx; maxiter=1000, rtol=1e-12)

    err = norm(phi[2:N-1, 2:N-1] .- phi_exact[2:N-1, 2:N-1], Inf)
    @test err < 1e-3  # O(h²) discretization error
    @test niter < 500

    @info "CG 128×128: niter=$niter, error=$err"
end
