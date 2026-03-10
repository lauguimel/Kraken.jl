using Test
using Kraken

@testset "Helmholtz Solver" begin
    @testset "sigma=0 reduces to identity" begin
        N = 17
        dx = 1.0 / (N - 1)
        rhs = rand(N, N)
        phi = zeros(N, N)
        phi, niter = Kraken.solve_helmholtz!(phi, rhs, dx, 0.0)
        @test maximum(abs.(phi - rhs)) < 1e-6
        @test niter == 0
    end

    @testset "Known analytic solution" begin
        N = 33
        dx = 1.0 / (N - 1)
        sigma = 0.01
        x = [(i - 1) * dx for i in 1:N]
        y = [(j - 1) * dx for j in 1:N]
        # phi_exact = cos(pi*x) * cos(pi*y) — satisfies Neumann BCs (zero normal derivative)
        # (I - sigma*nabla^2) phi = phi + sigma * 2*pi^2 * phi = (1 + 2*sigma*pi^2) * phi
        phi_exact = [cos(π * x[i]) * cos(π * y[j]) for i in 1:N, j in 1:N]
        rhs = (1 + 2 * sigma * π^2) .* phi_exact
        phi = zeros(N, N)
        phi, niter = Kraken.solve_helmholtz!(phi, rhs, dx, sigma)
        @test maximum(abs.(phi - phi_exact)) < 0.01
        @info "Helmholtz analytic test: max error = $(maximum(abs.(phi - phi_exact))), niter = $niter"
    end
end

@testset "DCT Helmholtz Solver" begin
    @testset "Known analytic solution (DCT)" begin
        N = 33
        dx = 1.0 / (N - 1)
        sigma = 0.01
        x = [(i - 1) * dx for i in 1:N]
        y = [(j - 1) * dx for j in 1:N]
        # phi_exact = cos(pi*x) * cos(pi*y) — satisfies Neumann BCs
        # (I - sigma*nabla^2) phi = (1 + 2*sigma*pi^2) * phi
        phi_exact = [cos(π * x[i]) * cos(π * y[j]) for i in 1:N, j in 1:N]
        rhs = (1 + 2 * sigma * π^2) .* phi_exact
        phi = zeros(N, N)
        phi, niter = Kraken.solve_helmholtz_dct!(phi, rhs, dx, sigma)
        @test niter == 0  # Direct solver
        max_err = maximum(abs.(phi - phi_exact))
        @test max_err < 0.01
        @info "DCT Helmholtz analytic test: max error = $max_err"
    end

    @testset "DCT Helmholtz with pre-computed eigenvalues" begin
        N = 33
        dx = 1.0 / (N - 1)
        sigma = 0.005
        inv_dx2 = 1.0 / (dx * dx)
        eig = zeros(N, N)
        for l in 1:N, k in 1:N
            eig[k, l] = 2.0 * inv_dx2 * (cos(π * (k - 1) / (N - 1)) - 1.0) +
                         2.0 * inv_dx2 * (cos(π * (l - 1) / (N - 1)) - 1.0)
        end
        x = [(i - 1) * dx for i in 1:N]
        y = [(j - 1) * dx for j in 1:N]
        phi_exact = [cos(π * x[i]) * cos(π * y[j]) for i in 1:N, j in 1:N]
        rhs = (1 + 2 * sigma * π^2) .* phi_exact
        phi = zeros(N, N)
        phi, niter = Kraken.solve_helmholtz_dct!(phi, rhs, dx, sigma;
                                                   poisson_eigenvalues=eig)
        @test niter == 0
        @test maximum(abs.(phi - phi_exact)) < 0.01
    end

    @testset "DCT vs CG Helmholtz agreement" begin
        N = 17
        dx = 1.0 / (N - 1)
        sigma = 0.01
        rhs = rand(N, N)
        phi_cg = zeros(N, N)
        phi_dct = zeros(N, N)
        phi_cg, _ = Kraken.solve_helmholtz!(phi_cg, rhs, dx, sigma; rtol=1e-10)
        phi_dct, _ = Kraken.solve_helmholtz_dct!(phi_dct, rhs, dx, sigma)
        max_diff = maximum(abs.(phi_cg - phi_dct))
        @test max_diff < 0.01
        @info "DCT vs CG Helmholtz max diff = $max_diff"
    end
end

@testset "Implicit Projection" begin
    @testset "Cavity convergence with implicit scheme" begin
        u, v, p, conv = Kraken.run_cavity(N=17, Re=100.0, max_steps=3000,
                                           tol=1e-4, time_scheme=:implicit,
                                           verbose=true)
        @test conv == true
    end

    @testset "Implicit vs explicit similarity" begin
        u_imp, v_imp, _, conv_imp = Kraken.run_cavity(N=17, Re=100.0, max_steps=3000,
                                                        tol=1e-4, time_scheme=:implicit)
        u_exp, v_exp, _, conv_exp = Kraken.run_cavity(N=17, Re=100.0, max_steps=5000,
                                                        tol=1e-4, time_scheme=:explicit)
        @test conv_imp
        @test conv_exp
        # Both should converge to similar steady state (same physics)
        @test maximum(abs.(Array(u_imp) - Array(u_exp))) < 0.1
        @test maximum(abs.(Array(v_imp) - Array(v_exp))) < 0.1
        @info "Implicit vs explicit max u diff = $(maximum(abs.(Array(u_imp) - Array(u_exp))))"
    end
end
