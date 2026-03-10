using Test
using Kraken

@testset "Multigrid Poisson Solver" begin
    # Test 1: Residual convergence for known RHS (cos solution, Neumann-compatible)
    @testset "Residual for cos solution N=$N" for N in [17, 33, 65]
        dx = 1.0 / (N - 1)
        x = [(i-1)*dx for i in 1:N]
        y = [(j-1)*dx for j in 1:N]
        rhs = [-2π^2*cos(π*x[i])*cos(π*y[j]) for i in 1:N, j in 1:N]
        phi = zeros(N, N)
        phi, niter = Kraken.solve_poisson_mg!(phi, rhs, dx; rtol=1e-8)
        # Check residual on interior points
        res = zeros(N, N)
        for j in 2:N-1, i in 2:N-1
            res[i,j] = (phi[i-1,j]+phi[i+1,j]+phi[i,j-1]+phi[i,j+1]-4phi[i,j])/dx^2 - rhs[i,j]
        end
        @test maximum(abs.(res[2:N-1,2:N-1])) < 1e-4
    end

    # Test 2: Random RHS — residual decreases to tolerance
    @testset "Random RHS convergence" begin
        N = 33; dx = 1.0 / (N-1)
        rhs = randn(N, N); rhs .-= sum(rhs)/(N*N)
        phi = zeros(N, N)
        phi, niter = Kraken.solve_poisson_mg!(phi, rhs, dx; rtol=1e-8)
        # Check residual
        res = zeros(N, N)
        for j in 2:N-1, i in 2:N-1
            res[i,j] = (phi[i-1,j]+phi[i+1,j]+phi[i,j-1]+phi[i,j+1]-4phi[i,j])/dx^2 - rhs[i,j]
        end
        @test maximum(abs.(res[2:N-1,2:N-1])) < 0.02
    end

    # Test 3: MG gives same cavity result as CG
    @testset "MG vs CG cavity" begin
        u_cg, v_cg, p_cg, conv_cg = Kraken.run_cavity(N=17, Re=100.0, max_steps=500, tol=1e-4, poisson_solver=:cg)
        u_mg, v_mg, p_mg, conv_mg = Kraken.run_cavity(N=17, Re=100.0, max_steps=500, tol=1e-4, poisson_solver=:mg)
        @test maximum(abs.(Array(u_cg) - Array(u_mg))) < 0.1
    end
end
