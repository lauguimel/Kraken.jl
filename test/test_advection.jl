using Test
using Kraken

@testset "TVD Van Leer advection" begin
    N = 34  # interior = 32
    dx = 1.0 / (N - 2)

    # Uniform flow u=1, v=0 advecting a linear profile f=x → exact ∂f/∂x = 1
    u = ones(N, N)
    v = zeros(N, N)
    f = [(i - 1.5) * dx for i in 1:N, j in 1:N]  # linear in x
    out = zeros(N, N)
    Kraken.advect!(out, u, v, f, dx; scheme=:tvd)
    # For a linear profile, TVD should give exact result (no limiting needed)
    interior = out[2:N-1, 2:N-1]
    @test all(abs.(interior .- 1.0) .< 0.01)

    # TVD should be more accurate than upwind for smooth profiles
    # Test with sin(2πx) profile
    f_sin = [sin(2π * (i - 1.5) * dx) for i in 1:N, j in 1:N]
    exact_dfdx = [2π * cos(2π * (i - 1.5) * dx) for i in 1:N, j in 1:N]

    out_upwind = zeros(N, N)
    out_tvd = zeros(N, N)
    Kraken.advect!(out_upwind, u, v, f_sin, dx; scheme=:upwind)
    Kraken.advect!(out_tvd, u, v, f_sin, dx; scheme=:tvd)

    # Use L2 norm for robust comparison (max-norm can be dominated by boundary effects)
    diff_upwind = out_upwind[3:N-2, 3:N-2] .- exact_dfdx[3:N-2, 3:N-2]
    diff_tvd = out_tvd[3:N-2, 3:N-2] .- exact_dfdx[3:N-2, 3:N-2]
    l2_upwind = sqrt(sum(diff_upwind .^ 2) / length(diff_upwind))
    l2_tvd = sqrt(sum(diff_tvd .^ 2) / length(diff_tvd))
    @test l2_tvd < l2_upwind  # TVD should be more accurate in L2 norm
end

@testset "Upwind advection backward compatibility" begin
    N = 34
    dx = 1.0 / (N - 2)

    u = ones(N, N)
    v = zeros(N, N)
    f = [(i - 1.5) * dx for i in 1:N, j in 1:N]

    out_default = zeros(N, N)
    out_explicit = zeros(N, N)
    Kraken.advect!(out_default, u, v, f, dx)
    Kraken.advect!(out_explicit, u, v, f, dx; scheme=:upwind)

    @test out_default == out_explicit
end
