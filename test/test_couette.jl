using Test
using Kraken

@testset "Couette 2D" begin
    Nx, Ny = 4, 32
    ν = 0.1
    u_wall = 0.05

    result = run_couette_2d(; Nx=Nx, Ny=Ny, ν=ν, u_wall=u_wall, max_steps=10000)

    # Analytical linear profile (half-way BB: walls at y=0.5 and y=Ny+0.5)
    # j=1 → u ≈ u_wall, j=Ny → u ≈ 0
    # u(y) = u_wall * (Ny + 0.5 - j) / Ny
    u_analytical = [u_wall * (Ny + 0.5 - j) / Ny for j in 1:Ny]
    u_numerical = result.ux[2, :]

    # Interior points
    errors = abs.(u_numerical[2:end-1] .- u_analytical[2:end-1])
    max_rel_err = maximum(errors) / u_wall

    @test max_rel_err < 0.02  # 2% L∞ relative error
    @info "Couette 2D: L∞ relative error = $(round(max_rel_err, digits=5))"

    # uy should be essentially zero
    @test maximum(abs.(result.uy)) < 1e-6
end
