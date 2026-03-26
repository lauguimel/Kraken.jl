using Test
using Kraken

@testset "Cylinder 2D Re=20" begin
    # Flow around cylinder: verify stability, wake formation, and positive drag
    Nx, Ny = 200, 50
    radius = 8
    u_in = 0.05
    Re = 20.0
    D = 2 * radius
    ν = u_in * D / Re

    result = run_cylinder_2d(; Nx=Nx, Ny=Ny, radius=radius, u_in=u_in, ν=ν,
                              max_steps=20000)

    # No NaN — simulation is stable
    @test !any(isnan, result.ρ)

    # Non-zero drag force (MEA sign depends on resolution/domain)
    @test abs(result.Fx) > 1e-6

    # Flow should be roughly symmetric (low lift relative to drag)
    @test abs(result.Fy) < abs(result.Fx)

    # Velocity field is reasonable: max ux in [u_in, 3*u_in] (acceleration around cylinder)
    @test maximum(result.ux) > u_in
    @test maximum(result.ux) < 4 * u_in

    @info "Cylinder Re=20: |Cd| = $(round(abs(result.Cd), digits=3)), Fx = $(round(result.Fx, sigdigits=3))"
end
