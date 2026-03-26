using Test
using Kraken

@testset "Thermal DDF" begin

    @testset "Heat conduction 1D" begin
        # Pure conduction (no flow): linear temperature profile at steady state
        # Hot wall (j=1) = 1.0, cold wall (j=Ny) = 0.0, periodic x
        Nx, Ny = 4, 32
        α = 0.1  # thermal diffusivity
        ω_T = Float64(1.0 / (3.0 * α + 0.5))

        # Initialize thermal populations to uniform T=0.5
        g_in  = zeros(Float64, Nx, Ny, 9)
        g_out = zeros(Float64, Nx, Ny, 9)
        Temp  = zeros(Float64, Nx, Ny)
        ux    = zeros(Float64, Nx, Ny)  # no flow
        uy    = zeros(Float64, Nx, Ny)

        w = Kraken.weights(D2Q9())
        for j in 1:Ny, i in 1:Nx, q in 1:9
            g_in[i, j, q] = w[q] * 0.5
        end
        copy!(g_out, g_in)
        is_solid = zeros(Bool, Nx, Ny)

        for step in 1:5000
            Kraken.stream_periodic_x_wall_y_2d!(g_out, g_in, Nx, Ny)
            Kraken.apply_fixed_temp_south_2d!(g_out, 1.0, Nx)
            Kraken.apply_fixed_temp_north_2d!(g_out, 0.0, Nx, Ny)
            Kraken.collide_thermal_2d!(g_out, ux, uy, ω_T)
            g_in, g_out = g_out, g_in
        end

        Kraken.compute_temperature_2d!(Temp, g_in)
        T_profile = Temp[2, :]

        # Analytical: linear from T_hot at wall (y=0.5) to T_cold at wall (y=Ny+0.5)
        T_analytical = [1.0 - (j - 0.5) / Ny for j in 1:Ny]

        err = maximum(abs.(T_profile .- T_analytical))
        @test err < 0.02
        @info "Heat conduction: max error = $(round(err, digits=5))"
    end

    @testset "Rayleigh-Bénard onset" begin
        # Below critical Ra (~1708): no convection, linear T profile
        # Above critical Ra: convection rolls develop
        result_sub = run_rayleigh_benard_2d(; Nx=64, Ny=16, Ra=1000.0, max_steps=10000)
        result_sup = run_rayleigh_benard_2d(; Nx=64, Ny=16, Ra=5000.0, max_steps=20000)

        # Sub-critical: velocity should remain very small
        max_uy_sub = maximum(abs.(result_sub.uy))

        # Super-critical: vertical velocity should develop (convection)
        max_uy_sup = maximum(abs.(result_sup.uy))

        @test max_uy_sup > max_uy_sub * 5  # convection much stronger above Ra_c
        @test !any(isnan, result_sub.Temp)
        @test !any(isnan, result_sup.Temp)

        @info "RB sub-critical (Ra=1000): max|uy| = $(round(max_uy_sub, sigdigits=3))"
        @info "RB super-critical (Ra=5000): max|uy| = $(round(max_uy_sup, sigdigits=3))"
    end
end
