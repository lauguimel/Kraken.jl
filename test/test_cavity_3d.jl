using Test
using Kraken
using KernelAbstractions

@testset "3D Cavity Zou-He all faces" begin

    @testset "Cavity 3D Zou-He top + bounce-back walls" begin
        N = 16
        u_lid = 0.02
        ν = 0.01  # Re = u_lid * N / ν = 32
        max_steps = 3000

        config = LBMConfig(D3Q19(); Nx=N, Ny=N, Nz=N, ν=ν, u_lid=u_lid, max_steps=max_steps)
        state = initialize_3d(config, Float64)
        f_in, f_out = state.f_in, state.f_out
        ρ = state.ρ
        ux, uy, uz = state.ux, state.uy, state.uz
        is_solid = state.is_solid
        ω = Float64(omega(config))
        Nx, Ny, Nz = N, N, N

        for step in 1:max_steps
            stream_3d!(f_out, f_in, Nx, Ny, Nz)

            # Top face: Zou-He velocity (u_lid, 0, 0)
            apply_zou_he_top_3d!(f_out, u_lid, Nx, Ny, Nz)

            # Other 5 faces: bounce-back
            apply_bounce_back_walls_3d!(f_out, Nx, Ny, Nz)

            # BGK collision
            collide_3d!(f_out, is_solid, ω)

            # Macroscopic fields
            compute_macroscopic_3d!(ρ, ux, uy, uz, f_out)

            # Swap
            f_in, f_out = f_out, f_in
        end

        ρ_cpu  = Array(ρ)
        ux_cpu = Array(ux)
        uy_cpu = Array(uy)
        uz_cpu = Array(uz)

        # No NaN
        @test !any(isnan, ρ_cpu)
        @test !any(isnan, ux_cpu)
        @test !any(isnan, uy_cpu)
        @test !any(isnan, uz_cpu)

        # Density variation < 5% (coarse grid, not fully converged)
        ρ_mean = sum(ρ_cpu) / length(ρ_cpu)
        ρ_max_dev = maximum(abs.(ρ_cpu .- ρ_mean)) / ρ_mean
        @test ρ_max_dev < 0.05

        # Mass conservation: total mass close to initial (coarse grid tolerance)
        @test abs(ρ_mean - 1.0) < 0.05

        # ux at lid-adjacent cells: check mean over interior of top face
        ux_top = ux_cpu[2:Nx-1, 2:Ny-1, Nz]
        mean_ux_top = sum(ux_top) / length(ux_top)
        @test abs(mean_ux_top - u_lid) / u_lid < 0.6  # coarse grid tolerance

        @info "Cavity 3D Zou-He: ρ_max_dev=$(round(ρ_max_dev, digits=5)), mean_ux_top=$(round(mean_ux_top, digits=5))"
    end

    @testset "Zou-He velocity all 6 faces — mass conservation" begin
        # Small box: impose zero velocity on all 6 faces via Zou-He
        # Should maintain equilibrium (no flow, ρ≈1 everywhere)
        N = 8
        ν = 0.1
        max_steps = 500

        config = LBMConfig(D3Q19(); Nx=N, Ny=N, Nz=N, ν=ν, u_lid=0.0, max_steps=max_steps)
        state = initialize_3d(config, Float64)
        f_in, f_out = state.f_in, state.f_out
        ρ = state.ρ
        ux, uy, uz = state.ux, state.uy, state.uz
        is_solid = state.is_solid
        ω = Float64(omega(config))

        for step in 1:max_steps
            stream_3d!(f_out, f_in, N, N, N)

            # All 6 faces: Zou-He velocity with zero velocity
            apply_zou_he_top_3d!(f_out, 0.0, N, N, N)
            apply_zou_he_bottom_3d!(f_out, 0.0, 0.0, N, N)
            apply_zou_he_west_3d!(f_out, 0.0, 0.0, 0.0, N, N)
            apply_zou_he_east_3d!(f_out, 0.0, 0.0, 0.0, N, N, N)
            apply_zou_he_south_3d!(f_out, 0.0, 0.0, 0.0, N, N)
            apply_zou_he_north_3d!(f_out, 0.0, 0.0, 0.0, N, N, N)

            collide_3d!(f_out, is_solid, ω)
            compute_macroscopic_3d!(ρ, ux, uy, uz, f_out)
            f_in, f_out = f_out, f_in
        end

        ρ_cpu = Array(ρ)
        ux_cpu = Array(ux)
        uy_cpu = Array(uy)
        uz_cpu = Array(uz)

        # Should stay at equilibrium: ρ≈1, u≈0
        @test !any(isnan, ρ_cpu)
        @test abs(sum(ρ_cpu) / length(ρ_cpu) - 1.0) < 0.001
        @test maximum(abs.(ux_cpu)) < 1e-10
        @test maximum(abs.(uy_cpu)) < 1e-10
        @test maximum(abs.(uz_cpu)) < 1e-10

        @info "Zou-He 6 faces zero vel: ρ_mean=$(round(sum(ρ_cpu)/length(ρ_cpu), digits=8)), max|u|=$(round(maximum(abs.(ux_cpu)), sigdigits=3))"
    end

    @testset "Zou-He pressure outlet 3D — no NaN" begin
        # Quick test: west inlet + east pressure outlet
        N = 8
        ν = 0.1
        u_in = 0.01
        max_steps = 200

        config = LBMConfig(D3Q19(); Nx=N, Ny=N, Nz=N, ν=ν, u_lid=0.0, max_steps=max_steps)
        state = initialize_3d(config, Float64)
        f_in, f_out = state.f_in, state.f_out
        ρ = state.ρ
        ux, uy, uz = state.ux, state.uy, state.uz
        is_solid = state.is_solid
        ω = Float64(omega(config))

        for step in 1:max_steps
            stream_3d!(f_out, f_in, N, N, N)

            # West: velocity inlet
            apply_zou_he_west_3d!(f_out, u_in, 0.0, 0.0, N, N)
            # East: pressure outlet
            apply_zou_he_pressure_east_3d!(f_out, N, N, N; ρ_out=1.0)
            # Top/bottom: Zou-He zero velocity
            apply_zou_he_top_3d!(f_out, 0.0, N, N, N)
            apply_zou_he_bottom_3d!(f_out, 0.0, 0.0, N, N)
            # South/north: Zou-He zero velocity
            apply_zou_he_south_3d!(f_out, 0.0, 0.0, 0.0, N, N)
            apply_zou_he_north_3d!(f_out, 0.0, 0.0, 0.0, N, N, N)

            collide_3d!(f_out, is_solid, ω)
            compute_macroscopic_3d!(ρ, ux, uy, uz, f_out)
            f_in, f_out = f_out, f_in
        end

        ρ_cpu = Array(ρ)
        @test !any(isnan, ρ_cpu)
        @test !any(isinf, ρ_cpu)

        @info "Pressure outlet 3D: ρ range=[$(round(minimum(ρ_cpu), digits=5)), $(round(maximum(ρ_cpu), digits=5))]"
    end
end
