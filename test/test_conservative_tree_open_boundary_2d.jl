using Test
using Kraken

function _cell_rho_ux_2d(Fcell, volume)
    rho = mass_F(Fcell) / volume
    ux = (momentum_F(Fcell)[1] / volume) / rho
    return rho, ux
end

@testset "Conservative tree open boundary 2D" begin
    @testset "composite Zou-He west covers fine inlet and coarse gaps" begin
        nx, ny = 7, 6
        volume_coarse = 1.0
        volume_fine = 0.25
        u_in = 0.035
        patch = create_conservative_tree_patch_2d(1:2, 2:5)
        coarse = zeros(Float64, nx, ny, 9)
        fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, 1.0, 0.0, 0.0)
        fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, 1.0, 0.0, 0.0)

        apply_composite_zou_he_west_F_2d!(
            coarse, patch, u_in, volume_coarse, volume_fine)

        for J in (1, 6)
            rho, ux = _cell_rho_ux_2d(@view(coarse[1, J, :]), volume_coarse)
            @test isfinite(rho)
            @test isapprox(ux, u_in; atol=1e-14, rtol=0)
        end
        for jf in axes(patch.fine_F, 2)
            rho, ux = _cell_rho_ux_2d(@view(patch.fine_F[1, jf, :]), volume_fine)
            @test isfinite(rho)
            @test isapprox(ux, u_in; atol=1e-14, rtol=0)
        end
        for J in patch.parent_j_range
            il, jl = 1, J - first(patch.parent_j_range) + 1
            shadow = @view patch.coarse_shadow_F[il, jl, :]
            children = @view patch.fine_F[1:2, (2 * jl - 1):(2 * jl), :]
            expected = zeros(Float64, 9)
            coalesce_F_2d!(expected, children)
            @test isapprox(shadow, expected; atol=1e-14, rtol=0)
        end
    end

    @testset "composite Zou-He east covers fine outlet and coarse gaps" begin
        nx, ny = 7, 6
        volume_coarse = 1.0
        volume_fine = 0.25
        rho_out = 1.07
        patch = create_conservative_tree_patch_2d(6:7, 2:5)
        coarse = zeros(Float64, nx, ny, 9)
        fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, 1.0, 0.02, 0.0)
        fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, 1.0, 0.02, 0.0)

        apply_composite_zou_he_pressure_east_F_2d!(
            coarse, patch, volume_coarse, volume_fine; rho_out=rho_out)

        for J in (1, 6)
            rho, ux = _cell_rho_ux_2d(@view(coarse[nx, J, :]), volume_coarse)
            @test isfinite(ux)
            @test isapprox(rho, rho_out; atol=1e-14, rtol=0)
        end
        for jf in axes(patch.fine_F, 2)
            rho, ux = _cell_rho_ux_2d(@view(patch.fine_F[end, jf, :]), volume_fine)
            @test isfinite(ux)
            @test isapprox(rho, rho_out; atol=1e-14, rtol=0)
        end
    end

    @testset "open route step imposes fine inlet and outlet moments" begin
        nx, ny = 6, 5
        volume_coarse = 1.0
        volume_fine = 0.25
        u_in = 0.025
        rho_out = 1.03
        patch = create_conservative_tree_patch_2d(1:nx, 2:4)
        patch_next = create_conservative_tree_patch_2d(1:nx, 2:4)
        topology = create_conservative_tree_topology_2d(nx, ny, patch)
        coarse = zeros(Float64, nx, ny, 9)
        coarse_next = similar(coarse)
        fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, 1.0, u_in, 0.0)
        fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, 1.0, u_in, 0.0)

        stream_composite_routes_zou_he_x_wall_y_F_2d!(
            coarse_next, patch_next, coarse, patch, topology;
            u_in=u_in, rho_out=rho_out,
            volume_coarse=volume_coarse, volume_fine=volume_fine)

        for J in (1, ny)
            rho_w, ux_w = _cell_rho_ux_2d(@view(coarse_next[1, J, :]), volume_coarse)
            rho_e, ux_e = _cell_rho_ux_2d(@view(coarse_next[nx, J, :]), volume_coarse)
            @test isfinite(rho_w)
            @test isfinite(ux_e)
            @test isapprox(ux_w, u_in; atol=1e-14, rtol=0)
            @test isapprox(rho_e, rho_out; atol=1e-14, rtol=0)
        end
        for jf in axes(patch_next.fine_F, 2)
            rho_w, ux_w = _cell_rho_ux_2d(@view(patch_next.fine_F[1, jf, :]), volume_fine)
            rho_e, ux_e = _cell_rho_ux_2d(@view(patch_next.fine_F[end, jf, :]), volume_fine)
            @test isfinite(rho_w)
            @test isfinite(ux_e)
            @test isapprox(ux_w, u_in; atol=1e-14, rtol=0)
            @test isapprox(rho_e, rho_out; atol=1e-14, rtol=0)
        end
        @test active_mass_F(coarse_next, patch_next) > 0
    end

    @testset "short open channel with inlet-spanning patch remains finite" begin
        nx, ny = 8, 6
        volume_coarse = 1.0
        volume_fine = 0.25
        u_in = 0.015
        rho_out = 1.0
        omega = 1.0
        patch = create_conservative_tree_patch_2d(1:4, 2:5)
        patch_next = create_conservative_tree_patch_2d(1:4, 2:5)
        topology = create_conservative_tree_topology_2d(nx, ny, patch)
        coarse = zeros(Float64, nx, ny, 9)
        coarse_next = similar(coarse)
        fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, 1.0, u_in, 0.0)
        fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, 1.0, u_in, 0.0)
        apply_composite_zou_he_west_F_2d!(
            coarse, patch, u_in, volume_coarse, volume_fine)
        apply_composite_zou_he_pressure_east_F_2d!(
            coarse, patch, volume_coarse, volume_fine; rho_out=rho_out)
        mass_initial = active_mass_F(coarse, patch)

        for _ in 1:10
            collide_BGK_composite_F_2d!(
                coarse, patch, volume_coarse, volume_fine, omega, omega)
            stream_composite_routes_zou_he_x_wall_y_F_2d!(
                coarse_next, patch_next, coarse, patch, topology;
                u_in=u_in, rho_out=rho_out,
                volume_coarse=volume_coarse, volume_fine=volume_fine)
            coarse, coarse_next = coarse_next, coarse
            patch, patch_next = patch_next, patch
        end

        mass_final = active_mass_F(coarse, patch)
        @test isfinite(mass_final)
        @test abs(mass_final - mass_initial) / mass_initial < 0.08
        for jf in axes(patch.fine_F, 2)
            _, ux = _cell_rho_ux_2d(@view(patch.fine_F[1, jf, :]), volume_fine)
            @test isapprox(ux, u_in; atol=1e-14, rtol=0)
        end
    end
end
