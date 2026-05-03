using Test
using Kraken

function _stream_zero_boundary_packets_3d!(coarse_F, patch, topology)
    for idx in topology.boundary_links
        link = topology.links[idx]
        cell = topology.cells[link.src]
        if cell.level == 0
            coarse_F[cell.i, cell.j, cell.k, link.q] = 0
        else
            i0 = 2 * first(patch.parent_i_range) - 1
            j0 = 2 * first(patch.parent_j_range) - 1
            k0 = 2 * first(patch.parent_k_range) - 1
            patch.fine_F[cell.i - i0 + 1,
                         cell.j - j0 + 1,
                         cell.k - k0 + 1,
                         link.q] = 0
        end
    end
    return coarse_F, patch
end

@testset "Conservative tree route streaming 3D" begin
    Nx, Ny, Nz = 7, 8, 6
    patch_template = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
    topology = create_conservative_tree_topology_3d(Nx, Ny, Nz, patch_template)

    @testset "direct same-level route moves one packet" begin
        patch_in = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        patch_out = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        coarse_in = zeros(Float64, Nx, Ny, Nz, 19)
        coarse_out = similar(coarse_in)
        coarse_in[1, 4, 2, 2] = 3.25

        stream_composite_routes_interior_F_3d!(
            coarse_out, patch_out, coarse_in, patch_in, topology)

        @test coarse_out[2, 4, 2, 2] == 3.25
        @test isapprox(active_mass_F_3d(coarse_out, patch_out), 3.25; atol=1e-14, rtol=0)
    end

    @testset "coarse to fine face route splits one packet" begin
        patch_in = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        patch_out = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        coarse_in = zeros(Float64, Nx, Ny, Nz, 19)
        coarse_out = similar(coarse_in)
        coarse_in[2, 4, 2, 2] = 4.0

        stream_composite_routes_interior_F_3d!(
            coarse_out, patch_out, coarse_in, patch_in, topology)

        @test patch_out.fine_F[1, 1, 1, 2] == 1.0
        @test patch_out.fine_F[1, 2, 1, 2] == 1.0
        @test patch_out.fine_F[1, 1, 2, 2] == 1.0
        @test patch_out.fine_F[1, 2, 2, 2] == 1.0
        @test patch_out.coarse_shadow_F[1, 1, 1, 2] == 4.0
        @test isapprox(active_mass_F_3d(coarse_out, patch_out), 4.0; atol=1e-14, rtol=0)
    end

    @testset "coarse to fine edge route splits one packet" begin
        patch_in = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        patch_out = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        coarse_in = zeros(Float64, Nx, Ny, Nz, 19)
        coarse_out = similar(coarse_in)
        coarse_in[2, 3, 2, 8] = 5.0

        stream_composite_routes_interior_F_3d!(
            coarse_out, patch_out, coarse_in, patch_in, topology)

        @test patch_out.fine_F[1, 1, 1, 8] == 2.5
        @test patch_out.fine_F[1, 1, 2, 8] == 2.5
        @test patch_out.coarse_shadow_F[1, 1, 1, 8] == 5.0
        @test isapprox(active_mass_F_3d(coarse_out, patch_out), 5.0; atol=1e-14, rtol=0)
    end

    @testset "fine to coarse face routes coalesce by accumulation" begin
        patch_in = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        patch_out = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        coarse_in = zeros(Float64, Nx, Ny, Nz, 19)
        coarse_out = similar(coarse_in)
        patch_in.fine_F[1, 1, 1, 3] = 1.25
        patch_in.fine_F[1, 2, 1, 3] = 2.0
        patch_in.fine_F[1, 1, 2, 3] = 0.5
        patch_in.fine_F[1, 2, 2, 3] = 0.25

        stream_composite_routes_interior_F_3d!(
            coarse_out, patch_out, coarse_in, patch_in, topology)

        @test coarse_out[2, 4, 2, 3] == 4.0
        @test isapprox(active_mass_F_3d(coarse_out, patch_out), 4.0; atol=1e-14, rtol=0)
    end

    @testset "fine to coarse edge routes coalesce by accumulation" begin
        patch_in = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        patch_out = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        coarse_in = zeros(Float64, Nx, Ny, Nz, 19)
        coarse_out = similar(coarse_in)
        patch_in.fine_F[1, 1, 1, 11] = 3.0
        patch_in.fine_F[1, 1, 2, 11] = 4.0

        stream_composite_routes_interior_F_3d!(
            coarse_out, patch_out, coarse_in, patch_in, topology)

        @test coarse_out[2, 3, 2, 11] == 7.0
        @test isapprox(active_mass_F_3d(coarse_out, patch_out), 7.0; atol=1e-14, rtol=0)
    end

    @testset "all non-boundary routes conserve active population sums" begin
        patch_in = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        patch_out = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        coarse_in = zeros(Float64, Nx, Ny, Nz, 19)
        coarse_out = similar(coarse_in)

        for q in 1:19, k in axes(coarse_in, 3), j in axes(coarse_in, 2), i in axes(coarse_in, 1)
            if !(i in patch_in.parent_i_range &&
                 j in patch_in.parent_j_range &&
                 k in patch_in.parent_k_range)
                coarse_in[i, j, k, q] = q + i / 32 + j / 64 + k / 128
            end
        end
        for q in 1:19, k in axes(patch_in.fine_F, 3), j in axes(patch_in.fine_F, 2), i in axes(patch_in.fine_F, 1)
            patch_in.fine_F[i, j, k, q] = q / 2 + i / 128 + j / 256 + k / 512
        end
        _stream_zero_boundary_packets_3d!(coarse_in, patch_in, topology)
        pop0 = active_population_sums_F_3d(coarse_in, patch_in)
        moments0 = collect(active_moments_F_3d(coarse_in, patch_in))

        stream_composite_routes_interior_F_3d!(
            coarse_out, patch_out, coarse_in, patch_in, topology)

        @test isapprox(active_population_sums_F_3d(coarse_out, patch_out), pop0;
                       atol=1e-11, rtol=0)
        @test isapprox(collect(active_moments_F_3d(coarse_out, patch_out)), moments0;
                       atol=1e-11, rtol=0)
    end

    @testset "layout checks catch mismatched patch" begin
        patch_in = create_conservative_tree_patch_3d(3:4, 4:5, 2:3)
        wrong_patch = create_conservative_tree_patch_3d(4:5, 4:5, 2:3)
        coarse_in = zeros(Float64, Nx, Ny, Nz, 19)
        coarse_out = similar(coarse_in)
        @test_throws ArgumentError stream_composite_routes_interior_F_3d!(
            coarse_out, wrong_patch, coarse_in, patch_in, topology)
    end
end
