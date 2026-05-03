using Test
using Kraken

function _stream_cell_id(topology, level, i, j)
    for (id, cell) in pairs(topology.cells)
        if cell.level == level && cell.i == i && cell.j == j
            return id
        end
    end
    error("cell not found: level=$level i=$i j=$j")
end

function _stream_zero_boundary_packets!(coarse_F, patch, topology)
    for idx in topology.boundary_links
        link = topology.links[idx]
        cell = topology.cells[link.src]
        if cell.level == 0
            coarse_F[cell.i, cell.j, link.q] = 0
        else
            i0 = 2 * first(patch.parent_i_range) - 1
            j0 = 2 * first(patch.parent_j_range) - 1
            patch.fine_F[cell.i - i0 + 1, cell.j - j0 + 1, link.q] = 0
        end
    end
    return coarse_F, patch
end

function _stream_zero_y_boundary_packets!(coarse_F, patch, topology)
    ny = size(coarse_F, 2)
    for idx in topology.boundary_links
        link = topology.links[idx]
        cell = topology.cells[link.src]
        cy = d2q9_cy(link.q)
        if cell.level == 0
            j_raw = cell.j + cy
            1 <= j_raw <= ny && continue
            coarse_F[cell.i, cell.j, link.q] = 0
        else
            j_raw = cell.j + cy
            1 <= j_raw <= 2 * ny && continue
            i0 = 2 * first(patch.parent_i_range) - 1
            j0 = 2 * first(patch.parent_j_range) - 1
            patch.fine_F[cell.i - i0 + 1, cell.j - j0 + 1, link.q] = 0
        end
    end
    return coarse_F, patch
end

_stream_moving_wall_delta(volume, rho_wall, wall_u, q) =
    volume * rho_wall * wall_u * d2q9_cx(q) / 6

@testset "Conservative tree route streaming 2D" begin
    Nx, Ny = 9, 10
    patch_in = create_conservative_tree_patch_2d(3:5, 4:6)
    patch_out = create_conservative_tree_patch_2d(3:5, 4:6)
    topology = create_conservative_tree_topology_2d(Nx, Ny, patch_in)

    @testset "direct same-level route moves one packet" begin
        coarse_in = zeros(Float64, Nx, Ny, 9)
        coarse_out = similar(coarse_in)
        coarse_in[1, 5, 2] = 3.25

        stream_composite_routes_interior_F_2d!(
            coarse_out, patch_out, coarse_in, patch_in, topology)

        @test coarse_out[2, 5, 2] == 3.25
        @test isapprox(active_mass_F(coarse_out, patch_out), 3.25; atol=1e-14, rtol=0)
    end

    @testset "coarse to fine face route splits one packet" begin
        coarse_in = zeros(Float64, Nx, Ny, 9)
        coarse_out = similar(coarse_in)
        patch_in = create_conservative_tree_patch_2d(3:5, 4:6)
        patch_out = create_conservative_tree_patch_2d(3:5, 4:6)
        coarse_in[2, 5, 2] = 4.0

        stream_composite_routes_interior_F_2d!(
            coarse_out, patch_out, coarse_in, patch_in, topology)

        @test patch_out.fine_F[1, 3, 2] == 2.0
        @test patch_out.fine_F[1, 4, 2] == 2.0
        @test isapprox(active_mass_F(coarse_out, patch_out), 4.0; atol=1e-14, rtol=0)
    end

    @testset "coarse to fine corner route sends one packet" begin
        coarse_in = zeros(Float64, Nx, Ny, 9)
        coarse_out = similar(coarse_in)
        patch_in = create_conservative_tree_patch_2d(3:5, 4:6)
        patch_out = create_conservative_tree_patch_2d(3:5, 4:6)
        coarse_in[2, 3, 6] = 5.5

        stream_composite_routes_interior_F_2d!(
            coarse_out, patch_out, coarse_in, patch_in, topology)

        @test patch_out.fine_F[1, 1, 6] == 5.5
        @test isapprox(active_mass_F(coarse_out, patch_out), 5.5; atol=1e-14, rtol=0)
    end

    @testset "fine to coarse face routes coalesce by accumulation" begin
        coarse_in = zeros(Float64, Nx, Ny, 9)
        coarse_out = similar(coarse_in)
        patch_in = create_conservative_tree_patch_2d(3:5, 4:6)
        patch_out = create_conservative_tree_patch_2d(3:5, 4:6)
        patch_in.fine_F[1, 3, 4] = 1.25
        patch_in.fine_F[1, 4, 4] = 2.75

        stream_composite_routes_interior_F_2d!(
            coarse_out, patch_out, coarse_in, patch_in, topology)

        @test coarse_out[2, 5, 4] == 4.0
        @test isapprox(active_mass_F(coarse_out, patch_out), 4.0; atol=1e-14, rtol=0)
    end

    @testset "fine to coarse corner route sends one packet" begin
        coarse_in = zeros(Float64, Nx, Ny, 9)
        coarse_out = similar(coarse_in)
        patch_in = create_conservative_tree_patch_2d(3:5, 4:6)
        patch_out = create_conservative_tree_patch_2d(3:5, 4:6)
        patch_in.fine_F[1, 1, 8] = 7.0

        stream_composite_routes_interior_F_2d!(
            coarse_out, patch_out, coarse_in, patch_in, topology)

        @test coarse_out[2, 3, 8] == 7.0
        @test isapprox(active_mass_F(coarse_out, patch_out), 7.0; atol=1e-14, rtol=0)
    end

    @testset "all non-boundary routes conserve active population sums" begin
        coarse_in = zeros(Float64, Nx, Ny, 9)
        coarse_out = similar(coarse_in)
        patch_in = create_conservative_tree_patch_2d(3:5, 4:6)
        patch_out = create_conservative_tree_patch_2d(3:5, 4:6)

        for q in 1:9, j in axes(coarse_in, 2), i in axes(coarse_in, 1)
            if !(i in patch_in.parent_i_range && j in patch_in.parent_j_range)
                coarse_in[i, j, q] = q + i / 32 + j / 64 + i * j / 4096
            end
        end
        for q in 1:9, j in axes(patch_in.fine_F, 2), i in axes(patch_in.fine_F, 1)
            patch_in.fine_F[i, j, q] = q / 2 + i / 128 + j / 256 + i * j / 8192
        end
        _stream_zero_boundary_packets!(coarse_in, patch_in, topology)
        pop0 = active_population_sums_F(coarse_in, patch_in)

        stream_composite_routes_interior_F_2d!(
            coarse_out, patch_out, coarse_in, patch_in, topology)

        @test isapprox(active_population_sums_F(coarse_out, patch_out), pop0;
                       atol=1e-12, rtol=0)
        @test isapprox(active_mass_F(coarse_out, patch_out), active_mass_F(coarse_in, patch_in);
                       atol=1e-12, rtol=0)
    end

    @testset "layout checks catch mismatched patch" begin
        coarse_in = zeros(Float64, Nx, Ny, 9)
        coarse_out = similar(coarse_in)
        wrong_patch = create_conservative_tree_patch_2d(4:6, 4:6)
        @test_throws ArgumentError stream_composite_routes_interior_F_2d!(
            coarse_out, wrong_patch, coarse_in, patch_in, topology)
    end

    @testset "periodic x wraps coarse boundary packets" begin
        coarse_in = zeros(Float64, Nx, Ny, 9)
        coarse_out = similar(coarse_in)
        patch_in = create_conservative_tree_patch_2d(3:5, 4:6)
        patch_out = create_conservative_tree_patch_2d(3:5, 4:6)
        coarse_in[1, 5, 4] = 2.5
        coarse_in[Nx, 5, 2] = 3.5

        stream_composite_routes_periodic_x_F_2d!(
            coarse_out, patch_out, coarse_in, patch_in, topology)

        @test coarse_out[Nx, 5, 4] == 2.5
        @test coarse_out[1, 5, 2] == 3.5
        @test isapprox(active_mass_F(coarse_out, patch_out), 6.0; atol=1e-14, rtol=0)
    end

    @testset "periodic x wraps coarse boundary packets into fine patch" begin
        nx, ny = 6, 6
        patch_in = create_conservative_tree_patch_2d(5:6, 3:4)
        patch_out = create_conservative_tree_patch_2d(5:6, 3:4)
        topology = create_conservative_tree_topology_2d(nx, ny, patch_in)
        coarse_in = zeros(Float64, nx, ny, 9)
        coarse_out = similar(coarse_in)
        coarse_in[1, 3, 4] = 8.0

        stream_composite_routes_periodic_x_F_2d!(
            coarse_out, patch_out, coarse_in, patch_in, topology)

        @test patch_out.fine_F[4, 1, 4] == 4.0
        @test patch_out.fine_F[4, 2, 4] == 4.0
        @test isapprox(active_mass_F(coarse_out, patch_out), 8.0; atol=1e-14, rtol=0)
    end

    @testset "periodic x wraps fine boundary packets to coarse cells" begin
        nx, ny = 6, 6
        patch_in = create_conservative_tree_patch_2d(1:2, 3:4)
        patch_out = create_conservative_tree_patch_2d(1:2, 3:4)
        topology = create_conservative_tree_topology_2d(nx, ny, patch_in)
        coarse_in = zeros(Float64, nx, ny, 9)
        coarse_out = similar(coarse_in)
        patch_in.fine_F[1, 1, 4] = 6.0

        stream_composite_routes_periodic_x_F_2d!(
            coarse_out, patch_out, coarse_in, patch_in, topology)

        @test coarse_out[nx, 3, 4] == 6.0
        @test isapprox(active_mass_F(coarse_out, patch_out), 6.0; atol=1e-14, rtol=0)
    end

    @testset "periodic x keeps all non-y-boundary population sums" begin
        coarse_in = zeros(Float64, Nx, Ny, 9)
        coarse_out = similar(coarse_in)
        patch_in = create_conservative_tree_patch_2d(3:5, 4:6)
        patch_out = create_conservative_tree_patch_2d(3:5, 4:6)
        topology = create_conservative_tree_topology_2d(Nx, Ny, patch_in)

        for q in 1:9, j in axes(coarse_in, 2), i in axes(coarse_in, 1)
            if !(i in patch_in.parent_i_range && j in patch_in.parent_j_range)
                coarse_in[i, j, q] = 0.25q + i / 41 + j / 67 + i * j / 8192
            end
        end
        for q in 1:9, j in axes(patch_in.fine_F, 2), i in axes(patch_in.fine_F, 1)
            patch_in.fine_F[i, j, q] = 0.125q + i / 131 + j / 257 + i * j / 16384
        end
        _stream_zero_y_boundary_packets!(coarse_in, patch_in, topology)
        pop0 = active_population_sums_F(coarse_in, patch_in)

        stream_composite_routes_periodic_x_F_2d!(
            coarse_out, patch_out, coarse_in, patch_in, topology)

        @test isapprox(active_population_sums_F(coarse_out, patch_out), pop0;
                       atol=1e-12, rtol=0)
        @test isapprox(active_mass_F(coarse_out, patch_out), active_mass_F(coarse_in, patch_in);
                       atol=1e-12, rtol=0)
    end

    @testset "periodic x wall y bounces coarse boundary packets" begin
        coarse_in = zeros(Float64, Nx, Ny, 9)
        coarse_out = similar(coarse_in)
        patch_in = create_conservative_tree_patch_2d(3:5, 4:6)
        patch_out = create_conservative_tree_patch_2d(3:5, 4:6)
        topology = create_conservative_tree_topology_2d(Nx, Ny, patch_in)
        coarse_in[4, 1, 5] = 2.0
        coarse_in[6, Ny, 3] = 3.0
        coarse_in[1, 1, 8] = 4.0

        stream_composite_routes_periodic_x_wall_y_F_2d!(
            coarse_out, patch_out, coarse_in, patch_in, topology)

        @test coarse_out[4, 1, d2q9_opposite(5)] == 2.0
        @test coarse_out[6, Ny, d2q9_opposite(3)] == 3.0
        @test coarse_out[1, 1, d2q9_opposite(8)] == 4.0
        @test isapprox(active_mass_F(coarse_out, patch_out), 9.0; atol=1e-14, rtol=0)
    end

    @testset "periodic x wall y bounces fine boundary packets" begin
        nx, ny = 6, 6
        patch_in = create_conservative_tree_patch_2d(3:4, 5:6)
        patch_out = create_conservative_tree_patch_2d(3:4, 5:6)
        topology = create_conservative_tree_topology_2d(nx, ny, patch_in)
        coarse_in = zeros(Float64, nx, ny, 9)
        coarse_out = similar(coarse_in)
        patch_in.fine_F[2, 4, 3] = 7.0

        stream_composite_routes_periodic_x_wall_y_F_2d!(
            coarse_out, patch_out, coarse_in, patch_in, topology)

        @test patch_out.fine_F[2, 4, d2q9_opposite(3)] == 7.0
        @test isapprox(active_mass_F(coarse_out, patch_out), 7.0; atol=1e-14, rtol=0)
    end

    @testset "periodic x wall y conserves total active mass" begin
        coarse_in = zeros(Float64, Nx, Ny, 9)
        coarse_out = similar(coarse_in)
        patch_in = create_conservative_tree_patch_2d(3:5, 4:6)
        patch_out = create_conservative_tree_patch_2d(3:5, 4:6)
        topology = create_conservative_tree_topology_2d(Nx, Ny, patch_in)

        for q in 1:9, j in axes(coarse_in, 2), i in axes(coarse_in, 1)
            if !(i in patch_in.parent_i_range && j in patch_in.parent_j_range)
                coarse_in[i, j, q] = 0.375q + i / 43 + j / 71 + i * j / 8192
            end
        end
        for q in 1:9, j in axes(patch_in.fine_F, 2), i in axes(patch_in.fine_F, 1)
            patch_in.fine_F[i, j, q] = 0.1875q + i / 137 + j / 263 + i * j / 16384
        end
        mass0 = active_mass_F(coarse_in, patch_in)

        stream_composite_routes_periodic_x_wall_y_F_2d!(
            coarse_out, patch_out, coarse_in, patch_in, topology)

        @test isapprox(active_mass_F(coarse_out, patch_out), mass0; atol=1e-11, rtol=0)
    end

    @testset "periodic x moving wall y corrects coarse diagonals" begin
        coarse_in = zeros(Float64, Nx, Ny, 9)
        coarse_out = similar(coarse_in)
        patch_in = create_conservative_tree_patch_2d(3:5, 4:6)
        patch_out = create_conservative_tree_patch_2d(3:5, 4:6)
        topology = create_conservative_tree_topology_2d(Nx, Ny, patch_in)
        u_south = 0.06
        u_north = 0.03
        coarse_in[4, 1, 8] = 2.0
        coarse_in[6, Ny, 6] = 3.0

        stream_composite_routes_periodic_x_moving_wall_y_F_2d!(
            coarse_out, patch_out, coarse_in, patch_in, topology;
            u_south=u_south, u_north=u_north)

        @test coarse_out[4, 1, 6] ==
              2.0 + _stream_moving_wall_delta(1.0, 1.0, u_south, 6)
        @test coarse_out[6, Ny, 8] ==
              3.0 + _stream_moving_wall_delta(1.0, 1.0, u_north, 8)
    end

    @testset "periodic x moving wall y corrects fine diagonals" begin
        nx, ny = 6, 6
        patch_in = create_conservative_tree_patch_2d(3:4, 5:6)
        patch_out = create_conservative_tree_patch_2d(3:4, 5:6)
        topology = create_conservative_tree_topology_2d(nx, ny, patch_in)
        coarse_in = zeros(Float64, nx, ny, 9)
        coarse_out = similar(coarse_in)
        u_north = 0.08
        patch_in.fine_F[2, 4, 6] = 5.0

        stream_composite_routes_periodic_x_moving_wall_y_F_2d!(
            coarse_out, patch_out, coarse_in, patch_in, topology;
            u_north=u_north, volume_fine=0.25)

        @test patch_out.fine_F[2, 4, 8] ==
              5.0 + _stream_moving_wall_delta(0.25, 1.0, u_north, 8)
    end

    @testset "periodic x moving wall y injects tangential momentum at rest" begin
        coarse_in = zeros(Float64, Nx, Ny, 9)
        coarse_out = similar(coarse_in)
        patch_in = create_conservative_tree_patch_2d(3:5, 4:6)
        patch_out = create_conservative_tree_patch_2d(3:5, 4:6)
        topology = create_conservative_tree_topology_2d(Nx, Ny, patch_in)
        fill_equilibrium_integrated_D2Q9!(coarse_in, 1.0, 1.0, 0.0, 0.0)
        fill_equilibrium_integrated_D2Q9!(patch_in.fine_F, 0.25, 1.0, 0.0, 0.0)
        mass0 = active_mass_F(coarse_in, patch_in)
        mx0 = active_momentum_F(coarse_in, patch_in)[1]

        stream_composite_routes_periodic_x_moving_wall_y_F_2d!(
            coarse_out, patch_out, coarse_in, patch_in, topology;
            u_north=0.04, volume_coarse=1.0, volume_fine=0.25)

        @test isapprox(active_mass_F(coarse_out, patch_out), mass0; atol=1e-12, rtol=0)
        @test active_momentum_F(coarse_out, patch_out)[1] > mx0
    end
end
