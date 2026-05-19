using Test
using KernelAbstractions
using Kraken

const FVFD_ATOL = 1e-12

function _fvfd_optional_metal_backend()
    try
        @eval using Metal
        return Metal.functional() ? Metal.MetalBackend() : nothing
    catch
        return nothing
    end
end

@testset "FVFD operator library 2D" begin
    @testset "domain BC aliases match log-FV compatibility codes" begin
        bc = Kraken.FVFDDomainBC2D(; west=:periodic, east=:open, south=:wall, north=:symmetry)
        @test bc.west == Kraken.LOGFV_BC_PERIODIC
        @test bc.east == Kraken.LOGFV_BC_OPEN
        @test bc.south == Kraken.LOGFV_BC_WALL
        @test bc.north == Kraken.LOGFV_BC_WALL

        log_bc = Kraken.LogFVDomainBC2D(; west=:periodic, east=:open, south=:wall, north=:wall)
        @test log_bc isa Kraken.FVFDDomainBC2D
        @test log_bc == Kraken.FVFDDomainBC2D(; west=:periodic, east=:open, south=:wall, north=:wall)

        field_bc = Kraken.FVFDFieldBC2D(;
            west=[1.0], east=[2.0], south=[3.0], north=[4.0],
        )
        @test field_bc.west == [1.0]
        @test field_bc.east == [2.0]
        @test field_bc.south == [3.0]
        @test field_bc.north == [4.0]
        @test Kraken.LogFVFieldBC2D === Kraken.FVFDFieldBC2D
    end

    @testset "field BC lowering transfers host values to backend vectors" begin
        Nx, Ny = 4, 3
        bc = Kraken.FVFDDomainBC2D(;
            west=:open, east=:periodic, south=:wall, north=:open,
        )
        field_bc = Kraken.FVFDFieldBC2D(;
            west=[1.0, 1.5, 2.0],
            east=[9.0],
            south=3.25,
            north=[4.0, 4.5, 5.0, 5.5],
        )

        lowered = Kraken.fvfd_transfer_field_bc_2d(
            field_bc, KernelAbstractions.CPU(), Float32, Nx, Ny, bc; name=:phi_bc,
        )
        @test eltype(lowered.west) == Float32
        @test lowered.west ≈ Float32[1.0, 1.5, 2.0] atol=0 rtol=0
        @test lowered.east == zeros(Float32, Ny)
        @test lowered.south == fill(Float32(3.25), Nx)
        @test lowered.north ≈ Float32[4.0, 4.5, 5.0, 5.5] atol=0 rtol=0

        lowered_logfv = Kraken.logfv_transfer_field_bc_2d(
            field_bc, KernelAbstractions.CPU(), Float32, Nx, Ny, bc; name=:phi_bc,
        )
        @test lowered_logfv.west == lowered.west
        @test lowered_logfv.north == lowered.north

        bad_bc = Kraken.FVFDFieldBC2D(;
            west=[1.0, 2.0],
            east=zeros(Float64, Ny),
            south=zeros(Float64, Nx),
            north=zeros(Float64, Nx),
        )
        @test_throws DimensionMismatch Kraken.fvfd_transfer_field_bc_2d(
            bad_bc, KernelAbstractions.CPU(), Float64, Nx, Ny, bc; name=:bad_bc,
        )
    end

    @testset "q_wall lowering ignores pure halfway and averages curved normals" begin
        q_wall = zeros(Float64, 6, 5, 9)
        q_wall[3, 3, 3] = 0.5
        embedded = Kraken.fvfd_embedded_boundary_from_qwall_2d(q_wall)
        @test embedded.wall_inv_distance[3, 3] == 0.0
        @test embedded.wall_fraction[3, 3] == 0.0

        q_wall[3, 3, 6] = 0.51
        q_wall[3, 3, 7] = 0.51
        embedded = Kraken.fvfd_embedded_boundary_from_qwall_2d(q_wall)
        @test embedded.wall_nx[3, 3] ≈ 0.0 atol=FVFD_ATOL
        @test embedded.wall_ny[3, 3] ≈ -1.0 atol=FVFD_ATOL
        @test inv(embedded.wall_inv_distance[3, 3]) ≈ (0.5 + 0.51 + 0.51) / 3 atol=FVFD_ATOL
        @test embedded.wall_distance[3, 3] ≈ inv(embedded.wall_inv_distance[3, 3]) atol=FVFD_ATOL
        @test embedded.cut_count[3, 3] == 3
        @test 0.9 < embedded.cell_fraction[3, 3] <= 1.0

        q_wall .= 0.0
        q_wall[3, 3, 6] = 0.37
        embedded = Kraken.fvfd_embedded_boundary_from_qwall_2d(q_wall)
        @test embedded.wall_nx[3, 3] ≈ -sqrt(2) / 2 atol=FVFD_ATOL
        @test embedded.wall_ny[3, 3] ≈ -sqrt(2) / 2 atol=FVFD_ATOL
        @test embedded.cut_count[3, 3] == 1
        @test embedded.west_fraction[3, 3] ≈ 1.0 atol=FVFD_ATOL
        @test embedded.south_fraction[3, 3] ≈ 1.0 atol=FVFD_ATOL
        @test embedded.east_fraction[3, 3] ≈ 0.74 atol=FVFD_ATOL
        @test embedded.north_fraction[3, 3] ≈ 0.74 atol=FVFD_ATOL
        @test embedded.wall_fraction[3, 3] ≈ hypot(0.26, 0.26) atol=FVFD_ATOL

        q_wall, _ = Kraken.precompute_q_wall_cylinder(900, 120, 450.0, 59.5, 30.0)
        embedded = Kraken.fvfd_embedded_boundary_from_qwall_2d(q_wall)
        @test embedded.cut_count[439, 33] == 4
        @test embedded.wall_nx[439, 33] ≈ -0.382683432 atol=1e-8
        @test embedded.wall_ny[439, 33] ≈ -0.923879533 atol=1e-8
        @test embedded.wall_distance[439, 33] ≈ inv(embedded.wall_inv_distance[439, 33]) atol=FVFD_ATOL
        @test 0.49 < embedded.cell_fraction[439, 33] < 0.51
        @test embedded.wall_fraction[439, 33] > 0.0
    end

    @testset "embedded velocity gradient uses FVFD geometry" begin
        Nx, Ny = 6, 5
        is_solid = falses(Nx, Ny)
        bc = Kraken.fvfd_openx_wally_bcspec_2d()
        q_wall = zeros(Float64, Nx, Ny, 9)
        q_wall[3, 3, 2] = 0.25
        embedded_h = Kraken.fvfd_embedded_boundary_from_qwall_2d(
            q_wall; include_axis_aligned=true,
        )
        embedded = Kraken.fvfd_transfer_embedded_boundary_2d(
            embedded_h, KernelAbstractions.CPU(), Float64,
        )

        wall_distance = embedded_h.wall_distance[3, 3]
        @test wall_distance ≈ 0.375 atol=FVFD_ATOL
        ux = zeros(Float64, Nx, Ny)
        uy = zeros(Float64, Nx, Ny)
        ux[3, 3] = wall_distance
        uy[3, 3] = 2 * wall_distance
        dudx = zeros(Float64, Nx, Ny)
        dudy = zeros(Float64, Nx, Ny)
        dvdx = zeros(Float64, Nx, Ny)
        dvdy = zeros(Float64, Nx, Ny)

        Kraken.fvfd_velocity_gradient_embedded_2d!(
            dudx, dudy, dvdx, dvdy, ux, uy, is_solid, 1.0, 1.0, bc, embedded,
        )
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        @test dudx[3, 3] ≈ -1.0 atol=FVFD_ATOL
        @test dudy[3, 3] ≈ 0.0 atol=FVFD_ATOL
        @test dvdx[3, 3] ≈ -2.0 atol=FVFD_ATOL
        @test dvdy[3, 3] ≈ 0.0 atol=FVFD_ATOL

        geometry = Kraken.FVFDGeometry2D(
            is_solid, embedded, Kraken.FVFDPatch2D(1.0, 1.0), bc,
        )
        fill!(dudx, 0.0)
        fill!(dudy, 0.0)
        fill!(dvdx, 0.0)
        fill!(dvdy, 0.0)
        Kraken.fvfd_velocity_gradient_embedded_2d!(
            dudx, dudy, dvdx, dvdy, ux, uy, geometry,
        )
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        @test dudx[3, 3] ≈ -1.0 atol=FVFD_ATOL
        @test dvdx[3, 3] ≈ -2.0 atol=FVFD_ATOL
    end

    @testset "affine velocity gradient is exact with domain BCs and solids" begin
        Nx, Ny = 8, 7
        dx, dy = 0.25, 0.4
        ax, ay = 0.31, -0.17
        bx, by = -0.23, 0.29
        ux = [0.4 + ax * ((i - 0.5) * dx) + ay * ((j - 0.5) * dy) for i in 1:Nx, j in 1:Ny]
        uy = [-0.2 + bx * ((i - 0.5) * dx) + by * ((j - 0.5) * dy) for i in 1:Nx, j in 1:Ny]
        is_solid = falses(Nx, Ny)
        is_solid[4, 4] = true
        bc = Kraken.FVFDDomainBC2D(; west=:open, east=:open, south=:wall, north=:wall)

        dudx = zeros(Float64, Nx, Ny)
        dudy = zeros(Float64, Nx, Ny)
        dvdx = zeros(Float64, Nx, Ny)
        dvdy = zeros(Float64, Nx, Ny)
        Kraken.fvfd_velocity_gradient_2d!(
            dudx, dudy, dvdx, dvdy, ux, uy, is_solid, dx, dy, bc,
        )
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        for j in 1:Ny, i in 1:Nx
            if is_solid[i, j]
                @test dudx[i, j] == 0.0
                @test dudy[i, j] == 0.0
                @test dvdx[i, j] == 0.0
                @test dvdy[i, j] == 0.0
            else
                @test dudx[i, j] ≈ ax atol=5e-14 rtol=5e-14
                @test dudy[i, j] ≈ ay atol=5e-14 rtol=5e-14
                @test dvdx[i, j] ≈ bx atol=5e-14 rtol=5e-14
                @test dvdy[i, j] ≈ by atol=5e-14 rtol=5e-14
            end
        end
    end

    @testset "constant velocity has zero gradient for every domain BC code" begin
        Nx, Ny = 9, 8
        ux = fill(0.37, Nx, Ny)
        uy = fill(-0.21, Nx, Ny)
        is_solid = falses(Nx, Ny)
        is_solid[4, 5] = true

        bcs = (
            Kraken.FVFDDomainBC2D(; west=:periodic, east=:periodic, south=:periodic, north=:periodic),
            Kraken.FVFDDomainBC2D(; west=:open, east=:open, south=:wall, north=:wall),
            Kraken.FVFDDomainBC2D(; west=:wall, east=:wall, south=:open, north=:open),
        )

        for bc in bcs
            dudx = fill(NaN, Nx, Ny)
            dudy = fill(NaN, Nx, Ny)
            dvdx = fill(NaN, Nx, Ny)
            dvdy = fill(NaN, Nx, Ny)
            Kraken.fvfd_velocity_gradient_2d!(
                dudx, dudy, dvdx, dvdy, ux, uy, is_solid, 0.3, 0.4, bc,
            )
            KernelAbstractions.synchronize(KernelAbstractions.CPU())

            @test maximum(abs, dudx) <= 10eps(Float64)
            @test maximum(abs, dudy) <= 10eps(Float64)
            @test maximum(abs, dvdx) <= 10eps(Float64)
            @test maximum(abs, dvdy) <= 10eps(Float64)
        end
    end

    @testset "periodic sine gradient converges at second order" begin
        errors = Float64[]
        for N in (16, 32, 64)
            Nx = N
            Ny = N
            Lx = 1.0
            Ly = 1.25
            dx = Lx / Nx
            dy = Ly / Ny
            kx = 2pi / Lx
            ky = 2pi / Ly
            ux = [sin(kx * ((i - 0.5) * dx)) for i in 1:Nx, j in 1:Ny]
            uy = [cos(ky * ((j - 0.5) * dy)) for i in 1:Nx, j in 1:Ny]
            is_solid = falses(Nx, Ny)
            bc = Kraken.FVFDDomainBC2D(;
                west=:periodic, east=:periodic, south=:periodic, north=:periodic,
            )

            dudx = zeros(Float64, Nx, Ny)
            dudy = zeros(Float64, Nx, Ny)
            dvdx = zeros(Float64, Nx, Ny)
            dvdy = zeros(Float64, Nx, Ny)
            Kraken.fvfd_velocity_gradient_2d!(
                dudx, dudy, dvdx, dvdy, ux, uy, is_solid, dx, dy, bc,
            )
            KernelAbstractions.synchronize(KernelAbstractions.CPU())

            exact_dudx = [kx * cos(kx * ((i - 0.5) * dx)) for i in 1:Nx, j in 1:Ny]
            exact_dvdy = [-ky * sin(ky * ((j - 0.5) * dy)) for i in 1:Nx, j in 1:Ny]
            push!(errors, sqrt(sum(abs2, dudx .- exact_dudx) / length(dudx)))
            push!(errors, sqrt(sum(abs2, dvdy .- exact_dvdy) / length(dvdy)))
            @test maximum(abs, dudy) <= 10eps(Float64)
            @test maximum(abs, dvdx) <= 10eps(Float64)
        end

        @test errors[3] < 0.30 * errors[1]
        @test errors[5] < 0.30 * errors[3]
        @test errors[4] < 0.30 * errors[2]
        @test errors[6] < 0.30 * errors[4]
    end

    @testset "Couette and Poiseuille analytical gradients are exact" begin
        Nx, Ny = 7, 9
        dx, dy = 0.5, 0.25
        is_solid = falses(Nx, Ny)
        bc = Kraken.fvfd_openx_wally_bcspec_2d()

        shear = -0.42
        ux_couette = [0.11 + shear * ((j - 0.5) * dy) for i in 1:Nx, j in 1:Ny]
        uy_zero = zeros(Float64, Nx, Ny)
        out = [zeros(Float64, Nx, Ny) for _ in 1:4]
        Kraken.fvfd_velocity_gradient_2d!(out..., ux_couette, uy_zero, is_solid, dx, dy, bc)
        KernelAbstractions.synchronize(KernelAbstractions.CPU())
        @test maximum(abs, out[1]) <= FVFD_ATOL
        @test out[2] ≈ fill(shear, Nx, Ny) atol=5e-14 rtol=5e-14
        @test maximum(abs, out[3]) <= FVFD_ATOL
        @test maximum(abs, out[4]) <= FVFD_ATOL

        a = -0.33
        b = 0.08
        c = 0.19
        ux_poiseuille = [
            a * ((j - 0.5) * dy)^2 + b * ((j - 0.5) * dy) + c for i in 1:Nx, j in 1:Ny
        ]
        fill!.(out, 0.0)
        Kraken.fvfd_velocity_gradient_2d!(out..., ux_poiseuille, uy_zero, is_solid, dx, dy, bc)
        KernelAbstractions.synchronize(KernelAbstractions.CPU())
        exact_dudy = [2a * ((j - 0.5) * dy) + b for i in 1:Nx, j in 1:Ny]
        @test maximum(abs, out[1]) <= FVFD_ATOL
        @test out[2] ≈ exact_dudy atol=5e-14 rtol=5e-14
        @test maximum(abs, out[3]) <= FVFD_ATOL
        @test maximum(abs, out[4]) <= FVFD_ATOL
    end

    @testset "quadratic gradient is exact around an internal solid cell" begin
        Nx, Ny = 9, 9
        dx, dy = 0.2, 0.3
        is_solid = falses(Nx, Ny)
        is_solid[5, 5] = true
        bc = Kraken.FVFDDomainBC2D(; west=:open, east=:open, south=:open, north=:open)

        ux = [
            0.7 + 0.2 * ((i - 0.5) * dx) - 0.4 * ((j - 0.5) * dy) +
            0.13 * ((i - 0.5) * dx)^2 - 0.08 * ((j - 0.5) * dy)^2 +
            0.05 * ((i - 0.5) * dx) * ((j - 0.5) * dy)
            for i in 1:Nx, j in 1:Ny
        ]
        uy = [
            -0.3 - 0.11 * ((i - 0.5) * dx) + 0.17 * ((j - 0.5) * dy) -
            0.07 * ((i - 0.5) * dx)^2 + 0.09 * ((j - 0.5) * dy)^2 -
            0.04 * ((i - 0.5) * dx) * ((j - 0.5) * dy)
            for i in 1:Nx, j in 1:Ny
        ]

        dudx = zeros(Float64, Nx, Ny)
        dudy = zeros(Float64, Nx, Ny)
        dvdx = zeros(Float64, Nx, Ny)
        dvdy = zeros(Float64, Nx, Ny)
        Kraken.fvfd_velocity_gradient_2d!(
            dudx, dudy, dvdx, dvdy, ux, uy, is_solid, dx, dy, bc,
        )
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        for j in 1:Ny, i in 1:Nx
            x = (i - 0.5) * dx
            y = (j - 0.5) * dy
            if is_solid[i, j]
                @test dudx[i, j] == 0.0
                @test dudy[i, j] == 0.0
                @test dvdx[i, j] == 0.0
                @test dvdy[i, j] == 0.0
            else
                @test dudx[i, j] ≈ 0.2 + 0.26 * x + 0.05 * y atol=5e-14 rtol=5e-14
                @test dudy[i, j] ≈ -0.4 - 0.16 * y + 0.05 * x atol=5e-14 rtol=5e-14
                @test dvdx[i, j] ≈ -0.11 - 0.14 * x - 0.04 * y atol=5e-14 rtol=5e-14
                @test dvdy[i, j] ≈ 0.17 + 0.18 * y - 0.04 * x atol=5e-14 rtol=5e-14
            end
        end
    end

    @testset "geometry lowering transfers full FVFD geometry" begin
        Nx, Ny = 5, 4
        is_solid = falses(Nx, Ny)
        is_solid[2, 3] = true
        q_wall = zeros(Float64, Nx, Ny, 9)
        q_wall[3, 2, 6] = 0.37
        bc = Kraken.fvfd_openx_wally_bcspec_2d()

        geometry_h = Kraken.fvfd_geometry_from_lbm_2d(is_solid, q_wall, 0.2, 0.3, bc)
        geometry = Kraken.fvfd_transfer_geometry_2d(
            geometry_h, KernelAbstractions.CPU(), Float32,
        )

        @test geometry.patch.dx === Float32(0.2)
        @test geometry.patch.dy === Float32(0.3)
        @test geometry.patch.level == 0
        @test Array(geometry.is_solid) == is_solid
        @test eltype(geometry.embedded.wall_nx) === Float32
        @test Array(geometry.embedded.wall_nx)[3, 2] ≈ -Float32(sqrt(2) / 2) atol=1f-6
        @test Array(geometry.embedded.wall_ny)[3, 2] ≈ -Float32(sqrt(2) / 2) atol=1f-6
        @test eltype(geometry.embedded.cell_fraction) === Float32
        @test eltype(geometry.embedded.wall_fraction) === Float32
        @test Array(geometry.embedded.wall_fraction)[3, 2] ≈ Float32(hypot(0.26, 0.26)) atol=1f-6
        @test Array(geometry.embedded.east_fraction)[3, 2] ≈ Float32(0.74) atol=1f-6
        @test Array(geometry.embedded.north_fraction)[3, 2] ≈ Float32(0.74) atol=1f-6
        @test Array(geometry.embedded.cut_count)[3, 2] == 1
    end

    @testset "tensor divergence is exact with FVFD geometry and domain BCs" begin
        Nx, Ny = 8, 7
        dx, dy = 0.3, 0.2
        axx, axy = 0.07, -0.03
        bxy, byy = 0.02, 0.05
        tauxx = [
            0.1 + axx * ((i - 0.5) * dx) - 0.01 * ((j - 0.5) * dy)
            for i in 1:Nx, j in 1:Ny
        ]
        tauxy = [
            -0.2 + axy * ((i - 0.5) * dx) + bxy * ((j - 0.5) * dy)
            for i in 1:Nx, j in 1:Ny
        ]
        tauyy = [
            0.3 + 0.04 * ((i - 0.5) * dx) + byy * ((j - 0.5) * dy)
            for i in 1:Nx, j in 1:Ny
        ]
        is_solid = falses(Nx, Ny)
        bc = Kraken.fvfd_openx_wally_bcspec_2d()
        geometry = Kraken.FVFDGeometry2D(
            is_solid,
            Kraken.fvfd_empty_embedded_boundary_2d(Nx, Ny),
            Kraken.FVFDPatch2D(dx, dy),
            bc,
        )
        fx = similar(tauxx)
        fy = similar(tauxx)

        Kraken.fvfd_tensor_divergence_2d!(fx, fy, tauxx, tauxy, tauyy, geometry)
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        for j in 1:Ny, i in 1:Nx
            @test fx[i, j] ≈ axx + bxy atol=5e-14 rtol=5e-14
            @test fy[i, j] ≈ axy + byy atol=5e-14 rtol=5e-14
        end
    end

    @testset "embedded tensor divergence balances constant stress on q_wall cuts" begin
        Nx, Ny = 5, 5
        q_wall = zeros(Float64, Nx, Ny, 9)
        q_wall[3, 3, 6] = 0.37
        is_solid = falses(Nx, Ny)
        bc = Kraken.FVFDDomainBC2D(;
            west=:periodic, east=:periodic, south=:periodic, north=:periodic,
        )
        geometry = Kraken.fvfd_geometry_from_lbm_2d(is_solid, q_wall, 1.0, 1.0, bc)
        tauxx = fill(1.2, Nx, Ny)
        tauxy = fill(-0.4, Nx, Ny)
        tauyy = fill(0.7, Nx, Ny)
        fx = fill(NaN, Nx, Ny)
        fy = fill(NaN, Nx, Ny)

        Kraken.fvfd_tensor_divergence_embedded_2d!(fx, fy, tauxx, tauxy, tauyy, geometry)
        @test abs(fx[3, 3]) <= FVFD_ATOL
        @test abs(fy[3, 3]) <= FVFD_ATOL

        fill!(fx, NaN)
        fill!(fy, NaN)
        Kraken.logfv_polymer_force_embedded_bc_aware_2d!(
            fx, fy, tauxx, tauxy, tauyy, geometry,
        )
        @test abs(fx[3, 3]) <= FVFD_ATOL
        @test abs(fy[3, 3]) <= FVFD_ATOL
    end

    @testset "coherent half-plane embedded geometry balances constant stress globally" begin
        Nx, Ny = 9, 9
        bc = Kraken.FVFDDomainBC2D(;
            west=:open, east=:open, south=:open, north=:open,
        )
        geometry = Kraken.fvfd_geometry_from_halfplane_2d(
            Nx, Ny, 1.0, 1.0, bc, 1.0, 1.0, -7.0,
        )
        @test count(geometry.embedded.cut_count .> 0) == 7
        @test count(geometry.is_solid) == 21
        cut_mask = geometry.embedded.cut_count .> 0
        @test all(geometry.embedded.cell_fraction[cut_mask] .≈ 0.5)
        @test all(geometry.embedded.wall_fraction[cut_mask] .≈ sqrt(2))

        tauxx = fill(1.2, Nx, Ny)
        tauxy = fill(-0.4, Nx, Ny)
        tauyy = fill(0.7, Nx, Ny)
        fx = fill(NaN, Nx, Ny)
        fy = fill(NaN, Nx, Ny)

        Kraken.fvfd_tensor_divergence_embedded_2d!(fx, fy, tauxx, tauxy, tauyy, geometry)
        fluid_mask = .!geometry.is_solid
        @test maximum(abs, fx[fluid_mask]) <= 5e-14
        @test maximum(abs, fy[fluid_mask]) <= 5e-14

        tx = fill(NaN, Nx, Ny)
        ty = fill(NaN, Nx, Ny)
        Kraken.fvfd_embedded_wall_traction_2d!(tx, ty, tauxx, tauxy, tauyy, geometry)
        @test sum(tx) ≈ 7 * (1.2 - 0.4) atol=5e-14 rtol=5e-14
        @test sum(ty) ≈ 7 * (-0.4 + 0.7) atol=5e-14 rtol=5e-14

        fill!(tx, NaN)
        fill!(ty, NaN)
        Kraken.logfv_embedded_wall_traction_2d!(tx, ty, tauxx, tauxy, tauyy, geometry)
        @test sum(tx) ≈ 7 * (1.2 - 0.4) atol=5e-14 rtol=5e-14
        @test sum(ty) ≈ 7 * (-0.4 + 0.7) atol=5e-14 rtol=5e-14

        metal_backend = _fvfd_optional_metal_backend()
        if metal_backend !== nothing
            FT = Float32
            geometry_d = Kraken.fvfd_transfer_geometry_2d(geometry, metal_backend, FT)
            tauxx_d = KernelAbstractions.allocate(metal_backend, FT, Nx, Ny)
            tauxy_d = KernelAbstractions.allocate(metal_backend, FT, Nx, Ny)
            tauyy_d = KernelAbstractions.allocate(metal_backend, FT, Nx, Ny)
            copyto!(tauxx_d, fill(FT(1.2), Nx, Ny))
            copyto!(tauxy_d, fill(FT(-0.4), Nx, Ny))
            copyto!(tauyy_d, fill(FT(0.7), Nx, Ny))
            fx_d = KernelAbstractions.zeros(metal_backend, FT, Nx, Ny)
            fy_d = KernelAbstractions.zeros(metal_backend, FT, Nx, Ny)

            Kraken.fvfd_tensor_divergence_embedded_2d!(
                fx_d, fy_d, tauxx_d, tauxy_d, tauyy_d, geometry_d,
            )
            fx_h = Array(fx_d)
            fy_h = Array(fy_d)
            @test maximum(abs, fx_h[fluid_mask]) <= 2e-5
            @test maximum(abs, fy_h[fluid_mask]) <= 2e-5

            tx_d = KernelAbstractions.zeros(metal_backend, FT, Nx, Ny)
            ty_d = KernelAbstractions.zeros(metal_backend, FT, Nx, Ny)
            Kraken.fvfd_embedded_wall_traction_2d!(
                tx_d, ty_d, tauxx_d, tauxy_d, tauyy_d, geometry_d,
            )
            @test Float64(sum(Array(tx_d))) ≈ 7 * (1.2 - 0.4) atol=2e-5 rtol=2e-5
            @test Float64(sum(Array(ty_d))) ≈ 7 * (-0.4 + 0.7) atol=2e-5 rtol=2e-5
        else
            @test true
        end
    end

    @testset "coherent circle embedded geometry balances constant stress and traction" begin
        Nx, Ny = 64, 64
        cx, cy, radius = 32.0, 32.0, 10.0
        bc = Kraken.FVFDDomainBC2D(;
            west=:open, east=:open, south=:open, north=:open,
        )
        geometry = Kraken.fvfd_geometry_from_circle_2d(
            Nx, Ny, 1.0, 1.0, bc, cx, cy, radius; samples=32,
        )
        cut_mask = geometry.embedded.cut_count .> 0
        @test count(cut_mask) == 68
        @test count(geometry.is_solid) == 276
        @test sum(geometry.embedded.wall_fraction) ≈ 2π * radius rtol=5e-3
        @test abs(sum(geometry.embedded.wall_fraction .* geometry.embedded.wall_nx)) <= FVFD_ATOL
        @test abs(sum(geometry.embedded.wall_fraction .* geometry.embedded.wall_ny)) <= FVFD_ATOL
        normal_dot_radial = Float64[]
        for idx in findall(cut_mask)
            i, j = Tuple(idx)
            x = (i - 0.5) - cx
            y = (j - 0.5) - cy
            r = hypot(x, y)
            push!(
                normal_dot_radial,
                geometry.embedded.wall_nx[i, j] * x / r +
                geometry.embedded.wall_ny[i, j] * y / r,
            )
        end
        @test minimum(normal_dot_radial) > 0.95

        shifted = Kraken.fvfd_embedded_boundary_from_circle_2d(
            100, 40, 50.5, 20.0, 10.0; samples=64,
        )
        near_tangent = (shifted.cut_count .> 0) .&
                       (shifted.cell_fraction .> 0.45) .&
                       (shifted.cell_fraction .< 0.55)
        @test count(near_tangent) > 0
        @test minimum(shifted.wall_distance[near_tangent]) > 0.2
        @test maximum(shifted.wall_inv_distance[near_tangent]) < 5.0

        tauxx = fill(1.2, Nx, Ny)
        tauxy = fill(-0.4, Nx, Ny)
        tauyy = fill(0.7, Nx, Ny)
        fx = fill(NaN, Nx, Ny)
        fy = fill(NaN, Nx, Ny)
        Kraken.fvfd_tensor_divergence_embedded_2d!(fx, fy, tauxx, tauxy, tauyy, geometry)
        fluid_mask = .!geometry.is_solid
        @test maximum(abs, fx[fluid_mask]) <= 5e-14
        @test maximum(abs, fy[fluid_mask]) <= 5e-14

        tx = fill(NaN, Nx, Ny)
        ty = fill(NaN, Nx, Ny)
        Kraken.fvfd_embedded_wall_traction_2d!(tx, ty, tauxx, tauxy, tauyy, geometry)
        @test abs(sum(tx)) <= 5e-13
        @test abs(sum(ty)) <= 5e-13
    end

    @testset "BSD force correction is exact with FVFD geometry and domain BCs" begin
        Nx, Ny = 8, 7
        dx, dy = 0.3, 0.2
        axx, axy = 0.04, -0.01
        ayx, ayy = -0.06, 0.03
        zeta = 0.75
        nu_p = 0.09
        is_solid = falses(Nx, Ny)
        ux = [
            0.1 + axx * ((i - 0.5) * dx)^2 + axy * ((j - 0.5) * dy)^2
            for i in 1:Nx, j in 1:Ny
        ]
        uy = [
            -0.2 + ayx * ((i - 0.5) * dx)^2 + ayy * ((j - 0.5) * dy)^2
            for i in 1:Nx, j in 1:Ny
        ]
        fx_poly = [0.02 + 0.01 * (i - 0.5) * dx for i in 1:Nx, j in 1:Ny]
        fy_poly = [-0.03 + 0.02 * (j - 0.5) * dy for i in 1:Nx, j in 1:Ny]
        bc = Kraken.fvfd_openx_wally_bcspec_2d()
        geometry = Kraken.FVFDGeometry2D(
            is_solid,
            Kraken.fvfd_empty_embedded_boundary_2d(Nx, Ny),
            Kraken.FVFDPatch2D(dx, dy),
            bc,
        )
        fx_total = similar(fx_poly)
        fy_total = similar(fy_poly)

        Kraken.fvfd_bsd_force_2d!(
            fx_total, fy_total, fx_poly, fy_poly, ux, uy, geometry, zeta, nu_p,
        )
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        expected_lap_ux = 2 * axx + 2 * axy
        expected_lap_uy = 2 * ayx + 2 * ayy
        for j in 1:Ny, i in 1:Nx
            @test fx_total[i, j] ≈
                  fx_poly[i, j] - zeta * nu_p * expected_lap_ux atol=5e-14 rtol=5e-14
            @test fy_total[i, j] ≈
                  fy_poly[i, j] - zeta * nu_p * expected_lap_uy atol=5e-14 rtol=5e-14
        end
    end

    @testset "face velocity lowering honors FVFD field BCs and solid masks" begin
        Nx, Ny = 5, 4
        ux = [0.2 * i - 0.03 * j for i in 1:Nx, j in 1:Ny]
        uy = [-0.1 * i + 0.07 * j for i in 1:Nx, j in 1:Ny]
        is_solid = falses(Nx, Ny)
        is_solid[2, 2] = true
        bc = Kraken.FVFDDomainBC2D(;
            west=:open, east=:open, south=:wall, north=:periodic,
        )
        ux_bc = Kraken.FVFDFieldBC2D(;
            west=[1.5 + 0.1j for j in 1:Ny],
            east=[2.5 + 0.1j for j in 1:Ny],
            south=zeros(Float64, Nx),
            north=zeros(Float64, Nx),
        )
        uy_bc = Kraken.FVFDFieldBC2D(;
            west=zeros(Float64, Ny),
            east=zeros(Float64, Ny),
            south=[-3.0 - 0.2i for i in 1:Nx],
            north=[4.0 + 0.2i for i in 1:Nx],
        )
        ux_face = fill(NaN, Nx + 1, Ny)
        uy_face = fill(NaN, Nx, Ny + 1)

        Kraken.fvfd_cell_velocity_to_faces_2d!(
            ux_face, uy_face, ux, uy, is_solid, ux_bc, uy_bc, bc,
        )
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        for j in 1:Ny
            @test ux_face[1, j] ≈ ux_bc.west[j] atol=FVFD_ATOL
            @test ux_face[Nx + 1, j] ≈ ux_bc.east[j] atol=FVFD_ATOL
        end
        @test ux_face[2, 2] == 0.0
        @test ux_face[3, 2] == 0.0
        for i in 1:Nx
            @test uy_face[i, 1] == 0.0
            @test uy_face[i, Ny + 1] ≈ (uy[i, Ny] + uy[i, 1]) / 2 atol=FVFD_ATOL
        end

        geometry = Kraken.FVFDGeometry2D(
            is_solid,
            Kraken.fvfd_empty_embedded_boundary_2d(Nx, Ny),
            Kraken.FVFDPatch2D(1.0, 1.0),
            bc,
        )
        ux_face_geometry = fill(NaN, Nx + 1, Ny)
        uy_face_geometry = fill(NaN, Nx, Ny + 1)
        Kraken.fvfd_cell_velocity_to_faces_2d!(
            ux_face_geometry, uy_face_geometry, ux, uy, geometry, ux_bc, uy_bc,
        )
        KernelAbstractions.synchronize(KernelAbstractions.CPU())
        @test ux_face_geometry ≈ ux_face atol=FVFD_ATOL rtol=FVFD_ATOL
        @test uy_face_geometry ≈ uy_face atol=FVFD_ATOL rtol=FVFD_ATOL
    end

    @testset "embedded face velocity lowering applies q_wall aperture fractions" begin
        Nx, Ny = 5, 5
        ux = ones(Float64, Nx, Ny)
        uy = fill(2.0, Nx, Ny)
        is_solid = falses(Nx, Ny)
        q_wall = zeros(Float64, Nx, Ny, 9)
        q_wall[3, 3, 6] = 0.37
        bc = Kraken.FVFDDomainBC2D(;
            west=:periodic, east=:periodic, south=:periodic, north=:periodic,
        )
        geometry_h = Kraken.fvfd_geometry_from_lbm_2d(is_solid, q_wall, 1.0, 1.0, bc)
        ux_bc = Kraken.FVFDFieldBC2D(;
            west=zeros(Float64, Ny), east=zeros(Float64, Ny),
            south=zeros(Float64, Nx), north=zeros(Float64, Nx),
        )
        uy_bc = Kraken.FVFDFieldBC2D(;
            west=zeros(Float64, Ny), east=zeros(Float64, Ny),
            south=zeros(Float64, Nx), north=zeros(Float64, Nx),
        )
        ux_face_regular = fill(NaN, Nx + 1, Ny)
        uy_face_regular = fill(NaN, Nx, Ny + 1)
        ux_face_embedded = fill(NaN, Nx + 1, Ny)
        uy_face_embedded = fill(NaN, Nx, Ny + 1)
        ux_face_logfv = fill(NaN, Nx + 1, Ny)
        uy_face_logfv = fill(NaN, Nx, Ny + 1)

        Kraken.fvfd_cell_velocity_to_faces_2d!(
            ux_face_regular, uy_face_regular, ux, uy, geometry_h, ux_bc, uy_bc,
        )
        Kraken.fvfd_cell_velocity_to_faces_embedded_2d!(
            ux_face_embedded, uy_face_embedded, ux, uy, geometry_h, ux_bc, uy_bc,
        )
        Kraken.logfv_cell_velocity_to_faces_embedded_2d!(
            ux_face_logfv, uy_face_logfv, ux, uy, geometry_h, ux_bc, uy_bc,
        )
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        @test ux_face_regular[4, 3] ≈ 1.0 atol=FVFD_ATOL
        @test uy_face_regular[3, 4] ≈ 2.0 atol=FVFD_ATOL
        @test ux_face_embedded[4, 3] ≈ 0.74 atol=FVFD_ATOL
        @test uy_face_embedded[3, 4] ≈ 1.48 atol=FVFD_ATOL
        @test ux_face_embedded[3, 3] ≈ 1.0 atol=FVFD_ATOL
        @test uy_face_embedded[3, 3] ≈ 2.0 atol=FVFD_ATOL
        @test ux_face_logfv ≈ ux_face_embedded atol=FVFD_ATOL rtol=FVFD_ATOL
        @test uy_face_logfv ≈ uy_face_embedded atol=FVFD_ATOL rtol=FVFD_ATOL

        metal_backend = _fvfd_optional_metal_backend()
        if metal_backend !== nothing
            FT = Float32
            geometry_d = Kraken.fvfd_transfer_geometry_2d(geometry_h, metal_backend, FT)
            ux_d = KernelAbstractions.allocate(metal_backend, FT, Nx, Ny)
            uy_d = KernelAbstractions.allocate(metal_backend, FT, Nx, Ny)
            copyto!(ux_d, fill(one(FT), Nx, Ny))
            copyto!(uy_d, fill(FT(2), Nx, Ny))
            ux_bc_d = Kraken.fvfd_transfer_field_bc_2d(
                ux_bc, metal_backend, FT, Nx, Ny, bc; name=:ux_bc,
            )
            uy_bc_d = Kraken.fvfd_transfer_field_bc_2d(
                uy_bc, metal_backend, FT, Nx, Ny, bc; name=:uy_bc,
            )
            ux_face_d = KernelAbstractions.zeros(metal_backend, FT, Nx + 1, Ny)
            uy_face_d = KernelAbstractions.zeros(metal_backend, FT, Nx, Ny + 1)

            Kraken.fvfd_cell_velocity_to_faces_embedded_2d!(
                ux_face_d, uy_face_d, ux_d, uy_d, geometry_d, ux_bc_d, uy_bc_d,
            )

            @test Float64(Array(ux_face_d)[4, 3]) ≈ 0.74 atol=2e-5 rtol=2e-5
            @test Float64(Array(uy_face_d)[3, 4]) ≈ 1.48 atol=2e-5 rtol=2e-5
        else
            @test true
        end
    end

    @testset "embedded advection wrapper preserves constants through q_wall apertures" begin
        Nx, Ny = 5, 5
        ux = ones(Float64, Nx, Ny)
        uy = fill(2.0, Nx, Ny)
        q_wall = zeros(Float64, Nx, Ny, 9)
        q_wall[3, 3, 6] = 0.37
        is_solid = falses(Nx, Ny)
        bc = Kraken.FVFDDomainBC2D(;
            west=:periodic, east=:periodic, south=:periodic, north=:periodic,
        )
        geometry = Kraken.fvfd_geometry_from_lbm_2d(is_solid, q_wall, 1.0, 1.0, bc)
        ux_bc = Kraken.FVFDFieldBC2D(;
            west=zeros(Float64, Ny), east=zeros(Float64, Ny),
            south=zeros(Float64, Nx), north=zeros(Float64, Nx),
        )
        uy_bc = Kraken.FVFDFieldBC2D(;
            west=zeros(Float64, Ny), east=zeros(Float64, Ny),
            south=zeros(Float64, Nx), north=zeros(Float64, Nx),
        )
        phi_bc = Kraken.FVFDFieldBC2D(;
            west=zeros(Float64, Ny), east=zeros(Float64, Ny),
            south=zeros(Float64, Nx), north=zeros(Float64, Nx),
        )

        phi = fill(3.0, Nx, Ny)
        phi_out = fill(NaN, Nx, Ny)
        ux_face = fill(NaN, Nx + 1, Ny)
        uy_face = fill(NaN, Nx, Ny + 1)
        Kraken.fvfd_advect_upwind_embedded_2d!(
            phi_out, phi, phi_bc,
            ux_face, uy_face, ux, uy,
            geometry, ux_bc, uy_bc, 0.2,
        )
        @test phi_out ≈ phi atol=FVFD_ATOL rtol=FVFD_ATOL
        @test ux_face[4, 3] ≈ 0.74 atol=FVFD_ATOL
        @test uy_face[3, 4] ≈ 1.48 atol=FVFD_ATOL

        psixx = fill(0.3, Nx, Ny)
        psixy = fill(-0.04, Nx, Ny)
        psiyy = fill(0.2, Nx, Ny)
        psixx_out = fill(NaN, Nx, Ny)
        psixy_out = fill(NaN, Nx, Ny)
        psiyy_out = fill(NaN, Nx, Ny)
        fill!(ux_face, NaN)
        fill!(uy_face, NaN)
        Kraken.logfv_advect_upwind_embedded_2d!(
            psixx_out, psixy_out, psiyy_out,
            psixx, psixy, psiyy,
            phi_bc, phi_bc, phi_bc,
            ux_face, uy_face, ux, uy,
            geometry, ux_bc, uy_bc, 0.2,
        )
        @test psixx_out ≈ psixx atol=FVFD_ATOL rtol=FVFD_ATOL
        @test psixy_out ≈ psixy atol=FVFD_ATOL rtol=FVFD_ATOL
        @test psiyy_out ≈ psiyy atol=FVFD_ATOL rtol=FVFD_ATOL
        @test ux_face[4, 3] ≈ 0.74 atol=FVFD_ATOL
        @test uy_face[3, 4] ≈ 1.48 atol=FVFD_ATOL
    end

    @testset "open field BC validation rejects mismatched lengths" begin
        Nx, Ny = 4, 3
        ux = zeros(Float64, Nx, Ny)
        uy = zeros(Float64, Nx, Ny)
        is_solid = falses(Nx, Ny)
        ux_face = zeros(Float64, Nx + 1, Ny)
        uy_face = zeros(Float64, Nx, Ny + 1)
        bc = Kraken.FVFDDomainBC2D(; west=:open, east=:periodic, south=:wall, north=:open)

        ux_bad = Kraken.FVFDFieldBC2D(;
            west=zeros(Float64, Ny + 1),
            east=zeros(Float64, Ny),
            south=zeros(Float64, Nx),
            north=zeros(Float64, Nx),
        )
        uy_bad = Kraken.FVFDFieldBC2D(;
            west=zeros(Float64, Ny),
            east=zeros(Float64, Ny),
            south=zeros(Float64, Nx),
            north=zeros(Float64, Nx - 1),
        )
        @test_throws DimensionMismatch Kraken.fvfd_cell_velocity_to_faces_2d!(
            ux_face, uy_face, ux, uy, is_solid, ux_bad, uy_bad, bc,
        )

        phi = [0.1i - 0.2j for i in 1:Nx, j in 1:Ny]
        phi_out = similar(phi)
        phi_bad = Kraken.FVFDFieldBC2D(;
            west=zeros(Float64, Ny + 1),
            east=zeros(Float64, Ny),
            south=zeros(Float64, Nx),
            north=zeros(Float64, Nx - 1),
        )
        @test_throws DimensionMismatch Kraken.fvfd_advect_upwind_2d!(
            phi_out, phi, phi_bad, ux_face, uy_face, is_solid, 1.0, 1.0, bc, 0.1,
        )

        matrix_bc = Kraken.FVFDFieldBC2D(; west=phi, east=phi, south=phi, north=phi)
        fill!(ux_face, 0.0)
        fill!(uy_face, 0.0)
        Kraken.fvfd_advect_upwind_2d!(
            phi_out, phi, matrix_bc, ux_face, uy_face, is_solid,
            1.0, 1.0, Kraken.fvfd_periodicx_wally_bcspec_2d(), 0.1,
        )
        KernelAbstractions.synchronize(KernelAbstractions.CPU())
        @test phi_out == phi
    end

    @testset "scalar upwind advection is exact for affine fields with field BCs" begin
        Nx, Ny = 7, 6
        dx, dy = 0.5, 0.25
        u0, v0 = 0.2, 0.1
        a, bx, by = 0.7, -0.13, 0.19
        x(i) = (i - 0.5) * dx
        y(j) = (j - 0.5) * dy
        phi = [a + bx * x(i) + by * y(j) for i in 1:Nx, j in 1:Ny]
        phi_bc = Kraken.FVFDFieldBC2D(;
            west=[a + bx * (-0.5 * dx) + by * y(j) for j in 1:Ny],
            east=[a + bx * ((Nx + 0.5) * dx) + by * y(j) for j in 1:Ny],
            south=[a + bx * x(i) + by * (-0.5 * dy) for i in 1:Nx],
            north=[a + bx * x(i) + by * ((Ny + 0.5) * dy) for i in 1:Nx],
        )
        ux_face = fill(u0, Nx + 1, Ny)
        uy_face = fill(v0, Nx, Ny + 1)
        is_solid = falses(Nx, Ny)
        out = similar(phi)
        dt = 0.3
        bc = Kraken.FVFDDomainBC2D(; west=:open, east=:open, south=:open, north=:open)

        Kraken.fvfd_advect_upwind_2d!(
            out, phi, phi_bc, ux_face, uy_face, is_solid, dx, dy, bc, dt,
        )
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        expected = phi .- dt .* (u0 * bx + v0 * by)
        @test out ≈ expected atol=5e-14 rtol=5e-14

        u0, v0 = -0.17, -0.09
        fill!(ux_face, u0)
        fill!(uy_face, v0)
        fill!(out, NaN)
        Kraken.fvfd_advect_upwind_2d!(
            out, phi, phi_bc, ux_face, uy_face, is_solid, dx, dy, bc, dt,
        )
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        expected = phi .- dt .* (u0 * bx + v0 * by)
        @test out ≈ expected atol=5e-14 rtol=5e-14
    end

    @testset "periodic scalar and tensor advection wrap boundary faces" begin
        Nx, Ny = 5, 4
        dx, dy = 0.75, 0.5
        dt = 0.11
        bc = Kraken.FVFDDomainBC2D(;
            west=:periodic, east=:periodic, south=:periodic, north=:periodic,
        )
        is_solid = falses(Nx, Ny)
        ux_face = [
            (-1.0)^(I + J) * (0.07 + 0.01I - 0.002J)
            for I in 1:(Nx + 1), J in 1:Ny
        ]
        uy_face = [
            (-1.0)^(I + J + 1) * (0.05 - 0.003I + 0.008J)
            for I in 1:Nx, J in 1:(Ny + 1)
        ]
        phi = [0.2 + 0.17i - 0.09j + 0.01i * j for i in 1:Nx, j in 1:Ny]
        dummy_bc = Kraken.FVFDFieldBC2D(;
            west=fill(99.0, Ny),
            east=fill(98.0, Ny),
            south=fill(97.0, Nx),
            north=fill(96.0, Nx),
        )

        function periodic_reference(field)
            out = similar(field)
            inv_dx = inv(dx)
            inv_dy = inv(dy)
            for j in 1:Ny, i in 1:Nx
                ip = i == Nx ? 1 : i + 1
                im = i == 1 ? Nx : i - 1
                jp = j == Ny ? 1 : j + 1
                jm = j == 1 ? Ny : j - 1
                ue = ux_face[i + 1, j]
                uw = ux_face[i, j]
                vn = uy_face[i, j + 1]
                vs = uy_face[i, j]
                phie = ue >= 0 ? field[i, j] : field[ip, j]
                phiw = uw >= 0 ? field[im, j] : field[i, j]
                phin = vn >= 0 ? field[i, j] : field[i, jp]
                phis = vs >= 0 ? field[i, jm] : field[i, j]
                flux_div = (ue * phie - uw * phiw) * inv_dx +
                           (vn * phin - vs * phis) * inv_dy
                divu = (ue - uw) * inv_dx + (vn - vs) * inv_dy
                out[i, j] = field[i, j] - dt * (flux_div - field[i, j] * divu)
            end
            return out
        end

        scalar_out = similar(phi)
        Kraken.fvfd_advect_upwind_2d!(
            scalar_out, phi, dummy_bc, ux_face, uy_face, is_solid, dx, dy, bc, dt,
        )
        KernelAbstractions.synchronize(KernelAbstractions.CPU())
        @test scalar_out ≈ periodic_reference(phi) atol=5e-14 rtol=5e-14

        psixx = phi
        psixy = [0.4 - 0.04i + 0.06j for i in 1:Nx, j in 1:Ny]
        psiyy = [-0.2 + 0.03i - 0.05j + 0.02i * j for i in 1:Nx, j in 1:Ny]
        sym_out = [zeros(Float64, Nx, Ny) for _ in 1:3]
        Kraken.fvfd_sym2_advect_upwind_2d!(
            sym_out...,
            psixx, psixy, psiyy,
            dummy_bc, dummy_bc, dummy_bc,
            ux_face, uy_face, is_solid, dx, dy, bc, dt,
        )
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        @test sym_out[1] ≈ periodic_reference(psixx) atol=5e-14 rtol=5e-14
        @test sym_out[2] ≈ periodic_reference(psixy) atol=5e-14 rtol=5e-14
        @test sym_out[3] ≈ periodic_reference(psiyy) atol=5e-14 rtol=5e-14
    end

    @testset "log-FV face and advection wrappers match FVFD operators" begin
        Nx, Ny = 6, 5
        ux = [0.04 * i - 0.02 * j for i in 1:Nx, j in 1:Ny]
        uy = [-0.03 * i + 0.05 * j for i in 1:Nx, j in 1:Ny]
        is_solid = falses(Nx, Ny)
        is_solid[3, 3] = true
        bc = Kraken.FVFDDomainBC2D(; west=:open, east=:open, south=:open, north=:open)
        ux_west = [0.21 + 0.01j for j in 1:Ny]
        ux_east = [0.31 + 0.02j for j in 1:Ny]
        uy_south = [-0.24 - 0.01i for i in 1:Nx]
        uy_north = [0.18 + 0.015i for i in 1:Nx]

        ux_face_fvfd = fill(NaN, Nx + 1, Ny)
        uy_face_fvfd = fill(NaN, Nx, Ny + 1)
        ux_face_logfv = fill(NaN, Nx + 1, Ny)
        uy_face_logfv = fill(NaN, Nx, Ny + 1)
        Kraken.fvfd_cell_velocity_to_faces_2d!(
            ux_face_fvfd, uy_face_fvfd, ux, uy, is_solid,
            ux_west, ux_east, uy_south, uy_north, bc,
        )
        Kraken.logfv_cell_velocity_to_faces_bc_aware_2d!(
            ux_face_logfv, uy_face_logfv, ux, uy, is_solid,
            ux_west, ux_east, uy_south, uy_north, bc,
        )
        KernelAbstractions.synchronize(KernelAbstractions.CPU())
        @test ux_face_logfv ≈ ux_face_fvfd atol=FVFD_ATOL rtol=FVFD_ATOL
        @test uy_face_logfv ≈ uy_face_fvfd atol=FVFD_ATOL rtol=FVFD_ATOL

        psixx = [0.2 + 0.01i - 0.02j for i in 1:Nx, j in 1:Ny]
        psixy = [-0.1 + 0.03i + 0.01j for i in 1:Nx, j in 1:Ny]
        psiyy = [0.4 - 0.02i + 0.04j for i in 1:Nx, j in 1:Ny]
        west_xx = [0.11 + 0.01j for j in 1:Ny]
        west_xy = [0.21 + 0.01j for j in 1:Ny]
        west_yy = [0.31 + 0.01j for j in 1:Ny]
        east_xx = [0.12 + 0.01j for j in 1:Ny]
        east_xy = [0.22 + 0.01j for j in 1:Ny]
        east_yy = [0.32 + 0.01j for j in 1:Ny]
        south_xx = [0.13 + 0.01i for i in 1:Nx]
        south_xy = [0.23 + 0.01i for i in 1:Nx]
        south_yy = [0.33 + 0.01i for i in 1:Nx]
        north_xx = [0.14 + 0.01i for i in 1:Nx]
        north_xy = [0.24 + 0.01i for i in 1:Nx]
        north_yy = [0.34 + 0.01i for i in 1:Nx]
        fvfd_out = [zeros(Float64, Nx, Ny) for _ in 1:3]
        logfv_out = [zeros(Float64, Nx, Ny) for _ in 1:3]
        dt = 0.07
        dx_adv, dy_adv = 0.7, 0.45

        Kraken.fvfd_sym2_advect_upwind_2d!(
            fvfd_out...,
            psixx, psixy, psiyy,
            Kraken.FVFDFieldBC2D(west_xx, east_xx, south_xx, north_xx),
            Kraken.FVFDFieldBC2D(west_xy, east_xy, south_xy, north_xy),
            Kraken.FVFDFieldBC2D(west_yy, east_yy, south_yy, north_yy),
            ux_face_fvfd, uy_face_fvfd, is_solid, dx_adv, dy_adv, bc, dt,
        )
        Kraken.logfv_advect_upwind_bc_aware_2d!(
            logfv_out...,
            psixx, psixy, psiyy,
            west_xx, west_xy, west_yy,
            east_xx, east_xy, east_yy,
            south_xx, south_xy, south_yy,
            north_xx, north_xy, north_yy,
            ux_face_logfv, uy_face_logfv, is_solid, dx_adv, dy_adv, bc, dt,
        )
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        for k in 1:3
            @test logfv_out[k] ≈ fvfd_out[k] atol=FVFD_ATOL rtol=FVFD_ATOL
            @test logfv_out[k][3, 3] == 0.0
        end
    end

    @testset "log-FV gradient wrappers match FVFD operator" begin
        Nx, Ny = 6, 5
        is_solid = falses(Nx, Ny)
        ux = [0.1 * i - 0.03 * j for i in 1:Nx, j in 1:Ny]
        uy = [-0.04 * i + 0.07 * j for i in 1:Nx, j in 1:Ny]
        bc = Kraken.fvfd_openx_wally_bcspec_2d()

        fvfd_out = [zeros(Float64, Nx, Ny) for _ in 1:4]
        logfv_out = [zeros(Float64, Nx, Ny) for _ in 1:4]
        Kraken.fvfd_velocity_gradient_2d!(fvfd_out..., ux, uy, is_solid, 1.0, 1.0, bc)
        Kraken.logfv_velocity_gradient_bc_aware_2d!(logfv_out..., ux, uy, is_solid, 1.0, 1.0, bc)
        KernelAbstractions.synchronize(KernelAbstractions.CPU())

        for k in 1:4
            @test logfv_out[k] ≈ fvfd_out[k] atol=FVFD_ATOL rtol=FVFD_ATOL
        end
    end
end
