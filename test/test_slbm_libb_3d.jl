using Test
using Kraken
using KernelAbstractions

# ===========================================================================
# WP-3D-4 — SLBM TRT + LI-BB 3D unit tests.
#
# Three families:
#   1. Mass + momentum conservation (uniform flow, periodic, no body)
#   2. Bit-exact equivalence with halfway-BB pull-stream on a uniform
#      Cartesian mesh (interior cells, no cut links)
#   3. Sphere q_wall sanity (cut-link count, q_w ∈ (0, 1])
# ===========================================================================

@testset "SLBM TRT + LI-BB 3D" begin
    backend = KernelAbstractions.CPU()
    T = Float64

    # ----------------------------------------------------------------
    # Helper: build a uniform Cartesian 3D mesh + SLBM geometry.
    # ----------------------------------------------------------------
    function _build_uniform_geom(Nx, Ny, Nz; periodic::Bool)
        if periodic
            mesh = build_mesh_3d(
                (ξ, η, ζ) -> (T(Nx) * ξ, T(Ny) * η, T(Nz) * ζ);
                Nξ=Nx, Nη=Ny, Nζ=Nz,
                periodic_ξ=true, periodic_η=true, periodic_ζ=true,
                dx_ref=one(T), FT=T)
        else
            mesh = cartesian_mesh_3d(; x_min=0.0, x_max=T(Nx - 1),
                                       y_min=0.0, y_max=T(Ny - 1),
                                       z_min=0.0, z_max=T(Nz - 1),
                                       Nx=Nx, Ny=Ny, Nz=Nz, FT=T)
        end
        geom_h = build_slbm_geometry_3d(mesh; local_cfl=false)
        geom = transfer_slbm_geometry_3d(geom_h, backend)
        return mesh, geom
    end

    # ----------------------------------------------------------------
    # 1. Conservation: uniform velocity field stays uniform forever.
    #
    # On a periodic mesh with no body and a velocity field initialised
    # to equilibrium with constant (u0, v0, w0), nothing should evolve:
    # f_neq = 0, collision is a no-op, semi-Lagrangian pull lands on
    # neighbour nodes carrying identical equilibrium populations.
    # ----------------------------------------------------------------
    @testset "uniform flow stays uniform (periodic, no body)" begin
        Nx, Ny, Nz = 16, 16, 16
        mesh, geom = _build_uniform_geom(Nx, Ny, Nz; periodic=true)
        u0, v0, w0 = T(0.05), T(0.02), T(-0.03)
        ν = T(0.05)

        is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny, Nz)
        fill!(is_solid, false)
        q_wall   = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19); fill!(q_wall, zero(T))
        uw_x     = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19); fill!(uw_x, zero(T))
        uw_y     = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19); fill!(uw_y, zero(T))
        uw_z     = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19); fill!(uw_z, zero(T))

        f_h = zeros(T, Nx, Ny, Nz, 19)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx, q in 1:19
            f_h[i, j, k, q] = Kraken.equilibrium(D3Q19(), one(T), u0, v0, w0, q)
        end
        fa = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19); copyto!(fa, f_h)
        fb = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19); fill!(fb, zero(T))
        ρ  = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz); fill!(ρ, one(T))
        ux = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz); fill!(ux, zero(T))
        uy = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz); fill!(uy, zero(T))
        uz = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz); fill!(uz, zero(T))

        for _ in 1:50
            slbm_trt_libb_step_3d!(fb, fa, ρ, ux, uy, uz, is_solid,
                                    q_wall, uw_x, uw_y, uw_z, geom, ν)
            KernelAbstractions.synchronize(backend)
            fa, fb = fb, fa
        end

        @test maximum(abs.(Array(ρ) .- one(T))) < 1e-12
        @test maximum(abs.(Array(ux) .- u0))   < 1e-12
        @test maximum(abs.(Array(uy) .- v0))   < 1e-12
        @test maximum(abs.(Array(uz) .- w0))   < 1e-12
    end

    # ----------------------------------------------------------------
    # 2. Bit-exact equivalence on a uniform Cartesian mesh.
    #
    # On a uniform Cartesian mesh, the precomputed departure indices
    # land EXACTLY on neighbour nodes (Δi = -c_qx, Δj = -c_qy, Δk = -c_qz
    # are integers). Trilinear interpolation then collapses to a single
    # node read. With no cut links, ApplyLiBBPrePhase_3D is a no-op.
    #
    # Hence slbm_trt_libb_step_3d! and fused_trt_libb_v2_step_3d! must
    # produce IDENTICAL post-collision populations on interior cells.
    # We check the interior i ∈ 2:Nx-1, j ∈ 2:Ny-1, k ∈ 2:Nz-1 because
    # the two kernels handle the domain edges differently (clamp-to-self
    # vs halfway-BB swap).
    # ----------------------------------------------------------------
    @testset "interior bit-exact match with fused_trt_libb_v2_step_3d!" begin
        Nx, Ny, Nz = 12, 10, 10
        mesh, geom = _build_uniform_geom(Nx, Ny, Nz; periodic=false)

        # No cut-links scenario: tag NO solid cells.
        is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny, Nz); fill!(is_solid, false)
        q_wall   = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19); fill!(q_wall, zero(T))
        uw_x     = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19); fill!(uw_x, zero(T))
        uw_y     = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19); fill!(uw_y, zero(T))
        uw_z     = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19); fill!(uw_z, zero(T))

        # Random-ish initial state (use a smooth field so both fall back
        # to interior reads predictably).
        f0 = zeros(T, Nx, Ny, Nz, 19)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx, q in 1:19
            ux0 = T(0.04) * sin(2π * (i - 1) / Nx)
            uy0 = T(0.02) * cos(2π * (j - 1) / Ny)
            uz0 = T(0.01) * sin(2π * (k - 1) / Nz)
            f0[i, j, k, q] = Kraken.equilibrium(D3Q19(), one(T), ux0, uy0, uz0, q)
        end

        # Path A: SLBM
        fa_A = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19); copyto!(fa_A, f0)
        fb_A = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19); fill!(fb_A, zero(T))
        ρA  = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz)
        uxA = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz)
        uyA = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz)
        uzA = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz)
        slbm_trt_libb_step_3d!(fb_A, fa_A, ρA, uxA, uyA, uzA, is_solid,
                                q_wall, uw_x, uw_y, uw_z, geom, T(0.05))
        KernelAbstractions.synchronize(backend)

        # Path B: halfway-BB Cartesian
        fa_B = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19); copyto!(fa_B, f0)
        fb_B = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz, 19); fill!(fb_B, zero(T))
        ρB  = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz)
        uxB = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz)
        uyB = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz)
        uzB = KernelAbstractions.allocate(backend, T, Nx, Ny, Nz)
        fused_trt_libb_v2_step_3d!(fb_B, fa_B, ρB, uxB, uyB, uzB, is_solid,
                                     q_wall, uw_x, uw_y, uw_z,
                                     Nx, Ny, Nz, T(0.05))
        KernelAbstractions.synchronize(backend)

        # Compare interior only.
        fA = Array(fb_A); fB = Array(fb_B)
        max_diff = maximum(abs.(fA[2:Nx-1, 2:Ny-1, 2:Nz-1, :] .-
                                 fB[2:Nx-1, 2:Ny-1, 2:Nz-1, :]))
        @test max_diff < 1e-12
    end

    # ----------------------------------------------------------------
    # 3. Sphere q_wall sanity.
    #
    # For a sphere of radius R inside the mesh, the precomputation
    # should flag a positive number of cut links and every q_w must
    # lie in (0, 1] (we use 0.5 as a safe fallback if the discriminant
    # is negative — only when the geometry is degenerate).
    # ----------------------------------------------------------------
    @testset "sphere q_wall sanity" begin
        Nx, Ny, Nz = 30, 20, 20
        mesh = cartesian_mesh_3d(; x_min=0.0, x_max=T(Nx - 1),
                                   y_min=0.0, y_max=T(Ny - 1),
                                   z_min=0.0, z_max=T(Nz - 1),
                                   Nx=Nx, Ny=Ny, Nz=Nz, FT=T)
        cx, cy, cz, R = T(15), T(10), T(10), T(4)

        is_solid_h = zeros(Bool, Nx, Ny, Nz)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            dx = mesh.X[i, j, k] - cx
            dy = mesh.Y[i, j, k] - cy
            dz = mesh.Z[i, j, k] - cz
            if dx^2 + dy^2 + dz^2 ≤ R^2
                is_solid_h[i, j, k] = true
            end
        end
        n_solid = count(is_solid_h)
        @test n_solid > 0

        q_wall, uwx, uwy, uwz =
            precompute_q_wall_slbm_sphere_3d(mesh, is_solid_h, cx, cy, cz, R; FT=T)
        n_cut = count(q_wall .> 0)
        @test n_cut > 0
        @test minimum(q_wall[q_wall .> 0]) > zero(T)
        @test maximum(q_wall) ≤ one(T)
        @test all(uwx .== zero(T))   # stationary sphere
        @test all(uwy .== zero(T))
        @test all(uwz .== zero(T))
    end
end
