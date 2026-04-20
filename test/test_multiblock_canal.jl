using Test
using Kraken

# =====================================================================
# Phase A.5b canal validation with ghost-layer f arrays.
#
# With pre-step ghost-fill on extended block arrays, each block's step
# reads VALID neighbour data at the interior-boundary (no halfway-BB
# fallback) and the interface is bit-exact vs single-block.
#
# What's NOT yet implemented: a helper that pre-fills PHYSICAL-wall
# ghosts with halfway-BB reflection. Without that, rows adjacent to a
# physical wall still differ from single-block. For this test we use
# a CLEAN configuration with no y-walls (south/north tagged as
# :interface → we close the y-direction by wrapping with a small
# periodic self-block, but easier: we just omit south/north BCs and
# let the kernel's halfway-BB fire at the extended outer ghosts where
# nothing reads it). We check:
#
# 1. Ghost fill + step preserves uniform rest state bit-exact (sanity).
# 2. After step with uniform-flow init, the two blocks' interior-boundary
#    COLUMNS at the x-interface agree EXACTLY (the ghost fill makes them
#    consistent — halo-strict couldn't achieve this at u ≠ 0).
# 3. Long-run smoke: 1000 steps with ghost-layer pipeline stays finite.
# =====================================================================

@testset "Multi-block canal with ghost layer (v0.3 Phase A.5b)" begin

    # Test geometry: left = [0, 5] × [0, 4],  right = [5, 10] × [0, 4].
    # Both blocks same Ny_phys = 5, Nx_phys = 5 (= 5 cells interior each).
    # For bit-exact vs single-block we'd also need physical-wall ghost
    # fill — scope deferred; here we assert INTERFACE consistency only.
    Nx_phys = 5; Ny_phys = 5
    ν = 0.1; ω = 1.0 / (3ν + 0.5)

    function build_mbm()
        mesh_A = cartesian_mesh(; x_min=0.0, x_max=5.0, y_min=0.0, y_max=4.0,
                                   Nx=Nx_phys, Ny=Ny_phys)
        mesh_B = cartesian_mesh(; x_min=5.0, x_max=10.0, y_min=0.0, y_max=4.0,
                                   Nx=Nx_phys, Ny=Ny_phys)
        blk_A = Block(:A, mesh_A; west=:inlet, east=:interface,
                                   south=:wall, north=:wall)
        blk_B = Block(:B, mesh_B; west=:interface, east=:outlet,
                                   south=:wall, north=:wall)
        iface = Interface(; from=(:A, :east), to=(:B, :west))
        return MultiBlockMesh2D([blk_A, blk_B]; interfaces=[iface])
    end

    function init_uniform_eq!(state::BlockState2D, ρ0, ux0, uy0)
        usq = ux0 * ux0 + uy0 * uy0
        Nx_ext, Ny_ext = ext_dims(state)
        @inbounds for j in 1:Ny_ext, i in 1:Nx_ext, q in 1:9
            state.f[i, j, q] = Kraken.feq_2d(Val(q), ρ0, ux0, uy0, usq)
        end
    end

    # ---- 1. Rest (u = 0) uniform remains uniform ---------------------
    @testset "u = 0 uniform equilibrium is preserved by ghost-layer step" begin
        mbm = build_mbm()
        @test isempty(sanity_check_multiblock(mbm; verbose=false))

        states = [allocate_block_state_2d(b; n_ghost=1) for b in mbm.blocks]
        init_uniform_eq!(states[1], 1.0, 0.0, 0.0)
        init_uniform_eq!(states[2], 1.0, 0.0, 0.0)

        f_out_A = similar(states[1].f); f_out_B = similar(states[2].f)
        is_solid_A = zeros(Bool, ext_dims(states[1])...)
        is_solid_B = zeros(Bool, ext_dims(states[2])...)

        for step in 1:10
            exchange_ghost_2d!(mbm, states)
            Nx_ext_A, Ny_ext_A = ext_dims(states[1])
            Nx_ext_B, Ny_ext_B = ext_dims(states[2])
            fused_bgk_step!(f_out_A, states[1].f, states[1].ρ, states[1].ux, states[1].uy,
                            is_solid_A, Nx_ext_A, Ny_ext_A, ω)
            fused_bgk_step!(f_out_B, states[2].f, states[2].ρ, states[2].ux, states[2].uy,
                            is_solid_B, Nx_ext_B, Ny_ext_B, ω)
            states[1].f, f_out_A = f_out_A, states[1].f
            states[2].f, f_out_B = f_out_B, states[2].f
        end

        # At u = 0 the feq is symmetric in ±x/±y so halfway-BB at outer
        # ghosts gives the same value as the init; interior should still
        # be exactly the initial equilibrium.
        ref = Kraken.feq_2d(Val(1), 1.0, 0.0, 0.0, 0.0)
        int_A = interior_f(states[1])
        int_B = interior_f(states[2])
        @test all(int_A[:, :, 1] .≈ ref)
        @test all(int_B[:, :, 1] .≈ ref)
        # Interface columns agree bit-exact in A and B
        @test int_A[Nx_phys, :, :] == int_B[1, :, :]
    end

    # ---- 2. u ≠ 0: step 1 interface bit-exact ----------------------
    # The ghost-layer exchange correctly fills A's east ghost from B's
    # interior-west (and vice versa) so the step at the physical-
    # interface column reads VALID populations on both sides. After ONE
    # step starting from uniform flow, A's interior-east column and B's
    # interior-west column match bit-exact.
    #
    # Multi-step bit-exactness additionally requires pre-filling
    # PHYSICAL-WALL ghosts with halfway-BB reflection so the wall-side
    # pollution does not propagate inward. That helper is deferred.
    @testset "u = 0.05 step-1 interface bit-exact match" begin
        # Torus: A west = B east, A east = B west (two interfaces, periodic x)
        mesh_A = cartesian_mesh(; x_min=0.0, x_max=5.0, y_min=0.0, y_max=4.0,
                                   Nx=Nx_phys, Ny=Ny_phys)
        mesh_B = cartesian_mesh(; x_min=5.0, x_max=10.0, y_min=0.0, y_max=4.0,
                                   Nx=Nx_phys, Ny=Ny_phys)
        mbm = build_mbm()
        @test isempty(sanity_check_multiblock(mbm; verbose=false))
        u0 = 0.05
        states = [allocate_block_state_2d(b; n_ghost=1) for b in mbm.blocks]
        init_uniform_eq!(states[1], 1.0, u0, 0.0)
        init_uniform_eq!(states[2], 1.0, u0, 0.0)
        f_out_A = similar(states[1].f); f_out_B = similar(states[2].f)
        is_solid_A = zeros(Bool, ext_dims(states[1])...)
        is_solid_B = zeros(Bool, ext_dims(states[2])...)

        # Single step; post-step, A's interior-east column at
        # (i = Nx_phys + n_ghost) and B's interior-west column at
        # (i = n_ghost + 1) must match bit-exactly. This is the fix
        # over halo-strict (cf. test_multiblock_canal old version in git)
        # where the interface agreement was machine-epsilon garbage due
        # to both blocks computing BGK on halfway-BB-polluted moments.
        exchange_ghost_2d!(mbm, states)
        Nx_ext_A, Ny_ext_A = ext_dims(states[1])
        Nx_ext_B, Ny_ext_B = ext_dims(states[2])
        fused_bgk_step!(f_out_A, states[1].f, states[1].ρ, states[1].ux, states[1].uy,
                        is_solid_A, Nx_ext_A, Ny_ext_A, ω)
        fused_bgk_step!(f_out_B, states[2].f, states[2].ρ, states[2].ux, states[2].uy,
                        is_solid_B, Nx_ext_B, Ny_ext_B, ω)
        ng = states[1].n_ghost
        a_col = view(f_out_A, Nx_phys + ng, (ng + 1):(ng + Ny_phys), :)
        b_col = view(f_out_B, ng + 1,        (ng + 1):(ng + Ny_phys), :)
        @test a_col == b_col     # bit-exact: identical ops on identical data
    end

    # ---- 3. Long-run smoke ------------------------------------------
    @testset "1000-step run produces finite populations" begin
        mbm = build_mbm()
        u0 = 0.02
        states = [allocate_block_state_2d(b; n_ghost=1) for b in mbm.blocks]
        init_uniform_eq!(states[1], 1.0, u0, 0.0)
        init_uniform_eq!(states[2], 1.0, u0, 0.0)
        f_out_A = similar(states[1].f); f_out_B = similar(states[2].f)
        is_solid_A = zeros(Bool, ext_dims(states[1])...)
        is_solid_B = zeros(Bool, ext_dims(states[2])...)

        for step in 1:1000
            exchange_ghost_2d!(mbm, states)
            Nx_ext_A, Ny_ext_A = ext_dims(states[1])
            Nx_ext_B, Ny_ext_B = ext_dims(states[2])
            fused_bgk_step!(f_out_A, states[1].f, states[1].ρ, states[1].ux, states[1].uy,
                            is_solid_A, Nx_ext_A, Ny_ext_A, ω)
            fused_bgk_step!(f_out_B, states[2].f, states[2].ρ, states[2].ux, states[2].uy,
                            is_solid_B, Nx_ext_B, Ny_ext_B, ω)
            states[1].f, f_out_A = f_out_A, states[1].f
            states[2].f, f_out_B = f_out_B, states[2].f
        end
        @test all(isfinite, interior_f(states[1]))
        @test all(isfinite, interior_f(states[2]))
    end
end
