using Test
using Kraken

@testset "Multi-block ghost exchange (v0.3 Phase A.4)" begin

    # Build a horizontally-split 2-block mesh: left = [0,0.5] × [0,1],
    # right = [0.5,1] × [0,1]. Shared column at x = 0.5 (f_left[Nξ,:,:]
    # ↔ f_right[1,:,:]).
    function mk_we_mbm(; Nx_each=5, Ny=4, T=Float64)
        mesh_l = cartesian_mesh(; x_min=0.0, x_max=0.5,
                                   y_min=0.0, y_max=1.0, Nx=Nx_each, Ny=Ny, FT=T)
        mesh_r = cartesian_mesh(; x_min=0.5, x_max=1.0,
                                   y_min=0.0, y_max=1.0, Nx=Nx_each, Ny=Ny, FT=T)
        blk_l = Block(:left,  mesh_l; west=:inlet, east=:interface,
                                       south=:wall, north=:wall)
        blk_r = Block(:right, mesh_r; west=:interface, east=:outlet,
                                       south=:wall, north=:wall)
        iface = Interface(; from=(:left, :east), to=(:right, :west))
        mbm = MultiBlockMesh2D([blk_l, blk_r]; interfaces=[iface])
        return mbm, mesh_l.Nξ, mesh_l.Nη
    end

    # Vertically-split 2-block mesh: bottom = [0,1] × [0,0.5], top = [0,1] × [0.5,1].
    function mk_sn_mbm(; Nx=4, Ny_each=5, T=Float64)
        mesh_b = cartesian_mesh(; x_min=0.0, x_max=1.0,
                                   y_min=0.0, y_max=0.5, Nx=Nx, Ny=Ny_each, FT=T)
        mesh_t = cartesian_mesh(; x_min=0.0, x_max=1.0,
                                   y_min=0.5, y_max=1.0, Nx=Nx, Ny=Ny_each, FT=T)
        blk_b = Block(:bot, mesh_b; west=:wall, east=:wall,
                                     south=:inlet, north=:interface)
        blk_t = Block(:top, mesh_t; west=:wall, east=:wall,
                                     south=:interface, north=:outlet)
        iface = Interface(; from=(:bot, :north), to=(:top, :south))
        mbm = MultiBlockMesh2D([blk_b, blk_t]; interfaces=[iface])
        return mbm, mesh_b.Nξ, mesh_b.Nη
    end

    # ---- West-East exchange invariants ---------------------------------
    @testset "W-E: +cqx copied left→right; -cqx copied right→left" begin
        mbm, Nξ, Nη = mk_we_mbm()
        f_l = zeros(Float64, Nξ, Nη, 9)
        f_r = zeros(Float64, Nξ, Nη, 9)
        # mark every edge-interface population with a known signature
        for q in 1:9
            f_l[Nξ, :, q] .= 1.0 + 0.01 * q     # "left value"  for each q at interface
            f_r[1,  :, q] .= 100.0 + 0.01 * q   # "right value" for each q at interface
        end
        exchange_ghost_2d!(mbm, [f_l, f_r])
        # +cqx: q ∈ (2,6,9) — should have been copied LEFT → RIGHT
        for q in (2, 6, 9)
            @test all(f_r[1, :, q] .== 1.0 + 0.01 * q)   # right now holds left's value
            @test all(f_l[Nξ, :, q] .== 1.0 + 0.01 * q)  # left unchanged
        end
        # -cqx: q ∈ (4,7,8) — RIGHT → LEFT
        for q in (4, 7, 8)
            @test all(f_l[Nξ, :, q] .== 100.0 + 0.01 * q)
            @test all(f_r[1,  :, q] .== 100.0 + 0.01 * q)
        end
        # cqx=0: q ∈ (1,3,5) — synced to left's value (deterministic)
        for q in (1, 3, 5)
            @test all(f_r[1, :, q] .== 1.0 + 0.01 * q)
            @test all(f_l[Nξ, :, q] .== 1.0 + 0.01 * q)
        end
    end

    @testset "W-E: away-from-edge populations untouched" begin
        mbm, Nξ, Nη = mk_we_mbm()
        f_l = rand(Float64, Nξ, Nη, 9)
        f_r = rand(Float64, Nξ, Nη, 9)
        f_l_before = copy(f_l); f_r_before = copy(f_r)
        exchange_ghost_2d!(mbm, [f_l, f_r])
        # only column i = Nξ (left) and i = 1 (right) should change
        @test f_l[1:Nξ-1, :, :] == f_l_before[1:Nξ-1, :, :]
        @test f_r[2:Nξ,   :, :] == f_r_before[2:Nξ,   :, :]
    end

    @testset "W-E: swapped interface (from=right:west, to=left:east) matches" begin
        mesh_l = cartesian_mesh(; x_min=0.0, x_max=0.5, y_min=0.0, y_max=1.0, Nx=5, Ny=4)
        mesh_r = cartesian_mesh(; x_min=0.5, x_max=1.0, y_min=0.0, y_max=1.0, Nx=5, Ny=4)
        blk_l = Block(:left,  mesh_l; west=:inlet,     east=:interface,
                                       south=:wall,    north=:wall)
        blk_r = Block(:right, mesh_r; west=:interface, east=:outlet,
                                       south=:wall,    north=:wall)
        # Swapped interface direction — should yield the same exchange
        iface = Interface(; from=(:right, :west), to=(:left, :east))
        mbm = MultiBlockMesh2D([blk_l, blk_r]; interfaces=[iface])
        f_l = rand(Float64, 5, 4, 9)
        f_r = rand(Float64, 5, 4, 9)
        # Save the expected "canonical direction" result
        f_l_c = copy(f_l); f_r_c = copy(f_r)
        iface_c = Interface(; from=(:left, :east), to=(:right, :west))
        mbm_c = MultiBlockMesh2D([blk_l, blk_r]; interfaces=[iface_c])
        exchange_ghost_2d!(mbm_c, [f_l_c, f_r_c])
        exchange_ghost_2d!(mbm,   [f_l,   f_r])
        @test f_l == f_l_c
        @test f_r == f_r_c
    end

    # ---- South-North exchange invariants -------------------------------
    @testset "S-N: +cqy bot→top; -cqy top→bot" begin
        mbm, Nξ, Nη = mk_sn_mbm()
        f_b = zeros(Float64, Nξ, Nη, 9)
        f_t = zeros(Float64, Nξ, Nη, 9)
        for q in 1:9
            f_b[:, Nη, q] .= 10.0 + 0.1 * q
            f_t[:, 1,  q] .= -10.0 - 0.1 * q
        end
        exchange_ghost_2d!(mbm, [f_b, f_t])
        for q in (3, 6, 7)     # +cqy
            @test all(f_t[:, 1, q] .== 10.0 + 0.1 * q)
            @test all(f_b[:, Nη, q] .== 10.0 + 0.1 * q)
        end
        for q in (5, 8, 9)     # -cqy
            @test all(f_b[:, Nη, q] .== -10.0 - 0.1 * q)
            @test all(f_t[:, 1, q] .== -10.0 - 0.1 * q)
        end
        for q in (1, 2, 4)     # cqy = 0
            @test all(f_t[:, 1, q] .== 10.0 + 0.1 * q)
            @test all(f_b[:, Nη, q] .== 10.0 + 0.1 * q)
        end
    end

    # ---- Idempotence: applying exchange twice changes nothing ----------
    @testset "exchange is idempotent when both sides already agree" begin
        mbm, Nξ, Nη = mk_we_mbm()
        f_l = rand(Float64, Nξ, Nη, 9)
        f_r = rand(Float64, Nξ, Nη, 9)
        exchange_ghost_2d!(mbm, [f_l, f_r])   # first exchange → sides now agree
        f_l_snap = copy(f_l); f_r_snap = copy(f_r)
        exchange_ghost_2d!(mbm, [f_l, f_r])   # second exchange → no change
        @test f_l == f_l_snap
        @test f_r == f_r_snap
    end

    # ---- Error paths ---------------------------------------------------
    @testset "length mismatch throws" begin
        mbm, _, _ = mk_we_mbm()
        @test_throws ErrorException exchange_ghost_2d!(mbm, [zeros(3,3,9)])
    end

    @testset "unsupported same-normal edge pair throws" begin
        # east ↔ east would mean two blocks with their :east edges stuck
        # together — MVP does not support (would require flip).
        mesh = cartesian_mesh(; x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, Nx=4, Ny=4)
        blk_a = Block(:a, mesh; west=:w, east=:interface, south=:s, north=:n)
        blk_b = Block(:b, mesh; west=:w, east=:interface, south=:s, north=:n)
        iface = Interface(; from=(:a, :east), to=(:b, :east))
        mbm = MultiBlockMesh2D([blk_a, blk_b]; interfaces=[iface])
        f = [rand(4, 4, 9), rand(4, 4, 9)]
        @test_throws ErrorException exchange_ghost_2d!(mbm, f)
    end
end
