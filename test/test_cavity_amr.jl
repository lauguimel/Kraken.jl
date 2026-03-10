using Test
using Kraken

@testset "AMR Projection" begin

    @testset "AMR cavity BCs" begin
        # Create 8x8 tree (level 3)
        tree = QuadTree(Float64, 3)
        refine_uniformly!(tree, 3)
        add_field!(tree, :u)
        add_field!(tree, :v)

        # Set all to 0.5 first
        foreach_leaf(tree) do level, i, j
            set_field!(tree, :u, level, i, j, 0.5)
            set_field!(tree, :v, level, i, j, 0.5)
        end

        apply_velocity_bc_amr!(tree, :u, :v)

        n = 8  # 2^3
        # Top row (j=8), interior cells (not corners): u=1
        for i in 2:7
            @test get_field(tree, :u, 3, i, 8) ≈ 1.0
            @test get_field(tree, :v, 3, i, 8) ≈ 0.0
        end

        # Bottom row (j=1): u=0, v=0
        for i in 1:8
            @test get_field(tree, :u, 3, i, 1) ≈ 0.0
            @test get_field(tree, :v, 3, i, 1) ≈ 0.0
        end

        # Left wall (i=1): u=0, v=0
        for j in 1:8
            @test get_field(tree, :u, 3, 1, j) ≈ 0.0
            @test get_field(tree, :v, 3, 1, j) ≈ 0.0
        end

        # Right wall (i=8): u=0, v=0
        for j in 1:8
            @test get_field(tree, :u, 3, 8, j) ≈ 0.0
            @test get_field(tree, :v, 3, 8, j) ≈ 0.0
        end

        # Corners: u=0
        @test get_field(tree, :u, 3, 1, 8) ≈ 0.0
        @test get_field(tree, :u, 3, 8, 8) ≈ 0.0

        # Interior cells: unchanged
        @test get_field(tree, :u, 3, 4, 4) ≈ 0.5
        @test get_field(tree, :v, 3, 4, 4) ≈ 0.5
    end

    @testset "AMR cavity convergence" begin
        # Small case: 16x16, Re=100, run enough steps
        tree, converged = run_cavity_amr(N=16, Re=100.0, cfl=0.2,
                                          max_steps=5000, tol=1e-6,
                                          verbose=false)

        # Should run without error
        @test tree isa QuadTree{Float64}
        @test nleaves(tree) == 16 * 16

        # Check velocity profile: the lid drives flow, so near the top
        # boundary (j = n-1, one cell below lid) u should be positive
        level = 4
        n = 16

        # One cell below the lid (j=n-1) at mid-x should have positive u
        u_near_lid = get_field(tree, :u, level, n ÷ 2, n - 1)
        @test u_near_lid > 0.0

        # The flow near the bottom should be weaker than near the top
        u_near_bot = get_field(tree, :u, level, n ÷ 2, 2)
        @test abs(u_near_bot) < abs(u_near_lid)
    end

    @testset "AMR cavity vs uniform" begin
        # Run both solvers at N=16
        Re = 100.0
        N = 16

        # Uniform grid solver
        u_uni, v_uni, p_uni, conv_uni = run_cavity(N=N, Re=Re, cfl=0.2,
                                                     max_steps=5000, tol=1e-6)

        # AMR solver (uniform refinement = same grid)
        tree, conv_amr = run_cavity_amr(N=N, Re=Re, cfl=0.2,
                                         max_steps=5000, tol=1e-6)

        # Compare u-velocity at interior points
        # Note: uniform grid uses node-based indexing (1:N), AMR uses cell-centered
        # The grids are slightly different (dx = 1/(N-1) vs h = 1/N), so we compare
        # qualitatively: extract center-line profiles and check L2 difference is bounded

        level = round(Int, log2(N))
        n_amr = 2^level

        # Extract u at mid-x line from AMR
        mid_i = n_amr ÷ 2
        u_amr_profile = [get_field(tree, :u, level, mid_i, j) for j in 1:n_amr]

        # Extract u at mid-x line from uniform (interior + boundary)
        mid_i_uni = N ÷ 2
        u_uni_profile = [u_uni[mid_i_uni, j] for j in 1:N]

        # Both should have similar shape: positive near top, negative/zero near bottom
        # The grids differ slightly so we just check L2 norm of difference is bounded
        # Interpolate: since N_amr = N, compare same-length vectors
        # But uniform has N points and AMR has N points, spacing differs slightly
        # Just check that both have similar max velocity
        max_u_amr = maximum(u_amr_profile)
        max_u_uni = maximum(u_uni_profile)

        # Both should have lid velocity ~1.0 at top
        @test max_u_amr ≈ 1.0 atol=0.01
        @test max_u_uni ≈ 1.0 atol=0.01

        # L2 difference of profiles (rough comparison)
        # Normalize by number of points
        n_compare = min(length(u_amr_profile), length(u_uni_profile))
        l2_diff = sqrt(sum((u_amr_profile[k] - u_uni_profile[k])^2 for k in 1:n_compare) / n_compare)
        @test l2_diff < 0.15
    end

end
