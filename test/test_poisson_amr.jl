using Test
using Kraken

@testset "AMR Poisson Solver" begin
    @testset "Uniform grid matches analytic (Neumann BC)" begin
        # Level 4 uniform (16x16) grid
        # Use cos(πx)cos(πy) which satisfies Neumann BCs (zero normal derivative)
        # ∇²[cos(πx)cos(πy)] = -2π²cos(πx)cos(πy)
        tree = QuadTree(Float64, 5)
        refine_uniformly!(tree, 4)
        @test nleaves(tree) == 16 * 16

        add_field!(tree, :phi; initial_value=0.0)
        add_field!(tree, :rhs)
        add_field!(tree, :res)

        initialize_field!(tree, :rhs, (x, y) -> -2π^2 * cos(π * x) * cos(π * y))

        converged, n_iters, final_r = solve_poisson_amr!(tree, :phi, :rhs;
            n_vcycles=20, n_smooth=4, rtol=1e-8, verbose=false)

        @test converged

        # Check L2 error vs exact = cos(πx)cos(πy) (has zero mean over [0,1]²)
        l2_err = 0.0
        l2_exact = 0.0
        for (level, i, j) in tree.leaf_list
            x, y = cell_center(tree, level, i, j)
            h = cell_size(tree, level)
            exact = cos(π * x) * cos(π * y)
            computed = get_field(tree, :phi, level, i, j)
            l2_err += (computed - exact)^2 * h^2
            l2_exact += exact^2 * h^2
        end
        rel_err = sqrt(l2_err / l2_exact)
        @test rel_err < 0.01  # < 1% relative error on 16×16
    end

    @testset "Coarser grid solves correctly" begin
        # 8x8 grid — coarser, larger error expected
        tree = QuadTree(Float64, 5)
        refine_uniformly!(tree, 3)
        @test nleaves(tree) == 8 * 8

        add_field!(tree, :phi; initial_value=0.0)
        add_field!(tree, :rhs)
        add_field!(tree, :res)

        initialize_field!(tree, :rhs, (x, y) -> -2π^2 * cos(π * x) * cos(π * y))

        converged, _, _ = solve_poisson_amr!(tree, :phi, :rhs;
            n_vcycles=20, n_smooth=4, rtol=1e-6, verbose=false)

        @test converged

        l2_err = 0.0
        l2_exact = 0.0
        for (level, i, j) in tree.leaf_list
            x, y = cell_center(tree, level, i, j)
            h = cell_size(tree, level)
            exact = cos(π * x) * cos(π * y)
            computed = get_field(tree, :phi, level, i, j)
            l2_err += (computed - exact)^2 * h^2
            l2_exact += exact^2 * h^2
        end
        @test sqrt(l2_err / l2_exact) < 0.05  # < 5% on 8×8
    end

    @testset "Residual decreases" begin
        tree = QuadTree(Float64, 4)
        refine_uniformly!(tree, 3)
        add_field!(tree, :phi; initial_value=0.0)
        add_field!(tree, :rhs)
        add_field!(tree, :res)
        initialize_field!(tree, :rhs, (x, y) -> -8.0 * π^2 * cos(2π * x) * cos(2π * y))

        # Compute initial residual norm
        compute_residual_all!(tree, :res, :phi, :rhs)
        r0 = residual_norm(tree, :res)

        vcycle_amr!(tree, :phi, :rhs, :res, 3)
        compute_residual_all!(tree, :res, :phi, :rhs)
        r1 = residual_norm(tree, :res)

        @test r1 < r0  # residual should decrease
    end

    @testset "refine_uniformly!" begin
        tree = QuadTree(Float64, 5)
        refine_uniformly!(tree, 3)
        @test nleaves(tree) == 8 * 8
        # All leaves should be at level 3
        for (l, _, _) in tree.leaf_list
            @test l == 3
        end
    end

    @testset "Residual norm zero for exact solution" begin
        tree = QuadTree(Float64, 4)
        refine_uniformly!(tree, 2)
        add_field!(tree, :phi; initial_value=0.0)
        add_field!(tree, :rhs; initial_value=0.0)
        add_field!(tree, :res; initial_value=0.0)

        compute_residual_all!(tree, :res, :phi, :rhs)
        @test residual_norm(tree, :res) ≈ 0.0 atol = 1e-14
    end

    @testset "Convergence rate" begin
        # V-cycle should reduce residual by >10x per cycle
        tree = QuadTree(Float64, 5)
        refine_uniformly!(tree, 4)
        add_field!(tree, :phi; initial_value=0.0)
        add_field!(tree, :rhs)
        add_field!(tree, :res)
        initialize_field!(tree, :rhs, (x, y) -> -2π^2 * cos(π * x) * cos(π * y))

        # Manually remove mean from rhs for compatibility
        Kraken._remove_mean!(tree, :rhs)

        compute_residual_all!(tree, :res, :phi, :rhs)
        r0 = residual_norm(tree, :res)

        vcycle_amr!(tree, :phi, :rhs, :res, 4)

        max_level = maximum(l for (l, _, _) in tree.leaf_list)
        Kraken._remove_mean_grid!(tree.fields[:phi][max_level+1], 2^max_level)

        compute_residual_all!(tree, :res, :phi, :rhs)
        r1 = residual_norm(tree, :res)

        @test r1 / r0 < 0.1
    end

    @testset "Higher frequency test" begin
        # Test with cos(2πx)cos(2πy), ∇² = -8π²cos(2πx)cos(2πy)
        tree = QuadTree(Float64, 5)
        refine_uniformly!(tree, 4)
        add_field!(tree, :phi; initial_value=0.0)
        add_field!(tree, :rhs)
        add_field!(tree, :res)
        initialize_field!(tree, :rhs, (x, y) -> -8π^2 * cos(2π * x) * cos(2π * y))

        converged, _, _ = solve_poisson_amr!(tree, :phi, :rhs;
            n_vcycles=20, n_smooth=4, rtol=1e-8, verbose=false)

        @test converged

        l2_err = 0.0
        l2_exact = 0.0
        for (level, i, j) in tree.leaf_list
            x, y = cell_center(tree, level, i, j)
            h = cell_size(tree, level)
            exact = cos(2π * x) * cos(2π * y)
            computed = get_field(tree, :phi, level, i, j)
            l2_err += (computed - exact)^2 * h^2
            l2_exact += exact^2 * h^2
        end
        @test sqrt(l2_err / l2_exact) < 0.05  # < 5% on 16×16 for 2nd mode
    end
end
