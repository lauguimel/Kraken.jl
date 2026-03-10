using Test
using Kraken

@testset "Quadtree AMR" begin
    @testset "Construction and basic ops" begin
        tree = QuadTree(Float64, 6)
        @test nleaves(tree) == 1
        add_field!(tree, :phi)
        @test get_field(tree, :phi, 0, 1, 1) == 0.0
        # Cell center of root cell
        x, y = cell_center(tree, 0, 1, 1)
        @test x == 0.5
        @test y == 0.5
        # Cell size
        @test cell_size(tree, 0) == 1.0
        @test cell_size(tree, 1) == 0.5
        @test cell_size(tree, 2) == 0.25
    end

    @testset "Uniform refinement to level 2" begin
        tree = QuadTree(Float64, 6)
        add_field!(tree, :phi)
        # Refine root -> 4 cells at level 1
        refine!(tree, 0, 1, 1)
        @test nleaves(tree) == 4
        # Root is no longer active
        @test tree.active[1][1, 1] == false
        # All level 1 cells are active
        for i in 1:2, j in 1:2
            @test tree.active[2][i, j] == true
        end
        # Refine all level 1 -> 16 cells at level 2
        for i in 1:2, j in 1:2
            refine!(tree, 1, i, j)
        end
        @test nleaves(tree) == 16
    end

    @testset "Coarsening" begin
        tree = QuadTree(Float64, 6)
        add_field!(tree, :phi)
        refine!(tree, 0, 1, 1)
        @test nleaves(tree) == 4
        # Set different values in children
        set_field!(tree, :phi, 1, 1, 1, 1.0)
        set_field!(tree, :phi, 1, 2, 1, 2.0)
        set_field!(tree, :phi, 1, 1, 2, 3.0)
        set_field!(tree, :phi, 1, 2, 2, 4.0)
        coarsen!(tree, 1, 1, 1)
        @test nleaves(tree) == 1
        # Parent should have average = (1+2+3+4)/4 = 2.5
        @test get_field(tree, :phi, 0, 1, 1) == 2.5
    end

    @testset "Neighbor finding - same level" begin
        tree = QuadTree(Float64, 6)
        add_field!(tree, :phi)
        refine!(tree, 0, 1, 1)  # 4 cells at level 1
        # Cell (1,1,1) right neighbor should be (1,2,1)
        nb = find_neighbor(tree, 1, 1, 1, :right)
        @test nb == (1, 2, 1)
        # Cell (1,2,1) left neighbor should be (1,1,1)
        nb = find_neighbor(tree, 1, 2, 1, :left)
        @test nb == (1, 1, 1)
        # Cell (1,1,1) top neighbor should be (1,1,2)
        nb = find_neighbor(tree, 1, 1, 1, :top)
        @test nb == (1, 1, 2)
        # Cell (1,1,1) left neighbor should be boundary
        nb = find_neighbor(tree, 1, 1, 1, :left)
        @test nb === nothing
        # Cell (1,1,1) bottom neighbor should be boundary
        nb = find_neighbor(tree, 1, 1, 1, :bottom)
        @test nb === nothing
    end

    @testset "Neighbor finding - coarse neighbor" begin
        tree = QuadTree(Float64, 6)
        add_field!(tree, :phi)
        refine!(tree, 0, 1, 1)     # 4 cells at level 1
        refine!(tree, 1, 1, 1)     # refine bottom-left -> 4 cells at level 2
        # Cell (2,2,1) right neighbor: the level 1 cell (1,2,1) should be found
        nb = find_neighbor(tree, 2, 2, 1, :right)
        @test nb == (1, 2, 1)
    end

    @testset "Neighbor finding - fine neighbors" begin
        tree = QuadTree(Float64, 6)
        add_field!(tree, :phi)
        refine!(tree, 0, 1, 1)     # 4 cells at level 1
        refine!(tree, 1, 2, 1)     # refine bottom-right at level 1
        # Cell (1,1,1) right neighbor: should return fine cells at level 2
        nb = find_neighbor(tree, 1, 1, 1, :right)
        @test nb isa Vector
        @test length(nb) == 2
        # The fine cells should be on the left face of the refined neighbor
        for (nl, ni, nj) in nb
            @test nl == 2
            @test ni == 3  # left column of children of (1,2,1)
        end
    end

    @testset "2:1 balance" begin
        tree = QuadTree(Float64, 6)
        add_field!(tree, :phi)
        # Refine to level 3 in one corner
        refine!(tree, 0, 1, 1)
        refine!(tree, 1, 1, 1)
        refine!(tree, 2, 1, 1)
        enforce_balance!(tree)
        # Check no leaf has neighbor 2+ levels different
        balanced = true
        for (level, i, j) in tree.leaf_list
            for dir in [:left, :right, :bottom, :top]
                nb = find_neighbor(tree, level, i, j, dir)
                if nb === nothing
                    continue
                end
                if nb isa Tuple{Int,Int,Int}
                    nb_level = nb[1]
                    if abs(level - nb_level) > 1
                        balanced = false
                    end
                elseif nb isa Vector
                    for (nb_level, _, _) in nb
                        if abs(level - nb_level) > 1
                            balanced = false
                        end
                    end
                end
            end
        end
        @test balanced
    end

    @testset "Adaptive refinement" begin
        tree = QuadTree(Float64, 4)
        add_field!(tree, :phi)
        # Refine uniformly to level 2 first
        refine!(tree, 0, 1, 1)
        for i in 1:2, j in 1:2
            refine!(tree, 1, i, j)
        end
        # Set phi = step function (sharp gradient along x=0.5)
        for (level, i, j) in tree.leaf_list
            x, y = cell_center(tree, level, i, j)
            set_field!(tree, :phi, level, i, j, x > 0.5 ? 1.0 : 0.0)
        end
        # Adapt - should refine near x=0.5
        n_before = nleaves(tree)
        adapt!(tree; field=:phi, threshold=0.1, max_level=4)
        @test nleaves(tree) > n_before    # more cells near discontinuity
        @test nleaves(tree) < 16 * 16     # fewer than uniform level 4
    end

    @testset "Field initialization" begin
        tree = QuadTree(Float64, 3)
        add_field!(tree, :phi)
        # Refine to level 2 uniformly
        refine!(tree, 0, 1, 1)
        for i in 1:2, j in 1:2
            refine!(tree, 1, i, j)
        end
        initialize_field!(tree, :phi, (x, y) -> sin(pi * x) * sin(pi * y))
        # Check a known cell center
        val = get_field(tree, :phi, 2, 1, 1)
        x, y = cell_center(tree, 2, 1, 1)
        @test val ≈ sin(pi * x) * sin(pi * y)
    end

    @testset "foreach_leaf" begin
        tree = QuadTree(Float64, 4)
        add_field!(tree, :phi)
        refine!(tree, 0, 1, 1)
        count = 0
        foreach_leaf((l, i, j) -> (count += 1), tree)
        @test count == 4
    end

    @testset "Cell center correctness" begin
        tree = QuadTree(Float64, 4)
        # Level 2: 4x4 grid, cell (1,1) center should be at (0.125, 0.125)
        x, y = cell_center(tree, 2, 1, 1)
        @test x ≈ 0.125
        @test y ≈ 0.125
        # Cell (4,4) center at (0.875, 0.875)
        x, y = cell_center(tree, 2, 4, 4)
        @test x ≈ 0.875
        @test y ≈ 0.875
    end
end
