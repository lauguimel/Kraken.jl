using Test
using Kraken

@testset "AMR Operators" begin

    # Helper: build a uniform quadtree at a given level
    function make_uniform_tree(target_level::Int; L=1.0)
        tree = QuadTree(Float64, max(target_level, 4); L=L)
        for lev in 0:(target_level - 1)
            n = 2^lev
            for j in 1:n, i in 1:n
                if tree.active[lev+1][i, j]
                    refine!(tree, lev, i, j)
                end
            end
        end
        tree
    end

    @testset "Laplacian of cos(πx)cos(πy) — Neumann-compatible" begin
        # φ = cos(πx)cos(πy) has ∂φ/∂n = 0 at all boundaries x=0,1 and y=0,1
        # ∇²φ = -2π²cos(πx)cos(πy)
        tree = make_uniform_tree(3)
        add_field!(tree, :phi)
        add_field!(tree, :lap)
        initialize_field!(tree, :phi, (x, y) -> cos(π * x) * cos(π * y))

        laplacian_amr!(tree, :lap, :phi)

        max_err = 0.0
        foreach_leaf(tree) do level, i, j
            computed = get_field(tree, :lap, level, i, j)
            x, y = cell_center(tree, level, i, j)
            exact = -2π^2 * cos(π * x) * cos(π * y)
            max_err = max(max_err, abs(computed - exact))
        end
        # Second-order FV on 8x8 grid (h=0.125), error ~ O(h²) ~ 0.15
        @test max_err < 1.0
    end

    @testset "Laplacian convergence (uniform grids)" begin
        # φ = cos(πx)cos(πy), check that error decreases with refinement
        errors = Float64[]
        for target_level in 2:4
            tree = make_uniform_tree(target_level)
            add_field!(tree, :phi)
            add_field!(tree, :lap)
            initialize_field!(tree, :phi, (x, y) -> cos(π * x) * cos(π * y))

            laplacian_amr!(tree, :lap, :phi)

            max_err = 0.0
            foreach_leaf(tree) do level, i, j
                computed = get_field(tree, :lap, level, i, j)
                x, y = cell_center(tree, level, i, j)
                exact = -2π^2 * cos(π * x) * cos(π * y)
                max_err = max(max_err, abs(computed - exact))
            end
            push!(errors, max_err)
        end
        # Error should decrease with refinement
        @test errors[2] < errors[1]
        @test errors[3] < errors[2]
        # Convergence ratio should be > 2 (second-order → ratio ~4)
        ratio = errors[1] / errors[2]
        @test ratio > 2.0
    end

    @testset "Laplacian of quadratic — interior cells" begin
        # φ = x² + y², ∇²φ = 4. Only check interior cells (boundary cells
        # have Neumann ghost that doesn't match the actual gradient).
        tree = make_uniform_tree(3)  # 8x8
        add_field!(tree, :phi)
        add_field!(tree, :lap)
        initialize_field!(tree, :phi, (x, y) -> x^2 + y^2)

        laplacian_amr!(tree, :lap, :phi)

        max_err = 0.0
        n_interior = 0
        foreach_leaf(tree) do level, i, j
            n = 2^level
            if i > 1 && i < n && j > 1 && j < n
                val = get_field(tree, :lap, level, i, j)
                max_err = max(max_err, abs(val - 4.0))
                n_interior += 1
            end
        end
        @test n_interior > 0
        @test max_err < 1e-10
    end

    @testset "Laplacian conservation (Neumann)" begin
        # With Neumann BCs, integral of ∇²φ over domain should be 0
        tree = make_uniform_tree(2)
        # Refine one quadrant further for non-uniformity
        refine!(tree, 2, 1, 1)

        add_field!(tree, :phi)
        add_field!(tree, :lap)
        initialize_field!(tree, :phi, (x, y) -> sin(2π * x) * cos(2π * y))

        laplacian_amr!(tree, :lap, :phi)

        integral = 0.0
        foreach_leaf(tree) do level, i, j
            h = cell_size(tree, level)
            integral += get_field(tree, :lap, level, i, j) * h^2
        end
        @test abs(integral) < 1e-10
    end

    @testset "Divergence of linear field — interior cells" begin
        # u = x, v = y → ∇·(u,v) = 2 at interior cells
        tree = make_uniform_tree(3)  # 8x8

        add_field!(tree, :u)
        add_field!(tree, :v)
        add_field!(tree, :div)
        initialize_field!(tree, :u, (x, y) -> x)
        initialize_field!(tree, :v, (x, y) -> y)

        divergence_amr!(tree, :div, :u, :v)

        max_err = 0.0
        n_interior = 0
        foreach_leaf(tree) do level, i, j
            n = 2^level
            if i > 1 && i < n && j > 1 && j < n
                val = get_field(tree, :div, level, i, j)
                max_err = max(max_err, abs(val - 2.0))
                n_interior += 1
            end
        end
        @test n_interior > 0
        @test max_err < 1e-10
    end

    @testset "Divergence of constant field" begin
        # u = 3, v = 5 → ∇·(u,v) = 0 everywhere (including boundaries)
        tree = make_uniform_tree(2)
        refine!(tree, 2, 1, 1)  # non-uniform

        add_field!(tree, :u)
        add_field!(tree, :v)
        add_field!(tree, :div)
        initialize_field!(tree, :u, (x, y) -> 3.0)
        initialize_field!(tree, :v, (x, y) -> 5.0)

        divergence_amr!(tree, :div, :u, :v)

        max_err = 0.0
        foreach_leaf(tree) do level, i, j
            max_err = max(max_err, abs(get_field(tree, :div, level, i, j)))
        end
        @test max_err < 1e-10
    end

    @testset "Gradient of linear field — interior cells" begin
        # φ = 2x + 3y → ∇φ = (2, 3) at interior cells
        tree = make_uniform_tree(3)  # 8x8

        add_field!(tree, :phi)
        add_field!(tree, :gx)
        add_field!(tree, :gy)
        initialize_field!(tree, :phi, (x, y) -> 2x + 3y)

        gradient_amr!(tree, :gx, :gy, :phi)

        max_err_x = 0.0
        max_err_y = 0.0
        n_interior = 0
        foreach_leaf(tree) do level, i, j
            n = 2^level
            if i > 1 && i < n && j > 1 && j < n
                gx_val = get_field(tree, :gx, level, i, j)
                gy_val = get_field(tree, :gy, level, i, j)
                max_err_x = max(max_err_x, abs(gx_val - 2.0))
                max_err_y = max(max_err_y, abs(gy_val - 3.0))
                n_interior += 1
            end
        end
        @test n_interior > 0
        @test max_err_x < 1e-10
        @test max_err_y < 1e-10
    end

    @testset "Gradient on adapted mesh — interior" begin
        # φ = 2x + 3y on non-uniform mesh
        tree = make_uniform_tree(2)
        refine!(tree, 2, 2, 2)

        add_field!(tree, :phi)
        add_field!(tree, :gx)
        add_field!(tree, :gy)
        initialize_field!(tree, :phi, (x, y) -> 2x + 3y)

        gradient_amr!(tree, :gx, :gy, :phi)

        # Check cells that are not on domain boundary and not at fine-coarse interface
        # For a linear field, even fine-coarse should be exact if distances are correct
        max_err = 0.0
        foreach_leaf(tree) do level, i, j
            x, y = cell_center(tree, level, i, j)
            h = cell_size(tree, level)
            # Skip cells on domain boundary (Neumann affects gradient there)
            if x > h && x < 1.0 - h && y > h && y < 1.0 - h
                gx_val = get_field(tree, :gx, level, i, j)
                gy_val = get_field(tree, :gy, level, i, j)
                err = max(abs(gx_val - 2.0), abs(gy_val - 3.0))
                max_err = max(max_err, err)
            end
        end
        # First-order at fine-coarse interfaces: coarse cell center is misaligned
        # in the transverse direction, causing O(h) error for gradient
        @test max_err < 1.0
    end

    @testset "Advection of uniform field" begin
        # u·∇φ = 0 when φ is constant
        tree = make_uniform_tree(2)
        refine!(tree, 2, 1, 1)

        add_field!(tree, :u)
        add_field!(tree, :v)
        add_field!(tree, :phi)
        add_field!(tree, :adv)
        initialize_field!(tree, :u, (x, y) -> 1.0)
        initialize_field!(tree, :v, (x, y) -> 1.0)
        initialize_field!(tree, :phi, (x, y) -> 5.0)

        advect_amr!(tree, :adv, :u, :v, :phi)

        max_err = 0.0
        foreach_leaf(tree) do level, i, j
            max_err = max(max_err, abs(get_field(tree, :adv, level, i, j)))
        end
        @test max_err < 1e-10
    end

    @testset "Advection with zero velocity" begin
        # u=0, v=0 → advection = 0 for any φ
        tree = make_uniform_tree(2)

        add_field!(tree, :u)
        add_field!(tree, :v)
        add_field!(tree, :phi)
        add_field!(tree, :adv)
        initialize_field!(tree, :u, (x, y) -> 0.0)
        initialize_field!(tree, :v, (x, y) -> 0.0)
        initialize_field!(tree, :phi, (x, y) -> sin(π * x) * cos(π * y))

        advect_amr!(tree, :adv, :u, :v, :phi)

        max_err = 0.0
        foreach_leaf(tree) do level, i, j
            max_err = max(max_err, abs(get_field(tree, :adv, level, i, j)))
        end
        @test max_err < 1e-10
    end

end
