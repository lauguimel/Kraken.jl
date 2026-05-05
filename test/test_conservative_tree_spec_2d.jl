using Test
using Kraken

@testset "Conservative tree nested spec 2D" begin
    @testset "programmatic nested tree preserves active volume" begin
        blocks = [
            ConservativeTreeRefineBlock2D("outer", 3:6, 2:5),
            ConservativeTreeRefineBlock2D("inner", 7:8, 5:6; parent="outer"),
        ]
        spec = create_conservative_tree_spec_2d(8, 6, blocks)

        @test spec.Nx == 8
        @test spec.Ny == 6
        @test spec.max_level == 2
        @test length(spec.cells) == 48 + 4 * 16 + 4 * 4
        @test length(spec.active_cells) == 48 - 16 + 64 - 4 + 16
        @test isapprox(active_volume(spec), 48.0; atol=1e-14, rtol=0)

        parent_id = conservative_tree_cell_id_2d(spec, 0, 3, 2)
        @test parent_id != 0
        @test !conservative_tree_is_active_leaf_2d(spec, parent_id)
        @test conservative_tree_children_2d(spec, parent_id) != (0, 0, 0, 0)

        inner_parent_id = conservative_tree_cell_id_2d(spec, 1, 7, 5)
        @test inner_parent_id != 0
        @test !conservative_tree_is_active_leaf_2d(spec, inner_parent_id)
        @test conservative_tree_children_2d(spec, inner_parent_id) != (0, 0, 0, 0)

        inner_leaf_id = conservative_tree_cell_id_2d(spec, 2, 13, 9)
        @test inner_leaf_id != 0
        @test conservative_tree_is_active_leaf_2d(spec, inner_leaf_id)
        @test spec.cells[inner_leaf_id].parent == inner_parent_id
    end

    @testset "four-level static nesting canary" begin
        blocks = [
            ConservativeTreeRefineBlock2D("L1", 17:48, 9:24),
            ConservativeTreeRefineBlock2D("L2", 49:80, 25:40; parent="L1"),
            ConservativeTreeRefineBlock2D("L3", 113:144, 57:72; parent="L2"),
            ConservativeTreeRefineBlock2D("L4", 241:272, 121:136; parent="L3"),
        ]
        spec = create_conservative_tree_spec_2d(64, 32, blocks)

        @test spec.max_level == 4
        @test isapprox(active_volume(spec), 64 * 32; atol=1e-12, rtol=0)
        @test spec.refine_level["L4"] == 4
        @test spec.refine_i_range["L4"] == 481:544
        @test spec.refine_j_range["L4"] == 241:272
    end

    @testset "invalid ownership and balance are rejected" begin
        @test_throws ArgumentError create_conservative_tree_spec_2d(8, 8, [
            ConservativeTreeRefineBlock2D("a", 2:4, 2:4),
            ConservativeTreeRefineBlock2D("b", 3:5, 3:5),
        ])

        @test_throws ArgumentError create_conservative_tree_spec_2d(8, 8, [
            ConservativeTreeRefineBlock2D("child", 3:4, 3:4; parent="missing"),
        ])

        @test_throws ArgumentError create_conservative_tree_spec_2d(8, 8, [
            ConservativeTreeRefineBlock2D("outer", 3:5, 3:5),
            ConservativeTreeRefineBlock2D("child", 1:2, 1:2; parent="outer"),
        ])

        unbalanced = [
            ConservativeTreeRefineBlock2D("outer", 3:4, 3:4),
            ConservativeTreeRefineBlock2D("edge_child", 5:5, 5:5; parent="outer"),
        ]
        @test_throws ArgumentError create_conservative_tree_spec_2d(8, 8, unbalanced)
        @test create_conservative_tree_spec_2d(
            8, 8, unbalanced; balance=2).max_level == 2
    end

    @testset ".krk nested refine blocks build a static spec" begin
        setup = parse_kraken("""
        Simulation nested_static D2Q9
        Domain L = 8 x 4  N = 16 x 8
        Physics nu = 0.1

        Refine outer { region = [2.0, 1.0, 6.0, 3.0], ratio = 2 }
        Refine inner { region = [3.0, 1.5, 4.0, 2.0], ratio = 2, parent = outer }

        Boundary west periodic
        Boundary east periodic
        Boundary south wall
        Boundary north wall
        Run 1 steps
        """)

        blocks = conservative_tree_refine_blocks_from_krk_2d(
            setup.domain, setup.refinements)
        @test blocks[1].name == "outer"
        @test blocks[1].i_range == 5:12
        @test blocks[1].j_range == 3:6
        @test blocks[2].name == "inner"
        @test blocks[2].parent == "outer"
        @test blocks[2].i_range == 13:16
        @test blocks[2].j_range == 7:8

        spec = create_conservative_tree_spec_from_krk_2d(setup)
        @test spec.max_level == 2
        @test isapprox(active_volume(spec), 16 * 8; atol=1e-12, rtol=0)
    end

    @testset "existing cylinder nested4 probe is a static DSL canary" begin
        setup = load_kraken("benchmarks/krk/amr_d_convergence_2d/cylinder_nested4_probe.krk")
        spec = create_conservative_tree_spec_from_krk_2d(setup)

        @test spec.max_level == 4
        @test length(spec.active_cells) == 8040
        @test isapprox(active_volume(spec), 24 * 14; atol=1e-12, rtol=0)
    end

    @testset ".krk static spec rejects bad nested contracts" begin
        missing_parent = parse_kraken("""
        Simulation missing_parent D2Q9
        Domain L = 1 x 1  N = 16 x 16
        Physics nu = 0.1
        Refine child { region = [0.25, 0.25, 0.5, 0.5], ratio = 2, parent = nope }
        Boundary west wall
        Boundary east wall
        Boundary south wall
        Boundary north wall
        Run 1 steps
        """)
        @test_throws ArgumentError create_conservative_tree_spec_from_krk_2d(missing_parent)

        child_outside = parse_kraken("""
        Simulation child_outside D2Q9
        Domain L = 1 x 1  N = 16 x 16
        Physics nu = 0.1
        Refine outer { region = [0.25, 0.25, 0.5, 0.5], ratio = 2 }
        Refine child { region = [0.75, 0.75, 0.875, 0.875], ratio = 2, parent = outer }
        Boundary west wall
        Boundary east wall
        Boundary south wall
        Boundary north wall
        Run 1 steps
        """)
        @test_throws ArgumentError create_conservative_tree_spec_from_krk_2d(child_outside)
    end
end
