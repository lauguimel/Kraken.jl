using Test
using Kraken

@testset "AMR D .krk validation helpers" begin
    convergence_dir = joinpath(dirname(@__DIR__), "benchmarks", "krk",
                               "amr_d_convergence_2d")
    showoff_dir = joinpath(dirname(@__DIR__), "benchmarks", "krk",
                           "amr_d_showoff_2d")

    cases = Dict(
        basename(path) => conservative_tree_amr_d_case_from_krk_2d(path)
        for path in sort(filter(endswith(".krk"), readdir(convergence_dir; join=true))))

    @test cases["poiseuille_xband_scale1.krk"].flow == :poiseuille
    @test cases["poiseuille_xband_scale1.krk"].boundary_policy == :periodic_x_wall_y
    @test cases["poiseuille_xband_scale1.krk"].runtime_supported

    @test cases["couette_scale1.krk"].flow == :couette
    @test cases["couette_scale1.krk"].boundary_policy == :periodic_x_moving_wall_y
    @test cases["couette_scale1.krk"].runtime_supported

    @test cases["bfs_scale1.krk"].flow == :bfs
    @test cases["bfs_scale1.krk"].boundary_policy == :open_x_wall_y
    @test cases["bfs_scale1.krk"].runtime_supported

    @test cases["square_scale1.krk"].flow == :square
    @test cases["square_scale1.krk"].wall_model == :halfway_bounceback_mask
    @test cases["square_scale1.krk"].runtime_supported

    @test cases["cylinder_scale1.krk"].flow == :cylinder
    @test cases["cylinder_scale1.krk"].wall_model == :halfway_bounceback_mask
    @test cases["cylinder_scale1.krk"].runtime_supported

    nested_cylinder = cases["cylinder_nested4_probe.krk"]
    @test nested_cylinder.spec_supported
    @test nested_cylinder.max_level == 4
    @test nested_cylinder.runtime_supported == false
    @test nested_cylinder.runtime_status == :nested_obstacle_runtime_pending

    nested_poiseuille = cases["poiseuille_nested4_channel.krk"]
    @test nested_poiseuille.max_level == 4
    @test nested_poiseuille.runtime_status == :subcycled_nested_channel
    result_p = run_conservative_tree_amr_d_case_from_krk_2d(
        joinpath(convergence_dir, "poiseuille_nested4_channel.krk");
        steps_override=2)
    @test result_p.max_level == 4
    @test result_p.steps == 2
    @test isfinite(result_p.relative_mass_drift)

    nested_couette = cases["couette_nested4_channel.krk"]
    @test nested_couette.max_level == 4
    @test nested_couette.runtime_status == :subcycled_nested_channel
    result_c = run_conservative_tree_amr_d_case_from_krk_2d(
        joinpath(convergence_dir, "couette_nested4_channel.krk");
        steps_override=2)
    @test result_c.max_level == 4
    @test result_c.steps == 2
    @test isfinite(result_c.relative_mass_drift)

    showoff = conservative_tree_amr_d_case_from_krk_2d(
        joinpath(showoff_dir, "cylinder_lift_re100_long_channel.krk"))
    @test showoff.flow == :cylinder_lift
    @test showoff.boundary_policy == :open_x_wall_y
    @test showoff.max_level == 3
    @test :lift in showoff.diagnostics
    @test showoff.runtime_supported == false
    @test showoff.runtime_status == :nested_obstacle_runtime_pending

    support = Dict(row.feature => row for row in conservative_tree_amr_d_support_matrix_2d())
    @test support[:periodic_x_wall_y].nested
    @test support[:periodic_x_moving_wall_y].nested
    @test support[:open_x_wall_y].single_patch
    @test support[:open_x_wall_y].nested == false
    @test support[:ibb].wall_model == :unsupported
    @test support[:libb].wall_model == :unsupported
end
