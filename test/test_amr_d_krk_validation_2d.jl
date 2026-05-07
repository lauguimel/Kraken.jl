using Test
using Kraken

function _test_amr_d_leaf_velocity_stats_2d(result; Fx=0.0)
    spec = result.spec
    F = result.F
    leaf_nx = Kraken._conservative_tree_level_size_2d(spec.Nx, spec.max_level)
    leaf_ny = Kraken._conservative_tree_level_size_2d(spec.Ny, spec.max_level)
    ux = Matrix{Float64}(undef, leaf_nx, leaf_ny)
    uy = Matrix{Float64}(undef, leaf_nx, leaf_ny)
    @inbounds for cell_id in spec.active_cells
        cell = spec.cells[cell_id]
        scale = 1 << (spec.max_level - cell.level)
        volume = cell.metrics.volume
        mass = zero(Float64)
        mx = zero(Float64)
        my = zero(Float64)
        for q in 1:9
            Fq = F[cell_id, q]
            mass += Fq
            mx += d2q9_cx(q) * Fq
            my += d2q9_cy(q) * Fq
        end
        rho = mass / volume
        fx = Kraken.conservative_tree_leaf_equivalent_force_2d(
            Fx, spec, cell.level)
        ux_cell = (mx / volume + fx / 2) / rho
        uy_cell = (my / volume) / rho
        for sj in 1:scale, si in 1:scale
            ux[(cell.i - 1) * scale + si, (cell.j - 1) * scale + sj] =
                ux_cell
            uy[(cell.i - 1) * scale + si, (cell.j - 1) * scale + sj] =
                uy_cell
        end
    end
    x_spread = maximum(maximum(@view ux[:, j]) - minimum(@view ux[:, j])
                       for j in 1:leaf_ny)
    return (;
        ux_min=minimum(ux),
        ux_max=maximum(ux),
        uy_linf=maximum(abs, uy),
        x_spread=x_spread)
end

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
    result_b = run_conservative_tree_amr_d_case_from_krk_2d(
        joinpath(convergence_dir, "bfs_scale1.krk"); steps_override=2)
    @test result_b.flow == :bfs_route_native
    @test result_b.steps == 2
    @test isfinite(result_b.mass_drift)

    @test cases["square_scale1.krk"].flow == :square
    @test cases["square_scale1.krk"].wall_model == :halfway_bounceback_mask
    @test cases["square_scale1.krk"].runtime_supported
    result_s = run_conservative_tree_amr_d_case_from_krk_2d(
        joinpath(convergence_dir, "square_scale1.krk"); steps_override=2)
    @test result_s.flow == :square_obstacle_route_native
    @test result_s.steps == 2
    @test isfinite(result_s.mass_drift)

    @test cases["cylinder_scale1.krk"].flow == :cylinder
    @test cases["cylinder_scale1.krk"].wall_model == :halfway_bounceback_mask
    @test cases["cylinder_scale1.krk"].runtime_supported
    result_y = run_conservative_tree_amr_d_case_from_krk_2d(
        joinpath(convergence_dir, "cylinder_scale1.krk"); steps_override=2)
    @test result_y.steps == 2
    @test isfinite(result_y.Cd)
    @test isfinite(result_y.mass_drift)

    nested_cylinder = cases["cylinder_nested4_probe.krk"]
    @test nested_cylinder.spec_supported
    @test nested_cylinder.max_level == 4
    @test nested_cylinder.runtime_supported
    @test nested_cylinder.runtime_status == :subcycled_nested_solid
    result_ny = run_conservative_tree_amr_d_case_from_krk_2d(
        joinpath(convergence_dir, "cylinder_nested4_probe.krk");
        steps_override=2)
    @test result_ny.flow == :cylinder_obstacle_subcycled
    @test result_ny.max_level == 4
    @test result_ny.steps == 2
    @test isfinite(result_ny.relative_mass_drift)

    nested_poiseuille = cases["poiseuille_nested4_channel.krk"]
    @test nested_poiseuille.max_level == 4
    @test nested_poiseuille.runtime_status == :subcycled_nested_channel
    result_p = run_conservative_tree_amr_d_case_from_krk_2d(
        joinpath(convergence_dir, "poiseuille_nested4_channel.krk");
        steps_override=2)
    @test result_p.max_level == 4
    @test result_p.steps == 2
    @test isfinite(result_p.relative_mass_drift)

    setup_limited = load_kraken(
        joinpath(convergence_dir, "poiseuille_nested4_channel.krk"))
    setup_limited.user_vars[:c2f_prolongation] = 1.0
    setup_limited.user_vars[:coarse_to_fine_predictor_weight] = 1.0
    result_limited = run_conservative_tree_amr_d_case_from_krk_2d(
        setup_limited; steps_override=2)
    @test result_limited.max_level == 4
    @test result_limited.steps == 2
    @test isfinite(result_limited.relative_mass_drift)

    setup_limited_bad_sampling = load_kraken(
        joinpath(convergence_dir, "poiseuille_nested4_channel.krk"))
    setup_limited_bad_sampling.user_vars[:c2f_prolongation] = 1.0
    setup_limited_bad_sampling.user_vars[:route_sampling] = 1.0
    @test_throws ArgumentError run_conservative_tree_amr_d_case_from_krk_2d(
        setup_limited_bad_sampling; steps_override=1)

    setup_bad_prolongation = load_kraken(
        joinpath(convergence_dir, "poiseuille_nested4_channel.krk"))
    setup_bad_prolongation.user_vars[:c2f_prolongation] = 2.0
    @test_throws ArgumentError run_conservative_tree_amr_d_case_from_krk_2d(
        setup_bad_prolongation; steps_override=1)

    setup_leaf_equiv = load_kraken(
        joinpath(convergence_dir, "poiseuille_nested4_channel.krk"))
    setup_leaf_equiv.user_vars[:route_sampling] = 0.0
    result_leaf_equiv = run_conservative_tree_amr_d_case_from_krk_2d(
        setup_leaf_equiv; steps_override=1)
    @test result_leaf_equiv.max_level == 4
    @test result_leaf_equiv.steps == 1
    @test isfinite(result_leaf_equiv.relative_mass_drift)

    setup_bad_sampling = load_kraken(
        joinpath(convergence_dir, "poiseuille_nested4_channel.krk"))
    setup_bad_sampling.user_vars[:route_sampling] = 3.0
    @test_throws ArgumentError run_conservative_tree_amr_d_case_from_krk_2d(
        setup_bad_sampling; steps_override=1)

    nested_couette = cases["couette_nested4_channel.krk"]
    @test nested_couette.max_level == 4
    @test nested_couette.runtime_status == :subcycled_nested_channel
    result_c = run_conservative_tree_amr_d_case_from_krk_2d(
        joinpath(convergence_dir, "couette_nested4_channel.krk");
        steps_override=2)
    @test result_c.max_level == 4
    @test result_c.steps == 2
    @test isfinite(result_c.relative_mass_drift)

    @test cases["poiseuille_xband_nested4_debug.krk"].max_level == 4
    @test cases["poiseuille_xband_nested4_debug.krk"].runtime_status ==
          :subcycled_nested_channel
    @test cases["poiseuille_yband_nested4_debug.krk"].max_level == 4
    @test cases["poiseuille_yband_nested4_debug.krk"].runtime_status ==
          :subcycled_nested_channel
    result_yband = run_conservative_tree_amr_d_case_from_krk_2d(
        joinpath(convergence_dir, "poiseuille_yband_nested4_debug.krk");
        steps_override=1)
    yband_stats = _test_amr_d_leaf_velocity_stats_2d(result_yband; Fx=1e-7)
    @test yband_stats.ux_min > 0
    @test yband_stats.uy_linf < 1e-12
    @test yband_stats.x_spread < 1e-12

    @test cases["poiseuille_yband_nested4_limited_debug.krk"].max_level == 4
    @test cases["poiseuille_yband_nested4_limited_debug.krk"].runtime_status ==
          :subcycled_nested_channel
    result_y_limited = run_conservative_tree_amr_d_case_from_krk_2d(
        joinpath(convergence_dir,
                 "poiseuille_yband_nested4_limited_debug.krk");
        steps_override=1)
    @test result_y_limited.max_level == 4
    @test result_y_limited.steps == 1
    @test isfinite(result_y_limited.relative_mass_drift)
    @test cases["poiseuille_wall_ybands_nested4_debug.krk"].max_level == 4
    @test cases["poiseuille_wall_ybands_nested4_debug.krk"].runtime_status ==
          :subcycled_nested_channel
    wall_ybands_spec = create_conservative_tree_spec_from_krk_2d(load_kraken(
        joinpath(convergence_dir,
                 "poiseuille_wall_ybands_nested4_debug.krk")))
    south_j = 1
    north_j = wall_ybands_spec.Ny << wall_ybands_spec.max_level
    @test any(wall_ybands_spec.cells[id].level ==
              wall_ybands_spec.max_level &&
              wall_ybands_spec.cells[id].j == south_j
              for id in wall_ybands_spec.active_cells)
    @test any(wall_ybands_spec.cells[id].level ==
              wall_ybands_spec.max_level &&
              wall_ybands_spec.cells[id].j == north_j
              for id in wall_ybands_spec.active_cells)
    @test cases["couette_yband_nested4_debug.krk"].max_level == 4
    @test cases["couette_yband_nested4_debug.krk"].runtime_status ==
          :subcycled_nested_channel
    couette_debug_spec = create_conservative_tree_spec_from_krk_2d(load_kraken(
        joinpath(convergence_dir, "couette_yband_nested4_debug.krk")))
    north_j = couette_debug_spec.Ny << couette_debug_spec.max_level
    @test any(couette_debug_spec.cells[id].level ==
              couette_debug_spec.max_level &&
              couette_debug_spec.cells[id].j == north_j
              for id in couette_debug_spec.active_cells)

    lift_probe = cases["cylinder_lift_nested4_probe.krk"]
    @test lift_probe.flow == :cylinder_lift
    @test lift_probe.max_level == 4
    @test lift_probe.runtime_supported == false
    @test lift_probe.runtime_status == :nested_obstacle_runtime_pending

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
    @test support[:halfway_bounceback_solid_mask].nested
    @test support[:ibb].wall_model == :unsupported
    @test support[:libb].wall_model == :unsupported
end
