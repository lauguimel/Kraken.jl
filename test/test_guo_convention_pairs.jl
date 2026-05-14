using Test
using Kraken

const GUO_PAIR_STEPS = 500
const GUO_PAIR_GX = 1e-5
const GUO_PAIR_GY = 0.0

function _guo_pair_initialize_lattice!(f)
    w = Kraken.weights(Kraken.D2Q9())
    @inbounds for q in 1:9
        f[:, :, q] .= w[q]
    end
    return f
end

function _guo_pair_standard_raw_halfstep(nx, ny, gx, gy, steps)
    f = zeros(Float64, nx, ny, 9)
    _guo_pair_initialize_lattice!(f)
    is_solid = falses(nx, ny)
    rho = ones(Float64, nx, ny)
    ux = zeros(Float64, nx, ny)
    uy = zeros(Float64, nx, ny)

    for _ in 1:steps
        # Production pair in run_poiseuille_2d at src/drivers/basic.jl:197-198.
        collide_guo_2d!(f, is_solid, 1.0, gx, gy)
    end
    compute_macroscopic_forced_2d!(rho, ux, uy, f, gx, gy)
    return sum(ux) / length(ux), sum(uy) / length(uy)
end

function _guo_pair_integrated_public(nx, ny, gx, gy, steps)
    F = zeros(Float64, nx, ny, 9)
    rho = ones(Float64, nx, ny)
    ux = zeros(Float64, nx, ny)
    uy = zeros(Float64, nx, ny)
    fill_equilibrium_integrated_D2Q9!(F, 1.0, 1.0, 0.0, 0.0)

    for _ in 1:steps
        # Integrated convention baseline used by test_amr_d_ladder.jl marche 3.
        collide_Guo_integrated_D2Q9!(F, 1.0, 1.0, gx, gy)
    end
    compute_macroscopic_2d!(rho, ux, uy, F; sync=true)
    return sum(ux) / length(ux), sum(uy) / length(uy)
end

function _guo_pair_composite_profile(nx, ny, gx, gy, steps)
    coarse = zeros(Float64, nx, ny, 9)
    patch = create_conservative_tree_patch_2d(9:24, 9:24; T=Float64)
    fill_equilibrium_integrated_D2Q9!(coarse, 1.0, 1.0, 0.0, 0.0)
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, 0.25, 1.0, 0.0, 0.0)

    for _ in 1:steps
        # Production pair in run_conservative_tree_poiseuille_macroflow_2d at
        # src/refinement/conservative_tree_2d.jl:1890-1900.
        collide_Guo_composite_F_2d!(coarse, patch, 1.0, 0.25, 1.0, 1.0, gx, gy)
    end
    profile = composite_leaf_mean_ux_profile(coarse, patch;
                                             volume_leaf=0.25, force_x=gx)
    return sum(profile) / length(profile), gy * steps
end

function _guo_pair_composite_velocity_field(nx, ny, gx, gy, steps)
    coarse = zeros(Float64, nx, ny, 9)
    patch = create_conservative_tree_patch_2d(9:24, 9:24; T=Float64)
    fill_equilibrium_integrated_D2Q9!(coarse, 1.0, 1.0, 0.0, 0.0)
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, 0.25, 1.0, 0.0, 0.0)

    for _ in 1:steps
        # Production pair feeding conservative_tree_velocity_gradient_patch_range_2d
        # at src/refinement/conservative_tree_streaming_2d.jl:1031-1032.
        collide_Guo_composite_F_2d!(coarse, patch, 1.0, 0.25, 1.0, 1.0, gx, gy)
    end
    velocity = composite_leaf_velocity_field_2d(coarse, patch;
                                                volume_leaf=0.25,
                                                force_x=gx, force_y=gy)
    return sum(velocity.ux) / length(velocity.ux),
           sum(velocity.uy) / length(velocity.uy)
end

function _guo_pair_leaf_solid_velocity(nx, ny, gx, gy, steps)
    F = zeros(Float64, nx, ny, 9)
    is_solid = falses(nx, ny)
    fill_equilibrium_integrated_D2Q9!(F, 1.0, 1.0, 0.0, 0.0)

    for _ in 1:steps
        # Production pair in _run_conservative_tree_periodic_solid_force_macroflow_2d
        # at src/refinement/conservative_tree_2d.jl:1943 and 1952-1953.
        collide_Guo_integrated_D2Q9!(F, is_solid, 1.0, 1.0, gx, gy)
    end
    return Kraken._leaf_fluid_mean_velocity_F(F, is_solid;
                                              volume=1.0,
                                              force_x=gx, force_y=gy)
end

function _guo_pair_leaf_solid_mean_ux(nx, ny, gx, gy, steps)
    F = zeros(Float64, nx, ny, 9)
    is_solid = falses(nx, ny)
    fill_equilibrium_integrated_D2Q9!(F, 1.0, 1.0, 0.0, 0.0)

    for _ in 1:steps
        # Production pair in run_conservative_tree_cylinder_macroflow_2d at
        # src/refinement/conservative_tree_2d.jl:2170 and 2187.
        collide_Guo_integrated_D2Q9!(F, is_solid, 1.0, 1.0, gx, gy)
    end
    return Kraken._leaf_fluid_mean_ux_F(F, is_solid; volume=1.0, force_x=gx),
           gy * steps
end

function _guo_pair_composite_solid_mean_ux(nx, ny, gx, gy, steps)
    coarse = zeros(Float64, nx, ny, 9)
    patch = create_conservative_tree_patch_2d(9:24, 9:24; T=Float64)
    topology = create_conservative_tree_topology_2d(nx, ny, patch)
    is_solid = falses(2 * nx, 2 * ny)
    fill_equilibrium_integrated_D2Q9!(coarse, 1.0, 1.0, 0.0, 0.0)
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, 0.25, 1.0, 0.0, 0.0)

    for _ in 1:steps
        # Production pair in run_conservative_tree_cylinder_obstacle_route_native_2d at
        # src/refinement/conservative_tree_streaming_2d.jl:1757 and 1779.
        Kraken.collide_Guo_composite_solid_F_2d!(
            coarse, patch, topology, is_solid, 1.0, 0.25, 1.0, 1.0, gx, gy)
    end
    leaf = zeros(Float64, 2 * nx, 2 * ny, 9)
    composite_to_leaf_F_2d!(leaf, coarse, patch)
    return Kraken._leaf_fluid_mean_ux_F(leaf, is_solid; volume=0.25, force_x=gx),
           gy * steps
end

function _guo_pair_subcycled_profile(nx, ny, gx, gy, steps)
    spec = create_conservative_tree_spec_2d(
        nx, ny, ConservativeTreeRefineBlock2D[])
    F = allocate_conservative_tree_F_2d(spec; T=Float64)
    initialize_conservative_tree_equilibrium_F_2d!(F, spec; rho=1.0)

    for _ in 1:steps
        # Production pair in run_conservative_tree_poiseuille_subcycled_2d at
        # src/refinement/conservative_tree_macroflows_subcycled_2d.jl:884-913.
        Kraken._collide_Guo_conservative_tree_active_ids_F_2d!(
            F, spec, spec.active_cells, 1.0, gx, gy)
    end
    profile = conservative_tree_leaf_mean_ux_profile_2d(
        F, spec; force_x=gx, level_scaled_force=true)
    return sum(profile) / length(profile), gy * steps
end

function _guo_pair_subcycled_solid_velocity(nx, ny, gx, gy, steps)
    spec = create_conservative_tree_spec_2d(
        nx, ny, ConservativeTreeRefineBlock2D[])
    F = allocate_conservative_tree_F_2d(spec; T=Float64)
    initialize_conservative_tree_equilibrium_F_2d!(F, spec; rho=1.0)
    is_solid = falses(nx, ny)

    for _ in 1:steps
        # Production collision path for subcycled solid macroflows; readout is
        # _subcycled_solid_macroflow_result_2d at
        # src/refinement/conservative_tree_macroflows_subcycled_2d.jl:639-641.
        Kraken._collide_Guo_conservative_tree_active_fluid_ids_F_2d!(
            F, spec, spec.active_cells, is_solid, 1.0, gx, gy)
    end
    return Kraken.conservative_tree_leaf_fluid_mean_velocity_2d(
        F, spec, is_solid; force_x=gx, force_y=gy, level_scaled_force=true)
end

function _test_guo_pair(name, runner, gx, gy, steps, tol)
    @testset "$name" begin
        mean_ux, mean_uy = runner(32, 32, gx, gy, steps)
        err_x = abs(mean_ux - gx * steps)
        err_y = abs(mean_uy - gy * steps)
        # Analytical truth: u_phys(N) = F*N on a periodic box from rest.
        @test err_x <= tol
        @test err_y <= tol
    end
end

@testset "Guo convention pairs (isothermal 2D)" begin
    gx = GUO_PAIR_GX
    gy = GUO_PAIR_GY
    steps = GUO_PAIR_STEPS
    # 50x machine-eps accumulated over N moment sums; safety factor covers
    # sum-of-moments noise over 32x32 cells.
    tol = 50 * eps(Float64) * steps

    _test_guo_pair("collide_guo_2d! + compute_macroscopic_forced_2d!",
                   _guo_pair_standard_raw_halfstep, gx, gy, steps, tol)
    _test_guo_pair("collide_Guo_integrated_D2Q9! + compute_macroscopic_2d!",
                   _guo_pair_integrated_public, gx, gy, steps, tol)
    _test_guo_pair("collide_Guo_composite_F_2d! + composite_leaf_mean_ux_profile",
                   _guo_pair_composite_profile, gx, gy, steps, tol)
    _test_guo_pair("collide_Guo_composite_F_2d! + composite_leaf_velocity_field_2d",
                   _guo_pair_composite_velocity_field, gx, gy, steps, tol)
    _test_guo_pair("collide_Guo_integrated_D2Q9! + _leaf_fluid_mean_velocity_F",
                   _guo_pair_leaf_solid_velocity, gx, gy, steps, tol)
    _test_guo_pair("collide_Guo_integrated_D2Q9! + _leaf_fluid_mean_ux_F",
                   _guo_pair_leaf_solid_mean_ux, gx, gy, steps, tol)
    _test_guo_pair("collide_Guo_composite_solid_F_2d! + _leaf_fluid_mean_ux_F",
                   _guo_pair_composite_solid_mean_ux, gx, gy, steps, tol)
    _test_guo_pair("_collide_Guo_conservative_tree_active_ids_F_2d! + conservative_tree_leaf_mean_ux_profile_2d",
                   _guo_pair_subcycled_profile, gx, gy, steps, tol)
    _test_guo_pair("_collide_Guo_conservative_tree_active_fluid_ids_F_2d! + conservative_tree_leaf_fluid_mean_velocity_2d",
                   _guo_pair_subcycled_solid_velocity, gx, gy, steps, tol)
end
