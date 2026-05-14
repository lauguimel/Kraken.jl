using Test
using Kraken

function _test_full_domain_nested_spec_2d(max_level::Integer)
    blocks = ConservativeTreeRefineBlock2D[]
    parent = ""
    for level in 1:Int(max_level)
        nx_level = 16 << (level - 1)
        ny_level = 12 << (level - 1)
        name = "L$(level)"
        push!(blocks, ConservativeTreeRefineBlock2D(
            name, 1:nx_level, 1:ny_level; parent=parent))
        parent = name
    end
    return create_conservative_tree_spec_2d(16, 12, blocks)
end

function _test_center_yband_nested_spec_2d()
    return create_conservative_tree_spec_2d(16, 12, [
        ConservativeTreeRefineBlock2D("C1", 1:16, 3:10),
        ConservativeTreeRefineBlock2D("C2", 1:32, 7:18; parent="C1"),
    ])
end

function _test_center_xband_nested_spec_2d()
    return create_conservative_tree_spec_2d(16, 12, [
        ConservativeTreeRefineBlock2D("X1", 5:12, 1:12),
        ConservativeTreeRefineBlock2D("X2", 11:22, 1:24; parent="X1"),
    ])
end

function _test_wall_refined_ybands_nested_spec_2d()
    return create_conservative_tree_spec_2d(16, 12, [
        ConservativeTreeRefineBlock2D("B1", 1:16, 1:5),
        ConservativeTreeRefineBlock2D("B2", 1:32, 1:8; parent="B1"),
        ConservativeTreeRefineBlock2D("T1", 1:16, 8:12),
        ConservativeTreeRefineBlock2D("T2", 1:32, 17:24; parent="T1"),
    ])
end

@inline _test_child_range_2d(r::UnitRange{Int}) =
    (2 * first(r) - 1):(2 * last(r))

@inline function _test_shrink_range_2d(r::UnitRange{Int};
                                       low::Bool=true,
                                       high::Bool=true,
                                       pad::Int=2)
    lo = first(r) + (low ? pad : 0)
    hi = last(r) - (high ? pad : 0)
    lo <= hi || throw(ArgumentError("shrunk range is empty"))
    return lo:hi
end

function _test_nested_band_spec_2d(kind::Symbol, max_level::Integer)
    ml = Int(max_level)
    1 <= ml <= 4 ||
        throw(ArgumentError("test nested band max_level must be in 1:4"))
    blocks = ConservativeTreeRefineBlock2D[]

    if kind == :xband
        ir = 5:12
        jr = 1:12
        parent = ""
        for level in 1:ml
            name = "X$(level)"
            push!(blocks, ConservativeTreeRefineBlock2D(
                name, ir, jr; parent=parent))
            parent = name
            ir = _test_shrink_range_2d(_test_child_range_2d(ir))
            jr = _test_child_range_2d(jr)
        end
    elseif kind == :yband
        ir = 1:16
        jr = 3:10
        parent = ""
        for level in 1:ml
            name = "Y$(level)"
            push!(blocks, ConservativeTreeRefineBlock2D(
                name, ir, jr; parent=parent))
            parent = name
            ir = _test_child_range_2d(ir)
            jr = _test_shrink_range_2d(_test_child_range_2d(jr))
        end
    elseif kind == :wall_ybands
        bottom_i = 1:16
        bottom_j = 1:5
        top_i = 1:16
        top_j = 8:12
        bottom_parent = ""
        top_parent = ""
        for level in 1:ml
            bottom_name = "B$(level)"
            top_name = "T$(level)"
            push!(blocks, ConservativeTreeRefineBlock2D(
                bottom_name, bottom_i, bottom_j; parent=bottom_parent))
            push!(blocks, ConservativeTreeRefineBlock2D(
                top_name, top_i, top_j; parent=top_parent))
            bottom_parent = bottom_name
            top_parent = top_name
            bottom_i = _test_child_range_2d(bottom_i)
            bottom_j = _test_shrink_range_2d(
                _test_child_range_2d(bottom_j); low=false, high=true)
            top_i = _test_child_range_2d(top_i)
            top_j = _test_shrink_range_2d(
                _test_child_range_2d(top_j); low=true, high=false)
        end
    else
        throw(ArgumentError("unknown nested band kind $kind"))
    end

    return create_conservative_tree_spec_2d(16, 12, blocks)
end

function _test_internal_xband_nested_spec_2d(max_level::Integer)
    ml = Int(max_level)
    1 <= ml <= 4 ||
        throw(ArgumentError("test nested band max_level must be in 1:4"))
    blocks = ConservativeTreeRefineBlock2D[]
    ir = 5:12
    jr = 3:10
    parent = ""
    for level in 1:ml
        name = "XI$(level)"
        push!(blocks, ConservativeTreeRefineBlock2D(
            name, ir, jr; parent=parent))
        parent = name
        ir = _test_shrink_range_2d(_test_child_range_2d(ir))
        jr = _test_shrink_range_2d(_test_child_range_2d(jr))
    end
    return create_conservative_tree_spec_2d(16, 12, blocks)
end

function _test_wall_closed_xband_nested_spec_2d(max_level::Integer)
    blocks = Kraken.conservative_tree_wall_closed_xband_refine_blocks_2d(
        "XWC", 16, 12, 5:12, Int(max_level))
    return create_conservative_tree_spec_2d(16, 12, blocks)
end

function _test_wall_touch_xband_one_level_spec_2d()
    return create_conservative_tree_spec_2d(6, 4, [
        ConservativeTreeRefineBlock2D("X", 3:4, 1:4),
    ])
end

function _test_wall_phase_three_level_spec_2d()
    return create_conservative_tree_spec_2d(8, 4, [
        ConservativeTreeRefineBlock2D("L1", 4:5, 1:4),
        ConservativeTreeRefineBlock2D("L2", 8:9, 1:8; parent="L1"),
    ])
end

function _test_wall_phase_xband_max2_spec_2d()
    return create_conservative_tree_spec_2d(16, 12, [
        ConservativeTreeRefineBlock2D("X1", 5:12, 1:12),
        ConservativeTreeRefineBlock2D("X2", 11:22, 1:24; parent="X1"),
    ])
end

function _test_wall_phase_bulk_touch_max_level_2_spec_2d()
    return create_conservative_tree_spec_2d(8, 8, [
        ConservativeTreeRefineBlock2D("L1", 4:5, 3:6),
        ConservativeTreeRefineBlock2D("L2", 8:9, 7:10; parent="L1"),
    ])
end

function _test_affine_operator_one_level_spec_2d()
    return create_conservative_tree_spec_2d(12, 8, [
        ConservativeTreeRefineBlock2D("L1", 6:7, 3:6),
    ])
end

function _test_affine_operator_two_level_spec_2d()
    return create_conservative_tree_spec_2d(12, 8, [
        ConservativeTreeRefineBlock2D("L1", 5:8, 3:6),
        ConservativeTreeRefineBlock2D("L2", 12:13, 6:11; parent="L1"),
    ])
end

function _test_cartesian_poiseuille_profile_2d(max_level::Integer,
                                               steps::Integer;
                                               Fx=1e-7,
                                               omega=1.0,
                                               rho0=1.0)
    scale = 1 << Int(max_level)
    nx = 16 * scale
    ny = 12 * scale
    volume = 1.0 / (scale * scale)
    F = zeros(Float64, nx, ny, 9)
    Ftmp = similar(F)
    fill_equilibrium_integrated_D2Q9!(F, volume, rho0, 0.0, 0.0)

    for _ in 1:(Int(steps) * scale)
        collide_Guo_integrated_D2Q9!(F, volume, omega, Fx, 0.0)
        stream_periodic_x_wall_y_F_2d!(Ftmp, F)
        F, Ftmp = Ftmp, F
    end

    profile = zeros(Float64, ny)
    for j in 1:ny
        ux_sum = 0.0
        for i in 1:nx
            cell = @view F[i, j, :]
            rho = mass_F(cell) / volume
            mx = momentum_F(cell)[1]
            ux_sum += (mx / volume) / rho
        end
        profile[j] = ux_sum / nx
    end
    return profile
end

function _test_cartesian_couette_profile_2d(max_level::Integer,
                                            steps::Integer;
                                            U=1e-4,
                                            omega=1.0,
                                            rho0=1.0)
    scale = 1 << Int(max_level)
    nx = 16 * scale
    ny = 12 * scale
    volume = 1.0 / (scale * scale)
    F = zeros(Float64, nx, ny, 9)
    Ftmp = similar(F)
    fill_equilibrium_integrated_D2Q9!(F, volume, rho0, 0.0, 0.0)

    for _ in 1:(Int(steps) * scale)
        collide_BGK_integrated_D2Q9!(F, volume, omega)
        stream_periodic_x_moving_wall_y_F_2d!(
            Ftmp, F; u_south=0.0, u_north=U, rho_wall=rho0,
            volume=volume)
        F, Ftmp = Ftmp, F
    end

    profile = zeros(Float64, ny)
    for j in 1:ny
        ux_sum = 0.0
        for i in 1:nx
            cell = @view F[i, j, :]
            rho = mass_F(cell) / volume
            mx = momentum_F(cell)[1]
            ux_sum += (mx / volume) / rho
        end
        profile[j] = ux_sum / nx
    end
    return profile
end

function _test_profile_linf_2d(a, b)
    length(a) == length(b) ||
        throw(ArgumentError("profile lengths differ"))
    return maximum(abs(Float64(a[i]) - Float64(b[i]))
                   for i in eachindex(a, b))
end

function _test_active_field_bounds_2d(result; force_x=0.0,
                                      level_scaled_force::Bool=false)
    spec = result.spec
    F = result.F
    rho_min = Inf
    rho_max = -Inf
    ux_min = Inf
    ux_max = -Inf
    @inbounds for cell_id in spec.active_cells
        cell = spec.cells[cell_id]
        volume = Float64(cell.metrics.volume)
        mass = 0.0
        mx = 0.0
        for q in 1:9
            fq = Float64(F[cell_id, q])
            mass += fq
            mx += Float64(Kraken.d2q9_cx(q)) * fq
        end
        rho = mass / volume
        fx = level_scaled_force ?
             Float64(Kraken.conservative_tree_leaf_equivalent_force_2d(
                force_x, spec, cell.level)) :
             Float64(force_x)
        ux = (mx / volume + fx / 2) / rho
        rho_min = min(rho_min, rho)
        rho_max = max(rho_max, rho)
        ux_min = min(ux_min, ux)
        ux_max = max(ux_max, ux)
    end
    return (; rho_min, rho_max, ux_min, ux_max)
end

function _test_fill_uniform_tree_equilibrium_2d!(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D;
        rho=1.0,
        ux=0.0,
        uy=0.0)
    fill!(F, zero(eltype(F)))
    @inbounds for cell_id in spec.active_cells
        volume = spec.cells[cell_id].metrics.volume
        for q in 1:9
            F[cell_id, q] = volume * equilibrium(D2Q9(), rho, ux, uy, q)
        end
    end
    return F
end

function _test_fill_diagonal_x_odd_tree_perturbation_2d!(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D;
        epsilon=1e-6)
    fill!(F, zero(eltype(F)))
    @inbounds for cell_id in spec.active_cells
        volume = spec.cells[cell_id].metrics.volume
        packet = epsilon * volume
        F[cell_id, 6] = packet
        F[cell_id, 9] = packet
        F[cell_id, 7] = -packet
        F[cell_id, 8] = -packet
    end
    return F
end

function _test_leaf_rho_level_boundary_metrics_2d(
        spec::ConservativeTreeSpec2D,
        F::AbstractMatrix;
        rho0=1.0,
        margin=2)
    leaf_nx = spec.Nx << spec.max_level
    leaf_ny = spec.Ny << spec.max_level
    rho = fill(NaN, leaf_nx, leaf_ny)
    level = fill(-1, leaf_nx, leaf_ny)
    @inbounds for cell_id in spec.active_cells
        cell = spec.cells[cell_id]
        scale = 1 << (spec.max_level - cell.level)
        mass = zero(eltype(F))
        for q in 1:9
            mass += F[cell_id, q]
        end
        cell_rho = mass / cell.metrics.volume
        i0 = (cell.i - 1) * scale + 1
        i1 = cell.i * scale
        j0 = (cell.j - 1) * scale + 1
        j1 = cell.j * scale
        rho[i0:i1, j0:j1] .= cell_rho
        level[i0:i1, j0:j1] .= cell.level
    end

    jlo = 1 + margin
    jhi = leaf_ny - margin
    max_x_jump = 0.0
    max_y_jump = 0.0
    max_abs_dev = 0.0
    x_count = 0
    y_count = 0
    @inbounds for i in 1:(leaf_nx - 1), j in jlo:jhi
        level[i, j] == level[i + 1, j] && continue
        x_count += 1
        a = Float64(rho[i, j])
        b = Float64(rho[i + 1, j])
        max_x_jump = max(max_x_jump, abs(b - a))
        max_abs_dev = max(max_abs_dev, abs(a - rho0), abs(b - rho0))
    end
    @inbounds for i in 1:leaf_nx, j in jlo:(jhi - 1)
        level[i, j] == level[i, j + 1] && continue
        y_count += 1
        a = Float64(rho[i, j])
        b = Float64(rho[i, j + 1])
        max_y_jump = max(max_y_jump, abs(b - a))
        max_abs_dev = max(max_abs_dev, abs(a - rho0), abs(b - rho0))
    end
    return (;
        x_count, y_count, max_x_jump, max_y_jump, max_abs_dev)
end

@inline _test_affine_population_value_2d(x, y) =
    1.0 + 0.2 * x - 0.13 * y

function _test_cell_leaf_bounds_2d(spec::ConservativeTreeSpec2D,
                                   cell_id::Int)
    cell = spec.cells[cell_id]
    scale = 1 << (spec.max_level - cell.level)
    i0 = (cell.i - 1) * scale + 1
    i1 = cell.i * scale
    j0 = (cell.j - 1) * scale + 1
    j1 = cell.j * scale
    return i0, i1, j0, j1
end

function _test_fill_single_q_affine_tree_2d!(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        q::Int)
    fill!(F, zero(eltype(F)))
    leaf_nx = spec.Nx << spec.max_level
    leaf_ny = spec.Ny << spec.max_level
    leaf_scale = 1 << spec.max_level
    leaf_volume = 1.0 / Float64(leaf_scale * leaf_scale)
    @inbounds for cell_id in spec.active_cells
        i0, i1, j0, j1 = _test_cell_leaf_bounds_2d(spec, cell_id)
        for jj in j0:j1, ii in i0:i1
            x = (Float64(ii) - 0.5) / leaf_nx
            y = (Float64(jj) - 0.5) / leaf_ny
            F[cell_id, q] += leaf_volume *
                             _test_affine_population_value_2d(x, y)
        end
    end
    return F
end

function _test_affine_expected_after_leaf_stream_2d(
        spec::ConservativeTreeSpec2D,
        cell_id::Int,
        q::Int)
    leaf_nx = spec.Nx << spec.max_level
    leaf_ny = spec.Ny << spec.max_level
    leaf_scale = 1 << spec.max_level
    leaf_volume = 1.0 / Float64(leaf_scale * leaf_scale)
    shift_i = leaf_scale * d2q9_cx(q)
    shift_j = leaf_scale * d2q9_cy(q)
    i0, i1, j0, j1 = _test_cell_leaf_bounds_2d(spec, cell_id)

    total = 0.0
    @inbounds for jj in j0:j1, ii in i0:i1
        dep_i = mod1(ii - shift_i, leaf_nx)
        dep_j = jj - shift_j
        1 <= dep_j <= leaf_ny || return nothing
        x = (Float64(dep_i) - 0.5) / leaf_nx
        y = (Float64(dep_j) - 0.5) / leaf_ny
        total += leaf_volume * _test_affine_population_value_2d(x, y)
    end
    return total
end

function _test_leaf_level_map_2d(spec::ConservativeTreeSpec2D)
    leaf_nx = spec.Nx << spec.max_level
    leaf_ny = spec.Ny << spec.max_level
    levels = fill(-1, leaf_nx, leaf_ny)
    @inbounds for cell_id in spec.active_cells
        cell = spec.cells[cell_id]
        i0, i1, j0, j1 = _test_cell_leaf_bounds_2d(spec, cell_id)
        levels[i0:i1, j0:j1] .= cell.level
    end
    return levels
end

function _test_touches_level_jump_2d(levels::AbstractMatrix{Int},
                                     i0::Int, i1::Int,
                                     j0::Int, j1::Int)
    nx, ny = size(levels)
    own = levels[i0, j0]
    @inbounds for j in max(1, j0 - 1):min(ny, j1 + 1),
                  i in max(1, i0 - 1):min(nx, i1 + 1)
        levels[i, j] == own || return true
    end
    return false
end

function _test_subcycled_affine_operator_metrics_2d(
        spec::ConservativeTreeSpec2D,
        q::Int,
        route_sampling::Symbol)
    table = create_conservative_tree_route_table_2d(
        spec; periodic_x=true, sampling=route_sampling)
    Fin = allocate_conservative_tree_F_2d(spec; T=Float64)
    Fout = similar(Fin)
    _test_fill_single_q_affine_tree_2d!(Fin, spec, q)
    Kraken.stream_conservative_tree_subcycled_buffered_routes_F_2d!(
        Fout, Fin, spec, table; boundary=:periodic_x_wall_y,
        interface_time_scaling=route_sampling == :level_native ?
            :level_native : :leaf_equivalent,
        coarse_to_fine_predictor_weight=0,
        interface_balance=false)

    levels = _test_leaf_level_map_2d(spec)
    max_l0_bulk = 0.0
    max_refined_or_interface = 0.0
    max_all = 0.0
    l0_bulk_count = 0
    refined_or_interface_count = 0
    @inbounds for cell_id in spec.active_cells
        expected = _test_affine_expected_after_leaf_stream_2d(
            spec, cell_id, q)
        expected === nothing && continue
        err = abs(Float64(Fout[cell_id, q]) - expected)
        max_all = max(max_all, err)
        i0, i1, j0, j1 = _test_cell_leaf_bounds_2d(spec, cell_id)
        near_interface = _test_touches_level_jump_2d(
            levels, i0, i1, j0, j1)
        if spec.cells[cell_id].level == 0 && !near_interface
            l0_bulk_count += 1
            max_l0_bulk = max(max_l0_bulk, err)
        else
            refined_or_interface_count += 1
            max_refined_or_interface = max(max_refined_or_interface, err)
        end
    end
    return (; max_all, max_l0_bulk, max_refined_or_interface,
            l0_bulk_count, refined_or_interface_count)
end

function _test_subcycled_rest_maxdiff_2d(spec, route_sampling::Symbol)
    table = create_conservative_tree_route_table_2d(
        spec; periodic_x=true, sampling=route_sampling)
    F = allocate_conservative_tree_F_2d(spec)
    G = similar(F)
    initialize_conservative_tree_equilibrium_F_2d!(F, spec; rho=1.0)
    Kraken.stream_conservative_tree_subcycled_buffered_routes_F_2d!(
        G, F, spec, table; boundary=:periodic_x_wall_y,
        interface_time_scaling=route_sampling == :level_native ?
            :level_native : :leaf_equivalent)
    return maximum(abs.(G[spec.active_cells, :] .-
                        F[spec.active_cells, :]))
end

function _test_subcycled_rest_prestream_maxdiff_2d(
        spec::ConservativeTreeSpec2D;
        mode::Symbol,
        omega=1.0)
    table = create_conservative_tree_route_table_2d(
        spec; periodic_x=true, sampling=:level_native)
    F = allocate_conservative_tree_F_2d(spec; T=Float64)
    G = similar(F)
    initialize_conservative_tree_equilibrium_F_2d!(F, spec; rho=1.0)
    schedule = Kraken.create_conservative_tree_subcycle_schedule_2d(
        spec.max_level)
    route_bank = Kraken.create_conservative_tree_subcycle_spatial_ledger_bank_2d(
        spec; schedule=schedule, T=Float64)
    state_bank = Kraken.create_conservative_tree_subcycle_buffer_bank_2d(
        spec; schedule=schedule, T=Float64)
    active_ids_by_level = state_bank.active_ids_by_level
    Fsource = similar(F)
    Fscratch = similar(F)
    pre_stream_level! = if mode == :noop
        (Flevel, local_spec, level, event) -> nothing
    elseif mode == :bgk
        (Flevel, local_spec, level, event) ->
            Kraken._collide_BGK_conservative_tree_active_ids_F_2d!(
                Flevel, local_spec, active_ids_by_level[level + 1],
                Kraken.conservative_tree_leaf_equivalent_omega_2d(
                    omega, local_spec, level))
    else
        throw(ArgumentError("mode must be :noop or :bgk"))
    end
    Kraken.stream_conservative_tree_subcycled_buffered_routes_F_2d!(
        G, F, spec, table; boundary=:periodic_x_wall_y,
        interface_time_scaling=:level_native,
        coarse_to_fine_predictor_weight=0,
        pre_stream_level! = pre_stream_level!,
        schedule=schedule, route_bank=route_bank, state_bank=state_bank,
        Fsource=Fsource, Fscratch=Fscratch)
    return maximum(abs.(G[spec.active_cells, :] .-
                        F[spec.active_cells, :]))
end

function _test_subcycled_unit_packet_level_native_2d(
        spec::ConservativeTreeSpec2D,
        src_id::Int,
        q::Int)
    table = create_conservative_tree_route_table_2d(
        spec; periodic_x=true, sampling=:level_native)
    F = allocate_conservative_tree_F_2d(spec; T=Float64)
    G = similar(F)
    fill!(F, 0.0)
    fill!(G, 0.0)
    F[src_id, q] = 1.0
    Kraken.stream_conservative_tree_subcycled_buffered_routes_F_2d!(
        G, F, spec, table; boundary=:periodic_x_wall_y,
        interface_time_scaling=:level_native,
        coarse_to_fine_predictor_weight=0)
    return G
end

function _test_subcycled_rest_row_maxdiffs_2d(spec, route_sampling::Symbol)
    table = create_conservative_tree_route_table_2d(
        spec; periodic_x=true, sampling=route_sampling)
    F = allocate_conservative_tree_F_2d(spec)
    G = similar(F)
    initialize_conservative_tree_equilibrium_F_2d!(F, spec; rho=1.0)
    Kraken.stream_conservative_tree_subcycled_buffered_routes_F_2d!(
        G, F, spec, table; boundary=:periodic_x_wall_y,
        interface_time_scaling=route_sampling == :level_native ?
            :level_native : :leaf_equivalent)

    rows = Dict{Tuple{Int,Int},Float64}()
    @inbounds for cell_id in spec.active_cells
        cell = spec.cells[cell_id]
        key = (cell.level, cell.j)
        rowdiff = get(rows, key, 0.0)
        for q in 1:9
            rowdiff = max(rowdiff, abs(Float64(G[cell_id, q] - F[cell_id, q])))
        end
        rows[key] = rowdiff
    end
    return rows
end

@testset "Conservative tree subcycling ledger 2D" begin
    @testset "generic recursive schedule is level agnostic" begin
        schedule = Kraken.create_conservative_tree_subcycle_schedule_2d(3)

        @test schedule.max_level == 3
        @test schedule.ratio == 2
        @test schedule.finest_ticks == 8
        @test schedule.level_step_ticks == [8, 4, 2, 1]
        @test Kraken.conservative_tree_subcycle_advance_counts_2d(schedule) ==
              [1, 2, 4, 8]

        sync_counts = Kraken.conservative_tree_subcycle_sync_counts_2d(schedule)
        for level in 0:2
            expected = 2^level
            @test sync_counts[(:sync_down, level, level + 1)] == expected
            @test sync_counts[(:sync_up, level + 1, level)] == expected
        end

        tick0 = Kraken.conservative_tree_subcycle_events_at_tick_2d(schedule, 0)
        @test [(event.phase, event.src_level, event.dst_level)
               for event in tick0] == [(:sync_down, 0, 1),
                                       (:sync_down, 1, 2),
                                       (:sync_down, 2, 3)]

        tick2 = Kraken.conservative_tree_subcycle_events_at_tick_2d(schedule, 2)
        @test (:advance, 3, 3) in
              [(event.phase, event.src_level, event.dst_level)
               for event in tick2]
        @test (:sync_up, 3, 2) in
              [(event.phase, event.src_level, event.dst_level)
               for event in tick2]
        @test (:advance, 2, 2) in
              [(event.phase, event.src_level, event.dst_level)
               for event in tick2]
        @test (:sync_down, 2, 3) in
              [(event.phase, event.src_level, event.dst_level)
               for event in tick2]

        tick8 = Kraken.conservative_tree_subcycle_events_at_tick_2d(schedule, 8)
        @test [(event.phase, event.src_level, event.dst_level)
               for event in tick8][end-1:end] == [(:sync_up, 1, 0),
                                                  (:advance, 0, 0)]
    end

    @testset "schedule contracts reject invalid inputs" begin
        @test Kraken.create_conservative_tree_subcycle_schedule_2d(0).events ==
              [Kraken.ConservativeTreeSubcycleEvent2D(1, :advance, 0, 0)]
        @test_throws ArgumentError Kraken.create_conservative_tree_subcycle_schedule_2d(-1)
        @test_throws ArgumentError Kraken.create_conservative_tree_subcycle_schedule_2d(2; ratio=1)
        schedule = Kraken.create_conservative_tree_subcycle_schedule_2d(2)
        @test_throws ArgumentError Kraken.conservative_tree_subcycle_events_at_tick_2d(
            schedule, 5)
    end

    @testset "subcycle state buffers keep algorithm roles disjoint" begin
        spec = create_conservative_tree_spec_2d(8, 6, [
            ConservativeTreeRefineBlock2D("outer", 3:6, 2:5),
            ConservativeTreeRefineBlock2D("inner", 7:8, 5:6; parent="outer"),
        ])
        bank = Kraken.create_conservative_tree_subcycle_buffer_bank_2d(spec)
        F = allocate_conservative_tree_F_2d(spec)
        for cell_id in spec.active_cells, q in 1:9
            F[cell_id, q] = 10 * spec.cells[cell_id].level + cell_id + q / 10
        end

        Kraken.conservative_tree_subcycle_store_active_owned_2d!(bank, F)
        level1_id = first(id for id in spec.active_cells
                          if spec.cells[id].level == 1)
        buffers = bank.levels[2]
        owned_before = buffers.owned[level1_id, 2]
        buffers.ghost_from_coarse[level1_id, 2] = 7.0
        buffers.reflux_to_coarse[level1_id, 2] = 3.0

        @test buffers.owned[level1_id, 2] == owned_before
        Kraken.conservative_tree_subcycle_apply_reflux_to_owned_level_2d!(
            bank, 1)
        @test buffers.owned[level1_id, 2] == owned_before + 3.0
        @test buffers.reflux_to_coarse[level1_id, 2] == 0.0
        @test buffers.ghost_from_coarse[level1_id, 2] == 7.0

        Frestored = allocate_conservative_tree_F_2d(spec)
        Kraken.conservative_tree_subcycle_restore_owned_level_2d!(
            Frestored, bank, 1)
        @test Frestored[level1_id, 2] == owned_before + 3.0
        level0_id = first(id for id in spec.active_cells
                          if spec.cells[id].level == 0)
        @test Frestored[level0_id, 2] == 0.0
    end

    @testset "subcycle restriction is conservative bottom-up" begin
        spec = create_conservative_tree_spec_2d(8, 6, [
            ConservativeTreeRefineBlock2D("outer", 3:6, 2:5),
            ConservativeTreeRefineBlock2D("inner", 7:8, 5:6; parent="outer"),
        ])
        bank = Kraken.create_conservative_tree_subcycle_buffer_bank_2d(spec)
        for cell_id in spec.active_cells, q in 1:9
            level = spec.cells[cell_id].level
            bank.levels[level + 1].owned[cell_id, q] =
                cell_id + q / 10 + 100 * level
        end

        function active_descendant_sum(parent_id, q)
            children = spec.children[parent_id]
            if children == (0, 0, 0, 0)
                cell = spec.cells[parent_id]
                return cell.active ?
                    bank.levels[cell.level + 1].owned[parent_id, q] : 0.0
            end
            return sum(active_descendant_sum(child_id, q)
                       for child_id in children)
        end

        Kraken.conservative_tree_subcycle_restrict_all_levels_2d!(bank)
        for (cell_id, cell) in pairs(spec.cells)
            spec.children[cell_id] == (0, 0, 0, 0) && continue
            buffers = bank.levels[cell.level + 1]
            for q in 1:9
                @test buffers.restrict_to_parent[cell_id, q] ==
                      active_descendant_sum(cell_id, q)
            end
        end
    end

    @testset "subcycle coarse ghosts are conservative and non-owned" begin
        spec = create_conservative_tree_spec_2d(6, 5, [
            ConservativeTreeRefineBlock2D("fine", 3:4, 2:3),
        ])
        bank = Kraken.create_conservative_tree_subcycle_buffer_bank_2d(spec)
        Fparent = allocate_conservative_tree_F_2d(spec)
        parent_id = conservative_tree_cell_id_2d(spec, 0, 3, 2)
        children = conservative_tree_children_2d(spec, parent_id)
        for q in 1:9
            Fparent[parent_id, q] = 4q
        end

        Kraken.conservative_tree_subcycle_prolong_F_to_child_ghost_2d!(
            bank, Fparent, 0)
        child_buffers = bank.levels[2]
        for q in 1:9
            @test sum(child_buffers.ghost_from_coarse[collect(children), q]) ==
                  Fparent[parent_id, q]
            @test all(child_buffers.ghost_from_coarse[child_id, q] == q
                      for child_id in children)
            @test all(child_buffers.owned[child_id, q] == 0
                      for child_id in children)
        end
    end

    @testset "scheduler binds one L/L+1 interface ledger" begin
        schedule = Kraken.create_conservative_tree_subcycle_schedule_2d(1)
        bank = Kraken.create_conservative_tree_subcycle_ledger_bank_2d(schedule)
        down = only(event for event in schedule.events
                    if event.phase == :sync_down)
        up = only(event for event in schedule.events
                  if event.phase == :sync_up)
        advances = [event for event in schedule.events
                    if event.phase == :advance && event.src_level == 1]

        @test Kraken.conservative_tree_subcycle_local_substep_2d(
            schedule, 0, advances[1].tick) == 1
        @test Kraken.conservative_tree_subcycle_local_substep_2d(
            schedule, 0, advances[2].tick) == 2

        ledger = Kraken.conservative_tree_subcycle_sync_down_face_2d!(
            bank, down, 12.0, 2, :west)
        half1 = zeros(Float64, 2, 2, 9)
        half2 = zeros(Float64, 2, 2, 9)
        half1[2, 1, 2] = 1.25
        half1[2, 2, 2] = 2.75
        half2[2, 1, 2] = 2.0
        half2[2, 2, 2] = 3.0
        Kraken.conservative_tree_subcycle_accumulate_advance_face_2d!(
            bank, advances[1], half1, 2, :east)
        Kraken.conservative_tree_subcycle_accumulate_advance_face_2d!(
            bank, advances[2], half2, 2, :east)

        @test Kraken.conservative_tree_subcycle_sync_up_ledger_2d(bank, up) ===
              ledger
        sums = conservative_tree_subcycle_orientation_sums_2d(ledger)
        @test sums.coarse_to_fine[2] == 12.0
        @test sums.fine_to_coarse[2] == 9.0

        Kraken.reset_conservative_tree_subcycle_pair_2d!(bank, 0)
        ledger = Kraken.conservative_tree_subcycle_pair_ledger_2d(bank, 0)
        Kraken.conservative_tree_subcycle_sync_down_corner_2d!(
            bank, down, 7.0, 6, :southwest)
        corner_half = zeros(Float64, 2, 2, 9)
        corner_half[2, 2, 6] = 2.0
        for event in advances
            Kraken.conservative_tree_subcycle_accumulate_advance_corner_2d!(
                bank, event, corner_half, 6, :northeast)
        end
        sums = conservative_tree_subcycle_orientation_sums_2d(ledger)
        @test sums.coarse_to_fine[6] == 7.0
        @test sums.fine_to_coarse[6] == 4.0
    end

    @testset "scheduler binds all adjacent level-pair ledgers recursively" begin
        schedule = Kraken.create_conservative_tree_subcycle_schedule_2d(3)
        bank = Kraken.create_conservative_tree_subcycle_ledger_bank_2d(schedule)

        for event in schedule.events
            if event.phase == :sync_down
                Fq = 10.0 * (event.src_level + 1)
                Kraken.conservative_tree_subcycle_sync_down_face_2d!(
                    bank, event, Fq, 2, :west)
            elseif event.phase == :advance && event.src_level > 0
                half = zeros(Float64, 2, 2, 9)
                half[2, 1, 2] = 1.0
                half[2, 2, 2] = 1.0
                Kraken.conservative_tree_subcycle_accumulate_advance_face_2d!(
                    bank, event, half, 2, :east)
            elseif event.phase == :sync_up
                @test Kraken.conservative_tree_subcycle_sync_up_ledger_2d(
                    bank, event) ===
                      Kraken.conservative_tree_subcycle_pair_ledger_2d(
                          bank, event.dst_level)
            end
        end

        for parent in 0:2
            ledger = Kraken.conservative_tree_subcycle_pair_ledger_2d(bank, parent)
            sums = conservative_tree_subcycle_orientation_sums_2d(ledger)
            sync_down_count = 2^parent
            child_advance_count = 2^(parent + 1)

            @test sums.coarse_to_fine[2] ==
                  sync_down_count * 10.0 * (parent + 1)
            @test sums.fine_to_coarse[2] == child_advance_count * 2.0
            @test ledger.fine_to_coarse[2, 1] == sync_down_count * 2.0
            @test ledger.fine_to_coarse[2, 2] == sync_down_count * 2.0
        end
    end

    @testset "spatial route ledgers apply one L/L+1 interface to F rows" begin
        spec = create_conservative_tree_spec_2d(6, 5, [
            ConservativeTreeRefineBlock2D("fine", 3:4, 2:3),
        ])
        table = create_conservative_tree_route_table_2d(spec)
        schedule = Kraken.create_conservative_tree_subcycle_schedule_2d(spec.max_level)
        bank = Kraken.create_conservative_tree_subcycle_spatial_ledger_bank_2d(
            spec; schedule=schedule)
        down = only(event for event in schedule.events
                    if event.phase == :sync_down)
        up = only(event for event in schedule.events
                  if event.phase == :sync_up)
        advances = [event for event in schedule.events
                    if event.phase == :advance && event.src_level == 1]

        Fin = allocate_conservative_tree_F_2d(spec)
        Fout = allocate_conservative_tree_F_2d(spec)
        coarse_west = conservative_tree_cell_id_2d(spec, 0, 2, 2)
        refined_parent = conservative_tree_cell_id_2d(spec, 0, 3, 2)
        children = conservative_tree_children_2d(spec, refined_parent)
        Fin[coarse_west, 2] = 12.0
        split_expected = sum(route.weight * Fin[route.src, route.q]
                             for route in table.routes
                             if route.src == coarse_west &&
                                route.q == 2 &&
                                (route.kind == SPLIT_FACE ||
                                 route.kind == SPLIT_CORNER)) * 2

        Kraken.conservative_tree_subcycle_sync_down_routes_F_2d!(
            bank, down, Fin, table)
        Kraken.conservative_tree_subcycle_apply_child_advance_injection_F_2d!(
            Fout, bank, advances[1])
        @test isapprox(sum(Fout[collect(children), 2]), split_expected / 2;
                       atol=1e-14, rtol=0)
        Kraken.conservative_tree_subcycle_apply_child_advance_injection_F_2d!(
            Fout, bank, advances[2])
        @test isapprox(sum(Fout[collect(children), 2]), split_expected;
                       atol=1e-14, rtol=0)

        Kraken.reset_conservative_tree_subcycle_spatial_bank_2d!(bank)
        fill!(Fin, 0.0)
        fill!(Fout, 0.0)
        fine_west = conservative_tree_cell_id_2d(spec, 1, 5, 3)
        Fin[fine_west, 4] = 1.25
        Kraken.conservative_tree_subcycle_accumulate_advance_routes_F_2d!(
            bank, advances[1], Fin, table)
        Fin[fine_west, 4] = 2.75
        Kraken.conservative_tree_subcycle_accumulate_advance_routes_F_2d!(
            bank, advances[2], Fin, table)
        Kraken.conservative_tree_subcycle_apply_sync_up_F_2d!(
            Fout, bank, up)
        @test isapprox(Fout[coarse_west, 4], 2.0; atol=1e-14, rtol=0)
    end

    @testset "spatial route ledgers apply eq/neq alpha to interface packets" begin
        spec = create_conservative_tree_spec_2d(6, 5, [
            ConservativeTreeRefineBlock2D("fine", 3:4, 2:3),
        ])
        table = create_conservative_tree_route_table_2d(spec)
        schedule = Kraken.create_conservative_tree_subcycle_schedule_2d(spec.max_level)
        bank = Kraken.create_conservative_tree_subcycle_spatial_ledger_bank_2d(
            spec; schedule=schedule)
        F = allocate_conservative_tree_F_2d(spec)
        coarse_west = conservative_tree_cell_id_2d(spec, 0, 2, 2)
        fill_equilibrium_integrated_D2Q9!(
            @view(F[coarse_west, :]), 1.0, 1.0, 0.03, 0.0)
        delta = 2e-4 / 4
        F[coarse_west, 6] += delta
        F[coarse_west, 7] -= delta
        F[coarse_west, 8] += delta
        F[coarse_west, 9] -= delta
        route = first(route for route in table.routes
                      if route.src == coarse_west &&
                         route.q == 6 &&
                         route.kind == SPLIT_CORNER)

        Kraken.conservative_tree_subcycle_deposit_coarse_to_fine_route_2d!(
            bank, F, route; alpha=0.25)
        parent_id = spec.cells[route.dst].parent
        ledger = Kraken.conservative_tree_subcycle_spatial_ledger_2d(
            bank, parent_id)

        expected = ledger.ratio * reconstructed_integrated_D2Q9_packet(
            @view(F[coarse_west, :]), 1.0, route.q, route.weight; alpha=0.25)
        raw = ledger.ratio * route.weight * F[coarse_west, route.q]
        @test isapprox(sum(ledger.coarse_to_fine[:, :, route.q, :]),
                       expected; atol=1e-14, rtol=0)
        @test abs(expected - raw) > 1e-6
    end

    @testset "spatial route ledgers recurse over all adjacent pairs" begin
        spec = create_conservative_tree_spec_2d(8, 6, [
            ConservativeTreeRefineBlock2D("outer", 3:6, 2:5),
            ConservativeTreeRefineBlock2D("inner", 7:8, 5:6; parent="outer"),
        ])
        table = create_conservative_tree_route_table_2d(spec)
        schedule = Kraken.create_conservative_tree_subcycle_schedule_2d(spec.max_level)
        bank = Kraken.create_conservative_tree_subcycle_spatial_ledger_bank_2d(
            spec; schedule=schedule)
        Fin = allocate_conservative_tree_F_2d(spec)
        Fout = allocate_conservative_tree_F_2d(spec)
        for cell_id in spec.active_cells
            for q in 1:9
                Fin[cell_id, q] = spec.cells[cell_id].metrics.volume
            end
        end

        for event in schedule.events
            if event.phase == :sync_down
                Kraken.conservative_tree_subcycle_sync_down_routes_F_2d!(
                    bank, event, Fin, table)
            elseif event.phase == :advance && event.src_level > 0
                Kraken.conservative_tree_subcycle_accumulate_advance_routes_F_2d!(
                    bank, event, Fin, table)
                Kraken.conservative_tree_subcycle_apply_child_advance_injection_F_2d!(
                    Fout, bank, event)
            elseif event.phase == :sync_up
                Kraken.conservative_tree_subcycle_apply_sync_up_F_2d!(
                    Fout, bank, event)
            end
        end

        for parent_level in 0:(spec.max_level - 1)
            pair = Kraken.conservative_tree_subcycle_spatial_pair_ledgers_2d(
                bank, parent_level)
            @test !isempty(pair)
            @test sum(sum(ledger.coarse_to_fine) for ledger in values(pair)) > 0
            @test sum(sum(ledger.fine_to_coarse) for ledger in values(pair)) > 0
        end
        @test sum(Fout) > 0
    end

    @testset "subcycled transport matches route scatter without refinement" begin
        spec = create_conservative_tree_spec_2d(
            4, 4, ConservativeTreeRefineBlock2D[])
        table = create_conservative_tree_route_table_2d(spec)
        Fin = allocate_conservative_tree_F_2d(spec)
        Froute = allocate_conservative_tree_F_2d(spec)
        Fsub = allocate_conservative_tree_F_2d(spec)
        for cell_id in spec.active_cells
            for q in 1:9
                Fin[cell_id, q] = cell_id + q / 10
            end
        end

        stream_conservative_tree_routes_F_2d!(
            Froute, Fin, spec, table; boundary=:bounceback)
        Kraken.stream_conservative_tree_subcycled_routes_F_2d!(
            Fsub, Fin, spec, table; boundary=:bounceback)
        @test Fsub == Froute
    end

    @testset "buffered subcycled transport matches route scatter without refinement" begin
        spec = create_conservative_tree_spec_2d(
            4, 4, ConservativeTreeRefineBlock2D[])
        table = create_conservative_tree_route_table_2d(spec)
        Fin = allocate_conservative_tree_F_2d(spec)
        Froute = allocate_conservative_tree_F_2d(spec)
        Fsub = allocate_conservative_tree_F_2d(spec)
        for cell_id in spec.active_cells
            for q in 1:9
                Fin[cell_id, q] = 2cell_id + q / 7
            end
        end

        stream_conservative_tree_routes_F_2d!(
            Froute, Fin, spec, table; boundary=:bounceback)
        Kraken.stream_conservative_tree_subcycled_buffered_routes_F_2d!(
            Fsub, Fin, spec, table; boundary=:bounceback)
        @test Fsub == Froute
    end

    @testset "single-level subcycled transport preserves closed rest mass" begin
        spec = create_conservative_tree_spec_2d(6, 5, [
            ConservativeTreeRefineBlock2D("fine", 3:4, 2:3),
        ])
        table = create_conservative_tree_route_table_2d(spec)
        Fin = allocate_conservative_tree_F_2d(spec)
        Fout = allocate_conservative_tree_F_2d(spec)
        w = weights(D2Q9())
        for cell_id in spec.active_cells
            volume = spec.cells[cell_id].metrics.volume
            for q in 1:9
                Fin[cell_id, q] = w[q] * volume
            end
        end

        Kraken.stream_conservative_tree_subcycled_routes_F_2d!(
            Fout, Fin, spec, table; boundary=:bounceback)

        @test isapprox(sum(active_population_sums_F_2d(Fout, spec)),
                       sum(active_population_sums_F_2d(Fin, spec));
                       atol=1e-12, rtol=0)
        @test maximum(abs.(Fout[spec.active_cells, :] .-
                           Fin[spec.active_cells, :])) <= 1e-14
        diag = Kraken.diagnose_conservative_tree_subcycled_rest_2d(spec, table)
        @test abs(diag.active_drift) <= 1e-12
        @test diag.max_active_abs <= 1e-14
        @test maximum(abs.(diag.level_drift)) <= 1e-12
        @test maximum(abs.(diag.orientation_drift)) <= 1e-12
    end

    @testset "single-level buffered subcycled transport preserves rest" begin
        spec = create_conservative_tree_spec_2d(6, 5, [
            ConservativeTreeRefineBlock2D("fine", 3:4, 2:3),
        ])
        table = create_conservative_tree_route_table_2d(spec)
        Fin = allocate_conservative_tree_F_2d(spec)
        Fout = allocate_conservative_tree_F_2d(spec)
        w = weights(D2Q9())
        for cell_id in spec.active_cells
            volume = spec.cells[cell_id].metrics.volume
            for q in 1:9
                Fin[cell_id, q] = w[q] * volume
            end
        end

        Kraken.stream_conservative_tree_subcycled_buffered_routes_F_2d!(
            Fout, Fin, spec, table; boundary=:bounceback)

        @test isapprox(sum(active_population_sums_F_2d(Fout, spec)),
                       sum(active_population_sums_F_2d(Fin, spec));
                       atol=1e-12, rtol=0)
        @test maximum(abs.(Fout[spec.active_cells, :] .-
                           Fin[spec.active_cells, :])) <= 1e-14
    end

    @testset "nested subcycled transport rest state is the next closure gate" begin
        spec = create_conservative_tree_spec_2d(8, 6, [
            ConservativeTreeRefineBlock2D("outer", 3:6, 2:5),
            ConservativeTreeRefineBlock2D("inner", 7:8, 5:6; parent="outer"),
        ])
        table = create_conservative_tree_route_table_2d(spec)
        Fin = allocate_conservative_tree_F_2d(spec)
        Fout = allocate_conservative_tree_F_2d(spec)
        w = weights(D2Q9())
        for cell_id in spec.active_cells
            volume = spec.cells[cell_id].metrics.volume
            for q in 1:9
                Fin[cell_id, q] = w[q] * volume
            end
        end

        Kraken.stream_conservative_tree_subcycled_routes_F_2d!(
            Fout, Fin, spec, table; boundary=:bounceback)
        diag = Kraken.diagnose_conservative_tree_subcycled_rest_2d(spec, table)

        @test_broken isapprox(sum(active_population_sums_F_2d(Fout, spec)),
                              sum(active_population_sums_F_2d(Fin, spec));
                              atol=1e-12, rtol=0)
        @test_broken maximum(abs.(Fout[spec.active_cells, :] .-
                                  Fin[spec.active_cells, :])) <= 1e-14
        @test_broken abs(diag.active_drift) <= 1e-12
        @test_broken diag.max_active_abs <= 1e-14
    end

    @testset "nested buffered subcycled transport preserves rest state" begin
        spec = create_conservative_tree_spec_2d(8, 6, [
            ConservativeTreeRefineBlock2D("outer", 3:6, 2:5),
            ConservativeTreeRefineBlock2D("inner", 7:8, 5:6; parent="outer"),
        ])
        table = create_conservative_tree_route_table_2d(spec)
        Fin = allocate_conservative_tree_F_2d(spec)
        Fout = allocate_conservative_tree_F_2d(spec)
        w = weights(D2Q9())
        for cell_id in spec.active_cells
            volume = spec.cells[cell_id].metrics.volume
            for q in 1:9
                Fin[cell_id, q] = w[q] * volume
            end
        end

        Kraken.stream_conservative_tree_subcycled_buffered_routes_F_2d!(
            Fout, Fin, spec, table; boundary=:bounceback)

        @test isapprox(sum(active_population_sums_F_2d(Fout, spec)),
                       sum(active_population_sums_F_2d(Fin, spec));
                       atol=1e-12, rtol=0)
        @test maximum(abs.(Fout[spec.active_cells, :] .-
                           Fin[spec.active_cells, :])) <= 1e-14
    end

    @testset "four-level buffered subcycled transport preserves rest state" begin
        spec = create_conservative_tree_spec_2d(16, 12, [
            ConservativeTreeRefineBlock2D("L1", 5:12, 3:10),
            ConservativeTreeRefineBlock2D("L2", 13:20, 7:14; parent="L1"),
            ConservativeTreeRefineBlock2D("L3", 29:36, 17:24; parent="L2"),
            ConservativeTreeRefineBlock2D("L4", 61:68, 37:44; parent="L3"),
        ])
        table = create_conservative_tree_route_table_2d(spec)
        Fin = allocate_conservative_tree_F_2d(spec)
        Fout = allocate_conservative_tree_F_2d(spec)
        w = weights(D2Q9())
        for cell_id in spec.active_cells
            volume = spec.cells[cell_id].metrics.volume
            for q in 1:9
                Fin[cell_id, q] = w[q] * volume
            end
        end

        Kraken.stream_conservative_tree_subcycled_buffered_routes_F_2d!(
            Fout, Fin, spec, table; boundary=:bounceback)

        @test spec.max_level == 4
        @test isapprox(sum(active_population_sums_F_2d(Fout, spec)),
                       sum(active_population_sums_F_2d(Fin, spec));
                       atol=1e-12, rtol=0)
        @test maximum(abs.(Fout[spec.active_cells, :] .-
                           Fin[spec.active_cells, :])) <= 1e-14
    end

    @testset "four-level buffered periodic-x wall-y transport preserves rest" begin
        spec = create_conservative_tree_spec_2d(16, 12, [
            ConservativeTreeRefineBlock2D("L1", 5:12, 3:10),
            ConservativeTreeRefineBlock2D("L2", 13:20, 7:14; parent="L1"),
            ConservativeTreeRefineBlock2D("L3", 29:36, 17:24; parent="L2"),
            ConservativeTreeRefineBlock2D("L4", 61:68, 37:44; parent="L3"),
        ])
        table = create_conservative_tree_route_table_2d(spec; periodic_x=true)
        Fin = allocate_conservative_tree_F_2d(spec)
        Fout = allocate_conservative_tree_F_2d(spec)
        w = weights(D2Q9())
        for cell_id in spec.active_cells
            volume = spec.cells[cell_id].metrics.volume
            for q in 1:9
                Fin[cell_id, q] = w[q] * volume
            end
        end

        Kraken.stream_conservative_tree_subcycled_buffered_routes_F_2d!(
            Fout, Fin, spec, table; boundary=:periodic_x_wall_y)

        @test spec.max_level == 4
        @test isapprox(sum(active_population_sums_F_2d(Fout, spec)),
                       sum(active_population_sums_F_2d(Fin, spec));
                       atol=1e-12, rtol=0)
        @test maximum(abs.(Fout[spec.active_cells, :] .-
                           Fin[spec.active_cells, :])) <= 1e-14

        fill!(Fout, 0.0)
        Kraken.stream_conservative_tree_subcycled_buffered_routes_F_2d!(
            Fout, Fin, spec, table; boundary=:periodic_x_wall_y,
            coarse_to_fine_predictor_weight=0.5)
        @test isapprox(sum(active_population_sums_F_2d(Fout, spec)),
                       sum(active_population_sums_F_2d(Fin, spec));
                       atol=1e-12, rtol=0)
        @test maximum(abs.(Fout[spec.active_cells, :] .-
                           Fin[spec.active_cells, :])) <= 1e-14

        fill!(Fout, 0.0)
        Kraken.stream_conservative_tree_subcycled_buffered_routes_F_2d!(
            Fout, Fin, spec, table; boundary=:periodic_x_wall_y,
            coarse_to_fine_prolongation=:limited_linear)
        @test isapprox(sum(active_population_sums_F_2d(Fout, spec)),
                       sum(active_population_sums_F_2d(Fin, spec));
                       atol=1e-12, rtol=0)
        @test maximum(abs.(Fout[spec.active_cells, :] .-
                           Fin[spec.active_cells, :])) <= 1e-14
    end

    @testset "level-native vertical wall-corner rest canary" begin
        one_level_wall_touch_x = _test_wall_touch_xband_one_level_spec_2d()
        internal_x = _test_internal_xband_nested_spec_2d(4)
        wall_touch_x = _test_nested_band_spec_2d(:xband, 4)
        wall_closed_x = _test_wall_closed_xband_nested_spec_2d(4)

        @test _test_subcycled_rest_maxdiff_2d(
            one_level_wall_touch_x, :level_native) <= 1e-14
        @test _test_subcycled_rest_maxdiff_2d(
            wall_touch_x, :leaf_equivalent) <= 1e-14
        @test _test_subcycled_rest_maxdiff_2d(
            wall_closed_x, :leaf_equivalent) <= 1e-14
        @test _test_subcycled_rest_maxdiff_2d(
            internal_x, :level_native) <= 1e-14
        @test _test_subcycled_rest_maxdiff_2d(
            wall_touch_x, :level_native) <= 1e-14

        one_level_rows = _test_subcycled_rest_row_maxdiffs_2d(
            one_level_wall_touch_x, :level_native)
        nested_rows = _test_subcycled_rest_row_maxdiffs_2d(
            wall_touch_x, :level_native)
        @test maximum(values(one_level_rows)) <= 1e-14
        @test maximum(values(nested_rows)) <= 1e-14
        @test one_level_rows[(0, 1)] <= 1e-14
        @test one_level_rows[(1, 1)] <= 1e-14
        @test one_level_rows[(1, 2)] <= 1e-14
    end

    @testset "level-native wall-phase three-level packet canary" begin
        spec = _test_wall_phase_three_level_spec_2d()

        south_src = conservative_tree_cell_id_2d(spec, 2, 15, 4)
        south_oracle_dst = conservative_tree_cell_id_2d(spec, 0, 3, 1)
        south_spill_dst = conservative_tree_cell_id_2d(spec, 1, 7, 1)
        @test south_src > 0
        @test south_oracle_dst > 0
        @test south_spill_dst > 0

        south = _test_subcycled_unit_packet_level_native_2d(
            spec, south_src, 8)
        @test abs(Float64(south[south_oracle_dst, 6]) - 1.0) <= 1e-14
        @test abs(Float64(south[south_spill_dst, 6])) <= 1e-14
        @test abs(Float64(sum(south[spec.active_cells, :])) - 1.0) <= 1e-14

        north_src = conservative_tree_cell_id_2d(spec, 2, 15, 13)
        north_oracle_dst = conservative_tree_cell_id_2d(spec, 0, 3, 4)
        north_spill_dst = conservative_tree_cell_id_2d(spec, 1, 7, 8)
        @test north_src > 0
        @test north_oracle_dst > 0
        @test north_spill_dst > 0

        north = _test_subcycled_unit_packet_level_native_2d(
            spec, north_src, 7)
        @test abs(Float64(north[north_oracle_dst, 9]) - 1.0) <= 1e-14
        @test abs(Float64(north[north_spill_dst, 9])) <= 1e-14
        @test abs(Float64(sum(north[spec.active_cells, :])) - 1.0) <= 1e-14
    end

    @testset "level-native wall-only packet guard" begin
        spec = _test_wall_phase_three_level_spec_2d()
        src = conservative_tree_cell_id_2d(spec, 0, 1, 1)
        dst_self = conservative_tree_cell_id_2d(spec, 0, 1, 1)
        dst_east = conservative_tree_cell_id_2d(spec, 0, 2, 1)
        dst_wrap = conservative_tree_cell_id_2d(spec, 0, 8, 1)
        @test src > 0
        @test dst_self > 0
        @test dst_east > 0
        @test dst_wrap > 0

        sw = _test_subcycled_unit_packet_level_native_2d(spec, src, 8)
        @test abs(Float64(sw[dst_self, 6]) - 0.5) <= 1e-14
        @test abs(Float64(sw[dst_east, 6]) - 0.25) <= 1e-14
        @test abs(Float64(sw[dst_wrap, 6]) - 0.25) <= 1e-14
        @test abs(Float64(sum(sw[spec.active_cells, :])) - 1.0) <= 1e-14

        se = _test_subcycled_unit_packet_level_native_2d(spec, src, 9)
        @test abs(Float64(se[dst_self, 7]) - 0.5) <= 1e-14
        @test abs(Float64(se[dst_east, 7]) - 0.25) <= 1e-14
        @test abs(Float64(se[dst_wrap, 7]) - 0.25) <= 1e-14
        @test abs(Float64(sum(se[spec.active_cells, :])) - 1.0) <= 1e-14
    end

    @testset "level-native wall-adjacent C2F dipole canary" begin
        spec = _test_wall_phase_three_level_spec_2d()
        src = conservative_tree_cell_id_2d(spec, 1, 7, 1)
        dsts = [
            conservative_tree_cell_id_2d(spec, 2, 17, 1),
            conservative_tree_cell_id_2d(spec, 2, 18, 1),
            conservative_tree_cell_id_2d(spec, 2, 17, 2),
            conservative_tree_cell_id_2d(spec, 2, 18, 2),
        ]
        @test src > 0
        @test all(>(0), dsts)

        east = _test_subcycled_unit_packet_level_native_2d(spec, src, 2)
        c2f_weights = [Float64(east[dst, 2]) for dst in dsts]
        @test_broken maximum(abs.(c2f_weights .- 0.25)) <= 1e-14
        @test abs(Float64(sum(east[spec.active_cells, :])) - 1.0) <= 1e-14
    end

    @testset "level-native pre-stream wall-phase rest gates" begin
        mini = _test_wall_phase_three_level_spec_2d()
        xband = _test_wall_phase_xband_max2_spec_2d()
        bulk = _test_wall_phase_bulk_touch_max_level_2_spec_2d()

        @test _test_subcycled_rest_prestream_maxdiff_2d(
            mini; mode=:noop) <= 1e-14
        @test _test_subcycled_rest_prestream_maxdiff_2d(
            xband; mode=:noop) <= 1e-14
        @test _test_subcycled_rest_prestream_maxdiff_2d(
            bulk; mode=:noop) <= 1e-14

        @test _test_subcycled_rest_prestream_maxdiff_2d(
            mini; mode=:bgk, omega=1.0) <= 1e-14
        @test _test_subcycled_rest_prestream_maxdiff_2d(
            xband; mode=:bgk, omega=1.0) <= 1e-14
        @test _test_subcycled_rest_prestream_maxdiff_2d(
            bulk; mode=:bgk, omega=1.0) <= 1e-14
    end

    @testset "uniform streamwise equilibrium isolates x-normal interface rho defect" begin
        metrics = Dict{Symbol,NamedTuple}()
        diagonal_metrics = Dict{Symbol,NamedTuple}()
        specs = Dict(
            :xband_center_only => _test_nested_band_spec_2d(:xband, 4),
            :xband_wall_closed => _test_wall_closed_xband_nested_spec_2d(4),
            :yband => _test_nested_band_spec_2d(:yband, 4),
        )
        for kind in (:xband_center_only, :xband_wall_closed, :yband)
            spec = specs[kind]
            table = create_conservative_tree_route_table_2d(
                spec; periodic_x=true, sampling=:leaf_equivalent)
            Fin = allocate_conservative_tree_F_2d(spec)
            Fout = similar(Fin)
            _test_fill_uniform_tree_equilibrium_2d!(
                Fin, spec; rho=1.0, ux=1e-4, uy=0.0)

            Kraken.stream_conservative_tree_subcycled_buffered_routes_F_2d!(
                Fout, Fin, spec, table; boundary=:periodic_x_wall_y)

            metrics[kind] = _test_leaf_rho_level_boundary_metrics_2d(
                spec, Fout; rho0=1.0)

            _test_fill_diagonal_x_odd_tree_perturbation_2d!(
                Fin, spec; epsilon=1e-6)
            fill!(Fout, 0.0)
            Kraken.stream_conservative_tree_subcycled_buffered_routes_F_2d!(
                Fout, Fin, spec, table; boundary=:periodic_x_wall_y)
            diagonal_metrics[kind] =
                _test_leaf_rho_level_boundary_metrics_2d(
                    spec, Fout; rho0=0.0)
        end

        @test metrics[:xband_center_only].x_count > 0
        @test_broken metrics[:xband_center_only].max_x_jump <= 1e-14
        @test_broken metrics[:xband_center_only].max_abs_dev <= 1e-14
        @test diagonal_metrics[:xband_center_only].x_count > 0
        @test_broken diagonal_metrics[:xband_center_only].max_x_jump <= 1e-18
        @test_broken diagonal_metrics[:xband_center_only].max_abs_dev <= 1e-18

        @test metrics[:xband_wall_closed].x_count > 0
        @test metrics[:xband_wall_closed].max_x_jump <= 1e-14
        @test metrics[:xband_wall_closed].max_abs_dev <= 1e-14
        @test diagonal_metrics[:xband_wall_closed].x_count > 0
        @test diagonal_metrics[:xband_wall_closed].max_x_jump <= 1e-18
        @test diagonal_metrics[:xband_wall_closed].max_abs_dev <= 1e-18

        @test metrics[:yband].y_count > 0
        @test metrics[:yband].max_y_jump <= 1e-14
        @test metrics[:yband].max_abs_dev <= 1e-14
        @test diagonal_metrics[:yband].y_count > 0
        @test diagonal_metrics[:yband].max_y_jump <= 1e-18
        @test diagonal_metrics[:yband].max_abs_dev <= 1e-18
    end

    @testset "level-native route sampling is isolated behind explicit scaling" begin
        direct_spec = create_conservative_tree_spec_2d(6, 4, [
            ConservativeTreeRefineBlock2D("patch", 3:4, 2:3),
        ])
        src = conservative_tree_cell_id_2d(direct_spec, 0, 1, 2)
        leaf_table = create_conservative_tree_route_table_2d(
            direct_spec; sampling=:leaf_equivalent)
        native_table = create_conservative_tree_route_table_2d(
            direct_spec; sampling=:level_native)
        leaf_routes = [leaf_table.routes[rid] for rid in leaf_table.direct_routes
                       if leaf_table.routes[rid].src == src &&
                          leaf_table.routes[rid].q == 2]
        native_routes = [native_table.routes[rid] for rid in native_table.direct_routes
                         if native_table.routes[rid].src == src &&
                            native_table.routes[rid].q == 2]

        @test length(leaf_routes) == 2
        @test sort([route.weight for route in leaf_routes]) == [0.5, 0.5]
        @test length(native_routes) == 1
        @test native_routes[1].weight == 1.0
        @test direct_spec.cells[native_routes[1].dst].i == 2

        c2f_src = conservative_tree_cell_id_2d(direct_spec, 0, 2, 2)
        c2f_routes = [native_table.routes[rid] for rid in native_table.interface_routes
                      if native_table.routes[rid].src == c2f_src &&
                         native_table.routes[rid].q == 2]
        @test length(c2f_routes) == 2
        @test all(route.kind == SPLIT_FACE for route in c2f_routes)
        @test sort([route.weight for route in c2f_routes]) == [0.5, 0.5]

        spec = create_conservative_tree_nested_channel_spec_2d(2)
        table = create_conservative_tree_route_table_2d(
            spec; periodic_x=true, sampling=:level_native)
        southwest_corner = [route for route in table.routes
                            if route.src == 20 && route.q == 6]
        @test length(southwest_corner) == 1
        @test southwest_corner[1].dst == 193
        @test southwest_corner[1].kind == SPLIT_CORNER
        @test southwest_corner[1].weight == 0.5

        southeast_touch = [route for route in table.routes
                           if route.src == 21 && route.q == 7]
        @test length(southeast_touch) == 2
        @test any(route.dst == 193 && route.kind == SPLIT_CORNER &&
                  route.weight == 0.5 for route in southeast_touch)
        @test any(route.dst == 36 && route.kind == DIRECT &&
                  route.weight == 0.75 for route in southeast_touch)

        face_diagonal = [route for route in table.routes
                         if route.src == 21 && route.q == 6]
        @test length(face_diagonal) == 2
        @test all(route.kind == SPLIT_CORNER for route in face_diagonal)
        @test sort([route.weight for route in face_diagonal]) == [0.5, 0.5]

        wall_touch_spec = create_conservative_tree_spec_2d(16, 12, [
            ConservativeTreeRefineBlock2D("X1", 6:10, 1:12),
        ])
        wall_touch_table = create_conservative_tree_route_table_2d(
            wall_touch_spec; periodic_x=true, sampling=:level_native)
        wall_corner = [route for route in wall_touch_table.routes
                       if route.src == 5 && route.q == 9]
        @test length(wall_corner) == 2
        @test any(route.kind == SPLIT_CORNER && route.weight == 0.5
                  for route in wall_corner)
        @test any(route.kind == ROUTE_BOUNDARY && route.weight == 0.5
                  for route in wall_corner)

        wall_partner = [route for route in wall_touch_table.routes
                        if route.src == 5 && route.q == 8]
        @test length(wall_partner) == 1
        @test wall_partner[1].kind == ROUTE_BOUNDARY
        @test wall_partner[1].weight == 1.0

        Fin = allocate_conservative_tree_F_2d(spec)
        Fout = allocate_conservative_tree_F_2d(spec)
        w = weights(D2Q9())
        for cell_id in spec.active_cells
            volume = spec.cells[cell_id].metrics.volume
            for q in 1:5
                Fin[cell_id, q] = w[q] * volume
            end
        end

        Kraken.stream_conservative_tree_subcycled_buffered_routes_F_2d!(
            Fout, Fin, spec, table; boundary=:periodic_x_wall_y,
            interface_time_scaling=:level_native)

        @test isapprox(sum(active_population_sums_F_2d(Fout, spec)),
                       sum(active_population_sums_F_2d(Fin, spec));
                       atol=1e-12, rtol=0)
        @test maximum(abs.(Fout[spec.active_cells, :] .-
                           Fin[spec.active_cells, :])) <= 1e-14

        fill!(Fin, 0.0)
        fill!(Fout, 0.0)
        for cell_id in spec.active_cells
            volume = spec.cells[cell_id].metrics.volume
            for q in 1:9
                Fin[cell_id, q] = w[q] * volume
            end
        end

        Kraken.stream_conservative_tree_subcycled_buffered_routes_F_2d!(
            Fout, Fin, spec, table; boundary=:periodic_x_wall_y,
            interface_time_scaling=:level_native)

        @test isapprox(sum(active_population_sums_F_2d(Fout, spec)),
                       sum(active_population_sums_F_2d(Fin, spec));
                       atol=1e-12, rtol=0)
        @test maximum(abs.(Fout[spec.active_cells, :] .-
                           Fin[spec.active_cells, :])) <= 1e-14

        spec4 = create_conservative_tree_nested_channel_spec_2d(4)
        table4 = create_conservative_tree_route_table_2d(
            spec4; periodic_x=true, sampling=:level_native)
        Fin4 = allocate_conservative_tree_F_2d(spec4)
        Fout4 = allocate_conservative_tree_F_2d(spec4)
        for cell_id in spec4.active_cells
            volume = spec4.cells[cell_id].metrics.volume
            for q in 1:9
                Fin4[cell_id, q] = w[q] * volume
            end
        end

        Kraken.stream_conservative_tree_subcycled_buffered_routes_F_2d!(
            Fout4, Fin4, spec4, table4; boundary=:periodic_x_wall_y,
            interface_time_scaling=:level_native)
        @test isapprox(sum(active_population_sums_F_2d(Fout4, spec4)),
                       sum(active_population_sums_F_2d(Fin4, spec4));
                       atol=1e-12, rtol=0)
        @test maximum(abs.(Fout4[spec4.active_cells, :] .-
                           Fin4[spec4.active_cells, :])) <= 1e-14

        Finw = allocate_conservative_tree_F_2d(wall_touch_spec)
        Foutw = allocate_conservative_tree_F_2d(wall_touch_spec)
        for cell_id in wall_touch_spec.active_cells
            volume = wall_touch_spec.cells[cell_id].metrics.volume
            for q in 1:9
                Finw[cell_id, q] = w[q] * volume
            end
        end
        Kraken.stream_conservative_tree_subcycled_buffered_routes_F_2d!(
            Foutw, Finw, wall_touch_spec, wall_touch_table;
            boundary=:periodic_x_wall_y, interface_time_scaling=:level_native)
        @test isapprox(sum(active_population_sums_F_2d(Foutw, wall_touch_spec)),
                       sum(active_population_sums_F_2d(Finw, wall_touch_spec));
                       atol=1e-12, rtol=0)
        @test maximum(abs.(Foutw[wall_touch_spec.active_cells, :] .-
                           Finw[wall_touch_spec.active_cells, :])) <= 1e-14

        @test_throws ArgumentError run_conservative_tree_poiseuille_subcycled_2d(
            max_level=2, steps=1, route_sampling=:level_native,
            coarse_to_fine_prolongation=:limited_linear)
    end

    @testset "analytical affine leaf-equivalent operator canaries" begin
        one_level = _test_affine_operator_one_level_spec_2d()
        two_level = _test_affine_operator_two_level_spec_2d()

        for q in (2, 3, 4, 5)
            leaf = _test_subcycled_affine_operator_metrics_2d(
                one_level, q, :leaf_equivalent)
            native = _test_subcycled_affine_operator_metrics_2d(
                one_level, q, :level_native)

            @test leaf.l0_bulk_count > 0
            @test native.l0_bulk_count == leaf.l0_bulk_count
            @test native.refined_or_interface_count > 0
            @test native.max_l0_bulk <= 1e-14

            @test_broken leaf.max_l0_bulk <= 1e-14
            @test native.max_refined_or_interface <= 1e-14
        end

        for q in (6, 7, 8, 9)
            native = _test_subcycled_affine_operator_metrics_2d(
                one_level, q, :level_native)
            @test native.refined_or_interface_count > 0
            @test native.max_l0_bulk <= 1e-14
            @test native.max_refined_or_interface <= 1e-14
        end

        for q in 2:9
            nested = _test_subcycled_affine_operator_metrics_2d(
                two_level, q, :level_native)
            @test nested.l0_bulk_count > 0
            @test nested.max_l0_bulk <= 1e-14
            @test nested.refined_or_interface_count > 0
            @test nested.max_refined_or_interface <= 1e-14
        end
    end

    @testset "full-domain nested Poiseuille matches Cartesian at same physical time" begin
        spec = _test_full_domain_nested_spec_2d(2)
        amr = run_conservative_tree_poiseuille_subcycled_2d(
            max_level=2, spec=spec, steps=12, Fx=1e-7, omega=1.0)
        cart_profile = _test_cartesian_poiseuille_profile_2d(
            2, 12; Fx=1e-7, omega=1.0)

        @test spec.max_level == 2
        @test length(spec.active_cells) == 16 * 12 * 4^2
        @test maximum(abs.(amr.ux_profile .- cart_profile)) < 1e-14
        @test amr.relative_mass_drift < 1e-13
    end

    @testset "short-time wall-normal Poiseuille improves with wall coverage" begin
        steps = 96
        center = run_conservative_tree_poiseuille_subcycled_2d(
            max_level=2, spec=_test_center_yband_nested_spec_2d(),
            steps=steps, Fx=1e-7, omega=1.0,
            route_sampling=:leaf_equivalent)
        walls = run_conservative_tree_poiseuille_subcycled_2d(
            max_level=2, spec=_test_wall_refined_ybands_nested_spec_2d(),
            steps=steps, Fx=1e-7, omega=1.0,
            route_sampling=:leaf_equivalent)
        cart = _test_cartesian_poiseuille_profile_2d(
            2, steps; Fx=1e-7, omega=1.0)

        center_diff = center.ux_profile .- cart
        walls_diff = walls.ux_profile .- cart
        center_l2 = sqrt(sum(center_diff .^ 2) / length(center_diff))
        walls_l2 = sqrt(sum(walls_diff .^ 2) / length(walls_diff))

        @test maximum(abs.(walls_diff)) < maximum(abs.(center_diff))
        @test walls_l2 < 0.2 * center_l2
        @test abs(maximum(walls.ux_profile) - maximum(cart)) <
              abs(maximum(center.ux_profile) - maximum(cart))
    end

    @testset "coarse-fine temporal predictor reduces wall-normal bias" begin
        steps = 192
        cart = _test_cartesian_poiseuille_profile_2d(
            2, steps; Fx=1e-7, omega=1.0)

        center_flat = run_conservative_tree_poiseuille_subcycled_2d(
            max_level=2, spec=_test_center_yband_nested_spec_2d(),
            steps=steps, Fx=1e-7, omega=1.0,
            route_sampling=:leaf_equivalent,
            coarse_to_fine_predictor_weight=0,
            enforce_mass=false)
        center_predicted = run_conservative_tree_poiseuille_subcycled_2d(
            max_level=2, spec=_test_center_yband_nested_spec_2d(),
            steps=steps, Fx=1e-7, omega=1.0,
            route_sampling=:leaf_equivalent,
            enforce_mass=false)
        flat_diff = center_flat.ux_profile .- cart
        predicted_diff = center_predicted.ux_profile .- cart
        flat_l2 = sqrt(sum(flat_diff .^ 2) / length(flat_diff))
        predicted_l2 = sqrt(sum(predicted_diff .^ 2) / length(predicted_diff))

        x_flat = run_conservative_tree_poiseuille_subcycled_2d(
            max_level=2, spec=_test_center_xband_nested_spec_2d(),
            steps=steps, Fx=1e-7, omega=1.0,
            route_sampling=:leaf_equivalent,
            coarse_to_fine_predictor_weight=0,
            enforce_mass=false)
        x_predicted = run_conservative_tree_poiseuille_subcycled_2d(
            max_level=2, spec=_test_center_xband_nested_spec_2d(),
            steps=steps, Fx=1e-7, omega=1.0,
            route_sampling=:leaf_equivalent,
            enforce_mass=false)
        x_flat_linf = maximum(abs.(x_flat.ux_profile .- cart))
        x_predicted_linf = maximum(abs.(x_predicted.ux_profile .- cart))

        @test predicted_l2 < 0.75 * flat_l2
        @test x_predicted_linf <= 1.05 * x_flat_linf
    end

    @testset "nested bands stay close to Poiseuille analytic profile" begin
        steps = 1024
        xband = run_conservative_tree_poiseuille_subcycled_2d(
            max_level=2, spec=_test_center_xband_nested_spec_2d(),
            steps=steps, Fx=1e-7, omega=1.0,
            route_sampling=:leaf_equivalent, enforce_mass=false)
        yband = run_conservative_tree_poiseuille_subcycled_2d(
            max_level=2, spec=_test_center_yband_nested_spec_2d(),
            steps=steps, Fx=1e-7, omega=1.0,
            route_sampling=:level_native, enforce_mass=false)
        wall_bands = run_conservative_tree_poiseuille_subcycled_2d(
            max_level=2, spec=_test_wall_refined_ybands_nested_spec_2d(),
            steps=steps, Fx=1e-7, omega=1.0,
            route_sampling=:level_native, enforce_mass=false)

        @test xband.linf_error < 3e-5
        @test yband.linf_error < 3e-5
        @test wall_bands.linf_error < 2e-5
        @test xband.relative_mass_drift < 1e-8
        @test yband.relative_mass_drift < 1e-12
        @test wall_bands.relative_mass_drift < 1e-12
    end

    @testset "nested route-mode matrix tracks same-time Cartesian transients" begin
        steps = 16
        kinds = (:xband, :yband, :wall_ybands)
        routes = (:leaf_equivalent, :level_native)
        for max_level in 2:4
            cart_poiseuille = _test_cartesian_poiseuille_profile_2d(
                max_level, steps; Fx=1e-7, omega=1.0)
            cart_couette = _test_cartesian_couette_profile_2d(
                max_level, steps; U=1e-4, omega=1.0)
            for kind in kinds, route in routes
                spec = kind == :xband ?
                    _test_wall_closed_xband_nested_spec_2d(max_level) :
                    _test_nested_band_spec_2d(kind, max_level)
                poiseuille = run_conservative_tree_poiseuille_subcycled_2d(
                    max_level=max_level, spec=spec, steps=steps,
                    Fx=1e-7, omega=1.0, route_sampling=route,
                    enforce_mass=false)
                couette = run_conservative_tree_couette_subcycled_2d(
                    max_level=max_level, spec=spec, steps=steps,
                    U=1e-4, omega=1.0, route_sampling=route,
                    enforce_mass=false)

                @test _test_profile_linf_2d(
                    poiseuille.ux_profile, cart_poiseuille) < 1e-4
                @test _test_profile_linf_2d(
                    couette.ux_profile, cart_couette) < 1e-4
                p_bounds = _test_active_field_bounds_2d(
                    poiseuille; force_x=1e-7, level_scaled_force=true)
                if kind == :xband && route == :level_native
                    @test_broken p_bounds.rho_min > 0.999
                    @test_broken p_bounds.rho_max < 1.001
                    @test_broken p_bounds.ux_min > -1e-6
                else
                    @test p_bounds.rho_min > 0.999
                    @test p_bounds.rho_max < 1.001
                    @test p_bounds.ux_min > -1e-6
                end
                guard = conservative_tree_mass_roundoff_rtol_2d(
                    Float64, steps, max_level;
                    active_cell_count=length(spec.active_cells))
                @test poiseuille.relative_mass_drift <= max(guard, 1e-10)
                @test couette.relative_mass_drift <= max(guard, 1e-10)
            end
        end
    end

    @testset "subcycled Poiseuille macroflow runs from level 1 to 4" begin
        for max_level in 1:4
            result = run_conservative_tree_poiseuille_subcycled_2d(
                max_level=max_level, steps=8, Fx=1e-7)
            cart = run_cartesian_channel_mass_reference_2d(
                flow=:poiseuille, max_level=max_level, steps=8, Fx=1e-7)
            guard = conservative_tree_mass_roundoff_rtol_2d(
                Float64, result.steps, max_level)
            @test result.flow == :poiseuille_subcycled
            @test result.max_level == max_level
            @test result.steps == 8
            @test all(isfinite, result.ux_profile)
            @test all(isfinite, result.analytic_profile)
            @test isfinite(result.l2_error)
            @test isfinite(result.linf_error)
            @test result.max_raw_relative_mass_drift <= guard
            @test result.relative_mass_drift <=
                  max(cart.relative_mass_drift, 10eps(Float64))
            @test maximum(result.ux_profile) > 0
            @test result.active_cell_count < result.leaf_equivalent_cell_count
        end
    end

    @testset "mass roundoff guard scales with active leaf count" begin
        small = conservative_tree_mass_roundoff_rtol_2d(
            Float64, 1, 4; active_cell_count=1)
        band = conservative_tree_mass_roundoff_rtol_2d(
            Float64, 1, 4; active_cell_count=20_000)
        @test band > small
        @test band < 20 * small
    end

    @testset "leaf-equivalent physics scales by AMR-D level" begin
        spec = Kraken.create_conservative_tree_nested_channel_spec_2d(4)
        omega_fine = 1.0
        force_fine = 1e-7

        @test Kraken.conservative_tree_leaf_equivalent_level_scale_2d(
            spec, 4) == 1
        @test Kraken.conservative_tree_leaf_equivalent_level_scale_2d(
            spec, 0) == 16
        @test Kraken.conservative_tree_leaf_equivalent_force_2d(
            force_fine, spec, 0) == 16force_fine
        @test Kraken.conservative_tree_leaf_equivalent_force_2d(
            force_fine, spec, 4) == force_fine

        tau_fine = inv(omega_fine)
        tau_coarse = inv(Kraken.conservative_tree_leaf_equivalent_omega_2d(
            omega_fine, spec, 0))
        @test tau_coarse - 0.5 ≈ (tau_fine - 0.5) / 16
        @test Kraken.conservative_tree_leaf_equivalent_omega_2d(
            omega_fine, spec, 4) == omega_fine
        @test_throws ArgumentError Kraken.conservative_tree_leaf_equivalent_omega_2d(
            2.0, spec, 0)
    end

    @testset "subcycled Couette macroflow runs from level 1 to 4" begin
        for max_level in 1:4
            result = run_conservative_tree_couette_subcycled_2d(
                max_level=max_level, steps=8, U=1e-4)
            cart = run_cartesian_channel_mass_reference_2d(
                flow=:couette, max_level=max_level, steps=8, U=1e-4)
            guard = conservative_tree_mass_roundoff_rtol_2d(
                Float64, result.steps, max_level)
            @test result.flow == :couette_subcycled
            @test result.max_level == max_level
            @test all(isfinite, result.ux_profile)
            @test isfinite(result.l2_error)
            @test isfinite(result.linf_error)
            @test result.max_raw_relative_mass_drift <= guard
            @test result.relative_mass_drift <=
                  max(cart.relative_mass_drift, 10eps(Float64))
            @test result.ux_profile[end] > result.ux_profile[1]
        end
    end

    @testset "subcycled macroflow compiles with Float32 storage" begin
        result = run_conservative_tree_poiseuille_subcycled_2d(
            max_level=1, steps=1, Fx=Float32(1e-7), T=Float32)
        @test eltype(result.F) == Float32
        @test eltype(result.ux_profile) == Float32
        @test all(isfinite, result.ux_profile)
        @test result.max_raw_relative_mass_drift <=
              conservative_tree_mass_roundoff_rtol_2d(Float32, 1, 1)
    end

    @testset "scheduled ledger binding rejects wrong events" begin
        schedule = Kraken.create_conservative_tree_subcycle_schedule_2d(2)
        bank = Kraken.create_conservative_tree_subcycle_ledger_bank_2d(schedule)
        down = first(event for event in schedule.events
                     if event.phase == :sync_down)
        advance0 = first(event for event in schedule.events
                         if event.phase == :advance && event.src_level == 0)
        advance1 = first(event for event in schedule.events
                         if event.phase == :advance && event.src_level == 1)

        half = zeros(Float64, 2, 2, 9)
        @test_throws ArgumentError Kraken.conservative_tree_subcycle_sync_down_face_2d!(
            bank, advance1, 1.0, 2, :west)
        @test_throws ArgumentError Kraken.conservative_tree_subcycle_accumulate_advance_face_2d!(
            bank, down, half, 2, :east)
        @test_throws ArgumentError Kraken.conservative_tree_subcycle_accumulate_advance_face_2d!(
            bank, advance0, half, 2, :east)
        @test_throws ArgumentError Kraken.conservative_tree_subcycle_sync_up_ledger_2d(
            bank, down)
        @test_throws ArgumentError Kraken.conservative_tree_subcycle_pair_ledger_2d(
            bank, 2)
        @test_throws ArgumentError Kraken.conservative_tree_subcycle_local_substep_2d(
            schedule, 0, 1)
    end

    @testset "coarse-to-fine face packet is consumed once over two half steps" begin
        ledger = create_conservative_tree_subcycle_ledger_2d()
        conservative_tree_subcycle_deposit_coarse_to_fine_face_2d!(
            ledger, 12.0, 2, :west)
        sums = conservative_tree_subcycle_orientation_sums_2d(ledger)
        totals = conservative_tree_subcycle_total_sums_2d(ledger)

        @test ledger.ratio == 2
        @test conservative_tree_subcycle_weights_2d(ledger) == [0.5, 0.5]
        @test sums.coarse_to_fine[2] == 12.0
        @test sum(sums.coarse_to_fine) == 12.0
        @test all(iszero, sums.coarse_to_fine[[1, 3, 4, 5, 6, 7, 8, 9]])
        @test totals.coarse_to_fine == 12.0
        @test totals.fine_to_coarse == 0.0

        for substep in 1:2
            @test ledger.coarse_to_fine[1, 1, 2, substep] == 3.0
            @test ledger.coarse_to_fine[1, 2, 2, substep] == 3.0
            @test sum(ledger.coarse_to_fine[:, :, 2, substep]) == 6.0
        end
    end

    @testset "coarse-to-fine corner packet is split only in time" begin
        ledger = create_conservative_tree_subcycle_ledger_2d()
        conservative_tree_subcycle_deposit_coarse_to_fine_corner_2d!(
            ledger, 7.0, 6, :southwest)
        sums = conservative_tree_subcycle_orientation_sums_2d(ledger)

        @test sums.coarse_to_fine[6] == 7.0
        @test sum(sums.coarse_to_fine) == 7.0
        for substep in 1:2
            @test ledger.coarse_to_fine[1, 1, 6, substep] == 3.5
            @test sum(ledger.coarse_to_fine[:, :, 6, substep]) == 3.5
        end
    end

    @testset "fine-to-coarse face packets accumulate by half step" begin
        ledger = create_conservative_tree_subcycle_ledger_2d()
        half1 = zeros(Float64, 2, 2, 9)
        half2 = zeros(Float64, 2, 2, 9)
        half1[2, 1, 2] = 1.25
        half1[2, 2, 2] = 2.75
        half2[2, 1, 2] = 2.0
        half2[2, 2, 2] = 3.0

        conservative_tree_subcycle_accumulate_fine_to_coarse_face_2d!(
            ledger, half1, 2, :east, 1)
        conservative_tree_subcycle_accumulate_fine_to_coarse_face_2d!(
            ledger, half2, 2, :east, 2)
        sums = conservative_tree_subcycle_orientation_sums_2d(ledger)

        @test ledger.fine_to_coarse[2, 1] == 4.0
        @test ledger.fine_to_coarse[2, 2] == 5.0
        @test sums.fine_to_coarse[2] == 9.0
        @test sum(sums.fine_to_coarse) == 9.0
        @test sum(sums.coarse_to_fine) == 0.0
    end

    @testset "full cycle ledger preserves expected orientation totals" begin
        ledger = create_conservative_tree_subcycle_ledger_2d()
        conservative_tree_subcycle_deposit_coarse_to_fine_face_2d!(
            ledger, 6.0, 2, :west)
        conservative_tree_subcycle_deposit_coarse_to_fine_corner_2d!(
            ledger, 4.0, 6, :southwest)

        face_half = zeros(Float64, 2, 2, 9)
        face_half[2, 1, 2] = 1.5
        face_half[2, 2, 2] = 1.5
        corner_half = zeros(Float64, 2, 2, 9)
        corner_half[2, 2, 6] = 2.0
        for substep in 1:2
            conservative_tree_subcycle_accumulate_fine_to_coarse_face_2d!(
                ledger, face_half, 2, :east, substep)
            conservative_tree_subcycle_accumulate_fine_to_coarse_corner_2d!(
                ledger, corner_half, 6, :northeast, substep)
        end

        sums = conservative_tree_subcycle_orientation_sums_2d(ledger)
        totals = conservative_tree_subcycle_total_sums_2d(ledger)
        @test sums.coarse_to_fine[2] == 6.0
        @test sums.coarse_to_fine[6] == 4.0
        @test sums.fine_to_coarse[2] == 6.0
        @test sums.fine_to_coarse[6] == 4.0
        @test totals.coarse_to_fine == 10.0
        @test totals.fine_to_coarse == 10.0

        reset_conservative_tree_subcycle_ledger_2d!(ledger)
        empty = conservative_tree_subcycle_total_sums_2d(ledger)
        @test empty.coarse_to_fine == 0.0
        @test empty.fine_to_coarse == 0.0
    end

    @testset "rest equilibrium: axis face deposit and accumulate balance per cycle" begin
        # Setup: a coarse cell at rest equilibrium (rho=1, V_c=1) sends its q-th
        # population through the entry face into the fine patch. The fine patch
        # children are at fine equilibrium (rho=1, V_f=0.25) and contribute back
        # via the opposite direction through the same physical face over 2 sub-steps.
        # The ledger must reflect a balanced cycle for axis directions q in 2:5
        # because a face has 2 children on each side and a cycle has 2 sub-steps:
        #   coarse->fine[q]   = w_q * V_c
        #   fine->coarse[opp] = 2 children * w_opp * V_f * 2 sub-steps = w_opp * V_c
        # With w_q == w_opp for axis directions, both totals are equal.
        Vc = 1.0
        Vf = 0.25
        wq_axis = 1 / 9  # D2Q9 weight for axis directions
        face_in_for = Dict(2 => :west, 3 => :south, 4 => :east, 5 => :north)
        face_out_for = Dict(2 => :east, 3 => :north, 4 => :west, 5 => :south)

        for q in (2, 3, 4, 5)
            opp = d2q9_opposite(q)
            ledger = create_conservative_tree_subcycle_ledger_2d()

            Fq_eq_coarse = wq_axis * Vc
            conservative_tree_subcycle_deposit_coarse_to_fine_face_2d!(
                ledger, Fq_eq_coarse, q, face_in_for[q])

            fine_block = zeros(Float64, 2, 2, 9)
            for ic in 1:2, jc in 1:2
                fine_block[ic, jc, opp] = wq_axis * Vf
            end
            for substep in 1:2
                conservative_tree_subcycle_accumulate_fine_to_coarse_face_2d!(
                    ledger, fine_block, opp, face_out_for[opp], substep)
            end

            sums = conservative_tree_subcycle_orientation_sums_2d(ledger)
            @test isapprox(sums.coarse_to_fine[q], wq_axis * Vc;
                           atol=1e-15, rtol=0)
            @test isapprox(sums.fine_to_coarse[opp], wq_axis * Vc;
                           atol=1e-15, rtol=0)
            @test isapprox(sums.coarse_to_fine[q], sums.fine_to_coarse[opp];
                           atol=1e-15, rtol=0)
            # All other orientations must be zero.
            for qz in 1:9
                qz == q && continue
                qz == opp && continue
                @test sums.coarse_to_fine[qz] == 0.0
                @test sums.fine_to_coarse[qz] == 0.0
            end
        end
    end

    @testset "rest equilibrium: corner deposit per substep matches fine equilibrium" begin
        # Setup: coarse cell sends its diagonal q (in 6:9) through one corner.
        # The ledger splits the packet evenly across 2 sub-steps; each sub-step's
        # corner deposit lands in 1 fine child cell. The deposit value per
        # sub-step (= w_q * V_c / 2) must equal the fine equilibrium of that q
        # (= w_q * V_f * 2) up to the well-known 2x corner geometric factor.
        # Concretely, ledger.coarse_to_fine[ix, iy, q, substep] = w_q * V_c / 2
        # = w_q * V_f * 2; the 2x prefactor reflects that 1 coarse step covers 2
        # fine half-steps so the per-sub-step deposit is twice the per-fine-cell
        # equilibrium. This is documented as a known imbalance that the time
        # integrator must compensate via Filippova-Hanel rescaling and reflux.
        Vc = 1.0
        Vf = 0.25
        wq_corner = 1 / 36
        corner_in_for = Dict(6 => :southwest, 7 => :southeast,
                             8 => :northeast, 9 => :northwest)
        corner_out_for = Dict(6 => :northeast, 7 => :northwest,
                              8 => :southwest, 9 => :southeast)

        for q in (6, 7, 8, 9)
            opp = d2q9_opposite(q)
            ledger = create_conservative_tree_subcycle_ledger_2d()

            Fq_eq_coarse = wq_corner * Vc
            conservative_tree_subcycle_deposit_coarse_to_fine_corner_2d!(
                ledger, Fq_eq_coarse, q, corner_in_for[q])

            fine_block = zeros(Float64, 2, 2, 9)
            for ic in 1:2, jc in 1:2
                fine_block[ic, jc, opp] = wq_corner * Vf
            end
            for substep in 1:2
                conservative_tree_subcycle_accumulate_fine_to_coarse_corner_2d!(
                    ledger, fine_block, opp, corner_out_for[opp], substep)
            end

            sums = conservative_tree_subcycle_orientation_sums_2d(ledger)
            # coarse->fine[q] = wq_corner * Vc (full coarse packet over 1 cycle)
            @test isapprox(sums.coarse_to_fine[q], wq_corner * Vc;
                           atol=1e-15, rtol=0)
            # fine->coarse[opp] = 1 child * wq_corner * Vf * 2 substeps
            #                   = wq_corner * Vc / 2
            @test isapprox(sums.fine_to_coarse[opp], wq_corner * Vc / 2;
                           atol=1e-15, rtol=0)
            # The 2x corner imbalance is INTRINSIC to the subcycling geometry,
            # not a ledger bug. The future time integrator must rescale
            # (Filippova-Hanel) or accept this as residual on coarse.
            @test sums.fine_to_coarse[opp] == sums.coarse_to_fine[q] / 2
        end
    end

    @testset "ledger rejects unsupported contracts" begin
        ledger = create_conservative_tree_subcycle_ledger_2d()
        bad_block = zeros(Float64, 3, 2, 9)

        @test_throws ArgumentError create_conservative_tree_subcycle_ledger_2d(ratio=3)
        @test_throws ArgumentError conservative_tree_subcycle_deposit_coarse_to_fine_face_2d!(
            ledger, 1.0, 2, :east)
        @test_throws ArgumentError conservative_tree_subcycle_deposit_coarse_to_fine_corner_2d!(
            ledger, 1.0, 2, :southwest)
        @test_throws ArgumentError conservative_tree_subcycle_accumulate_fine_to_coarse_face_2d!(
            ledger, bad_block, 2, :east, 1)
        @test_throws ArgumentError conservative_tree_subcycle_accumulate_fine_to_coarse_face_2d!(
            ledger, zeros(Float64, 2, 2, 9), 2, :east, 3)
    end
end
