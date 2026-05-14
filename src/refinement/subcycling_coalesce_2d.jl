function _conservative_tree_subcycle_accumulate_fine_to_coarse_packet_unchecked_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        F::AbstractMatrix,
        src_id::Integer,
        dst_id::Integer,
        q::Integer,
        weight,
        kind::RouteKind,
        substep::Integer,
        route_packet_slot::Integer=0;
        alpha=1,
        interface_time_scaling::Symbol=:leaf_equivalent,
        current_tick::Integer=0,
        delayed_same_level_transit::Bool=true,
        periodic_x::Bool=false)
    kind == COALESCE_FACE || kind == COALESCE_CORNER ||
        throw(ArgumentError("route must be a fine-to-coarse coalesce route"))

    spec = bank.spec
    child_id = Int(src_id)
    child = spec.cells[child_id]
    child.level > 0 ||
        throw(ArgumentError("fine-to-coarse route source must have a parent"))
    parent_id = child.parent
    parent = spec.cells[parent_id]
    dst_cell_id = Int(dst_id)
    dst_cell_id == 0 || spec.cells[dst_cell_id].level == parent.level ||
        throw(ArgumentError("fine-to-coarse route destination level mismatch"))
    src_q = _check_d2q9_q(q)
    dst_q = src_q
    step = Int(substep)
    1 <= step <= bank.schedule.ratio ||
        throw(ArgumentError("substep must be inside 1:$(bank.schedule.ratio)"))
    packet_slot_i = Int(route_packet_slot)
    if interface_time_scaling == :level_native && kind == COALESCE_CORNER
        if delayed_same_level_transit
            delayed_dst, delayed_q, delayed_tick =
                _conservative_tree_level_native_delayed_same_level_transit_state_2d(
                    spec, bank.schedule, child_id, src_q, Int(current_tick),
                    dst_cell_id; periodic_x=periodic_x)
            if delayed_dst != 0
                packet = _conservative_tree_f2c_time_factor_2d(
                    bank, interface_time_scaling) *
                    _subcycle_cell_route_packet_2d(
                        F, spec, child_id, src_q, weight; alpha=alpha)
                _push_conservative_tree_delayed_same_level_packet_2d!(
                    bank, delayed_tick, child.level, delayed_dst, delayed_q,
                    packet)
                return bank
            end
        end
        transit_dst, transit_q =
            _conservative_tree_level_native_corner_child_transit_state_2d(
                spec, child_id, src_q, step, bank.schedule.ratio,
                dst_cell_id; periodic_x=periodic_x)
        if transit_dst != 0
            transit_parent_id = spec.cells[transit_dst].parent
            transit_parent_id != 0 ||
                throw(ArgumentError("native child transit destination must have a parent"))
            transit_parent = spec.cells[transit_parent_id]
            transit_parent.level == parent.level ||
                throw(ArgumentError("native child transit parent level mismatch"))
            pair = _conservative_tree_packed_ledger_pair_2d(
                bank, parent.level)
            transit_slot = _conservative_tree_packed_ledger_slot_2d(
                bank, transit_parent_id)
            ix, iy = _conservative_tree_child_index_in_parent_2d(
                transit_parent, spec.cells[transit_dst])
            packet = _conservative_tree_f2c_time_factor_2d(
                bank, interface_time_scaling) *
                _subcycle_cell_route_packet_2d(
                    F, spec, child_id, src_q, weight; alpha=alpha)
            pair.coarse_to_fine[
                ix, iy, transit_q, bank.schedule.ratio, transit_slot] +=
                packet
            return bank
        end
        native_dst, native_q = _conservative_tree_level_native_corner_reflux_state_2d(
            spec, child_id, src_q, step, bank.schedule.ratio, dst_cell_id;
            periodic_x=periodic_x)
        if native_dst != dst_cell_id || native_q != dst_q
            dst_cell_id = native_dst
            dst_q = native_q
            packet_slot_i = 0
        end
    end
    dst_cell_id == 0 || spec.cells[dst_cell_id].level == parent.level ||
        throw(ArgumentError("fine-to-coarse native destination level mismatch"))
    pair = _conservative_tree_packed_ledger_pair_2d(bank, parent.level)
    slot = _conservative_tree_packed_ledger_slot_2d(bank, parent_id)
    packet = _conservative_tree_f2c_time_factor_2d(
        bank, interface_time_scaling) *
        _subcycle_cell_route_packet_2d(
            F, spec, child_id, src_q, weight; alpha=alpha)
    pair.fine_to_coarse[dst_q, step, slot] += packet
    dst_cell_id != 0 ||
        throw(ArgumentError("fine-to-coarse route must have a spatial destination"))
    packet_slot = packet_slot_i
    if packet_slot == 0
        packet_slot = _ensure_conservative_tree_route_packet_cache_2d!(
            bank, parent.level, dst_cell_id, dst_q)
    end
    cache = bank.route_packet_caches[parent.level + 1]
    cache.packets[(packet_slot - 1) * bank.schedule.ratio + step] += packet
    return bank
end

function _conservative_tree_level_native_corner_child_transit_state_2d(
        spec::ConservativeTreeSpec2D,
        src_id::Int,
        q::Int,
        substep::Int,
        ratio::Int,
        fallback_dst::Int;
        periodic_x::Bool=false)
    src = spec.cells[src_id]
    src.active || return 0, q
    src.level > 0 || return 0, q
    remaining_strides = ratio - substep + 1
    remaining_strides > 1 || return 0, q
    parent = spec.cells[src.parent]
    nx_child = _conservative_tree_level_size_2d(spec.Nx, src.level)
    ny_child = _conservative_tree_level_size_2d(spec.Ny, src.level)
    i_cur = src.i
    j_cur = src.j
    q_cur = q

    advanced, i_cur, j_cur, q_cur =
        _conservative_tree_phase_advance_periodic_x_wall_y_2d(
            i_cur, j_cur, q_cur, nx_child, ny_child;
            periodic_x=periodic_x)
    advanced || return 0, q
    first_owner = _active_leaf_covering_sample_2d(
        spec, src.level, i_cur, j_cur)
    first_owner == fallback_dst || return 0, q_cur
    first = spec.cells[first_owner]
    first.active || return 0, q_cur
    first.level == parent.level || return 0, q_cur
    spec.children[first_owner] == (0, 0, 0, 0) || return 0, q_cur

    for _ in 2:remaining_strides
        advanced, i_cur, j_cur, q_cur =
            _conservative_tree_phase_advance_periodic_x_wall_y_2d(
                i_cur, j_cur, q_cur, nx_child, ny_child;
                periodic_x=periodic_x)
        advanced || return 0, q
    end

    dst_id = conservative_tree_cell_id_2d(
        spec, src.level, i_cur, j_cur)
    dst_id == 0 && return 0, q_cur
    dst = spec.cells[dst_id]
    dst.active || return 0, q_cur
    dst.parent != 0 || return 0, q_cur
    dst_parent = spec.cells[dst.parent]
    dst_parent.level == parent.level || return 0, q_cur
    spec.children[dst.parent] != (0, 0, 0, 0) || return 0, q_cur
    return dst_id, q_cur
end

function _conservative_tree_level_native_delayed_same_level_transit_state_2d(
        spec::ConservativeTreeSpec2D,
        schedule::ConservativeTreeSubcycleSchedule2D,
        src_id::Int,
        q::Int,
        current_tick::Int,
        fallback_dst::Int;
        periodic_x::Bool=false)
    src = spec.cells[src_id]
    src.active || return 0, q, 0
    src.level > 0 || return 0, q, 0
    stride = schedule.level_step_ticks[src.level + 1]
    current_tick > 0 || return 0, q, 0
    current_tick % stride == 0 || return 0, q, 0
    nx_level = _conservative_tree_level_size_2d(spec.Nx, src.level)
    ny_level = _conservative_tree_level_size_2d(spec.Ny, src.level)
    i_cur = src.i
    j_cur = src.j
    q_cur = q
    tick = current_tick
    first_stride = true

    while tick <= schedule.finest_ticks
        advanced, i_cur, j_cur, q_cur =
            _conservative_tree_phase_advance_periodic_x_wall_y_2d(
                i_cur, j_cur, q_cur, nx_level, ny_level;
                periodic_x=periodic_x)
        advanced || return 0, q, 0
        owner_id = _active_leaf_covering_sample_2d(
            spec, src.level, i_cur, j_cur)
        owner_id == 0 && return 0, q_cur, 0
        owner = spec.cells[owner_id]
        if owner.level == src.level
            first_stride && return 0, q_cur, 0
            return owner_id, q_cur, tick
        end
        owner.level < src.level || return 0, q_cur, 0
        first_stride && owner_id != fallback_dst && return 0, q_cur, 0
        first_stride = false
        tick += stride
    end
    return 0, q_cur, 0
end

function _push_conservative_tree_delayed_same_level_packet_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D{T},
        tick::Integer,
        level::Integer,
        dst_id::Integer,
        q::Integer,
        packet) where T
    push!(bank.delayed_same_level_ticks, Int(tick))
    push!(bank.delayed_same_level_levels, Int(level))
    push!(bank.delayed_same_level_dst_ids, Int(dst_id))
    push!(bank.delayed_same_level_qs, _check_d2q9_q(q))
    push!(bank.delayed_same_level_packets, T(packet))
    return bank
end

function _conservative_tree_level_native_corner_reflux_state_2d(
        spec::ConservativeTreeSpec2D,
        src_id::Int,
        q::Int,
        substep::Int,
        ratio::Int,
        fallback_dst::Int;
        periodic_x::Bool=false)
    src = spec.cells[src_id]
    src.level > 0 || return fallback_dst, q
    parent = spec.cells[src.parent]
    remaining_strides = ratio - substep + 1
    nx_child = _conservative_tree_level_size_2d(spec.Nx, src.level)
    ny_child = _conservative_tree_level_size_2d(spec.Ny, src.level)
    i_final = src.i
    j_final = src.j
    q_final = q
    for _ in 1:remaining_strides
        trial_i = i_final + d2q9_cx(q_final)
        trial_j = j_final + d2q9_cy(q_final)
        if periodic_x
            trial_i = mod1(trial_i, nx_child)
        elseif !(1 <= trial_i <= nx_child)
            return fallback_dst, q
        end
        if !(1 <= trial_j <= ny_child)
            q_final = d2q9_opposite(q_final)
            continue
        end
        i_final = trial_i
        j_final = trial_j
    end
    dst_i = div(i_final + 1, 2)
    dst_j = div(j_final + 1, 2)
    dst_id = conservative_tree_cell_id_2d(
        spec, parent.level, dst_i, dst_j)
    dst_id == 0 && return fallback_dst, q_final
    spec.cells[dst_id].active || return fallback_dst, q_final
    return dst_id, q_final
end

function _conservative_tree_level_native_corner_reflux_dst_2d(
        spec::ConservativeTreeSpec2D,
        src_id::Int,
        q::Int,
        substep::Int,
        ratio::Int,
        fallback_dst::Int;
        periodic_x::Bool=false)
    dst, _ = _conservative_tree_level_native_corner_reflux_state_2d(
        spec, src_id, q, substep, ratio, fallback_dst;
        periodic_x=periodic_x)
    return dst
end

function _conservative_tree_subcycle_accumulate_fine_to_coarse_packet_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        F::AbstractMatrix,
        src_id::Integer,
        dst_id::Integer,
        q::Integer,
        weight,
        kind::RouteKind,
        substep::Integer,
        route_packet_slot::Integer=0;
        alpha=1,
        interface_time_scaling::Symbol=:leaf_equivalent,
        current_tick::Integer=0,
        delayed_same_level_transit::Bool=true,
        periodic_x::Bool=false)
    _check_conservative_tree_subcycle_spatial_F_2d(F, bank)
    return _conservative_tree_subcycle_accumulate_fine_to_coarse_packet_unchecked_2d!(
        bank, F, src_id, dst_id, q, weight, kind, substep,
        route_packet_slot; alpha=alpha,
        interface_time_scaling=interface_time_scaling,
        current_tick=current_tick,
        delayed_same_level_transit=delayed_same_level_transit,
        periodic_x=periodic_x)
end

function conservative_tree_subcycle_accumulate_fine_to_coarse_route_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        F::AbstractMatrix,
        route::ConservativeTreeRoute2D,
        substep::Integer,
        route_packet_slot::Integer=0;
        alpha=1,
        interface_time_scaling::Symbol=:leaf_equivalent,
        current_tick::Integer=0,
        delayed_same_level_transit::Bool=true,
        periodic_x::Bool=false)
    return _conservative_tree_subcycle_accumulate_fine_to_coarse_packet_2d!(
        bank, F, route.src, route.dst, route.q, route.weight, route.kind,
        substep, route_packet_slot; alpha=alpha,
        interface_time_scaling=interface_time_scaling,
        current_tick=current_tick,
        delayed_same_level_transit=delayed_same_level_transit,
        periodic_x=periodic_x)
end

@inline function _conservative_tree_fast_leaf_f2c_enabled_2d(
        alpha,
        interface_time_scaling::Symbol)
    return alpha == 1 && interface_time_scaling == :leaf_equivalent
end

function _conservative_tree_subcycle_accumulate_compact_f2c_routes_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        event::ConservativeTreeSubcycleEvent2D,
        F::AbstractMatrix)
    parent_level, substep = _check_subcycle_spatial_child_advance_event_2d(
        bank, event)
    child_level = parent_level + 1
    pair = bank.ledger_pairs[parent_level + 1]
    cache = bank.route_packet_caches[parent_level + 1]
    srcs = bank.f2c_route_srcs_by_child_level[child_level + 1]
    qs = bank.f2c_route_qs_by_child_level[child_level + 1]
    weights = bank.f2c_route_weights_by_child_level[child_level + 1]
    parent_slots = bank.f2c_route_parent_slots_by_child_level[child_level + 1]
    packet_slots = bank.f2c_route_packet_slots_by_child_level[child_level + 1]
    factor = _conservative_tree_f2c_time_factor_2d(bank, :leaf_equivalent)
    ratio = bank.schedule.ratio
    @inbounds for idx in eachindex(srcs)
        q = qs[idx]
        packet = factor * weights[idx] * F[srcs[idx], q]
        pair.fine_to_coarse[q, substep, parent_slots[idx]] += packet
        cache.packets[(packet_slots[idx] - 1) * ratio + substep] += packet
    end
    return bank
end

function _conservative_tree_subcycle_accumulate_compact_inactive_f2c_routes_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        event::ConservativeTreeSubcycleEvent2D,
        F::AbstractMatrix)
    parent_level, substep = _check_subcycle_spatial_child_advance_event_2d(
        bank, event)
    child_level = parent_level + 1
    pair = bank.ledger_pairs[parent_level + 1]
    cache = bank.route_packet_caches[parent_level + 1]
    srcs = bank.inactive_f2c_route_srcs_by_child_level[child_level + 1]
    qs = bank.inactive_f2c_route_qs_by_child_level[child_level + 1]
    parent_slots =
        bank.inactive_f2c_route_parent_slots_by_child_level[child_level + 1]
    packet_slots =
        bank.inactive_f2c_route_packet_slots_by_child_level[child_level + 1]
    factor = _conservative_tree_f2c_time_factor_2d(bank, :leaf_equivalent)
    ratio = bank.schedule.ratio
    @inbounds for idx in eachindex(srcs)
        q = qs[idx]
        packet = factor * F[srcs[idx], q]
        pair.fine_to_coarse[q, substep, parent_slots[idx]] += packet
        cache.packets[(packet_slots[idx] - 1) * ratio + substep] += packet
    end
    return bank
end

function conservative_tree_subcycle_accumulate_advance_routes_F_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        event::ConservativeTreeSubcycleEvent2D,
        F::AbstractMatrix,
        table::ConservativeTreeRouteTable2D;
        alpha=1,
        interface_time_scaling::Symbol=:leaf_equivalent,
        delayed_same_level_transit::Bool=true,
        periodic_x::Bool=false)
    parent_level, substep = _check_subcycle_spatial_child_advance_event_2d(
        bank, event)
    _check_conservative_tree_subcycle_route_table_2d(table)
    _check_conservative_tree_subcycle_spatial_F_2d(F, bank)
    if !_conservative_tree_subcycle_route_packet_cache_ready_2d(
            bank, table; periodic_x=periodic_x)
        prepare_conservative_tree_subcycle_route_packet_cache_2d!(
            bank, table; periodic_x=periodic_x)
    end
    if _conservative_tree_fast_leaf_f2c_enabled_2d(
            alpha, interface_time_scaling)
        return _conservative_tree_subcycle_accumulate_compact_f2c_routes_2d!(
            bank, event, F)
    end
    child_level = parent_level + 1

    @inbounds for route_pos in table.coalesce_route_ranges_by_child_level[child_level + 1]
        route_id = table.interface_routes[route_pos]
        route = table.routes[route_id]
        packet_slot = bank.route_packet_slot_by_route[route_id]
        _conservative_tree_subcycle_accumulate_fine_to_coarse_packet_unchecked_2d!(
            bank, F, route.src, route.dst, route.q, route.weight, route.kind,
            substep, packet_slot; alpha=alpha,
            interface_time_scaling=interface_time_scaling,
            current_tick=event.tick,
            delayed_same_level_transit=delayed_same_level_transit,
            periodic_x=periodic_x)
    end
    return bank
end

function _conservative_tree_inactive_parent_coalesce_route_spec_2d(
        spec::ConservativeTreeSpec2D,
        src_id::Int,
        q::Int;
        periodic_x::Bool=false)
    src = spec.cells[src_id]
    src.level > 0 || return 0, DIRECT
    src.active && return 0, DIRECT
    spec.children[src_id] == (0, 0, 0, 0) && return 0, DIRECT

    i_dst = src.i + d2q9_cx(q)
    j_dst = src.j + d2q9_cy(q)
    nx_level = _conservative_tree_level_size_2d(spec.Nx, src.level)
    ny_level = _conservative_tree_level_size_2d(spec.Ny, src.level)
    if periodic_x
        i_dst = mod1(i_dst, nx_level)
    elseif !(1 <= i_dst <= nx_level)
        return 0, DIRECT
    end
    1 <= j_dst <= ny_level || return 0, DIRECT

    parent = spec.cells[src.parent]
    in_parent = (2 * parent.i - 1 <= i_dst <= 2 * parent.i) &&
                (2 * parent.j - 1 <= j_dst <= 2 * parent.j)
    in_parent && return 0, DIRECT

    dst_i = div(i_dst + 1, 2)
    dst_j = div(j_dst + 1, 2)
    dst_id = conservative_tree_cell_id_2d(
        spec, parent.level, dst_i, dst_j)
    dst_id == 0 && return 0, DIRECT
    kind = (i_dst < 2 * parent.i - 1 || i_dst > 2 * parent.i) &&
           (j_dst < 2 * parent.j - 1 || j_dst > 2 * parent.j) ?
           COALESCE_CORNER : COALESCE_FACE
    return dst_id, kind
end

function _conservative_tree_inactive_parent_coalesce_dst_2d(
        spec::ConservativeTreeSpec2D,
        src_id::Int,
        q::Int;
        periodic_x::Bool=false)
    dst_id, _ = _conservative_tree_inactive_parent_coalesce_route_spec_2d(
        spec, src_id, q; periodic_x=periodic_x)
    return dst_id
end

function _conservative_tree_inactive_parent_coalesce_routes_2d(
        spec::ConservativeTreeSpec2D,
        src_id::Int,
        q::Int;
        periodic_x::Bool=false)
    dst_id, kind = _conservative_tree_inactive_parent_coalesce_route_spec_2d(
        spec, src_id, q; periodic_x=periodic_x)
    dst_id == 0 && return ConservativeTreeRoute2D[]
    return [ConservativeTreeRoute2D(src_id, dst_id, q, 1.0, kind)]
end

function conservative_tree_subcycle_accumulate_inactive_parent_routes_F_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        event::ConservativeTreeSubcycleEvent2D,
        F::AbstractMatrix;
        alpha=1,
        interface_time_scaling::Symbol=:leaf_equivalent,
        delayed_same_level_transit::Bool=true,
        periodic_x::Bool=false)
    parent_level, substep = _check_subcycle_spatial_child_advance_event_2d(
        bank, event)
    child_level = parent_level + 1
    _check_conservative_tree_subcycle_spatial_F_2d(F, bank)
    if _conservative_tree_fast_leaf_f2c_enabled_2d(
            alpha, interface_time_scaling)
        return _conservative_tree_subcycle_accumulate_compact_inactive_f2c_routes_2d!(
            bank, event, F)
    end

    inactive_ids = bank.inactive_refined_ids_by_level[child_level + 1]
    inactive_slots = bank.inactive_route_packet_slots_by_level[child_level + 1]
    inactive_dsts = bank.inactive_route_packet_dsts_by_level[child_level + 1]
    inactive_kinds = bank.inactive_route_packet_kinds_by_level[child_level + 1]
    @inbounds for (local_idx, src_id) in enumerate(inactive_ids)
        bank.spec.cells[src_id].active && continue
        for q in 1:9
            packet_slot = inactive_slots[local_idx, q]
            packet_slot == 0 && continue
            dst_id = inactive_dsts[local_idx, q]
            dst_id == 0 && continue
            kind = inactive_kinds[local_idx, q]
            _conservative_tree_subcycle_accumulate_fine_to_coarse_packet_unchecked_2d!(
                bank, F, src_id, dst_id, q, 1.0, kind, substep,
                packet_slot; alpha=alpha,
                interface_time_scaling=interface_time_scaling,
                current_tick=event.tick,
                delayed_same_level_transit=delayed_same_level_transit,
                periodic_x=periodic_x)
        end
    end
    return bank
end

function conservative_tree_subcycle_apply_delayed_same_level_packets_F_2d!(
        F::AbstractMatrix,
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        event::ConservativeTreeSubcycleEvent2D)
    event.phase == :advance || return F
    _check_conservative_tree_subcycle_spatial_F_2d(F, bank)
    level = event.src_level
    tick = event.tick
    @inbounds for idx in eachindex(bank.delayed_same_level_packets)
        bank.delayed_same_level_ticks[idx] == tick || continue
        bank.delayed_same_level_levels[idx] == level || continue
        packet = bank.delayed_same_level_packets[idx]
        iszero(packet) && continue
        F[bank.delayed_same_level_dst_ids[idx],
          bank.delayed_same_level_qs[idx]] += packet
        bank.delayed_same_level_packets[idx] = zero(packet)
    end
    return F
end

function conservative_tree_subcycle_apply_fine_to_coarse_F_2d!(
        F::AbstractMatrix,
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        parent_cell_id::Integer;
        boundary::Symbol=:skip,
        u_south=0,
        u_north=0,
        rho_wall=1)
    _check_conservative_tree_subcycle_spatial_F_2d(F, bank)
    policy = _check_conservative_tree_boundary_policy_2d(boundary)
    parent_id = Int(parent_cell_id)
    parent = bank.spec.cells[parent_id]
    pair = _conservative_tree_packed_ledger_pair_2d(bank, parent.level)
    slot = _conservative_tree_packed_ledger_slot_2d(bank, parent_id)

    @inbounds for q in 1:9
        packet = zero(eltype(F))
        for substep in 1:bank.schedule.ratio
            packet += pair.fine_to_coarse[q, substep, slot]
        end
        iszero(packet) && continue

        dst_id = conservative_tree_cell_id_2d(
            bank.spec, parent.level, parent.i + d2q9_cx(q),
            parent.j + d2q9_cy(q))
        if dst_id == 0
            if policy != :skip
                reflected = packet
                if policy == :periodic_x_moving_wall_y
                    cy = d2q9_cy(q)
                    cy != 0 ||
                        throw(ArgumentError("periodic-x wall-y boundary policy received an x-boundary route; rebuild the route table with periodic_x=true"))
                    wall_u = cy < 0 ? u_south : u_north
                    reflected += _moving_wall_delta(parent.metrics.volume,
                                                    rho_wall, wall_u,
                                                    d2q9_opposite(q))
                elseif policy == :periodic_x_wall_y
                    d2q9_cy(q) != 0 ||
                        throw(ArgumentError("periodic-x wall-y boundary policy received an x-boundary route; rebuild the route table with periodic_x=true"))
                end
                F[parent_id, d2q9_opposite(q)] += reflected
            end
        else
            F[dst_id, q] += packet
        end
    end
    return F
end

function conservative_tree_subcycle_apply_fine_to_coarse_pair_F_2d!(
        F::AbstractMatrix,
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        parent_level::Integer;
        boundary::Symbol=:skip,
        u_south=0,
        u_north=0,
        rho_wall=1)
    _check_conservative_tree_subcycle_spatial_F_2d(F, bank)
    parent = _check_conservative_tree_pair_level_2d(
        bank.schedule, parent_level)
    for parent_id in bank.refined_parent_ids_by_level[parent + 1]
        conservative_tree_subcycle_apply_fine_to_coarse_F_2d!(
            F, bank, parent_id; boundary=boundary, u_south=u_south,
            u_north=u_north, rho_wall=rho_wall)
    end
    return F
end

function conservative_tree_subcycle_apply_sync_up_F_2d!(
        F::AbstractMatrix,
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        event::ConservativeTreeSubcycleEvent2D;
        boundary::Symbol=:skip,
        u_south=0,
        u_north=0,
        rho_wall=1)
    parent_level = _check_subcycle_spatial_sync_up_event_2d(bank, event)
    cache = bank.route_packet_caches[parent_level + 1]
    if !isempty(cache.dst_ids)
        _check_conservative_tree_subcycle_spatial_F_2d(F, bank)
        @inbounds for slot in eachindex(cache.dst_ids)
            packet = zero(eltype(F))
            offset = (slot - 1) * bank.schedule.ratio
            for substep in 1:bank.schedule.ratio
                packet += cache.packets[offset + substep]
            end
            iszero(packet) && continue
            dst_id = cache.dst_ids[slot]
            q = cache.qs[slot]
            F[dst_id, q] += packet
        end
        return F
    end
    return conservative_tree_subcycle_apply_fine_to_coarse_pair_F_2d!(
        F, bank, parent_level; boundary=boundary, u_south=u_south,
        u_north=u_north, rho_wall=rho_wall)
end

function conservative_tree_subcycle_accumulate_advance_face_2d!(
        bank::ConservativeTreeSubcycleLedgerBank2D,
        event::ConservativeTreeSubcycleEvent2D,
        fine_half_step::AbstractArray{<:Any,3},
        q::Integer,
        face::Symbol)
    parent, substep = _check_subcycle_child_advance_event_2d(
        bank.schedule, event)
    ledger = conservative_tree_subcycle_pair_ledger_2d(bank, parent)
    conservative_tree_subcycle_accumulate_fine_to_coarse_face_2d!(
        ledger, fine_half_step, q, face, substep)
    return ledger
end

function conservative_tree_subcycle_accumulate_advance_corner_2d!(
        bank::ConservativeTreeSubcycleLedgerBank2D,
        event::ConservativeTreeSubcycleEvent2D,
        fine_half_step::AbstractArray{<:Any,3},
        q::Integer,
        corner::Symbol)
    parent, substep = _check_subcycle_child_advance_event_2d(
        bank.schedule, event)
    ledger = conservative_tree_subcycle_pair_ledger_2d(bank, parent)
    conservative_tree_subcycle_accumulate_fine_to_coarse_corner_2d!(
        ledger, fine_half_step, q, corner, substep)
    return ledger
end

function conservative_tree_subcycle_sync_up_ledger_2d(
        bank::ConservativeTreeSubcycleLedgerBank2D,
        event::ConservativeTreeSubcycleEvent2D)
    parent = _check_subcycle_sync_up_event_2d(bank.schedule, event)
    return conservative_tree_subcycle_pair_ledger_2d(bank, parent)
end

function conservative_tree_subcycle_accumulate_fine_to_coarse_face_2d!(
        ledger::ConservativeTreeSubcycleLedger2D,
        fine_half_step::AbstractArray{<:Any,3},
        q::Integer,
        face::Symbol,
        substep::Integer)
    _check_child_block_2d(fine_half_step, "fine_half_step")
    qi = _check_d2q9_q(Int(q))
    step = _check_subcycle_step_2d(ledger, substep)
    ledger.fine_to_coarse[qi, step] +=
        coalesce_fine_to_coarse_face_F(fine_half_step, qi, face)
    return ledger
end

function conservative_tree_subcycle_accumulate_fine_to_coarse_corner_2d!(
        ledger::ConservativeTreeSubcycleLedger2D,
        fine_half_step::AbstractArray{<:Any,3},
        q::Integer,
        corner::Symbol,
        substep::Integer)
    _check_child_block_2d(fine_half_step, "fine_half_step")
    qi = _check_d2q9_q(Int(q))
    step = _check_subcycle_step_2d(ledger, substep)
    ledger.fine_to_coarse[qi, step] +=
        coalesce_fine_to_coarse_corner_F(fine_half_step, qi, corner)
    return ledger
end
