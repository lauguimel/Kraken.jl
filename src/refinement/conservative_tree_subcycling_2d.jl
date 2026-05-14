include("subcycling_schedule_2d.jl")



































function prepare_conservative_tree_subcycle_route_packet_cache_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        table::ConservativeTreeRouteTable2D;
        periodic_x::Bool=false)
    _check_conservative_tree_subcycle_route_table_2d(table)
    resize!(bank.route_packet_slot_by_route, length(table.routes))
    fill!(bank.route_packet_slot_by_route, 0)
    resize!(bank.inactive_route_packet_slots_by_level,
            bank.spec.max_level + 1)
    resize!(bank.inactive_route_packet_dsts_by_level,
            bank.spec.max_level + 1)
    resize!(bank.inactive_route_packet_kinds_by_level,
            bank.spec.max_level + 1)
    @inbounds for level in 0:bank.spec.max_level
        empty!(bank.f2c_route_srcs_by_child_level[level + 1])
        empty!(bank.f2c_route_qs_by_child_level[level + 1])
        empty!(bank.f2c_route_weights_by_child_level[level + 1])
        empty!(bank.f2c_route_parent_slots_by_child_level[level + 1])
        empty!(bank.f2c_route_packet_slots_by_child_level[level + 1])
        empty!(bank.inactive_f2c_route_srcs_by_child_level[level + 1])
        empty!(bank.inactive_f2c_route_qs_by_child_level[level + 1])
        empty!(bank.inactive_f2c_route_parent_slots_by_child_level[level + 1])
        empty!(bank.inactive_f2c_route_packet_slots_by_child_level[level + 1])
        ids = bank.inactive_refined_ids_by_level[level + 1]
        slots = isassigned(bank.inactive_route_packet_slots_by_level,
                           level + 1) ?
                bank.inactive_route_packet_slots_by_level[level + 1] :
                zeros(Int, 0, 9)
        if size(slots, 1) != length(ids) || size(slots, 2) != 9
            slots = zeros(Int, length(ids), 9)
        else
            fill!(slots, 0)
        end
        bank.inactive_route_packet_slots_by_level[level + 1] = slots
        dsts = isassigned(bank.inactive_route_packet_dsts_by_level,
                          level + 1) ?
               bank.inactive_route_packet_dsts_by_level[level + 1] :
               zeros(Int, 0, 9)
        if size(dsts, 1) != length(ids) || size(dsts, 2) != 9
            dsts = zeros(Int, length(ids), 9)
        else
            fill!(dsts, 0)
        end
        bank.inactive_route_packet_dsts_by_level[level + 1] = dsts
        kinds = isassigned(bank.inactive_route_packet_kinds_by_level,
                           level + 1) ?
                bank.inactive_route_packet_kinds_by_level[level + 1] :
                fill(DIRECT, 0, 9)
        if size(kinds, 1) != length(ids) || size(kinds, 2) != 9
            kinds = fill(DIRECT, length(ids), 9)
        else
            fill!(kinds, DIRECT)
        end
        bank.inactive_route_packet_kinds_by_level[level + 1] = kinds
    end

    @inbounds for route_id in table.interface_routes
        route = table.routes[route_id]
        route.kind == COALESCE_FACE || route.kind == COALESCE_CORNER || continue
        route.dst == 0 && continue
        child = bank.spec.cells[route.src]
        child.level > 0 || continue
        parent = bank.spec.cells[child.parent]
        slot = _ensure_conservative_tree_route_packet_cache_2d!(
            bank, parent.level, route.dst, route.q)
        bank.route_packet_slot_by_route[route_id] = slot
        push!(bank.f2c_route_srcs_by_child_level[child.level + 1],
              route.src)
        push!(bank.f2c_route_qs_by_child_level[child.level + 1],
              route.q)
        push!(bank.f2c_route_weights_by_child_level[child.level + 1],
              route.weight)
        push!(bank.f2c_route_parent_slots_by_child_level[child.level + 1],
              bank.parent_ledger_slot[child.parent])
        push!(bank.f2c_route_packet_slots_by_child_level[child.level + 1],
              slot)
    end
    @inbounds for parent_level in 0:(bank.spec.max_level - 1)
        child_level = parent_level + 1
        inactive_ids = bank.inactive_refined_ids_by_level[child_level + 1]
        inactive_slots = bank.inactive_route_packet_slots_by_level[child_level + 1]
        inactive_dsts = bank.inactive_route_packet_dsts_by_level[child_level + 1]
        inactive_kinds = bank.inactive_route_packet_kinds_by_level[child_level + 1]
        for (local_idx, src_id) in enumerate(inactive_ids)
            src = bank.spec.cells[src_id]
            src.active && continue
            for q in 1:9
                dst_id, kind = _conservative_tree_inactive_parent_coalesce_route_spec_2d(
                    bank.spec, src_id, q; periodic_x=periodic_x)
                dst_id == 0 && continue
                inactive_dsts[local_idx, q] = dst_id
                inactive_kinds[local_idx, q] = kind
                inactive_slots[local_idx, q] =
                    _ensure_conservative_tree_route_packet_cache_2d!(
                    bank, parent_level, dst_id, q)
                push!(bank.inactive_f2c_route_srcs_by_child_level[child_level + 1],
                      src_id)
                push!(bank.inactive_f2c_route_qs_by_child_level[child_level + 1],
                      q)
                push!(bank.inactive_f2c_route_parent_slots_by_child_level[child_level + 1],
                      bank.parent_ledger_slot[src.parent])
                push!(bank.inactive_f2c_route_packet_slots_by_child_level[child_level + 1],
                      inactive_slots[local_idx, q])
            end
        end
    end
    bank.route_packet_cache_valid[1] = true
    bank.route_packet_cache_route_objectid[1] = objectid(table.routes)
    bank.route_packet_cache_periodic_x[1] = periodic_x
    return bank
end
function _conservative_tree_subcycle_route_packet_cache_ready_2d(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        table::ConservativeTreeRouteTable2D;
        periodic_x::Bool=false)
    return bank.route_packet_cache_valid[1] &&
           bank.route_packet_cache_route_objectid[1] == objectid(table.routes) &&
           length(bank.route_packet_slot_by_route) == length(table.routes) &&
           bank.route_packet_cache_periodic_x[1] == periodic_x
end

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





function _check_conservative_tree_leaf_solid_mask_2d(
        spec::ConservativeTreeSpec2D,
        is_solid::AbstractArray{Bool,2})
    leaf_nx = _conservative_tree_level_size_2d(spec.Nx, spec.max_level)
    leaf_ny = _conservative_tree_level_size_2d(spec.Ny, spec.max_level)
    size(is_solid) == (leaf_nx, leaf_ny) ||
        throw(ArgumentError("is_solid must match the finest leaf-equivalent grid"))
    return is_solid
end

@inline function _conservative_tree_cell_leaf_bounds_2d(
        spec::ConservativeTreeSpec2D,
        cell::ConservativeTreeCell2D)
    scale = 1 << (spec.max_level - cell.level)
    i0 = (cell.i - 1) * scale + 1
    i1 = cell.i * scale
    j0 = (cell.j - 1) * scale + 1
    j1 = cell.j * scale
    return i0, i1, j0, j1
end

function _conservative_tree_cell_solid_status_2d(
        spec::ConservativeTreeSpec2D,
        cell::ConservativeTreeCell2D,
        is_solid::AbstractArray{Bool,2})
    i0, i1, j0, j1 = _conservative_tree_cell_leaf_bounds_2d(spec, cell)
    any_solid = false
    all_solid = true
    @inbounds for j in j0:j1, i in i0:i1
        solid = is_solid[i, j]
        any_solid |= solid
        all_solid &= solid
    end
    all_solid && return :solid
    any_solid && return :partial
    return :fluid
end

function _conservative_tree_cell_is_solid_2d(
        spec::ConservativeTreeSpec2D,
        cell::ConservativeTreeCell2D,
        is_solid::Union{Nothing,AbstractArray{Bool,2}})
    is_solid === nothing && return false
    status = _conservative_tree_cell_solid_status_2d(spec, cell, is_solid)
    status == :partial &&
        throw(ArgumentError("active AMR-D solid cells must be fully resolved by refinement"))
    return status == :solid
end

function validate_conservative_tree_solid_mask_resolved_2d(
        spec::ConservativeTreeSpec2D,
        table::ConservativeTreeRouteTable2D,
        is_solid::AbstractArray{Bool,2})
    _check_conservative_tree_leaf_solid_mask_2d(spec, is_solid)
    @inbounds for cell_id in spec.active_cells
        status = _conservative_tree_cell_solid_status_2d(
            spec, spec.cells[cell_id], is_solid)
        status == :partial &&
            throw(ArgumentError("AMR-D solid mask cuts active cell $cell_id; refine the solid band"))
    end
    @inbounds for route_id in table.interface_routes
        route = table.routes[route_id]
        for cell_id in (route.src, route.dst)
            cell_id == 0 && continue
            status = _conservative_tree_cell_solid_status_2d(
                spec, spec.cells[cell_id], is_solid)
            status == :fluid && continue
            throw(ArgumentError("AMR-D solid mask touches an interface route; keep refinement interfaces away from solids"))
        end
    end
    return is_solid
end








@inline function _subcycle_route_packet_2d(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        route::ConservativeTreeRoute2D;
        alpha=1)
    return _subcycle_cell_route_packet_2d(
        F, spec, route.src, route.q, route.weight; alpha=alpha)
end

@inline function _subcycle_cell_route_packet_2d(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        src_id::Integer,
        q::Integer,
        weight;
        alpha=1)
    T = typeof(zero(eltype(F)) + weight + alpha)
    a = T(alpha)
    qi = _check_d2q9_q(q)
    src = Int(src_id)
    if a == one(a)
        return T(weight) * T(F[src, qi])
    end
    cell = spec.cells[src]
    return reconstructed_integrated_D2Q9_packet(
        @view(F[src, :]), cell.metrics.volume, qi, weight; alpha=alpha)
end

@inline function _check_conservative_tree_coarse_to_fine_prolongation_2d(
        prolongation::Symbol)
    prolongation in (:flat, :limited_linear) ||
        throw(ArgumentError("coarse_to_fine_prolongation must be :flat or :limited_linear"))
    return prolongation
end

@inline function _conservative_tree_same_level_Fq_2d(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        level::Int,
        i::Int,
        j::Int,
        q::Int;
        periodic_x::Bool=false)
    nx = _conservative_tree_level_size_2d(spec.Nx, level)
    ny = _conservative_tree_level_size_2d(spec.Ny, level)
    ii = periodic_x ? mod1(i, nx) : i
    1 <= ii <= nx && 1 <= j <= ny ||
        return false, zero(eltype(F))
    cell_id = conservative_tree_cell_id_2d(spec, level, ii, j)
    cell_id == 0 && return false, zero(eltype(F))
    return true, F[cell_id, q]
end

function _conservative_tree_limited_same_level_slope_x_2d(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        src_id::Int,
        q::Int;
        periodic_x::Bool=false)
    src = spec.cells[src_id]
    center = F[src_id, q]
    has_left, left_value = _conservative_tree_same_level_Fq_2d(
        F, spec, src.level, src.i - 1, src.j, q; periodic_x=periodic_x)
    has_right, right_value = _conservative_tree_same_level_Fq_2d(
        F, spec, src.level, src.i + 1, src.j, q; periodic_x=periodic_x)
    if has_left && has_right
        return _minmod(center - left_value, right_value - center)
    elseif has_left
        return center - left_value
    elseif has_right
        return right_value - center
    end
    return zero(center)
end

function _conservative_tree_limited_same_level_slope_y_2d(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        src_id::Int,
        q::Int)
    src = spec.cells[src_id]
    center = F[src_id, q]
    has_south, south_value = _conservative_tree_same_level_Fq_2d(
        F, spec, src.level, src.i, src.j - 1, q)
    has_north, north_value = _conservative_tree_same_level_Fq_2d(
        F, spec, src.level, src.i, src.j + 1, q)
    if has_south && has_north
        return _minmod(center - south_value, north_value - center)
    elseif has_south
        return center - south_value
    elseif has_north
        return north_value - center
    end
    return zero(center)
end

function _conservative_tree_limited_linear_child_packet_2d(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        src_id::Int,
        q::Int,
        si::Int,
        sj::Int,
        scale::Int;
        periodic_x::Bool=false)
    center = F[src_id, q]
    sx = _conservative_tree_limited_same_level_slope_x_2d(
        F, spec, src_id, q; periodic_x=periodic_x)
    sy = _conservative_tree_limited_same_level_slope_y_2d(
        F, spec, src_id, q)

    area = inv(typeof(center)(scale * scale))
    max_offset = typeof(center)(scale - 1) / typeof(center)(2 * scale)
    max_delta = (abs(sx) + abs(sy)) * max_offset * area
    base = center * area
    if max_delta > zero(max_delta) && base < max_delta
        theta = base / max_delta
        sx *= theta
        sy *= theta
    end

    xoff = (typeof(center)(si) - (typeof(center)(scale) + one(center)) / 2) /
        typeof(center)(scale)
    yoff = (typeof(center)(sj) - (typeof(center)(scale) + one(center)) / 2) /
        typeof(center)(scale)
    return base + (sx * xoff + sy * yoff) * area
end

function _conservative_tree_limited_linear_sampled_route_packet_2d(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        route::ConservativeTreeRoute2D;
        periodic_x::Bool=false)
    src = spec.cells[route.src]
    sample_level = spec.max_level
    src.level < sample_level ||
        throw(ArgumentError("limited-linear sampled route source is already finest"))
    q = _check_d2q9_q(route.q)
    cx = d2q9_cx(q)
    cy = d2q9_cy(q)
    scale = 1 << (sample_level - src.level)
    nx_sample = _conservative_tree_level_size_2d(spec.Nx, sample_level)
    packet = zero(eltype(F))

    @inbounds for sj in 1:scale, si in 1:scale
        sample_i = (src.i - 1) * scale + si + cx
        sample_j = (src.j - 1) * scale + sj + cy
        if periodic_x
            sample_i = mod1(sample_i, nx_sample)
        end
        dst_id = _active_leaf_covering_sample_2d(
            spec, sample_level, sample_i, sample_j)
        dst_id == route.dst || continue
        kind = _route_kind_for_level_pair_2d(src, spec.cells[dst_id], q)
        kind == route.kind || continue
        packet += _conservative_tree_limited_linear_child_packet_2d(
            F, spec, route.src, q, si, sj, scale; periodic_x=periodic_x)
    end
    return packet
end

function _subcycle_coarse_to_fine_route_packet_2d(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        route::ConservativeTreeRoute2D;
        alpha=1,
        coarse_to_fine_prolongation::Symbol=:flat,
        periodic_x::Bool=false)
    mode = _check_conservative_tree_coarse_to_fine_prolongation_2d(
        coarse_to_fine_prolongation)
    if mode == :limited_linear
        a = typeof(zero(eltype(F)) + alpha)(alpha)
        a == one(a) ||
            throw(ArgumentError("limited-linear coarse-to-fine prolongation currently requires alpha_c2f = 1"))
        return _conservative_tree_limited_linear_sampled_route_packet_2d(
            F, spec, route; periodic_x=periodic_x)
    end
    return _subcycle_route_packet_2d(F, spec, route; alpha=alpha)
end

@inline function _check_conservative_tree_coarse_to_fine_state_2d(
        coarse_to_fine_state::Symbol)
    coarse_to_fine_state in (:owned, :postcollision) ||
        throw(ArgumentError("coarse_to_fine_state must be :owned or :postcollision"))
    return coarse_to_fine_state
end

@inline function _check_conservative_tree_coarse_to_fine_predictor_weight_2d(
        weight)
    w = Float64(weight)
    0 <= w <= 1 ||
        throw(ArgumentError("coarse_to_fine_predictor_weight must be in [0, 1]"))
    return weight
end

@inline function _conservative_tree_periodic_x_policy_2d(policy::Symbol)
    return policy in (:periodic_x_wall_y, :periodic_x_moving_wall_y)
end

@inline function _check_conservative_tree_interface_time_scaling_2d(
        scaling::Symbol)
    scaling in (:leaf_equivalent, :level_native) ||
        throw(ArgumentError("interface_time_scaling must be :leaf_equivalent or :level_native"))
    return scaling
end

@inline function _conservative_tree_c2f_time_factor_2d(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        scaling::Symbol)
    mode = _check_conservative_tree_interface_time_scaling_2d(scaling)
    return mode == :leaf_equivalent ? bank.schedule.ratio : 1
end

@inline function _conservative_tree_f2c_time_factor_2d(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D{T},
        scaling::Symbol) where T
    mode = _check_conservative_tree_interface_time_scaling_2d(scaling)
    return mode == :leaf_equivalent ? inv(T(bank.schedule.ratio)) : one(T)
end

function conservative_tree_subcycle_deposit_coarse_to_fine_route_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        F::AbstractMatrix,
        route::ConservativeTreeRoute2D;
        alpha=1,
        interface_time_scaling::Symbol=:leaf_equivalent,
        coarse_to_fine_prolongation::Symbol=:flat,
        periodic_x::Bool=false)
    _check_conservative_tree_subcycle_spatial_F_2d(F, bank)
    route.kind == SPLIT_FACE || route.kind == SPLIT_CORNER ||
        throw(ArgumentError("route must be a coarse-to-fine split route"))
    route.dst != 0 ||
        throw(ArgumentError("coarse-to-fine split route must have a child destination"))

    spec = bank.spec
    src = spec.cells[route.src]
    child = spec.cells[route.dst]
    child.level == src.level + 1 ||
        throw(ArgumentError("coarse-to-fine route must cross one level"))
    parent_id = child.parent
    parent = spec.cells[parent_id]
    parent.level == src.level ||
        throw(ArgumentError("coarse-to-fine route destination parent level mismatch"))
    pair = _conservative_tree_packed_ledger_pair_2d(bank, parent.level)
    slot = _conservative_tree_packed_ledger_slot_2d(bank, parent_id)
    ix, iy = _conservative_tree_child_index_in_parent_2d(parent, child)
    qi = _check_d2q9_q(route.q)
    packet = _conservative_tree_c2f_time_factor_2d(
        bank, interface_time_scaling) *
        _subcycle_coarse_to_fine_route_packet_2d(
            F, spec, route; alpha=alpha,
            coarse_to_fine_prolongation=coarse_to_fine_prolongation,
            periodic_x=periodic_x)

    @inbounds for substep in 1:bank.schedule.ratio
        pair.coarse_to_fine[ix, iy, qi, substep, slot] +=
            packet / bank.schedule.ratio
    end
    return bank
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


function conservative_tree_subcycle_sync_down_level_native_phase_routes_F_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        event::ConservativeTreeSubcycleEvent2D,
        F::AbstractMatrix;
        periodic_x::Bool=false)
    parent_level = _check_subcycle_spatial_sync_down_event_2d(bank, event)
    _check_conservative_tree_subcycle_spatial_F_2d(F, bank)
    spec = bank.spec
    pair = _conservative_tree_packed_ledger_pair_2d(bank, parent_level)
    child_level = parent_level + 1
    ratio = bank.schedule.ratio
    nx_child = _conservative_tree_level_size_2d(spec.Nx, child_level)
    ny_child = _conservative_tree_level_size_2d(spec.Ny, child_level)

    @inbounds for src_id in spec.active_cells
        src = spec.cells[src_id]
        src.level == parent_level || continue
        for q in 1:9
            cx = d2q9_cx(q)
            cy = d2q9_cy(q)
            cx == 0 && cy == 0 && continue
            for sj in 1:ratio, si in 1:ratio
                pos_i = (src.i - 1) * ratio + si
                pos_j = (src.j - 1) * ratio + sj
                qcur = q
                packet = _conservative_tree_limited_linear_child_packet_2d(
                    F, spec, src_id, q, si, sj, ratio;
                    periodic_x=periodic_x)
                for substep in 1:ratio
                    advanced, dst_i, dst_j, qnext =
                        _conservative_tree_phase_advance_periodic_x_wall_y_2d(
                            pos_i, pos_j, qcur, nx_child, ny_child;
                            periodic_x=periodic_x)
                    advanced || break
                    pos_i = dst_i
                    pos_j = dst_j
                    qcur = qnext
                    dst_id = conservative_tree_cell_id_2d(
                        spec, child_level, dst_i, dst_j)
                    dst_id == 0 && continue
                    child = spec.cells[dst_id]
                    parent_id = child.parent
                    parent_id == 0 && continue
                    spec.cells[parent_id].level == parent_level || continue
                    spec.children[parent_id] == (0, 0, 0, 0) && continue
                    parent_id == src_id && continue
                    slot = _conservative_tree_packed_ledger_slot_2d(
                        bank, parent_id)
                    ix, iy = _conservative_tree_child_index_in_parent_2d(
                        spec.cells[parent_id], child)
                    pair.coarse_to_fine[ix, iy, qcur, substep, slot] += packet
                    break
                end
            end
        end
    end
    return bank
end

function _conservative_tree_level_native_phase_c2f_safe_2d(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        parent_level::Int)
    return true
end

function conservative_tree_subcycle_sync_down_routes_F_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        event::ConservativeTreeSubcycleEvent2D,
        F::AbstractMatrix,
        table::ConservativeTreeRouteTable2D;
        alpha=1,
        interface_time_scaling::Symbol=:leaf_equivalent,
        coarse_to_fine_prolongation::Symbol=:flat,
        periodic_x::Bool=false,
        phase_resolved_level_native::Bool=true)
    parent_level = _check_subcycle_spatial_sync_down_event_2d(bank, event)
    _check_conservative_tree_subcycle_route_table_2d(table)
    _check_conservative_tree_subcycle_spatial_F_2d(F, bank)
    if phase_resolved_level_native &&
       interface_time_scaling == :level_native &&
       coarse_to_fine_prolongation == :flat &&
       alpha == 1 &&
       _conservative_tree_level_native_phase_c2f_safe_2d(bank, parent_level)
        return conservative_tree_subcycle_sync_down_level_native_phase_routes_F_2d!(
            bank, event, F; periodic_x=periodic_x)
    end

    @inbounds for route_pos in table.split_route_ranges_by_parent_level[parent_level + 1]
        route_id = table.interface_routes[route_pos]
        route = table.routes[route_id]
        conservative_tree_subcycle_deposit_coarse_to_fine_route_2d!(
            bank, F, route; alpha=alpha,
            interface_time_scaling=interface_time_scaling,
            coarse_to_fine_prolongation=coarse_to_fine_prolongation,
            periodic_x=periodic_x)
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

function conservative_tree_subcycle_apply_coarse_to_fine_F_2d!(
        F::AbstractMatrix,
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        parent_cell_id::Integer,
        substep::Integer)
    _check_conservative_tree_subcycle_spatial_F_2d(F, bank)
    parent_id = Int(parent_cell_id)
    parent = bank.spec.cells[parent_id]
    pair = _conservative_tree_packed_ledger_pair_2d(bank, parent.level)
    slot = _conservative_tree_packed_ledger_slot_2d(bank, parent_id)
    step = Int(substep)
    1 <= step <= bank.schedule.ratio ||
        throw(ArgumentError("substep must be inside 1:$(bank.schedule.ratio)"))
    children = bank.spec.children[parent_id]
    children == (0, 0, 0, 0) &&
        throw(ArgumentError("parent_cell_id does not identify a refined parent"))

    @inbounds for iy in 1:2, ix in 1:2
        child_id = children[_conservative_tree_child_slot_2d(ix, iy)]
        child_id == 0 && continue
        for q in 1:9
            F[child_id, q] += pair.coarse_to_fine[ix, iy, q, step, slot]
        end
    end
    return F
end

function conservative_tree_subcycle_apply_coarse_to_fine_pair_F_2d!(
        F::AbstractMatrix,
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        parent_level::Integer,
        substep::Integer)
    _check_conservative_tree_subcycle_spatial_F_2d(F, bank)
    parent = _check_conservative_tree_pair_level_2d(
        bank.schedule, parent_level)
    for parent_id in bank.refined_parent_ids_by_level[parent + 1]
        conservative_tree_subcycle_apply_coarse_to_fine_F_2d!(
            F, bank, parent_id, substep)
    end
    return F
end

function conservative_tree_subcycle_apply_child_advance_injection_F_2d!(
        F::AbstractMatrix,
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        event::ConservativeTreeSubcycleEvent2D)
    parent_level, substep = _check_subcycle_spatial_child_advance_event_2d(
        bank, event)
    return conservative_tree_subcycle_apply_coarse_to_fine_pair_F_2d!(
        F, bank, parent_level, substep)
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

function _stream_conservative_tree_flat_direct_level_routes_F_2d!(
        Fout::AbstractMatrix,
        Fin::AbstractMatrix,
        table::ConservativeTreeRouteTable2D,
        level::Int)
    srcs = table.direct_route_srcs_by_level[level + 1]
    dsts = table.direct_route_dsts_by_level[level + 1]
    qs = table.direct_route_qs_by_level[level + 1]
    weights = table.direct_route_weights_by_level[level + 1]
    if table.direct_route_unique_dsts_by_level[level + 1] &&
            Threads.nthreads() > 1 && length(srcs) >= 4096
        Threads.@threads for idx in eachindex(srcs)
            @inbounds begin
                src_id = srcs[idx]
                q = qs[idx]
                Fout[dsts[idx], q] += weights[idx] * Fin[src_id, q]
            end
        end
    else
        @inbounds for idx in eachindex(srcs)
            src_id = srcs[idx]
            q = qs[idx]
            Fout[dsts[idx], q] += weights[idx] * Fin[src_id, q]
        end
    end
    return Fout
end

function _stream_conservative_tree_native_direct_level_routes_F_2d!(
        Fout::AbstractMatrix,
        Fin::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        level::Int;
        periodic_x::Bool=false)
    @inbounds for cell_id in spec.active_cells
        src = spec.cells[cell_id]
        src.level == level || continue
        nx = _conservative_tree_level_size_2d(spec.Nx, level)
        ny = _conservative_tree_level_size_2d(spec.Ny, level)
        for q in 1:9
            i_dst = src.i + d2q9_cx(q)
            j_dst = src.j + d2q9_cy(q)
            if periodic_x
                i_dst = mod1(i_dst, nx)
            end
            1 <= i_dst <= nx && 1 <= j_dst <= ny || continue
            dst_id = conservative_tree_cell_id_2d(
                spec, level, i_dst, j_dst)
            dst_id == 0 && continue
            spec.cells[dst_id].active || continue
            Fout[dst_id, q] += Fin[cell_id, q]
        end
    end
    return Fout
end

function _conservative_tree_child_position_enters_refined_parent_2d(
        spec::ConservativeTreeSpec2D,
        child_level::Int,
        i::Int,
        j::Int,
        parent_level::Int;
        periodic_x::Bool=false)
    nx_child = _conservative_tree_level_size_2d(spec.Nx, child_level)
    ny_child = _conservative_tree_level_size_2d(spec.Ny, child_level)
    ii = periodic_x ? mod1(i, nx_child) : i
    1 <= ii <= nx_child && 1 <= j <= ny_child || return false
    child_id = conservative_tree_cell_id_2d(spec, child_level, ii, j)
    child_id == 0 && return false
    child = spec.cells[child_id]
    parent_id = child.parent
    parent_id == 0 && return false
    parent = spec.cells[parent_id]
    parent.level == parent_level || return false
    return spec.children[parent_id] != (0, 0, 0, 0)
end

function _stream_conservative_tree_level_native_phase_direct_level_routes_F_2d!(
        Fout::AbstractMatrix,
        Fin::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        level::Int;
        periodic_x::Bool=false,
        ratio::Int=2)
    child_level = level + 1
    child_level <= spec.max_level || return Fout
    nx_child = _conservative_tree_level_size_2d(spec.Nx, child_level)
    ny_child = _conservative_tree_level_size_2d(spec.Ny, child_level)

    @inbounds for cell_id in spec.active_cells
        src = spec.cells[cell_id]
        src.level == level || continue
        for q in 1:9
            for sj in 1:ratio, si in 1:ratio
                pos_i = (src.i - 1) * ratio + si
                pos_j = (src.j - 1) * ratio + sj
                qcur = q
                enters_fine = false
                alive = true
                for substep in 1:ratio
                    advanced, trial_i, trial_j, qnext =
                        _conservative_tree_phase_advance_periodic_x_wall_y_2d(
                            pos_i, pos_j, qcur, nx_child, ny_child;
                            periodic_x=periodic_x)
                    if !advanced
                        alive = false
                        break
                    end
                    pos_i = trial_i
                    pos_j = trial_j
                    qcur = qnext
                    if _conservative_tree_child_position_enters_refined_parent_2d(
                            spec, child_level, pos_i, pos_j, level;
                            periodic_x=periodic_x)
                        enters_fine = true
                        break
                    end
                end
                alive || continue
                enters_fine && continue
                1 <= pos_i <= nx_child && 1 <= pos_j <= ny_child ||
                    continue
                dst_id = _active_leaf_covering_sample_2d(
                    spec, child_level, pos_i, pos_j)
                dst_id == 0 && continue
                spec.cells[dst_id].level == level || continue
                packet = _conservative_tree_limited_linear_child_packet_2d(
                    Fin, spec, cell_id, q, si, sj, ratio;
                    periodic_x=periodic_x)
                Fout[dst_id, qcur] += packet
            end
        end
    end
    return Fout
end

function _stream_conservative_tree_compact_boundary_level_routes_F_2d!(
        Fout::AbstractMatrix,
        Fin::AbstractMatrix,
        table::ConservativeTreeRouteTable2D,
        level::Int,
        policy::Symbol;
        u_south=0,
        u_north=0,
        rho_wall=1)
    policy == :skip && return Fout
    srcs = table.boundary_route_srcs_by_level[level + 1]
    qs = table.boundary_route_qs_by_level[level + 1]
    opps = table.boundary_route_opposite_qs_by_level[level + 1]
    cys = table.boundary_route_cys_by_level[level + 1]
    weights = table.boundary_route_weights_by_level[level + 1]
    volumes = table.boundary_route_volumes_by_level[level + 1]
    if Threads.nthreads() > 1 && length(srcs) >= 4096
        Threads.@threads for idx in eachindex(srcs)
            @inbounds begin
                src_id = srcs[idx]
                q = qs[idx]
                opp = opps[idx]
                cy = cys[idx]
                if policy == :periodic_x_wall_y ||
                   policy == :periodic_x_moving_wall_y
                    cy != 0 ||
                        throw(ArgumentError("periodic-x wall-y boundary policy received an x-boundary route; rebuild the route table with periodic_x=true"))
                end
                packet = weights[idx] * Fin[src_id, q]
                if policy == :periodic_x_moving_wall_y
                    wall_u = cy < 0 ? u_south : u_north
                    packet += weights[idx] * _moving_wall_delta(
                        volumes[idx], rho_wall, wall_u, opp)
                end
                Fout[src_id, opp] += packet
            end
        end
    else
        @inbounds for idx in eachindex(srcs)
            src_id = srcs[idx]
            q = qs[idx]
            opp = opps[idx]
            cy = cys[idx]
            if policy == :periodic_x_wall_y ||
               policy == :periodic_x_moving_wall_y
                cy != 0 ||
                    throw(ArgumentError("periodic-x wall-y boundary policy received an x-boundary route; rebuild the route table with periodic_x=true"))
            end
            packet = weights[idx] * Fin[src_id, q]
            if policy == :periodic_x_moving_wall_y
                wall_u = cy < 0 ? u_south : u_north
                packet += weights[idx] * _moving_wall_delta(
                    volumes[idx], rho_wall, wall_u, opp)
            end
            Fout[src_id, opp] += packet
        end
    end
    return Fout
end

function _conservative_tree_source_q_can_touch_y_wall_2d(
        spec::ConservativeTreeSpec2D,
        src_id::Int,
        q::Int,
        ticks::Int=1 << spec.max_level)
    cy = d2q9_cy(q)
    cy == 0 && return false
    src = spec.cells[src_id]
    _, _, j0, j1 = _conservative_tree_cell_leaf_bounds_2d(spec, src)
    leaf_ny = _conservative_tree_level_size_2d(spec.Ny, spec.max_level)
    return cy < 0 ? j0 <= ticks : j1 > leaf_ny - ticks
end

function _conservative_tree_source_q_touches_wall_and_interface_2d(
        spec::ConservativeTreeSpec2D,
        src_id::Int,
        q::Int,
        ticks::Int=1 << spec.max_level;
        periodic_x::Bool=false)
    _conservative_tree_source_q_can_touch_y_wall_2d(
        spec, src_id, q, ticks) || return false
    src = spec.cells[src_id]
    i0, i1, j0, j1 = _conservative_tree_cell_leaf_bounds_2d(spec, src)
    nx_leaf = _conservative_tree_level_size_2d(spec.Nx, spec.max_level)
    ny_leaf = _conservative_tree_level_size_2d(spec.Ny, spec.max_level)
    @inbounds for leaf_j in j0:j1, leaf_i in i0:i1
        pos_i = leaf_i
        pos_j = leaf_j
        qcur = q
        touched_wall = false
        changed_level = false
        owner_id = _active_leaf_covering_sample_2d(
            spec, spec.max_level, pos_i, pos_j)
        for _ in 1:ticks
            advanced, next_i, next_j, next_q =
                _conservative_tree_phase_advance_periodic_x_wall_y_2d(
                    pos_i, pos_j, qcur, nx_leaf, ny_leaf;
                    periodic_x=periodic_x)
            advanced || break
            if next_i == pos_i && next_j == pos_j && next_q != qcur
                touched_wall = true
            end
            dst_id = _active_leaf_covering_sample_2d(
                spec, spec.max_level, next_i, next_j)
            if owner_id != 0 && dst_id != 0 &&
               spec.cells[owner_id].level != spec.cells[dst_id].level
                changed_level = true
            end
            touched_wall && changed_level && return true
            pos_i = next_i
            pos_j = next_j
            qcur = next_q
            owner_id = dst_id
        end
    end
    return false
end

function _conservative_tree_wall_phase_owner_advance_tick_2d(
        schedule::ConservativeTreeSubcycleSchedule2D,
        level::Int,
        leaf_tick::Int)
    level_ticks = schedule.level_step_ticks[level + 1]
    return cld(leaf_tick, level_ticks) * level_ticks
end

function _conservative_tree_wall_phase_interval_start_tick_2d(
        schedule::ConservativeTreeSubcycleSchedule2D,
        level::Int,
        leaf_tick::Int)
    level_ticks = schedule.level_step_ticks[level + 1]
    return div(leaf_tick - 1, level_ticks) * level_ticks
end

function _conservative_tree_level_coord_from_leaf_2d(
        spec::ConservativeTreeSpec2D,
        level::Int,
        leaf_i::Int,
        leaf_j::Int)
    scale = 1 << (spec.max_level - level)
    return cld(leaf_i, scale), cld(leaf_j, scale)
end

function _push_conservative_tree_wall_phase_event_mask_2d!(
        seen::Set{Tuple{Symbol,Int,Int,Int,Int}},
        phases::Vector{Symbol},
        ticks::Vector{Int},
        levels::Vector{Int},
        src_ids::Vector{Int},
        src_qs::Vector{Int},
        phase::Symbol,
        tick::Int,
        level::Int,
        src_id::Int,
        q::Int)
    key = (phase, tick, level, src_id, q)
    key in seen && return nothing
    push!(seen, key)
    push!(phases, phase)
    push!(ticks, tick)
    push!(levels, level)
    push!(src_ids, src_id)
    push!(src_qs, q)
    return nothing
end

function _append_conservative_tree_wall_phase_one_tick_scatter_2d!(
        spec::ConservativeTreeSpec2D,
        schedule::ConservativeTreeSubcycleSchedule2D,
        src_id::Int,
        q::Int,
        leaf_tick::Int,
        event_mask_seen::Set{Tuple{Symbol,Int,Int,Int,Int}},
        event_mask_phases::Vector{Symbol},
        event_mask_ticks::Vector{Int},
        event_mask_levels::Vector{Int},
        event_mask_src_ids::Vector{Int},
        event_mask_src_qs::Vector{Int},
        sync_ticks::Vector{Int},
        sync_parent_levels::Vector{Int},
        sync_child_substeps::Vector{Int},
        sync_src_ids::Vector{Int},
        sync_src_qs::Vector{Int},
        sync_parent_ids::Vector{Int},
        sync_ixs::Vector{Int},
        sync_iys::Vector{Int},
        sync_dst_qs::Vector{Int},
        sync_weights::Vector{T},
        advance_ticks::Vector{Int},
        advance_levels::Vector{Int},
        advance_src_ids::Vector{Int},
        advance_src_qs::Vector{Int},
        advance_dst_ids::Vector{Int},
        advance_dst_qs::Vector{Int},
        advance_weights::Vector{T},
        reflux_ticks::Vector{Int},
        reflux_levels::Vector{Int},
        reflux_src_ids::Vector{Int},
        reflux_src_qs::Vector{Int},
        reflux_dst_ids::Vector{Int},
        reflux_dst_qs::Vector{Int},
        reflux_weights::Vector{T};
        periodic_x::Bool=false) where {T}
    src = spec.cells[src_id]
    level = src.level
    advance_tick = _conservative_tree_wall_phase_owner_advance_tick_2d(
        schedule, level, leaf_tick)

    if level == spec.max_level
        nx = _conservative_tree_level_size_2d(spec.Nx, level)
        ny = _conservative_tree_level_size_2d(spec.Ny, level)
        advanced, next_i, next_j, next_q =
            _conservative_tree_phase_advance_periodic_x_wall_y_2d(
                src.i, src.j, q, nx, ny; periodic_x=periodic_x)
        advanced || return nothing
        dst_id = _active_leaf_covering_sample_2d(
            spec, level, next_i, next_j)
        dst_id == 0 && return nothing
        dst = spec.cells[dst_id]
        _push_conservative_tree_wall_phase_event_mask_2d!(
            event_mask_seen, event_mask_phases, event_mask_ticks,
            event_mask_levels, event_mask_src_ids, event_mask_src_qs,
            :advance, advance_tick, level, src_id, q)
        if dst.level < level
            push!(reflux_ticks, advance_tick)
            push!(reflux_levels, level)
            push!(reflux_src_ids, src_id)
            push!(reflux_src_qs, q)
            push!(reflux_dst_ids, dst_id)
            push!(reflux_dst_qs, next_q)
            push!(reflux_weights, one(T))
        elseif dst.level == level
            push!(advance_ticks, advance_tick)
            push!(advance_levels, level)
            push!(advance_src_ids, src_id)
            push!(advance_src_qs, q)
            push!(advance_dst_ids, dst_id)
            push!(advance_dst_qs, next_q)
            push!(advance_weights, one(T))
        end
        return nothing
    end

    ratio = 2
    child_level = level + 1
    nx_child = _conservative_tree_level_size_2d(spec.Nx, child_level)
    ny_child = _conservative_tree_level_size_2d(spec.Ny, child_level)
    weight = one(T) / T(ratio * ratio)
    sync_tick = _conservative_tree_wall_phase_interval_start_tick_2d(
        schedule, level, leaf_tick)

    @inbounds for sj in 1:ratio, si in 1:ratio
        pos_i = (src.i - 1) * ratio + si
        pos_j = (src.j - 1) * ratio + sj
        qcur = q
        alive = true
        entered_fine = false
        for substep in 1:ratio
            advanced, next_i, next_j, next_q =
                _conservative_tree_phase_advance_periodic_x_wall_y_2d(
                    pos_i, pos_j, qcur, nx_child, ny_child;
                    periodic_x=periodic_x)
            if !advanced
                alive = false
                break
            end
            pos_i = next_i
            pos_j = next_j
            qcur = next_q
            child_id = conservative_tree_cell_id_2d(
                spec, child_level, pos_i, pos_j)
            child_id == 0 && continue
            parent_id = spec.cells[child_id].parent
            parent_id == 0 && continue
            parent = spec.cells[parent_id]
            if parent.level == level &&
               spec.children[parent_id] != (0, 0, 0, 0) &&
               parent_id != src_id
                _push_conservative_tree_wall_phase_event_mask_2d!(
                    event_mask_seen, event_mask_phases, event_mask_ticks,
                    event_mask_levels, event_mask_src_ids, event_mask_src_qs,
                    :sync_down, sync_tick, level, src_id, q)
                ix, iy = _conservative_tree_child_index_in_parent_2d(
                    parent, spec.cells[child_id])
                push!(sync_ticks, sync_tick)
                push!(sync_parent_levels, level)
                push!(sync_child_substeps, substep)
                push!(sync_src_ids, src_id)
                push!(sync_src_qs, q)
                push!(sync_parent_ids, parent_id)
                push!(sync_ixs, ix)
                push!(sync_iys, iy)
                push!(sync_dst_qs, qcur)
                push!(sync_weights, weight)
                entered_fine = true
                break
            end
        end
        alive || continue
        entered_fine && continue
        dst_id = _active_leaf_covering_sample_2d(
            spec, child_level, pos_i, pos_j)
        dst_id == 0 && continue
        dst = spec.cells[dst_id]
        _push_conservative_tree_wall_phase_event_mask_2d!(
            event_mask_seen, event_mask_phases, event_mask_ticks,
            event_mask_levels, event_mask_src_ids, event_mask_src_qs,
            :advance, advance_tick, level, src_id, q)
        if dst.level < level
            push!(reflux_ticks, advance_tick)
            push!(reflux_levels, level)
            push!(reflux_src_ids, src_id)
            push!(reflux_src_qs, q)
            push!(reflux_dst_ids, dst_id)
            push!(reflux_dst_qs, qcur)
            push!(reflux_weights, weight)
        elseif dst.level == level
            push!(advance_ticks, advance_tick)
            push!(advance_levels, level)
            push!(advance_src_ids, src_id)
            push!(advance_src_qs, q)
            push!(advance_dst_ids, dst_id)
            push!(advance_dst_qs, qcur)
            push!(advance_weights, weight)
        end
    end
    return nothing
end

function _conservative_tree_scatter_leaf_trace_packet_2d!(
        Fout::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        src_id::Int,
        q::Int,
        value;
        periodic_x::Bool=false,
        ticks::Int=1 << spec.max_level)
    src = spec.cells[src_id]
    i0, i1, j0, j1 = _conservative_tree_cell_leaf_bounds_2d(spec, src)
    nsub = (i1 - i0 + 1) * (j1 - j0 + 1)
    packet = value / nsub
    nx_leaf = _conservative_tree_level_size_2d(spec.Nx, spec.max_level)
    ny_leaf = _conservative_tree_level_size_2d(spec.Ny, spec.max_level)
    touched_wall = false

    @inbounds for j in j0:j1, i in i0:i1
        pos_i = i
        pos_j = j
        qcur = q
        for _ in 1:ticks
            advanced, next_i, next_j, next_q =
                _conservative_tree_phase_advance_periodic_x_wall_y_2d(
                    pos_i, pos_j, qcur, nx_leaf, ny_leaf;
                    periodic_x=periodic_x)
            advanced || break
            if next_i == pos_i && next_j == pos_j && next_q != qcur
                touched_wall = true
            end
            pos_i = next_i
            pos_j = next_j
            qcur = next_q
        end
        dst_id = _active_leaf_covering_sample_2d(
            spec, spec.max_level, pos_i, pos_j)
        dst_id == 0 && continue
        Fout[dst_id, qcur] += packet
    end
    return touched_wall
end

function _prepare_conservative_tree_wall_phase_scatter_2d(
        spec::ConservativeTreeSpec2D;
        periodic_x::Bool=false,
        ticks::Int=1 << spec.max_level,
        schedule::ConservativeTreeSubcycleSchedule2D=
            create_conservative_tree_subcycle_schedule_2d(spec.max_level),
        T::Type{<:Real}=Float64)
    source_mask = falses(length(spec.cells), 9)
    src_ids = Int[]
    src_qs = Int[]
    dst_ids = Int[]
    dst_qs = Int[]
    weights = T[]
    event_mask_seen = Set{Tuple{Symbol,Int,Int,Int,Int}}()
    event_mask_phases = Symbol[]
    event_mask_ticks = Int[]
    event_mask_levels = Int[]
    event_mask_src_ids = Int[]
    event_mask_src_qs = Int[]
    sync_ticks = Int[]
    sync_parent_levels = Int[]
    sync_child_substeps = Int[]
    sync_src_ids = Int[]
    sync_src_qs = Int[]
    sync_parent_ids = Int[]
    sync_ixs = Int[]
    sync_iys = Int[]
    sync_dst_qs = Int[]
    sync_weights = T[]
    advance_ticks = Int[]
    advance_levels = Int[]
    advance_src_ids = Int[]
    advance_src_qs = Int[]
    advance_dst_ids = Int[]
    advance_dst_qs = Int[]
    advance_weights = T[]
    reflux_ticks = Int[]
    reflux_levels = Int[]
    reflux_src_ids = Int[]
    reflux_src_qs = Int[]
    reflux_dst_ids = Int[]
    reflux_dst_qs = Int[]
    reflux_weights = T[]
    substep_sources_seen = Set{Tuple{Int,Int,Int}}()
    nx_leaf = _conservative_tree_level_size_2d(spec.Nx, spec.max_level)
    ny_leaf = _conservative_tree_level_size_2d(spec.Ny, spec.max_level)

    @inbounds for src_id in spec.active_cells, q in 1:9
        _conservative_tree_source_q_can_touch_y_wall_2d(
            spec, src_id, q, ticks) || continue
        source_mask[src_id, q] = true
        src = spec.cells[src_id]
        i0, i1, j0, j1 = _conservative_tree_cell_leaf_bounds_2d(spec, src)
        nsub = (i1 - i0 + 1) * (j1 - j0 + 1)
        weight = one(T) / T(nsub)
        local_weights = Dict{Tuple{Int,Int},T}()
        for j in j0:j1, i in i0:i1
            pos_i = i
            pos_j = j
            qcur = q
            for _ in 1:ticks
                advanced, next_i, next_j, next_q =
                    _conservative_tree_phase_advance_periodic_x_wall_y_2d(
                        pos_i, pos_j, qcur, nx_leaf, ny_leaf;
                        periodic_x=periodic_x)
                advanced || break
                pos_i = next_i
                pos_j = next_j
                qcur = next_q
            end
            dst_id = _active_leaf_covering_sample_2d(
                spec, spec.max_level, pos_i, pos_j)
            dst_id == 0 && continue
            key = (dst_id, qcur)
            local_weights[key] = get(local_weights, key, zero(T)) + weight
        end
        for ((dst_id, dst_q), dst_weight) in local_weights
            push!(src_ids, src_id)
            push!(src_qs, q)
            push!(dst_ids, dst_id)
            push!(dst_qs, dst_q)
            push!(weights, dst_weight)
        end

        _conservative_tree_source_q_touches_wall_and_interface_2d(
            spec, src_id, q, ticks; periodic_x=periodic_x) || continue
        for j in j0:j1, i in i0:i1
            pos_i = i
            pos_j = j
            qcur = q
            for leaf_tick in 1:ticks
                owner_id = _active_leaf_covering_sample_2d(
                    spec, spec.max_level, pos_i, pos_j)
                owner_id == 0 && break
                owner_level = spec.cells[owner_id].level
                owner_tick =
                    _conservative_tree_wall_phase_owner_advance_tick_2d(
                        schedule, owner_level, leaf_tick)
                key = (owner_tick, owner_id, qcur)
                if !(key in substep_sources_seen)
                    push!(substep_sources_seen, key)
                    _append_conservative_tree_wall_phase_one_tick_scatter_2d!(
                        spec, schedule, owner_id, qcur, leaf_tick,
                        event_mask_seen, event_mask_phases, event_mask_ticks,
                        event_mask_levels, event_mask_src_ids,
                        event_mask_src_qs, sync_ticks, sync_parent_levels,
                        sync_child_substeps, sync_src_ids, sync_src_qs,
                        sync_parent_ids, sync_ixs, sync_iys, sync_dst_qs,
                        sync_weights, advance_ticks, advance_levels,
                        advance_src_ids, advance_src_qs, advance_dst_ids,
                        advance_dst_qs, advance_weights, reflux_ticks,
                        reflux_levels, reflux_src_ids, reflux_src_qs,
                        reflux_dst_ids, reflux_dst_qs, reflux_weights;
                        periodic_x=periodic_x)
                end
                advanced, next_i, next_j, next_q =
                    _conservative_tree_phase_advance_periodic_x_wall_y_2d(
                        pos_i, pos_j, qcur, nx_leaf, ny_leaf;
                        periodic_x=periodic_x)
                advanced || break
                pos_i = next_i
                pos_j = next_j
                qcur = next_q
            end
        end
    end

    return ConservativeTreeWallPhaseScatter2D{T}(
        source_mask, src_ids, src_qs, dst_ids, dst_qs, weights,
        event_mask_phases, event_mask_ticks, event_mask_levels,
        event_mask_src_ids, event_mask_src_qs, sync_ticks,
        sync_parent_levels, sync_child_substeps, sync_src_ids, sync_src_qs,
        sync_parent_ids, sync_ixs, sync_iys, sync_dst_qs, sync_weights,
        advance_ticks, advance_levels, advance_src_ids, advance_src_qs,
        advance_dst_ids, advance_dst_qs, advance_weights, reflux_ticks,
        reflux_levels, reflux_src_ids, reflux_src_qs, reflux_dst_ids,
        reflux_dst_qs, reflux_weights)
end

function _cached_conservative_tree_wall_phase_scatter_2d(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D{T};
        periodic_x::Bool=false) where {T}
    if !isempty(bank.wall_phase_scatter_cache) &&
       !isempty(bank.wall_phase_scatter_cache_periodic_x) &&
       bank.wall_phase_scatter_cache_periodic_x[1] == periodic_x
        return bank.wall_phase_scatter_cache[1]
    end
    scatter = _prepare_conservative_tree_wall_phase_scatter_2d(
        bank.spec; periodic_x=periodic_x, schedule=bank.schedule, T=T)
    empty!(bank.wall_phase_scatter_cache)
    empty!(bank.wall_phase_scatter_cache_periodic_x)
    push!(bank.wall_phase_scatter_cache, scatter)
    push!(bank.wall_phase_scatter_cache_periodic_x, periodic_x)
    return scatter
end

function _mask_conservative_tree_wall_phase_sources_2d!(
        F::AbstractMatrix,
        scatter::ConservativeTreeWallPhaseScatter2D)
    @inbounds for src_id in axes(scatter.source_mask, 1), q in 1:9
        scatter.source_mask[src_id, q] || continue
        F[src_id, q] = zero(eltype(F))
    end
    return F
end

function _apply_conservative_tree_wall_phase_scatter_2d!(
        Fout::AbstractMatrix,
        Fin::AbstractMatrix,
        scatter::ConservativeTreeWallPhaseScatter2D)
    @inbounds for idx in eachindex(scatter.src_ids)
        Fout[scatter.dst_ids[idx], scatter.dst_qs[idx]] +=
            eltype(Fout)(scatter.weights[idx]) *
            Fin[scatter.src_ids[idx], scatter.src_qs[idx]]
    end
    return Fout
end

function _copy_mask_conservative_tree_wall_phase_event_sources_2d!(
        Fmasked::AbstractMatrix,
        Fsource::AbstractMatrix,
        scatter::ConservativeTreeWallPhaseScatter2D,
        phase::Symbol,
        level::Int,
        tick::Int)
    copyto!(Fmasked, Fsource)
    active = false
    @inbounds for idx in eachindex(scatter.event_mask_phases)
        scatter.event_mask_phases[idx] == phase || continue
        scatter.event_mask_ticks[idx] == tick || continue
        scatter.event_mask_levels[idx] == level || continue
        Fmasked[scatter.event_mask_src_ids[idx],
                scatter.event_mask_src_qs[idx]] = zero(eltype(Fmasked))
        active = true
    end
    return active
end

function _apply_conservative_tree_wall_phase_sync_down_scatter_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        event::ConservativeTreeSubcycleEvent2D,
        Fsource::AbstractMatrix,
        scatter::ConservativeTreeWallPhaseScatter2D)
    parent_level = event.src_level
    tick = event.tick
    @inbounds for idx in eachindex(scatter.sync_ticks)
        scatter.sync_ticks[idx] == tick || continue
        scatter.sync_parent_levels[idx] == parent_level || continue
        parent_id = scatter.sync_parent_ids[idx]
        pair = _conservative_tree_packed_ledger_pair_2d(bank, parent_level)
        slot = _conservative_tree_packed_ledger_slot_2d(bank, parent_id)
        packet = eltype(pair.coarse_to_fine)(scatter.sync_weights[idx]) *
                 Fsource[scatter.sync_src_ids[idx], scatter.sync_src_qs[idx]]
        pair.coarse_to_fine[
            scatter.sync_ixs[idx],
            scatter.sync_iys[idx],
            scatter.sync_dst_qs[idx],
            scatter.sync_child_substeps[idx],
            slot] += packet
    end
    return bank
end

function _apply_conservative_tree_wall_phase_advance_scatter_2d!(
        Fout::AbstractMatrix,
        Fsource::AbstractMatrix,
        scatter::ConservativeTreeWallPhaseScatter2D,
        level::Int,
        tick::Int)
    @inbounds for idx in eachindex(scatter.advance_ticks)
        scatter.advance_ticks[idx] == tick || continue
        scatter.advance_levels[idx] == level || continue
        Fout[scatter.advance_dst_ids[idx], scatter.advance_dst_qs[idx]] +=
            eltype(Fout)(scatter.advance_weights[idx]) *
            Fsource[scatter.advance_src_ids[idx], scatter.advance_src_qs[idx]]
    end
    return Fout
end

function _apply_conservative_tree_wall_phase_reflux_scatter_2d!(
        state_bank::ConservativeTreeSubcycleBufferBank2D,
        Fsource::AbstractMatrix,
        scatter::ConservativeTreeWallPhaseScatter2D,
        level::Int,
        tick::Int)
    reflux = state_bank.levels[1].reflux_to_coarse
    @inbounds for idx in eachindex(scatter.reflux_ticks)
        scatter.reflux_ticks[idx] == tick || continue
        scatter.reflux_levels[idx] == level || continue
        reflux[scatter.reflux_dst_ids[idx], scatter.reflux_dst_qs[idx]] +=
            eltype(reflux)(scatter.reflux_weights[idx]) *
            Fsource[scatter.reflux_src_ids[idx], scatter.reflux_src_qs[idx]]
    end
    return state_bank
end

function _conservative_tree_wall_phase_prestream_sources_unchanged_2d!(
        state_bank::ConservativeTreeSubcycleBufferBank2D,
        Fsource::AbstractMatrix,
        Fscratch::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        schedule::ConservativeTreeSubcycleSchedule2D,
        pre_stream_level!,
        scatter::ConservativeTreeWallPhaseScatter2D)
    pre_stream_level! === nothing && return true
    advance_events = Vector{Union{Nothing,ConservativeTreeSubcycleEvent2D}}(
        nothing, spec.max_level + 1)
    for event in schedule.events
        event.phase == :advance || continue
        advance_events[event.src_level + 1] === nothing || continue
        advance_events[event.src_level + 1] = event
    end

    @inbounds for level in 0:spec.max_level
        event = advance_events[level + 1]
        event === nothing && return false
        fill!(Fsource, zero(eltype(Fsource)))
        _copy_conservative_tree_level_rows_2d!(
            Fsource, state_bank.levels[level + 1].owned,
            state_bank.active_ids_by_level[level + 1])
        conservative_tree_subcycle_apply_restriction_to_inactive_level_F_2d!(
            Fsource, state_bank, level)
        copyto!(Fscratch, Fsource)
        has_wall_source = false
        for cell_id in axes(scatter.source_mask, 1), q in 1:9
            scatter.source_mask[cell_id, q] || continue
            spec.cells[cell_id].level == level || continue
            has_wall_source = true
            delta = eltype(Fsource)(
                1.0e-7 * (1 + mod(cell_id + 3 * q + 5 * level, 7)))
            Fsource[cell_id, q] += delta
            Fscratch[cell_id, q] += delta
        end
        has_wall_source || continue
        pre_stream_level!(Fscratch, spec, level, event)
        for cell_id in axes(scatter.source_mask, 1), q in 1:9
            scatter.source_mask[cell_id, q] || continue
            spec.cells[cell_id].level == level || continue
            before = Fsource[cell_id, q]
            after = Fscratch[cell_id, q]
            tol = 32 * eps(Float64) *
                  max(one(Float64), abs(Float64(before)),
                      abs(Float64(after)))
            abs(Float64(after - before)) <= tol || return false
        end
    end
    return true
end

function _conservative_tree_prestream_state_unchanged_2d!(
        state_bank::ConservativeTreeSubcycleBufferBank2D,
        Fsource::AbstractMatrix,
        Fscratch::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        schedule::ConservativeTreeSubcycleSchedule2D,
        pre_stream_level!)
    pre_stream_level! === nothing && return true
    advance_events = Vector{Union{Nothing,ConservativeTreeSubcycleEvent2D}}(
        nothing, spec.max_level + 1)
    for event in schedule.events
        event.phase == :advance || continue
        advance_events[event.src_level + 1] === nothing || continue
        advance_events[event.src_level + 1] = event
    end

    @inbounds for level in 0:spec.max_level
        event = advance_events[level + 1]
        event === nothing && return false
        fill!(Fsource, zero(eltype(Fsource)))
        _copy_conservative_tree_level_rows_2d!(
            Fsource, state_bank.levels[level + 1].owned,
            state_bank.active_ids_by_level[level + 1])
        conservative_tree_subcycle_apply_restriction_to_inactive_level_F_2d!(
            Fsource, state_bank, level)
        copyto!(Fscratch, Fsource)
        pre_stream_level!(Fscratch, spec, level, event)
        for (cell_id, cell) in pairs(spec.cells)
            cell.level == level || continue
            for q in 1:9
                before = Fsource[cell_id, q]
                after = Fscratch[cell_id, q]
                tol = 32 * eps(Float64) *
                      max(one(Float64), abs(Float64(before)),
                          abs(Float64(after)))
                abs(Float64(after - before)) <= tol || return false
            end
        end
    end
    return true
end

function _conservative_tree_apply_prestream_over_schedule_2d!(
        Fpost::AbstractMatrix,
        Fin::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        schedule::ConservativeTreeSubcycleSchedule2D,
        pre_stream_level!)
    copyto!(Fpost, Fin)
    pre_stream_level! === nothing && return Fpost
    for event in schedule.events
        event.phase == :advance || continue
        pre_stream_level!(Fpost, spec, event.src_level, event)
    end
    return Fpost
end

function _stream_conservative_tree_level_native_wall_phase_transport_F_2d!(
        Fout::AbstractMatrix,
        Fin::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        table::ConservativeTreeRouteTable2D;
        boundary::Symbol=:periodic_x_wall_y,
        u_south=0,
        u_north=0,
        rho_wall=1,
        alpha_c2f=1,
        alpha_f2c=1,
        coarse_to_fine_state::Symbol=:owned,
        pre_stream_level! = nothing,
        schedule=nothing,
        route_bank=nothing,
        state_bank=nothing,
        Fsource=nothing,
        Fscratch=nothing,
        scatter=nothing,
        scatter_source=nothing)
    periodic_x = true
    scatter_run = scatter === nothing ?
        (route_bank === nothing ?
            (schedule === nothing ?
                _prepare_conservative_tree_wall_phase_scatter_2d(
                    spec; periodic_x=periodic_x, T=eltype(Fout)) :
                _prepare_conservative_tree_wall_phase_scatter_2d(
                    spec; periodic_x=periodic_x, schedule=schedule,
                    T=eltype(Fout))) :
            _cached_conservative_tree_wall_phase_scatter_2d(
                route_bank; periodic_x=periodic_x)) :
        scatter
    Fmasked = copy(Fin)
    _mask_conservative_tree_wall_phase_sources_2d!(Fmasked, scatter_run)

    stream_conservative_tree_subcycled_buffered_routes_F_2d!(
        Fout, Fmasked, spec, table; boundary=boundary,
        u_south=u_south, u_north=u_north, rho_wall=rho_wall,
        alpha_c2f=alpha_c2f, alpha_f2c=alpha_f2c,
        interface_time_scaling=:level_native,
        coarse_to_fine_prolongation=:flat,
        coarse_to_fine_state=coarse_to_fine_state,
        coarse_to_fine_predictor_weight=0,
        pre_stream_level! = pre_stream_level!,
        schedule=schedule, route_bank=route_bank, state_bank=state_bank,
        Fsource=Fsource, Fscratch=Fscratch,
        wall_phase_transport_correction=false,
        level_native_delayed_same_level_transit=
            pre_stream_level! === nothing && scatter_source === nothing)

    return _apply_conservative_tree_wall_phase_scatter_2d!(
        Fout, scatter_source === nothing ? Fin : scatter_source, scatter_run)
end

# WallPhaseScatter2D implementation contract:
# - The scatter is a substep-keyed route family, independent of
#   phase_level_source_valid and of the coarse-to-fine predictor gate.
# - Covered (src_id, q) entries must opt out of direct, boundary, C2F, F2C, and
#   inactive-parent route families before any replacement packet is injected.
# - All scatter reads consume F[src_id, q] after the level-local
#   pre_stream_level! hook, at both sync_down and advance event sites.
# - sync_down(parent_level): parent-owned source packets that enter a refined
#   child during the child interval are deposited in pair.coarse_to_fine at the
#   exact child substep and child slot.
# - advance(level): post pre_stream_level! source packets that remain on the
#   same active level are added to Fscratch_run. Packets landing on a coarser
#   owner must not bypass the level-native fine-to-coarse/corner redirection:
#   v38-v39 route-family replacement audits show that direct reflux deposits
#   over-replace the native F2C family at wall-terminating xfaces.
# - advance(level): packets landing in a finer owner are illegal here unless
#   represented by the earlier sync_down table for that parent interval.
# - Scatter deposits are always additive: Fdst[dst_id, dst_q] += packet.
# - periodic-x wrapping is part of the precomputed leaf trace, including the
#   small Nx cases where wraparound changes the integer residual factor.
function _stream_conservative_tree_direct_level_routes_F_2d!(
        Fout::AbstractMatrix,
        Fin::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        table::ConservativeTreeRouteTable2D,
        level::Int,
        policy::Symbol;
        u_south=0,
        u_north=0,
        rho_wall=1,
        is_solid=nothing,
        coarse_to_fine_prolongation::Symbol=:flat,
        periodic_x::Bool=false,
        phase_resolved_level_native::Bool=false,
        subcycle_ratio::Int=2)
    _check_conservative_tree_stream_args_2d(Fout, Fin, spec)
    prolongation = _check_conservative_tree_coarse_to_fine_prolongation_2d(
        coarse_to_fine_prolongation)

    if phase_resolved_level_native && is_solid === nothing &&
       prolongation != :limited_linear && level < spec.max_level
        _stream_conservative_tree_level_native_phase_direct_level_routes_F_2d!(
            Fout, Fin, spec, level; periodic_x=periodic_x,
            ratio=subcycle_ratio)
    elseif is_solid === nothing && prolongation != :limited_linear
        _stream_conservative_tree_flat_direct_level_routes_F_2d!(
            Fout, Fin, table, level)
    elseif is_solid === nothing
        @inbounds for route_pos in table.direct_route_ranges_by_level[level + 1]
            route_id = table.direct_routes[route_pos]
            route = table.routes[route_id]
            src_cell = spec.cells[route.src]
            packet = if prolongation == :limited_linear &&
                        route.weight < 1.0 &&
                        src_cell.level < spec.max_level &&
                        table.source_q_has_split_route[route.src, route.q]
                _conservative_tree_limited_linear_sampled_route_packet_2d(
                    Fin, spec, route; periodic_x=periodic_x)
            else
                route.weight * Fin[route.src, route.q]
            end
            Fout[route.dst, route.q] += packet
        end
    else
        @inbounds for route_pos in table.direct_route_ranges_by_level[level + 1]
            route_id = table.direct_routes[route_pos]
            route = table.routes[route_id]
            src_cell = spec.cells[route.src]
            _conservative_tree_cell_is_solid_2d(spec, src_cell, is_solid) &&
                continue
            packet = if prolongation == :limited_linear &&
                        route.weight < 1.0 &&
                        src_cell.level < spec.max_level &&
                        table.source_q_has_split_route[route.src, route.q]
                _conservative_tree_limited_linear_sampled_route_packet_2d(
                    Fin, spec, route; periodic_x=periodic_x)
            else
                route.weight * Fin[route.src, route.q]
            end
            if _conservative_tree_cell_is_solid_2d(
                    spec, spec.cells[route.dst], is_solid)
                Fout[route.src, d2q9_opposite(route.q)] +=
                    packet
            else
                Fout[route.dst, route.q] += packet
            end
        end
    end

    phase_resolved_level_native && is_solid === nothing && return Fout

    if is_solid === nothing
        _stream_conservative_tree_compact_boundary_level_routes_F_2d!(
            Fout, Fin, table, level, policy; u_south=u_south,
            u_north=u_north, rho_wall=rho_wall)
    else
        @inbounds for route_pos in table.boundary_route_ranges_by_level[level + 1]
            route_id = table.boundary_routes[route_pos]
            route = table.routes[route_id]
            src_cell = spec.cells[route.src]
            _conservative_tree_cell_is_solid_2d(spec, src_cell, is_solid) &&
                continue
            if policy != :skip
                Fout[route.src, d2q9_opposite(route.q)] +=
                    _conservative_tree_boundary_reflection_packet_2d(
                        Fin, spec, route, policy; u_south=u_south,
                        u_north=u_north, rho_wall=rho_wall)
            end
        end
    end
    return Fout
end

function _copy_conservative_tree_level_rows_2d!(
        Fdst::AbstractMatrix,
        Fsrc::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        level::Int)
    @inbounds for (cell_id, cell) in pairs(spec.cells)
        cell.level == level || continue
        for q in 1:9
            Fdst[cell_id, q] = Fsrc[cell_id, q]
        end
    end
    return Fdst
end

function _copy_conservative_tree_level_rows_2d!(
        Fdst::AbstractMatrix,
        Fsrc::AbstractMatrix,
        ids::AbstractVector{<:Integer})
    @inbounds for raw_id in ids
        cell_id = Int(raw_id)
        for q in 1:9
            Fdst[cell_id, q] = Fsrc[cell_id, q]
        end
    end
    return Fdst
end

function _add_and_clear_conservative_tree_level_rows_2d!(
        Fdst::AbstractMatrix,
        Fpending::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        level::Int)
    @inbounds for (cell_id, cell) in pairs(spec.cells)
        cell.level == level || continue
        for q in 1:9
            Fdst[cell_id, q] += Fpending[cell_id, q]
            Fpending[cell_id, q] = zero(eltype(Fpending))
        end
    end
    return Fdst
end

function _copy_conservative_tree_active_level_rows_2d!(
        Fdst::AbstractMatrix,
        Fsrc::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        level::Int)
    @inbounds for cell_id in spec.active_cells
        cell = spec.cells[cell_id]
        cell.level == level || continue
        for q in 1:9
            Fdst[cell_id, q] = Fsrc[cell_id, q]
        end
    end
    return Fdst
end

function _copy_conservative_tree_active_level_rows_2d!(
        Fdst::AbstractMatrix,
        Fsrc::AbstractMatrix,
        ids::AbstractVector{<:Integer})
    return _copy_conservative_tree_level_rows_2d!(Fdst, Fsrc, ids)
end

"""
stream_conservative_tree_subcycled_routes_F_2d!(Fout, Fin, spec, table;
                                                    boundary=:skip,
                                                    alpha_c2f=1,
                                                    alpha_f2c=1)

CPU reference transport for the recursive AMR-D subcycle schedule. Same-level
routes are advanced only when their level receives an `:advance` event.
Coarse/fine interface routes are routed through spatial ledgers: split routes
are deposited on `:sync_down`, injected into child rows during each child
advance, and coalesce routes are accumulated during child advances then applied
on `:sync_up`.

This is still a transport skeleton: no collision, forcing, or physical open
boundary closure is performed here.

`alpha_c2f` and `alpha_f2c` rescale the non-equilibrium part of interface
packets. The default `1` preserves the original packet transport exactly.
"""
function stream_conservative_tree_subcycled_routes_F_2d!(
        Fout::AbstractMatrix,
        Fin::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        table::ConservativeTreeRouteTable2D;
        boundary::Symbol=:skip,
        u_south=0,
        u_north=0,
        rho_wall=1,
        alpha_c2f=1,
        alpha_f2c=1,
        interface_time_scaling::Symbol=:leaf_equivalent,
        is_solid=nothing)
    _check_conservative_tree_stream_args_2d(Fout, Fin, spec)
    policy = _check_conservative_tree_boundary_policy_2d(boundary)
    periodic_x = _conservative_tree_periodic_x_policy_2d(policy)
    is_solid === nothing ||
        validate_conservative_tree_solid_mask_resolved_2d(
            spec, table, is_solid)
    if spec.max_level == 0
        return stream_conservative_tree_routes_F_2d!(
            Fout, Fin, spec, table; boundary=boundary, u_south=u_south,
            u_north=u_north, rho_wall=rho_wall)
    end

    schedule = create_conservative_tree_subcycle_schedule_2d(spec.max_level)
    bank = create_conservative_tree_subcycle_spatial_ledger_bank_2d(
        spec; schedule=schedule, T=eltype(Fout))
    Fstate = copy(Fin)
    Fscratch = similar(Fout)
    Fpending = similar(Fout)
    fill!(Fpending, zero(eltype(Fpending)))

    for event in schedule.events
        if event.phase == :sync_down
            reset_conservative_tree_subcycle_spatial_pair_2d!(
                bank, event.src_level)
            conservative_tree_subcycle_sync_down_routes_F_2d!(
                bank, event, Fstate, table; alpha=alpha_c2f,
                interface_time_scaling=interface_time_scaling,
                periodic_x=periodic_x)
        elseif event.phase == :advance
            fill!(Fscratch, zero(eltype(Fscratch)))
            level = event.src_level
            if level > 0
                _add_and_clear_conservative_tree_level_rows_2d!(
                    Fstate, Fpending, spec, level)
                conservative_tree_subcycle_accumulate_advance_routes_F_2d!(
                    bank, event, Fstate, table; alpha=alpha_f2c,
                    interface_time_scaling=interface_time_scaling,
                    periodic_x=periodic_x)
                conservative_tree_subcycle_accumulate_inactive_parent_routes_F_2d!(
                    bank, event, Fstate; alpha=alpha_f2c,
                    interface_time_scaling=interface_time_scaling,
                    periodic_x=periodic_x)
            end
            _stream_conservative_tree_direct_level_routes_F_2d!(
                Fscratch, Fstate, spec, table, level, policy;
                u_south=u_south, u_north=u_north, rho_wall=rho_wall,
                is_solid=is_solid)
            if level > 0
                conservative_tree_subcycle_apply_child_advance_injection_F_2d!(
                    Fscratch, bank, event)
            end
            _add_and_clear_conservative_tree_level_rows_2d!(
                Fscratch, Fpending, spec, level)
            _copy_conservative_tree_level_rows_2d!(
                Fstate, Fscratch, spec, level)
        elseif event.phase == :sync_up
            conservative_tree_subcycle_apply_sync_up_F_2d!(
                Fpending, bank, event; boundary=boundary, u_south=u_south,
                u_north=u_north, rho_wall=rho_wall)
        else
            throw(ArgumentError("unknown subcycle event phase $(event.phase)"))
        end
    end

    copyto!(Fout, Fstate)
    return Fout
end

"""
stream_conservative_tree_subcycled_buffered_routes_F_2d!(Fout, Fin, spec, table;
                                                             boundary=:skip,
                                                             alpha_c2f=1,
                                                             alpha_f2c=1)

CPU reference transport that drives the recursive AMR-D schedule through the
explicit subcycle buffer contract. Unlike the legacy skeleton, it keeps
committed level state (`owned`), coarse-to-fine injections
(`ghost_from_coarse`), fine-to-coarse reflux (`reflux_to_coarse`), and
fine-to-parent restriction (`restrict_to_parent`) in distinct buffers.

This remains transport-only: no collision, forcing, or physical open-boundary
closure is performed here.

`alpha_c2f` and `alpha_f2c` are the Filippova-Hanel style non-equilibrium
rescaling factors for coarse-to-fine and fine-to-coarse interface packets.
`interface_time_scaling=:leaf_equivalent` keeps the historical packet weights:
coarse-to-fine split packets are distributed across child half-steps and
fine-to-coarse packets are accumulated with the reciprocal half-step weight.
`interface_time_scaling=:level_native` is experimental and only intended for
route tables built with `sampling=:level_native`; it preserves global rest mass
for flat coarse-to-fine packets at nested interfaces.
`coarse_to_fine_predictor_weight` blends coarse-to-fine packets between the
committed parent state (`0`) and a local post-collision predictor (`1`).
Macro-flow runners use a conservative partial blend; transport-only callers
default to `0`.
`pre_stream_level!`, when provided, is called as
`pre_stream_level!(Fsource, spec, level, event)` immediately before direct
routes for that level are streamed. Macro-flow runners use it to apply local
collision at each level's own subcycled advance.
"""
function stream_conservative_tree_subcycled_buffered_routes_F_2d!(
        Fout::AbstractMatrix,
        Fin::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        table::ConservativeTreeRouteTable2D;
        boundary::Symbol=:skip,
        u_south=0,
        u_north=0,
        rho_wall=1,
        alpha_c2f=1,
        alpha_f2c=1,
        interface_time_scaling::Symbol=:leaf_equivalent,
        coarse_to_fine_prolongation::Symbol=:flat,
        coarse_to_fine_state::Symbol=:owned,
        coarse_to_fine_predictor_weight=0,
        pre_stream_level! = nothing,
        schedule = nothing,
        route_bank = nothing,
        state_bank = nothing,
        Fsource = nothing,
        Fscratch = nothing,
        is_solid=nothing,
        interface_balance::Bool=false,
        wall_phase_transport_correction::Bool=true,
        level_native_delayed_same_level_transit::Bool=true)
    _check_conservative_tree_stream_args_2d(Fout, Fin, spec)
    policy = _check_conservative_tree_boundary_policy_2d(boundary)
    periodic_x = _conservative_tree_periodic_x_policy_2d(policy)
    _check_conservative_tree_coarse_to_fine_state_2d(coarse_to_fine_state)
    _check_conservative_tree_coarse_to_fine_predictor_weight_2d(
        coarse_to_fine_predictor_weight)
    _check_conservative_tree_interface_time_scaling_2d(
        interface_time_scaling)
    _check_conservative_tree_coarse_to_fine_prolongation_2d(
        coarse_to_fine_prolongation)
    if interface_time_scaling == :level_native &&
       coarse_to_fine_prolongation == :limited_linear
        throw(ArgumentError("interface_time_scaling=:level_native is closed only with coarse_to_fine_prolongation=:flat"))
    end
    is_solid === nothing ||
        validate_conservative_tree_solid_mask_resolved_2d(
            spec, table, is_solid)
    if spec.max_level == 0
        return stream_conservative_tree_routes_F_2d!(
            Fout, Fin, spec, table; boundary=boundary, u_south=u_south,
            u_north=u_north, rho_wall=rho_wall)
    end
    if wall_phase_transport_correction &&
       policy == :periodic_x_wall_y &&
       table.sampling == :level_native &&
       interface_time_scaling == :level_native &&
       coarse_to_fine_prolongation == :flat &&
       coarse_to_fine_state == :owned &&
       iszero(coarse_to_fine_predictor_weight) &&
       alpha_c2f == 1 &&
       alpha_f2c == 1 &&
       pre_stream_level! === nothing &&
       route_bank === nothing &&
       state_bank === nothing &&
       Fsource === nothing &&
       Fscratch === nothing &&
       is_solid === nothing
        return _stream_conservative_tree_level_native_wall_phase_transport_F_2d!(
            Fout, Fin, spec, table; boundary=boundary, u_south=u_south,
            u_north=u_north, rho_wall=rho_wall, alpha_c2f=alpha_c2f,
            alpha_f2c=alpha_f2c, coarse_to_fine_state=coarse_to_fine_state,
            schedule=schedule)
    end

    schedule_run = schedule === nothing ?
        create_conservative_tree_subcycle_schedule_2d(spec.max_level) :
        schedule
    _check_conservative_tree_subcycle_spec_schedule_2d(spec, schedule_run)
    route_bank_run = route_bank === nothing ?
        create_conservative_tree_subcycle_spatial_ledger_bank_2d(
            spec; schedule=schedule_run, T=eltype(Fout)) :
        route_bank
    state_bank_run = state_bank === nothing ?
        create_conservative_tree_subcycle_buffer_bank_2d(
            spec; schedule=schedule_run, T=eltype(Fout)) :
        state_bank
    route_bank_run.spec === spec ||
        throw(ArgumentError("route_bank must belong to spec"))
    route_bank_run.schedule === schedule_run ||
        throw(ArgumentError("route_bank schedule must match schedule"))
    state_bank_run.spec === spec ||
        throw(ArgumentError("state_bank must belong to spec"))
    state_bank_run.schedule === schedule_run ||
        throw(ArgumentError("state_bank schedule must match schedule"))
    if !_conservative_tree_subcycle_route_packet_cache_ready_2d(
            route_bank_run, table; periodic_x=periodic_x)
        prepare_conservative_tree_subcycle_route_packet_cache_2d!(
            route_bank_run, table; periodic_x=periodic_x)
    end
    reset_conservative_tree_subcycle_spatial_bank_2d!(route_bank_run)
    reset_conservative_tree_subcycle_buffer_bank_2d!(state_bank_run)
    conservative_tree_subcycle_store_active_owned_2d!(state_bank_run, Fin)
    conservative_tree_subcycle_restrict_all_levels_2d!(state_bank_run)
    Fsource_run = Fsource === nothing ? similar(Fout) : Fsource
    Fscratch_run = Fscratch === nothing ? similar(Fout) : Fscratch
    _check_conservative_tree_F_2d(Fsource_run, spec)
    _check_conservative_tree_F_2d(Fscratch_run, spec)
    if wall_phase_transport_correction &&
       pre_stream_level! !== nothing &&
       policy == :periodic_x_wall_y &&
       table.sampling == :level_native &&
       interface_time_scaling == :level_native &&
       coarse_to_fine_prolongation == :flat &&
       coarse_to_fine_state == :owned &&
       (iszero(coarse_to_fine_predictor_weight) ||
        coarse_to_fine_predictor_weight == 1) &&
       alpha_c2f == 1 &&
       alpha_f2c == 1 &&
       is_solid === nothing
        scatter_run = _cached_conservative_tree_wall_phase_scatter_2d(
            route_bank_run; periodic_x=periodic_x)
        if (iszero(coarse_to_fine_predictor_weight) &&
            !isempty(scatter_run.src_ids)) ||
           (!iszero(coarse_to_fine_predictor_weight) &&
            !isempty(scatter_run.event_mask_src_ids))
            Fpost_run = similar(Fout)
            _conservative_tree_apply_prestream_over_schedule_2d!(
                Fpost_run, Fin, spec, schedule_run, pre_stream_level!)
            return _stream_conservative_tree_level_native_wall_phase_transport_F_2d!(
                Fout, Fpost_run, spec, table; boundary=boundary,
                u_south=u_south, u_north=u_north, rho_wall=rho_wall,
                alpha_c2f=alpha_c2f, alpha_f2c=alpha_f2c,
                coarse_to_fine_state=coarse_to_fine_state,
                pre_stream_level! = nothing,
                schedule=schedule_run, route_bank=route_bank_run,
                state_bank=state_bank_run, Fsource=Fsource_run,
                Fscratch=Fscratch_run, scatter=scatter_run,
                scatter_source=Fpost_run)
        end
    end
    # The substep-keyed scatter table is prepared above as a WIP contract, but
    # remains disabled until its per-event mask scope is exact on BGK+force.
    wall_phase_substep_scatter = nothing
    delayed_same_level_transit =
        level_native_delayed_same_level_transit &&
        pre_stream_level! === nothing
    Fmasked_run = wall_phase_substep_scatter === nothing ?
        Fsource_run : similar(Fout)
    Fwall_source_run = wall_phase_substep_scatter === nothing ?
        Fsource_run : similar(Fout)
    use_phase_level_native_direct =
        interface_time_scaling == :level_native &&
        coarse_to_fine_prolongation == :flat &&
        alpha_c2f == 1 &&
        is_solid === nothing
    phase_level_sources = use_phase_level_native_direct ?
        [similar(Fout) for _ in 1:spec.max_level] :
        typeof(Fout)[]
    phase_level_source_valid = falses(spec.max_level)
    for event in schedule_run.events
        if event.phase == :sync_down
            reset_conservative_tree_subcycle_spatial_pair_2d!(
                route_bank_run, event.src_level)
            parent_owned = state_bank_run.levels[event.src_level + 1].owned
            Fparent = parent_owned
            predictor_weight = coarse_to_fine_state == :postcollision ?
                one(eltype(Fsource_run)) :
                eltype(Fsource_run)(coarse_to_fine_predictor_weight)
            needs_level_source =
                predictor_weight > zero(predictor_weight) ||
                coarse_to_fine_prolongation == :limited_linear ||
                use_phase_level_native_direct ||
                wall_phase_substep_scatter !== nothing
            if needs_level_source
                fill!(Fsource_run, zero(eltype(Fsource_run)))
                _copy_conservative_tree_level_rows_2d!(
                    Fsource_run, parent_owned,
                    state_bank_run.active_ids_by_level[event.src_level + 1])
                conservative_tree_subcycle_apply_restriction_to_inactive_level_F_2d!(
                    Fsource_run, state_bank_run, event.src_level)
                Fparent = Fsource_run
            end
            if predictor_weight > zero(predictor_weight)
                fill!(Fscratch_run, zero(eltype(Fscratch_run)))
                _copy_conservative_tree_level_rows_2d!(
                    Fscratch_run, parent_owned,
                    state_bank_run.active_ids_by_level[event.src_level + 1])
                conservative_tree_subcycle_apply_restriction_to_inactive_level_F_2d!(
                    Fscratch_run, state_bank_run, event.src_level)
                pre_stream_level! === nothing ||
                    pre_stream_level!(Fscratch_run, spec, event.src_level, event)
                @inbounds for (cell_id, cell) in pairs(spec.cells)
                    cell.level == event.src_level || continue
                    for q in 1:9
                        Fsource_run[cell_id, q] =
                            (one(predictor_weight) - predictor_weight) *
                            Fsource_run[cell_id, q] +
                            predictor_weight * Fscratch_run[cell_id, q]
                    end
                end
                Fparent = Fsource_run
            end
            phase_source_ready = use_phase_level_native_direct &&
                (pre_stream_level! === nothing ||
                 predictor_weight == one(predictor_weight))
            phase_safe = phase_source_ready &&
                _conservative_tree_level_native_phase_c2f_safe_2d(
                    route_bank_run, event.src_level)
            if phase_safe
                copyto!(phase_level_sources[event.src_level + 1], Fparent)
                phase_level_source_valid[event.src_level + 1] = true
            elseif use_phase_level_native_direct
                phase_level_source_valid[event.src_level + 1] = false
            end
            Fparent_routes = Fparent
            Fparent_scatter = Fparent
            if wall_phase_substep_scatter !== nothing
                if pre_stream_level! !== nothing &&
                   predictor_weight < one(predictor_weight)
                    copyto!(Fwall_source_run, Fsource_run)
                    pre_stream_level!(
                        Fwall_source_run, spec, event.src_level, event)
                    Fparent_scatter = Fwall_source_run
                end
                if _copy_mask_conservative_tree_wall_phase_event_sources_2d!(
                        Fmasked_run, Fparent, wall_phase_substep_scatter,
                        :sync_down, event.src_level, event.tick)
                    Fparent_routes = Fmasked_run
                end
            end
            conservative_tree_subcycle_sync_down_routes_F_2d!(
                route_bank_run, event, Fparent_routes, table; alpha=alpha_c2f,
                interface_time_scaling=interface_time_scaling,
                coarse_to_fine_prolongation=coarse_to_fine_prolongation,
                periodic_x=periodic_x,
                phase_resolved_level_native=phase_safe)
            wall_phase_substep_scatter === nothing ||
                _apply_conservative_tree_wall_phase_sync_down_scatter_2d!(
                    route_bank_run, event, Fparent_scatter,
                    wall_phase_substep_scatter)
        elseif event.phase == :advance
            level = event.src_level
            buffers = state_bank_run.levels[level + 1]
            fill!(Fsource_run, zero(eltype(Fsource_run)))
            _copy_conservative_tree_level_rows_2d!(
                Fsource_run, buffers.owned,
                state_bank_run.active_ids_by_level[level + 1])
            conservative_tree_subcycle_apply_restriction_to_inactive_level_F_2d!(
                Fsource_run, state_bank_run, level)
            pre_stream_level! === nothing ||
                pre_stream_level!(Fsource_run, spec, level, event)
            Froute_source = Fsource_run
            if wall_phase_substep_scatter !== nothing &&
               _copy_mask_conservative_tree_wall_phase_event_sources_2d!(
                    Fmasked_run, Fsource_run, wall_phase_substep_scatter,
                    :advance, level, event.tick)
                Froute_source = Fmasked_run
            end

            phase_direct_active =
                use_phase_level_native_direct &&
                level < spec.max_level &&
                phase_level_source_valid[level + 1]

            if level > 0
                conservative_tree_subcycle_accumulate_advance_routes_F_2d!(
                    route_bank_run, event, Froute_source, table;
                    alpha=alpha_f2c,
                    interface_time_scaling=interface_time_scaling,
                    delayed_same_level_transit=delayed_same_level_transit,
                    periodic_x=periodic_x)
                conservative_tree_subcycle_accumulate_inactive_parent_routes_F_2d!(
                    route_bank_run, event, Froute_source; alpha=alpha_f2c,
                    interface_time_scaling=interface_time_scaling,
                    delayed_same_level_transit=delayed_same_level_transit,
                    periodic_x=periodic_x)
            end

            fill!(Fscratch_run, zero(eltype(Fscratch_run)))
            Fdirect_source = Froute_source
            if phase_direct_active
                Fdirect_source = phase_level_sources[level + 1]
                if wall_phase_substep_scatter !== nothing &&
                   _copy_mask_conservative_tree_wall_phase_event_sources_2d!(
                        Fmasked_run, Fdirect_source,
                        wall_phase_substep_scatter, :advance, level,
                        event.tick)
                    Fdirect_source = Fmasked_run
                end
            end
            _stream_conservative_tree_direct_level_routes_F_2d!(
                Fscratch_run, Fdirect_source, spec, table, level, policy;
                u_south=u_south, u_north=u_north, rho_wall=rho_wall,
                is_solid=is_solid,
                coarse_to_fine_prolongation=coarse_to_fine_prolongation,
                periodic_x=periodic_x,
                phase_resolved_level_native=phase_direct_active,
                subcycle_ratio=schedule_run.ratio)
            if phase_direct_active
                phase_level_source_valid[level + 1] = false
            end
            conservative_tree_subcycle_apply_delayed_same_level_packets_F_2d!(
                Fscratch_run, route_bank_run, event)
            if wall_phase_substep_scatter !== nothing
                _apply_conservative_tree_wall_phase_advance_scatter_2d!(
                    Fscratch_run, Fsource_run, wall_phase_substep_scatter,
                    level, event.tick)
                _apply_conservative_tree_wall_phase_reflux_scatter_2d!(
                    state_bank_run, Fsource_run, wall_phase_substep_scatter,
                    level, event.tick)
            end
            if level > 0
                fill!(buffers.ghost_from_coarse,
                      zero(eltype(buffers.ghost_from_coarse)))
                conservative_tree_subcycle_apply_child_advance_injection_F_2d!(
                    buffers.ghost_from_coarse, route_bank_run, event)
                conservative_tree_subcycle_add_and_clear_ghost_to_F_level_2d!(
                    Fscratch_run, state_bank_run, level)
            end
            conservative_tree_subcycle_add_and_clear_reflux_to_F_level_2d!(
                Fscratch_run, state_bank_run, level)
            _copy_conservative_tree_active_level_rows_2d!(
                buffers.owned, Fscratch_run,
                state_bank_run.active_ids_by_level[level + 1])
        elseif event.phase == :sync_up
            parent_level = _check_subcycle_spatial_sync_up_event_2d(
                route_bank_run, event)
            conservative_tree_subcycle_restrict_level_2d!(
                state_bank_run, parent_level)
            parent_reflux = state_bank_run.levels[parent_level + 1].reflux_to_coarse
            conservative_tree_subcycle_apply_sync_up_F_2d!(
                parent_reflux, route_bank_run, event; boundary=boundary,
                u_south=u_south, u_north=u_north, rho_wall=rho_wall)
        else
            throw(ArgumentError("unknown subcycle event phase $(event.phase)"))
        end
    end

    conservative_tree_subcycle_collect_active_owned_F_2d!(Fout, state_bank_run)
    return Fout
end

function diagnose_conservative_tree_subcycled_rest_2d(
        spec::ConservativeTreeSpec2D,
        table::ConservativeTreeRouteTable2D;
        boundary::Symbol=:bounceback,
        T::Type{<:Real}=Float64)
    Fin = allocate_conservative_tree_F_2d(spec; T=T)
    Fout = allocate_conservative_tree_F_2d(spec; T=T)
    w = weights(D2Q9())
    @inbounds for cell_id in spec.active_cells
        volume = T(spec.cells[cell_id].metrics.volume)
        for q in 1:9
            Fin[cell_id, q] = T(w[q]) * volume
        end
    end

    stream_conservative_tree_subcycled_routes_F_2d!(
        Fout, Fin, spec, table; boundary=boundary)

    active_initial = sum(Fin[spec.active_cells, :])
    active_final = sum(Fout[spec.active_cells, :])
    level_drift = zeros(T, spec.max_level + 1)
    @inbounds for level in 0:spec.max_level
        ids = [id for id in spec.active_cells if spec.cells[id].level == level]
        level_drift[level + 1] = sum(Fout[ids, :]) - sum(Fin[ids, :])
    end
    orientation_drift = active_population_sums_F_2d(Fout, spec) -
                        active_population_sums_F_2d(Fin, spec)
    return (active_initial=active_initial,
            active_final=active_final,
            active_drift=active_final - active_initial,
            max_active_abs=maximum(abs.(Fout[spec.active_cells, :] .-
                                        Fin[spec.active_cells, :])),
            level_drift=level_drift,
            orientation_drift=orientation_drift)
end

function conservative_tree_subcycle_sync_down_face_2d!(
        bank::ConservativeTreeSubcycleLedgerBank2D,
        event::ConservativeTreeSubcycleEvent2D,
        Fq,
        q::Integer,
        face::Symbol)
    parent = _check_subcycle_sync_down_event_2d(bank.schedule, event)
    ledger = conservative_tree_subcycle_pair_ledger_2d(bank, parent)
    conservative_tree_subcycle_deposit_coarse_to_fine_face_2d!(
        ledger, Fq, q, face)
    return ledger
end

function conservative_tree_subcycle_sync_down_corner_2d!(
        bank::ConservativeTreeSubcycleLedgerBank2D,
        event::ConservativeTreeSubcycleEvent2D,
        Fq,
        q::Integer,
        corner::Symbol)
    parent = _check_subcycle_sync_down_event_2d(bank.schedule, event)
    ledger = conservative_tree_subcycle_pair_ledger_2d(bank, parent)
    conservative_tree_subcycle_deposit_coarse_to_fine_corner_2d!(
        ledger, Fq, q, corner)
    return ledger
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




function conservative_tree_subcycle_deposit_coarse_to_fine_face_2d!(
        ledger::ConservativeTreeSubcycleLedger2D,
        Fq,
        q::Integer,
        face::Symbol)
    qi = _check_d2q9_q(Int(q))
    weights = conservative_tree_subcycle_weights_2d(ledger)
    @inbounds for substep in 1:ledger.ratio
        split_coarse_to_fine_face_F_2d!(
            @view(ledger.coarse_to_fine[:, :, :, substep]),
            Fq * weights[substep], qi, face)
    end
    return ledger
end

function conservative_tree_subcycle_deposit_coarse_to_fine_corner_2d!(
        ledger::ConservativeTreeSubcycleLedger2D,
        Fq,
        q::Integer,
        corner::Symbol)
    qi = _check_d2q9_q(Int(q))
    weights = conservative_tree_subcycle_weights_2d(ledger)
    @inbounds for substep in 1:ledger.ratio
        split_coarse_to_fine_corner_F_2d!(
            @view(ledger.coarse_to_fine[:, :, :, substep]),
            Fq * weights[substep], qi, corner)
    end
    return ledger
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
