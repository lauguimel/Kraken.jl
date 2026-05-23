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
    # M-H-ETA-8: inactive parent cells (refined to fine children) hold no
    # valid F values; treat as missing for slope stencils.
    spec.cells[cell_id].active || return false, zero(eltype(F))
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
    end
    # M-H-ETA-8: order-2 decentred FD when the opposite same-level neighbour
    # is refined (typical at 3-way vertex). Captures quadratic feq curvature
    # exactly when the second cell on the available side is also same-level.
    if has_left
        has_left2, left2_value = _conservative_tree_same_level_Fq_2d(
            F, spec, src.level, src.i - 2, src.j, q; periodic_x=periodic_x)
        if has_left2
            return (3 * center - 4 * left_value + left2_value) / 2
        end
        return center - left_value
    elseif has_right
        has_right2, right2_value = _conservative_tree_same_level_Fq_2d(
            F, spec, src.level, src.i + 2, src.j, q; periodic_x=periodic_x)
        if has_right2
            return (-3 * center + 4 * right_value - right2_value) / 2
        end
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
    end
    # M-H-ETA-8: order-2 decentred FD when the opposite same-level neighbour
    # is refined (typical at 3-way vertex).
    if has_south
        has_south2, south2_value = _conservative_tree_same_level_Fq_2d(
            F, spec, src.level, src.i, src.j - 2, q)
        if has_south2
            return (3 * center - 4 * south_value + south2_value) / 2
        end
        return center - south_value
    elseif has_north
        has_north2, north2_value = _conservative_tree_same_level_Fq_2d(
            F, spec, src.level, src.i, src.j + 2, q)
        if has_north2
            return (-3 * center + 4 * north_value - north2_value) / 2
        end
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
    if _krk_c2f_runtime_trace_enabled() &&
       ((route.src == 133 && route.q in (6, 7, 8, 9)) ||
        route.dst in (343, 364, 405, 426))
        _krk_c2f_runtime_log(
            string("C2F_DEPOSIT src_id=", route.src,
                   " src_level=", src.level,
                   " q=", route.q,
                   " dst_id=", route.dst,
                   " dst_level=", child.level,
                   " parent_id=", parent_id,
                   " route_weight=", route.weight,
                   " kind=", _krk_c2f_runtime_kind_name(route.kind),
                   " packet=", packet,
                   " per_substep=", packet / bank.schedule.ratio))
    end

    @inbounds for substep in 1:bank.schedule.ratio
        if _KRK_CFROUTE_TRACE
            _krk_cfroute_log!(:split_deposit, src.level, route.kind,
                              route.src, route.dst, qi, 0,
                              packet / bank.schedule.ratio)
        end
        pair.coarse_to_fine[ix, iy, qi, substep, slot] +=
            packet / bank.schedule.ratio
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
    _krk_c2f_runtime_log(
        string("SYNC_DOWN_NATIVE_ENTRY parent_level=", parent_level,
               " active_cells=", length(spec.active_cells)))
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
                    if _KRK_CFROUTE_TRACE
                        _krk_cfroute_log!(:split_native_phase, parent_level,
                                          SPLIT_FACE, src_id, dst_id, qcur,
                                          0, packet)
                    end
                    if _krk_c2f_runtime_trace_enabled() &&
                       ((src_id == 133 && q in (6, 7, 8, 9)) ||
                        dst_id in (343, 364, 405, 426))
                        _krk_c2f_runtime_log(
                            string("C2F_NATIVE_PHASE_DEPOSIT src_id=", src_id,
                                   " src_level=", src.level,
                                   " q_initial=", q,
                                   " q_deposit=", qcur,
                                   " dst_id=", dst_id,
                                   " dst_level=", child.level,
                                   " parent_id=", parent_id,
                                   " si=", si,
                                   " sj=", sj,
                                   " substep=", substep,
                                   " route_weight=NA",
                                   " kind=SPLIT_NATIVE_PHASE",
                                   " packet=", packet))
                    end
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
    if get(ENV, "KRK_TRACE_ENTER", "0") == "1"
        let _krk_trace_out = get(ENV, "KRK_TRACE_OUT", joinpath(".engineer_logs", "trace.jsonl"))
            mkpath(dirname(_krk_trace_out))
            open(_krk_trace_out, "a") do io
                println(io, """{"t_ns":$(time_ns()),"kernel":"cpu_sync_down_routes","file":"src/refinement/subcycling_explosion_2d.jl:373","extras":{"parent_level":$parent_level,"split_routes":$(length(table.split_route_ranges_by_parent_level[parent_level + 1]))}}""")
            end
        end
    end
    _check_conservative_tree_subcycle_route_table_2d(table)
    _check_conservative_tree_subcycle_spatial_F_2d(F, bank)
    _krk_c2f_runtime_log(
        string("SYNC_DOWN_ROUTES_ENTRY parent_level=", parent_level,
               " phase_resolved_level_native=", phase_resolved_level_native,
               " interface_time_scaling=", interface_time_scaling,
               " coarse_to_fine_prolongation=", coarse_to_fine_prolongation,
               " alpha=", alpha))
    if phase_resolved_level_native &&
       interface_time_scaling == :level_native &&
       coarse_to_fine_prolongation == :flat &&
       alpha == 1 &&
       _conservative_tree_level_native_phase_c2f_safe_2d(bank, parent_level)
        return conservative_tree_subcycle_sync_down_level_native_phase_routes_F_2d!(
            bank, event, F; periodic_x=periodic_x)
    end

    # M-H-ETA-7+8: for routes from coarse cells flagged at 3-way vertex
    # topology, substitute :limited_linear prolongation for the default :flat.
    # Combined with order-2 decentred FD on the slope (M-H-ETA-8, with the
    # active-check fix), this captures the linear part of feq variation
    # exactly. Residual ~O(α²·Δy²) ≈ 2e-4 for T8.
    # Only active in mode 0 (:leaf_equivalent) where :flat is the production
    # default. :level_native has its own corner_reflux machinery.
    use_three_way_substitution =
        interface_time_scaling == :leaf_equivalent &&
        coarse_to_fine_prolongation == :flat &&
        alpha == 1
    @inbounds for route_pos in table.split_route_ranges_by_parent_level[parent_level + 1]
        route_id = table.interface_routes[route_pos]
        route = table.routes[route_id]
        per_route_prolongation = coarse_to_fine_prolongation
        if use_three_way_substitution &&
           _is_three_way_flagged_src_2d(table, route.src)
            per_route_prolongation = :limited_linear
        end
        conservative_tree_subcycle_deposit_coarse_to_fine_route_2d!(
            bank, F, route; alpha=alpha,
            interface_time_scaling=interface_time_scaling,
            coarse_to_fine_prolongation=per_route_prolongation,
            periodic_x=periodic_x)
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
