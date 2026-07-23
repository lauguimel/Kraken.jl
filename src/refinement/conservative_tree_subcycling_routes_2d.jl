function conservative_tree_subcycle_local_substep_2d(
        schedule::ConservativeTreeSubcycleSchedule2D,
        parent_level::Integer,
        tick::Integer)
    parent = _check_conservative_tree_pair_level_2d(schedule, parent_level)
    t = Int(tick)
    0 < t <= schedule.finest_ticks ||
        throw(ArgumentError("tick must be a positive schedule tick"))

    parent_ticks = schedule.level_step_ticks[parent + 1]
    child_ticks = schedule.level_step_ticks[parent + 2]
    t % child_ticks == 0 ||
        throw(ArgumentError("tick is not aligned with the child level"))

    parent_start = div(t - 1, parent_ticks) * parent_ticks
    local_step = div(t - parent_start, child_ticks)
    1 <= local_step <= schedule.ratio ||
        throw(ArgumentError("tick is outside the parent subcycle interval"))
    return local_step
end

function _check_subcycle_sync_down_event_2d(
        schedule::ConservativeTreeSubcycleSchedule2D,
        event::ConservativeTreeSubcycleEvent2D)
    event.phase == :sync_down ||
        throw(ArgumentError("event must be :sync_down"))
    event.dst_level == event.src_level + 1 ||
        throw(ArgumentError("sync_down event must target an adjacent child level"))
    _check_conservative_tree_pair_level_2d(schedule, event.src_level)
    return event.src_level
end

function _check_subcycle_sync_up_event_2d(
        schedule::ConservativeTreeSubcycleSchedule2D,
        event::ConservativeTreeSubcycleEvent2D)
    event.phase == :sync_up ||
        throw(ArgumentError("event must be :sync_up"))
    event.src_level == event.dst_level + 1 ||
        throw(ArgumentError("sync_up event must target an adjacent parent level"))
    _check_conservative_tree_pair_level_2d(schedule, event.dst_level)
    return event.dst_level
end

function _check_subcycle_child_advance_event_2d(
        schedule::ConservativeTreeSubcycleSchedule2D,
        event::ConservativeTreeSubcycleEvent2D)
    event.phase == :advance ||
        throw(ArgumentError("event must be :advance"))
    event.src_level == event.dst_level ||
        throw(ArgumentError("advance event must have identical src/dst levels"))
    event.src_level > 0 ||
        throw(ArgumentError("level-0 advance has no parent subcycle ledger"))
    parent = event.src_level - 1
    substep = conservative_tree_subcycle_local_substep_2d(
        schedule, parent, event.tick)
    return parent, substep
end

function _check_subcycle_spatial_sync_down_event_2d(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        event::ConservativeTreeSubcycleEvent2D)
    parent = _check_subcycle_sync_down_event_2d(bank.schedule, event)
    _conservative_tree_packed_ledger_pair_2d(bank, parent)
    return parent
end

function _check_subcycle_spatial_sync_up_event_2d(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        event::ConservativeTreeSubcycleEvent2D)
    parent = _check_subcycle_sync_up_event_2d(bank.schedule, event)
    _conservative_tree_packed_ledger_pair_2d(bank, parent)
    return parent
end

function _check_subcycle_spatial_child_advance_event_2d(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        event::ConservativeTreeSubcycleEvent2D)
    parent, substep = _check_subcycle_child_advance_event_2d(
        bank.schedule, event)
    _conservative_tree_packed_ledger_pair_2d(bank, parent)
    return parent, substep
end

@inline function _subcycle_route_packet_2d(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        route::ConservativeTreeRoute2D;
        alpha=1)
    src = spec.cells[route.src]
    return reconstructed_integrated_D2Q9_packet(
        @view(F[route.src, :]), src.metrics.volume, route.q, route.weight;
        alpha=alpha)
end

function conservative_tree_subcycle_deposit_coarse_to_fine_route_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        F::AbstractMatrix,
        route::ConservativeTreeRoute2D;
        alpha=1)
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
    packet = bank.schedule.ratio * _subcycle_route_packet_2d(
        F, spec, route; alpha=alpha)

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
        alpha=1)
    return _conservative_tree_subcycle_accumulate_fine_to_coarse_packet_2d!(
        bank, F, route.src, route.dst, route.q, route.weight, route.kind,
        substep, route_packet_slot; alpha=alpha)
end

function conservative_tree_subcycle_sync_down_routes_F_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        event::ConservativeTreeSubcycleEvent2D,
        F::AbstractMatrix,
        table::ConservativeTreeRouteTable2D;
        alpha=1)
    parent_level = _check_subcycle_spatial_sync_down_event_2d(bank, event)
    _check_conservative_tree_subcycle_route_table_2d(table)
    _check_conservative_tree_subcycle_spatial_F_2d(F, bank)

    @inbounds for route_pos in table.split_route_ranges_by_parent_level[parent_level + 1]
        route_id = table.interface_routes[route_pos]
        route = table.routes[route_id]
        conservative_tree_subcycle_deposit_coarse_to_fine_route_2d!(
            bank, F, route; alpha=alpha)
    end
    return bank
end

function conservative_tree_subcycle_accumulate_advance_routes_F_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        event::ConservativeTreeSubcycleEvent2D,
        F::AbstractMatrix,
        table::ConservativeTreeRouteTable2D;
        alpha=1)
    parent_level, substep = _check_subcycle_spatial_child_advance_event_2d(
        bank, event)
    _check_conservative_tree_subcycle_route_table_2d(table)
    _check_conservative_tree_subcycle_spatial_F_2d(F, bank)
    if length(bank.route_packet_slot_by_route) < length(table.routes)
        prepare_conservative_tree_subcycle_route_packet_cache_2d!(bank, table)
    end
    child_level = parent_level + 1

    @inbounds for route_pos in table.coalesce_route_ranges_by_child_level[child_level + 1]
        route_id = table.interface_routes[route_pos]
        route = table.routes[route_id]
        packet_slot = bank.route_packet_slot_by_route[route_id]
        _conservative_tree_subcycle_accumulate_fine_to_coarse_packet_unchecked_2d!(
            bank, F, route.src, route.dst, route.q, route.weight, route.kind,
            substep, packet_slot; alpha=alpha)
    end
    return bank
end

function _conservative_tree_inactive_parent_coalesce_route_spec_2d(
        spec::ConservativeTreeSpec2D,
        src_id::Int,
        q::Int)
    src = spec.cells[src_id]
    src.level > 0 || return 0, DIRECT
    src.active && return 0, DIRECT
    spec.children[src_id] == (0, 0, 0, 0) && return 0, DIRECT

    i_dst = src.i + d2q9_cx(q)
    j_dst = src.j + d2q9_cy(q)
    nx_level = _conservative_tree_level_size_2d(spec.Nx, src.level)
    ny_level = _conservative_tree_level_size_2d(spec.Ny, src.level)
    1 <= i_dst <= nx_level && 1 <= j_dst <= ny_level ||
        return 0, DIRECT

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
        q::Int)
    dst_id, _ = _conservative_tree_inactive_parent_coalesce_route_spec_2d(
        spec, src_id, q)
    return dst_id
end

function _conservative_tree_inactive_parent_coalesce_routes_2d(
        spec::ConservativeTreeSpec2D,
        src_id::Int,
        q::Int)
    dst_id, kind = _conservative_tree_inactive_parent_coalesce_route_spec_2d(
        spec, src_id, q)
    dst_id == 0 && return ConservativeTreeRoute2D[]
    return [ConservativeTreeRoute2D(src_id, dst_id, q, 1.0, kind)]
end

function conservative_tree_subcycle_accumulate_inactive_parent_routes_F_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        event::ConservativeTreeSubcycleEvent2D,
        F::AbstractMatrix;
        alpha=1)
    parent_level, substep = _check_subcycle_spatial_child_advance_event_2d(
        bank, event)
    child_level = parent_level + 1
    _check_conservative_tree_subcycle_spatial_F_2d(F, bank)

    inactive_ids = bank.inactive_refined_ids_by_level[child_level + 1]
    inactive_slots = bank.inactive_route_packet_slots_by_level[child_level + 1]
    @inbounds for (local_idx, src_id) in enumerate(inactive_ids)
        bank.spec.cells[src_id].active && continue
        for q in 1:9
            packet_slot = inactive_slots[local_idx, q]
            packet_slot == 0 && continue
            dst_id, kind = _conservative_tree_inactive_parent_coalesce_route_spec_2d(
                bank.spec, src_id, q)
            dst_id == 0 && continue
            _conservative_tree_subcycle_accumulate_fine_to_coarse_packet_unchecked_2d!(
                bank, F, src_id, dst_id, q, 1.0, kind, substep,
                packet_slot; alpha=alpha)
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
        is_solid=nothing)
    _check_conservative_tree_stream_args_2d(Fout, Fin, spec)

    @inbounds for route_pos in table.direct_route_ranges_by_level[level + 1]
        route_id = table.direct_routes[route_pos]
        route = table.routes[route_id]
        src_cell = spec.cells[route.src]
        _conservative_tree_cell_is_solid_2d(spec, src_cell, is_solid) &&
            continue
        if _conservative_tree_cell_is_solid_2d(
                spec, spec.cells[route.dst], is_solid)
            Fout[route.src, d2q9_opposite(route.q)] +=
                route.weight * Fin[route.src, route.q]
        else
            Fout[route.dst, route.q] += route.weight * Fin[route.src, route.q]
        end
    end
    @inbounds for route_pos in table.boundary_route_ranges_by_level[level + 1]
        route_id = table.boundary_routes[route_pos]
        route = table.routes[route_id]
        _conservative_tree_cell_is_solid_2d(
            spec, spec.cells[route.src], is_solid) && continue
        if policy != :skip
            Fout[route.src, d2q9_opposite(route.q)] +=
                _conservative_tree_boundary_reflection_packet_2d(
                    Fin, spec, route, policy; u_south=u_south,
                    u_north=u_north, rho_wall=rho_wall)
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
        is_solid=nothing)
    _check_conservative_tree_stream_args_2d(Fout, Fin, spec)
    policy = _check_conservative_tree_boundary_policy_2d(boundary)
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
                bank, event, Fstate, table; alpha=alpha_c2f)
        elseif event.phase == :advance
            fill!(Fscratch, zero(eltype(Fscratch)))
            level = event.src_level
            if level > 0
                _add_and_clear_conservative_tree_level_rows_2d!(
                    Fstate, Fpending, spec, level)
                conservative_tree_subcycle_accumulate_advance_routes_F_2d!(
                    bank, event, Fstate, table; alpha=alpha_f2c)
                conservative_tree_subcycle_accumulate_inactive_parent_routes_F_2d!(
                    bank, event, Fstate; alpha=alpha_f2c)
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
`pre_stream_level!`, when provided, is called as
`pre_stream_level!(Fsource, spec, level, event)` immediately before direct
routes for that level are streamed. Macro-flow runners use it to apply local
collision at each level's own subcycled advance.
"""
