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
