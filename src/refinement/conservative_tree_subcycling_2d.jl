include("subcycling_schedule_2d.jl")
include("subcycling_explosion_2d.jl")
include("subcycling_coalesce_2d.jl")
include("subcycling_streaming_2d.jl")




































































































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
