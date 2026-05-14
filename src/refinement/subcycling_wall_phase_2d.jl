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
