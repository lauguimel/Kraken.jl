# Conservative subcycling ledgers for route-native D2Q9 AMR.
#
# This is not yet the full time integrator. It records the packet accounting
# needed for one coarse step and two fine half-steps so interface transfers can
# be tested before they are put in the hot loop.

struct ConservativeTreeSubcycleLedger2D{T}
    ratio::Int
    coarse_to_fine::Array{T,4}
    fine_to_coarse::Matrix{T}
end

struct ConservativeTreeSubcycleEvent2D
    tick::Int
    phase::Symbol
    src_level::Int
    dst_level::Int
end

struct ConservativeTreeSubcycleSchedule2D
    max_level::Int
    ratio::Int
    finest_ticks::Int
    level_step_ticks::Vector{Int}
    events::Vector{ConservativeTreeSubcycleEvent2D}
end

struct ConservativeTreeSubcycleLedgerBank2D{T}
    schedule::ConservativeTreeSubcycleSchedule2D
    pair_ledgers::Vector{ConservativeTreeSubcycleLedger2D{T}}
end

struct ConservativeTreeSubcycleSpatialLedgerBank2D{T}
    spec::ConservativeTreeSpec2D
    schedule::ConservativeTreeSubcycleSchedule2D
    pair_parent_ledgers::Vector{Dict{Int,ConservativeTreeSubcycleLedger2D{T}}}
    fine_to_coarse_route_packets::Vector{Dict{Tuple{Int,Int},Vector{T}}}
    refined_parent_ids_by_level::Vector{Vector{Int}}
    inactive_refined_ids_by_level::Vector{Vector{Int}}
end

function create_conservative_tree_subcycle_ledger_2d(;
        T::Type{<:Real}=Float64,
        ratio::Integer=2)
    r = Int(ratio)
    r == 2 || throw(ArgumentError("conservative-tree subcycling currently requires ratio = 2"))
    return ConservativeTreeSubcycleLedger2D{T}(
        r, zeros(T, 2, 2, 9, r), zeros(T, 9, r))
end

function _check_conservative_tree_schedule_ratio_2d(ratio::Integer)
    r = Int(ratio)
    r >= 2 || throw(ArgumentError("subcycling ratio must be >= 2"))
    return r
end

function _conservative_tree_level_step_ticks_2d(max_level::Int, ratio::Int)
    return [ratio^(max_level - level) for level in 0:max_level]
end

function _push_conservative_tree_schedule_interval_2d!(
        events::Vector{ConservativeTreeSubcycleEvent2D},
        level::Int,
        max_level::Int,
        ratio::Int,
        tick_start::Int,
        tick_end::Int)
    if level == max_level
        push!(events, ConservativeTreeSubcycleEvent2D(
            tick_end, :advance, level, level))
        return events
    end

    child = level + 1
    push!(events, ConservativeTreeSubcycleEvent2D(
        tick_start, :sync_down, level, child))

    interval_ticks = tick_end - tick_start
    interval_ticks % ratio == 0 ||
        throw(ArgumentError("subcycle interval is not divisible by ratio"))
    child_ticks = div(interval_ticks, ratio)
    for substep in 1:ratio
        child_start = tick_start + (substep - 1) * child_ticks
        child_end = child_start + child_ticks
        _push_conservative_tree_schedule_interval_2d!(
            events, child, max_level, ratio, child_start, child_end)
    end

    push!(events, ConservativeTreeSubcycleEvent2D(
        tick_end, :sync_up, child, level))
    push!(events, ConservativeTreeSubcycleEvent2D(
        tick_end, :advance, level, level))
    return events
end

"""
    create_conservative_tree_subcycle_schedule_2d(max_level; ratio=2)

Build a level-agnostic recursive subcycling calendar for one level-0 coarse
step. Time is expressed in integer ticks of the finest level. For `ratio = 2`,
level `l` advances every `2^(max_level-l)` finest ticks.

The event order is recursive and deterministic:

1. `:sync_down` from a parent level to its child at the beginning of that
   parent interval;
2. all child sub-intervals;
3. `:sync_up` from child to parent at the synchronization point;
4. `:advance` of the parent level.

This object owns no populations and performs no physics. It is the dispatch
contract that the future route/reflux kernels must follow for any number of
levels.
"""
function create_conservative_tree_subcycle_schedule_2d(max_level::Integer;
                                                       ratio::Integer=2)
    ml = Int(max_level)
    ml >= 0 || throw(ArgumentError("max_level must be nonnegative"))
    r = _check_conservative_tree_schedule_ratio_2d(ratio)
    finest_ticks = r^ml
    level_step_ticks = _conservative_tree_level_step_ticks_2d(ml, r)
    events = ConservativeTreeSubcycleEvent2D[]
    _push_conservative_tree_schedule_interval_2d!(
        events, 0, ml, r, 0, finest_ticks)
    return ConservativeTreeSubcycleSchedule2D(
        ml, r, finest_ticks, level_step_ticks, events)
end

include("conservative_tree_subcycle_buffers_2d.jl")

function create_conservative_tree_subcycle_ledger_bank_2d(
        schedule::ConservativeTreeSubcycleSchedule2D;
        T::Type{<:Real}=Float64)
    pair_ledgers = ConservativeTreeSubcycleLedger2D{T}[
        create_conservative_tree_subcycle_ledger_2d(T=T, ratio=schedule.ratio)
        for _ in 1:schedule.max_level
    ]
    return ConservativeTreeSubcycleLedgerBank2D{T}(schedule, pair_ledgers)
end

function create_conservative_tree_subcycle_ledger_bank_2d(max_level::Integer;
                                                          ratio::Integer=2,
                                                          T::Type{<:Real}=Float64)
    schedule = create_conservative_tree_subcycle_schedule_2d(
        max_level; ratio=ratio)
    return create_conservative_tree_subcycle_ledger_bank_2d(schedule; T=T)
end

function _check_conservative_tree_subcycle_spec_schedule_2d(
        spec::ConservativeTreeSpec2D,
        schedule::ConservativeTreeSubcycleSchedule2D)
    spec.max_level == schedule.max_level ||
        throw(ArgumentError("subcycle schedule max_level must match the tree spec"))
    schedule.ratio == 2 ||
        throw(ArgumentError("conservative-tree spatial subcycling requires ratio = 2"))
    return nothing
end

function create_conservative_tree_subcycle_spatial_ledger_bank_2d(
        spec::ConservativeTreeSpec2D;
        schedule::ConservativeTreeSubcycleSchedule2D=
            create_conservative_tree_subcycle_schedule_2d(spec.max_level),
        T::Type{<:Real}=Float64)
    _check_conservative_tree_subcycle_spec_schedule_2d(spec, schedule)
    pair_parent_ledgers = [
        Dict{Int,ConservativeTreeSubcycleLedger2D{T}}()
        for _ in 1:spec.max_level
    ]
    fine_to_coarse_route_packets = [
        Dict{Tuple{Int,Int},Vector{T}}()
        for _ in 1:spec.max_level
    ]
    refined_parent_ids_by_level = [Int[] for _ in 1:spec.max_level]
    inactive_refined_ids_by_level = [Int[] for _ in 0:spec.max_level]

    @inbounds for (cell_id, cell) in pairs(spec.cells)
        children = spec.children[cell_id]
        children == (0, 0, 0, 0) && continue
        push!(inactive_refined_ids_by_level[cell.level + 1], cell_id)
        cell.level < spec.max_level || continue
        pair_parent_ledgers[cell.level + 1][cell_id] =
            create_conservative_tree_subcycle_ledger_2d(T=T, ratio=schedule.ratio)
        push!(refined_parent_ids_by_level[cell.level + 1], cell_id)
    end
    return ConservativeTreeSubcycleSpatialLedgerBank2D{T}(
        spec, schedule, pair_parent_ledgers, fine_to_coarse_route_packets,
        refined_parent_ids_by_level, inactive_refined_ids_by_level)
end

function conservative_tree_subcycle_events_at_tick_2d(
        schedule::ConservativeTreeSubcycleSchedule2D,
        tick::Integer)
    t = Int(tick)
    0 <= t <= schedule.finest_ticks ||
        throw(ArgumentError("tick is outside the schedule"))
    return [event for event in schedule.events if event.tick == t]
end

function conservative_tree_subcycle_advance_counts_2d(
        schedule::ConservativeTreeSubcycleSchedule2D)
    counts = zeros(Int, schedule.max_level + 1)
    @inbounds for event in schedule.events
        event.phase == :advance || continue
        counts[event.src_level + 1] += 1
    end
    return counts
end

function conservative_tree_subcycle_sync_counts_2d(
        schedule::ConservativeTreeSubcycleSchedule2D)
    counts = Dict{Tuple{Symbol,Int,Int},Int}()
    @inbounds for event in schedule.events
        event.phase == :advance && continue
        key = (event.phase, event.src_level, event.dst_level)
        counts[key] = get(counts, key, 0) + 1
    end
    return counts
end

function _check_conservative_tree_pair_level_2d(
        schedule::ConservativeTreeSubcycleSchedule2D,
        parent_level::Integer)
    parent = Int(parent_level)
    0 <= parent < schedule.max_level ||
        throw(ArgumentError("parent_level must identify an adjacent level pair"))
    return parent
end

function conservative_tree_subcycle_pair_ledger_2d(
        bank::ConservativeTreeSubcycleLedgerBank2D,
        parent_level::Integer)
    parent = _check_conservative_tree_pair_level_2d(
        bank.schedule, parent_level)
    return bank.pair_ledgers[parent + 1]
end

function reset_conservative_tree_subcycle_bank_2d!(
        bank::ConservativeTreeSubcycleLedgerBank2D)
    for ledger in bank.pair_ledgers
        reset_conservative_tree_subcycle_ledger_2d!(ledger)
    end
    return bank
end

function reset_conservative_tree_subcycle_pair_2d!(
        bank::ConservativeTreeSubcycleLedgerBank2D,
        parent_level::Integer)
    reset_conservative_tree_subcycle_ledger_2d!(
        conservative_tree_subcycle_pair_ledger_2d(bank, parent_level))
    return bank
end

function conservative_tree_subcycle_spatial_pair_ledgers_2d(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        parent_level::Integer)
    parent = _check_conservative_tree_pair_level_2d(
        bank.schedule, parent_level)
    return bank.pair_parent_ledgers[parent + 1]
end

function conservative_tree_subcycle_spatial_ledger_2d(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        parent_cell_id::Integer)
    parent_id = Int(parent_cell_id)
    1 <= parent_id <= length(bank.spec.cells) ||
        throw(ArgumentError("parent_cell_id is outside the tree"))
    parent = bank.spec.cells[parent_id]
    _check_conservative_tree_pair_level_2d(bank.schedule, parent.level)
    pair = bank.pair_parent_ledgers[parent.level + 1]
    haskey(pair, parent_id) ||
        throw(ArgumentError("parent_cell_id does not identify a refined parent"))
    return pair[parent_id]
end

function reset_conservative_tree_subcycle_spatial_bank_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D)
    for pair in bank.pair_parent_ledgers
        for ledger in values(pair)
            reset_conservative_tree_subcycle_ledger_2d!(ledger)
        end
    end
    for packets in bank.fine_to_coarse_route_packets
        _zero_conservative_tree_route_packet_cache_2d!(packets)
    end
    return bank
end

function reset_conservative_tree_subcycle_spatial_pair_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        parent_level::Integer)
    pair = conservative_tree_subcycle_spatial_pair_ledgers_2d(
        bank, parent_level)
    for ledger in values(pair)
        reset_conservative_tree_subcycle_ledger_2d!(ledger)
    end
    parent = _check_conservative_tree_pair_level_2d(
        bank.schedule, parent_level)
    _zero_conservative_tree_route_packet_cache_2d!(
        bank.fine_to_coarse_route_packets[parent + 1])
    return bank
end

function _zero_conservative_tree_route_packet_cache_2d!(
        packets::Dict{Tuple{Int,Int},Vector{T}}) where T
    for values_by_substep in values(packets)
        fill!(values_by_substep, zero(T))
    end
    return packets
end

function _ensure_conservative_tree_route_packet_cache_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D{T},
        parent_level::Integer,
        dst_id::Integer,
        q::Integer) where T
    parent = _check_conservative_tree_pair_level_2d(
        bank.schedule, parent_level)
    packets = bank.fine_to_coarse_route_packets[parent + 1]
    key = (Int(dst_id), _check_d2q9_q(q))
    return get!(packets, key) do
        zeros(T, bank.schedule.ratio)
    end
end

function prepare_conservative_tree_subcycle_route_packet_cache_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        table::ConservativeTreeRouteTable2D)
    _check_conservative_tree_subcycle_route_table_2d(table)
    @inbounds for route_id in table.interface_routes
        route = table.routes[route_id]
        route.kind == COALESCE_FACE || route.kind == COALESCE_CORNER || continue
        route.dst == 0 && continue
        child = bank.spec.cells[route.src]
        child.level > 0 || continue
        parent = bank.spec.cells[child.parent]
        _ensure_conservative_tree_route_packet_cache_2d!(
            bank, parent.level, route.dst, route.q)
    end
    @inbounds for parent_level in 0:(bank.spec.max_level - 1)
        child_level = parent_level + 1
        for src_id in bank.inactive_refined_ids_by_level[child_level + 1]
            src = bank.spec.cells[src_id]
            src.active && continue
            for q in 1:9
                dst_id = _conservative_tree_inactive_parent_coalesce_dst_2d(
                    bank.spec, src_id, q)
                dst_id == 0 && continue
                _ensure_conservative_tree_route_packet_cache_2d!(
                    bank, parent_level, dst_id, q)
            end
        end
    end
    return bank
end

@inline function _conservative_tree_child_slot_2d(ix::Int, iy::Int)
    1 <= ix <= 2 || throw(ArgumentError("child ix must be 1 or 2"))
    1 <= iy <= 2 || throw(ArgumentError("child iy must be 1 or 2"))
    return ix + 2 * (iy - 1)
end

@inline function _conservative_tree_child_index_in_parent_2d(
        parent::ConservativeTreeCell2D,
        child::ConservativeTreeCell2D)
    child.level == parent.level + 1 ||
        throw(ArgumentError("child cell is not one level below parent"))
    ix = child.i - 2 * parent.i + 2
    iy = child.j - 2 * parent.j + 2
    1 <= ix <= 2 && 1 <= iy <= 2 ||
        throw(ArgumentError("child cell is not inside parent"))
    return ix, iy
end

function _check_conservative_tree_subcycle_spatial_F_2d(
        F::AbstractMatrix,
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D)
    _check_conservative_tree_F_2d(F, bank.spec)
    return nothing
end

function _check_conservative_tree_subcycle_route_table_2d(
        table::ConservativeTreeRouteTable2D)
    return table
end

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
    conservative_tree_subcycle_spatial_pair_ledgers_2d(bank, parent)
    return parent
end

function _check_subcycle_spatial_sync_up_event_2d(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        event::ConservativeTreeSubcycleEvent2D)
    parent = _check_subcycle_sync_up_event_2d(bank.schedule, event)
    conservative_tree_subcycle_spatial_pair_ledgers_2d(bank, parent)
    return parent
end

function _check_subcycle_spatial_child_advance_event_2d(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        event::ConservativeTreeSubcycleEvent2D)
    parent, substep = _check_subcycle_child_advance_event_2d(
        bank.schedule, event)
    conservative_tree_subcycle_spatial_pair_ledgers_2d(bank, parent)
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
    ledger = conservative_tree_subcycle_spatial_ledger_2d(bank, parent_id)
    ix, iy = _conservative_tree_child_index_in_parent_2d(parent, child)
    qi = _check_d2q9_q(route.q)
    packet = ledger.ratio * _subcycle_route_packet_2d(
        F, spec, route; alpha=alpha)

    @inbounds for substep in 1:ledger.ratio
        ledger.coarse_to_fine[ix, iy, qi, substep] += packet / ledger.ratio
    end
    return ledger
end

function conservative_tree_subcycle_accumulate_fine_to_coarse_route_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        F::AbstractMatrix,
        route::ConservativeTreeRoute2D,
        substep::Integer;
        alpha=1)
    _check_conservative_tree_subcycle_spatial_F_2d(F, bank)
    route.kind == COALESCE_FACE || route.kind == COALESCE_CORNER ||
        throw(ArgumentError("route must be a fine-to-coarse coalesce route"))

    spec = bank.spec
    child = spec.cells[route.src]
    child.level > 0 ||
        throw(ArgumentError("fine-to-coarse route source must have a parent"))
    parent_id = child.parent
    parent = spec.cells[parent_id]
    route.dst == 0 || spec.cells[route.dst].level == parent.level ||
        throw(ArgumentError("fine-to-coarse route destination level mismatch"))
    ledger = conservative_tree_subcycle_spatial_ledger_2d(bank, parent_id)
    step = _check_subcycle_step_2d(ledger, substep)
    qi = _check_d2q9_q(route.q)
    packet = _subcycle_route_packet_2d(
        F, spec, route; alpha=alpha) / ledger.ratio
    ledger.fine_to_coarse[qi, step] += packet
    route.dst != 0 ||
        throw(ArgumentError("fine-to-coarse route must have a spatial destination"))
    packets = _ensure_conservative_tree_route_packet_cache_2d!(
        bank, parent.level, route.dst, qi)
    packets[step] += packet
    return ledger
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
    child_level = parent_level + 1

    @inbounds for route_pos in table.coalesce_route_ranges_by_child_level[child_level + 1]
        route_id = table.interface_routes[route_pos]
        route = table.routes[route_id]
        conservative_tree_subcycle_accumulate_fine_to_coarse_route_2d!(
            bank, F, route, substep; alpha=alpha)
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

    @inbounds for src_id in bank.inactive_refined_ids_by_level[child_level + 1]
        bank.spec.cells[src_id].active && continue
        for q in 1:9
            dst_id, kind = _conservative_tree_inactive_parent_coalesce_route_spec_2d(
                bank.spec, src_id, q)
            dst_id == 0 && continue
            route = ConservativeTreeRoute2D(src_id, dst_id, q, 1.0, kind)
            conservative_tree_subcycle_accumulate_fine_to_coarse_route_2d!(
                bank, F, route, substep; alpha=alpha)
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
    ledger = conservative_tree_subcycle_spatial_ledger_2d(bank, parent_id)
    step = _check_subcycle_step_2d(ledger, substep)
    children = bank.spec.children[parent_id]
    children == (0, 0, 0, 0) &&
        throw(ArgumentError("parent_cell_id does not identify a refined parent"))

    @inbounds for iy in 1:2, ix in 1:2
        child_id = children[_conservative_tree_child_slot_2d(ix, iy)]
        child_id == 0 && continue
        for q in 1:9
            F[child_id, q] += ledger.coarse_to_fine[ix, iy, q, step]
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
    ledger = conservative_tree_subcycle_spatial_ledger_2d(bank, parent_id)
    parent = bank.spec.cells[parent_id]

    @inbounds for q in 1:9
        packet = zero(eltype(F))
        for substep in 1:ledger.ratio
            packet += ledger.fine_to_coarse[q, substep]
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
    packets = bank.fine_to_coarse_route_packets[parent_level + 1]
    if !isempty(packets)
        _check_conservative_tree_subcycle_spatial_F_2d(F, bank)
        for ((dst_id, q), substep_packets) in packets
            packet = sum(substep_packets)
            iszero(packet) && continue
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
        rho_wall=1)
    _check_conservative_tree_stream_args_2d(Fout, Fin, spec)

    @inbounds for route_pos in table.direct_route_ranges_by_level[level + 1]
        route_id = table.direct_routes[route_pos]
        route = table.routes[route_id]
        Fout[route.dst, route.q] += route.weight * Fin[route.src, route.q]
    end
    @inbounds for route_pos in table.boundary_route_ranges_by_level[level + 1]
        route_id = table.boundary_routes[route_pos]
        route = table.routes[route_id]
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
        alpha_f2c=1)
    _check_conservative_tree_stream_args_2d(Fout, Fin, spec)
    policy = _check_conservative_tree_boundary_policy_2d(boundary)
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
                u_south=u_south, u_north=u_north, rho_wall=rho_wall)
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
        pre_stream_level! = nothing,
        schedule = nothing,
        route_bank = nothing,
        state_bank = nothing,
        Fsource = nothing,
        Fscratch = nothing)
    _check_conservative_tree_stream_args_2d(Fout, Fin, spec)
    policy = _check_conservative_tree_boundary_policy_2d(boundary)
    if spec.max_level == 0
        return stream_conservative_tree_routes_F_2d!(
            Fout, Fin, spec, table; boundary=boundary, u_south=u_south,
            u_north=u_north, rho_wall=rho_wall)
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
    prepare_conservative_tree_subcycle_route_packet_cache_2d!(
        route_bank_run, table)
    reset_conservative_tree_subcycle_spatial_bank_2d!(route_bank_run)
    reset_conservative_tree_subcycle_buffer_bank_2d!(state_bank_run)
    conservative_tree_subcycle_store_active_owned_2d!(state_bank_run, Fin)

    Fsource_run = Fsource === nothing ? similar(Fout) : Fsource
    Fscratch_run = Fscratch === nothing ? similar(Fout) : Fscratch
    _check_conservative_tree_F_2d(Fsource_run, spec)
    _check_conservative_tree_F_2d(Fscratch_run, spec)

    for event in schedule_run.events
        if event.phase == :sync_down
            reset_conservative_tree_subcycle_spatial_pair_2d!(
                route_bank_run, event.src_level)
            parent_owned = state_bank_run.levels[event.src_level + 1].owned
            conservative_tree_subcycle_sync_down_routes_F_2d!(
                route_bank_run, event, parent_owned, table; alpha=alpha_c2f)
        elseif event.phase == :advance
            level = event.src_level
            buffers = state_bank_run.levels[level + 1]
            fill!(Fsource_run, zero(eltype(Fsource_run)))
            _copy_conservative_tree_level_rows_2d!(
                Fsource_run, buffers.owned, spec, level)
            conservative_tree_subcycle_apply_restriction_to_inactive_level_F_2d!(
                Fsource_run, state_bank_run, level)
            pre_stream_level! === nothing ||
                pre_stream_level!(Fsource_run, spec, level, event)

            if level > 0
                conservative_tree_subcycle_accumulate_advance_routes_F_2d!(
                    route_bank_run, event, Fsource_run, table; alpha=alpha_f2c)
                conservative_tree_subcycle_accumulate_inactive_parent_routes_F_2d!(
                    route_bank_run, event, Fsource_run; alpha=alpha_f2c)
            end

            fill!(Fscratch_run, zero(eltype(Fscratch_run)))
            _stream_conservative_tree_direct_level_routes_F_2d!(
                Fscratch_run, Fsource_run, spec, table, level, policy;
                u_south=u_south, u_north=u_north, rho_wall=rho_wall)
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
                buffers.owned, Fscratch_run, spec, level)
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

function reset_conservative_tree_subcycle_ledger_2d!(
        ledger::ConservativeTreeSubcycleLedger2D)
    ledger.coarse_to_fine .= 0
    ledger.fine_to_coarse .= 0
    return ledger
end

function conservative_tree_subcycle_weights_2d(ledger::ConservativeTreeSubcycleLedger2D{T}) where T
    ledger.ratio == 2 ||
        throw(ArgumentError("conservative-tree subcycle weights require ratio = 2"))
    return fill(inv(T(ledger.ratio)), ledger.ratio)
end

@inline function _check_subcycle_step_2d(ledger::ConservativeTreeSubcycleLedger2D,
                                         substep::Integer)
    step = Int(substep)
    1 <= step <= ledger.ratio ||
        throw(ArgumentError("substep must be inside 1:$(ledger.ratio)"))
    return step
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

function conservative_tree_subcycle_orientation_sums_2d(
        ledger::ConservativeTreeSubcycleLedger2D{T}) where T
    coarse_to_fine = zeros(T, 9)
    fine_to_coarse = zeros(T, 9)
    @inbounds for substep in 1:ledger.ratio, q in 1:9
        fine_to_coarse[q] += ledger.fine_to_coarse[q, substep]
        for j in 1:2, i in 1:2
            coarse_to_fine[q] += ledger.coarse_to_fine[i, j, q, substep]
        end
    end
    return (coarse_to_fine=coarse_to_fine, fine_to_coarse=fine_to_coarse)
end

function conservative_tree_subcycle_total_sums_2d(
        ledger::ConservativeTreeSubcycleLedger2D)
    sums = conservative_tree_subcycle_orientation_sums_2d(ledger)
    return (coarse_to_fine=sum(sums.coarse_to_fine),
            fine_to_coarse=sum(sums.fine_to_coarse))
end
