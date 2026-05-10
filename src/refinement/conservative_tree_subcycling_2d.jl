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

struct ConservativeTreeSubcyclePackedLedgerPair2D{T}
    parent_ids::Vector{Int}
    coarse_to_fine::Array{T,5}
    fine_to_coarse::Array{T,3}
end

struct ConservativeTreeSubcycleRoutePacketCache2D{T}
    key_to_slot::Dict{Tuple{Int,Int},Int}
    dst_ids::Vector{Int}
    qs::Vector{Int}
    packets::Vector{T}
end

struct ConservativeTreeSubcycleSpatialLedgerBank2D{T}
    spec::ConservativeTreeSpec2D
    schedule::ConservativeTreeSubcycleSchedule2D
    ledger_pairs::Vector{ConservativeTreeSubcyclePackedLedgerPair2D{T}}
    parent_ledger_slot::Vector{Int}
    route_packet_caches::Vector{ConservativeTreeSubcycleRoutePacketCache2D{T}}
    route_packet_slot_by_route::Vector{Int}
    inactive_route_packet_slots_by_level::Vector{Matrix{Int}}
    inactive_route_packet_dsts_by_level::Vector{Matrix{Int}}
    inactive_route_packet_kinds_by_level::Vector{Matrix{RouteKind}}
    f2c_route_srcs_by_child_level::Vector{Vector{Int}}
    f2c_route_qs_by_child_level::Vector{Vector{Int}}
    f2c_route_weights_by_child_level::Vector{Vector{Float64}}
    f2c_route_parent_slots_by_child_level::Vector{Vector{Int}}
    f2c_route_packet_slots_by_child_level::Vector{Vector{Int}}
    inactive_f2c_route_srcs_by_child_level::Vector{Vector{Int}}
    inactive_f2c_route_qs_by_child_level::Vector{Vector{Int}}
    inactive_f2c_route_parent_slots_by_child_level::Vector{Vector{Int}}
    inactive_f2c_route_packet_slots_by_child_level::Vector{Vector{Int}}
    route_packet_cache_valid::Vector{Bool}
    route_packet_cache_route_objectid::Vector{UInt}
    route_packet_cache_periodic_x::Vector{Bool}
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
    route_packet_caches = [
        ConservativeTreeSubcycleRoutePacketCache2D{T}(
            Dict{Tuple{Int,Int},Int}(), Int[], Int[], T[])
        for _ in 1:spec.max_level
    ]
    refined_parent_ids_by_level = [Int[] for _ in 1:spec.max_level]
    inactive_refined_ids_by_level = [Int[] for _ in 0:spec.max_level]
    parent_ledger_slot = zeros(Int, length(spec.cells))

    @inbounds for (cell_id, cell) in pairs(spec.cells)
        children = spec.children[cell_id]
        children == (0, 0, 0, 0) && continue
        push!(inactive_refined_ids_by_level[cell.level + 1], cell_id)
        cell.level < spec.max_level || continue
        push!(refined_parent_ids_by_level[cell.level + 1], cell_id)
    end

    ledger_pairs = Vector{ConservativeTreeSubcyclePackedLedgerPair2D{T}}(
        undef, spec.max_level)
    for parent_level in 0:(spec.max_level - 1)
        parent_ids = refined_parent_ids_by_level[parent_level + 1]
        nparents = length(parent_ids)
        @inbounds for (slot, parent_id) in enumerate(parent_ids)
            parent_ledger_slot[parent_id] = slot
        end
        ledger_pairs[parent_level + 1] =
            ConservativeTreeSubcyclePackedLedgerPair2D{T}(
                parent_ids,
                zeros(T, 2, 2, 9, schedule.ratio, nparents),
                zeros(T, 9, schedule.ratio, nparents))
    end

    return ConservativeTreeSubcycleSpatialLedgerBank2D{T}(
        spec, schedule, ledger_pairs, parent_ledger_slot,
        route_packet_caches, Int[], Matrix{Int}[], Matrix{Int}[],
        Matrix{RouteKind}[],
        [Int[] for _ in 0:spec.max_level],
        [Int[] for _ in 0:spec.max_level],
        [Float64[] for _ in 0:spec.max_level],
        [Int[] for _ in 0:spec.max_level],
        [Int[] for _ in 0:spec.max_level],
        [Int[] for _ in 0:spec.max_level],
        [Int[] for _ in 0:spec.max_level],
        [Int[] for _ in 0:spec.max_level],
        [Int[] for _ in 0:spec.max_level],
        Bool[false], UInt[0], Bool[false],
        refined_parent_ids_by_level,
        inactive_refined_ids_by_level)
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
    pair = bank.ledger_pairs[parent + 1]
    ledgers = Vector{ConservativeTreeSubcycleLedger2D{eltype(pair.fine_to_coarse)}}(
        undef, length(pair.parent_ids))
    @inbounds for slot in eachindex(pair.parent_ids)
        ledgers[slot] = ConservativeTreeSubcycleLedger2D(
            bank.schedule.ratio,
            copy(@view(pair.coarse_to_fine[:, :, :, :, slot])),
            copy(@view(pair.fine_to_coarse[:, :, slot])))
    end
    return ledgers
end

function _conservative_tree_packed_ledger_pair_2d(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        parent_level::Integer)
    parent = _check_conservative_tree_pair_level_2d(
        bank.schedule, parent_level)
    return bank.ledger_pairs[parent + 1]
end

function _conservative_tree_packed_ledger_slot_2d(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        parent_cell_id::Integer)
    parent_id = Int(parent_cell_id)
    1 <= parent_id <= length(bank.spec.cells) ||
        throw(ArgumentError("parent_cell_id is outside the tree"))
    parent = bank.spec.cells[parent_id]
    _check_conservative_tree_pair_level_2d(bank.schedule, parent.level)
    slot = bank.parent_ledger_slot[parent_id]
    slot != 0 ||
        throw(ArgumentError("parent_cell_id does not identify a refined parent"))
    return slot
end

function conservative_tree_subcycle_spatial_ledger_2d(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        parent_cell_id::Integer)
    parent_id = Int(parent_cell_id)
    1 <= parent_id <= length(bank.spec.cells) ||
        throw(ArgumentError("parent_cell_id is outside the tree"))
    parent = bank.spec.cells[parent_id]
    pair = _conservative_tree_packed_ledger_pair_2d(bank, parent.level)
    slot = _conservative_tree_packed_ledger_slot_2d(bank, parent_id)
    return ConservativeTreeSubcycleLedger2D(
        bank.schedule.ratio,
        copy(@view(pair.coarse_to_fine[:, :, :, :, slot])),
        copy(@view(pair.fine_to_coarse[:, :, slot])))
end

function reset_conservative_tree_subcycle_spatial_bank_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D)
    for pair in bank.ledger_pairs
        fill!(pair.coarse_to_fine, zero(eltype(pair.coarse_to_fine)))
        fill!(pair.fine_to_coarse, zero(eltype(pair.fine_to_coarse)))
    end
    for cache in bank.route_packet_caches
        _zero_conservative_tree_route_packet_cache_2d!(cache)
    end
    return bank
end

function reset_conservative_tree_subcycle_spatial_pair_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        parent_level::Integer)
    parent = _check_conservative_tree_pair_level_2d(
        bank.schedule, parent_level)
    pair = bank.ledger_pairs[parent + 1]
    fill!(pair.coarse_to_fine, zero(eltype(pair.coarse_to_fine)))
    fill!(pair.fine_to_coarse, zero(eltype(pair.fine_to_coarse)))
    _zero_conservative_tree_route_packet_cache_2d!(
        bank.route_packet_caches[parent + 1])
    return bank
end

function _zero_conservative_tree_route_packet_cache_2d!(
        cache::ConservativeTreeSubcycleRoutePacketCache2D)
    fill!(cache.packets, zero(eltype(cache.packets)))
    return cache
end

function _ensure_conservative_tree_route_packet_cache_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D{T},
        parent_level::Integer,
        dst_id::Integer,
        q::Integer) where T
    parent = _check_conservative_tree_pair_level_2d(
        bank.schedule, parent_level)
    cache = bank.route_packet_caches[parent + 1]
    key = (Int(dst_id), _check_d2q9_q(q))
    slot = get(cache.key_to_slot, key, 0)
    if slot == 0
        push!(cache.dst_ids, key[1])
        push!(cache.qs, key[2])
        slot = length(cache.dst_ids)
        cache.key_to_slot[key] = slot
        old_len = length(cache.packets)
        resize!(cache.packets, old_len + bank.schedule.ratio)
        @inbounds for idx in (old_len + 1):length(cache.packets)
            cache.packets[idx] = zero(T)
        end
    end
    return slot
end

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
    qi = _check_d2q9_q(q)
    step = Int(substep)
    1 <= step <= bank.schedule.ratio ||
        throw(ArgumentError("substep must be inside 1:$(bank.schedule.ratio)"))
    packet_slot_i = Int(route_packet_slot)
    if interface_time_scaling == :level_native && kind == COALESCE_CORNER
        native_dst = _conservative_tree_level_native_corner_reflux_dst_2d(
            spec, child_id, qi, step, bank.schedule.ratio, dst_cell_id;
            periodic_x=periodic_x)
        if native_dst != dst_cell_id
            dst_cell_id = native_dst
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
            F, spec, child_id, qi, weight; alpha=alpha)
    pair.fine_to_coarse[qi, step, slot] += packet
    dst_cell_id != 0 ||
        throw(ArgumentError("fine-to-coarse route must have a spatial destination"))
    packet_slot = packet_slot_i
    if packet_slot == 0
        packet_slot = _ensure_conservative_tree_route_packet_cache_2d!(
            bank, parent.level, dst_cell_id, qi)
    end
    cache = bank.route_packet_caches[parent.level + 1]
    cache.packets[(packet_slot - 1) * bank.schedule.ratio + step] += packet
    return bank
end

function _conservative_tree_level_native_corner_reflux_dst_2d(
        spec::ConservativeTreeSpec2D,
        src_id::Int,
        q::Int,
        substep::Int,
        ratio::Int,
        fallback_dst::Int;
        periodic_x::Bool=false)
    src = spec.cells[src_id]
    src.level > 0 || return fallback_dst
    parent = spec.cells[src.parent]
    remaining_strides = ratio - substep + 1
    i_final = src.i + d2q9_cx(q) * remaining_strides
    j_final = src.j + d2q9_cy(q) * remaining_strides
    nx_child = _conservative_tree_level_size_2d(spec.Nx, src.level)
    ny_child = _conservative_tree_level_size_2d(spec.Ny, src.level)
    if periodic_x
        i_final = mod1(i_final, nx_child)
    elseif !(1 <= i_final <= nx_child)
        return fallback_dst
    end
    1 <= j_final <= ny_child || return fallback_dst
    dst_i = div(i_final + 1, 2)
    dst_j = div(j_final + 1, 2)
    dst_id = conservative_tree_cell_id_2d(
        spec, parent.level, dst_i, dst_j)
    dst_id == 0 && return fallback_dst
    spec.cells[dst_id].active || return fallback_dst
    return dst_id
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
        periodic_x::Bool=false)
    _check_conservative_tree_subcycle_spatial_F_2d(F, bank)
    return _conservative_tree_subcycle_accumulate_fine_to_coarse_packet_unchecked_2d!(
        bank, F, src_id, dst_id, q, weight, kind, substep,
        route_packet_slot; alpha=alpha,
        interface_time_scaling=interface_time_scaling,
        periodic_x=periodic_x)
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
        periodic_x::Bool=false)
    return _conservative_tree_subcycle_accumulate_fine_to_coarse_packet_2d!(
        bank, F, route.src, route.dst, route.q, route.weight, route.kind,
        substep, route_packet_slot; alpha=alpha,
        interface_time_scaling=interface_time_scaling,
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

function conservative_tree_subcycle_sync_down_routes_F_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        event::ConservativeTreeSubcycleEvent2D,
        F::AbstractMatrix,
        table::ConservativeTreeRouteTable2D;
        alpha=1,
        interface_time_scaling::Symbol=:leaf_equivalent,
        coarse_to_fine_prolongation::Symbol=:flat,
        periodic_x::Bool=false)
    parent_level = _check_subcycle_spatial_sync_down_event_2d(bank, event)
    _check_conservative_tree_subcycle_route_table_2d(table)
    _check_conservative_tree_subcycle_spatial_F_2d(F, bank)

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
        is_solid=nothing,
        coarse_to_fine_prolongation::Symbol=:flat,
        periodic_x::Bool=false)
    _check_conservative_tree_stream_args_2d(Fout, Fin, spec)
    prolongation = _check_conservative_tree_coarse_to_fine_prolongation_2d(
        coarse_to_fine_prolongation)

    if is_solid === nothing && prolongation != :limited_linear
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

    if is_solid === nothing
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
                interface_time_scaling=interface_time_scaling)
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
        is_solid=nothing)
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
                coarse_to_fine_prolongation == :limited_linear
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
            conservative_tree_subcycle_sync_down_routes_F_2d!(
                route_bank_run, event, Fparent, table; alpha=alpha_c2f,
                interface_time_scaling=interface_time_scaling,
                coarse_to_fine_prolongation=coarse_to_fine_prolongation,
                periodic_x=periodic_x)
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

            if level > 0
                conservative_tree_subcycle_accumulate_advance_routes_F_2d!(
                    route_bank_run, event, Fsource_run, table;
                    alpha=alpha_f2c,
                    interface_time_scaling=interface_time_scaling,
                    periodic_x=periodic_x)
                conservative_tree_subcycle_accumulate_inactive_parent_routes_F_2d!(
                    route_bank_run, event, Fsource_run; alpha=alpha_f2c,
                    interface_time_scaling=interface_time_scaling,
                    periodic_x=periodic_x)
            end

            fill!(Fscratch_run, zero(eltype(Fscratch_run)))
            _stream_conservative_tree_direct_level_routes_F_2d!(
                Fscratch_run, Fsource_run, spec, table, level, policy;
                u_south=u_south, u_north=u_north, rho_wall=rho_wall,
                is_solid=is_solid,
                coarse_to_fine_prolongation=coarse_to_fine_prolongation,
                periodic_x=periodic_x)
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
