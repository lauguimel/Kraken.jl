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

struct ConservativeTreeWallPhaseScatter2D{T}
    source_mask::BitMatrix
    src_ids::Vector{Int}
    src_qs::Vector{Int}
    dst_ids::Vector{Int}
    dst_qs::Vector{Int}
    weights::Vector{T}
    event_mask_phases::Vector{Symbol}
    event_mask_ticks::Vector{Int}
    event_mask_levels::Vector{Int}
    event_mask_src_ids::Vector{Int}
    event_mask_src_qs::Vector{Int}
    sync_ticks::Vector{Int}
    sync_parent_levels::Vector{Int}
    sync_child_substeps::Vector{Int}
    sync_src_ids::Vector{Int}
    sync_src_qs::Vector{Int}
    sync_parent_ids::Vector{Int}
    sync_ixs::Vector{Int}
    sync_iys::Vector{Int}
    sync_dst_qs::Vector{Int}
    sync_weights::Vector{T}
    advance_ticks::Vector{Int}
    advance_levels::Vector{Int}
    advance_src_ids::Vector{Int}
    advance_src_qs::Vector{Int}
    advance_dst_ids::Vector{Int}
    advance_dst_qs::Vector{Int}
    advance_weights::Vector{T}
    reflux_ticks::Vector{Int}
    reflux_levels::Vector{Int}
    reflux_src_ids::Vector{Int}
    reflux_src_qs::Vector{Int}
    reflux_dst_ids::Vector{Int}
    reflux_dst_qs::Vector{Int}
    reflux_weights::Vector{T}
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
    wall_phase_scatter_cache::Vector{ConservativeTreeWallPhaseScatter2D{T}}
    wall_phase_scatter_cache_periodic_x::Vector{Bool}
    delayed_same_level_ticks::Vector{Int}
    delayed_same_level_levels::Vector{Int}
    delayed_same_level_dst_ids::Vector{Int}
    delayed_same_level_qs::Vector{Int}
    delayed_same_level_packets::Vector{T}
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
        ConservativeTreeWallPhaseScatter2D{T}[], Bool[],
        Int[], Int[], Int[], Int[], T[],
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
    empty!(bank.delayed_same_level_ticks)
    empty!(bank.delayed_same_level_levels)
    empty!(bank.delayed_same_level_dst_ids)
    empty!(bank.delayed_same_level_qs)
    empty!(bank.delayed_same_level_packets)
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

function _conservative_tree_phase_advance_periodic_x_wall_y_2d(
        i::Int,
        j::Int,
        q::Int,
        nx::Int,
        ny::Int;
        periodic_x::Bool=false)
    trial_i = i + d2q9_cx(q)
    trial_j = j + d2q9_cy(q)
    if periodic_x
        trial_i = mod1(trial_i, nx)
    elseif !(1 <= trial_i <= nx)
        return false, i, j, q
    end
    if !(1 <= trial_j <= ny)
        return true, i, j, d2q9_opposite(q)
    end
    return true, trial_i, trial_j, q
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

