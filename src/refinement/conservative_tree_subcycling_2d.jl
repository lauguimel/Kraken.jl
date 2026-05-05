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

    @inbounds for (cell_id, cell) in pairs(spec.cells)
        cell.level < spec.max_level || continue
        children = spec.children[cell_id]
        children == (0, 0, 0, 0) && continue
        pair_parent_ledgers[cell.level + 1][cell_id] =
            create_conservative_tree_subcycle_ledger_2d(T=T, ratio=schedule.ratio)
    end
    return ConservativeTreeSubcycleSpatialLedgerBank2D{T}(
        spec, schedule, pair_parent_ledgers)
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

@inline function _subcycle_route_packet_2d(F::AbstractMatrix,
                                           route::ConservativeTreeRoute2D)
    return route.weight * F[route.src, route.q]
end

function conservative_tree_subcycle_deposit_coarse_to_fine_route_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        F::AbstractMatrix,
        route::ConservativeTreeRoute2D)
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
    packet = ledger.ratio * _subcycle_route_packet_2d(F, route)

    @inbounds for substep in 1:ledger.ratio
        ledger.coarse_to_fine[ix, iy, qi, substep] += packet / ledger.ratio
    end
    return ledger
end

function conservative_tree_subcycle_accumulate_fine_to_coarse_route_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        F::AbstractMatrix,
        route::ConservativeTreeRoute2D,
        substep::Integer)
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
    ledger.fine_to_coarse[qi, step] +=
        _subcycle_route_packet_2d(F, route) / ledger.ratio
    return ledger
end

function conservative_tree_subcycle_sync_down_routes_F_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        event::ConservativeTreeSubcycleEvent2D,
        F::AbstractMatrix,
        table::ConservativeTreeRouteTable2D)
    parent_level = _check_subcycle_spatial_sync_down_event_2d(bank, event)
    _check_conservative_tree_subcycle_route_table_2d(table)
    _check_conservative_tree_subcycle_spatial_F_2d(F, bank)

    @inbounds for route_id in table.interface_routes
        route = table.routes[route_id]
        route.kind == SPLIT_FACE || route.kind == SPLIT_CORNER || continue
        bank.spec.cells[route.src].level == parent_level || continue
        bank.spec.cells[route.dst].level == parent_level + 1 || continue
        conservative_tree_subcycle_deposit_coarse_to_fine_route_2d!(
            bank, F, route)
    end
    return bank
end

function conservative_tree_subcycle_accumulate_advance_routes_F_2d!(
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        event::ConservativeTreeSubcycleEvent2D,
        F::AbstractMatrix,
        table::ConservativeTreeRouteTable2D)
    parent_level, substep = _check_subcycle_spatial_child_advance_event_2d(
        bank, event)
    _check_conservative_tree_subcycle_route_table_2d(table)
    _check_conservative_tree_subcycle_spatial_F_2d(F, bank)
    child_level = parent_level + 1

    @inbounds for route_id in table.interface_routes
        route = table.routes[route_id]
        route.kind == COALESCE_FACE || route.kind == COALESCE_CORNER || continue
        bank.spec.cells[route.src].level == child_level || continue
        conservative_tree_subcycle_accumulate_fine_to_coarse_route_2d!(
            bank, F, route, substep)
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
    pair = conservative_tree_subcycle_spatial_pair_ledgers_2d(
        bank, parent_level)
    for parent_id in keys(pair)
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
        boundary::Symbol=:skip)
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
            if policy == :bounceback
                F[parent_id, d2q9_opposite(q)] += packet
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
        boundary::Symbol=:skip)
    _check_conservative_tree_subcycle_spatial_F_2d(F, bank)
    pair = conservative_tree_subcycle_spatial_pair_ledgers_2d(
        bank, parent_level)
    for parent_id in keys(pair)
        conservative_tree_subcycle_apply_fine_to_coarse_F_2d!(
            F, bank, parent_id; boundary=boundary)
    end
    return F
end

function conservative_tree_subcycle_apply_sync_up_F_2d!(
        F::AbstractMatrix,
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        event::ConservativeTreeSubcycleEvent2D;
        boundary::Symbol=:skip)
    parent_level = _check_subcycle_spatial_sync_up_event_2d(bank, event)
    return conservative_tree_subcycle_apply_fine_to_coarse_pair_F_2d!(
        F, bank, parent_level; boundary=boundary)
end

function _stream_conservative_tree_direct_level_routes_F_2d!(
        Fout::AbstractMatrix,
        Fin::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        table::ConservativeTreeRouteTable2D,
        level::Int,
        policy::Symbol)
    _check_conservative_tree_stream_args_2d(Fout, Fin, spec)

    @inbounds for route_id in table.direct_routes
        route = table.routes[route_id]
        spec.cells[route.src].level == level || continue
        Fout[route.dst, route.q] += route.weight * Fin[route.src, route.q]
    end
    @inbounds for route_id in table.boundary_routes
        route = table.routes[route_id]
        spec.cells[route.src].level == level || continue
        if policy == :bounceback
            Fout[route.src, d2q9_opposite(route.q)] +=
                route.weight * Fin[route.src, route.q]
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

"""
    stream_conservative_tree_subcycled_routes_F_2d!(Fout, Fin, spec, table;
                                                    boundary=:skip)

CPU reference transport for the recursive AMR-D subcycle schedule. Same-level
routes are advanced only when their level receives an `:advance` event.
Coarse/fine interface routes are routed through spatial ledgers: split routes
are deposited on `:sync_down`, injected into child rows during each child
advance, and coalesce routes are accumulated during child advances then applied
on `:sync_up`.

This is still a transport skeleton: no collision, forcing, or physical open
boundary closure is performed here.
"""
function stream_conservative_tree_subcycled_routes_F_2d!(
        Fout::AbstractMatrix,
        Fin::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        table::ConservativeTreeRouteTable2D;
        boundary::Symbol=:skip)
    _check_conservative_tree_stream_args_2d(Fout, Fin, spec)
    policy = _check_conservative_tree_boundary_policy_2d(boundary)
    if spec.max_level == 0
        return stream_conservative_tree_routes_F_2d!(
            Fout, Fin, spec, table; boundary=boundary)
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
                bank, event, Fstate, table)
        elseif event.phase == :advance
            fill!(Fscratch, zero(eltype(Fscratch)))
            level = event.src_level
            if level > 0
                conservative_tree_subcycle_accumulate_advance_routes_F_2d!(
                    bank, event, Fstate, table)
            end
            _stream_conservative_tree_direct_level_routes_F_2d!(
                Fscratch, Fstate, spec, table, level, policy)
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
                Fpending, bank, event; boundary=boundary)
        else
            throw(ArgumentError("unknown subcycle event phase $(event.phase)"))
        end
    end

    copyto!(Fout, Fstate)
    return Fout
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
