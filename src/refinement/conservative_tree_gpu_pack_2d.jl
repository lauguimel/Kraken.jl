# GPU-ready route packs for conservative-tree D2Q9 AMR.
#
# The pack is deliberately a structure-of-arrays made only of primitive Julia
# arrays. CPU tests replay the pack directly; a later patch can transfer these
# arrays to CUDA/Metal without changing the route contract.

using Atomix
using KernelAbstractions

struct ConservativeTreeGPURoutePack2D{T}
    cells_per_block::Int32
    block_level::Vector{UInt8}
    block_first_cell::Vector{Int32}
    block_count::Vector{Int32}
    cell_logical_id::Vector{Int32}
    cell_level::Vector{UInt8}
    cell_i::Vector{Int32}
    cell_j::Vector{Int32}
    route_src::Vector{Int32}
    route_dst::Vector{Int32}
    route_q::Vector{UInt8}
    route_kind::Vector{UInt8}
    route_weight::Vector{T}
    direct_routes::Vector{Int32}
    interface_routes::Vector{Int32}
    boundary_routes::Vector{Int32}
end

struct ConservativeTreeGPUPullRoutePack2D{T,AFirst,ACount,ASrc,AQ,AW}
    n_cells::Int32
    pull_first::AFirst
    pull_count::ACount
    pull_src::ASrc
    pull_q::AQ
    pull_weight::AW
end

struct ConservativeTreeGPUCellPack2D{T,AId,ALevel,AActive,AVolume}
    n_cells::Int32
    n_active::Int32
    active_cell_ids::AId
    cell_level::ALevel
    cell_active::AActive
    cell_volume::AVolume
end

struct ConservativeTreeGPUParentChildPack2D{AParent,AChild}
    parent_level::Int32
    nparents::Int32
    parent_ids::AParent
    child_ids::AChild
end

struct ConservativeTreeGPUC2FDepositPack2D{T,AFirst,ACount,ASrc,AQ,AW}
    ratio::Int32
    nparents::Int32
    ledger_first::AFirst
    ledger_count::ACount
    src::ASrc
    q::AQ
    weight::AW
end

struct ConservativeTreeGPUF2CDepositPack2D{T,AFirst,ACount,ASrc,AQ,AW,ADst,AQCache}
    ratio::Int32
    nslots::Int32
    slot_first::AFirst
    slot_count::ACount
    src::ASrc
    q::AQ
    weight::AW
    cache_dst_ids::ADst
    cache_qs::AQCache
end

struct ConservativeTreeGPUBoundaryCorrectionPack2D{T,ADst,AQ,AValue}
    nentries::Int32
    dst::ADst
    q::AQ
    value::AValue
end

function _allocate_conservative_tree_backend_array_2d(
        backend, ::Type{T}, dims::Integer...) where T
    dims_tuple = Tuple(Int(d) for d in dims)
    try
        return KernelAbstractions.allocate(backend, T, dims_tuple...)
    catch err
        try
            return KernelAbstractions.allocate(
                backend, T, dims_tuple; unified=false)
        catch
            if occursin("Metal", string(typeof(backend)))
                metal = Base.require(Base.PkgId(
                    Base.UUID("dde4c033-4e86-420c-a63e-0dd931031962"),
                    "Metal"))
                return Base.invokelatest(
                    getproperty(metal, :MtlArray), zeros(T, dims_tuple))
            end
            rethrow(err)
        end
    end
end

function _copy_conservative_tree_gpu_array_2d(backend, host::AbstractArray{T}) where T
    if occursin("Metal", string(typeof(backend)))
        metal = Base.require(Base.PkgId(
            Base.UUID("dde4c033-4e86-420c-a63e-0dd931031962"),
            "Metal"))
        return Base.invokelatest(getproperty(metal, :MtlArray), host)
    end
    dev = _allocate_conservative_tree_backend_array_2d(backend, T, size(host)...)
    copyto!(dev, host)
    return dev
end

@inline function _conservative_tree_gpu_stream_boundary_policy_2d(
        boundary::Symbol)
    boundary in (:skip, :bounceback, :periodic_x_wall_y,
                 :periodic_x_moving_wall_y) ||
        throw(ArgumentError("GPU pull route pack currently supports boundary=:skip, :bounceback, :periodic_x_wall_y, or :periodic_x_moving_wall_y"))
    return boundary
end

function _logical_to_packed_index_2d(packed::ConservativeTreePackedTopology2D)
    logical_to_packed = zeros(Int32, length(packed.logical_cell_to_block))
    @inbounds for (packed_index, logical_id) in pairs(packed.packed_cell_ids)
        logical_to_packed[logical_id] = Int32(packed_index)
    end
    return logical_to_packed
end

"""
    pack_conservative_tree_gpu_routes_2d(topology; cells_per_block=128, T=Float32)

Convert a conservative-tree topology into primitive route arrays suitable for
GPU transfer. Route destinations use packed-cell index `0` for boundaries.
"""
function pack_conservative_tree_gpu_routes_2d(
        topology::ConservativeTreeTopology2D;
        cells_per_block::Integer=128,
        T::Type{<:Real}=Float32)
    packed = pack_conservative_tree_topology_2d(topology;
                                                cells_per_block=cells_per_block)
    logical_to_packed = _logical_to_packed_index_2d(packed)

    block_level = UInt8[UInt8(block.level) for block in packed.blocks]
    block_first_cell = Int32[Int32(block.first_cell) for block in packed.blocks]
    block_count = Int32[Int32(block.count) for block in packed.blocks]

    cell_logical_id = Int32[Int32(id) for id in packed.packed_cell_ids]
    cell_level = Vector{UInt8}(undef, length(packed.packed_cell_ids))
    cell_i = Vector{Int32}(undef, length(packed.packed_cell_ids))
    cell_j = Vector{Int32}(undef, length(packed.packed_cell_ids))
    @inbounds for (packed_index, logical_id) in pairs(packed.packed_cell_ids)
        cell = topology.cells[logical_id]
        cell_level[packed_index] = UInt8(cell.level)
        cell_i[packed_index] = Int32(cell.i)
        cell_j[packed_index] = Int32(cell.j)
    end

    n_routes = length(topology.routes)
    route_src = Vector{Int32}(undef, n_routes)
    route_dst = Vector{Int32}(undef, n_routes)
    route_q = Vector{UInt8}(undef, n_routes)
    route_kind = Vector{UInt8}(undef, n_routes)
    route_weight = Vector{T}(undef, n_routes)

    @inbounds for (route_index, route) in pairs(topology.routes)
        src = logical_to_packed[route.src]
        src > 0 || throw(ArgumentError("route source is not packed"))
        route_src[route_index] = src
        route_dst[route_index] =
            route.kind == ROUTE_BOUNDARY ? Int32(0) : logical_to_packed[route.dst]
        route_q[route_index] = UInt8(route.q)
        route_kind[route_index] = UInt8(route.kind)
        route_weight[route_index] = T(route.weight)
    end

    return ConservativeTreeGPURoutePack2D{T}(
        Int32(packed.cells_per_block),
        block_level, block_first_cell, block_count,
        cell_logical_id, cell_level, cell_i, cell_j,
        route_src, route_dst, route_q, route_kind, route_weight,
        Int32.(packed.direct_routes),
        Int32.(packed.interface_routes),
        Int32.(packed.boundary_routes))
end

function conservative_tree_gpu_route_weight_sums_2d(
        pack::ConservativeTreeGPURoutePack2D{T}) where T
    sums = Dict{Tuple{Int32,UInt8},T}()
    @inbounds for route_index in eachindex(pack.route_src)
        key = (pack.route_src[route_index], pack.route_q[route_index])
        sums[key] = get(sums, key, zero(T)) + pack.route_weight[route_index]
    end
    return sums
end

function _pack_conservative_tree_gpu_pull_route_ids_2d(
        spec::ConservativeTreeSpec2D,
        table::ConservativeTreeRouteTable2D,
        route_ids;
        boundary::Symbol=:skip,
        T::Type{<:Real}=Float32)
    policy = _conservative_tree_gpu_stream_boundary_policy_2d(boundary)
    n_cells = length(spec.cells)
    buckets = [Tuple{Int32,UInt8,T}[] for _ in 1:(n_cells * 9)]

    @inbounds for route_id in route_ids
        route = table.routes[route_id]
        dst = route.dst
        dst_q = route.q
        if dst == 0
            policy == :skip && continue
            if policy == :periodic_x_wall_y
                d2q9_cy(route.q) != 0 ||
                    throw(ArgumentError("periodic-x wall-y GPU pack received an x-boundary route; rebuild the route table with periodic_x=true"))
            end
            dst = route.src
            dst_q = d2q9_opposite(route.q)
        end
        1 <= dst <= n_cells ||
            throw(ArgumentError("route destination is outside the spec"))
        bucket = (dst - 1) * 9 + dst_q
        push!(buckets[bucket],
              (Int32(route.src), UInt8(route.q), T(route.weight)))
    end

    pull_first = zeros(Int32, n_cells, 9)
    pull_count = zeros(Int32, n_cells, 9)
    n_entries = sum(length, buckets)
    pull_src = Vector{Int32}(undef, n_entries)
    pull_q = Vector{UInt8}(undef, n_entries)
    pull_weight = Vector{T}(undef, n_entries)

    cursor = 1
    @inbounds for cell in 1:n_cells, q in 1:9
        entries = buckets[(cell - 1) * 9 + q]
        if !isempty(entries)
            pull_first[cell, q] = Int32(cursor)
            pull_count[cell, q] = Int32(length(entries))
            for entry in entries
                pull_src[cursor] = entry[1]
                pull_q[cursor] = entry[2]
                pull_weight[cursor] = entry[3]
                cursor += 1
            end
        end
    end

    return ConservativeTreeGPUPullRoutePack2D{T,typeof(pull_first),
        typeof(pull_count),typeof(pull_src),typeof(pull_q),
        typeof(pull_weight)}(Int32(n_cells), pull_first, pull_count,
                             pull_src, pull_q, pull_weight)
end

"""
    pack_conservative_tree_gpu_pull_routes_2d(spec, table; boundary=:skip, T=Float32)

Build a no-atomic pull-stream pack for a conservative-tree route table. Each
output `(cell, q)` owns a short CSR list of incoming packets
`weight * Fin[src, src_q]`. Stationary wall boundary reflection is encoded as
normal pull entries to `opposite(q)` in the source cell.
"""
function pack_conservative_tree_gpu_pull_routes_2d(
        spec::ConservativeTreeSpec2D,
        table::ConservativeTreeRouteTable2D;
        boundary::Symbol=:skip,
        T::Type{<:Real}=Float32)
    return _pack_conservative_tree_gpu_pull_route_ids_2d(
        spec, table, eachindex(table.routes); boundary=boundary, T=T)
end

"""
    pack_conservative_tree_gpu_direct_level_pull_routes_2d(spec, table, level;
                                                           boundary=:skip,
                                                           T=Float32)

Build the no-atomic pull-stream pack for the scheduler's direct route pass at
one active level. Interface routes are deliberately excluded; coarse/fine
packets remain owned by the subcycling ledgers.
"""
function pack_conservative_tree_gpu_direct_level_pull_routes_2d(
        spec::ConservativeTreeSpec2D,
        table::ConservativeTreeRouteTable2D,
        level::Integer;
        boundary::Symbol=:skip,
        T::Type{<:Real}=Float32)
    l = Int(level)
    0 <= l <= spec.max_level ||
        throw(ArgumentError("level is outside the conservative-tree spec"))
    route_ids = Int[]
    @inbounds for route_pos in table.direct_route_ranges_by_level[l + 1]
        push!(route_ids, table.direct_routes[route_pos])
    end
    @inbounds for route_pos in table.boundary_route_ranges_by_level[l + 1]
        push!(route_ids, table.boundary_routes[route_pos])
    end
    return _pack_conservative_tree_gpu_pull_route_ids_2d(
        spec, table, route_ids; boundary=boundary, T=T)
end

function transfer_conservative_tree_gpu_pull_pack_2d(
        pack::ConservativeTreeGPUPullRoutePack2D{T},
        backend) where T
    pull_first = _copy_conservative_tree_gpu_array_2d(backend, pack.pull_first)
    pull_count = _copy_conservative_tree_gpu_array_2d(backend, pack.pull_count)
    pull_src = _copy_conservative_tree_gpu_array_2d(backend, pack.pull_src)
    pull_q = _copy_conservative_tree_gpu_array_2d(backend, pack.pull_q)
    pull_weight = _copy_conservative_tree_gpu_array_2d(backend,
                                                       pack.pull_weight)
    return ConservativeTreeGPUPullRoutePack2D{T,typeof(pull_first),
        typeof(pull_count),typeof(pull_src),typeof(pull_q),
        typeof(pull_weight)}(pack.n_cells, pull_first, pull_count,
                             pull_src, pull_q, pull_weight)
end

function pack_conservative_tree_gpu_cells_2d(
        spec::ConservativeTreeSpec2D;
        T::Type{<:Real}=Float32)
    n_cells = length(spec.cells)
    active_cell_ids = Int32.(spec.active_cells)
    cell_level = Vector{UInt8}(undef, n_cells)
    cell_active = Vector{UInt8}(undef, n_cells)
    cell_volume = Vector{T}(undef, n_cells)
    @inbounds for (cell_id, cell) in pairs(spec.cells)
        cell_level[cell_id] = UInt8(cell.level)
        cell_active[cell_id] = cell.active ? UInt8(1) : UInt8(0)
        cell_volume[cell_id] = T(cell.metrics.volume)
    end
    return ConservativeTreeGPUCellPack2D{T,typeof(active_cell_ids),
        typeof(cell_level),typeof(cell_active),typeof(cell_volume)}(
            Int32(n_cells), Int32(length(active_cell_ids)), active_cell_ids,
            cell_level, cell_active, cell_volume)
end

function transfer_conservative_tree_gpu_cell_pack_2d(
        pack::ConservativeTreeGPUCellPack2D{T},
        backend) where T
    active_cell_ids = _copy_conservative_tree_gpu_array_2d(
        backend, pack.active_cell_ids)
    cell_level = _copy_conservative_tree_gpu_array_2d(
        backend, pack.cell_level)
    cell_active = _copy_conservative_tree_gpu_array_2d(
        backend, pack.cell_active)
    cell_volume = _copy_conservative_tree_gpu_array_2d(
        backend, pack.cell_volume)
    return ConservativeTreeGPUCellPack2D{T,typeof(active_cell_ids),
        typeof(cell_level),typeof(cell_active),typeof(cell_volume)}(
            pack.n_cells, pack.n_active, active_cell_ids, cell_level,
            cell_active, cell_volume)
end

function pack_conservative_tree_gpu_parent_children_2d(
        spec::ConservativeTreeSpec2D,
        parent_level::Integer)
    l = Int(parent_level)
    0 <= l < spec.max_level ||
        throw(ArgumentError("parent_level must identify an adjacent level pair"))
    parent_ids = Int32[]
    @inbounds for (cell_id, cell) in pairs(spec.cells)
        cell.level == l || continue
        spec.children[cell_id] == (0, 0, 0, 0) && continue
        push!(parent_ids, Int32(cell_id))
    end
    child_ids = zeros(Int32, 4, length(parent_ids))
    @inbounds for (slot, parent_id) in pairs(parent_ids)
        children = spec.children[Int(parent_id)]
        for child_slot in 1:4
            child_ids[child_slot, slot] = Int32(children[child_slot])
        end
    end
    return ConservativeTreeGPUParentChildPack2D{typeof(parent_ids),
        typeof(child_ids)}(Int32(l), Int32(length(parent_ids)), parent_ids,
                           child_ids)
end

function transfer_conservative_tree_gpu_parent_child_pack_2d(
        pack::ConservativeTreeGPUParentChildPack2D,
        backend)
    parent_ids = _copy_conservative_tree_gpu_array_2d(backend, pack.parent_ids)
    child_ids = _copy_conservative_tree_gpu_array_2d(backend, pack.child_ids)
    return ConservativeTreeGPUParentChildPack2D{typeof(parent_ids),
        typeof(child_ids)}(pack.parent_level, pack.nparents, parent_ids,
                           child_ids)
end

@inline function _conservative_tree_gpu_c2f_bucket_2d(
        ix::Int, iy::Int, q::Int, slot::Int)
    child_slot = ix + 2 * (iy - 1)
    return (slot - 1) * 36 + (child_slot - 1) * 9 + q
end

function pack_conservative_tree_gpu_coarse_to_fine_deposit_2d(
        spec::ConservativeTreeSpec2D,
        table::ConservativeTreeRouteTable2D,
        parent_pack::ConservativeTreeGPUParentChildPack2D,
        parent_level::Integer;
        ratio::Integer=2,
        interface_time_scaling::Symbol=:leaf_equivalent,
        T::Type{<:Real}=Float32)
    l = Int(parent_level)
    l == Int(parent_pack.parent_level) ||
        throw(ArgumentError("parent_level must match parent_pack"))
    r = Int(ratio)
    r >= 1 || throw(ArgumentError("ratio must be positive"))
    interface_time_scaling in (:leaf_equivalent, :level_native) ||
        throw(ArgumentError("interface_time_scaling must be :leaf_equivalent or :level_native"))
    per_substep_factor = interface_time_scaling == :leaf_equivalent ?
                         one(T) : inv(T(r))

    parent_slot = zeros(Int, length(spec.cells))
    @inbounds for (slot, parent_id) in pairs(parent_pack.parent_ids)
        parent_slot[Int(parent_id)] = slot
    end

    nbuckets = Int(parent_pack.nparents) * 4 * 9
    buckets = [Tuple{Int32,UInt8,T}[] for _ in 1:nbuckets]
    @inbounds for route_pos in table.split_route_ranges_by_parent_level[l + 1]
        route_id = table.interface_routes[route_pos]
        route = table.routes[route_id]
        route.kind == SPLIT_FACE || route.kind == SPLIT_CORNER || continue
        child = spec.cells[route.dst]
        parent_id = child.parent
        slot = parent_slot[parent_id]
        slot != 0 ||
            throw(ArgumentError("coarse-to-fine route parent missing from GPU parent pack"))
        parent = spec.cells[parent_id]
        ix = child.i - 2 * parent.i + 2
        iy = child.j - 2 * parent.j + 2
        1 <= ix <= 2 && 1 <= iy <= 2 ||
            throw(ArgumentError("coarse-to-fine route child is outside its parent"))
        bucket = _conservative_tree_gpu_c2f_bucket_2d(
            ix, iy, route.q, slot)
        push!(buckets[bucket],
              (Int32(route.src), UInt8(route.q),
               T(route.weight) * per_substep_factor))
    end

    ledger_first = zeros(Int32, 2, 2, 9, Int(parent_pack.nparents))
    ledger_count = zeros(Int32, 2, 2, 9, Int(parent_pack.nparents))
    n_entries = sum(length, buckets)
    src = Vector{Int32}(undef, n_entries)
    q = Vector{UInt8}(undef, n_entries)
    weight = Vector{T}(undef, n_entries)

    cursor = 1
    @inbounds for slot in 1:Int(parent_pack.nparents), qidx in 1:9,
        iy in 1:2, ix in 1:2
        entries = buckets[_conservative_tree_gpu_c2f_bucket_2d(
            ix, iy, qidx, slot)]
        if !isempty(entries)
            ledger_first[ix, iy, qidx, slot] = Int32(cursor)
            ledger_count[ix, iy, qidx, slot] = Int32(length(entries))
            for entry in entries
                src[cursor] = entry[1]
                q[cursor] = entry[2]
                weight[cursor] = entry[3]
                cursor += 1
            end
        end
    end

    return ConservativeTreeGPUC2FDepositPack2D{T,typeof(ledger_first),
        typeof(ledger_count),typeof(src),typeof(q),typeof(weight)}(
            Int32(r), parent_pack.nparents, ledger_first, ledger_count,
            src, q, weight)
end

function transfer_conservative_tree_gpu_c2f_deposit_pack_2d(
        pack::ConservativeTreeGPUC2FDepositPack2D{T},
        backend) where T
    ledger_first = _copy_conservative_tree_gpu_array_2d(
        backend, pack.ledger_first)
    ledger_count = _copy_conservative_tree_gpu_array_2d(
        backend, pack.ledger_count)
    src = _copy_conservative_tree_gpu_array_2d(backend, pack.src)
    q = _copy_conservative_tree_gpu_array_2d(backend, pack.q)
    weight = _copy_conservative_tree_gpu_array_2d(backend, pack.weight)
    return ConservativeTreeGPUC2FDepositPack2D{T,typeof(ledger_first),
        typeof(ledger_count),typeof(src),typeof(q),typeof(weight)}(
            pack.ratio, pack.nparents, ledger_first, ledger_count,
            src, q, weight)
end

@inline function _conservative_tree_gpu_f2c_bucket_2d(
        substep::Int, slot::Int, ratio::Int)
    return (slot - 1) * ratio + substep
end

function _push_conservative_tree_gpu_f2c_deposit_route_2d!(
        buckets,
        route::ConservativeTreeRoute2D,
        slot::Integer,
        ratio::Integer,
        time_factor;
        T::Type{<:Real})
    slot_i = Int(slot)
    slot_i > 0 ||
        throw(ArgumentError("fine-to-coarse route is missing a packet-cache slot"))
    @inbounds for substep in 1:Int(ratio)
        push!(buckets[_conservative_tree_gpu_f2c_bucket_2d(
                  substep, slot_i, Int(ratio))],
              (Int32(route.src), UInt8(route.q),
               T(route.weight) * time_factor))
    end
    return buckets
end

function pack_conservative_tree_gpu_fine_to_coarse_deposit_2d(
        spec::ConservativeTreeSpec2D,
        table::ConservativeTreeRouteTable2D,
        bank::ConservativeTreeSubcycleSpatialLedgerBank2D,
        child_level::Integer;
        include_inactive_parents::Bool=true,
        interface_time_scaling::Symbol=:leaf_equivalent,
        periodic_x::Bool=false,
        T::Type{<:Real}=Float32)
    bank.spec === spec || throw(ArgumentError("bank must belong to spec"))
    _check_conservative_tree_subcycle_route_table_2d(table)
    child = Int(child_level)
    1 <= child <= spec.max_level ||
        throw(ArgumentError("child_level must identify an adjacent level pair"))
    parent_level = child - 1
    r = Int(bank.schedule.ratio)
    r >= 1 || throw(ArgumentError("subcycle ratio must be positive"))
    mode = _check_conservative_tree_interface_time_scaling_2d(
        interface_time_scaling)
    prepare_conservative_tree_subcycle_route_packet_cache_2d!(
        bank, table; periodic_x=periodic_x)

    cache = bank.route_packet_caches[parent_level + 1]
    slot_by_key = Dict{Tuple{Int,Int},Int}()
    cache_dst_ids = Int[]
    cache_qs = Int[]
    @inbounds for slot in eachindex(cache.dst_ids)
        dst_id = Int(cache.dst_ids[slot])
        qidx = Int(cache.qs[slot])
        slot_by_key[(dst_id, qidx)] = slot
        push!(cache_dst_ids, dst_id)
        push!(cache_qs, qidx)
    end

    function ensure_slot(dst_id::Int, qidx::Int)
        key = (dst_id, qidx)
        existing = get(slot_by_key, key, 0)
        existing != 0 && return existing
        slot = length(cache_dst_ids) + 1
        slot_by_key[key] = slot
        push!(cache_dst_ids, dst_id)
        push!(cache_qs, qidx)
        return slot
    end

    buckets = Dict{Tuple{Int,Int},Vector{Tuple{Int32,UInt8,T}}}()
    time_factor = mode == :leaf_equivalent ? inv(T(r)) : one(T)

    function push_entry!(slot::Int, substep::Int, src_id::Int,
                         qidx::Int, weight)
        key = (slot, substep)
        entries = get!(buckets, key, Tuple{Int32,UInt8,T}[])
        push!(entries, (Int32(src_id), UInt8(qidx), T(weight)))
        return nothing
    end

    @inbounds for route_pos in table.coalesce_route_ranges_by_child_level[child + 1]
        route_id = table.interface_routes[route_pos]
        route = table.routes[route_id]
        (route.kind == COALESCE_FACE || route.kind == COALESCE_CORNER) ||
            continue
        route.dst == 0 && continue
        for substep in 1:r
            dst_id = route.dst
            if mode == :level_native && route.kind == COALESCE_CORNER
                dst_id = _conservative_tree_level_native_corner_reflux_dst_2d(
                    spec, route.src, route.q, substep, r, dst_id;
                    periodic_x=periodic_x)
            end
            dst_id == 0 && continue
            slot = ensure_slot(Int(dst_id), Int(route.q))
            push_entry!(slot, substep, route.src, route.q,
                        T(route.weight) * time_factor)
        end
    end

    if include_inactive_parents
        inactive_ids = bank.inactive_refined_ids_by_level[child + 1]
        @inbounds for (local_idx, src_id) in enumerate(inactive_ids)
            spec.cells[src_id].active && continue
            for qidx in 1:9
                dst_id, kind =
                    _conservative_tree_inactive_parent_coalesce_route_spec_2d(
                    spec, src_id, qidx; periodic_x=periodic_x)
                dst_id == 0 && continue
                for substep in 1:r
                    native_dst = dst_id
                    if mode == :level_native && kind == COALESCE_CORNER
                        native_dst = _conservative_tree_level_native_corner_reflux_dst_2d(
                            spec, src_id, qidx, substep, r, dst_id;
                            periodic_x=periodic_x)
                    end
                    native_dst == 0 && continue
                    slot = ensure_slot(Int(native_dst), qidx)
                    push_entry!(slot, substep, src_id, qidx, time_factor)
                end
            end
        end
    end

    nslots = length(cache_dst_ids)
    slot_first = zeros(Int32, r, nslots)
    slot_count = zeros(Int32, r, nslots)
    n_entries = sum(length, values(buckets))
    src = Vector{Int32}(undef, n_entries)
    q = Vector{UInt8}(undef, n_entries)
    weight = Vector{T}(undef, n_entries)
    cursor = 1
    @inbounds for slot in 1:nslots, substep in 1:r
        entries = get(buckets, (slot, substep), Tuple{Int32,UInt8,T}[])
        if !isempty(entries)
            slot_first[substep, slot] = Int32(cursor)
            slot_count[substep, slot] = Int32(length(entries))
            for entry in entries
                src[cursor] = entry[1]
                q[cursor] = entry[2]
                weight[cursor] = entry[3]
                cursor += 1
            end
        end
    end

    cache_dst_ids = Int32.(cache_dst_ids)
    cache_qs = UInt8.(cache_qs)
    return ConservativeTreeGPUF2CDepositPack2D{T,typeof(slot_first),
        typeof(slot_count),typeof(src),typeof(q),typeof(weight),
        typeof(cache_dst_ids),typeof(cache_qs)}(
            Int32(r), Int32(nslots), slot_first, slot_count, src, q, weight,
            cache_dst_ids, cache_qs)
end

function transfer_conservative_tree_gpu_f2c_deposit_pack_2d(
        pack::ConservativeTreeGPUF2CDepositPack2D{T},
        backend) where T
    slot_first = _copy_conservative_tree_gpu_array_2d(
        backend, pack.slot_first)
    slot_count = _copy_conservative_tree_gpu_array_2d(
        backend, pack.slot_count)
    src = _copy_conservative_tree_gpu_array_2d(backend, pack.src)
    q = _copy_conservative_tree_gpu_array_2d(backend, pack.q)
    weight = _copy_conservative_tree_gpu_array_2d(backend, pack.weight)
    cache_dst_ids = _copy_conservative_tree_gpu_array_2d(
        backend, pack.cache_dst_ids)
    cache_qs = _copy_conservative_tree_gpu_array_2d(backend, pack.cache_qs)
    return ConservativeTreeGPUF2CDepositPack2D{T,typeof(slot_first),
        typeof(slot_count),typeof(src),typeof(q),typeof(weight),
        typeof(cache_dst_ids),typeof(cache_qs)}(
            pack.ratio, pack.nslots, slot_first, slot_count, src, q, weight,
            cache_dst_ids, cache_qs)
end

@inline function _conservative_tree_gpu_level_row_selected_2d(
        cell_level, cell_active, cell::Int, level::UInt8, active_only::Bool)
    cell_level[cell] == level || return false
    active_only && cell_active[cell] == UInt8(0) && return false
    return true
end

@kernel function _copy_conservative_tree_gpu_level_rows_2d_kernel!(
        Fdst, @Const(Fsrc), @Const(cell_level), @Const(cell_active),
        n_cells::Int32, level::UInt8, active_only::Bool)
    linear = @index(Global)
    total = Int(n_cells) * 9
    @inbounds if linear <= total
        cell = (linear - 1) ÷ 9 + 1
        q = (linear - 1) % 9 + 1
        if _conservative_tree_gpu_level_row_selected_2d(
                cell_level, cell_active, cell, level, active_only)
            Fdst[cell, q] = Fsrc[cell, q]
        end
    end
end

function copy_conservative_tree_gpu_level_rows_2d!(
        Fdst::AbstractMatrix,
        Fsrc::AbstractMatrix,
        pack::ConservativeTreeGPUCellPack2D,
        level::Integer;
        active_only::Bool=false,
        sync::Bool=true)
    0 <= Int(level) <= typemax(UInt8) ||
        throw(ArgumentError("level must fit in UInt8"))
    size(Fdst, 1) == Int(pack.n_cells) && size(Fsrc, 1) == Int(pack.n_cells) ||
        throw(ArgumentError("F matrices must match pack.n_cells"))
    size(Fdst, 2) == 9 && size(Fsrc, 2) == 9 ||
        throw(ArgumentError("F matrices must have 9 D2Q9 populations"))
    backend = KernelAbstractions.get_backend(Fdst)
    kernel! = _copy_conservative_tree_gpu_level_rows_2d_kernel!(backend)
    kernel!(Fdst, Fsrc, pack.cell_level, pack.cell_active, pack.n_cells,
            UInt8(level), active_only; ndrange=Int(pack.n_cells) * 9)
    sync && KernelAbstractions.synchronize(backend)
    return Fdst
end

@kernel function _add_and_clear_conservative_tree_gpu_level_rows_2d_kernel!(
        Fdst, Fpending, @Const(cell_level), @Const(cell_active),
        n_cells::Int32, level::UInt8, active_only::Bool)
    linear = @index(Global)
    total = Int(n_cells) * 9
    @inbounds if linear <= total
        cell = (linear - 1) ÷ 9 + 1
        q = (linear - 1) % 9 + 1
        if _conservative_tree_gpu_level_row_selected_2d(
                cell_level, cell_active, cell, level, active_only)
            Fdst[cell, q] += Fpending[cell, q]
            Fpending[cell, q] = zero(eltype(Fpending))
        end
    end
end

function add_and_clear_conservative_tree_gpu_level_rows_2d!(
        Fdst::AbstractMatrix,
        Fpending::AbstractMatrix,
        pack::ConservativeTreeGPUCellPack2D,
        level::Integer;
        active_only::Bool=false,
        sync::Bool=true)
    0 <= Int(level) <= typemax(UInt8) ||
        throw(ArgumentError("level must fit in UInt8"))
    size(Fdst, 1) == Int(pack.n_cells) &&
        size(Fpending, 1) == Int(pack.n_cells) ||
        throw(ArgumentError("F matrices must match pack.n_cells"))
    size(Fdst, 2) == 9 && size(Fpending, 2) == 9 ||
        throw(ArgumentError("F matrices must have 9 D2Q9 populations"))
    backend = KernelAbstractions.get_backend(Fdst)
    kernel! = _add_and_clear_conservative_tree_gpu_level_rows_2d_kernel!(backend)
    kernel!(Fdst, Fpending, pack.cell_level, pack.cell_active, pack.n_cells,
            UInt8(level), active_only; ndrange=Int(pack.n_cells) * 9)
    sync && KernelAbstractions.synchronize(backend)
    return Fdst
end

@kernel function _zero_conservative_tree_gpu_level_rows_2d_kernel!(
        F, @Const(cell_level), @Const(cell_active), n_cells::Int32,
        level::UInt8, active_only::Bool)
    linear = @index(Global)
    total = Int(n_cells) * 9
    @inbounds if linear <= total
        cell = (linear - 1) ÷ 9 + 1
        q = (linear - 1) % 9 + 1
        if _conservative_tree_gpu_level_row_selected_2d(
                cell_level, cell_active, cell, level, active_only)
            F[cell, q] = zero(eltype(F))
        end
    end
end

function zero_conservative_tree_gpu_level_rows_2d!(
        F::AbstractMatrix,
        pack::ConservativeTreeGPUCellPack2D,
        level::Integer;
        active_only::Bool=false,
        sync::Bool=true)
    0 <= Int(level) <= typemax(UInt8) ||
        throw(ArgumentError("level must fit in UInt8"))
    size(F, 1) == Int(pack.n_cells) && size(F, 2) == 9 ||
        throw(ArgumentError("F must match the conservative-tree cell pack"))
    backend = KernelAbstractions.get_backend(F)
    kernel! = _zero_conservative_tree_gpu_level_rows_2d_kernel!(backend)
    kernel!(F, pack.cell_level, pack.cell_active, pack.n_cells,
            UInt8(level), active_only; ndrange=Int(pack.n_cells) * 9)
    sync && KernelAbstractions.synchronize(backend)
    return F
end

@kernel function _restrict_conservative_tree_gpu_level_2d_kernel!(
        restrict_to_parent, @Const(owned), @Const(child_restrict),
        @Const(parent_ids), @Const(child_ids), @Const(cell_active),
        nparents::Int32)
    linear = @index(Global)
    total = Int(nparents) * 9
    @inbounds if linear <= total
        slot = (linear - 1) ÷ 9 + 1
        q = (linear - 1) % 9 + 1
        parent_id = Int(parent_ids[slot])
        acc = zero(eltype(restrict_to_parent))
        for child_slot in 1:4
            child_id = Int(child_ids[child_slot, slot])
            child_id == 0 && continue
            if cell_active[child_id] == UInt8(1)
                acc += owned[child_id, q]
            else
                acc += child_restrict[child_id, q]
            end
        end
        restrict_to_parent[parent_id, q] = acc
    end
end

function restrict_conservative_tree_gpu_level_2d!(
        restrict_to_parent::AbstractMatrix,
        owned::AbstractMatrix,
        child_restrict::AbstractMatrix,
        parent_pack::ConservativeTreeGPUParentChildPack2D,
        cell_pack::ConservativeTreeGPUCellPack2D;
        sync::Bool=true)
    size(restrict_to_parent, 1) == Int(cell_pack.n_cells) &&
        size(owned, 1) == Int(cell_pack.n_cells) &&
        size(child_restrict, 1) == Int(cell_pack.n_cells) ||
        throw(ArgumentError("F matrices must match pack.n_cells"))
    size(restrict_to_parent, 2) == 9 && size(owned, 2) == 9 &&
        size(child_restrict, 2) == 9 ||
        throw(ArgumentError("F matrices must have 9 D2Q9 populations"))
    backend = KernelAbstractions.get_backend(owned)
    kernel! = _restrict_conservative_tree_gpu_level_2d_kernel!(backend)
    kernel!(restrict_to_parent, owned, child_restrict, parent_pack.parent_ids,
            parent_pack.child_ids, cell_pack.cell_active, parent_pack.nparents;
            ndrange=Int(parent_pack.nparents) * 9)
    sync && KernelAbstractions.synchronize(backend)
    return restrict_to_parent
end

@kernel function _apply_conservative_tree_gpu_restriction_to_inactive_level_F_2d_kernel!(
        F, @Const(restrict_to_parent), @Const(cell_level),
        @Const(cell_active), n_cells::Int32, level::UInt8)
    linear = @index(Global)
    total = Int(n_cells) * 9
    @inbounds if linear <= total
        cell = (linear - 1) ÷ 9 + 1
        q = (linear - 1) % 9 + 1
        if cell_level[cell] == level && cell_active[cell] == UInt8(0)
            F[cell, q] = restrict_to_parent[cell, q]
        end
    end
end

function apply_conservative_tree_gpu_restriction_to_inactive_level_F_2d!(
        F::AbstractMatrix,
        restrict_to_parent::AbstractMatrix,
        pack::ConservativeTreeGPUCellPack2D,
        level::Integer;
        sync::Bool=true)
    0 <= Int(level) <= typemax(UInt8) ||
        throw(ArgumentError("level must fit in UInt8"))
    size(F, 1) == Int(pack.n_cells) &&
        size(restrict_to_parent, 1) == Int(pack.n_cells) ||
        throw(ArgumentError("F matrices must match pack.n_cells"))
    size(F, 2) == 9 && size(restrict_to_parent, 2) == 9 ||
        throw(ArgumentError("F matrices must have 9 D2Q9 populations"))
    backend = KernelAbstractions.get_backend(F)
    kernel! = _apply_conservative_tree_gpu_restriction_to_inactive_level_F_2d_kernel!(
        backend)
    kernel!(F, restrict_to_parent, pack.cell_level, pack.cell_active,
            pack.n_cells, UInt8(level); ndrange=Int(pack.n_cells) * 9)
    sync && KernelAbstractions.synchronize(backend)
    return F
end

@kernel function _apply_conservative_tree_gpu_coarse_to_fine_pair_F_2d_kernel!(
        F, @Const(coarse_to_fine), @Const(child_ids), nparents::Int32,
        substep::Int32)
    linear = @index(Global)
    total = Int(nparents) * 4 * 9
    @inbounds if linear <= total
        slot = (linear - 1) ÷ 36 + 1
        rem = (linear - 1) % 36
        child_slot = rem ÷ 9 + 1
        q = rem % 9 + 1
        child_id = Int(child_ids[child_slot, slot])
        if child_id != 0
            ix = (child_slot - 1) % 2 + 1
            iy = (child_slot - 1) ÷ 2 + 1
            F[child_id, q] += coarse_to_fine[ix, iy, q, Int(substep), slot]
        end
    end
end

function apply_conservative_tree_gpu_coarse_to_fine_pair_F_2d!(
        F::AbstractMatrix,
        coarse_to_fine::AbstractArray,
        pack::ConservativeTreeGPUParentChildPack2D,
        substep::Integer;
        sync::Bool=true)
    step = Int(substep)
    step >= 1 || throw(ArgumentError("substep must be positive"))
    size(F, 2) == 9 || throw(ArgumentError("F must have 9 D2Q9 populations"))
    size(coarse_to_fine, 1) == 2 && size(coarse_to_fine, 2) == 2 &&
        size(coarse_to_fine, 3) == 9 ||
        throw(ArgumentError("coarse_to_fine must have dimensions 2 x 2 x 9 x ratio x nparents"))
    step <= size(coarse_to_fine, 4) ||
        throw(ArgumentError("substep is outside the coarse_to_fine ledger"))
    size(coarse_to_fine, 5) == Int(pack.nparents) ||
        throw(ArgumentError("coarse_to_fine parent count must match pack"))
    backend = KernelAbstractions.get_backend(F)
    kernel! = _apply_conservative_tree_gpu_coarse_to_fine_pair_F_2d_kernel!(backend)
    kernel!(F, coarse_to_fine, pack.child_ids, pack.nparents, Int32(step);
            ndrange=Int(pack.nparents) * 4 * 9)
    sync && KernelAbstractions.synchronize(backend)
    return F
end

@kernel function _deposit_conservative_tree_gpu_coarse_to_fine_routes_F_2d_kernel!(
        coarse_to_fine, @Const(F), @Const(ledger_first),
        @Const(ledger_count), @Const(src), @Const(qsrc), @Const(weight),
        ratio::Int32, nparents::Int32)
    linear = @index(Global)
    total = Int(nparents) * 4 * 9
    @inbounds if linear <= total
        slot = (linear - 1) ÷ 36 + 1
        rem = (linear - 1) % 36
        child_slot = rem ÷ 9 + 1
        q = rem % 9 + 1
        ix = (child_slot - 1) % 2 + 1
        iy = (child_slot - 1) ÷ 2 + 1
        first_entry = Int(ledger_first[ix, iy, q, slot])
        count = Int(ledger_count[ix, iy, q, slot])
        packet = zero(eltype(coarse_to_fine))
        for offset in 0:(count - 1)
            entry = first_entry + offset
            packet += weight[entry] * F[Int(src[entry]), Int(qsrc[entry])]
        end
        if packet != zero(packet)
            for substep in 1:Int(ratio)
                coarse_to_fine[ix, iy, q, substep, slot] += packet
            end
        end
    end
end

function deposit_conservative_tree_gpu_coarse_to_fine_routes_F_2d!(
        coarse_to_fine::AbstractArray,
        F::AbstractMatrix,
        pack::ConservativeTreeGPUC2FDepositPack2D;
        sync::Bool=true)
    size(F, 2) == 9 || throw(ArgumentError("F must have 9 D2Q9 populations"))
    size(coarse_to_fine, 1) == 2 && size(coarse_to_fine, 2) == 2 &&
        size(coarse_to_fine, 3) == 9 ||
        throw(ArgumentError("coarse_to_fine must have dimensions 2 x 2 x 9 x ratio x nparents"))
    size(coarse_to_fine, 4) == Int(pack.ratio) ||
        throw(ArgumentError("coarse_to_fine ratio must match pack"))
    size(coarse_to_fine, 5) == Int(pack.nparents) ||
        throw(ArgumentError("coarse_to_fine parent count must match pack"))
    backend = KernelAbstractions.get_backend(F)
    kernel! = _deposit_conservative_tree_gpu_coarse_to_fine_routes_F_2d_kernel!(
        backend)
    kernel!(coarse_to_fine, F, pack.ledger_first, pack.ledger_count,
            pack.src, pack.q, pack.weight, pack.ratio, pack.nparents;
            ndrange=Int(pack.nparents) * 4 * 9)
    sync && KernelAbstractions.synchronize(backend)
    return coarse_to_fine
end

@kernel function _deposit_conservative_tree_gpu_fine_to_coarse_cache_F_2d_kernel!(
        cache_packets, @Const(F), @Const(slot_first), @Const(slot_count),
        @Const(src), @Const(qsrc), @Const(weight), ratio::Int32,
        nslots::Int32)
    linear = @index(Global)
    total = Int(nslots) * Int(ratio)
    @inbounds if linear <= total
        slot = (linear - 1) ÷ Int(ratio) + 1
        substep = (linear - 1) % Int(ratio) + 1
        first_entry = Int(slot_first[substep, slot])
        count = Int(slot_count[substep, slot])
        packet = zero(eltype(cache_packets))
        for offset in 0:(count - 1)
            entry = first_entry + offset
            packet += weight[entry] * F[Int(src[entry]), Int(qsrc[entry])]
        end
        if packet != zero(packet)
            cache_packets[(slot - 1) * Int(ratio) + substep] += packet
        end
    end
end

function deposit_conservative_tree_gpu_fine_to_coarse_cache_F_2d!(
        cache_packets::AbstractVector,
        F::AbstractMatrix,
        pack::ConservativeTreeGPUF2CDepositPack2D;
        sync::Bool=true)
    size(F, 2) == 9 || throw(ArgumentError("F must have 9 D2Q9 populations"))
    length(cache_packets) == Int(pack.nslots) * Int(pack.ratio) ||
        throw(ArgumentError("cache_packets length must match pack"))
    backend = KernelAbstractions.get_backend(F)
    kernel! = _deposit_conservative_tree_gpu_fine_to_coarse_cache_F_2d_kernel!(
        backend)
    kernel!(cache_packets, F, pack.slot_first, pack.slot_count, pack.src,
            pack.q, pack.weight, pack.ratio, pack.nslots;
            ndrange=Int(pack.nslots) * Int(pack.ratio))
    sync && KernelAbstractions.synchronize(backend)
    return cache_packets
end

@kernel function _deposit_conservative_tree_gpu_fine_to_coarse_cache_substep_F_2d_kernel!(
        cache_packets, @Const(F), @Const(slot_first), @Const(slot_count),
        @Const(src), @Const(qsrc), @Const(weight), substep::Int32,
        ratio::Int32, nslots::Int32)
    slot = @index(Global)
    @inbounds if slot <= Int(nslots)
        step = Int(substep)
        first_entry = Int(slot_first[step, slot])
        count = Int(slot_count[step, slot])
        packet = zero(eltype(cache_packets))
        for offset in 0:(count - 1)
            entry = first_entry + offset
            packet += weight[entry] * F[Int(src[entry]), Int(qsrc[entry])]
        end
        if packet != zero(packet)
            cache_packets[(slot - 1) * Int(ratio) + step] += packet
        end
    end
end

function deposit_conservative_tree_gpu_fine_to_coarse_cache_substep_F_2d!(
        cache_packets::AbstractVector,
        F::AbstractMatrix,
        pack::ConservativeTreeGPUF2CDepositPack2D,
        substep::Integer;
        sync::Bool=true)
    step = Int(substep)
    1 <= step <= Int(pack.ratio) ||
        throw(ArgumentError("substep is outside the fine-to-coarse pack ratio"))
    size(F, 2) == 9 || throw(ArgumentError("F must have 9 D2Q9 populations"))
    length(cache_packets) == Int(pack.nslots) * Int(pack.ratio) ||
        throw(ArgumentError("cache_packets length must match pack"))
    Int(pack.nslots) == 0 && return cache_packets
    backend = KernelAbstractions.get_backend(F)
    kernel! = _deposit_conservative_tree_gpu_fine_to_coarse_cache_substep_F_2d_kernel!(
        backend)
    kernel!(cache_packets, F, pack.slot_first, pack.slot_count, pack.src,
            pack.q, pack.weight, Int32(step), pack.ratio, pack.nslots;
            ndrange=Int(pack.nslots))
    sync && KernelAbstractions.synchronize(backend)
    return cache_packets
end

@kernel function _apply_conservative_tree_gpu_sync_up_cache_F_2d_kernel!(
        F, @Const(cache_packets), @Const(dst_ids), @Const(qs),
        ratio::Int32, nslots::Int32)
    slot = @index(Global)
    @inbounds if slot <= Int(nslots)
        packet = zero(eltype(F))
        offset = (slot - 1) * Int(ratio)
        for substep in 1:Int(ratio)
            packet += cache_packets[offset + substep]
        end
        if packet != zero(packet)
            F[Int(dst_ids[slot]), Int(qs[slot])] += packet
        end
    end
end

function apply_conservative_tree_gpu_sync_up_cache_F_2d!(
        F::AbstractMatrix,
        cache_packets::AbstractVector,
        pack::ConservativeTreeGPUF2CDepositPack2D;
        sync::Bool=true)
    size(F, 2) == 9 || throw(ArgumentError("F must have 9 D2Q9 populations"))
    length(cache_packets) == Int(pack.nslots) * Int(pack.ratio) ||
        throw(ArgumentError("cache_packets length must match pack"))
    backend = KernelAbstractions.get_backend(F)
    kernel! = _apply_conservative_tree_gpu_sync_up_cache_F_2d_kernel!(backend)
    kernel!(F, cache_packets, pack.cache_dst_ids, pack.cache_qs, pack.ratio,
            pack.nslots; ndrange=Int(pack.nslots))
    sync && KernelAbstractions.synchronize(backend)
    return F
end

function pack_conservative_tree_gpu_boundary_corrections_2d(
        spec::ConservativeTreeSpec2D,
        table::ConservativeTreeRouteTable2D,
        level::Integer;
        boundary::Symbol=:periodic_x_wall_y,
        u_south=0,
        u_north=0,
        rho_wall=1,
        T::Type{<:Real}=Float32)
    policy = _conservative_tree_gpu_stream_boundary_policy_2d(boundary)
    corrections = Dict{Tuple{Int,Int},T}()
    if policy == :periodic_x_moving_wall_y
        l = Int(level)
        @inbounds for route_pos in table.boundary_route_ranges_by_level[l + 1]
            route_id = table.boundary_routes[route_pos]
            route = table.routes[route_id]
            cy = d2q9_cy(route.q)
            cy != 0 ||
                throw(ArgumentError("periodic-x moving-wall GPU correction received an x-boundary route; rebuild the route table with periodic_x=true"))
            qdst = d2q9_opposite(route.q)
            wall_u = cy < 0 ? u_south : u_north
            volume = spec.cells[route.src].metrics.volume
            value = T(route.weight) *
                T(_moving_wall_delta(volume, rho_wall, wall_u, qdst))
            key = (route.src, qdst)
            corrections[key] = get(corrections, key, zero(T)) + value
        end
    end

    dst = Vector{Int32}(undef, length(corrections))
    q = Vector{UInt8}(undef, length(corrections))
    value = Vector{T}(undef, length(corrections))
    idx = 1
    for ((dst_id, qidx), v) in corrections
        dst[idx] = Int32(dst_id)
        q[idx] = UInt8(qidx)
        value[idx] = v
        idx += 1
    end
    return ConservativeTreeGPUBoundaryCorrectionPack2D{T,typeof(dst),
        typeof(q),typeof(value)}(Int32(length(dst)), dst, q, value)
end

function transfer_conservative_tree_gpu_boundary_correction_pack_2d(
        pack::ConservativeTreeGPUBoundaryCorrectionPack2D{T},
        backend) where T
    dst = _copy_conservative_tree_gpu_array_2d(backend, pack.dst)
    q = _copy_conservative_tree_gpu_array_2d(backend, pack.q)
    value = _copy_conservative_tree_gpu_array_2d(backend, pack.value)
    return ConservativeTreeGPUBoundaryCorrectionPack2D{T,typeof(dst),
        typeof(q),typeof(value)}(pack.nentries, dst, q, value)
end

@kernel function _apply_conservative_tree_gpu_boundary_corrections_2d_kernel!(
        F, @Const(dst), @Const(q), @Const(value), nentries::Int32)
    entry = @index(Global)
    @inbounds if entry <= Int(nentries)
        F[Int(dst[entry]), Int(q[entry])] += value[entry]
    end
end

function apply_conservative_tree_gpu_boundary_corrections_2d!(
        F::AbstractMatrix,
        pack::ConservativeTreeGPUBoundaryCorrectionPack2D;
        sync::Bool=true)
    Int(pack.nentries) == 0 && return F
    backend = KernelAbstractions.get_backend(F)
    kernel! = _apply_conservative_tree_gpu_boundary_corrections_2d_kernel!(
        backend)
    kernel!(F, pack.dst, pack.q, pack.value, pack.nentries;
            ndrange=Int(pack.nentries))
    sync && KernelAbstractions.synchronize(backend)
    return F
end

@kernel function _blend_conservative_tree_gpu_level_rows_2d_kernel!(
        Fdst, @Const(Fa), @Const(Fb), @Const(cell_level), n_cells::Int32,
        level::UInt8, weight)
    linear = @index(Global)
    total = Int(n_cells) * 9
    @inbounds if linear <= total
        cell = (linear - 1) ÷ 9 + 1
        if cell_level[cell] == level
            q = (linear - 1) % 9 + 1
            w = eltype(Fdst)(weight)
            Fdst[cell, q] = (one(w) - w) * Fa[cell, q] + w * Fb[cell, q]
        end
    end
end

function blend_conservative_tree_gpu_level_rows_2d!(
        Fdst::AbstractMatrix,
        Fa::AbstractMatrix,
        Fb::AbstractMatrix,
        pack::ConservativeTreeGPUCellPack2D,
        level::Integer,
        weight;
        sync::Bool=true)
    backend = KernelAbstractions.get_backend(Fdst)
    kernel! = _blend_conservative_tree_gpu_level_rows_2d_kernel!(backend)
    kernel!(Fdst, Fa, Fb, pack.cell_level, pack.n_cells, UInt8(level),
            weight; ndrange=Int(pack.n_cells) * 9)
    sync && KernelAbstractions.synchronize(backend)
    return Fdst
end

@kernel function _correct_conservative_tree_gpu_mass_2d_kernel!(
        F, cell_id::Int32, drift)
    idx = @index(Global)
    @inbounds if idx == 1
        F[Int(cell_id), 1] -= eltype(F)(drift)
    end
end

function correct_conservative_tree_gpu_mass_2d!(
        F::AbstractMatrix,
        cell_id::Integer,
        drift;
        sync::Bool=true)
    backend = KernelAbstractions.get_backend(F)
    kernel! = _correct_conservative_tree_gpu_mass_2d_kernel!(backend)
    kernel!(F, Int32(cell_id), drift; ndrange=1)
    sync && KernelAbstractions.synchronize(backend)
    return F
end

@kernel function _sum_conservative_tree_gpu_mass_2d_kernel!(
        mass_sum, @Const(F), nentries::Int32)
    idx = @index(Global)
    @inbounds if idx <= Int(nentries)
        Atomix.@atomic mass_sum[1] += F[idx]
    end
end

function sum_conservative_tree_gpu_mass_2d!(
        mass_sum::AbstractVector,
        F::AbstractMatrix;
        sync::Bool=true)
    length(mass_sum) == 1 ||
        throw(ArgumentError("mass_sum must be a one-element backend array"))
    fill!(mass_sum, zero(eltype(mass_sum)))
    backend = KernelAbstractions.get_backend(F)
    kernel! = _sum_conservative_tree_gpu_mass_2d_kernel!(backend)
    kernel!(mass_sum, F, Int32(length(F)); ndrange=length(F))
    sync && KernelAbstractions.synchronize(backend)
    return mass_sum
end

@kernel function _correct_conservative_tree_gpu_mass_from_sum_2d_kernel!(
        F, @Const(mass_sum), max_raw_relative_mass_drift,
        cell_id::Int32, target_mass, mass_denom)
    idx = @index(Global)
    @inbounds if idx == 1
        T = eltype(F)
        drift = T(mass_sum[1] - T(target_mass))
        denom = T(mass_denom)
        rel = abs(drift) / denom
        if rel > max_raw_relative_mass_drift[1]
            max_raw_relative_mass_drift[1] = rel
        end
        F[Int(cell_id), 1] -= drift
    end
end

function correct_conservative_tree_gpu_mass_from_sum_2d!(
        F::AbstractMatrix,
        mass_sum::AbstractVector,
        max_raw_relative_mass_drift::AbstractVector,
        cell_id::Integer,
        target_mass,
        mass_denom;
        sync::Bool=true)
    length(mass_sum) == 1 ||
        throw(ArgumentError("mass_sum must be a one-element backend array"))
    length(max_raw_relative_mass_drift) == 1 ||
        throw(ArgumentError("max_raw_relative_mass_drift must be a one-element backend array"))
    backend = KernelAbstractions.get_backend(F)
    kernel! = _correct_conservative_tree_gpu_mass_from_sum_2d_kernel!(backend)
    kernel!(F, mass_sum, max_raw_relative_mass_drift, Int32(cell_id),
            target_mass, mass_denom; ndrange=1)
    sync && KernelAbstractions.synchronize(backend)
    return F
end

function enforce_conservative_tree_gpu_mass_2d!(
        F::AbstractMatrix,
        mass_sum::AbstractVector,
        max_raw_relative_mass_drift::AbstractVector,
        cell_id::Integer,
        target_mass,
        mass_denom;
        sync::Bool=true)
    sum_conservative_tree_gpu_mass_2d!(mass_sum, F; sync=false)
    correct_conservative_tree_gpu_mass_from_sum_2d!(
        F, mass_sum, max_raw_relative_mass_drift, cell_id,
        target_mass, mass_denom; sync=false)
    sync && KernelAbstractions.synchronize(KernelAbstractions.get_backend(F))
    return F
end

function _allocate_conservative_tree_gpu_matrix_2d(backend,
                                                   ::Type{T},
                                                   n_cells::Integer) where T
    F = _allocate_conservative_tree_backend_array_2d(
        backend, T, Int(n_cells), 9)
    fill!(F, zero(T))
    return F
end

function _allocate_conservative_tree_gpu_array_2d(backend,
                                                  ::Type{T},
                                                  dims::Integer...) where T
    A = _allocate_conservative_tree_backend_array_2d(
        backend, T, Int.(dims)...)
    fill!(A, zero(T))
    return A
end

function create_conservative_tree_gpu_subcycle_workspace_2d(
        spec::ConservativeTreeSpec2D,
        table::ConservativeTreeRouteTable2D,
        backend;
        boundary::Symbol=:periodic_x_wall_y,
        interface_time_scaling::Symbol=:leaf_equivalent,
        periodic_x::Bool=false,
        u_south=0,
        u_north=0,
        rho_wall=1,
        T::Type{<:Real}=Float32)
    boundary_policy = _conservative_tree_gpu_stream_boundary_policy_2d(
        boundary)
    interface_mode = _check_conservative_tree_interface_time_scaling_2d(
        interface_time_scaling)
    schedule = create_conservative_tree_subcycle_schedule_2d(spec.max_level)
    route_bank = create_conservative_tree_subcycle_spatial_ledger_bank_2d(
        spec; schedule=schedule, T=T)
    prepare_conservative_tree_subcycle_route_packet_cache_2d!(
        route_bank, table; periodic_x=periodic_x)

    n_cells = length(spec.cells)
    cell_pack = transfer_conservative_tree_gpu_cell_pack_2d(
        pack_conservative_tree_gpu_cells_2d(spec; T=T), backend)
    direct_pulls = [
        transfer_conservative_tree_gpu_pull_pack_2d(
            pack_conservative_tree_gpu_direct_level_pull_routes_2d(
                spec, table, level; boundary=boundary_policy, T=T),
            backend)
        for level in 0:spec.max_level
    ]
    boundary_corrections = [
        transfer_conservative_tree_gpu_boundary_correction_pack_2d(
            pack_conservative_tree_gpu_boundary_corrections_2d(
                spec, table, level; boundary=boundary_policy,
                u_south=u_south, u_north=u_north, rho_wall=rho_wall, T=T),
            backend)
        for level in 0:spec.max_level
    ]

    parent_packs = ConservativeTreeGPUParentChildPack2D[]
    c2f_packs = ConservativeTreeGPUC2FDepositPack2D[]
    f2c_packs = ConservativeTreeGPUF2CDepositPack2D[]
    c2f_ledgers = Any[]
    f2c_caches = Any[]
    for parent_level in 0:(spec.max_level - 1)
        parent_host = pack_conservative_tree_gpu_parent_children_2d(
            spec, parent_level)
        push!(parent_packs,
              transfer_conservative_tree_gpu_parent_child_pack_2d(
                  parent_host, backend))
        c2f_host = pack_conservative_tree_gpu_coarse_to_fine_deposit_2d(
            spec, table, parent_host, parent_level; ratio=schedule.ratio,
            interface_time_scaling=interface_mode, T=T)
        push!(c2f_packs,
              transfer_conservative_tree_gpu_c2f_deposit_pack_2d(
                  c2f_host, backend))
        f2c_host = pack_conservative_tree_gpu_fine_to_coarse_deposit_2d(
            spec, table, route_bank, parent_level + 1;
            interface_time_scaling=interface_mode, periodic_x=periodic_x,
            T=T)
        push!(f2c_packs,
              transfer_conservative_tree_gpu_f2c_deposit_pack_2d(
                  f2c_host, backend))
        push!(c2f_ledgers,
              _allocate_conservative_tree_gpu_array_2d(
                  backend, T, 2, 2, 9, schedule.ratio,
                  Int(parent_host.nparents)))
        push!(f2c_caches,
              _allocate_conservative_tree_gpu_array_2d(
                  backend, T, Int(f2c_host.nslots) * schedule.ratio))
    end

    restricts = [
        _allocate_conservative_tree_gpu_matrix_2d(backend, T, n_cells)
        for _ in 0:spec.max_level
    ]
    return (;
        backend, schedule, boundary=boundary_policy,
        interface_time_scaling=interface_mode,
        cell_pack, direct_pulls, boundary_corrections, parent_packs,
        c2f_packs, f2c_packs, c2f_ledgers, f2c_caches,
        owned=_allocate_conservative_tree_gpu_matrix_2d(backend, T, n_cells),
        reflux=_allocate_conservative_tree_gpu_matrix_2d(backend, T, n_cells),
        restricts,
        Fsource=_allocate_conservative_tree_gpu_matrix_2d(backend, T, n_cells),
        Fscratch=_allocate_conservative_tree_gpu_matrix_2d(backend, T, n_cells))
end

function _conservative_tree_gpu_restrict_all_levels_2d!(workspace)
    for parent_level in (workspace.schedule.max_level - 1):-1:0
        fill!(workspace.restricts[parent_level + 1],
              zero(eltype(workspace.restricts[parent_level + 1])))
        restrict_conservative_tree_gpu_level_2d!(
            workspace.restricts[parent_level + 1], workspace.owned,
            workspace.restricts[parent_level + 2],
            workspace.parent_packs[parent_level + 1], workspace.cell_pack;
            sync=false)
    end
    return workspace
end

function _conservative_tree_gpu_level_source_2d!(
        Fdst,
        workspace,
        level::Integer)
    fill!(Fdst, zero(eltype(Fdst)))
    copy_conservative_tree_gpu_level_rows_2d!(
        Fdst, workspace.owned, workspace.cell_pack, level; sync=false)
    apply_conservative_tree_gpu_restriction_to_inactive_level_F_2d!(
        Fdst, workspace.restricts[Int(level) + 1], workspace.cell_pack,
        level; sync=false)
    return Fdst
end

function _conservative_tree_gpu_collide_level_2d!(
        F,
        workspace,
        spec::ConservativeTreeSpec2D,
        level::Integer;
        collision::Symbol,
        omega,
        Fx=0,
        Fy=0)
    l = Int(level)
    if collision == :guo
        return collide_Guo_conservative_tree_gpu_active_level_F_2d!(
            F, workspace.cell_pack, l,
            conservative_tree_leaf_equivalent_omega_2d(omega, spec, l),
            conservative_tree_leaf_equivalent_force_2d(Fx, spec, l),
            conservative_tree_leaf_equivalent_force_2d(Fy, spec, l);
            sync=false)
    elseif collision == :bgk
        return collide_BGK_conservative_tree_gpu_active_level_F_2d!(
            F, workspace.cell_pack, l,
            conservative_tree_leaf_equivalent_omega_2d(omega, spec, l);
            sync=false)
    end
    throw(ArgumentError("GPU subcycle collision must be :guo or :bgk"))
end

function stream_conservative_tree_subcycled_buffered_routes_gpu_F_2d!(
        Fout::AbstractMatrix,
        Fin::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        table::ConservativeTreeRouteTable2D,
        workspace;
        collision::Symbol=:guo,
        omega=1,
        Fx=0,
        Fy=0,
        coarse_to_fine_predictor_weight=0,
        sync::Bool=true)
    size(Fin, 1) == length(spec.cells) && size(Fout, 1) == length(spec.cells) ||
        throw(ArgumentError("F matrices must match the conservative-tree spec"))
    size(Fin, 2) == 9 && size(Fout, 2) == 9 ||
        throw(ArgumentError("F matrices must have 9 D2Q9 populations"))
    schedule = workspace.schedule
    fill!(workspace.owned, zero(eltype(workspace.owned)))
    fill!(workspace.reflux, zero(eltype(workspace.reflux)))
    for rbuf in workspace.restricts
        fill!(rbuf, zero(eltype(rbuf)))
    end
    for level in 0:spec.max_level
        copy_conservative_tree_gpu_level_rows_2d!(
            workspace.owned, Fin, workspace.cell_pack, level;
            active_only=true, sync=false)
    end
    _conservative_tree_gpu_restrict_all_levels_2d!(workspace)

    for event in schedule.events
        if event.phase == :sync_down
            parent_level = event.src_level
            fill!(workspace.c2f_ledgers[parent_level + 1],
                  zero(eltype(workspace.c2f_ledgers[parent_level + 1])))
            fill!(workspace.f2c_caches[parent_level + 1],
                  zero(eltype(workspace.f2c_caches[parent_level + 1])))

            predictor_weight = eltype(workspace.Fsource)(
                coarse_to_fine_predictor_weight)
            _conservative_tree_gpu_level_source_2d!(
                workspace.Fsource, workspace, parent_level)
            if predictor_weight > zero(predictor_weight)
                _conservative_tree_gpu_level_source_2d!(
                    workspace.Fscratch, workspace, parent_level)
                _conservative_tree_gpu_collide_level_2d!(
                    workspace.Fscratch, workspace, spec, parent_level;
                    collision=collision, omega=omega, Fx=Fx, Fy=Fy)
                blend_conservative_tree_gpu_level_rows_2d!(
                    workspace.Fsource, workspace.Fsource,
                    workspace.Fscratch, workspace.cell_pack, parent_level,
                    predictor_weight; sync=false)
            end
            deposit_conservative_tree_gpu_coarse_to_fine_routes_F_2d!(
                workspace.c2f_ledgers[parent_level + 1],
                workspace.Fsource,
                workspace.c2f_packs[parent_level + 1]; sync=false)
        elseif event.phase == :advance
            level = event.src_level
            _conservative_tree_gpu_level_source_2d!(
                workspace.Fsource, workspace, level)
            _conservative_tree_gpu_collide_level_2d!(
                workspace.Fsource, workspace, spec, level;
                collision=collision, omega=omega, Fx=Fx, Fy=Fy)

            if level > 0
                substep = conservative_tree_subcycle_local_substep_2d(
                    schedule, level - 1, event.tick)
                deposit_conservative_tree_gpu_fine_to_coarse_cache_substep_F_2d!(
                    workspace.f2c_caches[level], workspace.Fsource,
                    workspace.f2c_packs[level], substep; sync=false)
            end

            fill!(workspace.Fscratch, zero(eltype(workspace.Fscratch)))
            stream_conservative_tree_gpu_pull_routes_F_2d!(
                workspace.Fscratch, workspace.Fsource,
                workspace.direct_pulls[level + 1]; sync=false)
            apply_conservative_tree_gpu_boundary_corrections_2d!(
                workspace.Fscratch,
                workspace.boundary_corrections[level + 1]; sync=false)
            if level > 0
                substep = conservative_tree_subcycle_local_substep_2d(
                    schedule, level - 1, event.tick)
                apply_conservative_tree_gpu_coarse_to_fine_pair_F_2d!(
                    workspace.Fscratch, workspace.c2f_ledgers[level],
                    workspace.parent_packs[level], substep; sync=false)
            end
            add_and_clear_conservative_tree_gpu_level_rows_2d!(
                workspace.Fscratch, workspace.reflux, workspace.cell_pack,
                level; active_only=true, sync=false)
            copy_conservative_tree_gpu_level_rows_2d!(
                workspace.owned, workspace.Fscratch, workspace.cell_pack,
                level; active_only=true, sync=false)
        elseif event.phase == :sync_up
            parent_level = event.dst_level
            fill!(workspace.restricts[parent_level + 1],
                  zero(eltype(workspace.restricts[parent_level + 1])))
            restrict_conservative_tree_gpu_level_2d!(
                workspace.restricts[parent_level + 1], workspace.owned,
                workspace.restricts[parent_level + 2],
                workspace.parent_packs[parent_level + 1],
                workspace.cell_pack; sync=false)
            apply_conservative_tree_gpu_sync_up_cache_F_2d!(
                workspace.reflux, workspace.f2c_caches[parent_level + 1],
                workspace.f2c_packs[parent_level + 1]; sync=false)
        else
            throw(ArgumentError("unknown GPU subcycle event phase $(event.phase)"))
        end
    end

    fill!(Fout, zero(eltype(Fout)))
    for level in 0:spec.max_level
        copy_conservative_tree_gpu_level_rows_2d!(
            Fout, workspace.owned, workspace.cell_pack, level;
            active_only=true, sync=false)
    end
    sync && KernelAbstractions.synchronize(workspace.backend)
    return Fout
end

@kernel function _stream_conservative_tree_pull_routes_F_2d_kernel!(
        Fout, @Const(Fin), @Const(pull_first), @Const(pull_count),
        @Const(pull_src), @Const(pull_q), @Const(pull_weight),
        n_cells::Int32)
    linear = @index(Global)
    total = Int(n_cells) * 9
    @inbounds if linear <= total
        cell = (linear - 1) ÷ 9 + 1
        q = (linear - 1) % 9 + 1
        first_entry = Int(pull_first[cell, q])
        count = Int(pull_count[cell, q])
        acc = zero(eltype(Fout))
        for offset in 0:(count - 1)
            entry = first_entry + offset
            acc += pull_weight[entry] * Fin[Int(pull_src[entry]),
                                            Int(pull_q[entry])]
        end
        Fout[cell, q] = acc
    end
end

function stream_conservative_tree_gpu_pull_routes_F_2d!(
        Fout::AbstractMatrix,
        Fin::AbstractMatrix,
        pack::ConservativeTreeGPUPullRoutePack2D;
        sync::Bool=true)
    size(Fin, 1) == Int(pack.n_cells) && size(Fout, 1) == Int(pack.n_cells) ||
        throw(ArgumentError("F matrices must match pack.n_cells"))
    size(Fin, 2) == 9 && size(Fout, 2) == 9 ||
        throw(ArgumentError("F matrices must have 9 D2Q9 populations"))
    backend = KernelAbstractions.get_backend(Fin)
    kernel! = _stream_conservative_tree_pull_routes_F_2d_kernel!(backend)
    kernel!(Fout, Fin, pack.pull_first, pack.pull_count, pack.pull_src,
            pack.pull_q, pack.pull_weight, pack.n_cells;
            ndrange=Int(pack.n_cells) * 9)
    sync && KernelAbstractions.synchronize(backend)
    return Fout
end

@inline function _conservative_tree_gpu_cx_2d(q::Int)
    q == 2 && return 1
    q == 4 && return -1
    q == 6 && return 1
    q == 7 && return -1
    q == 8 && return -1
    q == 9 && return 1
    return 0
end

@inline function _conservative_tree_gpu_cy_2d(q::Int)
    q == 3 && return 1
    q == 5 && return -1
    q == 6 && return 1
    q == 7 && return 1
    q == 8 && return -1
    q == 9 && return -1
    return 0
end

@inline function _conservative_tree_gpu_weight_2d(q::Int, ::Type{T}) where T
    q == 1 && return T(4) / T(9)
    (q == 2 || q == 3 || q == 4 || q == 5) && return T(1) / T(9)
    return T(1) / T(36)
end

@inline function _conservative_tree_gpu_guo_integrated_q_2d(
        ::Val{Q}, Fq, volume, omega, rho, ux, uy, fx, fy, usq,
        guo_pref) where Q
    T = typeof(Fq + volume + omega + rho + ux + uy + fx + fy)
    cx = T(_conservative_tree_gpu_cx_2d(Q))
    cy = T(_conservative_tree_gpu_cy_2d(Q))
    w = _conservative_tree_gpu_weight_2d(Q, T)
    ci_dot_u = cx * ux + cy * uy
    ci_dot_F = cx * fx + cy * fy
    Sq = w * (T(3) * ((cx - ux) * fx + (cy - uy) * fy) +
              T(9) * ci_dot_u * ci_dot_F)
    f = Fq / volume
    feq = feq_2d(Val(Q), rho, ux, uy, usq)
    return volume * (f - omega * (f - feq) + guo_pref * Sq)
end

@inline function _conservative_tree_gpu_bgk_integrated_q_2d(
        ::Val{Q}, Fq, volume, omega, rho, ux, uy, usq) where Q
    f = Fq / volume
    feq = feq_2d(Val(Q), rho, ux, uy, usq)
    return volume * (f - omega * (f - feq))
end

@kernel function _collide_BGK_conservative_tree_active_level_F_2d_kernel!(
        F, @Const(active_cell_ids), @Const(cell_level), @Const(cell_volume),
        n_active::Int32, level::UInt8, omega)
    active_index = @index(Global)
    @inbounds if active_index <= Int(n_active)
        cell_id = Int(active_cell_ids[active_index])
        if cell_level[cell_id] == level
            T = eltype(F)
            volume = cell_volume[cell_id]
            f1 = F[cell_id, 1]; f2 = F[cell_id, 2]; f3 = F[cell_id, 3]
            f4 = F[cell_id, 4]; f5 = F[cell_id, 5]; f6 = F[cell_id, 6]
            f7 = F[cell_id, 7]; f8 = F[cell_id, 8]; f9 = F[cell_id, 9]
            mass_before = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
            mx = f2 - f4 + f6 - f7 - f8 + f9
            my = f3 - f5 + f6 + f7 - f8 - f9
            rho = mass_before / volume
            ux = mx / mass_before
            uy = my / mass_before
            usq = ux * ux + uy * uy
            womega = T(omega)

            g1 = _conservative_tree_gpu_bgk_integrated_q_2d(
                Val(1), f1, volume, womega, rho, ux, uy, usq)
            g2 = _conservative_tree_gpu_bgk_integrated_q_2d(
                Val(2), f2, volume, womega, rho, ux, uy, usq)
            g3 = _conservative_tree_gpu_bgk_integrated_q_2d(
                Val(3), f3, volume, womega, rho, ux, uy, usq)
            g4 = _conservative_tree_gpu_bgk_integrated_q_2d(
                Val(4), f4, volume, womega, rho, ux, uy, usq)
            g5 = _conservative_tree_gpu_bgk_integrated_q_2d(
                Val(5), f5, volume, womega, rho, ux, uy, usq)
            g6 = _conservative_tree_gpu_bgk_integrated_q_2d(
                Val(6), f6, volume, womega, rho, ux, uy, usq)
            g7 = _conservative_tree_gpu_bgk_integrated_q_2d(
                Val(7), f7, volume, womega, rho, ux, uy, usq)
            g8 = _conservative_tree_gpu_bgk_integrated_q_2d(
                Val(8), f8, volume, womega, rho, ux, uy, usq)
            g9 = _conservative_tree_gpu_bgk_integrated_q_2d(
                Val(9), f9, volume, womega, rho, ux, uy, usq)
            mass_after = g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8 + g9

            F[cell_id, 1] = g1 + mass_before - mass_after
            F[cell_id, 2] = g2
            F[cell_id, 3] = g3
            F[cell_id, 4] = g4
            F[cell_id, 5] = g5
            F[cell_id, 6] = g6
            F[cell_id, 7] = g7
            F[cell_id, 8] = g8
            F[cell_id, 9] = g9
        end
    end
end

function collide_BGK_conservative_tree_gpu_active_level_F_2d!(
        F::AbstractMatrix,
        pack::ConservativeTreeGPUCellPack2D,
        level::Integer,
        omega;
        sync::Bool=true)
    0 <= Int(level) <= typemax(UInt8) ||
        throw(ArgumentError("level must fit in UInt8"))
    size(F, 1) == Int(pack.n_cells) && size(F, 2) == 9 ||
        throw(ArgumentError("F must match the conservative-tree cell pack"))
    backend = KernelAbstractions.get_backend(F)
    kernel! = _collide_BGK_conservative_tree_active_level_F_2d_kernel!(backend)
    kernel!(F, pack.active_cell_ids, pack.cell_level, pack.cell_volume,
            pack.n_active, UInt8(level), omega; ndrange=Int(pack.n_active))
    sync && KernelAbstractions.synchronize(backend)
    return F
end

@kernel function _collide_Guo_conservative_tree_active_level_F_2d_kernel!(
        F, @Const(active_cell_ids), @Const(cell_level), @Const(cell_volume),
        n_active::Int32, level::UInt8, omega, Fx, Fy)
    active_index = @index(Global)
    @inbounds if active_index <= Int(n_active)
        cell_id = Int(active_cell_ids[active_index])
        if cell_level[cell_id] == level
            T = eltype(F)
            volume = cell_volume[cell_id]
            f1 = F[cell_id, 1]; f2 = F[cell_id, 2]; f3 = F[cell_id, 3]
            f4 = F[cell_id, 4]; f5 = F[cell_id, 5]; f6 = F[cell_id, 6]
            f7 = F[cell_id, 7]; f8 = F[cell_id, 8]; f9 = F[cell_id, 9]
            mass_before = zero(T)
            for q in 1:9
                mass_before += F[cell_id, q]
            end
            mx = f2 - f4 + f6 - f7 - f8 + f9
            my = f3 - f5 + f6 + f7 - f8 - f9
            rho = mass_before / volume
            fx = T(Fx)
            fy = T(Fy)
            womega = T(omega)
            ux = (mx / volume + fx / T(2)) / rho
            uy = (my / volume + fy / T(2)) / rho
            usq = ux * ux + uy * uy
            guo_pref = one(T) - womega / T(2)

            g1 = _conservative_tree_gpu_guo_integrated_q_2d(
                Val(1), f1, volume, womega, rho, ux, uy, fx, fy, usq,
                guo_pref)
            g2 = _conservative_tree_gpu_guo_integrated_q_2d(
                Val(2), f2, volume, womega, rho, ux, uy, fx, fy, usq,
                guo_pref)
            g3 = _conservative_tree_gpu_guo_integrated_q_2d(
                Val(3), f3, volume, womega, rho, ux, uy, fx, fy, usq,
                guo_pref)
            g4 = _conservative_tree_gpu_guo_integrated_q_2d(
                Val(4), f4, volume, womega, rho, ux, uy, fx, fy, usq,
                guo_pref)
            g5 = _conservative_tree_gpu_guo_integrated_q_2d(
                Val(5), f5, volume, womega, rho, ux, uy, fx, fy, usq,
                guo_pref)
            g6 = _conservative_tree_gpu_guo_integrated_q_2d(
                Val(6), f6, volume, womega, rho, ux, uy, fx, fy, usq,
                guo_pref)
            g7 = _conservative_tree_gpu_guo_integrated_q_2d(
                Val(7), f7, volume, womega, rho, ux, uy, fx, fy, usq,
                guo_pref)
            g8 = _conservative_tree_gpu_guo_integrated_q_2d(
                Val(8), f8, volume, womega, rho, ux, uy, fx, fy, usq,
                guo_pref)
            g9 = _conservative_tree_gpu_guo_integrated_q_2d(
                Val(9), f9, volume, womega, rho, ux, uy, fx, fy, usq,
                guo_pref)
            mass_after = g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8 + g9

            F[cell_id, 1] = g1 + mass_before - mass_after
            F[cell_id, 2] = g2
            F[cell_id, 3] = g3
            F[cell_id, 4] = g4
            F[cell_id, 5] = g5
            F[cell_id, 6] = g6
            F[cell_id, 7] = g7
            F[cell_id, 8] = g8
            F[cell_id, 9] = g9
        end
    end
end

function collide_Guo_conservative_tree_gpu_active_level_F_2d!(
        F::AbstractMatrix,
        pack::ConservativeTreeGPUCellPack2D,
        level::Integer,
        omega,
        Fx,
        Fy;
        sync::Bool=true)
    0 <= Int(level) <= typemax(UInt8) ||
        throw(ArgumentError("level must fit in UInt8"))
    size(F, 1) == Int(pack.n_cells) && size(F, 2) == 9 ||
        throw(ArgumentError("F must match the conservative-tree cell pack"))
    backend = KernelAbstractions.get_backend(F)
    kernel! = _collide_Guo_conservative_tree_active_level_F_2d_kernel!(backend)
    kernel!(F, pack.active_cell_ids, pack.cell_level, pack.cell_volume,
            pack.n_active, UInt8(level), omega, Fx, Fy;
            ndrange=Int(pack.n_active))
    sync && KernelAbstractions.synchronize(backend)
    return F
end

function advance_conservative_tree_gpu_direct_level_BGK_F_2d!(
        Fout::AbstractMatrix,
        Fin::AbstractMatrix,
        pull_pack::ConservativeTreeGPUPullRoutePack2D,
        cell_pack::ConservativeTreeGPUCellPack2D,
        level::Integer,
        omega;
        sync::Bool=true)
    collide_BGK_conservative_tree_gpu_active_level_F_2d!(
        Fin, cell_pack, level, omega; sync=false)
    stream_conservative_tree_gpu_pull_routes_F_2d!(
        Fout, Fin, pull_pack; sync=sync)
    return Fout
end

function advance_conservative_tree_gpu_direct_level_Guo_F_2d!(
        Fout::AbstractMatrix,
        Fin::AbstractMatrix,
        pull_pack::ConservativeTreeGPUPullRoutePack2D,
        cell_pack::ConservativeTreeGPUCellPack2D,
        level::Integer,
        omega,
        Fx,
        Fy;
        sync::Bool=true)
    collide_Guo_conservative_tree_gpu_active_level_F_2d!(
        Fin, cell_pack, level, omega, Fx, Fy; sync=false)
    stream_conservative_tree_gpu_pull_routes_F_2d!(
        Fout, Fin, pull_pack; sync=sync)
    return Fout
end

@inline function _gpu_pack_fine_local_2d(patch::ConservativeTreePatch2D,
                                         i_leaf::Int,
                                         j_leaf::Int)
    i0 = 2 * first(patch.parent_i_range) - 1
    j0 = 2 * first(patch.parent_j_range) - 1
    return i_leaf - i0 + 1, j_leaf - j0 + 1
end

@inline function _gpu_pack_cell_Fq_2d(coarse_F,
                                      patch::ConservativeTreePatch2D,
                                      pack::ConservativeTreeGPURoutePack2D,
                                      packed_cell::Int,
                                      q::Int)
    level = pack.cell_level[packed_cell]
    i = Int(pack.cell_i[packed_cell])
    j = Int(pack.cell_j[packed_cell])
    if level == UInt8(0)
        return coarse_F[i, j, q]
    end
    il, jl = _gpu_pack_fine_local_2d(patch, i, j)
    return patch.fine_F[il, jl, q]
end

@inline function _gpu_pack_add_cell_Fq_2d!(coarse_F,
                                           patch::ConservativeTreePatch2D,
                                           pack::ConservativeTreeGPURoutePack2D,
                                           packed_cell::Int,
                                           q::Int,
                                           value)
    level = pack.cell_level[packed_cell]
    i = Int(pack.cell_i[packed_cell])
    j = Int(pack.cell_j[packed_cell])
    if level == UInt8(0)
        coarse_F[i, j, q] += value
    else
        il, jl = _gpu_pack_fine_local_2d(patch, i, j)
        patch.fine_F[il, jl, q] += value
    end
    return nothing
end

function stream_conservative_tree_gpu_pack_interior_F_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D,
        pack::ConservativeTreeGPURoutePack2D)
    _check_composite_coarse_layout(coarse_in, patch_in)
    _check_composite_coarse_layout(coarse_out, patch_out)
    size(coarse_out) == size(coarse_in) ||
        throw(ArgumentError("coarse_out and coarse_in must have the same size"))
    patch_out.parent_i_range == patch_in.parent_i_range &&
        patch_out.parent_j_range == patch_in.parent_j_range ||
        throw(ArgumentError("patch_out and patch_in must cover the same parent range"))

    coarse_out .= 0
    patch_out.fine_F .= 0
    patch_out.coarse_shadow_F .= 0

    boundary_kind = UInt8(ROUTE_BOUNDARY)
    @inbounds for route_index in eachindex(pack.route_src)
        pack.route_kind[route_index] == boundary_kind && continue
        src = Int(pack.route_src[route_index])
        dst = Int(pack.route_dst[route_index])
        dst > 0 || throw(ArgumentError("non-boundary packed route has zero destination"))
        q = Int(pack.route_q[route_index])
        packet = _gpu_pack_cell_Fq_2d(coarse_in, patch_in, pack, src, q)
        _gpu_pack_add_cell_Fq_2d!(
            coarse_out, patch_out, pack, dst, q,
            pack.route_weight[route_index] * packet)
    end
    coalesce_patch_to_shadow_F_2d!(patch_out)
    return coarse_out, patch_out
end
