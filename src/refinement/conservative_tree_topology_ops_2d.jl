
"""
    create_conservative_tree_topology_2d(Nx, Ny, patch; coarse_volume=1.0,
                                         coarse_route_mode=:coarse)

Build static D2Q9 topology tables for a fixed ratio-2 conservative-tree patch.

The topology stores active coarse cells outside `patch`, inactive coarse
ledger cells under `patch`, and active fine children inside `patch`. Links and
routes are CPU-built tables intended to be packed or copied to a GPU-oriented
layout later; no LBM state is stored here.
"""
function create_conservative_tree_topology_2d(Nx::Integer,
                                               Ny::Integer,
                                               patch::ConservativeTreePatch2D;
                                               coarse_volume::Real=1.0,
                                               coarse_route_mode::Symbol=:coarse)
    Nx_i = Int(Nx)
    Ny_i = Int(Ny)
    _check_conservative_tree_topology_args(Nx_i, Ny_i, patch, coarse_volume)
    coarse_route_mode in (:coarse, :leaf_equivalent) ||
        throw(ArgumentError("coarse_route_mode must be :coarse or :leaf_equivalent"))

    cells, active_cells, cell_id_by_coord =
        _build_conservative_tree_cells_2d(Nx_i, Ny_i, patch, Float64(coarse_volume))

    links, routes,
    same_level_links, coarse_to_fine_links, fine_to_coarse_links,
    boundary_links, direct_routes, interface_routes, boundary_routes =
        _build_conservative_tree_links_2d(cells, active_cells, cell_id_by_coord,
                                          Nx_i, Ny_i, patch, coarse_route_mode)

    return ConservativeTreeTopology2D(cells, links, routes, active_cells,
                                      same_level_links, coarse_to_fine_links,
                                      fine_to_coarse_links, boundary_links,
                                      direct_routes, interface_routes,
                                      boundary_routes)
end

"""
    active_volume(topology)

Total volume of active cells in a conservative-tree topology.
"""
function active_volume(topology::ConservativeTreeTopology2D)
    total = 0.0
    @inbounds for id in topology.active_cells
        total += topology.cells[id].metrics.volume
    end
    return total
end

"""
    morton_key_2d(i, j)

Morton/Z-order key for one-based 2D integer coordinates.
"""
function morton_key_2d(i::Integer, j::Integer)
    i >= 1 || throw(ArgumentError("i must be >= 1"))
    j >= 1 || throw(ArgumentError("j must be >= 1"))

    x = UInt64(i - 1)
    y = UInt64(j - 1)
    key = UInt64(0)
    @inbounds for bit in 0:31
        key |= ((x >> bit) & UInt64(1)) << (2 * bit)
        key |= ((y >> bit) & UInt64(1)) << (2 * bit + 1)
    end
    return key
end

function _packed_route_kind_lists!(direct_routes::Vector{Int},
                                   interface_routes::Vector{Int},
                                   boundary_routes::Vector{Int},
                                   route_kind::RouteKind,
                                   route_index::Int)
    if route_kind == DIRECT
        push!(direct_routes, route_index)
    elseif route_kind == ROUTE_BOUNDARY
        push!(boundary_routes, route_index)
    else
        push!(interface_routes, route_index)
    end
    return nothing
end

function _pack_conservative_tree_cells_2d(topology::ConservativeTreeTopology2D,
                                          cells_per_block::Int)
    blocks = ConservativeTreeBlock2D[]
    packed_cell_ids = Int[]
    packed_morton_keys = UInt64[]
    logical_cell_to_block = zeros(Int, length(topology.cells))
    logical_cell_to_local = zeros(Int, length(topology.cells))

    levels = sort(unique(topology.cells[id].level for id in topology.active_cells))
    for level in levels
        ids = [id for id in topology.active_cells if topology.cells[id].level == level]
        sort!(ids; by=id -> morton_key_2d(topology.cells[id].i, topology.cells[id].j))

        for first_index in 1:cells_per_block:length(ids)
            last_index = min(first_index + cells_per_block - 1, length(ids))
            block_ids = @view ids[first_index:last_index]
            block_id = length(blocks) + 1
            first_cell = length(packed_cell_ids) + 1
            first_key = morton_key_2d(topology.cells[block_ids[1]].i,
                                      topology.cells[block_ids[1]].j)
            push!(blocks, ConservativeTreeBlock2D(level, first_cell,
                                                  length(block_ids), first_key))

            for (local_index, cell_id) in enumerate(block_ids)
                cell = topology.cells[cell_id]
                push!(packed_cell_ids, cell_id)
                push!(packed_morton_keys, morton_key_2d(cell.i, cell.j))
                logical_cell_to_block[cell_id] = block_id
                logical_cell_to_local[cell_id] = local_index
            end
        end
    end

    return blocks, packed_cell_ids, packed_morton_keys,
           logical_cell_to_block, logical_cell_to_local
end

"""
    pack_conservative_tree_topology_2d(topology; cells_per_block=128)

Pack active logical cells into fixed-size `(level, Morton)` blocks and remap
logical routes to `(block, local)` coordinates. Boundary routes use destination
`(0, 0)`.
"""
function pack_conservative_tree_topology_2d(topology::ConservativeTreeTopology2D;
                                            cells_per_block::Integer=128)
    cells_per_block_i = Int(cells_per_block)
    cells_per_block_i > 0 || throw(ArgumentError("cells_per_block must be positive"))

    blocks, packed_cell_ids, packed_morton_keys,
    logical_cell_to_block, logical_cell_to_local =
        _pack_conservative_tree_cells_2d(topology, cells_per_block_i)

    routes = ConservativeTreePackedRoute2D[]
    direct_routes = Int[]
    interface_routes = Int[]
    boundary_routes = Int[]

    for route in topology.routes
        src_block = logical_cell_to_block[route.src]
        src_local = logical_cell_to_local[route.src]
        src_block > 0 || throw(ArgumentError("route source is not an active packed cell"))

        if route.kind == ROUTE_BOUNDARY
            dst_block = 0
            dst_local = 0
        else
            dst_block = logical_cell_to_block[route.dst]
            dst_local = logical_cell_to_local[route.dst]
            dst_block > 0 || throw(ArgumentError("route destination is not an active packed cell"))
        end

        push!(routes, ConservativeTreePackedRoute2D(src_block, src_local,
                                                    dst_block, dst_local,
                                                    route.q, route.weight,
                                                    route.kind))
        _packed_route_kind_lists!(direct_routes, interface_routes, boundary_routes,
                                  route.kind, length(routes))
    end

    return ConservativeTreePackedTopology2D(cells_per_block_i, blocks,
                                            packed_cell_ids, packed_morton_keys,
                                            logical_cell_to_block,
                                            logical_cell_to_local,
                                            routes, direct_routes,
                                            interface_routes, boundary_routes)
end
