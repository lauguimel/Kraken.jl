# Static AMR topology tables for the conservative-tree D2Q9 prototype.
#
# This layer describes cells, logical D2Q9 links, and packet routes. It does
# not own LBM state and deliberately stays separate from the conservative
# projection/restriction oracle in conservative_tree_2d.jl.

@enum LinkKind::UInt8 SAME_LEVEL=0 COARSE_TO_FINE=1 FINE_TO_COARSE=2 BOUNDARY=3
@enum RouteKind::UInt8 DIRECT=0 SPLIT_FACE=1 SPLIT_CORNER=2 COALESCE_FACE=3 COALESCE_CORNER=4 ROUTE_BOUNDARY=5

abstract type AbstractCellMetrics end

struct CartesianMetrics2D <: AbstractCellMetrics
    volume::Float64
end

struct ConservativeTreeCell2D
    level::Int
    i::Int
    j::Int
    active::Bool
    metrics::CartesianMetrics2D
    parent::Int
end

struct ConservativeTreeLink2D
    src::Int
    q::Int
    kind::LinkKind
end

struct ConservativeTreeRoute2D
    src::Int
    dst::Int
    q::Int
    weight::Float64
    kind::RouteKind
end

struct ConservativeTreeTopology2D
    cells::Vector{ConservativeTreeCell2D}
    links::Vector{ConservativeTreeLink2D}
    routes::Vector{ConservativeTreeRoute2D}
    active_cells::Vector{Int}
    same_level_links::Vector{Int}
    coarse_to_fine_links::Vector{Int}
    fine_to_coarse_links::Vector{Int}
    boundary_links::Vector{Int}
    direct_routes::Vector{Int}
    interface_routes::Vector{Int}
    boundary_routes::Vector{Int}
end

struct ConservativeTreeBlock2D
    level::Int
    first_cell::Int
    count::Int
    morton_first::UInt64
end

struct ConservativeTreePackedRoute2D
    src_block::Int
    src_local::Int
    dst_block::Int
    dst_local::Int
    q::Int
    weight::Float64
    kind::RouteKind
end

struct ConservativeTreePackedTopology2D
    cells_per_block::Int
    blocks::Vector{ConservativeTreeBlock2D}
    packed_cell_ids::Vector{Int}
    packed_morton_keys::Vector{UInt64}
    logical_cell_to_block::Vector{Int}
    logical_cell_to_local::Vector{Int}
    routes::Vector{ConservativeTreePackedRoute2D}
    direct_routes::Vector{Int}
    interface_routes::Vector{Int}
    boundary_routes::Vector{Int}
end

@inline _tree_topology_key(level::Int, i::Int, j::Int) = (level, i, j)

@inline function _fine_global_i(I::Int, ix::Int)
    return 2 * I - 2 + ix
end

@inline function _fine_global_j(J::Int, iy::Int)
    return 2 * J - 2 + iy
end

@inline function _coarse_parent_from_fine(i::Int, j::Int)
    return (i + 1) >>> 1, (j + 1) >>> 1
end

@inline function _child_index_from_fine(i::Int, j::Int)
    return isodd(i) ? 1 : 2, isodd(j) ? 1 : 2
end

@inline function _inside_fine_patch(i::Int, j::Int, patch::ConservativeTreePatch2D)
    imin = 2 * first(patch.parent_i_range) - 1
    imax = 2 * last(patch.parent_i_range)
    jmin = 2 * first(patch.parent_j_range) - 1
    jmax = 2 * last(patch.parent_j_range)
    return imin <= i <= imax && jmin <= j <= jmax
end

@inline function _inside_leaf_domain(i::Int, j::Int, Nx::Int, Ny::Int)
    return 1 <= i <= 2 * Nx && 1 <= j <= 2 * Ny
end

function _check_conservative_tree_topology_args(Nx::Int,
                                                Ny::Int,
                                                patch::ConservativeTreePatch2D,
                                                coarse_volume::Real)
    Nx > 0 || throw(ArgumentError("Nx must be positive"))
    Ny > 0 || throw(ArgumentError("Ny must be positive"))
    coarse_volume > 0 || throw(ArgumentError("coarse_volume must be positive"))
    _check_conservative_tree_patch_layout(patch)
    first(patch.parent_i_range) >= 1 ||
        throw(ArgumentError("patch.parent_i_range starts outside the domain"))
    last(patch.parent_i_range) <= Nx ||
        throw(ArgumentError("patch.parent_i_range ends outside the domain"))
    first(patch.parent_j_range) >= 1 ||
        throw(ArgumentError("patch.parent_j_range starts outside the domain"))
    last(patch.parent_j_range) <= Ny ||
        throw(ArgumentError("patch.parent_j_range ends outside the domain"))
    return nothing
end

function _push_link_route!(links::Vector{ConservativeTreeLink2D},
                           routes::Vector{ConservativeTreeRoute2D},
                           same_level_links::Vector{Int},
                           coarse_to_fine_links::Vector{Int},
                           fine_to_coarse_links::Vector{Int},
                           boundary_links::Vector{Int},
                           direct_routes::Vector{Int},
                           interface_routes::Vector{Int},
                           boundary_routes::Vector{Int},
                           src::Int,
                           q::Int,
                           link_kind::LinkKind,
                           route_specs)
    push!(links, ConservativeTreeLink2D(src, q, link_kind))
    link_index = length(links)
    if link_kind == SAME_LEVEL
        push!(same_level_links, link_index)
    elseif link_kind == COARSE_TO_FINE
        push!(coarse_to_fine_links, link_index)
    elseif link_kind == FINE_TO_COARSE
        push!(fine_to_coarse_links, link_index)
    else
        push!(boundary_links, link_index)
    end

    for spec in route_specs
        dst, weight, route_kind = spec
        push!(routes, ConservativeTreeRoute2D(src, dst, q, Float64(weight), route_kind))
        route_index = length(routes)
        if route_kind == DIRECT
            push!(direct_routes, route_index)
        elseif route_kind == ROUTE_BOUNDARY
            push!(boundary_routes, route_index)
        else
            push!(interface_routes, route_index)
        end
    end
    return nothing
end

function _coarse_to_fine_route_specs(cell_id_by_coord,
                                     I_dst::Int,
                                     J_dst::Int,
                                     di::Int,
                                     dj::Int)
    if di != 0 && dj != 0
        ix = di < 0 ? 1 : 2
        iy = dj < 0 ? 1 : 2
        dst = cell_id_by_coord[_tree_topology_key(1,
                                                  _fine_global_i(I_dst, ix),
                                                  _fine_global_j(J_dst, iy))]
        return ((dst, 1.0, SPLIT_CORNER),)
    elseif di < 0
        return (
            (cell_id_by_coord[_tree_topology_key(1, _fine_global_i(I_dst, 1), _fine_global_j(J_dst, 1))], 0.5, SPLIT_FACE),
            (cell_id_by_coord[_tree_topology_key(1, _fine_global_i(I_dst, 1), _fine_global_j(J_dst, 2))], 0.5, SPLIT_FACE),
        )
    elseif di > 0
        return (
            (cell_id_by_coord[_tree_topology_key(1, _fine_global_i(I_dst, 2), _fine_global_j(J_dst, 1))], 0.5, SPLIT_FACE),
            (cell_id_by_coord[_tree_topology_key(1, _fine_global_i(I_dst, 2), _fine_global_j(J_dst, 2))], 0.5, SPLIT_FACE),
        )
    elseif dj < 0
        return (
            (cell_id_by_coord[_tree_topology_key(1, _fine_global_i(I_dst, 1), _fine_global_j(J_dst, 1))], 0.5, SPLIT_FACE),
            (cell_id_by_coord[_tree_topology_key(1, _fine_global_i(I_dst, 2), _fine_global_j(J_dst, 1))], 0.5, SPLIT_FACE),
        )
    else
        return (
            (cell_id_by_coord[_tree_topology_key(1, _fine_global_i(I_dst, 1), _fine_global_j(J_dst, 2))], 0.5, SPLIT_FACE),
            (cell_id_by_coord[_tree_topology_key(1, _fine_global_i(I_dst, 2), _fine_global_j(J_dst, 2))], 0.5, SPLIT_FACE),
        )
    end
end

@inline function _fine_to_coarse_route_kind(i_dst_leaf::Int,
                                            j_dst_leaf::Int,
                                            patch::ConservativeTreePatch2D)
    out_x = i_dst_leaf < 2 * first(patch.parent_i_range) - 1 ||
            i_dst_leaf > 2 * last(patch.parent_i_range)
    out_y = j_dst_leaf < 2 * first(patch.parent_j_range) - 1 ||
            j_dst_leaf > 2 * last(patch.parent_j_range)
    return out_x && out_y ? COALESCE_CORNER : COALESCE_FACE
end

function _build_conservative_tree_cells_2d(Nx::Int,
                                           Ny::Int,
                                           patch::ConservativeTreePatch2D,
                                           coarse_volume::Float64)
    cells = ConservativeTreeCell2D[]
    active_cells = Int[]
    cell_id_by_coord = Dict{Tuple{Int,Int,Int},Int}()

    coarse_metrics = CartesianMetrics2D(coarse_volume)
    fine_metrics = CartesianMetrics2D(coarse_volume / 4)

    for J in 1:Ny, I in 1:Nx
        active = !_inside_range(I, J, patch.parent_i_range, patch.parent_j_range)
        push!(cells, ConservativeTreeCell2D(0, I, J, active, coarse_metrics, 0))
        id = length(cells)
        cell_id_by_coord[_tree_topology_key(0, I, J)] = id
        active && push!(active_cells, id)
    end

    for J in patch.parent_j_range, I in patch.parent_i_range
        parent = cell_id_by_coord[_tree_topology_key(0, I, J)]
        for iy in 1:2, ix in 1:2
            i_f = _fine_global_i(I, ix)
            j_f = _fine_global_j(J, iy)
            push!(cells, ConservativeTreeCell2D(1, i_f, j_f, true, fine_metrics, parent))
            id = length(cells)
            cell_id_by_coord[_tree_topology_key(1, i_f, j_f)] = id
            push!(active_cells, id)
        end
    end

    return cells, active_cells, cell_id_by_coord
end

function _build_conservative_tree_links_2d(cells::Vector{ConservativeTreeCell2D},
                                           active_cells::Vector{Int},
                                           cell_id_by_coord,
                                           Nx::Int,
                                           Ny::Int,
                                           patch::ConservativeTreePatch2D)
    links = ConservativeTreeLink2D[]
    routes = ConservativeTreeRoute2D[]
    same_level_links = Int[]
    coarse_to_fine_links = Int[]
    fine_to_coarse_links = Int[]
    boundary_links = Int[]
    direct_routes = Int[]
    interface_routes = Int[]
    boundary_routes = Int[]

    for src in active_cells
        cell = cells[src]
        for q in 1:9
            cx = d2q9_cx(q)
            cy = d2q9_cy(q)

            if cell.level == 0
                I_dst = cell.i + cx
                J_dst = cell.j + cy
                if !(1 <= I_dst <= Nx && 1 <= J_dst <= Ny)
                    _push_link_route!(links, routes,
                                      same_level_links, coarse_to_fine_links,
                                      fine_to_coarse_links, boundary_links,
                                      direct_routes, interface_routes,
                                      boundary_routes,
                                      src, q, BOUNDARY,
                                      ((0, 1.0, ROUTE_BOUNDARY),))
                elseif _inside_range(I_dst, J_dst,
                                     patch.parent_i_range,
                                     patch.parent_j_range)
                    di = cell.i < first(patch.parent_i_range) ? -1 :
                         cell.i > last(patch.parent_i_range) ? 1 : 0
                    dj = cell.j < first(patch.parent_j_range) ? -1 :
                         cell.j > last(patch.parent_j_range) ? 1 : 0
                    specs = _coarse_to_fine_route_specs(cell_id_by_coord,
                                                        I_dst, J_dst, di, dj)
                    _push_link_route!(links, routes,
                                      same_level_links, coarse_to_fine_links,
                                      fine_to_coarse_links, boundary_links,
                                      direct_routes, interface_routes,
                                      boundary_routes,
                                      src, q, COARSE_TO_FINE, specs)
                else
                    dst = cell_id_by_coord[_tree_topology_key(0, I_dst, J_dst)]
                    _push_link_route!(links, routes,
                                      same_level_links, coarse_to_fine_links,
                                      fine_to_coarse_links, boundary_links,
                                      direct_routes, interface_routes,
                                      boundary_routes,
                                      src, q, SAME_LEVEL,
                                      ((dst, 1.0, DIRECT),))
                end
            else
                i_dst = cell.i + cx
                j_dst = cell.j + cy
                if _inside_fine_patch(i_dst, j_dst, patch)
                    dst = cell_id_by_coord[_tree_topology_key(1, i_dst, j_dst)]
                    _push_link_route!(links, routes,
                                      same_level_links, coarse_to_fine_links,
                                      fine_to_coarse_links, boundary_links,
                                      direct_routes, interface_routes,
                                      boundary_routes,
                                      src, q, SAME_LEVEL,
                                      ((dst, 1.0, DIRECT),))
                elseif !_inside_leaf_domain(i_dst, j_dst, Nx, Ny)
                    _push_link_route!(links, routes,
                                      same_level_links, coarse_to_fine_links,
                                      fine_to_coarse_links, boundary_links,
                                      direct_routes, interface_routes,
                                      boundary_routes,
                                      src, q, BOUNDARY,
                                      ((0, 1.0, ROUTE_BOUNDARY),))
                else
                    I_dst, J_dst = _coarse_parent_from_fine(i_dst, j_dst)
                    dst = cell_id_by_coord[_tree_topology_key(0, I_dst, J_dst)]
                    route_kind = _fine_to_coarse_route_kind(i_dst, j_dst, patch)
                    _push_link_route!(links, routes,
                                      same_level_links, coarse_to_fine_links,
                                      fine_to_coarse_links, boundary_links,
                                      direct_routes, interface_routes,
                                      boundary_routes,
                                      src, q, FINE_TO_COARSE,
                                      ((dst, 1.0, route_kind),))
                end
            end
        end
    end

    return links, routes,
           same_level_links, coarse_to_fine_links, fine_to_coarse_links,
           boundary_links, direct_routes, interface_routes, boundary_routes
end

"""
    create_conservative_tree_topology_2d(Nx, Ny, patch; coarse_volume=1.0)

Build static D2Q9 topology tables for a fixed ratio-2 conservative-tree patch.

The topology stores active coarse cells outside `patch`, inactive coarse
ledger cells under `patch`, and active fine children inside `patch`. Links and
routes are CPU-built tables intended to be packed or copied to a GPU-oriented
layout later; no LBM state is stored here.
"""
function create_conservative_tree_topology_2d(Nx::Integer,
                                              Ny::Integer,
                                              patch::ConservativeTreePatch2D;
                                              coarse_volume::Real=1.0)
    Nx_i = Int(Nx)
    Ny_i = Int(Ny)
    _check_conservative_tree_topology_args(Nx_i, Ny_i, patch, coarse_volume)

    cells, active_cells, cell_id_by_coord =
        _build_conservative_tree_cells_2d(Nx_i, Ny_i, patch, Float64(coarse_volume))

    links, routes,
    same_level_links, coarse_to_fine_links, fine_to_coarse_links,
    boundary_links, direct_routes, interface_routes, boundary_routes =
        _build_conservative_tree_links_2d(cells, active_cells, cell_id_by_coord,
                                          Nx_i, Ny_i, patch)

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
