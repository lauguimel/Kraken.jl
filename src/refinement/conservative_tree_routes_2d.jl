# Route-table construction for nested 2D conservative-tree specs.
#
# This is still a static correctness layer. It builds packet routes for active
# leaves, but it does not stream populations.

struct ConservativeTreeRouteTable2D
    links::Vector{ConservativeTreeLink2D}
    routes::Vector{ConservativeTreeRoute2D}
    same_level_links::Vector{Int}
    coarse_to_fine_links::Vector{Int}
    fine_to_coarse_links::Vector{Int}
    boundary_links::Vector{Int}
    direct_routes::Vector{Int}
    interface_routes::Vector{Int}
    boundary_routes::Vector{Int}
end

function _active_leaf_covering_sample_2d(spec::ConservativeTreeSpec2D,
                                         sample_level::Int,
                                         i::Int,
                                         j::Int)
    0 <= sample_level <= spec.max_level ||
        throw(ArgumentError("sample_level is outside the tree"))
    nx = _conservative_tree_level_size_2d(spec.Nx, sample_level)
    ny = _conservative_tree_level_size_2d(spec.Ny, sample_level)
    (1 <= i <= nx && 1 <= j <= ny) || return 0

    for level in sample_level:-1:0
        scale = 1 << (sample_level - level)
        ai = div(i - 1, scale) + 1
        aj = div(j - 1, scale) + 1
        cell_id = conservative_tree_cell_id_2d(spec, level, ai, aj)
        cell_id == 0 && continue
        spec.cells[cell_id].active && return cell_id
    end
    return 0
end

@inline function _route_kind_for_level_pair_2d(src::ConservativeTreeCell2D,
                                               dst::ConservativeTreeCell2D,
                                               q::Int)
    delta = dst.level - src.level
    if delta == 0
        return DIRECT
    end
    abs(delta) == 1 ||
        throw(ArgumentError("route crosses more than one AMR level"))
    is_corner = d2q9_cx(q) != 0 && d2q9_cy(q) != 0
    if delta == 1
        return is_corner ? SPLIT_CORNER : SPLIT_FACE
    end
    return is_corner ? COALESCE_CORNER : COALESCE_FACE
end

@inline function _link_kind_for_route_kinds_2d(kinds::Vector{RouteKind})
    has_boundary = false
    has_split = false
    has_coalesce = false
    @inbounds for kind in kinds
        if kind == ROUTE_BOUNDARY
            has_boundary = true
        elseif kind == SPLIT_FACE || kind == SPLIT_CORNER
            has_split = true
        elseif kind == COALESCE_FACE || kind == COALESCE_CORNER
            has_coalesce = true
        end
    end
    has_boundary && return BOUNDARY
    has_split && return COARSE_TO_FINE
    has_coalesce && return FINE_TO_COARSE
    return SAME_LEVEL
end

function _accumulate_route_spec_2d!(dsts::Vector{Int},
                                    weights::Vector{Float64},
                                    kinds::Vector{RouteKind},
                                    dst::Int,
                                    weight::Float64,
                                    kind::RouteKind)
    @inbounds for idx in eachindex(dsts)
        if dsts[idx] == dst && kinds[idx] == kind
            weights[idx] += weight
            return nothing
        end
    end
    push!(dsts, dst)
    push!(weights, weight)
    push!(kinds, kind)
    return nothing
end

function _route_specs_for_active_cell_2d(spec::ConservativeTreeSpec2D,
                                         src_id::Int,
                                         q::Int)
    src = spec.cells[src_id]
    sample_level = min(src.level + 1, spec.max_level)
    scale = 1 << (sample_level - src.level)
    cx = d2q9_cx(q)
    cy = d2q9_cy(q)
    weight = 1.0 / Float64(scale * scale)

    dsts = Int[]
    weights = Float64[]
    kinds = RouteKind[]
    for sj in 1:scale, si in 1:scale
        sample_i = (src.i - 1) * scale + si + cx
        sample_j = (src.j - 1) * scale + sj + cy
        dst_id = _active_leaf_covering_sample_2d(spec, sample_level,
                                                 sample_i, sample_j)
        if dst_id == 0
            _accumulate_route_spec_2d!(dsts, weights, kinds,
                                       0, weight, ROUTE_BOUNDARY)
        else
            dst = spec.cells[dst_id]
            kind = _route_kind_for_level_pair_2d(src, dst, q)
            _accumulate_route_spec_2d!(dsts, weights, kinds,
                                       dst_id, weight, kind)
        end
    end
    return dsts, weights, kinds
end

function _push_multilevel_link_routes_2d!(
        links::Vector{ConservativeTreeLink2D},
        routes::Vector{ConservativeTreeRoute2D},
        same_level_links::Vector{Int},
        coarse_to_fine_links::Vector{Int},
        fine_to_coarse_links::Vector{Int},
        boundary_links::Vector{Int},
        direct_routes::Vector{Int},
        interface_routes::Vector{Int},
        boundary_routes::Vector{Int},
        src_id::Int,
        q::Int,
        dsts::Vector{Int},
        weights::Vector{Float64},
        kinds::Vector{RouteKind})
    link_kind = _link_kind_for_route_kinds_2d(kinds)
    push!(links, ConservativeTreeLink2D(src_id, q, link_kind))
    link_id = length(links)
    if link_kind == SAME_LEVEL
        push!(same_level_links, link_id)
    elseif link_kind == COARSE_TO_FINE
        push!(coarse_to_fine_links, link_id)
    elseif link_kind == FINE_TO_COARSE
        push!(fine_to_coarse_links, link_id)
    else
        push!(boundary_links, link_id)
    end

    @inbounds for idx in eachindex(dsts)
        push!(routes, ConservativeTreeRoute2D(src_id, dsts[idx], q,
                                              weights[idx], kinds[idx]))
        route_id = length(routes)
        if kinds[idx] == DIRECT
            push!(direct_routes, route_id)
        elseif kinds[idx] == ROUTE_BOUNDARY
            push!(boundary_routes, route_id)
        else
            push!(interface_routes, route_id)
        end
    end
    return nothing
end

"""
    create_conservative_tree_route_table_2d(spec)

Build a static packet route table for every active leaf and D2Q9 population in
a nested conservative-tree spec. Routes are generated by leaf-equivalent
sampling one level below each source cell, then mapped back to the active leaf
that owns each sample.
"""
function create_conservative_tree_route_table_2d(spec::ConservativeTreeSpec2D)
    links = ConservativeTreeLink2D[]
    routes = ConservativeTreeRoute2D[]
    same_level_links = Int[]
    coarse_to_fine_links = Int[]
    fine_to_coarse_links = Int[]
    boundary_links = Int[]
    direct_routes = Int[]
    interface_routes = Int[]
    boundary_routes = Int[]

    for src_id in spec.active_cells
        for q in 1:9
            dsts, weights, kinds = _route_specs_for_active_cell_2d(spec, src_id, q)
            _push_multilevel_link_routes_2d!(
                links, routes, same_level_links, coarse_to_fine_links,
                fine_to_coarse_links, boundary_links, direct_routes,
                interface_routes, boundary_routes, src_id, q,
                dsts, weights, kinds)
        end
    end

    return ConservativeTreeRouteTable2D(
        links, routes, same_level_links, coarse_to_fine_links,
        fine_to_coarse_links, boundary_links, direct_routes, interface_routes,
        boundary_routes)
end
