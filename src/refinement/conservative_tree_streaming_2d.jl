# Route-driven streaming canaries for the conservative-tree D2Q9 prototype.
#
# This file is the first native-composite transport layer. It deliberately
# handles only routes already present in ConservativeTreeTopology2D. Boundary
# routes are explicit and skipped by default; native boundary closures are a
# later milestone.

@inline function _cell_Fq_2d(coarse_F::AbstractArray{<:Any,3},
                             patch::ConservativeTreePatch2D,
                             cell::ConservativeTreeCell2D,
                             q::Int)
    if cell.level == 0
        return coarse_F[cell.i, cell.j, q]
    else
        i0 = 2 * first(patch.parent_i_range) - 1
        j0 = 2 * first(patch.parent_j_range) - 1
        return patch.fine_F[cell.i - i0 + 1, cell.j - j0 + 1, q]
    end
end

@inline function _add_cell_Fq_2d!(coarse_F::AbstractArray{<:Any,3},
                                  patch::ConservativeTreePatch2D,
                                  cell::ConservativeTreeCell2D,
                                  q::Int,
                                  value)
    if cell.level == 0
        coarse_F[cell.i, cell.j, q] += value
    else
        i0 = 2 * first(patch.parent_i_range) - 1
        j0 = 2 * first(patch.parent_j_range) - 1
        patch.fine_F[cell.i - i0 + 1, cell.j - j0 + 1, q] += value
    end
    return nothing
end

function _check_route_stream_topology_layout(topology::ConservativeTreeTopology2D,
                                             coarse_F::AbstractArray{<:Any,3},
                                             patch::ConservativeTreePatch2D)
    _check_composite_coarse_layout(coarse_F, patch)
    nx = size(coarse_F, 1)
    ny = size(coarse_F, 2)
    @inbounds for id in topology.active_cells
        cell = topology.cells[id]
        if cell.level == 0
            cell.active || throw(ArgumentError("topology active_cells includes inactive coarse cell"))
            _inside_array_2d(coarse_F, cell.i, cell.j) ||
                throw(ArgumentError("topology coarse cell lies outside coarse_F"))
            _inside_range(cell.i, cell.j, patch.parent_i_range, patch.parent_j_range) &&
                throw(ArgumentError("topology coarse active cell lies inside patch"))
        elseif cell.level == 1
            _inside_fine_patch(cell.i, cell.j, patch) ||
                throw(ArgumentError("topology fine cell lies outside patch"))
            _inside_leaf_domain(cell.i, cell.j, nx, ny) ||
                throw(ArgumentError("topology fine cell lies outside leaf domain"))
        else
            throw(ArgumentError("only levels 0 and 1 are supported"))
        end
    end
    return nothing
end

"""
    stream_composite_routes_interior_F_2d!(coarse_out, patch_out,
                                           coarse_in, patch_in, topology;
                                           clear=true)

Scatter integrated D2Q9 populations along the non-boundary routes of a
`ConservativeTreeTopology2D`.

This is a surgical native-composite streaming primitive for the first AMR
milestone. It conserves every non-boundary route packet exactly up to roundoff.
Boundary routes are explicit in the topology and intentionally skipped here;
native periodic/wall/Zou-He closures are added in a later milestone.
"""
function stream_composite_routes_interior_F_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D,
        topology::ConservativeTreeTopology2D;
        clear::Bool=true)
    _check_composite_pair_layout(coarse_out, patch_out, coarse_in, patch_in)
    _check_route_stream_topology_layout(topology, coarse_in, patch_in)
    _check_route_stream_topology_layout(topology, coarse_out, patch_out)

    if clear
        coarse_out .= 0
        patch_out.fine_F .= 0
        patch_out.coarse_shadow_F .= 0
    end

    @inbounds for route in topology.routes
        route.kind == ROUTE_BOUNDARY && continue
        src_cell = topology.cells[route.src]
        dst_cell = topology.cells[route.dst]
        packet = _cell_Fq_2d(coarse_in, patch_in, src_cell, route.q)
        _add_cell_Fq_2d!(coarse_out, patch_out, dst_cell, route.q,
                         route.weight * packet)
    end
    coalesce_patch_to_shadow_F_2d!(patch_out)
    return coarse_out, patch_out
end

