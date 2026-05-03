# Route-driven streaming canaries for the conservative-tree D3Q19 prototype.
#
# This first transport layer handles only routes already present in
# ConservativeTreeTopology3D. Boundary routes are explicit and skipped by the
# interior primitive.

@inline function _cell_Fq_3d(coarse_F::AbstractArray{<:Any,4},
                             patch::ConservativeTreePatch3D,
                             cell::ConservativeTreeCell3D,
                             q::Int)
    if cell.level == 0
        return coarse_F[cell.i, cell.j, cell.k, q]
    else
        i0 = 2 * first(patch.parent_i_range) - 1
        j0 = 2 * first(patch.parent_j_range) - 1
        k0 = 2 * first(patch.parent_k_range) - 1
        return patch.fine_F[cell.i - i0 + 1,
                            cell.j - j0 + 1,
                            cell.k - k0 + 1,
                            q]
    end
end

@inline function _add_cell_Fq_3d!(coarse_F::AbstractArray{<:Any,4},
                                  patch::ConservativeTreePatch3D,
                                  cell::ConservativeTreeCell3D,
                                  q::Int,
                                  value)
    if cell.level == 0
        coarse_F[cell.i, cell.j, cell.k, q] += value
    else
        i0 = 2 * first(patch.parent_i_range) - 1
        j0 = 2 * first(patch.parent_j_range) - 1
        k0 = 2 * first(patch.parent_k_range) - 1
        patch.fine_F[cell.i - i0 + 1,
                     cell.j - j0 + 1,
                     cell.k - k0 + 1,
                     q] += value
    end
    return nothing
end

function _check_route_stream_topology_layout_3d(topology::ConservativeTreeTopology3D,
                                                coarse_F::AbstractArray{<:Any,4},
                                                patch::ConservativeTreePatch3D)
    _check_composite_coarse_layout_3d(coarse_F, patch)
    nx = size(coarse_F, 1)
    ny = size(coarse_F, 2)
    nz = size(coarse_F, 3)
    @inbounds for id in topology.active_cells
        cell = topology.cells[id]
        if cell.level == 0
            cell.active || throw(ArgumentError("topology active_cells includes inactive coarse cell"))
            (first(axes(coarse_F, 1)) <= cell.i <= last(axes(coarse_F, 1)) &&
             first(axes(coarse_F, 2)) <= cell.j <= last(axes(coarse_F, 2)) &&
             first(axes(coarse_F, 3)) <= cell.k <= last(axes(coarse_F, 3))) ||
                throw(ArgumentError("topology coarse cell lies outside coarse_F"))
            _inside_range_3d(cell.i, cell.j, cell.k,
                             patch.parent_i_range,
                             patch.parent_j_range,
                             patch.parent_k_range) &&
                throw(ArgumentError("topology coarse active cell lies inside patch"))
        elseif cell.level == 1
            _inside_fine_patch_3d(cell.i, cell.j, cell.k, patch) ||
                throw(ArgumentError("topology fine cell lies outside patch"))
            _inside_leaf_domain_3d(cell.i, cell.j, cell.k, nx, ny, nz) ||
                throw(ArgumentError("topology fine cell lies outside leaf domain"))
        else
            throw(ArgumentError("only levels 0 and 1 are supported"))
        end
    end
    return nothing
end

@inline function _scatter_route_packet_3d!(
        coarse_out::AbstractArray{<:Any,4},
        patch_out::ConservativeTreePatch3D,
        coarse_in::AbstractArray{<:Any,4},
        patch_in::ConservativeTreePatch3D,
        src_cell::ConservativeTreeCell3D,
        dst_cell::ConservativeTreeCell3D,
        q::Int,
        weight)
    packet = _cell_Fq_3d(coarse_in, patch_in, src_cell, q)
    _add_cell_Fq_3d!(coarse_out, patch_out, dst_cell, q, weight * packet)
    return nothing
end

function _stream_composite_routes_F_3d!(
        coarse_out::AbstractArray{<:Any,4},
        patch_out::ConservativeTreePatch3D,
        coarse_in::AbstractArray{<:Any,4},
        patch_in::ConservativeTreePatch3D,
        topology::ConservativeTreeTopology3D,
        boundary_policy::Symbol,
        clear::Bool)
    _check_composite_pair_layout_3d(coarse_out, patch_out, coarse_in, patch_in)
    _check_route_stream_topology_layout_3d(topology, coarse_in, patch_in)
    _check_route_stream_topology_layout_3d(topology, coarse_out, patch_out)

    if clear
        coarse_out .= 0
        patch_out.fine_F .= 0
        patch_out.coarse_shadow_F .= 0
    end

    @inbounds for route in topology.routes
        if route.kind == ROUTE_BOUNDARY_3D
            boundary_policy == :skip && continue
            throw(ArgumentError("unsupported route boundary policy: $boundary_policy"))
        end

        src_cell = topology.cells[route.src]
        dst_cell = topology.cells[route.dst]
        _scatter_route_packet_3d!(coarse_out, patch_out, coarse_in, patch_in,
                                  src_cell, dst_cell, route.q, route.weight)
    end
    coalesce_patch_to_shadow_F_3d!(patch_out)
    return coarse_out, patch_out
end

"""
    stream_composite_routes_interior_F_3d!(coarse_out, patch_out,
                                           coarse_in, patch_in, topology;
                                           clear=true)

Scatter integrated D3Q19 populations along the non-boundary routes of a
`ConservativeTreeTopology3D`.

This is a surgical native-composite streaming primitive for the 3D AMR route.
It conserves every non-boundary route packet exactly up to roundoff. Boundary
routes are explicit in the topology and intentionally skipped here.
"""
function stream_composite_routes_interior_F_3d!(
        coarse_out::AbstractArray{<:Any,4},
        patch_out::ConservativeTreePatch3D,
        coarse_in::AbstractArray{<:Any,4},
        patch_in::ConservativeTreePatch3D,
        topology::ConservativeTreeTopology3D;
        clear::Bool=true)
    return _stream_composite_routes_F_3d!(coarse_out, patch_out,
                                          coarse_in, patch_in,
                                          topology, :skip, clear)
end
