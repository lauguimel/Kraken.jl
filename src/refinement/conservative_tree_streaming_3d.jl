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

function _cell_id_by_coord_3d(topology::ConservativeTreeTopology3D)
    cell_id_by_coord = Dict{Tuple{Int,Int,Int,Int},Int}()
    @inbounds for (id, cell) in pairs(topology.cells)
        cell_id_by_coord[_tree_topology_key_3d(cell.level, cell.i, cell.j, cell.k)] = id
    end
    return cell_id_by_coord
end

@inline function _periodic_x_wrapped_3d(i::Int, nx::Int)
    return i < 1 ? nx : (i > nx ? 1 : i)
end

function _stream_periodic_x_boundary_route_3d!(
        coarse_out::AbstractArray{<:Any,4},
        patch_out::ConservativeTreePatch3D,
        coarse_in::AbstractArray{<:Any,4},
        patch_in::ConservativeTreePatch3D,
        topology::ConservativeTreeTopology3D,
        cell_id_by_coord,
        route::ConservativeTreeRoute3D,
        nx::Int,
        ny::Int,
        nz::Int)
    src_cell = topology.cells[route.src]
    q = route.q
    cx = d3q19_cx(q)
    cy = d3q19_cy(q)
    cz = d3q19_cz(q)
    cx == 0 && return false

    if src_cell.level == 0
        i_raw = src_cell.i + cx
        j_dst = src_cell.j + cy
        k_dst = src_cell.k + cz
        (i_raw < 1 || i_raw > nx) || return false
        (1 <= j_dst <= ny && 1 <= k_dst <= nz) || return false

        i_dst = _periodic_x_wrapped_3d(i_raw, nx)
        if _inside_range_3d(i_dst, j_dst, k_dst,
                            patch_in.parent_i_range,
                            patch_in.parent_j_range,
                            patch_in.parent_k_range)
            di = -cx
            dj = cy == 0 ? 0 : -cy
            dk = cz == 0 ? 0 : -cz
            specs = _coarse_to_fine_route_specs_3d(
                cell_id_by_coord, i_dst, j_dst, k_dst, di, dj, dk)
            @inbounds for spec in specs
                dst, weight, _ = spec
                dst_cell = topology.cells[dst]
                _scatter_route_packet_3d!(coarse_out, patch_out,
                                           coarse_in, patch_in,
                                           src_cell, dst_cell, q, weight)
            end
        else
            dst = cell_id_by_coord[_tree_topology_key_3d(0, i_dst, j_dst, k_dst)]
            dst_cell = topology.cells[dst]
            _scatter_route_packet_3d!(coarse_out, patch_out,
                                      coarse_in, patch_in,
                                      src_cell, dst_cell, q, route.weight)
        end
        return true
    end

    i_raw = src_cell.i + cx
    j_dst = src_cell.j + cy
    k_dst = src_cell.k + cz
    (i_raw < 1 || i_raw > 2 * nx) || return false
    (1 <= j_dst <= 2 * ny && 1 <= k_dst <= 2 * nz) || return false

    i_dst = _periodic_x_wrapped_3d(i_raw, 2 * nx)
    if _inside_fine_patch_3d(i_dst, j_dst, k_dst, patch_in)
        dst = cell_id_by_coord[_tree_topology_key_3d(1, i_dst, j_dst, k_dst)]
        dst_cell = topology.cells[dst]
        _scatter_route_packet_3d!(coarse_out, patch_out,
                                  coarse_in, patch_in,
                                  src_cell, dst_cell, q, route.weight)
    else
        I_dst, J_dst, K_dst = _coarse_parent_from_fine_3d(i_dst, j_dst, k_dst)
        dst = cell_id_by_coord[_tree_topology_key_3d(0, I_dst, J_dst, K_dst)]
        dst_cell = topology.cells[dst]
        _scatter_route_packet_3d!(coarse_out, patch_out,
                                  coarse_in, patch_in,
                                  src_cell, dst_cell, q, route.weight)
    end
    return true
end

function _stream_wall_yz_boundary_route_3d!(
        coarse_out::AbstractArray{<:Any,4},
        patch_out::ConservativeTreePatch3D,
        coarse_in::AbstractArray{<:Any,4},
        patch_in::ConservativeTreePatch3D,
        topology::ConservativeTreeTopology3D,
        route::ConservativeTreeRoute3D,
        ny::Int,
        nz::Int)
    src_cell = topology.cells[route.src]
    q = route.q
    cy = d3q19_cy(q)
    cz = d3q19_cz(q)

    if src_cell.level == 0
        j_raw = src_cell.j + cy
        k_raw = src_cell.k + cz
        (j_raw < 1 || j_raw > ny || k_raw < 1 || k_raw > nz) || return false
    else
        j_raw = src_cell.j + cy
        k_raw = src_cell.k + cz
        (j_raw < 1 || j_raw > 2 * ny || k_raw < 1 || k_raw > 2 * nz) || return false
    end

    packet = _cell_Fq_3d(coarse_in, patch_in, src_cell, q)
    _add_cell_Fq_3d!(coarse_out, patch_out, src_cell, d3q19_opposite(q),
                     route.weight * packet)
    return true
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

    nx = size(coarse_in, 1)
    ny = size(coarse_in, 2)
    nz = size(coarse_in, 3)
    cell_id_by_coord = boundary_policy in (:periodic_x, :periodic_x_wall_yz) ?
        _cell_id_by_coord_3d(topology) : nothing

    @inbounds for route in topology.routes
        if route.kind == ROUTE_BOUNDARY_3D
            boundary_policy == :skip && continue
            boundary_policy in (:periodic_x, :periodic_x_wall_yz) ||
                throw(ArgumentError("unsupported route boundary policy: $boundary_policy"))
            handled = false
            if boundary_policy == :periodic_x_wall_yz
                handled = _stream_wall_yz_boundary_route_3d!(
                    coarse_out, patch_out, coarse_in, patch_in,
                    topology, route, ny, nz)
            end
            handled || _stream_periodic_x_boundary_route_3d!(
                coarse_out, patch_out, coarse_in, patch_in, topology,
                cell_id_by_coord, route, nx, ny, nz)
            continue
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

function stream_composite_routes_periodic_x_F_3d!(
        coarse_out::AbstractArray{<:Any,4},
        patch_out::ConservativeTreePatch3D,
        coarse_in::AbstractArray{<:Any,4},
        patch_in::ConservativeTreePatch3D,
        topology::ConservativeTreeTopology3D;
        clear::Bool=true)
    return _stream_composite_routes_F_3d!(coarse_out, patch_out,
                                          coarse_in, patch_in,
                                          topology, :periodic_x, clear)
end

function stream_composite_routes_periodic_x_wall_yz_F_3d!(
        coarse_out::AbstractArray{<:Any,4},
        patch_out::ConservativeTreePatch3D,
        coarse_in::AbstractArray{<:Any,4},
        patch_in::ConservativeTreePatch3D,
        topology::ConservativeTreeTopology3D;
        clear::Bool=true)
    return _stream_composite_routes_F_3d!(coarse_out, patch_out,
                                          coarse_in, patch_in,
                                          topology, :periodic_x_wall_yz, clear)
end
