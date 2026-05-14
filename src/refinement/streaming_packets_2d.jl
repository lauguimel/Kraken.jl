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

@inline function _cell_view_2d(coarse_F::AbstractArray{<:Any,3},
                               patch::ConservativeTreePatch2D,
                               cell::ConservativeTreeCell2D)
    if cell.level == 0
        return @view coarse_F[cell.i, cell.j, :]
    else
        i0 = 2 * first(patch.parent_i_range) - 1
        j0 = 2 * first(patch.parent_j_range) - 1
        return @view patch.fine_F[cell.i - i0 + 1, cell.j - j0 + 1, :]
    end
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

function _check_route_solid_mask_layout(topology::ConservativeTreeTopology2D,
                                        coarse_F::AbstractArray{<:Any,3},
                                        patch::ConservativeTreePatch2D,
                                        is_solid::AbstractArray{Bool,2})
    size(is_solid) == (2 * size(coarse_F, 1), 2 * size(coarse_F, 2)) ||
        throw(ArgumentError("is_solid must have size (2*Nx, 2*Ny)"))

    @inbounds for id in topology.active_cells
        cell = topology.cells[id]
        cell.level == 1 && continue

        i0 = 2 * cell.i - 1
        j0 = 2 * cell.j - 1
        s11 = is_solid[i0, j0]
        s21 = is_solid[i0 + 1, j0]
        s12 = is_solid[i0, j0 + 1]
        s22 = is_solid[i0 + 1, j0 + 1]
        (s11 == s21 == s12 == s22) ||
            throw(ArgumentError("active coarse cells cannot be partially solid"))
    end
    return nothing
end

@inline function _cell_is_solid_2d(cell::ConservativeTreeCell2D,
                                   is_solid::AbstractArray{Bool,2})
    if cell.level == 0
        return is_solid[2 * cell.i - 1, 2 * cell.j - 1]
    end
    return is_solid[cell.i, cell.j]
end

function _cell_id_by_coord_2d(topology::ConservativeTreeTopology2D)
    cell_id_by_coord = Dict{Tuple{Int,Int,Int},Int}()
    @inbounds for (id, cell) in pairs(topology.cells)
        cell_id_by_coord[_tree_topology_key(cell.level, cell.i, cell.j)] = id
    end
    return cell_id_by_coord
end

@inline function _scatter_route_packet_2d!(coarse_out::AbstractArray{<:Any,3},
                                           patch_out::ConservativeTreePatch2D,
                                           coarse_in::AbstractArray{<:Any,3},
                                           patch_in::ConservativeTreePatch2D,
                                           src_cell::ConservativeTreeCell2D,
                                           dst_cell::ConservativeTreeCell2D,
                                           q::Int,
                                           weight)
    packet = _cell_Fq_2d(coarse_in, patch_in, src_cell, q)
    _add_cell_Fq_2d!(coarse_out, patch_out, dst_cell, q, weight * packet)
    return nothing
end

@inline function _coarse_child_limited_linear_Fq_2d(
        coarse_F::AbstractArray{<:Any,3},
        patch::ConservativeTreePatch2D,
        I::Int,
        J::Int,
        q::Int,
        ix::Int,
        iy::Int)
    center = _composite_parent_Fq(coarse_F, patch, I, J, q)
    sx = _limited_parent_slope_x(coarse_F, patch, I, J, q)
    sy = _limited_parent_slope_y(coarse_F, patch, I, J, q)

    max_delta = (abs(sx) + abs(sy)) / 16
    base = center / 4
    if max_delta > zero(max_delta) && base < max_delta
        theta = base / max_delta
        sx *= theta
        sy *= theta
    end

    xsign = ix == 1 ? -1 : 1
    ysign = iy == 1 ? -1 : 1
    return base + xsign * sx / 16 + ysign * sy / 16
end

function _scatter_limited_linear_coarse_route_packet_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D,
        topology::ConservativeTreeTopology2D,
        cell_id_by_coord,
        src_cell::ConservativeTreeCell2D,
        route::ConservativeTreeRoute2D,
        nx::Int,
        ny::Int)
    q = route.q
    cx = d2q9_cx(q)
    cy = d2q9_cy(q)
    split_kind = (src_cell.i < first(patch_in.parent_i_range) ||
                  src_cell.i > last(patch_in.parent_i_range)) &&
                 (src_cell.j < first(patch_in.parent_j_range) ||
                  src_cell.j > last(patch_in.parent_j_range)) ? SPLIT_CORNER : SPLIT_FACE
    handled = false

    @inbounds for iy in 1:2, ix in 1:2
        i_dst = _fine_global_i(src_cell.i, ix) + cx
        j_dst = _fine_global_j(src_cell.j, iy) + cy
        _inside_leaf_domain(i_dst, j_dst, nx, ny) || continue

        if _inside_fine_patch(i_dst, j_dst, patch_in)
            dst = cell_id_by_coord[_tree_topology_key(1, i_dst, j_dst)]
            kind = split_kind
        else
            I_dst, J_dst = _coarse_parent_from_fine(i_dst, j_dst)
            dst = cell_id_by_coord[_tree_topology_key(0, I_dst, J_dst)]
            kind = DIRECT
        end
        dst == route.dst && kind == route.kind || continue

        dst_cell = topology.cells[dst]
        value = _coarse_child_limited_linear_Fq_2d(
            coarse_in, patch_in, src_cell.i, src_cell.j, q, ix, iy)
        _add_cell_Fq_2d!(coarse_out, patch_out, dst_cell, q, value)
        handled = true
    end
    return handled
end

function _scatter_limited_linear_coarse_solid_route_packet_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D,
        topology::ConservativeTreeTopology2D,
        cell_id_by_coord,
        src_cell::ConservativeTreeCell2D,
        route::ConservativeTreeRoute2D,
        nx::Int,
        ny::Int,
        is_solid::AbstractArray{Bool,2})
    q = route.q
    cx = d2q9_cx(q)
    cy = d2q9_cy(q)
    split_kind = (src_cell.i < first(patch_in.parent_i_range) ||
                  src_cell.i > last(patch_in.parent_i_range)) &&
                 (src_cell.j < first(patch_in.parent_j_range) ||
                  src_cell.j > last(patch_in.parent_j_range)) ? SPLIT_CORNER : SPLIT_FACE
    handled = false

    @inbounds for iy in 1:2, ix in 1:2
        i_dst = _fine_global_i(src_cell.i, ix) + cx
        j_dst = _fine_global_j(src_cell.j, iy) + cy
        _inside_leaf_domain(i_dst, j_dst, nx, ny) || continue

        if _inside_fine_patch(i_dst, j_dst, patch_in)
            dst = cell_id_by_coord[_tree_topology_key(1, i_dst, j_dst)]
            kind = split_kind
        else
            I_dst, J_dst = _coarse_parent_from_fine(i_dst, j_dst)
            dst = cell_id_by_coord[_tree_topology_key(0, I_dst, J_dst)]
            kind = DIRECT
        end
        dst == route.dst && kind == route.kind || continue

        dst_cell = topology.cells[dst]
        value = _coarse_child_limited_linear_Fq_2d(
            coarse_in, patch_in, src_cell.i, src_cell.j, q, ix, iy)
        if _cell_is_solid_2d(dst_cell, is_solid)
            _add_cell_Fq_2d!(coarse_out, patch_out, src_cell,
                             d2q9_opposite(q), value)
        else
            _add_cell_Fq_2d!(coarse_out, patch_out, dst_cell, q, value)
        end
        handled = true
    end
    return handled
end

@inline function _scatter_solid_route_packet_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D,
        src_cell::ConservativeTreeCell2D,
        dst_cell::ConservativeTreeCell2D,
        q::Int,
        weight,
        is_solid::AbstractArray{Bool,2})
    _cell_is_solid_2d(src_cell, is_solid) && return nothing

    packet = _cell_Fq_2d(coarse_in, patch_in, src_cell, q)
    if _cell_is_solid_2d(dst_cell, is_solid)
        _add_cell_Fq_2d!(coarse_out, patch_out, src_cell,
                         d2q9_opposite(q), weight * packet)
    else
        _add_cell_Fq_2d!(coarse_out, patch_out, dst_cell, q, weight * packet)
    end
    return nothing
end

@inline function _periodic_x_wrapped(i::Int, nx::Int)
    return i < 1 ? nx : (i > nx ? 1 : i)
end

function _stream_periodic_x_leaf_equivalent_boundary_samples_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D,
        topology::ConservativeTreeTopology2D,
        cell_id_by_coord,
        src_cell::ConservativeTreeCell2D,
        q::Int,
        nx::Int,
        ny::Int;
        coarse_prolongation::Symbol=:flat)
    cx = d2q9_cx(q)
    cy = d2q9_cy(q)
    cx == 0 && return false

    handled = false
    @inbounds for iy in 1:2, ix in 1:2
        i_raw = _fine_global_i(src_cell.i, ix) + cx
        j_dst = _fine_global_j(src_cell.j, iy) + cy
        (i_raw < 1 || i_raw > 2 * nx) || continue
        1 <= j_dst <= 2 * ny || continue

        i_dst = _periodic_x_wrapped(i_raw, 2 * nx)
        if _inside_fine_patch(i_dst, j_dst, patch_in)
            dst = cell_id_by_coord[_tree_topology_key(1, i_dst, j_dst)]
        else
            I_dst, J_dst = _coarse_parent_from_fine(i_dst, j_dst)
            dst = cell_id_by_coord[_tree_topology_key(0, I_dst, J_dst)]
        end
        dst_cell = topology.cells[dst]
        if coarse_prolongation == :limited_linear
            value = _coarse_child_limited_linear_Fq_2d(
                coarse_in, patch_in, src_cell.i, src_cell.j, q, ix, iy)
            _add_cell_Fq_2d!(coarse_out, patch_out, dst_cell, q, value)
        else
            _scatter_route_packet_2d!(coarse_out, patch_out,
                                       coarse_in, patch_in,
                                       src_cell, dst_cell, q, 0.25)
        end
        handled = true
    end
    return handled
end

function _stream_periodic_x_wall_y_leaf_equivalent_boundary_samples_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D,
        topology::ConservativeTreeTopology2D,
        cell_id_by_coord,
        src_cell::ConservativeTreeCell2D,
        q::Int,
        nx::Int,
        ny::Int,
        u_south,
        u_north,
        rho_wall,
        volume_coarse;
        coarse_prolongation::Symbol=:flat)
    cx = d2q9_cx(q)
    cy = d2q9_cy(q)
    q_dst = d2q9_opposite(q)
    handled = false

    @inbounds for iy in 1:2, ix in 1:2
        i_leaf = _fine_global_i(src_cell.i, ix) + cx
        j_leaf = _fine_global_j(src_cell.j, iy) + cy
        _inside_leaf_domain(i_leaf, j_leaf, nx, ny) && continue

        value = if coarse_prolongation == :limited_linear
            _coarse_child_limited_linear_Fq_2d(
                coarse_in, patch_in, src_cell.i, src_cell.j, q, ix, iy)
        else
            0.25 * _cell_Fq_2d(coarse_in, patch_in, src_cell, q)
        end

        if j_leaf < 1 || j_leaf > 2 * ny
            wall_u = j_leaf < 1 ? u_south : u_north
            correction = _moving_wall_delta(
                volume_coarse / 4, rho_wall, wall_u, q_dst)
            _add_cell_Fq_2d!(coarse_out, patch_out, src_cell, q_dst,
                             value + correction)
            handled = true
        elseif i_leaf < 1 || i_leaf > 2 * nx
            i_dst = _periodic_x_wrapped(i_leaf, 2 * nx)
            if _inside_fine_patch(i_dst, j_leaf, patch_in)
                dst = cell_id_by_coord[_tree_topology_key(1, i_dst, j_leaf)]
            else
                I_dst, J_dst = _coarse_parent_from_fine(i_dst, j_leaf)
                dst = cell_id_by_coord[_tree_topology_key(0, I_dst, J_dst)]
            end
            dst_cell = topology.cells[dst]
            _add_cell_Fq_2d!(coarse_out, patch_out, dst_cell, q, value)
            handled = true
        end
    end
    return handled
end

function _stream_periodic_x_boundary_route_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D,
        topology::ConservativeTreeTopology2D,
        cell_id_by_coord,
        route::ConservativeTreeRoute2D,
        nx::Int,
        ny::Int;
        coarse_prolongation::Symbol=:flat)
    src_cell = topology.cells[route.src]
    q = route.q
    cx = d2q9_cx(q)
    cy = d2q9_cy(q)
    cx == 0 && return false

    if src_cell.level == 0
        i_raw = src_cell.i + cx
        j_dst = src_cell.j + cy
        (i_raw < 1 || i_raw > nx) || return false
        1 <= j_dst <= ny || return false

        i_dst = _periodic_x_wrapped(i_raw, nx)
        if route.weight < 1.0
            return _stream_periodic_x_leaf_equivalent_boundary_samples_2d!(
                coarse_out, patch_out, coarse_in, patch_in,
                topology, cell_id_by_coord, src_cell, q, nx, ny;
                coarse_prolongation=coarse_prolongation)
        end
        if _inside_range(i_dst, j_dst, patch_in.parent_i_range, patch_in.parent_j_range)
            specs = _coarse_to_fine_route_specs(cell_id_by_coord,
                                                src_cell.i, src_cell.j, q,
                                                patch_in, nx, ny;
                                                periodic_x=true)
            @inbounds for spec in specs
                dst, weight, _ = spec
                dst == 0 && continue
                dst_cell = topology.cells[dst]
                _scatter_route_packet_2d!(coarse_out, patch_out,
                                           coarse_in, patch_in,
                                           src_cell, dst_cell, q, weight)
            end
        else
            dst = cell_id_by_coord[_tree_topology_key(0, i_dst, j_dst)]
            dst_cell = topology.cells[dst]
            _scatter_route_packet_2d!(coarse_out, patch_out,
                                       coarse_in, patch_in,
                                       src_cell, dst_cell, q, route.weight)
        end
        return true
    end

    i_raw = src_cell.i + cx
    j_dst = src_cell.j + cy
    (i_raw < 1 || i_raw > 2 * nx) || return false
    1 <= j_dst <= 2 * ny || return false

    i_dst = _periodic_x_wrapped(i_raw, 2 * nx)
    if _inside_fine_patch(i_dst, j_dst, patch_in)
        dst = cell_id_by_coord[_tree_topology_key(1, i_dst, j_dst)]
        dst_cell = topology.cells[dst]
        _scatter_route_packet_2d!(coarse_out, patch_out,
                                   coarse_in, patch_in,
                                   src_cell, dst_cell, q, route.weight)
    else
        i_parent, j_parent = _coarse_parent_from_fine(i_dst, j_dst)
        dst = cell_id_by_coord[_tree_topology_key(0, i_parent, j_parent)]
        dst_cell = topology.cells[dst]
        _scatter_route_packet_2d!(coarse_out, patch_out,
                                   coarse_in, patch_in,
                                   src_cell, dst_cell, q, route.weight)
    end
    return true
end

function _stream_wall_y_boundary_route_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D,
        topology::ConservativeTreeTopology2D,
        route::ConservativeTreeRoute2D,
        ny::Int,
        u_south,
        u_north,
        rho_wall,
        volume_coarse,
        volume_fine;
        coarse_prolongation::Symbol=:flat)
    src_cell = topology.cells[route.src]
    q = route.q
    cy = d2q9_cy(q)
    cy == 0 && return false

    wall_u = u_south
    if src_cell.level == 0
        j_raw = src_cell.j + cy
        (j_raw < 1 || j_raw > ny) || return false
        wall_u = j_raw < 1 ? u_south : u_north
        if route.weight < 1.0
            q_dst = d2q9_opposite(q)
            handled = false
            @inbounds for iy in 1:2, ix in 1:2
                j_leaf = _fine_global_j(src_cell.j, iy) + cy
                (j_leaf < 1 || j_leaf > 2 * ny) || continue
                value = if coarse_prolongation == :limited_linear
                    _coarse_child_limited_linear_Fq_2d(
                        coarse_in, patch_in, src_cell.i, src_cell.j, q, ix, iy)
                else
                    0.25 * _cell_Fq_2d(coarse_in, patch_in, src_cell, q)
                end
                correction = _moving_wall_delta(
                    volume_coarse / 4, rho_wall, wall_u, q_dst)
                _add_cell_Fq_2d!(coarse_out, patch_out, src_cell, q_dst,
                                 value + correction)
                handled = true
            end
            return handled
        end
    else
        j_raw = src_cell.j + cy
        (j_raw < 1 || j_raw > 2 * ny) || return false
        wall_u = j_raw < 1 ? u_south : u_north
    end

    q_dst = d2q9_opposite(q)
    volume = src_cell.level == 0 ? volume_coarse : volume_fine
    packet = _cell_Fq_2d(coarse_in, patch_in, src_cell, q)
    correction = _moving_wall_delta(volume, rho_wall, wall_u, q_dst)
    _add_cell_Fq_2d!(coarse_out, patch_out, src_cell, q_dst,
                     route.weight * packet + correction)
    return true
end

