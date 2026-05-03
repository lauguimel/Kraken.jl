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

function _stream_periodic_x_boundary_route_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D,
        topology::ConservativeTreeTopology2D,
        cell_id_by_coord,
        route::ConservativeTreeRoute2D,
        nx::Int,
        ny::Int)
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
        if _inside_range(i_dst, j_dst, patch_in.parent_i_range, patch_in.parent_j_range)
            di = -cx
            dj = cy == 0 ? 0 : -cy
            specs = _coarse_to_fine_route_specs(cell_id_by_coord, i_dst, j_dst, di, dj)
            @inbounds for spec in specs
                dst, weight, _ = spec
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
        volume_fine)
    src_cell = topology.cells[route.src]
    q = route.q
    cy = d2q9_cy(q)
    cy == 0 && return false

    wall_u = u_south
    if src_cell.level == 0
        j_raw = src_cell.j + cy
        (j_raw < 1 || j_raw > ny) || return false
        wall_u = j_raw < 1 ? u_south : u_north
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

function _stream_composite_routes_F_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D,
        topology::ConservativeTreeTopology2D,
        boundary_policy::Symbol,
        clear::Bool;
        u_south=0,
        u_north=0,
        rho_wall=1,
        volume_coarse=1,
        volume_fine=0.25,
        is_solid=nothing)
    _check_composite_pair_layout(coarse_out, patch_out, coarse_in, patch_in)
    _check_route_stream_topology_layout(topology, coarse_in, patch_in)
    _check_route_stream_topology_layout(topology, coarse_out, patch_out)
    if is_solid !== nothing
        _check_route_solid_mask_layout(topology, coarse_in, patch_in, is_solid)
    end

    if clear
        coarse_out .= 0
        patch_out.fine_F .= 0
        patch_out.coarse_shadow_F .= 0
    end

    nx = size(coarse_in, 1)
    ny = size(coarse_in, 2)
    cell_id_by_coord = boundary_policy in (
        :periodic_x, :periodic_x_wall_y, :periodic_x_moving_wall_y) ?
        _cell_id_by_coord_2d(topology) : nothing

    @inbounds for route in topology.routes
        if route.kind == ROUTE_BOUNDARY
            boundary_policy == :skip && continue
            src_cell = topology.cells[route.src]
            is_solid !== nothing && _cell_is_solid_2d(src_cell, is_solid) && continue
            boundary_policy in (
                :periodic_x, :periodic_x_wall_y, :periodic_x_moving_wall_y) ||
                throw(ArgumentError("unsupported route boundary policy: $boundary_policy"))
            handled = false
            if boundary_policy in (:periodic_x_wall_y, :periodic_x_moving_wall_y)
                handled = _stream_wall_y_boundary_route_2d!(
                    coarse_out, patch_out, coarse_in, patch_in,
                    topology, route, ny, u_south, u_north, rho_wall,
                    volume_coarse, volume_fine)
            end
            handled || _stream_periodic_x_boundary_route_2d!(
                coarse_out, patch_out, coarse_in, patch_in,
                topology, cell_id_by_coord, route, nx, ny)
            continue
        end

        src_cell = topology.cells[route.src]
        dst_cell = topology.cells[route.dst]
        if is_solid === nothing
            _scatter_route_packet_2d!(coarse_out, patch_out, coarse_in, patch_in,
                                      src_cell, dst_cell, route.q, route.weight)
        else
            _scatter_solid_route_packet_2d!(
                coarse_out, patch_out, coarse_in, patch_in,
                src_cell, dst_cell, route.q, route.weight, is_solid)
        end
    end
    coalesce_patch_to_shadow_F_2d!(patch_out)
    return coarse_out, patch_out
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
    return _stream_composite_routes_F_2d!(coarse_out, patch_out,
                                          coarse_in, patch_in,
                                          topology, :skip, clear)
end

"""
    stream_composite_routes_periodic_x_F_2d!(coarse_out, patch_out,
                                             coarse_in, patch_in, topology;
                                             clear=true)

Scatter integrated D2Q9 populations along all interior routes and wrap
boundary packets that leave through the periodic x direction.

Packets leaving through y boundaries are still skipped. This keeps the
Milestone-1 boundary surface explicit: periodic x is native, wall and inlet/
outlet closures are added by later surgical patches.
"""
function stream_composite_routes_periodic_x_F_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D,
        topology::ConservativeTreeTopology2D;
        clear::Bool=true)
    return _stream_composite_routes_F_2d!(coarse_out, patch_out,
                                          coarse_in, patch_in,
                                          topology, :periodic_x, clear)
end

"""
    stream_composite_routes_periodic_x_wall_y_F_2d!(coarse_out, patch_out,
                                                    coarse_in, patch_in,
                                                    topology; clear=true)

Scatter integrated D2Q9 populations with periodic x wrapping and stationary
no-slip bounce-back for packets leaving through y boundaries.

This is still a transport-only primitive. Moving-wall and inlet/outlet
corrections belong to later boundary patches.
"""
function stream_composite_routes_periodic_x_wall_y_F_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D,
        topology::ConservativeTreeTopology2D;
        clear::Bool=true)
    return _stream_composite_routes_F_2d!(coarse_out, patch_out,
                                          coarse_in, patch_in,
                                          topology, :periodic_x_wall_y, clear)
end

"""
    stream_composite_routes_periodic_x_moving_wall_y_F_2d!(
        coarse_out, patch_out, coarse_in, patch_in, topology;
        u_south=0, u_north=0, rho_wall=1,
        volume_coarse=1, volume_fine=0.25, clear=true)

Scatter integrated D2Q9 populations with periodic x wrapping and moving-wall
bounce-back corrections on y boundaries.
"""
function stream_composite_routes_periodic_x_moving_wall_y_F_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D,
        topology::ConservativeTreeTopology2D;
        u_south=0,
        u_north=0,
        rho_wall=1,
        volume_coarse=1,
        volume_fine=0.25,
        clear::Bool=true)
    return _stream_composite_routes_F_2d!(coarse_out, patch_out,
                                          coarse_in, patch_in,
                                          topology, :periodic_x_moving_wall_y,
                                          clear;
                                          u_south=u_south,
                                          u_north=u_north,
                                          rho_wall=rho_wall,
                                          volume_coarse=volume_coarse,
                                          volume_fine=volume_fine)
end

function stream_composite_routes_periodic_x_wall_y_solid_F_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D,
        topology::ConservativeTreeTopology2D,
        is_solid::AbstractArray{Bool,2};
        clear::Bool=true)
    return _stream_composite_routes_F_2d!(coarse_out, patch_out,
                                          coarse_in, patch_in,
                                          topology, :periodic_x_wall_y, clear;
                                          is_solid=is_solid)
end

function collide_Guo_composite_solid_F_2d!(
        coarse_F::AbstractArray{<:Any,3},
        patch::ConservativeTreePatch2D,
        topology::ConservativeTreeTopology2D,
        is_solid::AbstractArray{Bool,2},
        volume_coarse,
        volume_fine,
        omega_coarse,
        omega_fine,
        Fx,
        Fy)
    _check_route_stream_topology_layout(topology, coarse_F, patch)
    _check_route_solid_mask_layout(topology, coarse_F, patch, is_solid)

    @inbounds for id in topology.active_cells
        cell = topology.cells[id]
        _cell_is_solid_2d(cell, is_solid) && continue
        volume = cell.level == 0 ? volume_coarse : volume_fine
        omega = cell.level == 0 ? omega_coarse : omega_fine
        collide_Guo_integrated_D2Q9!(_cell_view_2d(coarse_F, patch, cell),
                                     volume, omega, Fx, Fy)
    end
    coalesce_patch_to_shadow_F_2d!(patch)
    return coarse_F, patch
end

function regrid_conservative_tree_patch_F_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D)
    size(coarse_out) == size(coarse_in) ||
        throw(ArgumentError("coarse_out and coarse_in must have the same size"))
    _check_composite_coarse_layout(coarse_in, patch_in)
    _check_composite_coarse_layout(coarse_out, patch_out)

    leaf = similar(coarse_in, 2 * size(coarse_in, 1), 2 * size(coarse_in, 2), 9)
    composite_to_leaf_F_2d!(leaf, coarse_in, patch_in)
    leaf_to_composite_F_2d!(coarse_out, patch_out, leaf)
    return coarse_out, patch_out
end

function _source_parent_leaf_block_F_2d!(
        leaf_block::AbstractArray{<:Any,3},
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D,
        I::Int,
        J::Int)
    _check_child_block_2d(leaf_block, "leaf_block")
    if _inside_range(I, J, patch_in.parent_i_range, patch_in.parent_j_range)
        il, jl = _patch_local_parent_index(patch_in, I, J)
        leaf_block .= _child_block_view(patch_in.fine_F, il, jl)
    else
        _explode_limited_linear_composite_F_2d!(leaf_block, coarse_in, patch_in, I, J)
    end
    return leaf_block
end

function regrid_conservative_tree_patch_direct_F_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D)
    size(coarse_out) == size(coarse_in) ||
        throw(ArgumentError("coarse_out and coarse_in must have the same size"))
    _check_composite_coarse_layout(coarse_in, patch_in)
    _check_composite_coarse_layout(coarse_out, patch_out)
    coalesce_patch_to_shadow_F_2d!(patch_in)

    coarse_out .= 0
    patch_out.fine_F .= 0
    patch_out.coarse_shadow_F .= 0
    leaf_block = zeros(promote_type(eltype(coarse_in), eltype(patch_in.fine_F)),
                       2, 2, 9)

    @inbounds for J in axes(coarse_in, 2), I in axes(coarse_in, 1)
        _source_parent_leaf_block_F_2d!(leaf_block, coarse_in, patch_in, I, J)
        if _inside_range(I, J, patch_out.parent_i_range, patch_out.parent_j_range)
            il, jl = _patch_local_parent_index(patch_out, I, J)
            _child_block_view(patch_out.fine_F, il, jl) .= leaf_block
        else
            coalesce_F_2d!(@view(coarse_out[I, J, :]), leaf_block)
        end
    end

    coalesce_patch_to_shadow_F_2d!(patch_out)
    return coarse_out, patch_out
end

function conservative_tree_solid_mask_patch_range_2d(
        is_solid::AbstractArray{Bool,2};
        pad::Int=1)
    nx_leaf = size(is_solid, 1)
    ny_leaf = size(is_solid, 2)
    iseven(nx_leaf) && iseven(ny_leaf) ||
        throw(ArgumentError("is_solid dimensions must be even leaf-grid sizes"))
    pad >= 0 || throw(ArgumentError("pad must be nonnegative"))
    any(is_solid) || throw(ArgumentError("is_solid contains no solid cells"))

    i_min_leaf = typemax(Int)
    i_max_leaf = typemin(Int)
    j_min_leaf = typemax(Int)
    j_max_leaf = typemin(Int)
    @inbounds for j in axes(is_solid, 2), i in axes(is_solid, 1)
        if is_solid[i, j]
            i_min_leaf = min(i_min_leaf, i)
            i_max_leaf = max(i_max_leaf, i)
            j_min_leaf = min(j_min_leaf, j)
            j_max_leaf = max(j_max_leaf, j)
        end
    end

    nx = nx_leaf >>> 1
    ny = ny_leaf >>> 1
    i_min = max(1, cld(i_min_leaf, 2) - pad)
    i_max = min(nx, cld(i_max_leaf, 2) + pad)
    j_min = max(1, cld(j_min_leaf, 2) - pad)
    j_max = min(ny, cld(j_max_leaf, 2) + pad)
    return (i_range=i_min:i_max, j_range=j_min:j_max)
end

function adapt_conservative_tree_patch_to_solid_mask_2d(
        coarse_F::AbstractArray{T,3},
        patch::ConservativeTreePatch2D{T},
        is_solid::AbstractArray{Bool,2};
        pad::Int=1) where T
    _check_composite_coarse_layout(coarse_F, patch)
    size(is_solid) == (2 * size(coarse_F, 1), 2 * size(coarse_F, 2)) ||
        throw(ArgumentError("is_solid must have size (2*Nx, 2*Ny)"))
    ranges = conservative_tree_solid_mask_patch_range_2d(is_solid; pad=pad)
    patch_out = create_conservative_tree_patch_2d(
        ranges.i_range, ranges.j_range; T=T)
    coarse_out = similar(coarse_F)
    regrid_conservative_tree_patch_direct_F_2d!(coarse_out, patch_out, coarse_F, patch)
    return (coarse_F=coarse_out, patch=patch_out)
end

function vertical_facing_step_solid_mask_leaf_2d(
        Nx::Int,
        Ny::Int,
        step_i_range::AbstractUnitRange{<:Integer},
        step_height::Int)
    Nx > 0 || throw(ArgumentError("Nx must be positive"))
    Ny > 0 || throw(ArgumentError("Ny must be positive"))
    isempty(step_i_range) && throw(ArgumentError("step_i_range must be nonempty"))
    first(step_i_range) >= 1 && last(step_i_range) <= Nx ||
        throw(ArgumentError("step_i_range must be inside 1:Nx"))
    1 <= step_height < Ny ||
        throw(ArgumentError("step_height must be inside 1:Ny-1"))

    mask = falses(Nx, Ny)
    @inbounds for j in 1:step_height, i in Int(first(step_i_range)):Int(last(step_i_range))
        mask[i, j] = true
    end
    return mask
end

struct ConservativeTreeAdaptiveRun2D{T}
    flow::Symbol
    coarse_F::Array{T,3}
    patch::ConservativeTreePatch2D{T}
    patch_history::Vector{Tuple{UnitRange{Int},UnitRange{Int}}}
    ux_mean::T
    mass_initial::T
    mass_final::T
    mass_drift::T
    steps::Int
    regrid_every::Int
    regrid_count::Int
end

function run_conservative_tree_couette_route_native_2d(;
        Nx::Int=18,
        Ny::Int=14,
        patch_i_range::UnitRange{Int}=7:12,
        patch_j_range::UnitRange{Int}=5:10,
        U=0.04,
        omega=1.0,
        rho=1.0,
        steps::Int=3000,
        T::Type{<:Real}=Float64)
    U = T(U)
    omega = T(omega)
    rho = T(rho)
    volume_coarse = one(T)
    volume_fine = T(0.25)

    coarse = zeros(T, Nx, Ny, 9)
    coarse_next = similar(coarse)
    patch = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    topology = create_conservative_tree_topology_2d(Nx, Ny, patch)

    fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, rho, zero(T), zero(T))
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, rho, zero(T), zero(T))
    mass_initial = active_mass_F(coarse, patch)

    for _ in 1:steps
        collide_BGK_composite_F_2d!(coarse, patch, volume_coarse, volume_fine,
                                    omega, omega)
        stream_composite_routes_periodic_x_moving_wall_y_F_2d!(
            coarse_next, patch_next, coarse, patch, topology;
            u_north=U, rho_wall=rho,
            volume_coarse=volume_coarse, volume_fine=volume_fine)
        coarse, coarse_next = coarse_next, coarse
        patch, patch_next = patch_next, patch
    end

    profile = composite_leaf_mean_ux_profile(coarse, patch; volume_leaf=volume_fine)
    analytic = couette_analytic_profile_2d(length(profile), U)
    l2, linf = _profile_errors(profile, analytic)
    mass_final = active_mass_F(coarse, patch)

    return ConservativeTreeMacroFlow2D{T}(
        :couette_route_native, coarse, patch, profile, analytic, l2, linf,
        mass_initial, mass_final, mass_final - mass_initial, steps)
end

function run_conservative_tree_poiseuille_route_native_2d(;
        Nx::Int=18,
        Ny::Int=14,
        patch_i_range::UnitRange{Int}=7:12,
        patch_j_range::UnitRange{Int}=5:10,
        Fx=5e-5,
        omega=1.0,
        rho=1.0,
        steps::Int=3000,
        T::Type{<:Real}=Float64)
    Fx = T(Fx)
    omega = T(omega)
    rho = T(rho)
    volume_coarse = one(T)
    volume_fine = T(0.25)

    coarse = zeros(T, Nx, Ny, 9)
    coarse_next = similar(coarse)
    patch = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    topology = create_conservative_tree_topology_2d(Nx, Ny, patch)

    fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, rho, zero(T), zero(T))
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, rho, zero(T), zero(T))
    mass_initial = active_mass_F(coarse, patch)

    for _ in 1:steps
        collide_Guo_composite_F_2d!(coarse, patch, volume_coarse, volume_fine,
                                    omega, omega, Fx, zero(T))
        stream_composite_routes_periodic_x_wall_y_F_2d!(
            coarse_next, patch_next, coarse, patch, topology)
        coarse, coarse_next = coarse_next, coarse
        patch, patch_next = patch_next, patch
    end

    profile = composite_leaf_mean_ux_profile(coarse, patch;
                                             volume_leaf=volume_fine,
                                             force_x=Fx)
    analytic = poiseuille_analytic_profile_2d(length(profile), Fx, omega; rho=rho)
    l2, linf = _profile_errors(profile, analytic)
    mass_final = active_mass_F(coarse, patch)

    return ConservativeTreeMacroFlow2D{T}(
        :poiseuille_route_native, coarse, patch, profile, analytic, l2, linf,
        mass_initial, mass_final, mass_final - mass_initial, steps)
end

function run_conservative_tree_poiseuille_adaptive_route_native_2d(;
        Nx::Int=18,
        Ny::Int=14,
        patch_schedule::Tuple=((7:12, 5:10), (6:11, 4:9), (8:13, 5:10)),
        regrid_every::Int=80,
        Fx=5e-5,
        omega=1.0,
        rho=1.0,
        steps::Int=320,
        T::Type{<:Real}=Float64)
    steps > 0 || throw(ArgumentError("steps must be positive"))
    regrid_every > 0 || throw(ArgumentError("regrid_every must be positive"))
    isempty(patch_schedule) && throw(ArgumentError("patch_schedule must be nonempty"))

    Fx = T(Fx)
    omega = T(omega)
    rho = T(rho)
    volume_coarse = one(T)
    volume_fine = T(0.25)

    first_ranges = patch_schedule[1]
    patch = create_conservative_tree_patch_2d(first_ranges[1], first_ranges[2]; T=T)
    patch_next = create_conservative_tree_patch_2d(first_ranges[1], first_ranges[2]; T=T)
    topology = create_conservative_tree_topology_2d(Nx, Ny, patch)
    coarse = zeros(T, Nx, Ny, 9)
    coarse_next = similar(coarse)
    fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, rho, zero(T), zero(T))
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, rho, zero(T), zero(T))
    mass_initial = active_mass_F(coarse, patch)

    patch_history = Tuple{UnitRange{Int},UnitRange{Int}}[
        (patch.parent_i_range, patch.parent_j_range)
    ]
    regrid_count = 0

    for step in 1:steps
        collide_Guo_composite_F_2d!(coarse, patch, volume_coarse, volume_fine,
                                    omega, omega, Fx, zero(T))
        stream_composite_routes_periodic_x_wall_y_F_2d!(
            coarse_next, patch_next, coarse, patch, topology)
        coarse, coarse_next = coarse_next, coarse
        patch, patch_next = patch_next, patch

        if step < steps && step % regrid_every == 0
            regrid_count += 1
            ranges = patch_schedule[mod1(regrid_count + 1, length(patch_schedule))]
            new_patch = create_conservative_tree_patch_2d(ranges[1], ranges[2]; T=T)
            new_coarse = similar(coarse)
            regrid_conservative_tree_patch_direct_F_2d!(
                new_coarse, new_patch, coarse, patch)
            coarse = new_coarse
            patch = new_patch
            patch_next = create_conservative_tree_patch_2d(ranges[1], ranges[2]; T=T)
            coarse_next = similar(coarse)
            topology = create_conservative_tree_topology_2d(Nx, Ny, patch)
            push!(patch_history, (patch.parent_i_range, patch.parent_j_range))
        end
    end

    mass_final = active_mass_F(coarse, patch)
    ux_mean = sum(composite_leaf_mean_ux_profile(coarse, patch;
                                                 volume_leaf=volume_fine,
                                                 force_x=Fx)) / T(2 * Ny)

    return ConservativeTreeAdaptiveRun2D{T}(
        :poiseuille_adaptive_route_native, coarse, patch, patch_history,
        ux_mean, mass_initial, mass_final, mass_final - mass_initial,
        steps, regrid_every, regrid_count)
end

function validate_conservative_tree_route_native_phase_p_2d(;
        steps::Int=1000,
        T::Type{<:Real}=Float64)
    couette_route = run_conservative_tree_couette_route_native_2d(
        ; steps=steps, T=T)
    couette_oracle = run_conservative_tree_couette_macroflow_2d(
        ; Nx=size(couette_route.coarse_F, 1),
        Ny=size(couette_route.coarse_F, 2),
        patch_i_range=couette_route.patch.parent_i_range,
        patch_j_range=couette_route.patch.parent_j_range,
        steps=steps, T=T)
    couette_l2, couette_linf = _profile_errors(
        couette_route.ux_profile, couette_oracle.ux_profile)

    poiseuille_route = run_conservative_tree_poiseuille_route_native_2d(
        ; steps=steps, T=T)
    poiseuille_oracle = run_conservative_tree_poiseuille_macroflow_2d(
        ; Nx=size(poiseuille_route.coarse_F, 1),
        Ny=size(poiseuille_route.coarse_F, 2),
        patch_i_range=poiseuille_route.patch.parent_i_range,
        patch_j_range=poiseuille_route.patch.parent_j_range,
        steps=steps, T=T)
    poiseuille_l2, poiseuille_linf = _profile_errors(
        poiseuille_route.ux_profile, poiseuille_oracle.ux_profile)

    return (
        couette=(
            route=couette_route,
            oracle=couette_oracle,
            l2_delta=couette_l2,
            linf_delta=couette_linf,
        ),
        poiseuille=(
            route=poiseuille_route,
            oracle=poiseuille_oracle,
            l2_delta=poiseuille_l2,
            linf_delta=poiseuille_linf,
        ),
    )
end

function run_conservative_tree_square_obstacle_route_native_2d(;
        Nx::Int=24,
        Ny::Int=14,
        patch_i_range::UnitRange{Int}=8:17,
        patch_j_range::UnitRange{Int}=4:11,
        obstacle_i_range::UnitRange{Int}=22:27,
        obstacle_j_range::UnitRange{Int}=12:17,
        Fx=2e-5,
        Fy=0.0,
        omega=1.0,
        rho=1.0,
        steps::Int=1200,
        T::Type{<:Real}=Float64)
    Fx = T(Fx)
    Fy = T(Fy)
    omega = T(omega)
    rho = T(rho)
    volume_coarse = one(T)
    volume_fine = T(0.25)

    is_solid = square_solid_mask_leaf_2d(2 * Nx, 2 * Ny,
                                         obstacle_i_range, obstacle_j_range)
    coarse = zeros(T, Nx, Ny, 9)
    coarse_next = similar(coarse)
    patch = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    topology = create_conservative_tree_topology_2d(Nx, Ny, patch)
    leaf = zeros(T, 2 * Nx, 2 * Ny, 9)

    _check_route_solid_mask_layout(topology, coarse, patch, is_solid)

    fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, rho, zero(T), zero(T))
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, rho, zero(T), zero(T))
    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass_initial = _leaf_fluid_mass_F(leaf, is_solid)

    for _ in 1:steps
        collide_Guo_composite_solid_F_2d!(
            coarse, patch, topology, is_solid,
            volume_coarse, volume_fine, omega, omega, Fx, Fy)
        stream_composite_routes_periodic_x_wall_y_solid_F_2d!(
            coarse_next, patch_next, coarse, patch, topology, is_solid)
        coarse, coarse_next = coarse_next, coarse
        patch, patch_next = patch_next, patch
    end

    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass_final = _leaf_fluid_mass_F(leaf, is_solid)
    ux_mean, uy_mean = _leaf_fluid_mean_velocity_F(
        leaf, is_solid; volume=volume_fine, force_x=Fx, force_y=Fy)

    return ConservativeTreeSolidFlowResult2D{T}(
        :square_obstacle_route_native, coarse, patch, is_solid, ux_mean, uy_mean,
        mass_initial, mass_final, mass_final - mass_initial, steps)
end

function run_conservative_tree_vfs_route_native_2d(;
        Nx::Int=28,
        Ny::Int=14,
        patch_i_range::UnitRange{Int}=5:12,
        patch_j_range::UnitRange{Int}=1:8,
        step_i_range::UnitRange{Int}=14:19,
        step_height_leaf::Int=8,
        Fx=2e-5,
        Fy=0.0,
        omega=1.0,
        rho=1.0,
        steps::Int=900,
        T::Type{<:Real}=Float64)
    Fx = T(Fx)
    Fy = T(Fy)
    omega = T(omega)
    rho = T(rho)
    volume_coarse = one(T)
    volume_fine = T(0.25)

    is_solid = vertical_facing_step_solid_mask_leaf_2d(
        2 * Nx, 2 * Ny, step_i_range, step_height_leaf)
    coarse = zeros(T, Nx, Ny, 9)
    coarse_next = similar(coarse)
    patch = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    topology = create_conservative_tree_topology_2d(Nx, Ny, patch)
    leaf = zeros(T, 2 * Nx, 2 * Ny, 9)

    _check_route_solid_mask_layout(topology, coarse, patch, is_solid)

    fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, rho, zero(T), zero(T))
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, rho, zero(T), zero(T))
    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass_initial = _leaf_fluid_mass_F(leaf, is_solid)

    for _ in 1:steps
        collide_Guo_composite_solid_F_2d!(
            coarse, patch, topology, is_solid,
            volume_coarse, volume_fine, omega, omega, Fx, Fy)
        stream_composite_routes_periodic_x_wall_y_solid_F_2d!(
            coarse_next, patch_next, coarse, patch, topology, is_solid)
        coarse, coarse_next = coarse_next, coarse
        patch, patch_next = patch_next, patch
    end

    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass_final = _leaf_fluid_mass_F(leaf, is_solid)
    ux_mean, uy_mean = _leaf_fluid_mean_velocity_F(
        leaf, is_solid; volume=volume_fine, force_x=Fx, force_y=Fy)

    return ConservativeTreeSolidFlowResult2D{T}(
        :vfs_route_native, coarse, patch, is_solid, ux_mean, uy_mean,
        mass_initial, mass_final, mass_final - mass_initial, steps)
end
