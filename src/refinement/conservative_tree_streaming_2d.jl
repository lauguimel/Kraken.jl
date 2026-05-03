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
        volume_fine=0.25)
    _check_composite_pair_layout(coarse_out, patch_out, coarse_in, patch_in)
    _check_route_stream_topology_layout(topology, coarse_in, patch_in)
    _check_route_stream_topology_layout(topology, coarse_out, patch_out)

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
        _scatter_route_packet_2d!(coarse_out, patch_out, coarse_in, patch_in,
                                   src_cell, dst_cell, route.q, route.weight)
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
