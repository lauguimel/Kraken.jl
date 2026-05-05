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

@inline function _cell_view_3d(coarse_F::AbstractArray{<:Any,4},
                               patch::ConservativeTreePatch3D,
                               cell::ConservativeTreeCell3D)
    if cell.level == 0
        return @view coarse_F[cell.i, cell.j, cell.k, :]
    else
        i0 = 2 * first(patch.parent_i_range) - 1
        j0 = 2 * first(patch.parent_j_range) - 1
        k0 = 2 * first(patch.parent_k_range) - 1
        return @view patch.fine_F[cell.i - i0 + 1,
                                  cell.j - j0 + 1,
                                  cell.k - k0 + 1, :]
    end
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

function _check_route_solid_mask_layout_3d(topology::ConservativeTreeTopology3D,
                                           coarse_F::AbstractArray{<:Any,4},
                                           patch::ConservativeTreePatch3D,
                                           is_solid::AbstractArray{Bool,3})
    size(is_solid) == (2 * size(coarse_F, 1),
                       2 * size(coarse_F, 2),
                       2 * size(coarse_F, 3)) ||
        throw(ArgumentError("is_solid must have size (2*Nx, 2*Ny, 2*Nz)"))

    @inbounds for id in topology.active_cells
        cell = topology.cells[id]
        cell.level == 1 && continue

        i0 = 2 * cell.i - 1
        j0 = 2 * cell.j - 1
        k0 = 2 * cell.k - 1
        s000 = is_solid[i0, j0, k0]
        for dk in 0:1, dj in 0:1, di in 0:1
            is_solid[i0 + di, j0 + dj, k0 + dk] == s000 ||
                throw(ArgumentError("active coarse cells cannot be partially solid"))
        end
    end
    return nothing
end

@inline function _cell_is_solid_3d(cell::ConservativeTreeCell3D,
                                   is_solid::AbstractArray{Bool,3})
    if cell.level == 0
        return is_solid[2 * cell.i - 1, 2 * cell.j - 1, 2 * cell.k - 1]
    end
    return is_solid[cell.i, cell.j, cell.k]
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

@inline function _scatter_solid_route_packet_3d!(
        coarse_out::AbstractArray{<:Any,4},
        patch_out::ConservativeTreePatch3D,
        coarse_in::AbstractArray{<:Any,4},
        patch_in::ConservativeTreePatch3D,
        src_cell::ConservativeTreeCell3D,
        dst_cell::ConservativeTreeCell3D,
        q::Int,
        weight,
        is_solid::AbstractArray{Bool,3})
    _cell_is_solid_3d(src_cell, is_solid) && return nothing

    packet = _cell_Fq_3d(coarse_in, patch_in, src_cell, q)
    if _cell_is_solid_3d(dst_cell, is_solid)
        _add_cell_Fq_3d!(coarse_out, patch_out, src_cell,
                         d3q19_opposite(q), weight * packet)
    else
        _add_cell_Fq_3d!(coarse_out, patch_out, dst_cell, q, weight * packet)
    end
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
        nz::Int,
        is_solid)
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
                if is_solid === nothing
                    _scatter_route_packet_3d!(coarse_out, patch_out,
                                               coarse_in, patch_in,
                                               src_cell, dst_cell, q, weight)
                else
                    _scatter_solid_route_packet_3d!(
                        coarse_out, patch_out, coarse_in, patch_in,
                        src_cell, dst_cell, q, weight, is_solid)
                end
            end
        else
            dst = cell_id_by_coord[_tree_topology_key_3d(0, i_dst, j_dst, k_dst)]
            dst_cell = topology.cells[dst]
            if is_solid === nothing
                _scatter_route_packet_3d!(coarse_out, patch_out,
                                          coarse_in, patch_in,
                                          src_cell, dst_cell, q, route.weight)
            else
                _scatter_solid_route_packet_3d!(
                    coarse_out, patch_out, coarse_in, patch_in,
                    src_cell, dst_cell, q, route.weight, is_solid)
            end
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
        if is_solid === nothing
            _scatter_route_packet_3d!(coarse_out, patch_out,
                                      coarse_in, patch_in,
                                      src_cell, dst_cell, q, route.weight)
        else
            _scatter_solid_route_packet_3d!(
                coarse_out, patch_out, coarse_in, patch_in,
                src_cell, dst_cell, q, route.weight, is_solid)
        end
    else
        I_dst, J_dst, K_dst = _coarse_parent_from_fine_3d(i_dst, j_dst, k_dst)
        dst = cell_id_by_coord[_tree_topology_key_3d(0, I_dst, J_dst, K_dst)]
        dst_cell = topology.cells[dst]
        if is_solid === nothing
            _scatter_route_packet_3d!(coarse_out, patch_out,
                                      coarse_in, patch_in,
                                      src_cell, dst_cell, q, route.weight)
        else
            _scatter_solid_route_packet_3d!(
                coarse_out, patch_out, coarse_in, patch_in,
                src_cell, dst_cell, q, route.weight, is_solid)
        end
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
        clear::Bool;
        is_solid=nothing)
    _check_composite_pair_layout_3d(coarse_out, patch_out, coarse_in, patch_in)
    _check_route_stream_topology_layout_3d(topology, coarse_in, patch_in)
    _check_route_stream_topology_layout_3d(topology, coarse_out, patch_out)
    if is_solid !== nothing
        _check_route_solid_mask_layout_3d(topology, coarse_in, patch_in, is_solid)
    end

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
            src_cell = topology.cells[route.src]
            is_solid !== nothing && _cell_is_solid_3d(src_cell, is_solid) && continue
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
                cell_id_by_coord, route, nx, ny, nz, is_solid)
            continue
        end

        src_cell = topology.cells[route.src]
        dst_cell = topology.cells[route.dst]
        if is_solid === nothing
            _scatter_route_packet_3d!(coarse_out, patch_out, coarse_in, patch_in,
                                      src_cell, dst_cell, route.q, route.weight)
        else
            _scatter_solid_route_packet_3d!(
                coarse_out, patch_out, coarse_in, patch_in,
                src_cell, dst_cell, route.q, route.weight, is_solid)
        end
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

function stream_composite_routes_periodic_x_wall_yz_solid_F_3d!(
        coarse_out::AbstractArray{<:Any,4},
        patch_out::ConservativeTreePatch3D,
        coarse_in::AbstractArray{<:Any,4},
        patch_in::ConservativeTreePatch3D,
        topology::ConservativeTreeTopology3D,
        is_solid::AbstractArray{Bool,3};
        clear::Bool=true)
    return _stream_composite_routes_F_3d!(coarse_out, patch_out,
                                          coarse_in, patch_in,
                                          topology, :periodic_x_wall_yz, clear;
                                          is_solid=is_solid)
end

function collide_Guo_composite_solid_F_3d!(
        coarse_F::AbstractArray{<:Any,4},
        patch::ConservativeTreePatch3D,
        topology::ConservativeTreeTopology3D,
        is_solid::AbstractArray{Bool,3},
        volume_coarse,
        volume_fine,
        omega_coarse,
        omega_fine,
        Fx,
        Fy,
        Fz)
    _check_route_stream_topology_layout_3d(topology, coarse_F, patch)
    _check_route_solid_mask_layout_3d(topology, coarse_F, patch, is_solid)

    @inbounds for id in topology.active_cells
        cell = topology.cells[id]
        _cell_is_solid_3d(cell, is_solid) && continue
        volume = cell.level == 0 ? volume_coarse : volume_fine
        omega = cell.level == 0 ? omega_coarse : omega_fine
        collide_Guo_integrated_D3Q19!(
            _cell_view_3d(coarse_F, patch, cell),
            volume, omega, Fx, Fy, Fz)
    end
    coalesce_patch_to_shadow_F_3d!(patch)
    return coarse_F, patch
end

"""
Result bundle for a fixed-patch route-native 3D AMR macro-flow canary.
"""
struct ConservativeTreeMacroFlow3D{T}
    flow::Symbol
    coarse_F::Array{T,4}
    patch::ConservativeTreePatch3D{T}
    ux_profile::Matrix{T}
    oracle_ux_profile::Matrix{T}
    l2_error::T
    linf_error::T
    ux_mean::T
    uy_mean::T
    uz_mean::T
    mass_initial::T
    mass_final::T
    mass_drift::T
    relative_mass_drift::T
    steps::Int
end

function _fill_rest_composite_F_3d!(coarse_F::AbstractArray{T,4},
                                    patch::ConservativeTreePatch3D{T},
                                    rho::T) where {T}
    _check_composite_coarse_layout_3d(coarse_F, patch)
    z = zero(T)
    @inbounds for k in axes(coarse_F, 3), j in axes(coarse_F, 2), i in axes(coarse_F, 1)
        _inside_range_3d(i, j, k,
                         patch.parent_i_range,
                         patch.parent_j_range,
                         patch.parent_k_range) && continue
        fill_equilibrium_integrated_D3Q19!(
            @view(coarse_F[i, j, k, :]), one(T), rho, z, z, z)
    end
    fill_equilibrium_integrated_D3Q19!(
        patch.fine_F, one(T) / T(8), rho, z, z, z)
    coalesce_patch_to_shadow_F_3d!(patch)
    return coarse_F, patch
end

function _mean_velocity_with_force_3d(coarse_F::AbstractArray{T,4},
                                      patch::ConservativeTreePatch3D{T},
                                      Fx::T,
                                      Fy::T,
                                      Fz::T) where {T}
    m, mx, my, mz = active_moments_F_3d(coarse_F, patch)
    active_vol = T(size(coarse_F, 1) * size(coarse_F, 2) * size(coarse_F, 3))
    half_force = T(0.5) * active_vol
    return (mx + half_force * Fx) / m,
           (my + half_force * Fy) / m,
           (mz + half_force * Fz) / m
end

function _profile_errors_3d(profile::AbstractMatrix, reference::AbstractMatrix)
    size(profile) == size(reference) ||
        throw(ArgumentError("profile and reference must have the same size"))
    T = promote_type(eltype(profile), eltype(reference))
    n = length(profile)
    l2 = sqrt(sum((T(profile[i]) - T(reference[i]))^2 for i in eachindex(profile)) /
              T(n))
    linf = maximum(abs(T(profile[i]) - T(reference[i])) for i in eachindex(profile))
    return l2, linf
end

function _leaf_mean_ux_yz_profile_3d(F::AbstractArray{T,4};
                                     volume::T,
                                     force_x::T=zero(T)) where {T}
    size(F, 4) == 19 ||
        throw(ArgumentError("F must have 19 D3Q19 populations in dimension 4"))
    profile = zeros(T, size(F, 2), size(F, 3))
    @inbounds for k in axes(F, 3), j in axes(F, 2)
        ux_sum = zero(T)
        for i in axes(F, 1)
            cell = @view F[i, j, k, :]
            rho = mass_F_3d(cell) / volume
            mx = momentum_F_3d(cell)[1]
            ux_sum += (mx / volume + force_x / T(2)) / rho
        end
        profile[j, k] = ux_sum / T(size(F, 1))
    end
    return profile
end

function composite_leaf_mean_ux_yz_profile_3d(coarse_F::AbstractArray{T,4},
                                              patch::ConservativeTreePatch3D{T};
                                              volume_leaf::T=T(0.125),
                                              force_x::T=zero(T)) where {T}
    leaf = zeros(T, 2 * size(coarse_F, 1), 2 * size(coarse_F, 2),
                 2 * size(coarse_F, 3), 19)
    composite_to_leaf_F_3d!(leaf, coarse_F, patch)
    return _leaf_mean_ux_yz_profile_3d(leaf;
                                       volume=volume_leaf,
                                       force_x=force_x)
end

function _stream_leaf_periodic_x_wall_yz_F_3d!(out::AbstractArray{T,4},
                                               F::AbstractArray{T,4}) where {T}
    size(out) == size(F) ||
        throw(ArgumentError("out and F must have the same size"))
    size(F, 4) == 19 ||
        throw(ArgumentError("F must have 19 D3Q19 populations in dimension 4"))
    out .= 0
    nx, ny, nz = size(F, 1), size(F, 2), size(F, 3)
    @inbounds for k in axes(F, 3), j in axes(F, 2), i in axes(F, 1), q in 1:19
        i_dst = i + d3q19_cx(q)
        j_dst = j + d3q19_cy(q)
        k_dst = k + d3q19_cz(q)
        if j_dst < 1 || j_dst > ny || k_dst < 1 || k_dst > nz
            out[i, j, k, d3q19_opposite(q)] += F[i, j, k, q]
        else
            i_wrap = _periodic_x_wrapped_3d(i_dst, nx)
            out[i_wrap, j_dst, k_dst, q] += F[i, j, k, q]
        end
    end
    return out
end

function _run_leaf_poiseuille_oracle_3d(;
        Nx::Int,
        Ny::Int,
        Nz::Int,
        rho,
        omega,
        Fx,
        Fy,
        Fz,
        steps::Int,
        T::Type{<:Real})
    F = zeros(T, 2 * Nx, 2 * Ny, 2 * Nz, 19)
    F_next = similar(F)
    volume = one(T) / T(8)
    fill_equilibrium_integrated_D3Q19!(
        F, volume, T(rho), zero(T), zero(T), zero(T))

    @inbounds for _ in 1:steps
        collide_Guo_integrated_D3Q19!(
            F, volume, T(omega), T(Fx), T(Fy), T(Fz))
        _stream_leaf_periodic_x_wall_yz_F_3d!(F_next, F)
        F, F_next = F_next, F
    end

    return F
end

function _conservative_tree_channel_patch_ranges_3d(
        Nx::Int,
        Ny::Int,
        Nz::Int,
        patch_i_range::AbstractUnitRange{<:Integer},
        patch_j_range::AbstractUnitRange{<:Integer},
        patch_k_range::AbstractUnitRange{<:Integer},
        patch_strategy::Symbol)
    if patch_strategy == :compact
        return Int(first(patch_i_range)):Int(last(patch_i_range)),
               Int(first(patch_j_range)):Int(last(patch_j_range)),
               Int(first(patch_k_range)):Int(last(patch_k_range))
    elseif patch_strategy == :cross_section_buffered
        Nx >= 4 || throw(ArgumentError("Nx must be >= 4 for :cross_section_buffered"))
        margin_i = Nx >= 10 ? 2 : 1
        return (1 + margin_i):(Nx - margin_i), 1:Ny, 1:Nz
    elseif patch_strategy == :full
        return 1:Nx, 1:Ny, 1:Nz
    end
    throw(ArgumentError("unsupported 3D channel patch_strategy: $patch_strategy"))
end

"""
    run_conservative_tree_poiseuille_route_native_3d(; kwargs...)

Run a small fixed-patch D3Q19 route-native 3D channel canary. The domain is
periodic in x, bounce-back walled in y/z, and driven by a uniform Guo force.

This is the publication-D 3D smoke gate: it validates conservative fixed-patch
transport plus collision in 3D before any sphere or open-boundary claim.
"""
function run_conservative_tree_poiseuille_route_native_3d(;
        Nx::Int=8,
        Ny::Int=8,
        Nz::Int=6,
        patch_i_range::AbstractUnitRange{<:Integer}=3:6,
        patch_j_range::AbstractUnitRange{<:Integer}=3:6,
        patch_k_range::AbstractUnitRange{<:Integer}=2:5,
        patch_strategy::Symbol=:compact,
        rho=1.0,
        omega=1.0,
        Fx=2.0e-5,
        Fy=0.0,
        Fz=0.0,
        steps::Int=120,
        T::Type{<:Real}=Float64)
    isconcretetype(T) || throw(ArgumentError("T must be a concrete Real type"))
    Nx > 2 || throw(ArgumentError("Nx must be > 2"))
    Ny > 2 || throw(ArgumentError("Ny must be > 2"))
    Nz > 2 || throw(ArgumentError("Nz must be > 2"))
    steps >= 0 || throw(ArgumentError("steps must be nonnegative"))

    coarse = zeros(T, Nx, Ny, Nz, 19)
    coarse_next = similar(coarse)
    patch_i_range, patch_j_range, patch_k_range =
        _conservative_tree_channel_patch_ranges_3d(
            Nx, Ny, Nz,
            patch_i_range, patch_j_range, patch_k_range,
            patch_strategy)
    patch = create_conservative_tree_patch_3d(
        patch_i_range, patch_j_range, patch_k_range; T=T)
    patch_next = create_conservative_tree_patch_3d(
        patch_i_range, patch_j_range, patch_k_range; T=T)
    topology = create_conservative_tree_topology_3d(Nx, Ny, Nz, patch)

    rho_T = T(rho)
    omega_T = T(omega)
    Fx_T = T(Fx)
    Fy_T = T(Fy)
    Fz_T = T(Fz)
    _fill_rest_composite_F_3d!(coarse, patch, rho_T)
    mass0 = T(active_mass_F_3d(coarse, patch))

    @inbounds for _ in 1:steps
        collide_Guo_composite_F_3d!(
            coarse, patch, one(T), one(T) / T(8), omega_T, omega_T,
            Fx_T, Fy_T, Fz_T)
        stream_composite_routes_periodic_x_wall_yz_F_3d!(
            coarse_next, patch_next, coarse, patch, topology)
        coarse, coarse_next = coarse_next, coarse
        patch, patch_next = patch_next, patch
    end

    mass1 = T(active_mass_F_3d(coarse, patch))
    drift = mass1 - mass0
    ux_mean, uy_mean, uz_mean =
        _mean_velocity_with_force_3d(coarse, patch, Fx_T, Fy_T, Fz_T)
    profile = composite_leaf_mean_ux_yz_profile_3d(coarse, patch;
                                                   volume_leaf=one(T) / T(8),
                                                   force_x=Fx_T)
    oracle = _run_leaf_poiseuille_oracle_3d(;
        Nx=Nx, Ny=Ny, Nz=Nz, rho=rho_T, omega=omega_T,
        Fx=Fx_T, Fy=Fy_T, Fz=Fz_T, steps=steps, T=T)
    oracle_profile = _leaf_mean_ux_yz_profile_3d(oracle;
                                                 volume=one(T) / T(8),
                                                 force_x=Fx_T)
    l2, linf = _profile_errors_3d(profile, oracle_profile)
    return ConservativeTreeMacroFlow3D{T}(
        :poiseuille_route_native_3d,
        coarse,
        patch,
        profile,
        oracle_profile,
        T(l2),
        T(linf),
        T(ux_mean),
        T(uy_mean),
        T(uz_mean),
        mass0,
        mass1,
        drift,
        abs(drift) / mass0,
        steps)
end
