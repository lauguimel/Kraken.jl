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
        is_solid=nothing,
        coarse_prolongation::Symbol=:flat)
    _check_composite_pair_layout(coarse_out, patch_out, coarse_in, patch_in)
    _check_route_stream_topology_layout(topology, coarse_in, patch_in)
    _check_route_stream_topology_layout(topology, coarse_out, patch_out)
    coarse_prolongation in (:flat, :limited_linear) ||
        throw(ArgumentError("coarse_prolongation must be :flat or :limited_linear"))
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
    cell_id_by_coord = _cell_id_by_coord_2d(topology)

    @inbounds for route in topology.routes
        if route.kind == ROUTE_BOUNDARY
            boundary_policy == :skip && continue
            src_cell = topology.cells[route.src]
            is_solid !== nothing && _cell_is_solid_2d(src_cell, is_solid) && continue
            boundary_policy in (
                :periodic_x, :periodic_x_wall_y, :periodic_x_moving_wall_y,
                :open_x_wall_y) ||
                throw(ArgumentError("unsupported route boundary policy: $boundary_policy"))
            if src_cell.level == 0 && route.weight < 1.0 &&
                    boundary_policy in (:periodic_x_wall_y,
                                        :periodic_x_moving_wall_y)
                _stream_periodic_x_wall_y_leaf_equivalent_boundary_samples_2d!(
                    coarse_out, patch_out, coarse_in, patch_in,
                    topology, cell_id_by_coord, src_cell, route.q, nx, ny,
                    u_south, u_north, rho_wall, volume_coarse;
                    coarse_prolongation=coarse_prolongation)
                continue
            end
            handled = false
            if boundary_policy in (
                    :periodic_x_wall_y, :periodic_x_moving_wall_y, :open_x_wall_y)
                handled = _stream_wall_y_boundary_route_2d!(
                    coarse_out, patch_out, coarse_in, patch_in,
                    topology, route, ny, u_south, u_north, rho_wall,
                    volume_coarse, volume_fine;
                    coarse_prolongation=coarse_prolongation)
            end
            boundary_policy == :open_x_wall_y && continue
            handled || _stream_periodic_x_boundary_route_2d!(
                coarse_out, patch_out, coarse_in, patch_in,
                topology, cell_id_by_coord, route, nx, ny;
                coarse_prolongation=coarse_prolongation)
            continue
        end

        src_cell = topology.cells[route.src]
        dst_cell = topology.cells[route.dst]
        if is_solid === nothing
            if coarse_prolongation == :limited_linear &&
                    src_cell.level == 0 && route.weight < 1.0
                _scatter_limited_linear_coarse_route_packet_2d!(
                    coarse_out, patch_out, coarse_in, patch_in,
                    topology, cell_id_by_coord, src_cell, route, nx, ny)
            else
                _scatter_route_packet_2d!(coarse_out, patch_out, coarse_in, patch_in,
                                          src_cell, dst_cell, route.q, route.weight)
            end
        else
            if coarse_prolongation == :limited_linear &&
                    src_cell.level == 0 && route.weight < 1.0
                _scatter_limited_linear_coarse_solid_route_packet_2d!(
                    coarse_out, patch_out, coarse_in, patch_in,
                    topology, cell_id_by_coord, src_cell, route, nx, ny,
                    is_solid)
            else
                _scatter_solid_route_packet_2d!(
                    coarse_out, patch_out, coarse_in, patch_in,
                    src_cell, dst_cell, route.q, route.weight, is_solid)
            end
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
        clear::Bool=true,
        coarse_prolongation::Symbol=:flat)
    return _stream_composite_routes_F_2d!(coarse_out, patch_out,
                                          coarse_in, patch_in,
                                          topology, :skip, clear;
                                          coarse_prolongation=coarse_prolongation)
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
        clear::Bool=true,
        coarse_prolongation::Symbol=:flat)
    return _stream_composite_routes_F_2d!(coarse_out, patch_out,
                                          coarse_in, patch_in,
                                          topology, :periodic_x, clear;
                                          coarse_prolongation=coarse_prolongation)
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
        clear::Bool=true,
        coarse_prolongation::Symbol=:flat)
    return _stream_composite_routes_F_2d!(coarse_out, patch_out,
                                          coarse_in, patch_in,
                                          topology, :periodic_x_wall_y, clear;
                                          coarse_prolongation=coarse_prolongation)
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
        clear::Bool=true,
        coarse_prolongation::Symbol=:flat)
    return _stream_composite_routes_F_2d!(coarse_out, patch_out,
                                          coarse_in, patch_in,
                                          topology, :periodic_x_moving_wall_y,
                                          clear;
                                          u_south=u_south,
                                          u_north=u_north,
                                          rho_wall=rho_wall,
                                          volume_coarse=volume_coarse,
                                          volume_fine=volume_fine,
                                          coarse_prolongation=coarse_prolongation)
end

function stream_composite_routes_zou_he_x_wall_y_F_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D,
        topology::ConservativeTreeTopology2D;
        u_in=0,
        rho_out=1,
        rho_wall=1,
        volume_coarse=1,
        volume_fine=0.25,
        clear::Bool=true,
        coarse_prolongation::Symbol=:flat)
    _stream_composite_routes_F_2d!(coarse_out, patch_out,
                                   coarse_in, patch_in,
                                   topology, :open_x_wall_y, clear;
                                   rho_wall=rho_wall,
                                   volume_coarse=volume_coarse,
                                   volume_fine=volume_fine,
                                   coarse_prolongation=coarse_prolongation)
    apply_composite_zou_he_west_F_2d!(
        coarse_out, patch_out, u_in, volume_coarse, volume_fine)
    apply_composite_zou_he_pressure_east_F_2d!(
        coarse_out, patch_out, volume_coarse, volume_fine; rho_out=rho_out)
    return coarse_out, patch_out
end

function stream_composite_routes_zou_he_x_wall_y_solid_F_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D,
        topology::ConservativeTreeTopology2D,
        is_solid::AbstractArray{Bool,2};
        u_in=0,
        rho_out=1,
        rho_wall=1,
        volume_coarse=1,
        volume_fine=0.25,
        clear::Bool=true,
        coarse_prolongation::Symbol=:flat)
    _stream_composite_routes_F_2d!(coarse_out, patch_out,
                                   coarse_in, patch_in,
                                   topology, :open_x_wall_y, clear;
                                   rho_wall=rho_wall,
                                   volume_coarse=volume_coarse,
                                   volume_fine=volume_fine,
                                   is_solid=is_solid,
                                   coarse_prolongation=coarse_prolongation)
    apply_composite_zou_he_west_F_2d!(
        coarse_out, patch_out, is_solid, u_in, volume_coarse, volume_fine)
    apply_composite_zou_he_pressure_east_F_2d!(
        coarse_out, patch_out, is_solid, volume_coarse, volume_fine;
        rho_out=rho_out)
    return coarse_out, patch_out
end

function stream_composite_routes_periodic_x_wall_y_solid_F_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D,
        topology::ConservativeTreeTopology2D,
        is_solid::AbstractArray{Bool,2};
        clear::Bool=true,
        coarse_prolongation::Symbol=:flat)
    return _stream_composite_routes_F_2d!(coarse_out, patch_out,
                                          coarse_in, patch_in,
                                          topology, :periodic_x_wall_y, clear;
                                          is_solid=is_solid,
                                          coarse_prolongation=coarse_prolongation)
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

