# CPU reference streaming over nested conservative-tree route tables.
#
# This is a test-first transport primitive. Boundary handling is explicit and
# deliberately small so macro-flow boundary closures remain separate.

function _check_conservative_tree_stream_args_2d(Fout::AbstractMatrix,
                                                 Fin::AbstractMatrix,
                                                 spec::ConservativeTreeSpec2D)
    _check_conservative_tree_F_2d(Fout, spec)
    _check_conservative_tree_F_2d(Fin, spec)
    return nothing
end

@inline function _check_conservative_tree_boundary_policy_2d(policy::Symbol)
    policy in (:skip, :bounceback, :periodic_x_wall_y,
               :periodic_x_moving_wall_y) ||
        throw(ArgumentError("boundary policy must be :skip, :bounceback, " *
                            ":periodic_x_wall_y, or :periodic_x_moving_wall_y"))
    return policy
end

@inline function _conservative_tree_boundary_reflection_packet_2d(
        Fin::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        route::ConservativeTreeRoute2D,
        policy::Symbol;
        u_south=0,
        u_north=0,
        rho_wall=1)
    packet = route.weight * Fin[route.src, route.q]
    cy = d2q9_cy(route.q)
    if policy == :periodic_x_wall_y || policy == :periodic_x_moving_wall_y
        cy != 0 ||
            throw(ArgumentError("periodic-x wall-y boundary policy received an x-boundary route; rebuild the route table with periodic_x=true"))
    end
    policy == :periodic_x_moving_wall_y || return packet

    wall_u = cy < 0 ? u_south : u_north
    opp = d2q9_opposite(route.q)
    volume = spec.cells[route.src].metrics.volume
    return packet + route.weight * _moving_wall_delta(volume, rho_wall,
                                                      wall_u, opp)
end

"""
    stream_conservative_tree_routes_F_2d!(Fout, Fin, spec, table;
                                          boundary=:skip, u_south=0,
                                          u_north=0, rho_wall=1)

Scatter integrated D2Q9 populations through a static multilevel route table.
`Fout` is zeroed before scattering. Boundary routes are explicit:

- `boundary = :skip` drops boundary packets for packet-level route tests;
- `boundary = :bounceback` reflects boundary packets to `opposite(q)` in the
  source cell, useful for closed rest-state canaries.
- `boundary = :periodic_x_wall_y` is the same reflection contract for route
  tables built with `periodic_x=true`;
- `boundary = :periodic_x_moving_wall_y` adds the standard tangential moving
  wall correction on y-boundary packets.
"""
function stream_conservative_tree_routes_F_2d!(
        Fout::AbstractMatrix,
        Fin::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        table::ConservativeTreeRouteTable2D;
        boundary::Symbol=:skip,
        u_south=0,
        u_north=0,
        rho_wall=1)
    _check_conservative_tree_stream_args_2d(Fout, Fin, spec)
    policy = _check_conservative_tree_boundary_policy_2d(boundary)
    fill!(Fout, zero(eltype(Fout)))

    @inbounds for route in table.routes
        packet = route.weight * Fin[route.src, route.q]
        if route.dst == 0
            if policy != :skip
                reflected = policy == :bounceback ? packet :
                    _conservative_tree_boundary_reflection_packet_2d(
                        Fin, spec, route, policy; u_south=u_south,
                        u_north=u_north, rho_wall=rho_wall)
                Fout[route.src, d2q9_opposite(route.q)] += reflected
            end
        else
            Fout[route.dst, route.q] += packet
        end
    end
    return Fout
end
