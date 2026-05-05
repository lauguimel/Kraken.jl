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
    policy in (:skip, :bounceback) ||
        throw(ArgumentError("boundary policy must be :skip or :bounceback"))
    return policy
end

"""
    stream_conservative_tree_routes_F_2d!(Fout, Fin, spec, table;
                                          boundary=:skip)

Scatter integrated D2Q9 populations through a static multilevel route table.
`Fout` is zeroed before scattering. Boundary routes are explicit:

- `boundary = :skip` drops boundary packets for packet-level route tests;
- `boundary = :bounceback` reflects boundary packets to `opposite(q)` in the
  source cell, useful for closed rest-state canaries.
"""
function stream_conservative_tree_routes_F_2d!(
        Fout::AbstractMatrix,
        Fin::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        table::ConservativeTreeRouteTable2D;
        boundary::Symbol=:skip)
    _check_conservative_tree_stream_args_2d(Fout, Fin, spec)
    policy = _check_conservative_tree_boundary_policy_2d(boundary)
    fill!(Fout, zero(eltype(Fout)))

    @inbounds for route in table.routes
        packet = route.weight * Fin[route.src, route.q]
        if route.dst == 0
            if policy == :bounceback
                Fout[route.src, d2q9_opposite(route.q)] += packet
            end
        else
            Fout[route.dst, route.q] += packet
        end
    end
    return Fout
end
