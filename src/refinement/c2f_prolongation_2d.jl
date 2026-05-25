"""
    c2f_prolongation_2d.jl

Centralized coarse-to-fine (c2f) prolongation packets for AMR-D 2D
conservative-tree streaming.

Schemes:
- `:flat`: replicate the parent population with the route's weight. This is the
  C11 bit-exact replacement for `route.weight * F[route.src, route.q]`.

Reserved for future extraction missions: `:limited_linear`, `:biquadratic`,
and `:wall_aware`.
"""

struct C2FProlongationContext2D
    spec::ConservativeTreeSpec2D
    table::Union{Nothing,ConservativeTreeRouteTable2D}
    periodic_x::Bool
    alpha
    interface_time_scaling::Symbol
end

@inline function c2f_flat_route_packet_2d(
        F::AbstractMatrix,
        route::ConservativeTreeRoute2D,
        ctx::C2FProlongationContext2D)
    return route.weight * F[route.src, route.q]
end

@inline function c2f_route_packet_2d(
        scheme::Symbol,
        F::AbstractMatrix,
        route::ConservativeTreeRoute2D,
        ctx::C2FProlongationContext2D)
    scheme == :flat && return c2f_flat_route_packet_2d(F, route, ctx)
    throw(ArgumentError("c2f prolongation scheme $scheme is not implemented " *
                        "in C11 (only :flat)"))
end
