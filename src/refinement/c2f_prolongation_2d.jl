"""
    c2f_prolongation_2d.jl

Centralized coarse-to-fine (c2f) prolongation packets for AMR-D 2D
conservative-tree streaming.

Public packet API:
- `C2FProlongationContext2D`: host-side context for route-native dispatch.
- `c2f_route_packet_2d`: route-native packet dispatch.
- `c2f_subcycled_route_packet_2d`: CPU subcycled packet dispatch.
- `c2f_limited_linear_child_packet_2d`: CPU level-native phase child sample.

Schemes:
- `:flat`: replicate the parent population with the route's weight. This is the
  C11 bit-exact replacement for `route.weight * F[route.src, route.q]` and is
  valid for route-native dispatch, CPU subcycled routes, and GPU route packs.
- `:limited_linear`: current CPU subcycled sampled route reconstruction,
  extracted bit-exact from `subcycling_explosion_2d.jl`. It is valid only in
  CPU subcycled/sample contexts with `alpha_c2f = 1`. Route-native dispatch,
  GPU route packs, and wall-phase/wall-aware reconstruction remain C14+ work.

Reserved for future extraction missions: `:biquadratic` and `:wall_aware`.
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

@inline function _subcycle_route_packet_2d(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        route::ConservativeTreeRoute2D;
        alpha=1)
    return _subcycle_cell_route_packet_2d(
        F, spec, route.src, route.q, route.weight; alpha=alpha)
end

@inline function _subcycle_cell_route_packet_2d(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        src_id::Integer,
        q::Integer,
        weight;
        alpha=1)
    T = typeof(zero(eltype(F)) + weight + alpha)
    a = T(alpha)
    qi = _check_d2q9_q(q)
    src = Int(src_id)
    if a == one(a)
        return T(weight) * T(F[src, qi])
    end
    cell = spec.cells[src]
    return reconstructed_integrated_D2Q9_packet(
        @view(F[src, :]), cell.metrics.volume, qi, weight; alpha=alpha)
end

@inline function _check_conservative_tree_coarse_to_fine_prolongation_2d(
        prolongation::Symbol)
    prolongation in (:flat, :limited_linear) ||
        throw(ArgumentError("coarse_to_fine_prolongation must be :flat or :limited_linear"))
    return prolongation
end

@inline function _conservative_tree_same_level_Fq_2d(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        level::Int,
        i::Int,
        j::Int,
        q::Int;
        periodic_x::Bool=false)
    nx = _conservative_tree_level_size_2d(spec.Nx, level)
    ny = _conservative_tree_level_size_2d(spec.Ny, level)
    ii = periodic_x ? mod1(i, nx) : i
    1 <= ii <= nx && 1 <= j <= ny ||
        return false, zero(eltype(F))
    cell_id = conservative_tree_cell_id_2d(spec, level, ii, j)
    cell_id == 0 && return false, zero(eltype(F))
    # Inactive parent cells hold no valid F values; treat them as missing.
    spec.cells[cell_id].active || return false, zero(eltype(F))
    return true, F[cell_id, q]
end

function _conservative_tree_limited_same_level_slope_x_2d(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        src_id::Int,
        q::Int;
        periodic_x::Bool=false)
    src = spec.cells[src_id]
    center = F[src_id, q]
    has_left, left_value = _conservative_tree_same_level_Fq_2d(
        F, spec, src.level, src.i - 1, src.j, q; periodic_x=periodic_x)
    has_right, right_value = _conservative_tree_same_level_Fq_2d(
        F, spec, src.level, src.i + 1, src.j, q; periodic_x=periodic_x)
    if has_left && has_right
        return _minmod(center - left_value, right_value - center)
    end
    if has_left
        has_left2, left2_value = _conservative_tree_same_level_Fq_2d(
            F, spec, src.level, src.i - 2, src.j, q; periodic_x=periodic_x)
        if has_left2
            return (3 * center - 4 * left_value + left2_value) / 2
        end
        return center - left_value
    elseif has_right
        has_right2, right2_value = _conservative_tree_same_level_Fq_2d(
            F, spec, src.level, src.i + 2, src.j, q; periodic_x=periodic_x)
        if has_right2
            return (-3 * center + 4 * right_value - right2_value) / 2
        end
        return right_value - center
    end
    return zero(center)
end

function _conservative_tree_limited_same_level_slope_y_2d(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        src_id::Int,
        q::Int)
    src = spec.cells[src_id]
    center = F[src_id, q]
    has_south, south_value = _conservative_tree_same_level_Fq_2d(
        F, spec, src.level, src.i, src.j - 1, q)
    has_north, north_value = _conservative_tree_same_level_Fq_2d(
        F, spec, src.level, src.i, src.j + 1, q)
    if has_south && has_north
        return _minmod(center - south_value, north_value - center)
    end
    if has_south
        has_south2, south2_value = _conservative_tree_same_level_Fq_2d(
            F, spec, src.level, src.i, src.j - 2, q)
        if has_south2
            return (3 * center - 4 * south_value + south2_value) / 2
        end
        return center - south_value
    elseif has_north
        has_north2, north2_value = _conservative_tree_same_level_Fq_2d(
            F, spec, src.level, src.i, src.j + 2, q)
        if has_north2
            return (-3 * center + 4 * north_value - north2_value) / 2
        end
        return north_value - center
    end
    return zero(center)
end

"""
    c2f_limited_linear_child_packet_2d(F, spec, src_id, q, si, sj, scale;
                                       periodic_x=false)

Return the bit-exact C12 limited-linear child sample used by CPU subcycled
level-native phase routing. This helper is not wall-aware and is not valid
inside GPU kernels.
"""
function c2f_limited_linear_child_packet_2d(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        src_id::Int,
        q::Int,
        si::Int,
        sj::Int,
        scale::Int;
        periodic_x::Bool=false)
    center = F[src_id, q]
    sx = _conservative_tree_limited_same_level_slope_x_2d(
        F, spec, src_id, q; periodic_x=periodic_x)
    sy = _conservative_tree_limited_same_level_slope_y_2d(
        F, spec, src_id, q)

    area = inv(typeof(center)(scale * scale))
    max_offset = typeof(center)(scale - 1) / typeof(center)(2 * scale)
    max_delta = (abs(sx) + abs(sy)) * max_offset * area
    base = center * area
    if max_delta > zero(max_delta) && base < max_delta
        theta = base / max_delta
        sx *= theta
        sy *= theta
    end

    xoff = (typeof(center)(si) - (typeof(center)(scale) + one(center)) / 2) /
        typeof(center)(scale)
    yoff = (typeof(center)(sj) - (typeof(center)(scale) + one(center)) / 2) /
        typeof(center)(scale)
    return base + (sx * xoff + sy * yoff) * area
end

function _conservative_tree_limited_linear_sampled_route_packet_2d(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        route::ConservativeTreeRoute2D;
        periodic_x::Bool=false)
    src = spec.cells[route.src]
    sample_level = spec.max_level
    src.level < sample_level ||
        throw(ArgumentError("limited-linear sampled route source is already finest"))
    q = _check_d2q9_q(route.q)
    cx = d2q9_cx(q)
    cy = d2q9_cy(q)
    scale = 1 << (sample_level - src.level)
    nx_sample = _conservative_tree_level_size_2d(spec.Nx, sample_level)
    packet = zero(eltype(F))

    @inbounds for sj in 1:scale, si in 1:scale
        sample_i = (src.i - 1) * scale + si + cx
        sample_j = (src.j - 1) * scale + sj + cy
        if periodic_x
            sample_i = mod1(sample_i, nx_sample)
        end
        dst_id = _active_leaf_covering_sample_2d(
            spec, sample_level, sample_i, sample_j)
        dst_id == route.dst || continue
        kind = _route_kind_for_level_pair_2d(src, spec.cells[dst_id], q)
        kind == route.kind || continue
        packet += c2f_limited_linear_child_packet_2d(
            F, spec, route.src, q, si, sj, scale; periodic_x=periodic_x)
    end
    return packet
end

function c2f_limited_linear_route_packet_2d(
        F::AbstractMatrix,
        route::ConservativeTreeRoute2D,
        ctx::C2FProlongationContext2D)
    a = typeof(zero(eltype(F)) + ctx.alpha)(ctx.alpha)
    a == one(a) ||
        throw(ArgumentError("limited-linear coarse-to-fine prolongation currently requires alpha_c2f = 1"))
    return _conservative_tree_limited_linear_sampled_route_packet_2d(
        F, ctx.spec, route; periodic_x=ctx.periodic_x)
end

"""
    c2f_subcycled_route_packet_2d(F, spec, route; alpha=1,
                                  coarse_to_fine_prolongation=:flat,
                                  periodic_x=false)

Return a CPU subcycled coarse-to-fine route packet. `:flat` preserves the
legacy weighted-source packet for any `alpha`; `:limited_linear` uses the C12
sampled reconstruction and throws `ArgumentError` unless `alpha == 1`.
"""
function c2f_subcycled_route_packet_2d(
        F::AbstractMatrix,
        spec::ConservativeTreeSpec2D,
        route::ConservativeTreeRoute2D;
        alpha=1,
        coarse_to_fine_prolongation::Symbol=:flat,
        periodic_x::Bool=false)
    mode = _check_conservative_tree_coarse_to_fine_prolongation_2d(
        coarse_to_fine_prolongation)
    if mode == :limited_linear
        ctx = C2FProlongationContext2D(spec, nothing, periodic_x, alpha, :none)
        return c2f_limited_linear_route_packet_2d(F, route, ctx)
    end
    return _subcycle_route_packet_2d(F, spec, route; alpha=alpha)
end

@inline function c2f_route_packet_2d(
        scheme::Symbol,
        F::AbstractMatrix,
        route::ConservativeTreeRoute2D,
        ctx::C2FProlongationContext2D)
    # Route-native dispatch is intentionally conservative: only :flat is
    # enabled. C14+ should enable :limited_linear here once the wall-aware
    # scheme lands and route-native/GPU gates are documented together.
    scheme == :flat && return c2f_flat_route_packet_2d(F, route, ctx)
    scheme == :limited_linear &&
        throw(ArgumentError("route-native c2f prolongation :limited_linear " *
                            "is not enabled in C12"))
    throw(ArgumentError("c2f prolongation scheme $scheme is not implemented"))
end
