using KernelAbstractions
using LinearAlgebra

# --- Conformation tensor LBM (TRT, D2Q9) for viscoelastic flows ---
#
# Reference: Liu et al., "An improved lattice Boltzmann method with a
# novel conservative boundary scheme for viscoelastic fluid flows",
# arxiv 2508.16997 (Aug 2025).
#
# Three independent scalar D2Q9 advection-diffusion-reaction LBMs evolve
# C_xx, C_xy, C_yy. The diffusion is built into the LBM via the relaxation
# time: κ = (τp,1 - 0.5)/3. TRT (two-relaxation-time) gives better stability.
# Source S = Φ_αβ contains the upper-convected derivative + relaxation.

# D2Q9 indexing (Kraken convention):
#   1: rest (0,0)        opp = 1
#   2: E   (+1, 0)       opp = 4
#   3: N   ( 0,+1)       opp = 5
#   4: W   (-1, 0)       opp = 2
#   5: S   ( 0,-1)       opp = 3
#   6: NE  (+1,+1)       opp = 8
#   7: NW  (-1,+1)       opp = 9
#   8: SW  (-1,-1)       opp = 6
#   9: SE  (+1,-1)       opp = 7

@inline function _wall_aware_dx_2d(a, is_solid, i, j, Nx, ::Type{T}) where {T}
    plus_ok = i < Nx && !is_solid[i + 1, j]
    minus_ok = i > 1 && !is_solid[i - 1, j]
    plus2_ok = i < Nx - 1 && plus_ok && !is_solid[i + 2, j]
    minus2_ok = i > 2 && minus_ok && !is_solid[i - 2, j]
    if plus_ok && minus_ok
        return (a[i + 1, j] - a[i - 1, j]) / T(2)
    elseif plus2_ok
        return (-T(3) * a[i, j] + T(4) * a[i + 1, j] - a[i + 2, j]) / T(2)
    elseif minus2_ok
        return (T(3) * a[i, j] - T(4) * a[i - 1, j] + a[i - 2, j]) / T(2)
    elseif plus_ok
        return a[i + 1, j] - a[i, j]
    elseif minus_ok
        return a[i, j] - a[i - 1, j]
    else
        return zero(T)
    end
end

@inline function _wall_aware_dy_2d(a, is_solid, i, j, Ny, ::Type{T}) where {T}
    plus_ok = j < Ny && !is_solid[i, j + 1]
    minus_ok = j > 1 && !is_solid[i, j - 1]
    plus2_ok = j < Ny - 1 && plus_ok && !is_solid[i, j + 2]
    minus2_ok = j > 2 && minus_ok && !is_solid[i, j - 2]
    if plus_ok && minus_ok
        return (a[i, j + 1] - a[i, j - 1]) / T(2)
    elseif plus2_ok
        return (-T(3) * a[i, j] + T(4) * a[i, j + 1] - a[i, j + 2]) / T(2)
    elseif minus2_ok
        return (T(3) * a[i, j] - T(4) * a[i, j - 1] + a[i, j - 2]) / T(2)
    elseif plus_ok
        return a[i, j + 1] - a[i, j]
    elseif minus_ok
        return a[i, j] - a[i, j - 1]
    else
        return zero(T)
    end
end

struct ConformationGradientStencils2D{CT,DI,DJ,WI,WJ,WQ,IW,COUNT,FB}
    coeff::CT
    di::DI
    dj::DJ
    wall_i::WI
    wall_j::WJ
    wall_q::WQ
    is_wall::IW
    count::COUNT
    fallback::FB
    max_terms::Int
    mode::Symbol
end

function _upload_gradient_stencil_array(backend, a)
    host = a isa BitArray ? Array(a) : a
    b = KernelAbstractions.allocate(backend, eltype(host), size(host)...)
    copyto!(b, host)
    return b
end

function _push_gradient_stencil_term!(coeff, di, dj, wall_i, wall_j, wall_q,
                                      is_wall, count, i, j, axis, c, δi, δj,
                                      wi, wj, wq, wall_term::Bool, max_terms)
    abs(c) > 0 || return nothing
    slot = Int(count[i, j, axis]) + 1
    slot <= max_terms ||
        error("conformation gradient stencil exceeded max_terms=$(max_terms)")
    coeff[i, j, axis, slot] = c
    di[i, j, axis, slot] = Int32(δi)
    dj[i, j, axis, slot] = Int32(δj)
    wall_i[i, j, axis, slot] = Int32(wi)
    wall_j[i, j, axis, slot] = Int32(wj)
    wall_q[i, j, axis, slot] = Int32(wq)
    is_wall[i, j, axis, slot] = wall_term
    count[i, j, axis] = Int32(slot)
    return nothing
end

function _wall_aware_axis_terms_2d(is_solid, i, j, Nx, Ny, axis::Int)
    if axis == 1
        plus_ok = i < Nx && !is_solid[i + 1, j]
        minus_ok = i > 1 && !is_solid[i - 1, j]
        plus2_ok = i < Nx - 1 && plus_ok && !is_solid[i + 2, j]
        minus2_ok = i > 2 && minus_ok && !is_solid[i - 2, j]
        plus_ok && minus_ok && return ((0.5, 1, 0), (-0.5, -1, 0))
        plus2_ok && return ((-1.5, 0, 0), (2.0, 1, 0), (-0.5, 2, 0))
        minus2_ok && return ((1.5, 0, 0), (-2.0, -1, 0), (0.5, -2, 0))
        plus_ok && return ((-1.0, 0, 0), (1.0, 1, 0))
        minus_ok && return ((1.0, 0, 0), (-1.0, -1, 0))
    else
        plus_ok = j < Ny && !is_solid[i, j + 1]
        minus_ok = j > 1 && !is_solid[i, j - 1]
        plus2_ok = j < Ny - 1 && plus_ok && !is_solid[i, j + 2]
        minus2_ok = j > 2 && minus_ok && !is_solid[i, j - 2]
        plus_ok && minus_ok && return ((0.5, 0, 1), (-0.5, 0, -1))
        plus2_ok && return ((-1.5, 0, 0), (2.0, 0, 1), (-0.5, 0, 2))
        minus2_ok && return ((1.5, 0, 0), (-2.0, 0, -1), (0.5, 0, -2))
        plus_ok && return ((-1.0, 0, 0), (1.0, 0, 1))
        minus_ok && return ((1.0, 0, 0), (-1.0, 0, -1))
    end
    return ()
end

function _fill_wall_aware_axis_stencil!(coeff, di, dj, wall_i, wall_j, wall_q,
                                        is_wall, count, fallback,
                                        is_solid, i, j, Nx, Ny, axis,
                                        ::Type{T}, max_terms) where {T}
    fallback[i, j, axis] = true
    for (c, δi, δj) in _wall_aware_axis_terms_2d(is_solid, i, j, Nx, Ny, axis)
        _push_gradient_stencil_term!(
            coeff, di, dj, wall_i, wall_j, wall_q, is_wall, count,
            i, j, axis, T(c), δi, δj, 0, 0, 0, false, max_terms,
        )
    end
    return nothing
end

function _derivative_rhs_weights_at_zero_2d(positions, ::Type{T}) where {T}
    n = length(positions)
    if n >= 2
        rows = T[]
        for s in positions
            push!(rows, T(s))
            push!(rows, T(s) * T(s))
        end
        A = transpose(reshape(rows, 2, n))
        return vec((A \ Matrix{T}(LinearAlgebra.I, n, n))[1, :])
    elseif n == 1
        return T[one(T) / T(positions[1])]
    else
        return T[]
    end
end

function _fill_embedded_axis_stencil!(coeff, di, dj, wall_i, wall_j, wall_q,
                                      is_wall, count, fallback,
                                      is_solid, q_wall,
                                      i, j, Nx, Ny, axis, ::Type{T},
                                      max_terms) where {T}
    samples = NamedTuple[]
    dirs = axis == 1 ? ((2, 1.0), (4, -1.0)) : ((3, 1.0), (5, -1.0))
    for (q, sfluid) in dirs
        cx = Int(_D2Q9_CX[q])
        cy = Int(_D2Q9_CY[q])
        ii = i + cx
        jj = j + cy
        if 1 <= ii <= Nx && 1 <= jj <= Ny && !is_solid[ii, jj]
            push!(samples, (s=T(sfluid), is_wall=false,
                            di=cx, dj=cy, wi=0, wj=0, wq=0))
        elseif q_wall[i, j, q] > 0
            push!(samples, (s=T(sfluid) * T(q_wall[i, j, q]),
                            is_wall=true, di=0, dj=0,
                            wi=i, wj=j, wq=q))
        end
    end

    if length(samples) < 2 && !any(sample -> sample.is_wall, samples)
        _fill_wall_aware_axis_stencil!(coeff, di, dj, wall_i, wall_j, wall_q,
                                       is_wall, count, fallback, is_solid,
                                       i, j, Nx, Ny, axis, T, max_terms)
        return nothing
    end

    weights = _derivative_rhs_weights_at_zero_2d(
        [sample.s for sample in samples], T,
    )
    _push_gradient_stencil_term!(
        coeff, di, dj, wall_i, wall_j, wall_q, is_wall, count,
        i, j, axis, -sum(weights), 0, 0, 0, 0, 0, false, max_terms,
    )
    for (weight, sample) in zip(weights, samples)
        _push_gradient_stencil_term!(
            coeff, di, dj, wall_i, wall_j, wall_q, is_wall, count,
            i, j, axis, weight, sample.di, sample.dj,
            sample.wi, sample.wj, sample.wq, sample.is_wall, max_terms,
        )
    end
    return nothing
end

function _polyfit_features_conformation_2d(dx, dy, degree::Int)
    if degree == 4
        return [
            dx, dy,
            0.5 * dx^2, dx * dy, 0.5 * dy^2,
            dx^3 / 6, 0.5 * dx^2 * dy, 0.5 * dx * dy^2, dy^3 / 6,
            dx^4 / 24, dx^3 * dy / 6, 0.25 * dx^2 * dy^2,
            dx * dy^3 / 6, dy^4 / 24,
        ]
    elseif degree == 3
        return [
            dx, dy,
            0.5 * dx^2, dx * dy, 0.5 * dy^2,
            dx^3 / 6, 0.5 * dx^2 * dy, 0.5 * dx * dy^2, dy^3 / 6,
        ]
    else
        error("unsupported conformation gradient polynomial degree $(degree)")
    end
end

function _fill_wallfit_stencils!(coeff, di, dj, wall_i, wall_j, wall_q,
                                 is_wall, count, fallback, is_solid, q_wall,
                                 i, j, Nx, Ny, degree, radius, wall_weight,
                                 ::Type{T}, max_terms) where {T}
    ncoef = length(_polyfit_features_conformation_2d(one(T), zero(T), degree))
    rows = T[]
    samples = NamedTuple[]
    x0 = T(i - 1)
    y0 = T(j - 1)

    for δj in -radius:radius, δi in -radius:radius
        ii = i + δi
        jj = j + δj
        1 <= ii <= Nx && 1 <= jj <= Ny || continue
        is_solid[ii, jj] && continue
        δi^2 + δj^2 <= radius^2 || continue

        if !(δi == 0 && δj == 0)
            dist2 = δi^2 + δj^2
            rhs_scale = sqrt(one(T) / T(max(1, dist2)))
            append!(rows, rhs_scale .* _polyfit_features_conformation_2d(
                T(δi), T(δj), degree,
            ))
            push!(samples, (is_wall=false, di=δi, dj=δj,
                            wi=0, wj=0, wq=0, rhs_scale=rhs_scale))
        end

        for q in 2:9
            qw = q_wall[ii, jj, q]
            qw > 0 || continue
            wx = T(ii - 1) + T(qw) * T(_D2Q9_CX[q])
            wy = T(jj - 1) + T(qw) * T(_D2Q9_CY[q])
            dx = wx - x0
            dy = wy - y0
            dx^2 + dy^2 <= T(radius + 0.5)^2 || continue
            rhs_scale = sqrt(T(wall_weight) / max(T(0.25), dx^2 + dy^2))
            append!(rows, rhs_scale .* _polyfit_features_conformation_2d(
                dx, dy, degree,
            ))
            push!(samples, (is_wall=true, di=0, dj=0,
                            wi=ii, wj=jj, wq=q, rhs_scale=rhs_scale))
        end
    end

    nrows = length(samples)
    if nrows < ncoef
        _fill_wall_aware_axis_stencil!(coeff, di, dj, wall_i, wall_j, wall_q,
                                       is_wall, count, fallback, is_solid,
                                       i, j, Nx, Ny, 1, T, max_terms)
        _fill_wall_aware_axis_stencil!(coeff, di, dj, wall_i, wall_j, wall_q,
                                       is_wall, count, fallback, is_solid,
                                       i, j, Nx, Ny, 2, T, max_terms)
        return nothing
    end

    A = transpose(reshape(rows, ncoef, nrows))
    if LinearAlgebra.rank(A) < ncoef
        _fill_wall_aware_axis_stencil!(coeff, di, dj, wall_i, wall_j, wall_q,
                                       is_wall, count, fallback, is_solid,
                                       i, j, Nx, Ny, 1, T, max_terms)
        _fill_wall_aware_axis_stencil!(coeff, di, dj, wall_i, wall_j, wall_q,
                                       is_wall, count, fallback, is_solid,
                                       i, j, Nx, Ny, 2, T, max_terms)
        return nothing
    end

    weights = A \ Matrix{T}(LinearAlgebra.I, nrows, nrows)
    for axis in 1:2
        center_coeff = zero(T)
        row = axis
        for (weight, sample) in zip(vec(weights[row, :]), samples)
            c = weight * sample.rhs_scale
            center_coeff -= c
            _push_gradient_stencil_term!(
                coeff, di, dj, wall_i, wall_j, wall_q, is_wall, count,
                i, j, axis, c, sample.di, sample.dj,
                sample.wi, sample.wj, sample.wq, sample.is_wall, max_terms,
            )
        end
        _push_gradient_stencil_term!(
            coeff, di, dj, wall_i, wall_j, wall_q, is_wall, count,
            i, j, axis, center_coeff, 0, 0, 0, 0, 0, false, max_terms,
        )
    end
    return nothing
end

function precompute_conformation_gradient_stencils_2d(
        is_solid, q_wall; mode::Symbol=:embedded_axis,
        max_terms::Union{Nothing,Integer}=nothing, degree::Int=4,
        radius::Int=3, wall_weight::Real=16.0,
        backend=KernelAbstractions.CPU(), FT=eltype(q_wall))
    mode in (:embedded_axis, :wallfit4) ||
        error("unknown conformation gradient stencil mode $(mode); expected :embedded_axis or :wallfit4")
    Nx, Ny, _ = size(q_wall)
    slots = Int(something(max_terms, mode === :wallfit4 ? 64 : 4))
    coeff = zeros(FT, Nx, Ny, 2, slots)
    di = zeros(Int32, Nx, Ny, 2, slots)
    dj = zeros(Int32, Nx, Ny, 2, slots)
    wall_i = zeros(Int32, Nx, Ny, 2, slots)
    wall_j = zeros(Int32, Nx, Ny, 2, slots)
    wall_q = zeros(Int32, Nx, Ny, 2, slots)
    is_wall = falses(Nx, Ny, 2, slots)
    count = zeros(Int32, Nx, Ny, 2)
    fallback = falses(Nx, Ny, 2)

    @inbounds for j in 1:Ny, i in 1:Nx
        is_solid[i, j] && continue
        if mode === :embedded_axis
            _fill_embedded_axis_stencil!(
                coeff, di, dj, wall_i, wall_j, wall_q, is_wall, count,
                fallback, is_solid, q_wall, i, j, Nx, Ny, 1, FT, slots,
            )
            _fill_embedded_axis_stencil!(
                coeff, di, dj, wall_i, wall_j, wall_q, is_wall, count,
                fallback, is_solid, q_wall, i, j, Nx, Ny, 2, FT, slots,
            )
        else
            _fill_wallfit_stencils!(
                coeff, di, dj, wall_i, wall_j, wall_q, is_wall, count,
                fallback, is_solid, q_wall, i, j, Nx, Ny, degree, radius,
                wall_weight, FT, slots,
            )
        end
    end

    return ConformationGradientStencils2D(
        _upload_gradient_stencil_array(backend, coeff),
        _upload_gradient_stencil_array(backend, di),
        _upload_gradient_stencil_array(backend, dj),
        _upload_gradient_stencil_array(backend, wall_i),
        _upload_gradient_stencil_array(backend, wall_j),
        _upload_gradient_stencil_array(backend, wall_q),
        _upload_gradient_stencil_array(backend, is_wall),
        _upload_gradient_stencil_array(backend, count),
        _upload_gradient_stencil_array(backend, fallback),
        slots,
        mode,
    )
end

function conformation_gradient_stencil_stats_2d(stencils::ConformationGradientStencils2D)
    count = Array(stencils.count)
    is_wall = Array(stencils.is_wall)
    fallback = Array(stencils.fallback)
    Nx, Ny, _, _ = size(is_wall)
    max_count = 0
    max_wall_count = 0
    fallback_count = 0
    @inbounds for axis in 1:2, j in 1:Ny, i in 1:Nx
        n = Int(count[i, j, axis])
        max_count = max(max_count, n)
        nw = 0
        for slot in 1:n
            nw += is_wall[i, j, axis, slot] ? 1 : 0
        end
        max_wall_count = max(max_wall_count, nw)
        fallback_count += fallback[i, j, axis] ? 1 : 0
    end
    return (; max_count, max_wall_count, fallback_count,
              max_terms=stencils.max_terms, mode=stencils.mode)
end

@inline function _conformation_gradient_from_stencil_arrays_2d(
        a, uwall, coeff, di, dj, wall_i, wall_j, wall_q, is_wall, count,
        i, j, axis::Int, ::Type{T}) where {T}
    value = zero(T)
    n = Int(count[i, j, axis])
    @inbounds for slot in 1:n
        c = T(coeff[i, j, axis, slot])
        if is_wall[i, j, axis, slot]
            wi = Int(wall_i[i, j, axis, slot])
            wj = Int(wall_j[i, j, axis, slot])
            wq = Int(wall_q[i, j, axis, slot])
            value += c * T(uwall[wi, wj, wq])
        else
            ii = i + Int(di[i, j, axis, slot])
            jj = j + Int(dj[i, j, axis, slot])
            value += c * T(a[ii, jj])
        end
    end
    return value
end

@inline function _conformation_gradient_from_stencil_2d(
        a, uwall, stencils::ConformationGradientStencils2D, i, j,
        axis::Int, ::Type{T}) where {T}
    return _conformation_gradient_from_stencil_arrays_2d(
        a, uwall, stencils.coeff, stencils.di, stencils.dj,
        stencils.wall_i, stencils.wall_j, stencils.wall_q,
        stencils.is_wall, stencils.count, i, j, axis, T,
    )
end

function conformation_velocity_gradient_from_stencils_2d(
        ux, uy, uwx, uwy, stencils::ConformationGradientStencils2D, i, j)
    T = promote_type(eltype(ux), eltype(uy), eltype(uwx), eltype(uwy),
                     eltype(stencils.coeff))
    return (;
        dudx = _conformation_gradient_from_stencil_2d(ux, uwx, stencils, i, j, 1, T),
        dudy = _conformation_gradient_from_stencil_2d(ux, uwx, stencils, i, j, 2, T),
        dvdx = _conformation_gradient_from_stencil_2d(uy, uwy, stencils, i, j, 1, T),
        dvdy = _conformation_gradient_from_stencil_2d(uy, uwy, stencils, i, j, 2, T),
    )
end

@inline function conformation_source_2d(cxx::T, cxy::T, cyy::T,
                                        dudx::T, dudy::T, dvdx::T, dvdy::T,
                                        λ::T, component::Int) where {T<:AbstractFloat}
    return conformation_source_with_divergence_2d(
        cxx, cxy, cyy, dudx, dudy, dvdx, dvdy, dudx + dvdy, λ, component,
    )
end

@inline function conformation_source_with_divergence_2d(
        cxx::T, cxy::T, cyy::T,
        dudx::T, dudy::T, dvdx::T, dvdy::T,
        advective_divu::T, λ::T, component::Int) where {T<:AbstractFloat}
    inv_λ = one(T) / λ
    if component == 1
        return -inv_λ * (cxx - one(T)) + T(2) * (cxx*dudx + cxy*dudy) + cxx*advective_divu
    elseif component == 2
        return -inv_λ * cxy + (cxx*dvdx + cyy*dudy + cxy*(dudx + dvdy)) + cxy*advective_divu
    else
        return -inv_λ * (cyy - one(T)) + T(2) * (cxy*dvdx + cyy*dvdy) + cyy*advective_divu
    end
end

@inline function _apply_conformation_divergence_mode_2d(
        dudx::T, dudy::T, dvdx::T, dvdy::T, mode::Int) where {T<:AbstractFloat}
    if mode == 1 || mode == 2
        half_trace = T(0.5) * (dudx + dvdy)
        return dudx - half_trace, dudy, dvdx, dvdy - half_trace
    end
    return dudx, dudy, dvdx, dvdy
end

@inline function _conformation_divergence_mode_code(mode::Symbol)
    mode === :numerical && return 0
    mode === :trace_free && return 1
    mode === :trace_free_conservative && return 2
    error("unknown conformation_divergence_mode $(mode); expected :numerical, :trace_free, or :trace_free_conservative")
end

@inline function _conformation_advective_divergence_2d(
        raw_dudx::T, raw_dvdy::T, dudx::T, dvdy::T,
        mode::Int) where {T<:AbstractFloat}
    mode == 2 && return raw_dudx + raw_dvdy
    return dudx + dvdy
end

@kernel function collide_conformation_2d_kernel!(g, @Const(C_field), @Const(ux), @Const(uy),
                                                   @Const(C_xx_f), @Const(C_xy_f), @Const(C_yy_f),
                                                   @Const(is_solid),
                                                   tau_plus, tau_minus, lambda,
                                                   component, divergence_mode, Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds if !is_solid[i, j]
        T = eltype(g)
        φ = C_field[i, j]
        u = ux[i, j]
        v = uy[i, j]
        usq = u*u + v*v

        # Velocity gradient: centered in the bulk, one-sided next to
        # embedded solids/domain walls. Do not wrap x periodically in
        # inlet/outlet cylinder runs.
        dudx = _wall_aware_dx_2d(ux, is_solid, i, j, Nx, T)
        dvdx = _wall_aware_dx_2d(uy, is_solid, i, j, Nx, T)
        dudy = _wall_aware_dy_2d(ux, is_solid, i, j, Ny, T)
        dvdy = _wall_aware_dy_2d(uy, is_solid, i, j, Ny, T)
        raw_dudx = dudx
        raw_dvdy = dvdy
        dudx, dudy, dvdx, dvdy = _apply_conformation_divergence_mode_2d(
            dudx, dudy, dvdx, dvdy, divergence_mode)
        advective_divu = _conformation_advective_divergence_2d(
            raw_dudx, raw_dvdy, dudx, dvdy, divergence_mode)

        # Source term Φ_αβ
        cxx = C_xx_f[i, j]; cxy = C_xy_f[i, j]; cyy = C_yy_f[i, j]
        S = conformation_source_with_divergence_2d(
            cxx, cxy, cyy, dudx, dudy, dvdx, dvdy,
            advective_divu, T(lambda), component)

        # Pre-load all 9 populations
        g1 = g[i,j,1]; g2 = g[i,j,2]; g3 = g[i,j,3]; g4 = g[i,j,4]; g5 = g[i,j,5]
        g6 = g[i,j,6]; g7 = g[i,j,7]; g8 = g[i,j,8]; g9 = g[i,j,9]

        # Equilibria (reuse feq_2d with φ instead of ρ)
        ge1 = feq_2d(Val(1), φ, u, v, usq)
        ge2 = feq_2d(Val(2), φ, u, v, usq)
        ge3 = feq_2d(Val(3), φ, u, v, usq)
        ge4 = feq_2d(Val(4), φ, u, v, usq)
        ge5 = feq_2d(Val(5), φ, u, v, usq)
        ge6 = feq_2d(Val(6), φ, u, v, usq)
        ge7 = feq_2d(Val(7), φ, u, v, usq)
        ge8 = feq_2d(Val(8), φ, u, v, usq)
        ge9 = feq_2d(Val(9), φ, u, v, usq)

        ωp = one(T) / T(tau_plus)
        ωm = one(T) / T(tau_minus)
        half = T(0.5)
        wr = T(4/9); wa = T(1/9); we = T(1/36)
        source_linear = (one(T) - ωp * half) * T(3) * S
        S1 = wr * S
        S2 = wa * (S + source_linear * u)
        S3 = wa * (S + source_linear * v)
        S4 = wa * (S - source_linear * u)
        S5 = wa * (S - source_linear * v)
        S6 = we * (S + source_linear * (u + v))
        S7 = we * (S + source_linear * (-u + v))
        S8 = we * (S - source_linear * (u + v))
        S9 = we * (S + source_linear * (u - v))

        # TRT: each pair (q, opp) is collided together.
        # For q=1 (rest, self-opposite), only the symmetric part exists.
        nq1 = g1 - ge1   # already symmetric (opp=1)
        g[i,j,1] = g1 - ωp * nq1 + S1

        # Pair (2, 4) — E and W
        gp24 = (g2 + g4) * half;  gm24 = (g2 - g4) * half
        ep24 = (ge2 + ge4) * half; em24 = (ge2 - ge4) * half
        post2 = g2 - ωp*(gp24 - ep24) - ωm*(gm24 - em24)
        post4 = g4 - ωp*(gp24 - ep24) - ωm*(-(gm24 - em24))
        g[i,j,2] = post2 + S2
        g[i,j,4] = post4 + S4

        # Pair (3, 5) — N and S
        gp35 = (g3 + g5) * half;  gm35 = (g3 - g5) * half
        ep35 = (ge3 + ge5) * half; em35 = (ge3 - ge5) * half
        post3 = g3 - ωp*(gp35 - ep35) - ωm*(gm35 - em35)
        post5 = g5 - ωp*(gp35 - ep35) - ωm*(-(gm35 - em35))
        g[i,j,3] = post3 + S3
        g[i,j,5] = post5 + S5

        # Pair (6, 8) — NE and SW
        gp68 = (g6 + g8) * half;  gm68 = (g6 - g8) * half
        ep68 = (ge6 + ge8) * half; em68 = (ge6 - ge8) * half
        post6 = g6 - ωp*(gp68 - ep68) - ωm*(gm68 - em68)
        post8 = g8 - ωp*(gp68 - ep68) - ωm*(-(gm68 - em68))
        g[i,j,6] = post6 + S6
        g[i,j,8] = post8 + S8

        # Pair (7, 9) — NW and SE
        gp79 = (g7 + g9) * half;  gm79 = (g7 - g9) * half
        ep79 = (ge7 + ge9) * half; em79 = (ge7 - ge9) * half
        post7 = g7 - ωp*(gp79 - ep79) - ωm*(gm79 - em79)
        post9 = g9 - ωp*(gp79 - ep79) - ωm*(-(gm79 - em79))
        g[i,j,7] = post7 + S7
        g[i,j,9] = post9 + S9
    end
end

"""
    collide_conformation_2d!(g, C_field, ux, uy, C_xx, C_xy, C_yy, is_solid,
                              tau_plus, lambda; magic=0.25, component=1)

TRT collision + source for one scalar component of the conformation tensor.
`g` : D2Q9 distributions for that component, shape (Nx, Ny, 9)
`C_field` : the macroscopic value being evolved (= moments of g)
`C_xx, C_xy, C_yy` : all 3 components needed for the source Φ
`component` : 1=xx, 2=xy, 3=yy
`tau_plus = τp,1` — sets diffusion: κ = (tau_plus - 0.5)/3
`tau_minus = magic/(tau_plus - 0.5) + 0.5` — TRT magic-parameter relation
"""
function collide_conformation_2d!(g, C_field, ux, uy, C_xx, C_xy, C_yy, is_solid,
                                   tau_plus, lambda; magic=0.25, component=1,
                                   divergence_mode::Symbol=:numerical)
    backend = KernelAbstractions.get_backend(g)
    Nx, Ny = size(C_field)
    T = eltype(g)
    tau_minus = magic / (tau_plus - 0.5) + 0.5
    kernel! = collide_conformation_2d_kernel!(backend)
    kernel!(g, C_field, ux, uy, C_xx, C_xy, C_yy, is_solid,
            T(tau_plus), T(tau_minus), T(lambda),
            Int(component), _conformation_divergence_mode_code(divergence_mode),
            Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

@kernel function collide_conformation_2d_gradient_stencils_kernel!(
        g, @Const(C_field), @Const(ux), @Const(uy),
        @Const(C_xx_f), @Const(C_xy_f), @Const(C_yy_f), @Const(is_solid),
        @Const(uwx), @Const(uwy),
        @Const(grad_coeff), @Const(grad_di), @Const(grad_dj),
        @Const(grad_wall_i), @Const(grad_wall_j), @Const(grad_wall_q),
        @Const(grad_is_wall), @Const(grad_count),
        tau_plus, tau_minus, lambda, component, divergence_mode, Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds if !is_solid[i, j]
        T = eltype(g)
        φ = C_field[i, j]
        u = ux[i, j]
        v = uy[i, j]
        usq = u*u + v*v

        dudx = _conformation_gradient_from_stencil_arrays_2d(
            ux, uwx, grad_coeff, grad_di, grad_dj,
            grad_wall_i, grad_wall_j, grad_wall_q,
            grad_is_wall, grad_count, i, j, 1, T,
        )
        dudy = _conformation_gradient_from_stencil_arrays_2d(
            ux, uwx, grad_coeff, grad_di, grad_dj,
            grad_wall_i, grad_wall_j, grad_wall_q,
            grad_is_wall, grad_count, i, j, 2, T,
        )
        dvdx = _conformation_gradient_from_stencil_arrays_2d(
            uy, uwy, grad_coeff, grad_di, grad_dj,
            grad_wall_i, grad_wall_j, grad_wall_q,
            grad_is_wall, grad_count, i, j, 1, T,
        )
        dvdy = _conformation_gradient_from_stencil_arrays_2d(
            uy, uwy, grad_coeff, grad_di, grad_dj,
            grad_wall_i, grad_wall_j, grad_wall_q,
            grad_is_wall, grad_count, i, j, 2, T,
        )
        raw_dudx = dudx
        raw_dvdy = dvdy
        dudx, dudy, dvdx, dvdy = _apply_conformation_divergence_mode_2d(
            dudx, dudy, dvdx, dvdy, divergence_mode)
        advective_divu = _conformation_advective_divergence_2d(
            raw_dudx, raw_dvdy, dudx, dvdy, divergence_mode)

        cxx = C_xx_f[i, j]; cxy = C_xy_f[i, j]; cyy = C_yy_f[i, j]
        S = conformation_source_with_divergence_2d(
            cxx, cxy, cyy, dudx, dudy, dvdx, dvdy,
            advective_divu, T(lambda), component)

        g1 = g[i,j,1]; g2 = g[i,j,2]; g3 = g[i,j,3]; g4 = g[i,j,4]; g5 = g[i,j,5]
        g6 = g[i,j,6]; g7 = g[i,j,7]; g8 = g[i,j,8]; g9 = g[i,j,9]

        ge1 = feq_2d(Val(1), φ, u, v, usq)
        ge2 = feq_2d(Val(2), φ, u, v, usq)
        ge3 = feq_2d(Val(3), φ, u, v, usq)
        ge4 = feq_2d(Val(4), φ, u, v, usq)
        ge5 = feq_2d(Val(5), φ, u, v, usq)
        ge6 = feq_2d(Val(6), φ, u, v, usq)
        ge7 = feq_2d(Val(7), φ, u, v, usq)
        ge8 = feq_2d(Val(8), φ, u, v, usq)
        ge9 = feq_2d(Val(9), φ, u, v, usq)

        ωp = one(T) / T(tau_plus)
        ωm = one(T) / T(tau_minus)
        half = T(0.5)
        wr = T(4/9); wa = T(1/9); we = T(1/36)
        source_linear = (one(T) - ωp * half) * T(3) * S

        S1 = wr * S
        S2 = wa * (S + source_linear*u)
        S3 = wa * (S + source_linear*v)
        S4 = wa * (S - source_linear*u)
        S5 = wa * (S - source_linear*v)
        S6 = we * (S + source_linear*(u + v))
        S7 = we * (S + source_linear*(-u + v))
        S8 = we * (S - source_linear*(u + v))
        S9 = we * (S + source_linear*(u - v))

        g[i,j,1] = g1 - ωp * (g1 - ge1) + S1

        gp24 = (g2 + g4) * half;  gm24 = (g2 - g4) * half
        ep24 = (ge2 + ge4) * half; em24 = (ge2 - ge4) * half
        g[i,j,2] = g2 - ωp*(gp24 - ep24) - ωm*(gm24 - em24) + S2
        g[i,j,4] = g4 - ωp*(gp24 - ep24) - ωm*(-(gm24 - em24)) + S4

        gp35 = (g3 + g5) * half;  gm35 = (g3 - g5) * half
        ep35 = (ge3 + ge5) * half; em35 = (ge3 - ge5) * half
        g[i,j,3] = g3 - ωp*(gp35 - ep35) - ωm*(gm35 - em35) + S3
        g[i,j,5] = g5 - ωp*(gp35 - ep35) - ωm*(-(gm35 - em35)) + S5

        gp68 = (g6 + g8) * half;  gm68 = (g6 - g8) * half
        ep68 = (ge6 + ge8) * half; em68 = (ge6 - ge8) * half
        g[i,j,6] = g6 - ωp*(gp68 - ep68) - ωm*(gm68 - em68) + S6
        g[i,j,8] = g8 - ωp*(gp68 - ep68) - ωm*(-(gm68 - em68)) + S8

        gp79 = (g7 + g9) * half;  gm79 = (g7 - g9) * half
        ep79 = (ge7 + ge9) * half; em79 = (ge7 - ge9) * half
        g[i,j,7] = g7 - ωp*(gp79 - ep79) - ωm*(gm79 - em79) + S7
        g[i,j,9] = g9 - ωp*(gp79 - ep79) - ωm*(-(gm79 - em79)) + S9
    end
end

function collide_conformation_2d_with_gradient_stencils!(
        g, C_field, ux, uy, C_xx, C_xy, C_yy, is_solid,
        uwx, uwy, stencils::ConformationGradientStencils2D,
        tau_plus, lambda; magic=0.25, component=1,
        divergence_mode::Symbol=:numerical)
    backend = KernelAbstractions.get_backend(g)
    Nx, Ny = size(C_field)
    T = eltype(g)
    tau_minus = magic / (tau_plus - 0.5) + 0.5
    kernel! = collide_conformation_2d_gradient_stencils_kernel!(backend)
    kernel!(g, C_field, ux, uy, C_xx, C_xy, C_yy, is_solid,
            uwx, uwy,
            stencils.coeff, stencils.di, stencils.dj,
            stencils.wall_i, stencils.wall_j, stencils.wall_q,
            stencils.is_wall, stencils.count,
            T(tau_plus), T(tau_minus), T(lambda),
            Int(component), _conformation_divergence_mode_code(divergence_mode),
            Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

# =====================================================================
# Liu/Yu regularized TRT collision for the conformation CDE.
# This follows the moment reconstruction form of Liu Eq. (26): post-state is
# rebuilt from B₀ⁿᵉᵠ, Bαⁿᵉᵠ and Bαβⁿᵉᵠ instead of relaxing all population
# degrees of freedom directly. This matters when τp,1 is close to 0.5
# (high-Schmidt, low artificial diffusion).
# =====================================================================

@kernel function collide_conformation_regularized_2d_kernel!(g, @Const(C_field), @Const(ux), @Const(uy),
                                                               @Const(C_xx_f), @Const(C_xy_f), @Const(C_yy_f),
                                                               @Const(is_solid),
                                                               tau_plus, tau_minus, lambda,
                                                               component, divergence_mode, Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds if !is_solid[i, j]
        T = eltype(g)
        φ = C_field[i, j]
        u = ux[i, j]
        v = uy[i, j]
        usq = u*u + v*v

        dudx = _wall_aware_dx_2d(ux, is_solid, i, j, Nx, T)
        dvdx = _wall_aware_dx_2d(uy, is_solid, i, j, Nx, T)
        dudy = _wall_aware_dy_2d(ux, is_solid, i, j, Ny, T)
        dvdy = _wall_aware_dy_2d(uy, is_solid, i, j, Ny, T)
        raw_dudx = dudx
        raw_dvdy = dvdy
        dudx, dudy, dvdx, dvdy = _apply_conformation_divergence_mode_2d(
            dudx, dudy, dvdx, dvdy, divergence_mode)
        advective_divu = _conformation_advective_divergence_2d(
            raw_dudx, raw_dvdy, dudx, dvdy, divergence_mode)

        cxx = C_xx_f[i, j]; cxy = C_xy_f[i, j]; cyy = C_yy_f[i, j]
        S = conformation_source_with_divergence_2d(
            cxx, cxy, cyy, dudx, dudy, dvdx, dvdy,
            advective_divu, T(lambda), component)

        g1 = g[i,j,1]; g2 = g[i,j,2]; g3 = g[i,j,3]; g4 = g[i,j,4]; g5 = g[i,j,5]
        g6 = g[i,j,6]; g7 = g[i,j,7]; g8 = g[i,j,8]; g9 = g[i,j,9]

        ge1 = feq_2d(Val(1), φ, u, v, usq)
        ge2 = feq_2d(Val(2), φ, u, v, usq)
        ge3 = feq_2d(Val(3), φ, u, v, usq)
        ge4 = feq_2d(Val(4), φ, u, v, usq)
        ge5 = feq_2d(Val(5), φ, u, v, usq)
        ge6 = feq_2d(Val(6), φ, u, v, usq)
        ge7 = feq_2d(Val(7), φ, u, v, usq)
        ge8 = feq_2d(Val(8), φ, u, v, usq)
        ge9 = feq_2d(Val(9), φ, u, v, usq)

        d1 = g1 - ge1; d2 = g2 - ge2; d3 = g3 - ge3
        d4 = g4 - ge4; d5 = g5 - ge5; d6 = g6 - ge6
        d7 = g7 - ge7; d8 = g8 - ge8; d9 = g9 - ge9

        B0 = d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8 + d9
        Bx = d2 - d4 + d6 - d7 - d8 + d9
        By = d3 - d5 + d6 + d7 - d8 - d9
        cs2 = T(1/3)
        H0 = -cs2
        H1 = one(T) - cs2
        Bxx = H0*(d1 + d3 + d5) + H1*(d2 + d4 + d6 + d7 + d8 + d9)
        Byy = H0*(d1 + d2 + d4) + H1*(d3 + d5 + d6 + d7 + d8 + d9)
        Bxy = d6 - d7 + d8 - d9

        ωp = one(T) / T(tau_plus)
        ωm = one(T) / T(tau_minus)
        r1 = one(T) - ωp
        r2 = one(T) - ωm
        wr = T(4/9); wa = T(1/9); we = T(1/36)
        hpre = T(9/2)
        source_linear = (one(T) - ωp * T(0.5)) * T(3) * S

        R1 = r1 * B0 + r2 * hpre * (H0*Bxx + H0*Byy)
        R2 = r1 * (B0 + T(3)*Bx) + r2 * hpre * (H1*Bxx + H0*Byy)
        R3 = r1 * (B0 + T(3)*By) + r2 * hpre * (H0*Bxx + H1*Byy)
        R4 = r1 * (B0 - T(3)*Bx) + r2 * hpre * (H1*Bxx + H0*Byy)
        R5 = r1 * (B0 - T(3)*By) + r2 * hpre * (H0*Bxx + H1*Byy)
        R6 = r1 * (B0 + T(3)*(Bx + By)) + r2 * hpre * (H1*Bxx + T(2)*Bxy + H1*Byy)
        R7 = r1 * (B0 + T(3)*(-Bx + By)) + r2 * hpre * (H1*Bxx - T(2)*Bxy + H1*Byy)
        R8 = r1 * (B0 - T(3)*(Bx + By)) + r2 * hpre * (H1*Bxx + T(2)*Bxy + H1*Byy)
        R9 = r1 * (B0 + T(3)*(Bx - By)) + r2 * hpre * (H1*Bxx - T(2)*Bxy + H1*Byy)

        g[i,j,1] = ge1 + wr*R1 + wr*S
        g[i,j,2] = ge2 + wa*R2 + wa*(S + source_linear*u)
        g[i,j,3] = ge3 + wa*R3 + wa*(S + source_linear*v)
        g[i,j,4] = ge4 + wa*R4 + wa*(S - source_linear*u)
        g[i,j,5] = ge5 + wa*R5 + wa*(S - source_linear*v)
        g[i,j,6] = ge6 + we*R6 + we*(S + source_linear*(u + v))
        g[i,j,7] = ge7 + we*R7 + we*(S + source_linear*(-u + v))
        g[i,j,8] = ge8 + we*R8 + we*(S - source_linear*(u + v))
        g[i,j,9] = ge9 + we*R9 + we*(S + source_linear*(u - v))
    end
end

"""
    collide_conformation_regularized_2d!(...)

Regularized TRT-LBM collision for one conformation component, matching the
moment-reconstruction structure of Liu et al. Eq. (26).
"""
function collide_conformation_regularized_2d!(g, C_field, ux, uy, C_xx, C_xy, C_yy, is_solid,
                                                tau_plus, lambda; magic=0.25, component=1,
                                                divergence_mode::Symbol=:numerical)
    backend = KernelAbstractions.get_backend(g)
    Nx, Ny = size(C_field)
    T = eltype(g)
    tau_minus = magic / (tau_plus - 0.5) + 0.5
    kernel! = collide_conformation_regularized_2d_kernel!(backend)
    kernel!(g, C_field, ux, uy, C_xx, C_xy, C_yy, is_solid,
            T(tau_plus), T(tau_minus), T(lambda),
            Int(component), _conformation_divergence_mode_code(divergence_mode),
            Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

@kernel function collide_conformation_regularized_2d_gradient_stencils_kernel!(
        g, @Const(C_field), @Const(ux), @Const(uy),
        @Const(C_xx_f), @Const(C_xy_f), @Const(C_yy_f), @Const(is_solid),
        @Const(uwx), @Const(uwy),
        @Const(grad_coeff), @Const(grad_di), @Const(grad_dj),
        @Const(grad_wall_i), @Const(grad_wall_j), @Const(grad_wall_q),
        @Const(grad_is_wall), @Const(grad_count),
        tau_plus, tau_minus, lambda, component, divergence_mode, Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds if !is_solid[i, j]
        T = eltype(g)
        φ = C_field[i, j]
        u = ux[i, j]
        v = uy[i, j]
        usq = u*u + v*v

        dudx = _conformation_gradient_from_stencil_arrays_2d(
            ux, uwx, grad_coeff, grad_di, grad_dj,
            grad_wall_i, grad_wall_j, grad_wall_q,
            grad_is_wall, grad_count, i, j, 1, T,
        )
        dudy = _conformation_gradient_from_stencil_arrays_2d(
            ux, uwx, grad_coeff, grad_di, grad_dj,
            grad_wall_i, grad_wall_j, grad_wall_q,
            grad_is_wall, grad_count, i, j, 2, T,
        )
        dvdx = _conformation_gradient_from_stencil_arrays_2d(
            uy, uwy, grad_coeff, grad_di, grad_dj,
            grad_wall_i, grad_wall_j, grad_wall_q,
            grad_is_wall, grad_count, i, j, 1, T,
        )
        dvdy = _conformation_gradient_from_stencil_arrays_2d(
            uy, uwy, grad_coeff, grad_di, grad_dj,
            grad_wall_i, grad_wall_j, grad_wall_q,
            grad_is_wall, grad_count, i, j, 2, T,
        )
        raw_dudx = dudx
        raw_dvdy = dvdy
        dudx, dudy, dvdx, dvdy = _apply_conformation_divergence_mode_2d(
            dudx, dudy, dvdx, dvdy, divergence_mode)
        advective_divu = _conformation_advective_divergence_2d(
            raw_dudx, raw_dvdy, dudx, dvdy, divergence_mode)

        cxx = C_xx_f[i, j]; cxy = C_xy_f[i, j]; cyy = C_yy_f[i, j]
        S = conformation_source_with_divergence_2d(
            cxx, cxy, cyy, dudx, dudy, dvdx, dvdy,
            advective_divu, T(lambda), component)

        g1 = g[i,j,1]; g2 = g[i,j,2]; g3 = g[i,j,3]; g4 = g[i,j,4]; g5 = g[i,j,5]
        g6 = g[i,j,6]; g7 = g[i,j,7]; g8 = g[i,j,8]; g9 = g[i,j,9]

        ge1 = feq_2d(Val(1), φ, u, v, usq)
        ge2 = feq_2d(Val(2), φ, u, v, usq)
        ge3 = feq_2d(Val(3), φ, u, v, usq)
        ge4 = feq_2d(Val(4), φ, u, v, usq)
        ge5 = feq_2d(Val(5), φ, u, v, usq)
        ge6 = feq_2d(Val(6), φ, u, v, usq)
        ge7 = feq_2d(Val(7), φ, u, v, usq)
        ge8 = feq_2d(Val(8), φ, u, v, usq)
        ge9 = feq_2d(Val(9), φ, u, v, usq)

        d1 = g1 - ge1; d2 = g2 - ge2; d3 = g3 - ge3
        d4 = g4 - ge4; d5 = g5 - ge5; d6 = g6 - ge6
        d7 = g7 - ge7; d8 = g8 - ge8; d9 = g9 - ge9

        B0 = d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8 + d9
        Bx = d2 - d4 + d6 - d7 - d8 + d9
        By = d3 - d5 + d6 + d7 - d8 - d9
        cs2 = T(1/3)
        H0 = -cs2
        H1 = one(T) - cs2
        Bxx = H0*(d1 + d3 + d5) + H1*(d2 + d4 + d6 + d7 + d8 + d9)
        Byy = H0*(d1 + d2 + d4) + H1*(d3 + d5 + d6 + d7 + d8 + d9)
        Bxy = d6 - d7 + d8 - d9

        ωp = one(T) / T(tau_plus)
        ωm = one(T) / T(tau_minus)
        r1 = one(T) - ωp
        r2 = one(T) - ωm
        wr = T(4/9); wa = T(1/9); we = T(1/36)
        hpre = T(9/2)
        source_linear = (one(T) - ωp * T(0.5)) * T(3) * S

        R1 = r1 * B0 + r2 * hpre * (H0*Bxx + H0*Byy)
        R2 = r1 * (B0 + T(3)*Bx) + r2 * hpre * (H1*Bxx + H0*Byy)
        R3 = r1 * (B0 + T(3)*By) + r2 * hpre * (H0*Bxx + H1*Byy)
        R4 = r1 * (B0 - T(3)*Bx) + r2 * hpre * (H1*Bxx + H0*Byy)
        R5 = r1 * (B0 - T(3)*By) + r2 * hpre * (H0*Bxx + H1*Byy)
        R6 = r1 * (B0 + T(3)*(Bx + By)) + r2 * hpre * (H1*Bxx + T(2)*Bxy + H1*Byy)
        R7 = r1 * (B0 + T(3)*(-Bx + By)) + r2 * hpre * (H1*Bxx - T(2)*Bxy + H1*Byy)
        R8 = r1 * (B0 - T(3)*(Bx + By)) + r2 * hpre * (H1*Bxx + T(2)*Bxy + H1*Byy)
        R9 = r1 * (B0 + T(3)*(Bx - By)) + r2 * hpre * (H1*Bxx - T(2)*Bxy + H1*Byy)

        g[i,j,1] = ge1 + wr*R1 + wr*S
        g[i,j,2] = ge2 + wa*R2 + wa*(S + source_linear*u)
        g[i,j,3] = ge3 + wa*R3 + wa*(S + source_linear*v)
        g[i,j,4] = ge4 + wa*R4 + wa*(S - source_linear*u)
        g[i,j,5] = ge5 + wa*R5 + wa*(S - source_linear*v)
        g[i,j,6] = ge6 + we*R6 + we*(S + source_linear*(u + v))
        g[i,j,7] = ge7 + we*R7 + we*(S + source_linear*(-u + v))
        g[i,j,8] = ge8 + we*R8 + we*(S - source_linear*(u + v))
        g[i,j,9] = ge9 + we*R9 + we*(S + source_linear*(u - v))
    end
end

function collide_conformation_regularized_2d_with_gradient_stencils!(
        g, C_field, ux, uy, C_xx, C_xy, C_yy, is_solid,
        uwx, uwy, stencils::ConformationGradientStencils2D,
        tau_plus, lambda; magic=0.25, component=1,
        divergence_mode::Symbol=:numerical)
    backend = KernelAbstractions.get_backend(g)
    Nx, Ny = size(C_field)
    T = eltype(g)
    tau_minus = magic / (tau_plus - 0.5) + 0.5
    kernel! = collide_conformation_regularized_2d_gradient_stencils_kernel!(backend)
    kernel!(g, C_field, ux, uy, C_xx, C_xy, C_yy, is_solid,
            uwx, uwy,
            stencils.coeff, stencils.di, stencils.dj,
            stencils.wall_i, stencils.wall_j, stencils.wall_q,
            stencils.is_wall, stencils.count,
            T(tau_plus), T(tau_minus), T(lambda),
            Int(component), _conformation_divergence_mode_code(divergence_mode),
            Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

# =====================================================================
# Liu Eq. (26) diagnostic collision.
#
# This extends the regularized reconstruction with the two source terms that
# are explicit in Liu Eq. (26) but absent from the simpler diagnostic path:
#   - Ge_i, restricted here to the density-gradient part of Eq. (31)
#     because the cylinder validation has no external body force Fα.
#   - 0.5 ∂t Fe_i, using Eq. (36) with per-population source history.
# =====================================================================

@kernel function collide_conformation_liu_eq26_2d_kernel!(g, Fe_prev,
                                                            @Const(C_field), @Const(ux), @Const(uy),
                                                            @Const(ρ), @Const(C_xx_f), @Const(C_xy_f),
                                                            @Const(C_yy_f), @Const(is_solid),
                                                            tau_plus, tau_minus, lambda,
                                                            bneq_source_scale,
                                                            bneq_mass_scale,
                                                            bneq_second_moment_raw,
                                                            component, divergence_mode, Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds if !is_solid[i, j]
        T = eltype(g)
        φ = C_field[i, j]
        u = ux[i, j]
        v = uy[i, j]
        usq = u*u + v*v

        dudx = _wall_aware_dx_2d(ux, is_solid, i, j, Nx, T)
        dvdx = _wall_aware_dx_2d(uy, is_solid, i, j, Nx, T)
        dudy = _wall_aware_dy_2d(ux, is_solid, i, j, Ny, T)
        dvdy = _wall_aware_dy_2d(uy, is_solid, i, j, Ny, T)
        raw_dudx = dudx
        raw_dvdy = dvdy
        dudx, dudy, dvdx, dvdy = _apply_conformation_divergence_mode_2d(
            dudx, dudy, dvdx, dvdy, divergence_mode)
        advective_divu = _conformation_advective_divergence_2d(
            raw_dudx, raw_dvdy, dudx, dvdy, divergence_mode)

        cxx = C_xx_f[i, j]; cxy = C_xy_f[i, j]; cyy = C_yy_f[i, j]
        S = conformation_source_with_divergence_2d(
            cxx, cxy, cyy, dudx, dudy, dvdx, dvdy,
            advective_divu, T(lambda), component)

        ωp = one(T) / T(tau_plus)
        ωm = one(T) / T(tau_minus)
        r1 = one(T) - ωp
        r2 = one(T) - ωm
        coeff = one(T) - ωp * T(0.5)
        wr = T(4/9); wa = T(1/9); we = T(1/36)
        hpre = T(9/2)
        source_linear = coeff * T(3) * S

        F1 = wr * S
        F2 = wa * (S + source_linear * u)
        F3 = wa * (S + source_linear * v)
        F4 = wa * (S - source_linear * u)
        F5 = wa * (S - source_linear * v)
        F6 = we * (S + source_linear * (u + v))
        F7 = we * (S + source_linear * (-u + v))
        F8 = we * (S - source_linear * (u + v))
        F9 = we * (S + source_linear * (u - v))

        ge1 = feq_2d(Val(1), φ, u, v, usq)
        ge2 = feq_2d(Val(2), φ, u, v, usq)
        ge3 = feq_2d(Val(3), φ, u, v, usq)
        ge4 = feq_2d(Val(4), φ, u, v, usq)
        ge5 = feq_2d(Val(5), φ, u, v, usq)
        ge6 = feq_2d(Val(6), φ, u, v, usq)
        ge7 = feq_2d(Val(7), φ, u, v, usq)
        ge8 = feq_2d(Val(8), φ, u, v, usq)
        ge9 = feq_2d(Val(9), φ, u, v, usq)

        bscale = T(bneq_source_scale)
        d1 = g[i,j,1] - ge1 + bscale * F1
        d2 = g[i,j,2] - ge2 + bscale * F2
        d3 = g[i,j,3] - ge3 + bscale * F3
        d4 = g[i,j,4] - ge4 + bscale * F4
        d5 = g[i,j,5] - ge5 + bscale * F5
        d6 = g[i,j,6] - ge6 + bscale * F6
        d7 = g[i,j,7] - ge7 + bscale * F7
        d8 = g[i,j,8] - ge8 + bscale * F8
        d9 = g[i,j,9] - ge9 + bscale * F9

        B0 = T(bneq_mass_scale) * (d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8 + d9)
        Bx = d2 - d4 + d6 - d7 - d8 + d9
        By = d3 - d5 + d6 + d7 - d8 - d9
        cs2 = T(1/3)
        H0 = -cs2
        H1 = one(T) - cs2
        Bxx_h = H0*(d1 + d3 + d5) + H1*(d2 + d4 + d6 + d7 + d8 + d9)
        Byy_h = H0*(d1 + d2 + d4) + H1*(d3 + d5 + d6 + d7 + d8 + d9)
        Bxx_r = d2 + d4 + d6 + d7 + d8 + d9
        Byy_r = d3 + d5 + d6 + d7 + d8 + d9
        Bxx = bneq_second_moment_raw == 1 ? Bxx_r : Bxx_h
        Byy = bneq_second_moment_raw == 1 ? Byy_r : Byy_h
        Bxy = d6 - d7 + d8 - d9

        R1 = r1 * B0 + r2 * hpre * (H0*Bxx + H0*Byy)
        R2 = r1 * (B0 + T(3)*Bx) + r2 * hpre * (H1*Bxx + H0*Byy)
        R3 = r1 * (B0 + T(3)*By) + r2 * hpre * (H0*Bxx + H1*Byy)
        R4 = r1 * (B0 - T(3)*Bx) + r2 * hpre * (H1*Bxx + H0*Byy)
        R5 = r1 * (B0 - T(3)*By) + r2 * hpre * (H0*Bxx + H1*Byy)
        R6 = r1 * (B0 + T(3)*(Bx + By)) + r2 * hpre * (H1*Bxx + T(2)*Bxy + H1*Byy)
        R7 = r1 * (B0 + T(3)*(-Bx + By)) + r2 * hpre * (H1*Bxx - T(2)*Bxy + H1*Byy)
        R8 = r1 * (B0 - T(3)*(Bx + By)) + r2 * hpre * (H1*Bxx + T(2)*Bxy + H1*Byy)
        R9 = r1 * (B0 + T(3)*(Bx - By)) + r2 * hpre * (H1*Bxx - T(2)*Bxy + H1*Byy)

        drhodx = _wall_aware_dx_2d(ρ, is_solid, i, j, Nx, T)
        drhody = _wall_aware_dy_2d(ρ, is_solid, i, j, Ny, T)
        invρ = one(T) / max(ρ[i, j], eps(T))
        ge_coeff = -coeff * φ * invρ
        G1 = zero(T)
        G2 = wa * ge_coeff * drhodx
        G3 = wa * ge_coeff * drhody
        G4 = -wa * ge_coeff * drhodx
        G5 = -wa * ge_coeff * drhody
        G6 = we * ge_coeff * (drhodx + drhody)
        G7 = we * ge_coeff * (-drhodx + drhody)
        G8 = -we * ge_coeff * (drhodx + drhody)
        G9 = we * ge_coeff * (drhodx - drhody)

        oldF1 = Fe_prev[i,j,1]; oldF2 = Fe_prev[i,j,2]; oldF3 = Fe_prev[i,j,3]
        oldF4 = Fe_prev[i,j,4]; oldF5 = Fe_prev[i,j,5]; oldF6 = Fe_prev[i,j,6]
        oldF7 = Fe_prev[i,j,7]; oldF8 = Fe_prev[i,j,8]; oldF9 = Fe_prev[i,j,9]

        g[i,j,1] = ge1 + wr*R1 + G1 + F1 + T(0.5)*(F1 - oldF1)
        g[i,j,2] = ge2 + wa*R2 + G2 + F2 + T(0.5)*(F2 - oldF2)
        g[i,j,3] = ge3 + wa*R3 + G3 + F3 + T(0.5)*(F3 - oldF3)
        g[i,j,4] = ge4 + wa*R4 + G4 + F4 + T(0.5)*(F4 - oldF4)
        g[i,j,5] = ge5 + wa*R5 + G5 + F5 + T(0.5)*(F5 - oldF5)
        g[i,j,6] = ge6 + we*R6 + G6 + F6 + T(0.5)*(F6 - oldF6)
        g[i,j,7] = ge7 + we*R7 + G7 + F7 + T(0.5)*(F7 - oldF7)
        g[i,j,8] = ge8 + we*R8 + G8 + F8 + T(0.5)*(F8 - oldF8)
        g[i,j,9] = ge9 + we*R9 + G9 + F9 + T(0.5)*(F9 - oldF9)

        Fe_prev[i,j,1] = F1; Fe_prev[i,j,2] = F2; Fe_prev[i,j,3] = F3
        Fe_prev[i,j,4] = F4; Fe_prev[i,j,5] = F5; Fe_prev[i,j,6] = F6
        Fe_prev[i,j,7] = F7; Fe_prev[i,j,8] = F8; Fe_prev[i,j,9] = F9
    end
end

function collide_conformation_liu_eq26_2d!(g, Fe_prev, C_field, ux, uy, ρ,
                                            C_xx, C_xy, C_yy, is_solid,
                                            tau_plus, lambda; magic=0.25,
                                            bneq_source_scale=0.0,
                                            bneq_mass_scale=1.0,
                                            bneq_second_moment_raw=false,
                                            component=1,
                                            divergence_mode::Symbol=:numerical)
    backend = KernelAbstractions.get_backend(g)
    Nx, Ny = size(C_field)
    T = eltype(g)
    tau_minus = magic / (tau_plus - 0.5) + 0.5
    kernel! = collide_conformation_liu_eq26_2d_kernel!(backend)
    kernel!(g, Fe_prev, C_field, ux, uy, ρ, C_xx, C_xy, C_yy, is_solid,
            T(tau_plus), T(tau_minus), T(lambda),
            T(bneq_source_scale), T(bneq_mass_scale),
            bneq_second_moment_raw ? 1 : 0,
            Int(component), _conformation_divergence_mode_code(divergence_mode),
            Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

@kernel function collide_conformation_liu_eq26_2d_gradient_stencils_kernel!(
        g, Fe_prev, @Const(C_field), @Const(ux), @Const(uy),
        @Const(ρ), @Const(C_xx_f), @Const(C_xy_f), @Const(C_yy_f),
        @Const(is_solid), @Const(uwx), @Const(uwy),
        @Const(grad_coeff), @Const(grad_di), @Const(grad_dj),
        @Const(grad_wall_i), @Const(grad_wall_j), @Const(grad_wall_q),
        @Const(grad_is_wall), @Const(grad_count),
        tau_plus, tau_minus, lambda, bneq_source_scale,
        bneq_mass_scale, bneq_second_moment_raw,
        component, divergence_mode, Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds if !is_solid[i, j]
        T = eltype(g)
        φ = C_field[i, j]
        u = ux[i, j]
        v = uy[i, j]
        usq = u*u + v*v

        dudx = _conformation_gradient_from_stencil_arrays_2d(
            ux, uwx, grad_coeff, grad_di, grad_dj,
            grad_wall_i, grad_wall_j, grad_wall_q,
            grad_is_wall, grad_count, i, j, 1, T,
        )
        dudy = _conformation_gradient_from_stencil_arrays_2d(
            ux, uwx, grad_coeff, grad_di, grad_dj,
            grad_wall_i, grad_wall_j, grad_wall_q,
            grad_is_wall, grad_count, i, j, 2, T,
        )
        dvdx = _conformation_gradient_from_stencil_arrays_2d(
            uy, uwy, grad_coeff, grad_di, grad_dj,
            grad_wall_i, grad_wall_j, grad_wall_q,
            grad_is_wall, grad_count, i, j, 1, T,
        )
        dvdy = _conformation_gradient_from_stencil_arrays_2d(
            uy, uwy, grad_coeff, grad_di, grad_dj,
            grad_wall_i, grad_wall_j, grad_wall_q,
            grad_is_wall, grad_count, i, j, 2, T,
        )
        raw_dudx = dudx
        raw_dvdy = dvdy
        dudx, dudy, dvdx, dvdy = _apply_conformation_divergence_mode_2d(
            dudx, dudy, dvdx, dvdy, divergence_mode)
        advective_divu = _conformation_advective_divergence_2d(
            raw_dudx, raw_dvdy, dudx, dvdy, divergence_mode)

        cxx = C_xx_f[i, j]; cxy = C_xy_f[i, j]; cyy = C_yy_f[i, j]
        S = conformation_source_with_divergence_2d(
            cxx, cxy, cyy, dudx, dudy, dvdx, dvdy,
            advective_divu, T(lambda), component)

        ωp = one(T) / T(tau_plus)
        ωm = one(T) / T(tau_minus)
        r1 = one(T) - ωp
        r2 = one(T) - ωm
        coeff = one(T) - ωp * T(0.5)
        wr = T(4/9); wa = T(1/9); we = T(1/36)
        hpre = T(9/2)
        source_linear = coeff * T(3) * S

        F1 = wr * S
        F2 = wa * (S + source_linear * u)
        F3 = wa * (S + source_linear * v)
        F4 = wa * (S - source_linear * u)
        F5 = wa * (S - source_linear * v)
        F6 = we * (S + source_linear * (u + v))
        F7 = we * (S + source_linear * (-u + v))
        F8 = we * (S - source_linear * (u + v))
        F9 = we * (S + source_linear * (u - v))

        ge1 = feq_2d(Val(1), φ, u, v, usq)
        ge2 = feq_2d(Val(2), φ, u, v, usq)
        ge3 = feq_2d(Val(3), φ, u, v, usq)
        ge4 = feq_2d(Val(4), φ, u, v, usq)
        ge5 = feq_2d(Val(5), φ, u, v, usq)
        ge6 = feq_2d(Val(6), φ, u, v, usq)
        ge7 = feq_2d(Val(7), φ, u, v, usq)
        ge8 = feq_2d(Val(8), φ, u, v, usq)
        ge9 = feq_2d(Val(9), φ, u, v, usq)

        bscale = T(bneq_source_scale)
        d1 = g[i,j,1] - ge1 + bscale * F1
        d2 = g[i,j,2] - ge2 + bscale * F2
        d3 = g[i,j,3] - ge3 + bscale * F3
        d4 = g[i,j,4] - ge4 + bscale * F4
        d5 = g[i,j,5] - ge5 + bscale * F5
        d6 = g[i,j,6] - ge6 + bscale * F6
        d7 = g[i,j,7] - ge7 + bscale * F7
        d8 = g[i,j,8] - ge8 + bscale * F8
        d9 = g[i,j,9] - ge9 + bscale * F9

        B0 = T(bneq_mass_scale) * (d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8 + d9)
        Bx = d2 - d4 + d6 - d7 - d8 + d9
        By = d3 - d5 + d6 + d7 - d8 - d9
        cs2 = T(1/3)
        H0 = -cs2
        H1 = one(T) - cs2
        Bxx_h = H0*(d1 + d3 + d5) + H1*(d2 + d4 + d6 + d7 + d8 + d9)
        Byy_h = H0*(d1 + d2 + d4) + H1*(d3 + d5 + d6 + d7 + d8 + d9)
        Bxx_r = d2 + d4 + d6 + d7 + d8 + d9
        Byy_r = d3 + d5 + d6 + d7 + d8 + d9
        Bxx = bneq_second_moment_raw == 1 ? Bxx_r : Bxx_h
        Byy = bneq_second_moment_raw == 1 ? Byy_r : Byy_h
        Bxy = d6 - d7 + d8 - d9

        R1 = r1 * B0 + r2 * hpre * (H0*Bxx + H0*Byy)
        R2 = r1 * (B0 + T(3)*Bx) + r2 * hpre * (H1*Bxx + H0*Byy)
        R3 = r1 * (B0 + T(3)*By) + r2 * hpre * (H0*Bxx + H1*Byy)
        R4 = r1 * (B0 - T(3)*Bx) + r2 * hpre * (H1*Bxx + H0*Byy)
        R5 = r1 * (B0 - T(3)*By) + r2 * hpre * (H0*Bxx + H1*Byy)
        R6 = r1 * (B0 + T(3)*(Bx + By)) + r2 * hpre * (H1*Bxx + T(2)*Bxy + H1*Byy)
        R7 = r1 * (B0 + T(3)*(-Bx + By)) + r2 * hpre * (H1*Bxx - T(2)*Bxy + H1*Byy)
        R8 = r1 * (B0 - T(3)*(Bx + By)) + r2 * hpre * (H1*Bxx + T(2)*Bxy + H1*Byy)
        R9 = r1 * (B0 + T(3)*(Bx - By)) + r2 * hpre * (H1*Bxx - T(2)*Bxy + H1*Byy)

        drhodx = _wall_aware_dx_2d(ρ, is_solid, i, j, Nx, T)
        drhody = _wall_aware_dy_2d(ρ, is_solid, i, j, Ny, T)
        invρ = one(T) / max(ρ[i, j], eps(T))
        ge_coeff = -coeff * φ * invρ
        G1 = zero(T)
        G2 = wa * ge_coeff * drhodx
        G3 = wa * ge_coeff * drhody
        G4 = -wa * ge_coeff * drhodx
        G5 = -wa * ge_coeff * drhody
        G6 = we * ge_coeff * (drhodx + drhody)
        G7 = we * ge_coeff * (-drhodx + drhody)
        G8 = -we * ge_coeff * (drhodx + drhody)
        G9 = we * ge_coeff * (drhodx - drhody)

        oldF1 = Fe_prev[i,j,1]; oldF2 = Fe_prev[i,j,2]; oldF3 = Fe_prev[i,j,3]
        oldF4 = Fe_prev[i,j,4]; oldF5 = Fe_prev[i,j,5]; oldF6 = Fe_prev[i,j,6]
        oldF7 = Fe_prev[i,j,7]; oldF8 = Fe_prev[i,j,8]; oldF9 = Fe_prev[i,j,9]

        g[i,j,1] = ge1 + wr*R1 + G1 + F1 + T(0.5)*(F1 - oldF1)
        g[i,j,2] = ge2 + wa*R2 + G2 + F2 + T(0.5)*(F2 - oldF2)
        g[i,j,3] = ge3 + wa*R3 + G3 + F3 + T(0.5)*(F3 - oldF3)
        g[i,j,4] = ge4 + wa*R4 + G4 + F4 + T(0.5)*(F4 - oldF4)
        g[i,j,5] = ge5 + wa*R5 + G5 + F5 + T(0.5)*(F5 - oldF5)
        g[i,j,6] = ge6 + we*R6 + G6 + F6 + T(0.5)*(F6 - oldF6)
        g[i,j,7] = ge7 + we*R7 + G7 + F7 + T(0.5)*(F7 - oldF7)
        g[i,j,8] = ge8 + we*R8 + G8 + F8 + T(0.5)*(F8 - oldF8)
        g[i,j,9] = ge9 + we*R9 + G9 + F9 + T(0.5)*(F9 - oldF9)

        Fe_prev[i,j,1] = F1; Fe_prev[i,j,2] = F2; Fe_prev[i,j,3] = F3
        Fe_prev[i,j,4] = F4; Fe_prev[i,j,5] = F5; Fe_prev[i,j,6] = F6
        Fe_prev[i,j,7] = F7; Fe_prev[i,j,8] = F8; Fe_prev[i,j,9] = F9
    end
end

function collide_conformation_liu_eq26_2d_with_gradient_stencils!(
        g, Fe_prev, C_field, ux, uy, ρ, C_xx, C_xy, C_yy, is_solid,
        uwx, uwy, stencils::ConformationGradientStencils2D,
        tau_plus, lambda; magic=0.25,
        bneq_source_scale=0.0,
        bneq_mass_scale=1.0,
        bneq_second_moment_raw=false,
        component=1,
        divergence_mode::Symbol=:numerical)
    backend = KernelAbstractions.get_backend(g)
    Nx, Ny = size(C_field)
    T = eltype(g)
    tau_minus = magic / (tau_plus - 0.5) + 0.5
    kernel! = collide_conformation_liu_eq26_2d_gradient_stencils_kernel!(backend)
    kernel!(g, Fe_prev, C_field, ux, uy, ρ, C_xx, C_xy, C_yy, is_solid,
            uwx, uwy,
            stencils.coeff, stencils.di, stencils.dj,
            stencils.wall_i, stencils.wall_j, stencils.wall_q,
            stencils.is_wall, stencils.count,
            T(tau_plus), T(tau_minus), T(lambda),
            T(bneq_source_scale), T(bneq_mass_scale),
            bneq_second_moment_raw ? 1 : 0,
            Int(component), _conformation_divergence_mode_code(divergence_mode),
            Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

# ============================================================
# Initialization: g = g^eq with given C field and velocity field
# ============================================================

@kernel function init_conformation_field_2d_kernel!(g, @Const(C_field), @Const(ux), @Const(uy))
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(g)
        φ = C_field[i, j]
        u = ux[i, j]
        v = uy[i, j]
        usq = u*u + v*v
        g[i,j,1] = feq_2d(Val(1), φ, u, v, usq)
        g[i,j,2] = feq_2d(Val(2), φ, u, v, usq)
        g[i,j,3] = feq_2d(Val(3), φ, u, v, usq)
        g[i,j,4] = feq_2d(Val(4), φ, u, v, usq)
        g[i,j,5] = feq_2d(Val(5), φ, u, v, usq)
        g[i,j,6] = feq_2d(Val(6), φ, u, v, usq)
        g[i,j,7] = feq_2d(Val(7), φ, u, v, usq)
        g[i,j,8] = feq_2d(Val(8), φ, u, v, usq)
        g[i,j,9] = feq_2d(Val(9), φ, u, v, usq)
    end
end

function init_conformation_field_2d!(g, C_field, ux, uy)
    backend = KernelAbstractions.get_backend(g)
    Nx, Ny = size(C_field)
    kernel! = init_conformation_field_2d_kernel!(backend)
    kernel!(g, C_field, ux, uy; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

# ============================================================
# Macroscopic recovery: φ = sum_q g_q
# ============================================================

@kernel function compute_conformation_macro_2d_kernel!(C_field, @Const(g))
    i, j = @index(Global, NTuple)
    @inbounds begin
        C_field[i,j] = g[i,j,1] + g[i,j,2] + g[i,j,3] + g[i,j,4] + g[i,j,5] +
                       g[i,j,6] + g[i,j,7] + g[i,j,8] + g[i,j,9]
    end
end

function compute_conformation_macro_2d!(C_field, g)
    backend = KernelAbstractions.get_backend(g)
    Nx, Ny = size(C_field)
    kernel! = compute_conformation_macro_2d_kernel!(backend)
    kernel!(C_field, g; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

@kernel function apply_conformation_macro_source_shift_2d_kernel!(C_xx, C_xy, C_yy,
                                                                   @Const(ux), @Const(uy),
                                                                   @Const(is_solid),
                                                                   lambda, source_shift,
                                                                   Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds if !is_solid[i, j]
        T = eltype(C_xx)
        cxx = C_xx[i, j]
        cxy = C_xy[i, j]
        cyy = C_yy[i, j]

        dudx = _wall_aware_dx_2d(ux, is_solid, i, j, Nx, T)
        dvdx = _wall_aware_dx_2d(uy, is_solid, i, j, Nx, T)
        dudy = _wall_aware_dy_2d(ux, is_solid, i, j, Ny, T)
        dvdy = _wall_aware_dy_2d(uy, is_solid, i, j, Ny, T)
        divu = dudx + dvdy
        inv_λ = one(T) / T(lambda)
        shift = T(source_shift)

        S_xx = -inv_λ * (cxx - one(T)) + T(2) * (cxx*dudx + cxy*dudy) + cxx*divu
        S_xy = -inv_λ * cxy + (cxx*dvdx + cyy*dudy + cxy*(dudx + dvdy)) + cxy*divu
        S_yy = -inv_λ * (cyy - one(T)) + T(2) * (cxy*dvdx + cyy*dvdy) + cyy*divu

        C_xx[i, j] = cxx + shift * S_xx
        C_xy[i, j] = cxy + shift * S_xy
        C_yy[i, j] = cyy + shift * S_yy
    end
end

function apply_conformation_macro_source_shift_2d!(C_xx, C_xy, C_yy, ux, uy,
                                                    is_solid, lambda, source_shift)
    backend = KernelAbstractions.get_backend(C_xx)
    Nx, Ny = size(C_xx)
    T = eltype(C_xx)
    kernel! = apply_conformation_macro_source_shift_2d_kernel!(backend)
    kernel!(C_xx, C_xy, C_yy, ux, uy, is_solid, T(lambda), T(source_shift),
            Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

# ============================================================
# Inlet / outlet BC for conformation populations
# ============================================================
# At domain boundaries (west inlet, east outlet), stream_2d! applies
# bounce-back on g-populations which corrupts C. These routines force
# g = f^eq(C_prescribed, u_prescribed) at the boundary nodes, analogous
# to the Zou-He rebuild for the hydrodynamic f.

@kernel function _reset_conf_inlet_kernel!(g, @Const(C_inlet), @Const(u_profile))
    j = @index(Global)
    @inbounds begin
        T = eltype(g)
        φ = C_inlet[j]
        u = u_profile[j]
        usq = u * u
        g[1,j,1] = feq_2d(Val(1), φ, u, zero(T), usq)
        g[1,j,2] = feq_2d(Val(2), φ, u, zero(T), usq)
        g[1,j,3] = feq_2d(Val(3), φ, u, zero(T), usq)
        g[1,j,4] = feq_2d(Val(4), φ, u, zero(T), usq)
        g[1,j,5] = feq_2d(Val(5), φ, u, zero(T), usq)
        g[1,j,6] = feq_2d(Val(6), φ, u, zero(T), usq)
        g[1,j,7] = feq_2d(Val(7), φ, u, zero(T), usq)
        g[1,j,8] = feq_2d(Val(8), φ, u, zero(T), usq)
        g[1,j,9] = feq_2d(Val(9), φ, u, zero(T), usq)
    end
end

@kernel function _reset_conf_inlet_masked_kernel!(g, @Const(C_inlet),
                                                   @Const(u_profile), @Const(mask))
    j = @index(Global)
    @inbounds if mask[j]
        T = eltype(g)
        φ = C_inlet[j]
        u = u_profile[j]
        usq = u * u
        g[1,j,1] = feq_2d(Val(1), φ, u, zero(T), usq)
        g[1,j,2] = feq_2d(Val(2), φ, u, zero(T), usq)
        g[1,j,3] = feq_2d(Val(3), φ, u, zero(T), usq)
        g[1,j,4] = feq_2d(Val(4), φ, u, zero(T), usq)
        g[1,j,5] = feq_2d(Val(5), φ, u, zero(T), usq)
        g[1,j,6] = feq_2d(Val(6), φ, u, zero(T), usq)
        g[1,j,7] = feq_2d(Val(7), φ, u, zero(T), usq)
        g[1,j,8] = feq_2d(Val(8), φ, u, zero(T), usq)
        g[1,j,9] = feq_2d(Val(9), φ, u, zero(T), usq)
    end
end

function reset_conformation_inlet_2d!(g, C_inlet, u_profile, Ny)
    backend = KernelAbstractions.get_backend(g)
    kernel! = _reset_conf_inlet_kernel!(backend)
    kernel!(g, C_inlet, u_profile; ndrange=(Ny,))
    KernelAbstractions.synchronize(backend)
end

function reset_conformation_inlet_masked_2d!(g, C_inlet, u_profile, mask, Ny)
    backend = KernelAbstractions.get_backend(g)
    kernel! = _reset_conf_inlet_masked_kernel!(backend)
    kernel!(g, C_inlet, u_profile, mask; ndrange=(Ny,))
    KernelAbstractions.synchronize(backend)
end

@kernel function _reset_conf_outlet_kernel!(g, Nx)
    j = @index(Global)
    @inbounds begin
        # Zero-gradient: copy from Nx-1 to Nx
        for q in 1:9
            g[Nx, j, q] = g[Nx-1, j, q]
        end
    end
end

@kernel function _reset_conf_outlet_masked_kernel!(g, Nx, @Const(mask))
    j = @index(Global)
    @inbounds if mask[j]
        # Zero-gradient: copy from Nx-1 to Nx
        for q in 1:9
            g[Nx, j, q] = g[Nx-1, j, q]
        end
    end
end

function reset_conformation_outlet_2d!(g, Nx, Ny)
    backend = KernelAbstractions.get_backend(g)
    kernel! = _reset_conf_outlet_kernel!(backend)
    kernel!(g, Nx; ndrange=(Ny,))
    KernelAbstractions.synchronize(backend)
end

function reset_conformation_outlet_masked_2d!(g, Nx, Ny, mask)
    backend = KernelAbstractions.get_backend(g)
    kernel! = _reset_conf_outlet_masked_kernel!(backend)
    kernel!(g, Nx, mask; ndrange=(Ny,))
    KernelAbstractions.synchronize(backend)
end

# ============================================================
# Conservative non-equilibrium bounce-back (CNEBB) — Liu et al. 2025, Eqs (38-39)
# ============================================================
#
# Applied at fluid cells x_b adjacent to a solid neighbour, AFTER streaming.
# Required inputs:
#   - g_post : populations after streaming (some directions invalid because
#              they came from solid neighbours)
#   - g_pre  : populations BEFORE streaming = post-collision of previous step
#              in the same cell. Used to recover the populations that left
#              x_b toward the solid (and were lost during streaming).
#
# Algorithm at each fluid cell with at least one solid neighbour:
#   1. Γ = {q : neighbour(i,j, opp(q)) is fluid}  → g_post[q] is valid
#      H = complement                              → g_post[q] is unknown
#   2. φ = Σ_{q∈Γ} g_post[q] + Σ_{q∈H} g_pre[opp(q)]
#      (the second sum recovers populations that streamed out toward solid)
#   3. g^eq computed with this φ at u_wall = 0 (no-slip)
#   4. For each q ∈ H: g_post[q] = g^eq[q] + (g_post[opp(q)] - g^eq[opp(q)])
#   5. Rest-population rebalance: g_post[1] = φ - Σ_{q≠1} g_post[q]
#   6. Macroscopic update: C_field[i,j] = φ
#
# Convention for stream_2d!: g_post[i,j,q] receives from neighbour at
# (i - cx[q], j - cy[q]). So direction q is "valid" iff that source neighbour
# is fluid.

# D2Q9 opposite indices (Kraken convention from header above):
#   1↔1, 2↔4, 3↔5, 4↔2, 5↔3, 6↔8, 7↔9, 8↔6, 9↔7

@inline _opp_q(q) = q == 1 ? 1 :
                    q == 2 ? 4 : q == 4 ? 2 :
                    q == 3 ? 5 : q == 5 ? 3 :
                    q == 6 ? 8 : q == 8 ? 6 :
                    q == 7 ? 9 : 7

@inline _cx_q(q) = (0, 1, 0, -1, 0,  1, -1, -1,  1)[q]
@inline _cy_q(q) = (0, 0, 1,  0,-1,  1,  1, -1, -1)[q]
@inline _w_q(q) = q == 1 ? 4/9 :
                  (q == 2 || q == 3 || q == 4 || q == 5 ? 1/9 : 1/36)

@inline function _feq_q_2d(q, φ::T, u::T, v::T, usq::T) where {T}
    q == 1 && return feq_2d(Val(1), φ, u, v, usq)
    q == 2 && return feq_2d(Val(2), φ, u, v, usq)
    q == 3 && return feq_2d(Val(3), φ, u, v, usq)
    q == 4 && return feq_2d(Val(4), φ, u, v, usq)
    q == 5 && return feq_2d(Val(5), φ, u, v, usq)
    q == 6 && return feq_2d(Val(6), φ, u, v, usq)
    q == 7 && return feq_2d(Val(7), φ, u, v, usq)
    q == 8 && return feq_2d(Val(8), φ, u, v, usq)
    return feq_2d(Val(9), φ, u, v, usq)
end

@inline function _ylw_chi(qw::T, tau_plus::T) where {T}
    return qw < T(0.5) ? (T(2) * qw - one(T)) / (tau_plus - T(2)) :
                         (T(2) * qw - one(T)) / tau_plus
end

@inline function _ylw_ghost_us(qw::T, q_out, i, j, ux, uy, is_solid, Nx, Ny,
                               ub::T, vb::T, use_local_velocity) where {T}
    if qw < T(0.5)
        bi = i - _cx_q(q_out)
        bj = j - _cy_q(q_out)
        if use_local_velocity && 1 <= bi <= Nx && 1 <= bj <= Ny && !is_solid[bi, bj]
            return ux[bi, bj], uy[bi, bj]
        end
        return ub, vb
    end
    scale = (qw - one(T)) / qw
    return scale * ub, scale * vb
end

@inline function _feq_ylw_ghost_2d(q, φ::T, ub::T, vb::T,
                                   us::T, vs::T) where {T}
    w = q == 1 ? T(4/9) :
        (q == 2 || q == 3 || q == 4 || q == 5 ? T(1/9) : T(1/36))
    cx = T(_cx_q(q))
    cy = T(_cy_q(q))
    eu_s = cx * us + cy * vs
    eu_b = cx * ub + cy * vb
    ub2 = ub * ub + vb * vb
    return w * φ * (one(T) + T(3) * eu_s + T(4.5) * eu_b * eu_b - T(1.5) * ub2)
end

@inline function _feq_lit_2d(w::T, cx::T, cy::T, φ::T, u::T, v::T,
                             usq::T) where {T}
    eu = cx * u + cy * v
    return w * φ * (one(T) + T(3) * eu + T(4.5) * eu * eu - T(1.5) * usq)
end

@inline function _feq_ylw_lit_2d(w::T, cx::T, cy::T, φ::T, ub::T, vb::T,
                                 us::T, vs::T) where {T}
    eu_s = cx * us + cy * vs
    eu_b = cx * ub + cy * vb
    ub2 = ub * ub + vb * vb
    return w * φ * (one(T) + T(3) * eu_s + T(4.5) * eu_b * eu_b - T(1.5) * ub2)
end

@kernel function apply_ylw_a_2d_kernel!(g_post, @Const(g_pre), @Const(is_solid),
                                         @Const(q_wall), C_field,
                                         @Const(ux_bc), @Const(uy_bc),
                                         use_local_velocity, tau_plus, Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds if !is_solid[i, j]
        T = eltype(g_post)
        any_solid = false
        for q in 2:9
            ni = i + _cx_q(q)
            nj = j + _cy_q(q)
            if !(1 <= ni <= Nx && 1 <= nj <= Ny) || is_solid[ni, nj]
                any_solid = true
            end
        end

        if any_solid
            ub = use_local_velocity ? ux_bc[i, j] : zero(T)
            vb = use_local_velocity ? uy_bc[i, j] : zero(T)
            φb = C_field[i, j]
            outgoing = zero(T)
            incoming = zero(T)

            for q in 2:9
                si = i - _cx_q(q)
                sj = j - _cy_q(q)
                src_solid = !(1 <= si <= Nx && 1 <= sj <= Ny) ||
                            is_solid[si, sj]
                if src_solid
                    q_out = _opp_q(q)
                    qw_raw = q_wall[i, j, q_out]
                    qw = qw_raw > zero(T) ? qw_raw : T(0.5)
                    χ = _ylw_chi(qw, T(tau_plus))
                    us = ub
                    vs = vb
                    if qw < T(0.5)
                        bi = i - _cx_q(q_out)
                        bj = j - _cy_q(q_out)
                        if use_local_velocity && 1 <= bi <= Nx && 1 <= bj <= Ny &&
                           !is_solid[bi, bj]
                            us = ux_bc[bi, bj]
                            vs = uy_bc[bi, bj]
                        end
                    else
                        scale = (qw - one(T)) / qw
                        us = scale * ub
                        vs = scale * vb
                    end
                    geq_s = _feq_ylw_ghost_2d(q_out, φb, ub, vb, us, vs)
                    incoming_q = (one(T) - χ) * g_pre[i, j, q_out] + χ * geq_s
                    g_post[i, j, q] = incoming_q
                    outgoing += g_pre[i, j, q_out]
                    incoming += incoming_q
                end
            end

            g_post[i, j, 1] += outgoing - incoming
            φ = zero(T)
            for q in 1:9
                φ += g_post[i, j, q]
            end
            C_field[i, j] = φ
        end
    end
end

@kernel function apply_ylw_b_2d_kernel!(g_post, @Const(g_pre), @Const(is_solid),
                                         @Const(q_wall), C_field,
                                         @Const(ux_bc), @Const(uy_bc),
                                         use_local_velocity, tau_plus, Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds if !is_solid[i, j]
        T = eltype(g_post)
        any_solid = false
        for q in 2:9
            ni = i + _cx_q(q)
            nj = j + _cy_q(q)
            if !(1 <= ni <= Nx && 1 <= nj <= Ny) || is_solid[ni, nj]
                any_solid = true
            end
        end

        if any_solid
            ub = use_local_velocity ? ux_bc[i, j] : zero(T)
            vb = use_local_velocity ? uy_bc[i, j] : zero(T)
            numerator = zero(T)
            denominator = zero(T)

            for q in 2:9
                si = i - _cx_q(q)
                sj = j - _cy_q(q)
                src_solid = !(1 <= si <= Nx && 1 <= sj <= Ny) ||
                            is_solid[si, sj]
                if src_solid
                    q_out = _opp_q(q)
                    qw_raw = q_wall[i, j, q_out]
                    qw = qw_raw > zero(T) ? qw_raw : T(0.5)
                    χ = _ylw_chi(qw, T(tau_plus))
                    us = ub
                    vs = vb
                    if qw < T(0.5)
                        bi = i - _cx_q(q_out)
                        bj = j - _cy_q(q_out)
                        if use_local_velocity && 1 <= bi <= Nx && 1 <= bj <= Ny &&
                           !is_solid[bi, bj]
                            us = ux_bc[bi, bj]
                            vs = uy_bc[bi, bj]
                        end
                    else
                        scale = (qw - one(T)) / qw
                        us = scale * ub
                        vs = scale * vb
                    end
                    numerator += χ * g_pre[i, j, q_out]
                    denominator += χ * _feq_ylw_lit_2d(T(_w_q(q_out)),
                                                       T(_cx_q(q_out)),
                                                       T(_cy_q(q_out)),
                                                       one(T), ub, vb, us, vs)
                end
            end

            φs = abs(denominator) > T(100) * eps(T) ?
                 numerator / denominator : C_field[i, j]

            for q in 2:9
                si = i - _cx_q(q)
                sj = j - _cy_q(q)
                src_solid = !(1 <= si <= Nx && 1 <= sj <= Ny) ||
                            is_solid[si, sj]
                if src_solid
                    q_out = _opp_q(q)
                    qw_raw = q_wall[i, j, q_out]
                    qw = qw_raw > zero(T) ? qw_raw : T(0.5)
                    χ = _ylw_chi(qw, T(tau_plus))
                    us = ub
                    vs = vb
                    if qw < T(0.5)
                        bi = i - _cx_q(q_out)
                        bj = j - _cy_q(q_out)
                        if use_local_velocity && 1 <= bi <= Nx && 1 <= bj <= Ny &&
                           !is_solid[bi, bj]
                            us = ux_bc[bi, bj]
                            vs = uy_bc[bi, bj]
                        end
                    else
                        scale = (qw - one(T)) / qw
                        us = scale * ub
                        vs = scale * vb
                    end
                    geq_s = _feq_ylw_lit_2d(T(_w_q(q_out)),
                                            T(_cx_q(q_out)),
                                            T(_cy_q(q_out)),
                                            φs, ub, vb, us, vs)
                    g_post[i, j, q] = (one(T) - χ) * g_pre[i, j, q_out] + χ * geq_s
                end
            end

            φ = zero(T)
            for q in 1:9
                φ += g_post[i, j, q]
            end
            C_field[i, j] = φ
        end
    end
end

@kernel function apply_ylw_balance_2d_kernel!(g_post, @Const(g_pre), @Const(is_solid),
                                               @Const(q_wall), C_field,
                                               @Const(ux_bc), @Const(uy_bc),
                                               use_local_velocity, Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds if !is_solid[i, j]
        T = eltype(g_post)
        any_solid = false
        for q in 2:9
            ni = i + _cx_q(q)
            nj = j + _cy_q(q)
            if !(1 <= ni <= Nx && 1 <= nj <= Ny) || is_solid[ni, nj]
                any_solid = true
            end
        end

        if any_solid
            ub = use_local_velocity ? ux_bc[i, j] : zero(T)
            vb = use_local_velocity ? uy_bc[i, j] : zero(T)
            usq = ub * ub + vb * vb

            φb = C_field[i, j]
            outgoing = zero(T)
            incoming = zero(T)

            for q in 2:9
                si = i - _cx_q(q)
                sj = j - _cy_q(q)
                src_solid = !(1 <= si <= Nx && 1 <= sj <= Ny) ||
                            is_solid[si, sj]
                if src_solid
                    q_out = _opp_q(q)
                    geq_q = _feq_q_2d(q, φb, ub, vb, usq)
                    geq_out = _feq_q_2d(q_out, φb, ub, vb, usq)
                    incoming_q = geq_q + (g_post[i, j, q_out] - geq_out)
                    g_post[i, j, q] = incoming_q
                    outgoing += g_pre[i, j, q_out]
                    incoming += incoming_q
                end
            end

            g_post[i, j, 1] += outgoing - incoming
            φ = zero(T)
            for q in 1:9
                φ += g_post[i, j, q]
            end
            C_field[i, j] = φ
        end
    end
end

@kernel function apply_cnebb_strict_simple_2d_kernel!(g_post, @Const(g_pre),
                                                       @Const(is_solid),
                                                       C_field,
                                                       @Const(ux_bc), @Const(uy_bc),
                                                       use_local_velocity,
                                                       Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds if !is_solid[i, j]
        T = eltype(g_post)
        any_solid = false
        for q in 2:9
            ni = i + _cx_q(q)
            nj = j + _cy_q(q)
            if !(1 <= ni <= Nx && 1 <= nj <= Ny) || is_solid[ni, nj]
                any_solid = true
            end
        end

        if any_solid
            ub = use_local_velocity ? ux_bc[i, j] : zero(T)
            vb = use_local_velocity ? uy_bc[i, j] : zero(T)
            φ = g_post[i, j, 1]
            for q in 2:9
                si = i - _cx_q(q)
                sj = j - _cy_q(q)
                src_solid = !(1 <= si <= Nx && 1 <= sj <= Ny) ||
                            is_solid[si, sj]
                φ += src_solid ? g_pre[i, j, _opp_q(q)] : g_post[i, j, q]
            end

            usq = ub * ub + vb * vb
            for q in 2:9
                si = i - _cx_q(q)
                sj = j - _cy_q(q)
                src_solid = !(1 <= si <= Nx && 1 <= sj <= Ny) ||
                            is_solid[si, sj]
                if src_solid
                    q_out = _opp_q(q)
                    geq_q = _feq_q_2d(q, φ, ub, vb, usq)
                    geq_out = _feq_q_2d(q_out, φ, ub, vb, usq)
                    g_post[i, j, q] = geq_q + (g_post[i, j, q_out] - geq_out)
                end
            end

            nonrest = zero(T)
            for q in 2:9
                nonrest += g_post[i, j, q]
            end
            g_post[i, j, 1] = φ - nonrest
            C_field[i, j] = φ
        end
    end
end

@kernel function apply_cnebb_qaware_simple_2d_kernel!(g_post, @Const(g_pre),
                                                      @Const(is_solid),
                                                      @Const(q_wall),
                                                      C_field,
                                                      @Const(ux_bc), @Const(uy_bc),
                                                      use_local_velocity,
                                                      Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds if !is_solid[i, j]
        T = eltype(g_post)
        any_solid = false
        for q in 2:9
            ni = i + _cx_q(q)
            nj = j + _cy_q(q)
            if !(1 <= ni <= Nx && 1 <= nj <= Ny) || is_solid[ni, nj]
                any_solid = true
            end
        end

        if any_solid
            ub = use_local_velocity ? ux_bc[i, j] : zero(T)
            vb = use_local_velocity ? uy_bc[i, j] : zero(T)
            φ = g_post[i, j, 1]
            for q in 2:9
                si = i - _cx_q(q)
                sj = j - _cy_q(q)
                src_solid = !(1 <= si <= Nx && 1 <= sj <= Ny) ||
                            is_solid[si, sj]
                φ += src_solid ? g_pre[i, j, _opp_q(q)] : g_post[i, j, q]
            end

            usq = ub * ub + vb * vb
            for q in 2:9
                si = i - _cx_q(q)
                sj = j - _cy_q(q)
                src_solid = !(1 <= si <= Nx && 1 <= sj <= Ny) ||
                            is_solid[si, sj]
                if src_solid
                    q_out = _opp_q(q)
                    geq_q = _feq_q_2d(q, φ, ub, vb, usq)
                    geq_out = _feq_q_2d(q_out, φ, ub, vb, usq)
                    q_w = q_wall[i, j, q_out]
                    if q_w > zero(T) && abs(q_w - T(0.5)) > T(100) * eps(T)
                        neq_out = g_pre[i, j, q_out] - geq_out
                        neq_q = zero(T)
                        if q_w <= T(0.5)
                            bi = i - _cx_q(q_out)
                            bj = j - _cy_q(q_out)
                            geq_back_out = geq_out
                            if 1 <= bi <= Nx && 1 <= bj <= Ny && !is_solid[bi, bj]
                                us = use_local_velocity ? ux_bc[bi, bj] : zero(T)
                                vs = use_local_velocity ? uy_bc[bi, bj] : zero(T)
                                Cs = C_field[bi, bj]
                                geq_back_out = _feq_q_2d(q_out, Cs, us, vs,
                                                         us * us + vs * vs)
                            end
                            neq_back_out = g_post[i, j, q_out] - geq_back_out
                            neq_q = T(2) * q_w * neq_out +
                                    (one(T) - T(2) * q_w) * neq_back_out
                        else
                            inv_two_q = one(T) / (T(2) * q_w)
                            neq_here_q = g_pre[i, j, q] - geq_q
                            neq_q = inv_two_q * neq_out +
                                    (one(T) - inv_two_q) * neq_here_q
                        end
                        g_post[i, j, q] = geq_q + neq_q
                    else
                        g_post[i, j, q] = geq_q + (g_post[i, j, q_out] - geq_out)
                    end
                end
            end

            nonrest = zero(T)
            for q in 2:9
                nonrest += g_post[i, j, q]
            end
            g_post[i, j, 1] = φ - nonrest
            C_field[i, j] = φ
        end
    end
end

@kernel function apply_cnebb_field_qaware_2d_kernel!(g_post, @Const(g_pre),
                                                     @Const(is_solid),
                                                     @Const(q_wall),
                                                     use_q_wall, C_field,
                                                     @Const(ux_bc), @Const(uy_bc),
                                                     use_local_velocity,
                                                     Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds if !is_solid[i, j]
        T = eltype(g_post)
        any_solid = false
        for q in 2:9
            ni = i + _cx_q(q)
            nj = j + _cy_q(q)
            if !(1 <= ni <= Nx && 1 <= nj <= Ny) || is_solid[ni, nj]
                any_solid = true
            end
        end

        if any_solid
            φ = C_field[i, j]
            ub = use_local_velocity ? ux_bc[i, j] : zero(T)
            vb = use_local_velocity ? uy_bc[i, j] : zero(T)
            usq = ub * ub + vb * vb

            for q in 2:9
                si = i - _cx_q(q)
                sj = j - _cy_q(q)
                src_solid = !(1 <= si <= Nx && 1 <= sj <= Ny) ||
                            is_solid[si, sj]
                if src_solid
                    q_out = _opp_q(q)
                    geq_q = _feq_q_2d(q, φ, ub, vb, usq)
                    geq_out = _feq_q_2d(q_out, φ, ub, vb, usq)
                    q_w = use_q_wall ? q_wall[i, j, q_out] : zero(T)
                    if use_q_wall && q_w > zero(T) &&
                       abs(q_w - T(0.5)) > T(100) * eps(T)
                        neq_out = g_pre[i, j, q_out] - geq_out
                        if q_w <= T(0.5)
                            bi = i - _cx_q(q_out)
                            bj = j - _cy_q(q_out)
                            geq_back_out = geq_out
                            if 1 <= bi <= Nx && 1 <= bj <= Ny && !is_solid[bi, bj]
                                us = use_local_velocity ? ux_bc[bi, bj] : zero(T)
                                vs = use_local_velocity ? uy_bc[bi, bj] : zero(T)
                                Cs = C_field[bi, bj]
                                geq_back_out = _feq_q_2d(q_out, Cs, us, vs,
                                                         us * us + vs * vs)
                            end
                            neq_back_out = g_post[i, j, q_out] - geq_back_out
                            g_post[i, j, q] = geq_q + T(2) * q_w * neq_out +
                                              (one(T) - T(2) * q_w) * neq_back_out
                        else
                            inv_two_q = one(T) / (T(2) * q_w)
                            neq_here_q = g_pre[i, j, q] - geq_q
                            g_post[i, j, q] = geq_q + inv_two_q * neq_out +
                                              (one(T) - inv_two_q) * neq_here_q
                        end
                    else
                        g_post[i, j, q] = geq_q + (g_post[i, j, q_out] - geq_out)
                    end
                end
            end

            nonrest = zero(T)
            for q in 2:9
                nonrest += g_post[i, j, q]
            end
            g_post[i, j, 1] = φ - nonrest
        end
    end
end

@kernel function apply_cnebb_qaware_correction_2d_kernel!(g_post, @Const(g_pre),
                                                          @Const(is_solid),
                                                          @Const(q_wall),
                                                          C_field,
                                                          @Const(ux_bc), @Const(uy_bc),
                                                          use_local_velocity,
                                                          Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds if !is_solid[i, j]
        T = eltype(g_post)
        any_solid = false
        for q in 2:9
            si = i - _cx_q(q)
            sj = j - _cy_q(q)
            if !(1 <= si <= Nx && 1 <= sj <= Ny) || is_solid[si, sj]
                any_solid = true
            end
        end

        if any_solid
            φ = C_field[i, j]
            ub = use_local_velocity ? ux_bc[i, j] : zero(T)
            vb = use_local_velocity ? uy_bc[i, j] : zero(T)
            usq = ub * ub + vb * vb
            for q in 2:9
                si = i - _cx_q(q)
                sj = j - _cy_q(q)
                src_solid = !(1 <= si <= Nx && 1 <= sj <= Ny) ||
                            is_solid[si, sj]
                if src_solid
                    q_out = _opp_q(q)
                    q_w = q_wall[i, j, q_out]
                    if q_w > zero(T) && abs(q_w - T(0.5)) > T(100) * eps(T)
                        geq_q = _feq_q_2d(q, φ, ub, vb, usq)
                        geq_out = _feq_q_2d(q_out, φ, ub, vb, usq)
                        neq_out = g_pre[i, j, q_out] - geq_out
                        if q_w <= T(0.5)
                            bi = i - _cx_q(q_out)
                            bj = j - _cy_q(q_out)
                            geq_back_out = geq_out
                            if 1 <= bi <= Nx && 1 <= bj <= Ny && !is_solid[bi, bj]
                                us = use_local_velocity ? ux_bc[bi, bj] : zero(T)
                                vs = use_local_velocity ? uy_bc[bi, bj] : zero(T)
                                Cs = C_field[bi, bj]
                                geq_back_out = _feq_q_2d(q_out, Cs, us, vs,
                                                         us * us + vs * vs)
                            end
                            neq_back_out = g_post[i, j, q_out] - geq_back_out
                            g_post[i, j, q] = geq_q + T(2) * q_w * neq_out +
                                              (one(T) - T(2) * q_w) * neq_back_out
                        else
                            inv_two_q = one(T) / (T(2) * q_w)
                            neq_here_q = g_pre[i, j, q] - geq_q
                            g_post[i, j, q] = geq_q + inv_two_q * neq_out +
                                              (one(T) - inv_two_q) * neq_here_q
                        end
                    end
                end
            end

            nonrest = zero(T)
            for q in 2:9
                nonrest += g_post[i, j, q]
            end
            g_post[i, j, 1] = φ - nonrest
        end
    end
end

@kernel function apply_cnebb_qaware_dir_2d_kernel!(g_post, @Const(g_pre),
                                                   @Const(is_solid),
                                                   @Const(q_wall), C_field,
                                                   @Const(ux_bc), @Const(uy_bc),
                                                   use_local_velocity,
                                                   q, q_out, cxq, cyq,
                                                   cxout, cyout, wq, wout,
                                                   Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds if !is_solid[i, j]
        T = eltype(g_post)
        si = i - cxq
        sj = j - cyq
        src_solid = !(1 <= si <= Nx && 1 <= sj <= Ny) || is_solid[si, sj]
        if src_solid
            q_w = q_wall[i, j, q_out]
            if q_w > zero(T) && abs(q_w - T(0.5)) > T(100) * eps(T)
                φ = C_field[i, j]
                ub = use_local_velocity ? ux_bc[i, j] : zero(T)
                vb = use_local_velocity ? uy_bc[i, j] : zero(T)
                usq = ub * ub + vb * vb
                cxq_t = T(cxq)
                cyq_t = T(cyq)
                cxout_t = T(cxout)
                cyout_t = T(cyout)
                geq_q = _feq_lit_2d(T(wq), cxq_t, cyq_t, φ, ub, vb, usq)
                geq_out = _feq_lit_2d(T(wout), cxout_t, cyout_t, φ, ub, vb, usq)
                neq_out = g_pre[i, j, q_out] - geq_out
                if q_w <= T(0.5)
                    bi = i - cxout
                    bj = j - cyout
                    geq_back_out = geq_out
                    if 1 <= bi <= Nx && 1 <= bj <= Ny && !is_solid[bi, bj]
                        us = use_local_velocity ? ux_bc[bi, bj] : zero(T)
                        vs = use_local_velocity ? uy_bc[bi, bj] : zero(T)
                        Cs = C_field[bi, bj]
                        geq_back_out = _feq_lit_2d(T(wout), cxout_t, cyout_t,
                                                   Cs, us, vs, us * us + vs * vs)
                    end
                    neq_back_out = g_post[i, j, q_out] - geq_back_out
                    g_post[i, j, q] = geq_q + T(2) * q_w * neq_out +
                                      (one(T) - T(2) * q_w) * neq_back_out
                else
                    inv_two_q = one(T) / (T(2) * q_w)
                    neq_here_q = g_pre[i, j, q] - geq_q
                    g_post[i, j, q] = geq_q + inv_two_q * neq_out +
                                      (one(T) - inv_two_q) * neq_here_q
                end
            end
        end
    end
end

@kernel function apply_cnebb_field_qaware_dir_2d_kernel!(g_post, @Const(g_pre),
                                                         @Const(is_solid),
                                                         @Const(q_wall),
                                                         use_q_wall, C_field,
                                                         @Const(ux_bc),
                                                         @Const(uy_bc),
                                                         use_local_velocity,
                                                         q, q_out, cxq, cyq,
                                                         cxout, cyout, wq, wout,
                                                         Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds if !is_solid[i, j]
        T = eltype(g_post)
        si = i - cxq
        sj = j - cyq
        src_solid = !(1 <= si <= Nx && 1 <= sj <= Ny) || is_solid[si, sj]
        if src_solid
            φ = C_field[i, j]
            ub = use_local_velocity ? ux_bc[i, j] : zero(T)
            vb = use_local_velocity ? uy_bc[i, j] : zero(T)
            usq = ub * ub + vb * vb
            cxq_t = T(cxq)
            cyq_t = T(cyq)
            cxout_t = T(cxout)
            cyout_t = T(cyout)
            geq_q = _feq_lit_2d(T(wq), cxq_t, cyq_t, φ, ub, vb, usq)
            geq_out = _feq_lit_2d(T(wout), cxout_t, cyout_t, φ, ub, vb, usq)
            q_w = use_q_wall ? q_wall[i, j, q_out] : zero(T)
            if use_q_wall && q_w > zero(T) &&
               abs(q_w - T(0.5)) > T(100) * eps(T)
                neq_out = g_pre[i, j, q_out] - geq_out
                if q_w <= T(0.5)
                    bi = i - cxout
                    bj = j - cyout
                    geq_back_out = geq_out
                    if 1 <= bi <= Nx && 1 <= bj <= Ny && !is_solid[bi, bj]
                        us = use_local_velocity ? ux_bc[bi, bj] : zero(T)
                        vs = use_local_velocity ? uy_bc[bi, bj] : zero(T)
                        Cs = C_field[bi, bj]
                        geq_back_out = _feq_lit_2d(T(wout), cxout_t, cyout_t,
                                                   Cs, us, vs, us * us + vs * vs)
                    end
                    neq_back_out = g_post[i, j, q_out] - geq_back_out
                    g_post[i, j, q] = geq_q + T(2) * q_w * neq_out +
                                      (one(T) - T(2) * q_w) * neq_back_out
                else
                    inv_two_q = one(T) / (T(2) * q_w)
                    neq_here_q = g_pre[i, j, q] - geq_q
                    g_post[i, j, q] = geq_q + inv_two_q * neq_out +
                                      (one(T) - inv_two_q) * neq_here_q
                end
            else
                g_post[i, j, q] = geq_q + (g_post[i, j, q_out] - geq_out)
            end
        end
    end
end

@kernel function rebalance_wall_rest_to_field_2d_kernel!(g_post,
                                                         @Const(is_solid),
                                                         C_field, Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds if !is_solid[i, j]
        any_solid = false
        for q in 2:9
            si = i - _cx_q(q)
            sj = j - _cy_q(q)
            if !(1 <= si <= Nx && 1 <= sj <= Ny) || is_solid[si, sj]
                any_solid = true
            end
        end
        if any_solid
            T = eltype(g_post)
            nonrest = zero(T)
            for q in 2:9
                nonrest += g_post[i, j, q]
            end
            g_post[i, j, 1] = C_field[i, j] - nonrest
        end
    end
end

@kernel function apply_extrap_eq_conformation_2d_kernel!(g_post,
                                                         @Const(is_solid),
                                                         @Const(q_wall),
                                                         @Const(C_field),
                                                         @Const(ux),
                                                         @Const(uy),
                                                         Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds if !is_solid[i, j]
        any_cut = false
        for q in 2:9
            if q_wall[i, j, q] > 0
                any_cut = true
            end
        end

        if any_cut
            T = eltype(g_post)
            dCdx = _wall_aware_dx_2d(C_field, is_solid, i, j, Nx, T)
            dCdy = _wall_aware_dy_2d(C_field, is_solid, i, j, Ny, T)
            dudx = _wall_aware_dx_2d(ux, is_solid, i, j, Nx, T)
            dudy = _wall_aware_dy_2d(ux, is_solid, i, j, Ny, T)
            dvdx = _wall_aware_dx_2d(uy, is_solid, i, j, Nx, T)
            dvdy = _wall_aware_dy_2d(uy, is_solid, i, j, Ny, T)

            for q in 2:9
                cx = _cx_q(q)
                cy = _cy_q(q)
                si = i - cx
                sj = j - cy
                src_solid = !(1 <= si <= Nx && 1 <= sj <= Ny) || is_solid[si, sj]
                if src_solid
                    offx = T(-cx)
                    offy = T(-cy)
                    Cv = C_field[i, j] + offx * dCdx + offy * dCdy
                    uv = ux[i, j] + offx * dudx + offy * dudy
                    vv = uy[i, j] + offx * dvdx + offy * dvdy
                    g_post[i, j, q] = _feq_q_2d(q, Cv, uv, vv, uv * uv + vv * vv)
                end
            end

            nonrest = zero(T)
            for q in 2:9
                nonrest += g_post[i, j, q]
            end
            g_post[i, j, 1] = C_field[i, j] - nonrest
        end
    end
end

@kernel function apply_ylw_a_dir_2d_kernel!(g_post, @Const(g_pre),
                                            @Const(is_solid), @Const(q_wall),
                                            C_field, @Const(ux_bc), @Const(uy_bc),
                                            use_local_velocity, tau_plus,
                                            q, q_out, cxq, cyq, cxout, cyout,
                                            wout, Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds if !is_solid[i, j]
        T = eltype(g_post)
        si = i - cxq
        sj = j - cyq
        src_solid = !(1 <= si <= Nx && 1 <= sj <= Ny) || is_solid[si, sj]
        if src_solid
            ub = use_local_velocity ? ux_bc[i, j] : zero(T)
            vb = use_local_velocity ? uy_bc[i, j] : zero(T)
            qw_raw = q_wall[i, j, q_out]
            qw = qw_raw > zero(T) ? qw_raw : T(0.5)
            χ = _ylw_chi(qw, T(tau_plus))
            us = ub
            vs = vb
            if qw < T(0.5)
                bi = i - cxout
                bj = j - cyout
                if use_local_velocity && 1 <= bi <= Nx && 1 <= bj <= Ny &&
                   !is_solid[bi, bj]
                    us = ux_bc[bi, bj]
                    vs = uy_bc[bi, bj]
                end
            else
                scale = (qw - one(T)) / qw
                us = scale * ub
                vs = scale * vb
            end
            geq_s = _feq_ylw_lit_2d(T(wout), T(cxout), T(cyout), C_field[i, j],
                                    ub, vb, us, vs)
            g_post[i, j, q] = (one(T) - χ) * g_pre[i, j, q_out] + χ * geq_s
        end
    end
end

@kernel function apply_ylw_a_rest_balance_2d_kernel!(g_post, @Const(g_pre),
                                                     @Const(is_solid), C_field,
                                                     Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds if !is_solid[i, j]
        T = eltype(g_post)
        outgoing = zero(T)
        incoming = zero(T)
        any_solid = false
        for q in 2:9
            si = i - _cx_q(q)
            sj = j - _cy_q(q)
            src_solid = !(1 <= si <= Nx && 1 <= sj <= Ny) || is_solid[si, sj]
            if src_solid
                any_solid = true
                q_out = _opp_q(q)
                outgoing += g_pre[i, j, q_out]
                incoming += g_post[i, j, q]
            end
        end
        if any_solid
            g_post[i, j, 1] += outgoing - incoming
            φ = zero(T)
            for q in 1:9
                φ += g_post[i, j, q]
            end
            C_field[i, j] = φ
        end
    end
end

@kernel function apply_ylw_b_accumulate_dir_2d_kernel!(num, den, @Const(g_pre),
                                                       @Const(is_solid),
                                                       @Const(q_wall),
                                                       @Const(C_field),
                                                       @Const(ux_bc),
                                                       @Const(uy_bc),
                                                       use_local_velocity,
                                                       tau_plus, q, q_out,
                                                       cxq, cyq, cxout, cyout,
                                                       wout, Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds if !is_solid[i, j]
        T = eltype(num)
        si = i - cxq
        sj = j - cyq
        src_solid = !(1 <= si <= Nx && 1 <= sj <= Ny) || is_solid[si, sj]
        if src_solid
            ub = use_local_velocity ? ux_bc[i, j] : zero(T)
            vb = use_local_velocity ? uy_bc[i, j] : zero(T)
            qw_raw = q_wall[i, j, q_out]
            qw = qw_raw > zero(T) ? qw_raw : T(0.5)
            χ = _ylw_chi(qw, T(tau_plus))
            us = ub
            vs = vb
            if qw < T(0.5)
                bi = i - cxout
                bj = j - cyout
                if use_local_velocity && 1 <= bi <= Nx && 1 <= bj <= Ny &&
                   !is_solid[bi, bj]
                    us = ux_bc[bi, bj]
                    vs = uy_bc[bi, bj]
                end
            else
                scale = (qw - one(T)) / qw
                us = scale * ub
                vs = scale * vb
            end
            num[i, j] += χ * g_pre[i, j, q_out]
            den[i, j] += χ * _feq_ylw_lit_2d(T(wout), T(cxout), T(cyout),
                                             one(T), ub, vb, us, vs)
        end
    end
end

@kernel function apply_ylw_b_phi_from_accum_2d_kernel!(num, den,
                                                       @Const(is_solid),
                                                       C_field, Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds if !is_solid[i, j]
        T = eltype(C_field)
        any_solid = false
        for q in 2:9
            si = i - _cx_q(q)
            sj = j - _cy_q(q)
            if !(1 <= si <= Nx && 1 <= sj <= Ny) || is_solid[si, sj]
                any_solid = true
            end
        end
        if any_solid && abs(den[i, j]) > T(100) * eps(T)
            C_field[i, j] = num[i, j] / den[i, j]
        end
    end
end

@kernel function compute_wall_macro_sum_2d_kernel!(g_post, @Const(is_solid),
                                                   C_field, Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds if !is_solid[i, j]
        T = eltype(g_post)
        any_solid = false
        for q in 2:9
            si = i - _cx_q(q)
            sj = j - _cy_q(q)
            if !(1 <= si <= Nx && 1 <= sj <= Ny) || is_solid[si, sj]
                any_solid = true
            end
        end
        if any_solid
            φ = zero(T)
            for q in 1:9
                φ += g_post[i, j, q]
            end
            C_field[i, j] = φ
        end
    end
end

@kernel function reset_cutlink_conformation_equilibrium_2d_kernel!(
        g, @Const(C_field), @Const(ux_bc), @Const(uy_bc),
        @Const(is_solid), @Const(q_wall), use_local_velocity, Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds if !is_solid[i, j]
        T = eltype(g)
        any_cut = false
        for q in 2:9
            any_cut |= q_wall[i, j, q] > zero(T)
        end
        if any_cut
            φ = C_field[i, j]
            u = use_local_velocity ? ux_bc[i, j] : zero(T)
            v = use_local_velocity ? uy_bc[i, j] : zero(T)
            usq = u * u + v * v
            g[i, j, 1] = feq_2d(Val(1), φ, u, v, usq)
            g[i, j, 2] = feq_2d(Val(2), φ, u, v, usq)
            g[i, j, 3] = feq_2d(Val(3), φ, u, v, usq)
            g[i, j, 4] = feq_2d(Val(4), φ, u, v, usq)
            g[i, j, 5] = feq_2d(Val(5), φ, u, v, usq)
            g[i, j, 6] = feq_2d(Val(6), φ, u, v, usq)
            g[i, j, 7] = feq_2d(Val(7), φ, u, v, usq)
            g[i, j, 8] = feq_2d(Val(8), φ, u, v, usq)
            g[i, j, 9] = feq_2d(Val(9), φ, u, v, usq)
        end
    end
end

@kernel function apply_cnebb_2d_kernel!(g_post, @Const(g_pre), @Const(is_solid),
                                          @Const(q_wall), use_q_wall,
                                          C_field, @Const(ux_bc), @Const(uy_bc),
                                          use_local_velocity, phi_mode, Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds if !is_solid[i, j]
        T = eltype(g_post)

        # Detect if any neighbour is solid or outside the domain; if not,
        # nothing to do. Domain boundaries are no-slip walls for the
        # conformation populations too, and must get the same conservative
        # reconstruction as embedded solid boundaries.
        any_solid = false
        for q in 2:9
            ni = i + _cx_q(q)
            nj = j + _cy_q(q)
            if !(1 <= ni <= Nx && 1 <= nj <= Ny) || is_solid[ni, nj]
                any_solid = true
            end
        end

        if any_solid
            ub = use_local_velocity ? ux_bc[i, j] : zero(T)
            vb = use_local_velocity ? uy_bc[i, j] : zero(T)
            usq = ub * ub + vb * vb

            has_cut_link = false
            if use_q_wall
                for q in 2:9
                    has_cut_link |= q_wall[i, j, q] > zero(T)
                end
            end
            use_eq_gradient_phi = phi_mode == 3 ||
                                  (phi_mode == 4 && has_cut_link)

            Cb = C_field[i, j]
            ge_local_1 = feq_2d(Val(1), Cb, ub, vb, usq)
            ge_local_2 = feq_2d(Val(2), Cb, ub, vb, usq)
            ge_local_3 = feq_2d(Val(3), Cb, ub, vb, usq)
            ge_local_4 = feq_2d(Val(4), Cb, ub, vb, usq)
            ge_local_5 = feq_2d(Val(5), Cb, ub, vb, usq)
            ge_local_6 = feq_2d(Val(6), Cb, ub, vb, usq)
            ge_local_7 = feq_2d(Val(7), Cb, ub, vb, usq)
            ge_local_8 = feq_2d(Val(8), Cb, ub, vb, usq)
            ge_local_9 = feq_2d(Val(9), Cb, ub, vb, usq)

            # Step 1+2: compute φ conservatively.
            # phi_mode == 3 is a diagnostic wall-gradient correction: valid
            # incoming populations keep their non-equilibrium part, but their
            # equilibrium part is shifted from the source node to the local
            # wall-adjacent node. This preserves linear equilibrium profiles
            # exactly while leaving the production :pre_opp path unchanged.
            # phi_mode == 4 applies the same correction only on cells that
            # carry explicit cut-link geometry in q_wall. Domain walls fall
            # back to Yu/CNEBB :pre_opp recovery, which is required by the
            # planar Poiseuille CDE patch tests.
            φ = phi_mode == 2 ? C_field[i, j] : zero(T)
            if use_eq_gradient_phi
                φ = Cb + g_post[i, j, 1] - ge_local_1
                for q in 2:9
                    si = i - _cx_q(q)
                    sj = j - _cy_q(q)
                    src_solid = !(1 <= si <= Nx && 1 <= sj <= Ny) ||
                                is_solid[si, sj]
                    if src_solid
                        oq = _opp_q(q)
                        geq_local_oq = oq == 2 ? ge_local_2 : oq == 3 ? ge_local_3 : oq == 4 ? ge_local_4 :
                                        oq == 5 ? ge_local_5 : oq == 6 ? ge_local_6 : oq == 7 ? ge_local_7 :
                                        oq == 8 ? ge_local_8 : ge_local_9
                        φ += g_pre[i, j, oq] - geq_local_oq
                    else
                        us = use_local_velocity ? ux_bc[si, sj] : zero(T)
                        vs = use_local_velocity ? uy_bc[si, sj] : zero(T)
                        usq_src = us * us + vs * vs
                        Cs = C_field[si, sj]
                        geq_src_q = q == 2 ? feq_2d(Val(2), Cs, us, vs, usq_src) :
                                    q == 3 ? feq_2d(Val(3), Cs, us, vs, usq_src) :
                                    q == 4 ? feq_2d(Val(4), Cs, us, vs, usq_src) :
                                    q == 5 ? feq_2d(Val(5), Cs, us, vs, usq_src) :
                                    q == 6 ? feq_2d(Val(6), Cs, us, vs, usq_src) :
                                    q == 7 ? feq_2d(Val(7), Cs, us, vs, usq_src) :
                                    q == 8 ? feq_2d(Val(8), Cs, us, vs, usq_src) :
                                             feq_2d(Val(9), Cs, us, vs, usq_src)
                        φ += g_post[i, j, q] - geq_src_q
                    end
                end
            elseif phi_mode != 2
                # Rest population is always valid (q=1, no streaming)
                φ += g_post[i, j, 1]
                for q in 2:9
                    # Source neighbour for direction q in stream_2d is (i-cx, j-cy).
                    # If that source is solid, the default diagnostic recovers the
                    # missing value from g_pre[opp(q)]. `post_opp` tests the alternate
                    # reading where the known opposite post-streaming population
                    # supplies the conservative φ.
                    si = i - _cx_q(q)
                    sj = j - _cy_q(q)
                    src_solid = !(1 <= si <= Nx && 1 <= sj <= Ny) ||
                                is_solid[si, sj]
                    if src_solid
                        oq = _opp_q(q)
                        φ += phi_mode == 1 ? g_post[i, j, oq] : g_pre[i, j, oq]
                    else
                        φ += g_post[i, j, q]
                    end
                end
            end

            # Step 3: equilibrium at the boundary fluid node. Liu Eq. (39)
            # refers to gᵉᵠᵢ(x_b,t+Δt); in the production driver we use the
            # local hydrodynamic velocity at x_b. The zero-velocity wrapper is
            # retained for legacy diagnostics.
            ge1 = feq_2d(Val(1), φ, ub, vb, usq)
            ge2 = feq_2d(Val(2), φ, ub, vb, usq)
            ge3 = feq_2d(Val(3), φ, ub, vb, usq)
            ge4 = feq_2d(Val(4), φ, ub, vb, usq)
            ge5 = feq_2d(Val(5), φ, ub, vb, usq)
            ge6 = feq_2d(Val(6), φ, ub, vb, usq)
            ge7 = feq_2d(Val(7), φ, ub, vb, usq)
            ge8 = feq_2d(Val(8), φ, ub, vb, usq)
            ge9 = feq_2d(Val(9), φ, ub, vb, usq)

            # Step 4: reconstruct unknown populations via Eq. (39)
            # For each direction q whose source is solid, set
            #   g_post[q] = g^eq[q] + (g_post[opp(q)] - g^eq[opp(q)])
            # This is non-equilibrium bounce-back.
            for q in 2:9
                si = i - _cx_q(q)
                sj = j - _cy_q(q)
                src_solid = !(1 <= si <= Nx && 1 <= sj <= Ny) ||
                            is_solid[si, sj]
                if src_solid
                    oq = _opp_q(q)
                    geq_q  = q == 2 ? ge2 : q == 3 ? ge3 : q == 4 ? ge4 :
                             q == 5 ? ge5 : q == 6 ? ge6 : q == 7 ? ge7 :
                             q == 8 ? ge8 : ge9
                    geq_oq = oq == 2 ? ge2 : oq == 3 ? ge3 : oq == 4 ? ge4 :
                             oq == 5 ? ge5 : oq == 6 ? ge6 : oq == 7 ? ge7 :
                             oq == 8 ? ge8 : ge9
                    q_w = use_q_wall ? q_wall[i, j, oq] : zero(T)
                    if use_q_wall && q_w > zero(T) &&
                       abs(q_w - T(0.5)) > T(100) * eps(T)
                        neq_here_oq = g_pre[i, j, oq] - geq_oq
                        neq_q = zero(T)
                        if q_w <= T(0.5)
                            bi = i - _cx_q(oq)
                            bj = j - _cy_q(oq)
                            geq_back_oq = geq_oq
                            if 1 <= bi <= Nx && 1 <= bj <= Ny && !is_solid[bi, bj]
                                us = use_local_velocity ? ux_bc[bi, bj] : zero(T)
                                vs = use_local_velocity ? uy_bc[bi, bj] : zero(T)
                                Cs = C_field[bi, bj]
                                geq_back_oq = _feq_q_2d(oq, Cs, us, vs, us*us + vs*vs)
                            end
                            neq_back_oq = g_post[i, j, oq] - geq_back_oq
                            neq_q = T(2) * q_w * neq_here_oq +
                                    (one(T) - T(2) * q_w) * neq_back_oq
                        else
                            inv_two_q = one(T) / (T(2) * q_w)
                            neq_here_q = g_pre[i, j, q] - geq_q
                            neq_q = inv_two_q * neq_here_oq +
                                    (one(T) - inv_two_q) * neq_here_q
                        end
                        g_post[i, j, q] = geq_q + neq_q
                    else
                        g_post[i, j, q] = geq_q + (g_post[i, j, oq] - geq_oq)
                    end
                end
            end

            # Step 5: exact rest-population rebalance (Liu Eq. 40).
            nonrest = zero(T)
            for q in 2:9
                nonrest += g_post[i, j, q]
            end
            g_post[i, j, 1] = φ - nonrest

            # Step 6: update macroscopic field
            C_field[i, j] = φ
        end
    end
end

function reset_cutlink_conformation_equilibrium_2d!(g, C_field, is_solid, q_wall)
    backend = KernelAbstractions.get_backend(g)
    Nx, Ny = size(C_field)
    kernel! = reset_cutlink_conformation_equilibrium_2d_kernel!(backend)
    kernel!(g, C_field, C_field, C_field, is_solid, q_wall, false,
            Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
    return nothing
end

function reset_cutlink_conformation_equilibrium_2d!(g, C_field, ux, uy,
                                                    is_solid, q_wall)
    backend = KernelAbstractions.get_backend(g)
    Nx, Ny = size(C_field)
    kernel! = reset_cutlink_conformation_equilibrium_2d_kernel!(backend)
    kernel!(g, C_field, ux, uy, is_solid, q_wall, true,
            Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, C_field)

Conservative non-equilibrium bounce-back (CNEBB) for the conformation tensor
LBM at solid walls. Liu et al. 2025, Eqs. (38-39).

# Arguments
- `g_post` : populations after streaming (modified in place at near-wall fluid cells)
- `g_pre`  : populations before streaming (post-collision of previous step)
- `is_solid` : boolean mask
- `C_field`  : macroscopic conformation component, updated at near-wall cells

The scheme exactly conserves φ (one component of C) at the wall, eliminating
the polymer-stress leakage that plagues simple bounce-back / extrapolation
boundary treatments at high Wi.

Diagnostic `phi_mode` values:
- `:pre_opp` follows the conservative Liu/Yu recovery from outgoing
  pre-streaming populations.
- `:post_opp` uses the post-streaming opposite population for audit sweeps.
- `:field` pins `φ` to the previous macroscopic field at the wall node.
- `:eq_gradient` pins the local equilibrium part to the wall node and keeps
  transported non-equilibrium residuals; useful only for wall-gradient audits.
- `:eq_gradient_cutlink` applies `:eq_gradient` only on explicit `q_wall`
  cut-link cells and falls back to `:pre_opp` on domain walls.
"""
function apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, C_field;
                                      phi_mode::Symbol=:pre_opp)
    phi_mode === :pre_opp &&
        return apply_cnebb_strict_simple_conformation_2d!(g_post, g_pre,
                                                          is_solid, C_field)
    backend = KernelAbstractions.get_backend(g_post)
    Nx, Ny = size(C_field)
    kernel! = apply_cnebb_2d_kernel!(backend)
    kernel!(g_post, g_pre, is_solid, g_post, false, C_field, C_field, C_field,
            false, _cnebb_phi_mode_code(phi_mode), Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

function apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, q_wall, C_field;
                                      phi_mode::Symbol=:pre_opp)
    phi_mode === :pre_opp &&
        return apply_cnebb_qaware_simple_conformation_2d!(g_post, g_pre,
                                                          is_solid, q_wall,
                                                          C_field)
    backend = KernelAbstractions.get_backend(g_post)
    Nx, Ny = size(C_field)
    kernel! = apply_cnebb_2d_kernel!(backend)
    kernel!(g_post, g_pre, is_solid, q_wall, true, C_field, C_field, C_field,
            false, _cnebb_phi_mode_code(phi_mode), Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

function _cnebb_phi_mode_code(phi_mode::Symbol)
    phi_mode === :pre_opp && return 0
    phi_mode === :post_opp && return 1
    phi_mode === :field && return 2
    phi_mode === :eq_gradient && return 3
    phi_mode === :eq_gradient_cutlink && return 4
    error("unknown CNEBB phi mode $(phi_mode); expected :pre_opp, :post_opp, :field, :eq_gradient, or :eq_gradient_cutlink")
end

"""
    apply_cnebb_strict_simple_conformation_2d!(g_post, g_pre, is_solid, C_field, ux, uy)

Specialized Yu/CNEBB strict path for the default `:pre_opp` formulation.
It is algebraically equivalent to `apply_cnebb_conformation_2d!` with
`phi_mode=:pre_opp` and no `q_wall`, but avoids the diagnostic branches that
can make Metal compilation fragile.
"""
function apply_cnebb_strict_simple_conformation_2d!(g_post, g_pre, is_solid,
                                                    C_field, ux, uy)
    backend = KernelAbstractions.get_backend(g_post)
    Nx, Ny = size(C_field)
    kernel! = apply_cnebb_strict_simple_2d_kernel!(backend)
    kernel!(g_post, g_pre, is_solid, C_field, ux, uy, true,
            Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

function apply_cnebb_strict_simple_conformation_2d!(g_post, g_pre, is_solid,
                                                    C_field)
    backend = KernelAbstractions.get_backend(g_post)
    Nx, Ny = size(C_field)
    kernel! = apply_cnebb_strict_simple_2d_kernel!(backend)
    kernel!(g_post, g_pre, is_solid, C_field, C_field, C_field, false,
            Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

function apply_cnebb_qaware_simple_conformation_2d!(g_post, g_pre, is_solid,
                                                    q_wall, C_field, ux, uy)
    backend = KernelAbstractions.get_backend(g_post)
    Nx, Ny = size(C_field)
    apply_cnebb_strict_simple_conformation_2d!(g_post, g_pre, is_solid,
                                               C_field, ux, uy)
    dir_kernel! = apply_cnebb_qaware_dir_2d_kernel!(backend)
    for q in 2:9
        q_out = _opp_q(q)
        dir_kernel!(g_post, g_pre, is_solid, q_wall, C_field, ux, uy, true,
                    q, q_out, _cx_q(q), _cy_q(q), _cx_q(q_out), _cy_q(q_out),
                    eltype(g_post)(_w_q(q)), eltype(g_post)(_w_q(q_out)),
                    Nx, Ny; ndrange=(Nx, Ny))
    end
    rebalance_kernel! = rebalance_wall_rest_to_field_2d_kernel!(backend)
    rebalance_kernel!(g_post, is_solid, C_field, Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

function apply_cnebb_qaware_simple_conformation_2d!(g_post, g_pre, is_solid,
                                                    q_wall, C_field)
    backend = KernelAbstractions.get_backend(g_post)
    Nx, Ny = size(C_field)
    apply_cnebb_strict_simple_conformation_2d!(g_post, g_pre, is_solid,
                                               C_field)
    dir_kernel! = apply_cnebb_qaware_dir_2d_kernel!(backend)
    for q in 2:9
        q_out = _opp_q(q)
        dir_kernel!(g_post, g_pre, is_solid, q_wall, C_field, C_field, C_field,
                    false, q, q_out, _cx_q(q), _cy_q(q), _cx_q(q_out),
                    _cy_q(q_out), eltype(g_post)(_w_q(q)),
                    eltype(g_post)(_w_q(q_out)), Nx, Ny; ndrange=(Nx, Ny))
    end
    rebalance_kernel! = rebalance_wall_rest_to_field_2d_kernel!(backend)
    rebalance_kernel!(g_post, is_solid, C_field, Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

function apply_cnebb_field_conformation_2d!(g_post, g_pre, is_solid, C_field,
                                            ux, uy)
    backend = KernelAbstractions.get_backend(g_post)
    Nx, Ny = size(C_field)
    dir_kernel! = apply_cnebb_field_qaware_dir_2d_kernel!(backend)
    for q in 2:9
        q_out = _opp_q(q)
        dir_kernel!(g_post, g_pre, is_solid, g_post, false, C_field, ux, uy,
                    true, q, q_out, _cx_q(q), _cy_q(q), _cx_q(q_out),
                    _cy_q(q_out), eltype(g_post)(_w_q(q)),
                    eltype(g_post)(_w_q(q_out)), Nx, Ny; ndrange=(Nx, Ny))
    end
    rebalance_kernel! = rebalance_wall_rest_to_field_2d_kernel!(backend)
    rebalance_kernel!(g_post, is_solid, C_field, Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

function apply_cnebb_field_conformation_2d!(g_post, g_pre, is_solid, C_field)
    backend = KernelAbstractions.get_backend(g_post)
    Nx, Ny = size(C_field)
    dir_kernel! = apply_cnebb_field_qaware_dir_2d_kernel!(backend)
    for q in 2:9
        q_out = _opp_q(q)
        dir_kernel!(g_post, g_pre, is_solid, g_post, false, C_field,
                    C_field, C_field, false, q, q_out, _cx_q(q), _cy_q(q),
                    _cx_q(q_out), _cy_q(q_out), eltype(g_post)(_w_q(q)),
                    eltype(g_post)(_w_q(q_out)), Nx, Ny; ndrange=(Nx, Ny))
    end
    rebalance_kernel! = rebalance_wall_rest_to_field_2d_kernel!(backend)
    rebalance_kernel!(g_post, is_solid, C_field, Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

function apply_cnebb_field_conformation_2d!(g_post, g_pre, is_solid, q_wall,
                                            C_field, ux, uy)
    backend = KernelAbstractions.get_backend(g_post)
    Nx, Ny = size(C_field)
    dir_kernel! = apply_cnebb_field_qaware_dir_2d_kernel!(backend)
    for q in 2:9
        q_out = _opp_q(q)
        dir_kernel!(g_post, g_pre, is_solid, q_wall, true, C_field, ux, uy,
                    true, q, q_out, _cx_q(q), _cy_q(q), _cx_q(q_out),
                    _cy_q(q_out), eltype(g_post)(_w_q(q)),
                    eltype(g_post)(_w_q(q_out)), Nx, Ny; ndrange=(Nx, Ny))
    end
    rebalance_kernel! = rebalance_wall_rest_to_field_2d_kernel!(backend)
    rebalance_kernel!(g_post, is_solid, C_field, Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

function apply_cnebb_field_conformation_2d!(g_post, g_pre, is_solid, q_wall,
                                            C_field)
    backend = KernelAbstractions.get_backend(g_post)
    Nx, Ny = size(C_field)
    dir_kernel! = apply_cnebb_field_qaware_dir_2d_kernel!(backend)
    for q in 2:9
        q_out = _opp_q(q)
        dir_kernel!(g_post, g_pre, is_solid, q_wall, true, C_field,
                    C_field, C_field, false, q, q_out, _cx_q(q), _cy_q(q),
                    _cx_q(q_out), _cy_q(q_out), eltype(g_post)(_w_q(q)),
                    eltype(g_post)(_w_q(q_out)), Nx, Ny; ndrange=(Nx, Ny))
    end
    rebalance_kernel! = rebalance_wall_rest_to_field_2d_kernel!(backend)
    rebalance_kernel!(g_post, is_solid, C_field, Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

"""
    apply_extrap_eq_conformation_2d!(g_post, is_solid, q_wall, C_field, ux, uy)

Fill missing cut-link conformation populations from wall-aware extrapolated
equilibrium states, then rebalance the rest population to `C_field`.
"""
function apply_extrap_eq_conformation_2d!(g_post, is_solid, q_wall, C_field,
                                          ux, uy)
    backend = KernelAbstractions.get_backend(g_post)
    Nx, Ny = size(C_field)
    kernel! = apply_extrap_eq_conformation_2d_kernel!(backend)
    kernel!(g_post, is_solid, q_wall, C_field, ux, uy,
            Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

function apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, C_field, ux, uy;
                                      phi_mode::Symbol=:pre_opp)
    phi_mode === :pre_opp &&
        return apply_cnebb_strict_simple_conformation_2d!(g_post, g_pre,
                                                          is_solid, C_field,
                                                          ux, uy)
    backend = KernelAbstractions.get_backend(g_post)
    Nx, Ny = size(C_field)
    kernel! = apply_cnebb_2d_kernel!(backend)
    kernel!(g_post, g_pre, is_solid, g_post, false, C_field, ux, uy,
            true, _cnebb_phi_mode_code(phi_mode), Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

function apply_cnebb_conformation_2d!(g_post, g_pre, is_solid, q_wall, C_field,
                                      ux, uy; phi_mode::Symbol=:pre_opp)
    phi_mode === :pre_opp &&
        return apply_cnebb_qaware_simple_conformation_2d!(g_post, g_pre,
                                                          is_solid, q_wall,
                                                          C_field, ux, uy)
    backend = KernelAbstractions.get_backend(g_post)
    Nx, Ny = size(C_field)
    kernel! = apply_cnebb_2d_kernel!(backend)
    kernel!(g_post, g_pre, is_solid, q_wall, true, C_field, ux, uy,
            true, _cnebb_phi_mode_code(phi_mode), Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

"""
    apply_ylw_a_conformation_2d!(g_post, g_pre, is_solid, q_wall, C_field, ux, uy; tau_plus=1)

Yu-Li-Wen 2020 modified curved-wall scheme A, transposed to one scalar
conformation component: MLS reconstruction (Eq. 17) followed by rest
population leakage correction (Eq. 18).
"""
function apply_ylw_a_conformation_2d!(g_post, g_pre, is_solid, q_wall,
                                      C_field, ux, uy; tau_plus=1)
    backend = KernelAbstractions.get_backend(g_post)
    Nx, Ny = size(C_field)
    dir_kernel! = apply_ylw_a_dir_2d_kernel!(backend)
    for q in 2:9
        q_out = _opp_q(q)
        dir_kernel!(g_post, g_pre, is_solid, q_wall, C_field, ux, uy, true,
                    eltype(g_post)(tau_plus), q, q_out, _cx_q(q), _cy_q(q),
                    _cx_q(q_out), _cy_q(q_out), eltype(g_post)(_w_q(q_out)),
                    Nx, Ny; ndrange=(Nx, Ny))
    end
    rebalance_kernel! = apply_ylw_a_rest_balance_2d_kernel!(backend)
    rebalance_kernel!(g_post, g_pre, is_solid, C_field, Nx, Ny;
                      ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

function apply_ylw_a_conformation_2d!(g_post, g_pre, is_solid, q_wall,
                                      C_field; tau_plus=1)
    backend = KernelAbstractions.get_backend(g_post)
    Nx, Ny = size(C_field)
    dir_kernel! = apply_ylw_a_dir_2d_kernel!(backend)
    for q in 2:9
        q_out = _opp_q(q)
        dir_kernel!(g_post, g_pre, is_solid, q_wall, C_field, C_field, C_field,
                    false, eltype(g_post)(tau_plus), q, q_out, _cx_q(q),
                    _cy_q(q), _cx_q(q_out), _cy_q(q_out),
                    eltype(g_post)(_w_q(q_out)), Nx, Ny; ndrange=(Nx, Ny))
    end
    rebalance_kernel! = apply_ylw_a_rest_balance_2d_kernel!(backend)
    rebalance_kernel!(g_post, g_pre, is_solid, C_field, Nx, Ny;
                      ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

"""
    apply_ylw_b_conformation_2d!(g_post, g_pre, is_solid, q_wall, C_field, ux, uy; tau_plus=1)

Yu-Li-Wen 2020 modified curved-wall scheme B, transposed to one scalar
conformation component: MLS reconstruction (Eq. 17) with the fictitious
ghost density chosen from Eqs. 20-22 so that incoming and outgoing wall
fluxes balance locally.
"""
function apply_ylw_b_conformation_2d!(g_post, g_pre, is_solid, q_wall,
                                      C_field, ux, uy; tau_plus=1)
    backend = KernelAbstractions.get_backend(g_post)
    Nx, Ny = size(C_field)
    num = similar(C_field)
    den = similar(C_field)
    fill!(num, zero(eltype(g_post)))
    fill!(den, zero(eltype(g_post)))
    acc_kernel! = apply_ylw_b_accumulate_dir_2d_kernel!(backend)
    for q in 2:9
        q_out = _opp_q(q)
        acc_kernel!(num, den, g_pre, is_solid, q_wall, C_field, ux, uy, true,
                    eltype(g_post)(tau_plus), q, q_out, _cx_q(q), _cy_q(q),
                    _cx_q(q_out), _cy_q(q_out), eltype(g_post)(_w_q(q_out)),
                    Nx, Ny; ndrange=(Nx, Ny))
    end
    phi_kernel! = apply_ylw_b_phi_from_accum_2d_kernel!(backend)
    phi_kernel!(num, den, is_solid, C_field, Nx, Ny; ndrange=(Nx, Ny))
    dir_kernel! = apply_ylw_a_dir_2d_kernel!(backend)
    for q in 2:9
        q_out = _opp_q(q)
        dir_kernel!(g_post, g_pre, is_solid, q_wall, C_field, ux, uy, true,
                    eltype(g_post)(tau_plus), q, q_out, _cx_q(q), _cy_q(q),
                    _cx_q(q_out), _cy_q(q_out), eltype(g_post)(_w_q(q_out)),
                    Nx, Ny; ndrange=(Nx, Ny))
    end
    sum_kernel! = compute_wall_macro_sum_2d_kernel!(backend)
    sum_kernel!(g_post, is_solid, C_field, Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

function apply_ylw_b_conformation_2d!(g_post, g_pre, is_solid, q_wall,
                                      C_field; tau_plus=1)
    backend = KernelAbstractions.get_backend(g_post)
    Nx, Ny = size(C_field)
    num = similar(C_field)
    den = similar(C_field)
    fill!(num, zero(eltype(g_post)))
    fill!(den, zero(eltype(g_post)))
    acc_kernel! = apply_ylw_b_accumulate_dir_2d_kernel!(backend)
    for q in 2:9
        q_out = _opp_q(q)
        acc_kernel!(num, den, g_pre, is_solid, q_wall, C_field, C_field, C_field,
                    false, eltype(g_post)(tau_plus), q, q_out, _cx_q(q),
                    _cy_q(q), _cx_q(q_out), _cy_q(q_out),
                    eltype(g_post)(_w_q(q_out)), Nx, Ny; ndrange=(Nx, Ny))
    end
    phi_kernel! = apply_ylw_b_phi_from_accum_2d_kernel!(backend)
    phi_kernel!(num, den, is_solid, C_field, Nx, Ny; ndrange=(Nx, Ny))
    dir_kernel! = apply_ylw_a_dir_2d_kernel!(backend)
    for q in 2:9
        q_out = _opp_q(q)
        dir_kernel!(g_post, g_pre, is_solid, q_wall, C_field, C_field, C_field,
                    false, eltype(g_post)(tau_plus), q, q_out, _cx_q(q),
                    _cy_q(q), _cx_q(q_out), _cy_q(q_out),
                    eltype(g_post)(_w_q(q_out)), Nx, Ny; ndrange=(Nx, Ny))
    end
    sum_kernel! = compute_wall_macro_sum_2d_kernel!(backend)
    sum_kernel!(g_post, is_solid, C_field, Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

"""
    apply_ylw_balance_conformation_2d!(g_post, g_pre, is_solid, q_wall, C_field, ux, uy)

Diagnostic balance-only variant: local NEBB reconstruction for missing
populations, then YLW scheme-A rest-population incoming/outgoing correction.
"""
function apply_ylw_balance_conformation_2d!(g_post, g_pre, is_solid, q_wall,
                                            C_field, ux, uy)
    backend = KernelAbstractions.get_backend(g_post)
    Nx, Ny = size(C_field)
    kernel! = apply_ylw_balance_2d_kernel!(backend)
    kernel!(g_post, g_pre, is_solid, q_wall, C_field, ux, uy, true,
            Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

function apply_ylw_balance_conformation_2d!(g_post, g_pre, is_solid, q_wall,
                                            C_field)
    backend = KernelAbstractions.get_backend(g_post)
    Nx, Ny = size(C_field)
    kernel! = apply_ylw_balance_2d_kernel!(backend)
    kernel!(g_post, g_pre, is_solid, q_wall, C_field, C_field, C_field,
            false, Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end
