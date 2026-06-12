# ============================================================================
# src/platform/calibration.jl
# M-P2b-2: ParameterSpace, loss, fit, CalibResult
# ============================================================================

"""
    ParameterSpace(names, lower, upper; log_scale=nothing, fixed=nothing)

Bookkeeping for a named parameter vector with bounds, optional log-space
coordinates, and fixed/free masks.

`to_flat(ps, p)` packs only free parameters. `from_flat(ps, x, p0)` unpacks
`x` into a `NamedTuple`, reading fixed values from `p0`. `project!(ps, x)`
clips a flat vector to the feasible box, respecting log-space coordinates.
"""
struct ParameterSpace
    names::Vector{Symbol}
    lower::Vector{Float64}
    upper::Vector{Float64}
    log_scale::BitVector
    fixed::BitVector
end

function ParameterSpace(names, lower, upper;
                        log_scale=falses(length(names)),
                        fixed=falses(length(names)))
    n = length(names)
    length(lower) == n || throw(ArgumentError("lower must have length $n"))
    length(upper) == n || throw(ArgumentError("upper must have length $n"))
    length(log_scale) == n || throw(ArgumentError("log_scale must have length $n"))
    length(fixed) == n || throw(ArgumentError("fixed must have length $n"))

    names_v = [Symbol(name) for name in names]
    lower_v = Float64.(collect(lower))
    upper_v = Float64.(collect(upper))
    log_v = BitVector(log_scale)
    fixed_v = BitVector(fixed)

    for i in eachindex(names_v)
        lower_v[i] <= upper_v[i] ||
            throw(ArgumentError("lower[$i] must be <= upper[$i]"))
        if log_v[i] && !(lower_v[i] > 0.0 && upper_v[i] > 0.0)
            throw(ArgumentError("log-scale bounds for $(names_v[i]) must be positive"))
        end
    end

    return ParameterSpace(names_v, lower_v, upper_v, log_v, fixed_v)
end

"""Return the number of free optimisation variables in `ps`."""
n_free(ps::ParameterSpace) = count(!, ps.fixed)

function _ps_get(named::NamedTuple, name::Symbol)
    haskey(named, name) || throw(KeyError(name))
    return getfield(named, name)
end

"""Pack free parameters from a named tuple into a flat vector."""
function to_flat(ps::ParameterSpace, named::NamedTuple)
    v = Vector{Float64}(undef, n_free(ps))
    j = 0
    for i in eachindex(ps.names)
        ps.fixed[i] && continue
        j += 1
        val = Float64(_ps_get(named, ps.names[i]))
        if ps.log_scale[i]
            val > 0.0 || throw(ArgumentError("log-scale parameter $(ps.names[i]) must be positive"))
            v[j] = log(val)
        else
            v[j] = val
        end
    end
    return v
end

"""
    from_flat(ps, x, p_fixed)

Unpack free coordinates from `x` and fixed coordinates from `p_fixed`.
Returned values are in natural coordinates and clipped to `ps` bounds.
"""
function from_flat(ps::ParameterSpace, v::AbstractVector{<:Real},
                   p_fixed::NamedTuple)
    length(v) == n_free(ps) ||
        throw(ArgumentError("flat vector length $(length(v)) does not match n_free=$(n_free(ps))"))

    j = 0
    vals = Vector{Float64}(undef, length(ps.names))
    for i in eachindex(ps.names)
        name = ps.names[i]
        if ps.fixed[i]
            vals[i] = Float64(_ps_get(p_fixed, name))
        else
            j += 1
            val = Float64(v[j])
            val = ps.log_scale[i] ? exp(val) : val
            vals[i] = clamp(val, ps.lower[i], ps.upper[i])
        end
    end
    return NamedTuple{Tuple(ps.names)}(Tuple(vals))
end

function from_flat(ps::ParameterSpace, v::AbstractVector{<:Real})
    any(ps.fixed) &&
        throw(ArgumentError("from_flat(ps, v) needs p_fixed when ps has fixed parameters"))
    return from_flat(ps, v, (;))
end

"""Project a flat vector in-place onto the bounded feasible set."""
function project!(ps::ParameterSpace, v::AbstractVector{Float64})
    length(v) == n_free(ps) ||
        throw(ArgumentError("flat vector length $(length(v)) does not match n_free=$(n_free(ps))"))

    j = 0
    for i in eachindex(ps.names)
        ps.fixed[i] && continue
        j += 1
        if ps.log_scale[i]
            v[j] = clamp(v[j], log(ps.lower[i]), log(ps.upper[i]))
        else
            v[j] = clamp(v[j], ps.lower[i], ps.upper[i])
        end
    end
    return v
end

function _sumsq_delta(y_hat, y)
    d = y_hat .- y
    return d isa Number ? abs2(d) : sum(abs2, d)
end

"""
    loss(predictions::Vector{<:Prediction}, data; weights=nothing) -> Float64

Data-misfit loss `L = (1/2) sum_i w_i ||predictions[i].value - data[i].value||^2`.
Each `data[i]` must provide a `value` field. Weights default to one.
"""
function loss(predictions::AbstractVector{<:Prediction}, data; weights=nothing)
    length(predictions) == length(data) ||
        throw(ArgumentError("predictions and data must have the same length"))
    if weights !== nothing && length(weights) != length(predictions)
        throw(ArgumentError("weights must match predictions length"))
    end

    L = 0.0
    for i in eachindex(predictions)
        w = weights === nothing ? 1.0 : Float64(weights[i])
        L += 0.5 * w * _sumsq_delta(predictions[i].value, data[i].value)
    end
    return L
end

const _D2Q9_CX = (0.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, -1.0, 1.0)

function _extract_ux_from_f(f::Array{Float64,3})
    Nx, Ny, _ = size(f)
    ux = zeros(Float64, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        rho = 0.0
        momx = 0.0
        for q in 1:9
            fq = f[i, j, q]
            rho += fq
            momx += _D2Q9_CX[q] * fq
        end
        ux[i, j] = momx / rho
    end
    return ux
end

function _obs_lineprofile_ux_from_f(f::Array{Float64,3}, obs::LineProfile)
    ux = _extract_ux_from_f(f)
    vals = Vector{Float64}(undef, length(obs.indices))
    for (k, idx) in enumerate(obs.indices)
        vals[k] = ux[idx[1], idx[2]]
    end
    return Prediction(obs, vals)
end

function _is_cd_observable(obs)
    return obs isa FieldReduction && obs.field === :Cd && obs.reducer === identity
end

"""
    _dJ_df_lineprofile_ux(f_star, pred_ux, data_ux, indices) -> Array{Float64,3}

Assemble `dL/df` for `L = (1/2)||ux_pred(indices) - ux_data||^2`.
Nonzero entries appear only at the observed `(i,j)` indices.
"""
function _dJ_df_lineprofile_ux(f_star::Array{Float64,3},
                               pred_ux::AbstractVector{<:Real},
                               data_ux::AbstractVector{<:Real},
                               indices)
    dLdf = zeros(Float64, size(f_star))
    for (k, idx) in enumerate(indices)
        i, j = idx[1], idx[2]
        rho_ij = sum(@view f_star[i, j, :])
        ux_ij = Float64(pred_ux[k])
        res = ux_ij - Float64(data_ux[k])
        for q in 1:9
            dLdf[i, j, q] = res * (_D2Q9_CX[q] - ux_ij) / rho_ij
        end
    end
    return dLdf
end

"""
    CalibResult

Result of [`fit`](@ref): optimal natural-space parameters, final loss, loss and
gradient traces, iteration count, convergence flag, and a diagnostic message.
"""
struct CalibResult
    p_opt::NamedTuple
    loss_final::Float64
    loss_trace::Vector{Float64}
    grad_trace::Vector{Float64}
    n_iter::Int
    converged::Bool
    message::String
end

function _fit_weight(weights, i::Int)
    return weights === nothing ? 1.0 : Float64(weights[i])
end

function _insert_fit_params!(kw::Dict{Symbol,Any}, p_named::NamedTuple)
    for (k, v) in pairs(p_named)
        if k === :ν
            kw[:nu] = v
        elseif k === :radius || k === :u_in || k === :rho_out || k === :inlet ||
               k === :cx || k === :cy || k === :Nx || k === :Ny || k === :nu
            kw[k] = v
        end
    end
    return kw
end

"""
    fit(problem, method::LBM, data, p0, pspace; observables, kwargs...) -> CalibResult

Calibrate free parameters in `pspace` with projected gradient descent and
Armijo backtracking. `problem` must be a `NamedTuple` of fixed
`ad_forward_solve` geometry keywords; free parameters from `p0`/`pspace`
override matching entries (`:ν` maps to `nu`).

Currently supported observables are `LineProfile(:ux, indices)` and
`FieldReduction(:Cd, identity)`. Gradients use the steady adjoint chain:
`_ad_pvjp_nu` for scalar viscosity and `_ad_dqwall_terms` for radius.
"""
function fit(problem, ::LBM, data, p0::NamedTuple,
             pspace::ParameterSpace;
             observables,
             weights=nothing,
             max_iter::Int=100,
             step_size::Real=0.5,
             armijo_c::Real=1e-4,
             armijo_rho::Real=0.5,
             armijo_max_steps::Int=20,
             gtol::Real=1e-6,
             ftol::Real=1e-12,
             forward_tol::Real=1e-12,
             forward_max_steps::Int=200_000,
             gmres_tol::Real=1e-11,
             adjoint_tol::Real=AD_LINEAR_RES_TOL,
             gmres_restart::Int=240,
             gmres_max_restarts::Int=20,
             verbose::Bool=false)
    problem isa NamedTuple ||
        throw(ArgumentError("fit expects problem::NamedTuple with ad_forward_solve geometry keywords"))
    length(observables) == length(data) ||
        throw(ArgumentError("observables and data must have the same length"))
    if weights !== nothing && length(weights) != length(observables)
        throw(ArgumentError("weights must match observables length"))
    end

    x = project!(pspace, copy(to_flat(pspace, p0)))
    isempty(x) && throw(ArgumentError("ParameterSpace has no free parameters"))
    loss_trace = Float64[]
    grad_trace = Float64[]

    function forward_at(p_named::NamedTuple; f_init=nothing)
        kw = Dict{Symbol,Any}(pairs(problem))
        _insert_fit_params!(kw, p_named)
        kw[:tol] = Float64(forward_tol)
        kw[:max_steps] = forward_max_steps
        f_init === nothing || (kw[:f_init] = f_init)
        haskey(kw, :nu) ||
            throw(ArgumentError("fit needs a scalar viscosity from pspace name :ν or problem key :nu"))
        fwd = ad_forward_solve(; kw...)
        fwd.converged || @warn "fit: forward solve did not converge" residual=fwd.residual
        return fwd
    end

    function build_preds(fwd)
        preds = Prediction[]
        for obs in observables
            if obs isa LineProfile && obs.field === :ux
                push!(preds, _obs_lineprofile_ux_from_f(fwd.f_star, obs))
            elseif _is_cd_observable(obs)
                push!(preds, Prediction(obs, fwd.Cd))
            else
                error("fit: unsupported observable $(typeof(obs)); supported: LineProfile(:ux), FieldReduction(:Cd, identity)")
            end
        end
        return preds
    end

    function eval_at(x_flat; f_init=nothing)
        p_named = from_flat(pspace, x_flat, p0)
        fwd = forward_at(p_named; f_init=f_init)
        preds = build_preds(fwd)
        return loss(preds, data; weights=weights), preds, fwd, p_named
    end

    function compute_gradient_flat(fwd, preds, p_named)
        dLdf = zeros(Float64, size(fwd.f_star))
        cd_residual = 0.0

        for i in eachindex(observables)
            obs = observables[i]
            w = _fit_weight(weights, i)
            if obs isa LineProfile && obs.field === :ux
                dLdf .+= w .* _dJ_df_lineprofile_ux(fwd.f_star, preds[i].value,
                                                     data[i].value, obs.indices)
            elseif _is_cd_observable(obs)
                res = w * (Float64(preds[i].value) - Float64(data[i].value))
                dLdf .+= res .* _ad_dJdf(fwd.f_star, fwd.q_wall, fwd.u_ref,
                                         fwd.D, fwd.Nx, fwd.Ny)
                cd_residual += res
            else
                error("fit: gradient not implemented for observable $(typeof(obs))")
            end
        end

        apply_GtT = v -> _ad_vjp_GtT(fwd.f_star, v, fwd.q_wall, fwd.is_solid,
                                      fwd.u_profile, fwd.rho_out,
                                      fwd.s_plus, fwd.s_minus, fwd.Nx, fwd.Ny)
        adj = gmres_adjoint(apply_GtT, dLdf; tol=gmres_tol,
                            restart=gmres_restart,
                            max_restarts=gmres_max_restarts,
                            linear_tol=adjoint_tol)
        adj.converged || @warn "fit: adjoint solve did not converge" linres=adj.linres

        λ = adj.lambda
        g_flat = Float64[]
        for i in eachindex(pspace.names)
            pspace.fixed[i] && continue
            name = pspace.names[i]
            if name === :ν
                geom = LBMGeomParams(fwd.q_wall, fwd.is_solid, fwd.u_profile,
                                     fwd.rho_out, fwd.s_plus, fwd.s_minus,
                                     fwd.Nx, fwd.Ny)
                dL_dν = _ad_pvjp_nu(fwd.f_star, λ, LBMScalarParams(geom, p_named[:ν]))
                push!(g_flat, pspace.log_scale[i] ? dL_dν * p_named[:ν] : dL_dν)
            elseif name === :radius
                dq_dR = dq_wall_dR_cylinder(fwd.Nx, fwd.Ny, fwd.cx, fwd.cy,
                                            fwd.radius; FT=Float64)
                qwall_terms = _ad_dqwall_terms(fwd.f_star, λ, fwd.q_wall,
                                               fwd.is_solid, fwd.u_profile,
                                               fwd.rho_out, fwd.s_plus,
                                               fwd.s_minus, fwd.Nx, fwd.Ny,
                                               fwd.u_ref, fwd.D, dq_dR)
                terms = ad_assemble_radius_terms(fwd.Cd, fwd.radius, dq_dR,
                                                 qwall_terms.explicit,
                                                 qwall_terms.implicit)
                dL_dR = cd_residual * (terms.explicit_qwall + terms.direct_D) +
                         terms.implicit_qwall
                push!(g_flat, pspace.log_scale[i] ? dL_dR * p_named[:radius] : dL_dR)
            else
                error("fit: gradient not implemented for parameter :$(name)")
            end
        end
        return g_flat
    end

    L, preds, fwd, p_named = eval_at(x)
    converged = false
    message = "max_iter reached"

    # Barzilai-Borwein spectral step: updated each iteration from (Δx, Δg).
    # On iter=1 (no previous pair), falls back to step_size.
    # On iter≥2, α_BB = (Δx⋅Δg) / (Δg⋅Δg)  (BB2 / inverse formula),
    # clamped to [step_size * 1e-6, step_size * 1e3] to stay reasonable.
    # Armijo backtracking is still applied as a safeguard after the BB trial.
    x_prev = copy(x)
    g_prev = zeros(Float64, length(x))

    for iter in 1:max_iter
        g_flat = compute_gradient_flat(fwd, preds, p_named)
        gnorm = norm(g_flat)
        push!(loss_trace, L)
        push!(grad_trace, gnorm)
        verbose && @info "fit" iter L gnorm p_named

        if gnorm < gtol
            converged = true
            message = "gradient norm < gtol=$gtol"
            break
        end

        # Spectral (BB2) step: α = (Δx⋅Δg) / ||Δg||² — valid from iter 2 onward.
        α_bb = Float64(step_size)
        if iter >= 2
            Δx = x .- x_prev
            Δg = g_flat .- g_prev
            dgg = dot(Δg, Δg)
            if dgg > 0.0
                α_candidate = dot(Δx, Δg) / dgg
                # Clamp to a sensible range around the user's step_size
                α_bb = clamp(α_candidate, Float64(step_size) * 1e-6, Float64(step_size) * 1e3)
            end
        end

        accepted = false
        L_new = L
        α = α_bb
        x_new = copy(x)
        preds_new = preds
        fwd_new = fwd
        p_new = p_named

        for _bt in 1:armijo_max_steps
            trial = project!(pspace, copy(x .- α .* g_flat))
            L_trial, preds_trial, fwd_trial, p_trial = eval_at(trial; f_init=fwd.f_star)
            if L_trial <= L - Float64(armijo_c) * α * gnorm^2
                accepted = true
                L_new = L_trial
                x_new = trial
                preds_new = preds_trial
                fwd_new = fwd_trial
                p_new = p_trial
                break
            end
            α *= Float64(armijo_rho)
        end

        if !accepted
            message = "Armijo backtracking failed after $armijo_max_steps steps"
            break
        end

        ΔL = abs(L - L_new)
        x_prev = copy(x)
        g_prev = copy(g_flat)
        x = x_new
        L = L_new
        preds = preds_new
        fwd = fwd_new
        p_named = p_new

        if ΔL < ftol
            converged = true
            message = "loss change |ΔL| < ftol=$ftol"
            break
        end
    end

    if isempty(loss_trace) || loss_trace[end] != L
        push!(loss_trace, L)
        push!(grad_trace, 0.0)
    end

    return CalibResult(from_flat(pspace, x, p0), L, loss_trace, grad_trace,
                       length(loss_trace), converged, message)
end
