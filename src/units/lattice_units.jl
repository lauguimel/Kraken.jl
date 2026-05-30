const _SQRT3 = sqrt(3.0)

function _required(kw, key::Symbol)
    _has(kw, key) && return _get(kw, key, nothing)
    throw(ArgumentError("missing required units keyword `$key`"))
end

function _first_value(x)
    x isa Tuple && return first(x)
    x isa AbstractVector && return first(x)
    return x
end

function _length_gt_one(x)
    x === nothing && return false
    if x isa Tuple || x isa AbstractVector
        return length(x) > 1
    end
    return false
end

function _r_lu(kw)
    raw = _get(kw, :R_LU, _get(kw, :R, _get(kw, :radius, nothing)))
    raw === nothing && throw(ArgumentError("missing required units keyword `R_LU`"))
    return round(Int, _first_value(raw))
end

function _is_r_sweep(kw)
    _length_gt_one(_get(kw, :sweep_R, nothing)) && return true
    return _length_gt_one(_get(kw, :R_LU, nothing))
end

function _resolve_scaling(kw)
    scaling = _sym(_get(kw, :scaling, :auto))
    scaling === :auto && return _is_r_sweep(kw) ? :diffusive : :acoustic
    scaling in (:diffusive, :acoustic) ||
        throw(ArgumentError("scaling must be :auto, :diffusive, or :acoustic; got :$scaling"))
    return scaling
end

_viscosity_factor(beta, bsd_fraction) = beta + bsd_fraction * (one(beta) - beta)
_tau_from_nu(nu, fnu) = 0.5 + 3 * fnu * nu
_nu_from_tau(tau, fnu) = (tau - 0.5) / (3 * fnu)

function _max_steps(kw, geom::GeometryDescriptor, R_LU::Int, u, lambda)
    n_ft = _get(kw, :n_flow_through, _get(kw, :n_FT, 1.0))
    flow_steps = ceil(Int, Float64(n_ft) * (geom.L_up + geom.L_down) * R_LU / Float64(u))
    lambda_steps = isfinite(Float64(lambda)) ? ceil(Int, 5 * Float64(lambda)) : 0
    required = max(flow_steps, lambda_steps)
    provided = _get(kw, :max_steps, nothing)
    provided === nothing && return required
    return max(round(Int, provided), required)
end

function _real_factor(kw, key::Symbol, ::Type{T}) where {T}
    value = _get(kw, key, nothing)
    value === nothing && return T(NaN)
    return T(value)
end

function _make_units(::Type{T}; tau, nu, u, R_LU, scaling, max_steps,
                     nu_s=NaN, nu_p=NaN, lambda=NaN,
                     alpha=NaN, beta_thermal=NaN, kw=NamedTuple()) where {T}
    return LBMUnits{T}(
        T(tau),
        T(nu),
        T(u),
        Int(R_LU),
        T(u * _SQRT3),
        scaling,
        Int(max_steps),
        T(nu_s),
        T(nu_p),
        T(lambda),
        T(alpha),
        T(beta_thermal),
        _real_factor(kw, :dx_real, T),
        _real_factor(kw, :dt_real, T),
        _real_factor(kw, :rho_real, T),
    )
end

function nondim_to_lu(spec::NewtonianSpec{T}, kw, geom::GeometryDescriptor) where {T}
    R = _r_lu(kw)
    scaling = _resolve_scaling(kw)
    if scaling === :diffusive
        tau = T(_get(kw, :tau_target, 0.95))
        nu = T(_nu_from_tau(tau, one(T)))
        u = T(spec.Re * nu / R)
    else
        u = T(_get(kw, :u_target, _get(kw, :u_mean, 0.005)))
        nu = T(u * R / spec.Re)
        tau = T(_tau_from_nu(nu, one(T)))
    end
    max_steps = _max_steps(kw, geom, R, u, T(NaN))
    return _make_units(T; tau=tau, nu=nu, u=u, R_LU=R, scaling=scaling,
                       max_steps=max_steps, kw=kw)
end

function nondim_to_lu(spec::ViscoelasticSpec{T}, kw,
                      geom::GeometryDescriptor) where {T}
    R = _r_lu(kw)
    scaling = _resolve_scaling(kw)
    fnu = _viscosity_factor(spec.beta, spec.bsd_fraction)
    if scaling === :diffusive
        tau = T(_get(kw, :tau_target, 0.95))
        nu = T(_nu_from_tau(tau, fnu))
        u = T(spec.Re * nu / R)
    else
        u = T(_get(kw, :u_target, _get(kw, :u_mean, 0.005)))
        nu = T(u * R / spec.Re)
        tau = T(_tau_from_nu(nu, fnu))
    end
    nu_s = T(spec.beta * nu)
    nu_p = T((one(T) - spec.beta) * nu)
    lambda = T(spec.Wi * R / u)
    max_steps = _max_steps(kw, geom, R, u, lambda)
    return _make_units(T; tau=tau, nu=nu, u=u, R_LU=R, scaling=scaling,
                       max_steps=max_steps, nu_s=nu_s, nu_p=nu_p,
                       lambda=lambda, kw=kw)
end

function _driver_value(kw, keys::Tuple)
    for key in keys
        value = _get(kw, key, nothing)
        value === nothing || return value
    end
    return nothing
end

function lu_to_nondim(::Type{NewtonianSpec}, kw, geom::GeometryDescriptor)
    T = _get(kw, :T, Float64)
    R = _r_lu(kw)
    u = T(_driver_value(kw, (:u_LU, :u_mean)))
    nu_raw = _driver_value(kw, (:nu_total_LU, :nu_total, :nu))
    tau_raw = _driver_value(kw, (:tau_hydro, :tau))
    nu = nu_raw === nothing ? T((T(tau_raw) - T(0.5)) / T(3)) : T(nu_raw)
    tau = tau_raw === nothing ? T(_tau_from_nu(nu, one(T))) : T(tau_raw)
    Re = T(u * R / nu)
    scaling = _sym(_get(kw, :scaling, :acoustic))
    max_steps = _max_steps(kw, geom, R, u, T(NaN))
    units = _make_units(T; tau=tau, nu=nu, u=u, R_LU=R, scaling=scaling,
                        max_steps=max_steps, kw=kw)
    return NewtonianSpec{T}(Re), units
end

function lu_to_nondim(::Type{ViscoelasticSpec}, kw, geom::GeometryDescriptor)
    T = _get(kw, :T, Float64)
    R = _r_lu(kw)
    u = T(_driver_value(kw, (:u_LU, :u_mean)))
    nu_s_raw = _driver_value(kw, (:nu_s_LU, :nu_s))
    nu_p_raw = _driver_value(kw, (:nu_p_LU, :nu_p))
    nu_total_raw = _driver_value(kw, (:nu_total_LU, :nu_total, :nu))
    nu_s = nu_s_raw === nothing ? T(NaN) : T(nu_s_raw)
    nu_p = nu_p_raw === nothing ? T(NaN) : T(nu_p_raw)
    nu = nu_total_raw === nothing ? T(nu_s + nu_p) : T(nu_total_raw)
    beta = _get(kw, :beta, nothing)
    beta = beta === nothing ? T(nu_s / nu) : T(beta)
    bsd = T(_get(kw, :bsd_fraction, 1.0))
    fnu = _viscosity_factor(beta, bsd)
    tau_raw = _driver_value(kw, (:tau_hydro, :tau))
    tau = tau_raw === nothing ? T(_tau_from_nu(nu, fnu)) : T(tau_raw)
    lambda = T(_driver_value(kw, (:lambda_LU, :lambda)))
    Re = T(_get(kw, :Re, u * R / nu))
    Wi = T(_get(kw, :Wi, lambda * u / R))
    model = _normalize_model(_sym(_get(kw, :model, _get(kw, :polymer_model, :oldroyd_b))))
    L_max = T(_get(kw, :L_max, 0.0))
    scaling = _sym(_get(kw, :scaling, :acoustic))
    max_steps = _max_steps(kw, geom, R, u, lambda)
    units = _make_units(T; tau=tau, nu=nu, u=u, R_LU=R, scaling=scaling,
                        max_steps=max_steps, nu_s=nu_s, nu_p=nu_p,
                        lambda=lambda, kw=kw)
    return ViscoelasticSpec{T}(Re, Wi, beta, bsd, model, L_max), units
end

function intrinsic_unit_issues(units::LBMUnits, ::Type{T}) where {T}
    issues = Issue[]
    if units.tau_hydro < T(0.55)
        push!(issues, fatal_issue(:tau_below_trt_window,
            "tau_hydro=$(units.tau_hydro) is below the TRT lower bound 0.55"))
    elseif units.tau_hydro > T(1.5)
        push!(issues, fatal_issue(:tau_above_trt_window,
            "tau_hydro=$(units.tau_hydro) is above the TRT upper bound 1.5"))
    end
    if units.tau_hydro > T(1.2)
        push!(issues, warn_issue(:tau_above_magic,
            "tau_hydro=$(units.tau_hydro) is in the high-tau U-shape audit band"))
    end
    if units.Ma > T(0.05)
        push!(issues, fatal_issue(:mach_above_limit,
            "Ma=$(units.Ma) exceeds the 0.05 compressibility guardrail"))
    end
    if T === Float32 && units.tau_hydro < T(0.6)
        push!(issues, fatal_issue(:tau_float32_floor,
            "Float32 plans require tau_hydro >= 0.6"))
    end
    if isfinite(units.lambda_LU) && units.lambda_LU > T(1e5)
        push!(issues, warn_issue(:stiff_polymer_lambda,
            "lambda_LU=$(units.lambda_LU) exceeds the stiff-polymer audit threshold"))
    end
    if units.max_steps < 20_000
        push!(issues, warn_issue(:max_steps_low,
            "max_steps=$(units.max_steps) is below the 20,000 transient-resolution heuristic"))
    end
    return issues
end
