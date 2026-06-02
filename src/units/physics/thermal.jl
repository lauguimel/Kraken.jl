function _build_spec(::Type{ThermalBoussinesqSpec}, kw)
    T = _get(kw, :T, Float64)
    Re = T(_required(kw, :Re))
    Pr = T(_required(kw, :Pr))
    Ra = T(_required(kw, :Ra))
    Re > zero(T) || throw(ArgumentError("Re must be positive"))
    Pr > zero(T) || throw(ArgumentError("Pr must be positive"))
    Ra > zero(T) || throw(ArgumentError("Ra must be positive"))
    return ThermalBoussinesqSpec{T}(Re, Pr, Ra)
end

_compile_with_spec(::ThermalBoussinesqSpec) =
    throw(phase_stub_error(:thermal_boussinesq))

function nondim_to_lu(spec::ThermalBoussinesqSpec{T}, kw,
                      geom::GeometryDescriptor) where {T}
    hydro = nondim_to_lu(NewtonianSpec{T}(spec.Re), kw, geom)
    alpha = T(hydro.nu_total_LU / spec.Pr)
    beta_thermal = T(spec.Ra * hydro.nu_total_LU * alpha /
                     (T(hydro.R_LU)^3))
    return _make_units(T; tau=hydro.tau_hydro, nu=hydro.nu_total_LU,
                       u=hydro.u_LU, R_LU=hydro.R_LU,
                       scaling=hydro.scaling, max_steps=hydro.max_steps,
                       alpha=alpha, beta_thermal=beta_thermal, kw=kw)
end

function _compile_with_spec(spec::ThermalBoussinesqSpec{T}, kw,
                            geom::GeometryDescriptor, bc::BCConfig,
                            disc::DiscretizationConfig, refinement,
                            strict::Bool, preissues::Vector{Issue}) where {T}
    units = nondim_to_lu(spec, kw, geom)
    issues = vcat(preissues, _shared_validation_issues(units, spec, geom,
                                                       bc, disc, T))
    return _assemble_plan(spec, units, bc, geom, disc, refinement, issues,
                          String[], :compile, strict)
end

function lu_to_nondim(::Type{ThermalBoussinesqSpec}, kw,
                      geom::GeometryDescriptor)
    T = _get(kw, :T, Float64)
    R = _r_lu(kw)
    u = T(_required_driver_value(kw, (:u_LU, :u_mean), "u_LU"))
    nu_raw = _driver_value(kw, (:nu_total_LU, :nu_total, :nu))
    tau_raw = _driver_value(kw, (:tau_hydro, :tau))
    nu_raw === nothing && tau_raw === nothing &&
        throw(ArgumentError("missing required thermal driver keyword `nu_total_LU` or `tau_hydro`"))
    nu = nu_raw === nothing ? T((T(tau_raw) - T(0.5)) / T(3)) : T(nu_raw)
    tau = tau_raw === nothing ? T(_tau_from_nu(nu, one(T))) : T(tau_raw)
    alpha = T(_required_driver_value(kw, (:alpha_LU, :alpha,
        :thermal_alpha_LU, :thermal_alpha), "alpha_LU"))
    beta_thermal = T(_required_driver_value(kw, (:beta_thermal_LU,
        :beta_thermal, :thermal_beta_LU, :thermal_buoyancy_LU),
        "beta_thermal_LU"))
    Re = T(_get(kw, :Re, u * R / nu))
    Pr = T(_get(kw, :Pr, nu / alpha))
    Ra = T(_get(kw, :Ra, beta_thermal * T(R)^3 / (nu * alpha)))
    scaling = _sym(_get(kw, :scaling, :acoustic))
    max_steps = _max_steps(kw, geom, R, u, T(NaN))
    units = _make_units(T; tau=tau, nu=nu, u=u, R_LU=R,
                        scaling=scaling, max_steps=max_steps, alpha=alpha,
                        beta_thermal=beta_thermal, kw=kw)
    return ThermalBoussinesqSpec{T}(Re, Pr, Ra), units
end

_audit_with_spec_type(::Type{ThermalBoussinesqSpec}, kw,
                      geom::GeometryDescriptor) =
    lu_to_nondim(ThermalBoussinesqSpec, kw, geom)

function _required_driver_value(kw, keys::Tuple, label::AbstractString)
    value = _driver_value(kw, keys)
    value === nothing &&
        throw(ArgumentError("missing required thermal driver keyword `$label`"))
    return value
end

function _thermal_halfway_pred(units::LBMUnits, ::GeometryDescriptor)
    T = typeof(units.alpha_LU)
    issues = Issue[]
    if !(isfinite(units.alpha_LU) && units.alpha_LU > zero(T))
        push!(issues, fatal_issue(:thermal_alpha_nonpositive,
            "thermal diffusivity alpha_LU must be positive"))
    else
        tau_thermal = T(0.5) + T(3) * units.alpha_LU
        if tau_thermal < T(0.55)
            push!(issues, fatal_issue(:thermal_tau_below_floor,
                "HalfwayBB thermal scalar relaxation requires tau_thermal >= 0.55"))
        elseif tau_thermal > T(1.5)
            push!(issues, fatal_issue(:thermal_tau_above_ceiling,
                "HalfwayBB thermal scalar relaxation requires tau_thermal <= 1.5"))
        end
    end
    if !(isfinite(units.beta_thermal_LU) && units.beta_thermal_LU > zero(T))
        push!(issues, fatal_issue(:thermal_beta_nonpositive,
            "thermal buoyancy coefficient beta_thermal_LU must be positive"))
    end
    return issues
end

register_stability!(HalfwayBB, ThermalBoussinesqSpec, _thermal_halfway_pred)
register_bc_combo!((:velocity_parabolic, :zou_he_pressure,
                    :temperature_dirichlet, :temperature_dirichlet), :ok)
