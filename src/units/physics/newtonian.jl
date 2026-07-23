function _build_spec(::Type{NewtonianSpec}, kw)
    T = _get(kw, :T, Float64)
    Re = T(_required(kw, :Re))
    Re > zero(T) || throw(ArgumentError("Re must be positive"))
    return NewtonianSpec{T}(Re)
end

function _compile_with_spec(spec::NewtonianSpec{T}, kw,
                            geom::GeometryDescriptor, bc::BCConfig,
                            disc::DiscretizationConfig, refinement,
                            strict::Bool, preissues::Vector{Issue}) where {T}
    units = nondim_to_lu(spec, kw, geom)
    issues = vcat(preissues, _shared_validation_issues(units, spec, geom,
                                                       bc, disc, T))
    return _assemble_plan(spec, units, bc, geom, disc, refinement, issues,
                          String[], :compile, strict)
end

_audit_with_spec_type(::Type{NewtonianSpec}, kw, geom::GeometryDescriptor) =
    lu_to_nondim(NewtonianSpec, kw, geom)

function _driver_kwargs(plan::SimulationPlan, spec::NewtonianSpec)
    u = plan.units
    return (
        Re=spec.Re,
        Wi=typeof(u.tau_hydro)(NaN),
        beta=typeof(u.tau_hydro)(NaN),
        bsd_fraction=typeof(u.tau_hydro)(NaN),
        model=:newtonian,
        L_max=typeof(u.tau_hydro)(NaN),
        R_LU=u.R_LU,
        radius=u.R_LU,
        scaling=u.scaling,
        tau_hydro=u.tau_hydro,
        tau=u.tau_hydro,
        nu_total_LU=u.nu_total_LU,
        nu_total=u.nu_total_LU,
        nu_s_LU=u.nu_s_LU,
        nu_p_LU=u.nu_p_LU,
        nu_s=u.nu_s_LU,
        nu_p=u.nu_p_LU,
        lambda_LU=u.lambda_LU,
        lambda=u.lambda_LU,
        u_LU=u.u_LU,
        u_mean=u.u_LU,
        Ma=u.Ma,
        max_steps=u.max_steps,
        polymer_model=:none,
        advection_scheme=plan.discretization.advection_scheme,
        embedded_geometry=plan.discretization.embedded_geometry,
        embedded_gradient=plan.discretization.embedded_gradient,
        wall_bc=plan.bc.wall_bc,
    )
end
