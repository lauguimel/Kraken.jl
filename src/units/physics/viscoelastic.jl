function _normalize_model(model::Symbol)
    model === :oldroydb && return :oldroyd_b
    model === :oldroyd_b && return :oldroyd_b
    model === :fene_p && return :fene_p
    model === :log_conf && return :log_conf
    throw(ArgumentError("unsupported viscoelastic model :$model"))
end

function _build_spec(::Type{ViscoelasticSpec}, kw)
    T = _get(kw, :T, Float64)
    Re = T(_required(kw, :Re))
    Wi = T(_required(kw, :Wi))
    beta = T(_required(kw, :beta))
    bsd = T(_get(kw, :bsd_fraction, 1.0))
    model = _normalize_model(_sym(_get(kw, :model,
        _get(kw, :polymer_model, :oldroyd_b))))
    L_max = T(_get(kw, :L_max, model === :fene_p ? 10.0 : 0.0))
    Re > zero(T) || throw(ArgumentError("Re must be positive"))
    Wi >= zero(T) || throw(ArgumentError("Wi must be non-negative"))
    zero(T) < beta <= one(T) || throw(ArgumentError("beta must satisfy 0 < beta <= 1"))
    zero(T) <= bsd <= one(T) || throw(ArgumentError("bsd_fraction must satisfy 0 <= bsd_fraction <= 1"))
    return ViscoelasticSpec{T}(Re, Wi, beta, bsd, model, L_max)
end

function _compile_with_spec(spec::ViscoelasticSpec{T}, kw,
                            geom::GeometryDescriptor, bc::BCConfig,
                            disc::DiscretizationConfig, refinement,
                            strict::Bool, preissues::Vector{Issue}) where {T}
    units = nondim_to_lu(spec, kw, geom)
    issues = vcat(preissues, _shared_validation_issues(units, spec, geom,
                                                       bc, disc, T))
    return _assemble_plan(spec, units, bc, geom, disc, refinement, issues,
                          String[], :compile, strict)
end

_audit_with_spec_type(::Type{ViscoelasticSpec}, kw, geom::GeometryDescriptor) =
    lu_to_nondim(ViscoelasticSpec, kw, geom)

_driver_polymer_model(::Val{:oldroyd_b}) = :oldroydb
_driver_polymer_model(::Val{:fene_p}) = :fene_p
_driver_polymer_model(::Val{:log_conf}) = :log_conf
_driver_polymer_model(model::Symbol) = model === :oldroyd_b ? :oldroydb : model

function _driver_kwargs(plan::SimulationPlan, spec::ViscoelasticSpec)
    u = plan.units
    return (
        Re=spec.Re,
        Wi=spec.Wi,
        beta=spec.beta,
        bsd_fraction=spec.bsd_fraction,
        model=spec.model,
        L_max=spec.L_max,
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
        polymer_model=_driver_polymer_model(spec.model),
        advection_scheme=plan.discretization.advection_scheme,
        embedded_geometry=plan.discretization.embedded_geometry,
        embedded_gradient=plan.discretization.embedded_gradient,
        wall_bc=plan.bc.wall_bc,
    )
end
