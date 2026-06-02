const PHYSICS_REGISTRY = Dict{Symbol,Type}(
    :newtonian => NewtonianSpec,
    :viscoelastic => ViscoelasticSpec,
    :thermal_boussinesq => ThermalBoussinesqSpec,
    :power_law => PowerLawSpec,
    :multiphase => MultiphaseSpec,
    :mhd => MHDSpec,
)

const _GENERIC_KW = Set{Symbol}((
    :R_LU, :R, :radius, :scaling, :sweep_R, :tau_target, :u_target,
    :n_FT, :n_flow_through, :max_steps, :L_up, :L_down,
    :advection_scheme, :embedded_geometry, :embedded_gradient,
    :dx_real, :dt_real, :rho_real,
))

const _PHYSICS_KW = Dict{Symbol,Set{Symbol}}(
    :newtonian => Set{Symbol}((:Re,)),
    :viscoelastic => Set{Symbol}((
        :Re, :Wi, :beta, :bsd_fraction, :model, :L_max,
        :nu_s, :nu_p, :nu_s_LU, :nu_p_LU, :lambda, :lambda_LU,
        :u_mean, :u_LU, :nu_total, :nu_total_LU, :tau, :tau_hydro,
        :polymer_model, :polymer_substeps, :subcycle_relative_tolerance,
        :max_deformation_increment, :max_memory_deformation_increment,
        :max_polymer_substeps,
    )),
    :thermal_boussinesq => Set{Symbol}((:Re, :Pr, :Ra)),
    :power_law => Set{Symbol}((:Re, :n, :K)),
    :multiphase => Set{Symbol}((:Re,)),
    :mhd => Set{Symbol}((:Re, :Ha, :Rm)),
)

function register_physics!(sym::Symbol, spec_type::Type{<:AbstractPhysicsSpec})
    PHYSICS_REGISTRY[sym] = spec_type
    return spec_type
end

function unknown_kw_issues(physics::Symbol, keys_iter; audit_mode::Bool=false)
    allowed = union(_GENERIC_KW, get(_PHYSICS_KW, physics, Set{Symbol}()))
    unknown = sort([Symbol(k) for k in keys_iter if !(Symbol(k) in allowed)])
    isempty(unknown) && return Issue[]
    severity = audit_mode ? :warn : :error
    msg = "unknown units keyword(s) for :$physics: $(join(string.(unknown), ", "))"
    return [Issue(severity, :unknown_keyword, msg)]
end

function _build_spec end
function _compile_with_spec end
function _audit_with_spec_type end
