const STABILITY_REGISTRY = Dict{Tuple{Type,Type},Vector{Function}}()

const WALL_BC_REGISTRY = Dict{Symbol,Type{<:AbstractWallBC}}(
    :halfwayBB => HalfwayBB,
    :bouzidi_fl => BouzidiFL,
    :li_bb_v2 => LiBBV2,
    :mei_bouzidi => MeiBouzidi,
)

function _camel_to_snake_symbol(T::Type)
    name = replace(string(nameof(T)), r"BC$" => "")
    buf = IOBuffer()
    for (i, c) in enumerate(name)
        if i > 1 && isuppercase(c)
            print(buf, '_')
        end
        print(buf, lowercase(string(c)))
    end
    return Symbol(String(take!(buf)))
end

_wall_symbol(::Type{HalfwayBB}) = :halfwayBB
_wall_symbol(::Type{BouzidiFL}) = :bouzidi_fl
_wall_symbol(::Type{LiBBV2}) = :li_bb_v2
_wall_symbol(::Type{MeiBouzidi}) = :mei_bouzidi
_wall_symbol(T::Type{<:AbstractWallBC}) = _camel_to_snake_symbol(T)

function register_stability!(wall_type::Type{<:AbstractWallBC},
                             spec_type::Type{<:AbstractPhysicsSpec},
                             pred::Function)
    key = (wall_type, spec_type)
    list = get!(STABILITY_REGISTRY, key, Function[])
    pred in list || push!(list, pred)
    WALL_BC_REGISTRY[_wall_symbol(wall_type)] = wall_type
    return pred
end

function _wall_type(sym::Symbol)
    return get(WALL_BC_REGISTRY, sym, nothing)
end

function check_stability(units::LBMUnits, geom::GeometryDescriptor,
                         bc::BCConfig, spec::AbstractPhysicsSpec)
    wall_type = _wall_type(bc.wall_bc)
    wall_type === nothing &&
        return [error_issue(:wall_bc_unknown, "unknown wall_bc :$(bc.wall_bc)")]
    spec_type = Base.typename(typeof(spec)).wrapper
    funcs = get(STABILITY_REGISTRY, (wall_type, spec_type), Function[])
    issues = Issue[]
    for pred in funcs
        append!(issues, pred(units, geom))
    end
    return issues
end

function _halfway_pred(units::LBMUnits, ::GeometryDescriptor)
    issues = Issue[]
    units.tau_hydro < 0.55 &&
        push!(issues, fatal_issue(:halfway_tau_below_floor,
            "HalfwayBB requires tau_hydro >= 0.55"))
    units.tau_hydro > 1.5 &&
        push!(issues, fatal_issue(:halfway_tau_above_ceiling,
            "HalfwayBB requires tau_hydro <= 1.5"))
    units.Ma > 0.05 &&
        push!(issues, fatal_issue(:halfway_mach_above_limit,
            "HalfwayBB requires Ma <= 0.05"))
    return issues
end

function _bouzidi_ve_pred(units::LBMUnits, geom::GeometryDescriptor)
    issues = Issue[]
    units.tau_hydro < 0.6 &&
        push!(issues, fatal_issue(:bouzidi_tau_floor,
            "Bouzidi-FL requires tau_hydro >= 0.6"))
    units.tau_hydro > 1.5 &&
        push!(issues, fatal_issue(:bouzidi_tau_ceiling,
            "Bouzidi-FL requires tau_hydro <= 1.5"))
    if geom.q_wall_dist !== nothing &&
       any(q -> q < 0.05 || q > 0.95, geom.q_wall_dist)
        push!(issues, warn_issue(:q_wall_near_cliff,
            "Bouzidi-FL q_wall distribution has values near 0 or 1"))
    end
    return issues
end

register_stability!(HalfwayBB, NewtonianSpec, _halfway_pred)
register_stability!(HalfwayBB, ViscoelasticSpec, _halfway_pred)
register_stability!(BouzidiFL, ViscoelasticSpec, _bouzidi_ve_pred)
