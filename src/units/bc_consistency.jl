const BC_COMPATIBILITY = Dict{NTuple{4,Symbol},Symbol}(
    (:velocity_parabolic, :zou_he_pressure, :halfwayBB, :halfwayBB) => :ok,
    (:velocity_uniform, :zou_he_pressure, :halfwayBB, :halfwayBB) => :ok,
    (:periodic_x, :periodic_x, :halfwayBB, :halfwayBB) => :ok,
    (:velocity_parabolic, :periodic_x, :halfwayBB, :halfwayBB) => :error,
    (:velocity_parabolic, :zou_he_pressure, :wall, :wall) => :ok,
    (:velocity_uniform, :pressure, :wall, :wall) => :ok,
    (:velocity_uniform, :zou_he_pressure, :wall, :wall) => :ok,
    (:periodic_x, :periodic_x, :wall, :wall) => :ok,
)

function register_bc_combo!(key::NTuple{4,Symbol}, status::Symbol)
    status in (:ok, :warn, :error) ||
        throw(ArgumentError("BC combo status must be :ok, :warn, or :error"))
    BC_COMPATIBILITY[key] = status
    return status
end

function check_bc_consistency(bc::BCConfig)
    key = (bc.inlet, bc.outlet, bc.north_wall, bc.south_wall)
    status = get(BC_COMPATIBILITY, key, nothing)
    status === nothing &&
        return [error_issue(:bc_combo_unknown,
            "unknown BC combo $(key); register it with register_bc_combo!")]
    status === :ok && return Issue[]
    status === :warn && return [warn_issue(:bc_combo_warn,
        "borderline BC combo $(key)")]
    return [error_issue(:bc_combo_incompatible,
        "incompatible BC combo $(key)")]
end
