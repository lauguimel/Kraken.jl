# --- .krk sensitivity dispatch helpers ---

function _krk_sensitivity_real(value, label::Symbol)
    value isa Real && return Float64(value)
    throw(ArgumentError(
        "Sensitivity dispatch: `$label` must be numeric, got '$value'."))
end

function _krk_sensitivity_optional_numeric(setup::SimulationSetup,
                                           keys::Tuple{Vararg{Symbol}})
    for key in keys
        haskey(setup.physics.params, key) &&
            return _krk_sensitivity_real(setup.physics.params[key], key)
        haskey(setup.user_vars, key) &&
            return _krk_sensitivity_real(setup.user_vars[key], key)
    end
    return nothing
end

function _krk_sensitivity_required_numeric(setup::SimulationSetup,
                                           keys::Tuple{Vararg{Symbol}},
                                           label::AbstractString)
    value = _krk_sensitivity_optional_numeric(setup, keys)
    value !== nothing && return value
    throw(ArgumentError(
        "Sensitivity dispatch: missing $label. Provide it in Physics or Define."))
end

function _krk_sensitivity_expr_kwargs(setup::SimulationSetup; x=0.0,
                                      y=setup.domain.Ly / 2, z=0.0, t=0.0)
    dom = setup.domain
    dx = dom.Lx / dom.Nx
    dy = dom.Ly / dom.Ny
    dz = dom.Lz / dom.Nz
    return (; x=x, y=y, z=z, t=t, Lx=dom.Lx, Ly=dom.Ly, Lz=dom.Lz,
            Nx=dom.Nx, Ny=dom.Ny, Nz=dom.Nz, dx=dx, dy=dy, dz=dz)
end

function _krk_sensitivity_cylinder_radius(setup::SimulationSetup)
    for region in setup.regions
        region.kind === :obstacle || continue
        lname = lowercase(region.name)
        (occursin("cyl", lname) || occursin("circle", lname)) || continue
        for key in (:radius, :R)
            haskey(region.bc_values, key) &&
                return Float64(evaluate(region.bc_values[key]))
        end
    end
    value = _krk_sensitivity_optional_numeric(setup, (:radius, :R))
    value !== nothing && return value
    throw(ArgumentError(
        "Sensitivity dispatch: missing cylinder radius. Prefer " *
        "`Obstacle cylinder wall(radius = R) { ... }` or `Define R = ...`."))
end

function _krk_sensitivity_u_in(setup::SimulationSetup)
    value = _krk_sensitivity_optional_numeric(setup, (:u_in, :U, :U_in))
    value !== nothing && return value

    for bc in setup.boundaries
        bc.face === :west || continue
        bc.type === :velocity || continue
        haskey(bc.values, :ux) || continue
        kwargs = _krk_sensitivity_expr_kwargs(setup; x=0.0,
                                             y=setup.domain.Ly / 2)
        return abs(Float64(evaluate(bc.values[:ux]; kwargs...)))
    end
    throw(ArgumentError(
        "Sensitivity dispatch: missing inlet velocity. Provide `Define U = ...` " *
        "or a west velocity boundary with `ux = ...`."))
end

function _krk_sensitivity_rho_out(setup::SimulationSetup)
    value = _krk_sensitivity_optional_numeric(setup, (:rho_out, Symbol("ρ_out")))
    value !== nothing && return value

    for bc in setup.boundaries
        bc.face === :east || continue
        bc.type === :pressure || continue
        haskey(bc.values, :rho) || continue
        kwargs = _krk_sensitivity_expr_kwargs(setup; x=setup.domain.Lx,
                                             y=setup.domain.Ly / 2)
        return Float64(evaluate(bc.values[:rho]; kwargs...))
    end
    return 1.0
end

function _krk_sensitivity_inlet(setup::SimulationSetup)
    for key in (:inlet,)
        if haskey(setup.physics.params, key) || haskey(setup.user_vars, key)
            value = haskey(setup.physics.params, key) ?
                setup.physics.params[key] : setup.user_vars[key]
            value isa Symbol || throw(ArgumentError(
                "Sensitivity dispatch: `$key` must be a bare identifier."))
            return Symbol(lowercase(String(value)))
        end
    end
    for bc in setup.boundaries
        bc.face === :west || continue
        bc.type === :velocity || continue
        haskey(bc.values, :ux) || continue
        return is_spatial(bc.values[:ux]) ? :parabolic : :uniform
    end
    return :parabolic
end

"""
    run_krk_sensitivity(setup::SimulationSetup)

Dispatch a `.krk` `Sensitivity` request to `steady_shape_sensitivity`.
Returns the AD API NamedTuple: `(; value, gradient, qoi_value, solver,
terms, n_iter, ...)`.
"""
function run_krk_sensitivity(setup::SimulationSetup)
    request = setup.sensitivity
    request === nothing && throw(ArgumentError(
        "run_krk_sensitivity requires setup.sensitivity !== nothing"))

    dom = setup.domain
    nu = _krk_sensitivity_required_numeric(setup, (:nu, Symbol("ν")),
                                           "viscosity `nu`")
    kwargs = Dict{Symbol, Any}(
        :Nx => dom.Nx,
        :Ny => dom.Ny,
        :radius => _krk_sensitivity_cylinder_radius(setup),
        :u_in => _krk_sensitivity_u_in(setup),
        Symbol("ν") => nu,
        Symbol("ρ_out") => _krk_sensitivity_rho_out(setup),
        :qoi => request.qoi,
        :wrt => request.wrt,
        :max_steps => setup.max_steps,
        :inlet => _krk_sensitivity_inlet(setup),
    )

    cx = _krk_sensitivity_optional_numeric(setup, (:cx,))
    cy = _krk_sensitivity_optional_numeric(setup, (:cy,))
    cx !== nothing && (kwargs[:cx] = cx)
    cy !== nothing && (kwargs[:cy] = cy)

    for key in (:tol, :gmres_tol, :adjoint_tol)
        value = _krk_sensitivity_optional_numeric(setup, (key,))
        value !== nothing && (kwargs[key] = value)
    end

    return steady_shape_sensitivity(; kwargs...)
end
