# --- Generic simulation runner for .krk config files ---


"""
    run_simulation(filename::String; backend=CPU(), T=Float64,
                   max_steps=nothing, kwargs...) -> NamedTuple

Run an LBM simulation defined by a `.krk` configuration file.
Keyword arguments override `Define` defaults for parametric studies.
Pass `max_steps` to override the `Run N steps` directive (useful for tests).

Returns a NamedTuple with final fields on CPU: `(ρ, ux, uy, setup)`.
If the setup contains `Sensitivity { ... }`, returns the
`steady_shape_sensitivity` API result instead.

## Dispatch rules
The runner selects a backend driver based on `setup.modules`, `setup.lattice`,
`setup.refinements`, and the `setup.name` (case name):

1. `setup.sensitivity !== nothing` → steady AD shape sensitivity
2. `:advection_only in modules`  → pure VOF advection (no LBM solve)
3. `:twophase_vof   in modules`  → two-phase LBM with surface tension
4. `:axisymmetric   in modules`  → `run_hagen_poiseuille_2d` if the case
   name contains `hagen_poiseuille`, otherwise an informative error
5. `!isempty(setup.refinements)` → refined-grid drivers. Only refined
   natural convection is currently supported via the .krk runner;
   other refined cases raise an informative error (run them via the
   Julia API — see `create_refined_domain` / `create_thermal_patch_arrays`).
6. `:thermal in modules`         → `run_rayleigh_benard_2d`,
   `run_natural_convection_2d`, or a thermal-conduction fallback
   depending on the case name
7. `setup.lattice === :D3Q19`    → `run_cavity_3d` if the case name
   contains `cavity_3d`, otherwise an informative error
8. Default (`:D2Q9`, no modules) → generic single-phase LBM loop
   (existing behavior, compatible with all cavity/Poiseuille/Couette/
   Taylor-Green/cylinder examples).

# Example
```julia
result = run_simulation("examples/cavity.krk")
result = run_simulation("examples/cavity.krk"; Re=400, N=256)
result = run_simulation("examples/rayleigh_benard.krk"; max_steps=100)
```
"""
function run_simulation(filename::String;
                        backend=KernelAbstractions.CPU(), T=Float64,
                        callback::Union{Nothing,Function}=nothing,
                        callback_every::Int=100,
                        max_steps::Union{Nothing,Int}=nothing, kwargs...)
    setup = load_kraken(filename; kwargs...)
    if max_steps !== nothing
        setup = _override_max_steps(setup, max_steps)
    end
    return run_simulation(setup; backend=backend, T=T,
                          callback=callback, callback_every=callback_every)
end

"""Find the first output spec with format `fmt`, or nothing."""
function _find_output(setup::SimulationSetup, fmt::Symbol)
    idx = findfirst(o -> o.format == fmt, setup.outputs)
    return idx === nothing ? nothing : setup.outputs[idx]
end

"""Return a copy of `setup` with `max_steps` overridden."""
function _override_max_steps(setup::SimulationSetup, new_max::Int)
    return SimulationSetup(
        setup.name, setup.lattice, setup.domain, setup.physics,
        setup.user_vars, setup.regions, setup.boundaries, setup.initial,
        setup.modules, new_max,
        setup.outputs, setup.diagnostics, setup.refinements,
        setup.velocity_field, setup.rheology, setup.mesh, setup.units,
        setup.collision, setup.wall_bc, setup.sensitivity)
end

_krk_selector_symbol(sym::Symbol) = Symbol(lowercase(String(sym)))

function _normalize_collision_selector(sym::Symbol)
    mode = _krk_selector_symbol(sym)
    mode in (:bgk, :trt) || throw(ArgumentError(
        "Unsupported Physics collision='$sym'. Supported generic 2D values: bgk, trt."))
    return mode
end

function _normalize_wall_bc_selector(sym::Symbol)
    mode = _krk_selector_symbol(sym)
    mode in (:halfwaybb, :halfway_bb, :halfway) && return :halfwaybb
    mode === :libb && return :libb
    throw(ArgumentError(
        "Unsupported Physics wall_bc='$sym'. Supported generic 2D values: halfwaybb, libb."))
end

function _setup_with_global_stl_libb(setup::SimulationSetup)
    regions = GeometryRegion[]
    sizehint!(regions, length(setup.regions))
    found_stl_obstacle = false
    for region in setup.regions
        if region.kind === :obstacle && region.stl !== nothing
            found_stl_obstacle = true
            push!(regions, GeometryRegion(region.name, region.kind, region.condition,
                                          region.stl, :libb, region.bc_values))
        else
            push!(regions, region)
        end
    end
    found_stl_obstacle || throw(ArgumentError(
        "Physics wall_bc=libb in the generic 2D path currently requires an STL Obstacle. " *
        "Analytic-obstacle LI-BB dispatch is deferred."))
    # TODO(K1-followup): add LI-BB q-wall dispatch for analytic obstacles.
    return SimulationSetup(setup.name, setup.lattice, setup.domain, setup.physics,
                           setup.user_vars, regions, setup.boundaries,
                           setup.initial, setup.modules, setup.max_steps,
                           setup.outputs, setup.diagnostics, setup.refinements,
                           setup.velocity_field, setup.rheology, setup.mesh,
                           setup.units, setup.collision, :libb,
                           setup.sensitivity)
end

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

function _ensure_generic_trt_supported!(setup::SimulationSetup, has_body_force::Bool,
                                        has_rheology::Bool)
    if has_body_force || has_rheology
        throw(ArgumentError(
            "Physics collision=trt in the generic 2D path currently supports only " *
            "Newtonian flow without body force."))
    end
    bad = findfirst(bc -> bc.type !== :wall, setup.boundaries)
    bad === nothing && return nothing
    bc = setup.boundaries[bad]
    throw(ArgumentError(
        "Physics collision=trt in the generic 2D path currently supports only " *
        "static wall boundaries; got $(bc.face) $(bc.type)."))
end

"""
    run_simulation(setup::SimulationSetup; backend=CPU(), T=Float64,
                   callback=nothing, callback_every=100)

Run an LBM simulation from a parsed `SimulationSetup`.

If `callback` is provided, it is called every `callback_every` steps as
`callback(step, state)` where `state` is a NamedTuple `(; rho, ux, uy)` of
CPU arrays. Useful for live monitoring or custom post-processing.
"""
function run_simulation(setup::SimulationSetup;
                        backend=KernelAbstractions.CPU(), T=Float64,
                        callback::Union{Nothing,Function}=nothing,
                        callback_every::Int=100)
    # --- Sanity checks (tau, Mach, CFL) ---
    sanity_check(setup)

    # --- Dispatch to specialized runners based on modules ---
    if setup.sensitivity !== nothing
        T === Float64 || throw(ArgumentError(
            ".krk Sensitivity dispatch supports T=Float64 only."))
        return run_krk_sensitivity(setup)
    elseif setup.mesh !== nothing
        if :slbm_drag in setup.modules
            return _run_gmsh_slbm_drag(setup; backend=backend, T=T,
                                       callback=callback,
                                       callback_every=callback_every)
        end
        error("Mesh directive is present, but no mesh-capable runner was selected. " *
              "For Gmsh cylinder drag use `Module slbm_drag`.")
    elseif :advection_only in setup.modules
        return _run_advection_only(setup; backend=backend, T=T)
    elseif :twophase_vof in setup.modules
        return _run_twophase_vof(setup; backend=backend, T=T)
    elseif :axisymmetric in setup.modules
        return _run_axisymmetric(setup; backend=backend, T=T)
    elseif !isempty(setup.refinements)
        return _run_refined(setup; backend=backend, T=T,
                            callback=callback, callback_every=callback_every)
    elseif :thermal in setup.modules
        return _run_thermal(setup; backend=backend, T=T)
    elseif :viscoelastic in setup.modules
        return _run_viscoelastic(setup; backend=backend, T=T)
    elseif setup.lattice === :D3Q19
        return _run_d3q19(setup; backend=backend, T=T)
    end

    # --- Default: single-phase LBM ---
    dom = setup.domain
    Nx, Ny = dom.Nx, dom.Ny
    dx = dom.Lx / Nx
    dy = dom.Ly / Ny
    ν = setup.physics.params[:nu]
    ω = T(1.0 / (3.0 * ν + 0.5))
    collision = _normalize_collision_selector(setup.collision)
    wall_bc = _normalize_wall_bc_selector(setup.wall_bc)

    # --- Initialize state ---
    config = LBMConfig(D2Q9(); Nx=Nx, Ny=Ny, ν=ν, u_lid=0.0,
                       max_steps=setup.max_steps, output_interval=1000)
    state = initialize_2d(config, T; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid

    # --- Apply geometry ---
    _apply_geometry!(is_solid, setup, dx, dy)
    libb_setup = wall_bc === :libb ? _setup_with_global_stl_libb(setup) : setup
    stl_libb = _has_stl_libb_obstacle(libb_setup)

    # --- Apply initial conditions ---
    if setup.initial !== nothing
        _apply_initial_conditions!(f_in, f_out, setup, dx, dy, T)
    end

    # --- Select streaming kernel ---
    stream_fn! = _select_streaming_kernel(setup)

    # --- Select collision kernel ---
    has_body_force = !isempty(setup.physics.body_force)
    Fx_val = T(0)
    Fy_val = T(0)
    if has_body_force
        Fx_val = haskey(setup.physics.body_force, :Fx) ?
            T(evaluate(setup.physics.body_force[:Fx])) : T(0)
        Fy_val = haskey(setup.physics.body_force, :Fy) ?
            T(evaluate(setup.physics.body_force[:Fy])) : T(0)
    end

    # --- Non-Newtonian rheology ---
    has_rheology = !isempty(setup.rheology)
    rheology_model = nothing
    tau_field = nothing
    Fx_field = nothing
    Fy_field = nothing
    if has_rheology
        rs = first(r for r in setup.rheology if r.phase in (:default, :liquid))
        rheology_model = build_rheology_model(rs; FT=T)
        tau_field = KernelAbstractions.ones(backend, T, Nx, Ny)  # initial tau = 1
        if has_body_force
            Fx_field = KernelAbstractions.zeros(backend, T, Nx, Ny)
            Fy_field = KernelAbstractions.zeros(backend, T, Nx, Ny)
            Fx_field .= Fx_val
            Fy_field .= Fy_val
        end
    end
    if stl_libb && (has_body_force || has_rheology)
        error("STL wall=libb in the generic .krk runner supports only Newtonian flow without body force")
    end
    use_trt_step = !stl_libb && collision === :trt
    use_trt_step && _ensure_generic_trt_supported!(setup, has_body_force, has_rheology)

    # --- Build boundary handlers ---
    bc_handlers = _build_boundary_handlers(setup, dx, dy, Nx, Ny, T, backend)
    libb_q_wall = nothing
    libb_uw_x = nothing
    libb_uw_y = nothing
    libb_bcspec = nothing
    if stl_libb
        q_wall_cpu = _precompute_stl_libb_q_wall_2d(Array(is_solid), libb_setup, dx, dy, T)
        libb_q_wall = _copy_to_backend(backend, T, q_wall_cpu)
        libb_uw_x = KernelAbstractions.zeros(backend, T, Nx, Ny, 9)
        libb_uw_y = KernelAbstractions.zeros(backend, T, Nx, Ny, 9)
        libb_bcspec = _build_libb_bc_rebuild_spec_2d(libb_setup, dx, dy, Nx, Ny, T, backend)
    end

    # --- Setup output ---
    vtk_out = _find_output(setup, :vtk)
    pvd = nothing
    output_dir = ""
    if vtk_out !== nothing
        output_dir = setup_output_dir(vtk_out.directory)
        pvd = create_pvd(joinpath(output_dir, setup.name))
    end

    # PNG/GIF output setup
    png_out = _find_output(setup, :png)
    gif_out = _find_output(setup, :gif)
    gif_frames = _init_gif_frames(gif_out)
    _check_image_backend(png_out, gif_out)

    # --- Time loop ---
    for step in 1:setup.max_steps
        if stl_libb
            # Fused TRT LI-BB performs pull-streaming, wall treatment, collision,
            # and moment writes. Face BCs are rebuilt from f_in afterwards.
            fused_trt_libb_v2_step!(f_out, f_in, ρ, ux, uy, is_solid,
                                     libb_q_wall, libb_uw_x, libb_uw_y,
                                     Nx, Ny, T(ν))
            apply_bc_rebuild_2d!(f_out, f_in, libb_bcspec, ν, Nx, Ny)
        elseif use_trt_step
            fused_trt_step!(f_out, f_in, ρ, ux, uy, is_solid, Nx, Ny, T(ν))
        else
            # 1. Stream
            stream_fn!(f_out, f_in, Nx, Ny)

            # 2. Apply boundary conditions
            _apply_boundary_conditions!(f_out, bc_handlers, step, Nx, Ny, dx, dy, dom, T)

            # 3. Collide
            if has_rheology && has_body_force
                collide_rheology_guo_2d!(f_out, is_solid, rheology_model, tau_field,
                                          Fx_field, Fy_field)
            elseif has_rheology
                collide_rheology_2d!(f_out, is_solid, rheology_model, tau_field)
            elseif has_body_force
                collide_guo_2d!(f_out, is_solid, ω, Fx_val, Fy_val)
            else
                collide_2d!(f_out, is_solid, ω)
            end

            # 4. Macroscopic quantities
            if has_body_force
                compute_macroscopic_forced_2d!(ρ, ux, uy, f_out, Fx_val, Fy_val)
            else
                compute_macroscopic_2d!(ρ, ux, uy, f_out)
            end
        end

        # 5. Swap
        f_in, f_out = f_out, f_in

        # 6. VTK Output
        if vtk_out !== nothing && step % vtk_out.interval == 0
            _write_output(ρ, ux, uy, setup, vtk_out, pvd, output_dir, dx, step)
        end

        # 6b. PNG/GIF snapshots
        _maybe_save_png(png_out, ρ, ux, uy, setup, output_dir, step)
        _maybe_collect_gif(gif_out, gif_frames, ρ, ux, uy, step)

        # 7. Callback (live visualization / probes)
        if callback !== nothing && step % callback_every == 0
            callback(step, (; rho=Array(ρ), ux=Array(ux), uy=Array(uy)))
        end
    end

    # Finalize PVD
    if pvd !== nothing
        vtk_save(pvd)
    end

    # Finalize GIF
    _maybe_save_gif(gif_out, gif_frames, setup, output_dir)

    # Return on CPU
    result = (ρ=Array(ρ), ux=Array(ux), uy=Array(uy), setup=setup)
    if has_rheology
        result = merge(result, (tau_field=Array(tau_field),))
    end
    return result
end

function _ve_numeric_param(setup::SimulationSetup, key::Symbol, default=nothing)
    if haskey(setup.physics.params, key)
        return Float64(setup.physics.params[key])
    elseif haskey(setup.user_vars, key)
        return Float64(setup.user_vars[key])
    elseif default !== nothing
        return Float64(default)
    end
    throw(ArgumentError(
        "viscoelastic dispatch: missing parameter `$key`. " *
        "Provide it as `Define $key = ...` or `Physics $key = ...`."))
end

function _ve_symbol_param(setup::SimulationSetup, key::Symbol, default::Symbol)
    if haskey(setup.physics.params, key) || haskey(setup.user_vars, key)
        throw(ArgumentError(
            "viscoelastic dispatch: `$key` is symbolic, but `.krk` Define/Physics " *
            "values are numeric in the current parser. Omit `$key` to use `:$default` " *
            "or call the Julia driver directly."))
    end
    return default
end

function _ve_oldroydb_rheology(setup::SimulationSetup)
    idx = findfirst(rs -> rs.model === :oldroyd_b, setup.rheology)
    idx !== nothing && return setup.rheology[idx]
    throw(ArgumentError(
        "viscoelastic dispatch: cylinder requires " *
        "`Rheology oldroyd_b { nu_s = ..., nu_p = ..., lambda = ... }`."))
end

function _ve_rheology_param(rs::RheologySetup, key::Symbol)
    haskey(rs.params, key) && return Float64(rs.params[key])
    throw(ArgumentError(
        "viscoelastic dispatch: Rheology $(rs.model) is missing `$key`."))
end

function _ve_cylinder_obstacle_radius(setup::SimulationSetup)
    for region in setup.regions
        region_name = lowercase(region.name)
        (occursin("cylinder", region_name) || occursin("circle", region_name)) || continue
        for key in (:radius, :R)
            if haskey(region.bc_values, key)
                return Float64(evaluate(region.bc_values[key]))
            end
        end
    end
    return nothing
end

"""Dispatch viscoelastic cases to the existing production log-FV driver."""
function _run_viscoelastic(setup::SimulationSetup;
                           backend=KernelAbstractions.CPU(), T=Float64)
    name = lowercase(setup.name)
    if !occursin("cylinder", name)
        throw(ArgumentError(
            "viscoelastic dispatch: unrecognized case name '$(setup.name)'. " *
            "Known cases: cylinder."))
    end

    rs = _ve_oldroydb_rheology(setup)
    radius_from_obstacle = _ve_cylinder_obstacle_radius(setup)
    R = radius_from_obstacle === nothing ?
        _ve_numeric_param(setup, :R) : Float64(radius_from_obstacle)
    nu_s = _ve_rheology_param(rs, :nu_s)
    nu_p = _ve_rheology_param(rs, :nu_p)
    lambda = _ve_rheology_param(rs, :lambda)
    Re = _ve_numeric_param(setup, :Re, 1.0)
    nu_total = nu_s + nu_p
    u_mean = _ve_numeric_param(setup, :u_mean, nu_total * Re / R)
    L_up = _ve_numeric_param(setup, :L_up, 15.0)
    L_down = _ve_numeric_param(setup, :L_down, 15.0)
    bsd_fraction = _ve_numeric_param(setup, :bsd_fraction, 1.0)
    wall_bc = _ve_symbol_param(setup, :wall_bc, :halfwayBB)
    advection_scheme = _ve_symbol_param(setup, :advection_scheme, :muscl_superbee)
    avg_window = round(Int, 0.2 * setup.max_steps)

    result = run_viscoelastic_logfv_cylinder_coupled_2d(;
        radius=R,
        L_up=L_up,
        L_down=L_down,
        nu_s=nu_s,
        nu_p=nu_p,
        lambda=lambda,
        u_mean=u_mean,
        bsd_fraction=bsd_fraction,
        polymer_model=:oldroydb,
        wall_bc=wall_bc,
        advection_scheme=advection_scheme,
        max_steps=setup.max_steps,
        avg_window=avg_window,
        backend=backend,
        T=T,
    )
    return merge(result, (setup=setup,))
end

# --- Gmsh multi-block SLBM drag runner ---

function _setup_number(setup::SimulationSetup, keys, default)
    key_tuple = keys isa Tuple ? keys : (keys,)
    for key in key_tuple
        haskey(setup.physics.params, key) && return setup.physics.params[key]
        haskey(setup.user_vars, key) && return setup.user_vars[key]
    end
    return default
end

function _copy_to_backend(backend, ::Type{T}, host::AbstractArray) where T
    dev = KernelAbstractions.allocate(backend, T, size(host)...)
    copyto!(dev, T.(host))
    return dev
end

function _copy_bool_to_backend(backend, host::AbstractArray{Bool})
    dev = KernelAbstractions.allocate(backend, Bool, size(host)...)
    copyto!(dev, host)
    return dev
end

function _allocate_block_state_as(block::Block, ::Type{T}, backend, ng::Int) where T
    nx = block.mesh.Nξ + 2 * ng
    ny = block.mesh.Nη + 2 * ng
    f = KernelAbstractions.allocate(backend, T, nx, ny, 9); fill!(f, zero(T))
    rho = KernelAbstractions.allocate(backend, T, nx, ny); fill!(rho, one(T))
    ux = KernelAbstractions.allocate(backend, T, nx, ny); fill!(ux, zero(T))
    uy = KernelAbstractions.allocate(backend, T, nx, ny); fill!(uy, zero(T))
    return BlockState2D{T, typeof(f), typeof(rho)}(f, rho, ux, uy,
                                                    block.mesh.Nξ, block.mesh.Nη, ng)
end

@inline function _edge_node(block::Block, edge::Symbol, r::Int)
    edge === :west  && return 1, r
    edge === :east  && return block.mesh.Nξ, r
    edge === :south && return r, 1
    edge === :north && return r, block.mesh.Nη
    error("unknown edge $edge")
end

function _parabolic_channel_u(y, Ly, u_max)
    yy = clamp(Float64(y), 0.0, Float64(Ly))
    return 4.0 * Float64(u_max) * yy * (Float64(Ly) - yy) / Float64(Ly)^2
end


"""Apply initial conditions from expressions."""
function _apply_initial_conditions!(f_in, f_out, setup::SimulationSetup,
                                    dx, dy, ::Type{T}) where T
    Nx, Ny = setup.domain.Nx, setup.domain.Ny
    Lx, Ly = setup.domain.Lx, setup.domain.Ly

    ic = setup.initial
    w = weights(D2Q9())
    f_cpu = zeros(T, Nx, Ny, 9)

    for j in 1:Ny, i in 1:Nx
        x = (i - 0.5) * dx
        y = (j - 0.5) * dy
        kw = (; x=x, y=y, Lx=Lx, Ly=Ly, Nx=Float64(Nx), Ny=Float64(Ny), dx=dx, dy=dy)

        ρ_val = haskey(ic.fields, :rho) ? T(evaluate(ic.fields[:rho]; kw...)) : one(T)
        ux_val = haskey(ic.fields, :ux) ? T(evaluate(ic.fields[:ux]; kw...)) : zero(T)
        uy_val = haskey(ic.fields, :uy) ? T(evaluate(ic.fields[:uy]; kw...)) : zero(T)

        for q in 1:9
            f_cpu[i, j, q] = equilibrium(D2Q9(), ρ_val, ux_val, uy_val, q)
        end
    end

    copyto!(f_in, f_cpu)
    copyto!(f_out, f_cpu)
end
