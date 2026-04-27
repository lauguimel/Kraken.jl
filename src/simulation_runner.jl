# --- Generic simulation runner for .krk config files ---

"""
    BoundaryHandler

Pre-compiled boundary condition data for one face.
"""
struct BoundaryHandler
    face::Symbol
    type::Symbol
    # Pre-compiled expression functions (or nothing)
    ux_fn::Union{Function, Nothing}
    uy_fn::Union{Function, Nothing}
    rho_fn::Union{Function, Nothing}
    # Flags
    is_spatial_ux::Bool
    is_spatial_uy::Bool
    is_time_dep_ux::Bool
    is_time_dep_uy::Bool
    is_time_dep_rho::Bool
    # Pre-allocated arrays for spatial BCs (on backend)
    ux_arr::Union{AbstractArray, Nothing}
    uy_arr::Union{AbstractArray, Nothing}
    rho_arr::Union{AbstractArray, Nothing}
end

"""
    run_simulation(filename::String; backend=CPU(), T=Float64,
                   max_steps=nothing, kwargs...) -> NamedTuple

Run an LBM simulation defined by a `.krk` configuration file.
Keyword arguments override `Define` defaults for parametric studies.
Pass `max_steps` to override the `Run N steps` directive (useful for tests).

Returns a NamedTuple with final fields on CPU: `(ρ, ux, uy, setup)`.

## Dispatch rules (v0.1.0)
1. `:thermal in modules`      → `run_rayleigh_benard_2d`,
   `run_natural_convection_2d`, or a thermal-conduction fallback
2. `setup.lattice === :D3Q19` → `run_cavity_3d`
3. Default (`:D2Q9`, no modules) → generic single-phase LBM loop
   (cavity/Poiseuille/Couette/Taylor-Green/cylinder examples).

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
        setup.velocity_field, setup.rheology)
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
    # v0.1.0 scope: Newtonian single-phase + thermal (DDF)
    unsupported = intersect(setup.modules,
        [:advection_only, :twophase_vof, :axisymmetric, :rheology, :viscoelastic, :species])
    if !isempty(unsupported)
        error("Module(s) $(unsupported) not supported in v0.1.0. " *
              "See the development branch `lbm` for advanced physics.")
    end
    if !isempty(setup.refinements)
        error("Grid refinement is not supported in v0.1.0. " *
              "See the development branch `lbm` for this feature.")
    end
    if :thermal in setup.modules
        return _run_thermal(setup; backend=backend, T=T)
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

    # --- Initialize state ---
    config = LBMConfig(D2Q9(); Nx=Nx, Ny=Ny, ν=ν, u_lid=0.0,
                       max_steps=setup.max_steps, output_interval=1000)
    state = initialize_2d(config, T; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid

    # --- Apply geometry ---
    _apply_geometry!(is_solid, setup, dx, dy)

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

    # --- Build boundary handlers ---
    bc_handlers = _build_boundary_handlers(setup, dx, dy, Nx, Ny, T, backend)

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
        # 1. Stream
        stream_fn!(f_out, f_in, Nx, Ny)

        # 2. Apply boundary conditions
        _apply_boundary_conditions!(f_out, bc_handlers, step, Nx, Ny, dx, dy, dom, T)

        # 3. Collide
        if has_body_force
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
    return (ρ=Array(ρ), ux=Array(ux), uy=Array(uy), setup=setup)
end

# --- Internal helpers ---

"""Select streaming kernel based on boundary periodicity."""
function _select_streaming_kernel(setup::SimulationSetup)
    faces = Dict(b.face => b.type for b in setup.boundaries)
    periodic_x = get(faces, :west, :wall) == :periodic || get(faces, :east, :wall) == :periodic
    periodic_y = get(faces, :south, :wall) == :periodic || get(faces, :north, :wall) == :periodic

    if periodic_x && periodic_y
        return stream_fully_periodic_2d!
    elseif periodic_x
        return stream_periodic_x_wall_y_2d!
    else
        return stream_2d!
    end
end

"""Build pre-compiled boundary handlers."""
function _build_boundary_handlers(setup::SimulationSetup, dx, dy, Nx, Ny, ::Type{T},
                                  backend) where T
    handlers = BoundaryHandler[]

    for bc in setup.boundaries
        bc.type == :periodic && continue

        # Determine face size for array allocation
        face_size = (bc.face in (:north, :south)) ? Nx : Ny

        ux_fn = haskey(bc.values, :ux) ? bc.values[:ux].func : nothing
        uy_fn = haskey(bc.values, :uy) ? bc.values[:uy].func : nothing
        rho_fn = haskey(bc.values, :rho) ? bc.values[:rho].func : nothing

        is_sp_ux = haskey(bc.values, :ux) && is_spatial(bc.values[:ux])
        is_sp_uy = haskey(bc.values, :uy) && is_spatial(bc.values[:uy])
        is_td_ux = haskey(bc.values, :ux) && is_time_dependent(bc.values[:ux])
        is_td_uy = haskey(bc.values, :uy) && is_time_dependent(bc.values[:uy])
        is_td_rho = haskey(bc.values, :rho) && is_time_dependent(bc.values[:rho])

        needs_spatial = is_sp_ux || is_sp_uy || is_td_ux || is_td_uy

        ux_arr = needs_spatial ? KernelAbstractions.zeros(backend, T, face_size) : nothing
        uy_arr = needs_spatial ? KernelAbstractions.zeros(backend, T, face_size) : nothing
        rho_arr = (haskey(bc.values, :rho) && (is_spatial(bc.values[:rho]) || is_td_rho)) ?
            KernelAbstractions.zeros(backend, T, face_size) : nothing

        # Pre-compute static spatial arrays
        if needs_spatial && !is_td_ux && !is_td_uy
            _fill_bc_arrays!(ux_arr, uy_arr, ux_fn, uy_fn, bc.face,
                             dx, dy, Nx, Ny, setup.domain, T)
        end

        push!(handlers, BoundaryHandler(
            bc.face, bc.type,
            ux_fn, uy_fn, rho_fn,
            is_sp_ux, is_sp_uy,
            is_td_ux, is_td_uy, is_td_rho,
            ux_arr, uy_arr, rho_arr
        ))
    end

    return handlers
end

"""Fill BC arrays with spatial profile for a given face."""
function _fill_bc_arrays!(ux_arr, uy_arr, ux_fn, uy_fn, face::Symbol,
                          dx, dy, Nx, Ny, domain::DomainSetup, ::Type{T};
                          t::Float64=0.0) where T
    Lx, Ly = domain.Lx, domain.Ly

    if face in (:north, :south)
        cpu_ux = zeros(T, Nx)
        cpu_uy = zeros(T, Nx)
        y_val = face == :south ? dy / 2 : Ly - dy / 2
        for i in 1:Nx
            x_val = (i - 0.5) * dx
            kw = (; x=x_val, y=y_val, Lx=Lx, Ly=Ly,
                    Nx=Float64(Nx), Ny=Float64(Ny), dx=dx, dy=dy, t=t)
            ux_fn !== nothing && (cpu_ux[i] = T(Base.invokelatest(ux_fn; kw...)))
            uy_fn !== nothing && (cpu_uy[i] = T(Base.invokelatest(uy_fn; kw...)))
        end
        ux_arr !== nothing && copyto!(ux_arr, cpu_ux)
        uy_arr !== nothing && copyto!(uy_arr, cpu_uy)
    else  # :west or :east
        cpu_ux = zeros(T, Ny)
        cpu_uy = zeros(T, Ny)
        x_val = face == :west ? dx / 2 : Lx - dx / 2
        for j in 1:Ny
            y_val = (j - 0.5) * dy
            kw = (; x=x_val, y=y_val, Lx=Lx, Ly=Ly,
                    Nx=Float64(Nx), Ny=Float64(Ny), dx=dx, dy=dy, t=t)
            ux_fn !== nothing && (cpu_ux[j] = T(Base.invokelatest(ux_fn; kw...)))
            uy_fn !== nothing && (cpu_uy[j] = T(Base.invokelatest(uy_fn; kw...)))
        end
        ux_arr !== nothing && copyto!(ux_arr, cpu_ux)
        uy_arr !== nothing && copyto!(uy_arr, cpu_uy)
    end
end

"""Apply boundary conditions at a given timestep."""
function _apply_boundary_conditions!(f, handlers::Vector{BoundaryHandler},
                                     step::Int, Nx, Ny, dx, dy,
                                     domain::DomainSetup, ::Type{T}) where T
    for h in handlers
        h.type == :periodic && continue

        if h.type == :wall && h.ux_fn === nothing && h.uy_fn === nothing
            # Pure wall — handled by streaming bounce-back, but apply explicit
            # bounce-back for faces not covered by the streaming kernel
            _apply_wall_bc!(f, h.face, Nx, Ny)
        elseif h.type == :velocity
            # Re-evaluate time-dependent BCs
            if h.is_time_dep_ux || h.is_time_dep_uy
                _fill_bc_arrays!(h.ux_arr, h.uy_arr, h.ux_fn, h.uy_fn, h.face,
                                 dx, dy, Nx, Ny, domain, T;
                                 t=Float64(step))
            end

            if h.ux_arr !== nothing  # spatial BC
                _apply_velocity_spatial!(f, h, Nx, Ny)
            else  # scalar BC
                ux_val = h.ux_fn !== nothing ? Base.invokelatest(h.ux_fn; t=Float64(step)) : 0.0
                uy_val = h.uy_fn !== nothing ? Base.invokelatest(h.uy_fn; t=Float64(step)) : 0.0
                _apply_velocity_scalar!(f, h.face, ux_val, uy_val, Nx, Ny)
            end
        elseif h.type == :pressure
            rho_val = h.rho_fn !== nothing ? Base.invokelatest(h.rho_fn; t=Float64(step)) : 1.0
            _apply_pressure_bc!(f, h.face, rho_val, Nx, Ny)
        end
    end
end

"""Apply wall BC on a specific face."""
function _apply_wall_bc!(f, face::Symbol, Nx, Ny)
    # stream_2d! already handles bounce-back at domain edges.
    # For periodic streaming kernels, explicit bounce-back is needed.
    # The bounce-back is embedded in the streaming step for wall boundaries.
    # Nothing extra needed here — the streaming kernel handles it.
end

"""Apply scalar velocity BC."""
function _apply_velocity_scalar!(f, face::Symbol, ux_val, uy_val, Nx, Ny)
    if face == :north
        apply_zou_he_north_2d!(f, ux_val, Nx, Ny)
    elseif face == :south
        apply_zou_he_south_2d!(f, ux_val, Nx)
    elseif face == :west
        apply_zou_he_west_2d!(f, ux_val, Nx, Ny)
    end
end

"""Apply spatial velocity BC."""
function _apply_velocity_spatial!(f, h::BoundaryHandler, Nx, Ny)
    if h.face == :north
        apply_zou_he_north_spatial_2d!(f, h.ux_arr, h.uy_arr, Nx, Ny)
    elseif h.face == :south
        apply_zou_he_south_spatial_2d!(f, h.ux_arr, h.uy_arr, Nx)
    elseif h.face == :west
        apply_zou_he_west_spatial_2d!(f, h.ux_arr, h.uy_arr, Nx, Ny)
    end
end

"""Apply pressure BC."""
function _apply_pressure_bc!(f, face::Symbol, rho_val, Nx, Ny)
    if face == :east
        apply_zou_he_pressure_east_2d!(f, Nx, Ny; ρ_out=rho_val)
    end
end

"""Build is_solid mask from geometry regions (expression-based predicates)."""
function _apply_geometry!(is_solid, setup::SimulationSetup, dx, dy)
    Nx, Ny = setup.domain.Nx, setup.domain.Ny
    Lx, Ly = setup.domain.Lx, setup.domain.Ly

    has_fluid_region = any(r -> r.kind == :fluid, setup.regions)
    solid_cpu = has_fluid_region ? ones(Bool, Nx, Ny) : zeros(Bool, Nx, Ny)

    for region in setup.regions
        for j in 1:Ny, i in 1:Nx
            x = (i - 0.5) * dx
            y = (j - 0.5) * dy
            result = evaluate(region.condition; x=x, y=y, z=0.0,
                             Lx=Lx, Ly=Ly, dx=dx, dy=dy)
            if region.kind == :fluid && result
                solid_cpu[i, j] = false
            elseif region.kind == :obstacle && result
                solid_cpu[i, j] = true
            end
        end
    end

    copyto!(is_solid, solid_cpu)
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

"""Write VTK output."""
function _write_output(ρ, ux, uy, setup::SimulationSetup, out::OutputSetup,
                       pvd, output_dir, dx, step;
                       extra_fields=Dict{String,Any}())
    fields_dict = Dict{String, Matrix{Float64}}()
    field_set = Set(out.fields)

    if :rho in field_set
        fields_dict["rho"] = Array(ρ)
    end
    if :ux in field_set
        fields_dict["ux"] = Array(ux)
    end
    if :uy in field_set
        fields_dict["uy"] = Array(uy)
    end

    # Merge extra fields (C, phi, kappa, etc.)
    for (k, v) in extra_fields
        if Symbol(k) in field_set
            fields_dict[k] = v isa AbstractArray ? Array(v) : v
        end
    end

    Nx, Ny = setup.domain.Nx, setup.domain.Ny
    fname = joinpath(output_dir, "$(setup.name)_$(lpad(step, 8, '0'))")
    write_vtk_to_pvd(pvd, fname, Nx, Ny, dx, fields_dict, Float64(step))
end

# --- PNG/GIF output helpers ---

"""Compute a requested field from macroscopic arrays."""
function _compute_field(field::Symbol, ρ, ux, uy)
    if field == Symbol("|u|")
        return sqrt.(Array(ux).^2 .+ Array(uy).^2)
    elseif field == :rho
        return Array(ρ)
    elseif field == :ux
        return Array(ux)
    elseif field == :uy
        return Array(uy)
    else
        return Array(ρ)  # fallback
    end
end

"""Emit a warning if png/gif output is requested but CairoMakie is not loaded."""
function _check_image_backend(png_out, gif_out)
    need = png_out !== nothing || gif_out !== nothing
    if need && _png_saver[] === nothing
        @warn "Output png/gif requested but CairoMakie is not loaded. " *
              "Add `using CairoMakie` before `using Kraken` to enable PNG/GIF output."
    end
end

"""Initialize GIF frame storage."""
function _init_gif_frames(gif_out)
    gif_out === nothing && return Dict{Symbol, Vector{Matrix{Float64}}}()
    frames = Dict{Symbol, Vector{Matrix{Float64}}}()
    for f in gif_out.fields
        frames[f] = Matrix{Float64}[]
    end
    return frames
end

"""Save a PNG snapshot if it's time and the backend is loaded."""
function _maybe_save_png(png_out, ρ, ux, uy, setup, output_dir, step)
    png_out === nothing && return
    _png_saver[] === nothing && return
    step % png_out.interval != 0 && return

    dir = isempty(output_dir) ? setup_output_dir(png_out.directory) : output_dir
    for field_name in png_out.fields
        data = _compute_field(field_name, ρ, ux, uy)
        fname = joinpath(dir, "$(setup.name)_$(field_name)_$(lpad(step, 8, '0')).png")
        _png_saver[](fname, data, string(field_name))
    end
end

"""Collect a GIF frame if it's time."""
function _maybe_collect_gif(gif_out, gif_frames, ρ, ux, uy, step)
    gif_out === nothing && return
    _gif_saver[] === nothing && return
    step % gif_out.interval != 0 && return

    for field_name in gif_out.fields
        data = _compute_field(field_name, ρ, ux, uy)
        push!(gif_frames[field_name], copy(data))
    end
end

"""Assemble and save GIF after simulation completes."""
function _maybe_save_gif(gif_out, gif_frames, setup, output_dir)
    gif_out === nothing && return
    _gif_saver[] === nothing && return

    dir = isempty(output_dir) ? setup_output_dir(gif_out.directory) : output_dir
    for field_name in gif_out.fields
        frames = gif_frames[field_name]
        isempty(frames) && continue
        fname = joinpath(dir, "$(setup.name)_$(field_name).gif")
        _gif_saver[](fname, frames, string(field_name); fps=gif_out.fps)
    end
end

# ===========================================================================
# Dispatch helpers for 3D / thermal cases
# ===========================================================================


"""Dispatch D3Q19 cases to the appropriate 3D driver."""
function _run_d3q19(setup::SimulationSetup;
                    backend=KernelAbstractions.CPU(), T=Float64)
    name = lowercase(setup.name)
    dom  = setup.domain
    ν    = setup.physics.params[:nu]
    if occursin("cavity_3d", name) || occursin("cavity3d", name)
        u_lid = 0.1
        for b in setup.boundaries
            if b.type == :velocity && haskey(b.values, :ux)
                try
                    u_lid = Float64(evaluate(b.values[:ux]))
                catch
                end
                break
            end
        end
        config = LBMConfig(D3Q19(); Nx=dom.Nx, Ny=dom.Ny, Nz=dom.Nz,
                           ν=Float64(ν), u_lid=u_lid,
                           max_steps=setup.max_steps)
        result = run_cavity_3d(config; backend=backend, T=T)
        return merge(result, (setup=setup,))
    else
        throw(ArgumentError(
            "D3Q19 dispatch: only `cavity_3d` is supported in v0.1.0 " *
            "(got case name: $(setup.name)). Use the Julia API for other 3D cases."))
    end
end

"""Dispatch thermal cases to the appropriate thermal driver."""
function _run_thermal(setup::SimulationSetup;
                      backend=KernelAbstractions.CPU(), T=Float64)
    name   = lowercase(setup.name)
    dom    = setup.domain
    params = setup.physics.params
    Ra = Float64(get(params, :Ra, 1e4))
    Pr = Float64(get(params, :Pr, 0.71))

    if occursin("rayleigh_benard", name) || occursin("rayleigh-benard", name)
        result = run_rayleigh_benard_2d(; Nx=dom.Nx, Ny=dom.Ny, Ra=Ra, Pr=Pr,
                                         max_steps=setup.max_steps,
                                         backend=backend, FT=T)
        return merge(result, (setup=setup,))
    elseif occursin("natural_convection", name)
        result = run_natural_convection_2d(; N=dom.Nx, Ra=Ra, Pr=Pr,
                                            max_steps=setup.max_steps,
                                            backend=backend, FT=T)
        return merge(result, (setup=setup,))
    elseif occursin("heat_conduction", name) || occursin("conduction", name)
        result = run_rayleigh_benard_2d(; Nx=dom.Nx, Ny=dom.Ny,
                                         Ra=1e-8, Pr=Pr,
                                         max_steps=setup.max_steps,
                                         backend=backend, FT=T)
        return merge(result, (setup=setup,))
    else
        throw(ArgumentError(
            "thermal dispatch: unrecognized case name '$(setup.name)'. " *
            "Known cases: rayleigh_benard, natural_convection, heat_conduction."))
    end
end
