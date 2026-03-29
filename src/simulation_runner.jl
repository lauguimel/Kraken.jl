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
    run_simulation(filename::String; backend=CPU(), T=Float64) -> NamedTuple

Run an LBM simulation defined by a `.krk` configuration file.

Returns a NamedTuple with final fields on CPU: `(ρ, ux, uy, setup)`.

# Example
```julia
result = run_simulation("examples/cavity.krk")
result.ux  # velocity field
```
"""
function run_simulation(filename::String;
                        backend=KernelAbstractions.CPU(), T=Float64)
    setup = load_kraken(filename)
    return run_simulation(setup; backend=backend, T=T)
end

"""
    run_simulation(setup::SimulationSetup; backend=CPU(), T=Float64)

Run an LBM simulation from a parsed `SimulationSetup`.
"""
function run_simulation(setup::SimulationSetup;
                        backend=KernelAbstractions.CPU(), T=Float64)
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
    pvd = nothing
    output_dir = ""
    if setup.output !== nothing
        output_dir = setup_output_dir(setup.output.directory)
        pvd = create_pvd(joinpath(output_dir, setup.name))
    end

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

        # 6. Output
        if setup.output !== nothing && step % setup.output.interval == 0
            _write_output(ρ, ux, uy, setup, pvd, output_dir, dx, step)
        end
    end

    # Finalize PVD
    if pvd !== nothing
        vtk_save(pvd)
    end

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

"""Build is_solid mask from geometry regions (condition expressions or STL files)."""
function _apply_geometry!(is_solid, setup::SimulationSetup, dx, dy)
    Nx, Ny = setup.domain.Nx, setup.domain.Ny
    Lx, Ly = setup.domain.Lx, setup.domain.Ly

    has_fluid_region = any(r -> r.kind == :fluid, setup.regions)
    solid_cpu = has_fluid_region ? ones(Bool, Nx, Ny) : zeros(Bool, Nx, Ny)

    for region in setup.regions
        if region.stl !== nothing
            # STL-based geometry
            stl_mask = _voxelize_stl_region(region.stl, Nx, Ny, dx, dy)
            for j in 1:Ny, i in 1:Nx
                if region.kind == :fluid && stl_mask[i, j]
                    solid_cpu[i, j] = false
                elseif region.kind == :obstacle && stl_mask[i, j]
                    solid_cpu[i, j] = true
                end
            end
        else
            # Expression-based geometry
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
    end

    copyto!(is_solid, solid_cpu)
end

"""Load and voxelize an STL file for a 2D simulation (z-plane cross-section)."""
function _voxelize_stl_region(stl_src::STLSource, Nx, Ny, dx, dy)
    mesh = read_stl(stl_src.file)
    if stl_src.scale != 1.0 || stl_src.translate != (0.0, 0.0, 0.0)
        mesh = transform_mesh(mesh; scale=stl_src.scale, translate=stl_src.translate)
    end
    return voxelize_2d(mesh, Nx, Ny, dx, dy; z_slice=stl_src.z_slice)
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
function _write_output(ρ, ux, uy, setup::SimulationSetup, pvd, output_dir, dx, step)
    fields_dict = Dict{String, Matrix{Float64}}()
    field_set = Set(setup.output.fields)

    if :rho in field_set
        fields_dict["rho"] = Array(ρ)
    end
    if :ux in field_set
        fields_dict["ux"] = Array(ux)
    end
    if :uy in field_set
        fields_dict["uy"] = Array(uy)
    end

    Nx, Ny = setup.domain.Nx, setup.domain.Ny
    fname = joinpath(output_dir, "$(setup.name)_$(lpad(step, 8, '0'))")
    write_vtk_to_pvd(pvd, fname, Nx, Ny, dx, fields_dict, Float64(step))
end
