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

## Dispatch rules
The runner selects a backend driver based on `setup.modules`, `setup.lattice`,
`setup.refinements`, and the `setup.name` (case name):

1. `:advection_only in modules`  → pure VOF advection (no LBM solve)
2. `:twophase_vof   in modules`  → two-phase LBM with surface tension
3. `:axisymmetric   in modules`  → `run_hagen_poiseuille_2d` if the case
   name contains `hagen_poiseuille`, otherwise an informative error
4. `!isempty(setup.refinements)` → refined-grid drivers. Only refined
   natural convection is currently supported via the .krk runner;
   other refined cases raise an informative error (run them via the
   Julia API — see `create_refined_domain` / `create_thermal_patch_arrays`).
5. `:thermal in modules`         → `run_rayleigh_benard_2d`,
   `run_natural_convection_2d`, or a thermal-conduction fallback
   depending on the case name
6. `setup.lattice === :D3Q19`    → `run_cavity_3d` if the case name
   contains `cavity_3d`, otherwise an informative error
7. Default (`:D2Q9`, no modules) → generic single-phase LBM loop
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
        setup.velocity_field, setup.rheology, setup.mesh)
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
    if setup.mesh !== nothing
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

function _edge_profile_host(block::Block, edge::Symbol, ::Type{T}, Ly, u_max) where T
    n = edge_length(block, edge)
    profile = zeros(T, n)
    for r in 1:n
        i, j = _edge_node(block, edge, r)
        profile[r] = T(_parabolic_channel_u(block.mesh.Y[i, j], Ly, u_max))
    end
    return profile
end

function _physical_normal_from_tag(tag::Symbol)
    tag === :inlet && return :west
    tag === :outlet && return :east
    tag in (:wall_bot, :wall_bottom, :bottom) && return :south
    tag in (:wall_top, :wall_upper, :top) && return :north
    return :auto
end

function _mesh_drag_bc(block::Block, edge::Symbol, tag::Symbol,
                       setup::SimulationSetup, backend, ::Type{T},
                       Ly, u_max, rho_out) where T
    tag === INTERFACE_TAG && return InterfaceBC()
    if tag === :inlet
        profile = _copy_to_backend(backend, T,
                                   _edge_profile_host(block, edge, T, Ly, u_max))
        return ZouHeVelocity(profile, _physical_normal_from_tag(tag))
    elseif tag === :outlet
        return ZouHePressure(T(rho_out), _physical_normal_from_tag(tag))
    end
    return HalfwayBB()
end

function _mesh_drag_bcspec(block::Block, setup::SimulationSetup,
                           backend, ::Type{T}, Ly, u_max, rho_out) where T
    tags = block.boundary_tags
    return BCSpec2D(;
        west=_mesh_drag_bc(block, :west, tags.west, setup, backend, T,
                           Ly, u_max, rho_out),
        east=_mesh_drag_bc(block, :east, tags.east, setup, backend, T,
                           Ly, u_max, rho_out),
        south=_mesh_drag_bc(block, :south, tags.south, setup, backend, T,
                            Ly, u_max, rho_out),
        north=_mesh_drag_bc(block, :north, tags.north, setup, backend, T,
                            Ly, u_max, rho_out))
end

function _mesh_drag_noop_bcspec(block::Block)
    function bc_for(tag::Symbol)
        (tag === INTERFACE_TAG || tag === :interface) && return InterfaceBC()
        return HalfwayBB()
    end
    tags = block.boundary_tags
    return BCSpec2D(;
        west=bc_for(tags.west),
        east=bc_for(tags.east),
        south=bc_for(tags.south),
        north=bc_for(tags.north))
end

_edge_code_2d(edge::Symbol) =
    edge === :west ? 1 :
    edge === :east ? 2 :
    edge === :south ? 3 :
    edge === :north ? 4 :
    error("unknown edge $edge")

_normal_code_2d(normal::Symbol) =
    normal === :west ? 1 :
    normal === :east ? 2 :
    normal === :south ? 3 :
    normal === :north ? 4 :
    error("unknown physical normal $normal")

@kernel function _mesh_drag_physnorm_velocity_edge_2d!(f, edge_code::Int,
                                                       normal_code::Int,
                                                       profile, Nx::Int,
                                                       Ny::Int)
    r = @index(Global)
    T = eltype(f)
    i = edge_code == 1 ? 1  :
        edge_code == 2 ? Nx :
        r
    j = edge_code == 3 ? 1  :
        edge_code == 4 ? Ny :
        r
    @inbounds begin
        f1 = f[i, j, 1]; f2 = f[i, j, 2]; f3 = f[i, j, 3]
        f4 = f[i, j, 4]; f5 = f[i, j, 5]; f6 = f[i, j, 6]
        f7 = f[i, j, 7]; f8 = f[i, j, 8]; f9 = f[i, j, 9]
        u = profile[r]
        if normal_code == 1
            rho = (f1 + f3 + f5 + T(2) * (f4 + f7 + f8)) / (one(T) - u)
            f2 = f4 + T(2 / 3) * rho * u
            f6 = f8 - T(0.5) * (f3 - f5) + T(1 / 6) * rho * u
            f9 = f7 + T(0.5) * (f3 - f5) + T(1 / 6) * rho * u
        elseif normal_code == 2
            rho = (f1 + f3 + f5 + T(2) * (f2 + f6 + f9)) / (one(T) + u)
            f4 = f2 - T(2 / 3) * rho * u
            f7 = f9 - T(0.5) * (f3 - f5) - T(1 / 6) * rho * u
            f8 = f6 + T(0.5) * (f3 - f5) - T(1 / 6) * rho * u
        elseif normal_code == 3
            rho = (f1 + f2 + f4 + T(2) * (f5 + f8 + f9)) / (one(T) - u)
            f3 = f5 + T(2 / 3) * rho * u
            f6 = f8 + T(0.5) * (f4 - f2) + T(1 / 6) * rho * u
            f7 = f9 + T(0.5) * (f2 - f4) + T(1 / 6) * rho * u
        else
            rho = (f1 + f2 + f4 + T(2) * (f3 + f6 + f7)) / (one(T) + u)
            f5 = f3 - T(2 / 3) * rho * u
            f8 = f6 + T(0.5) * (f2 - f4) - T(1 / 6) * rho * u
            f9 = f7 + T(0.5) * (f4 - f2) - T(1 / 6) * rho * u
        end
        f[i, j, 1] = f1; f[i, j, 2] = f2; f[i, j, 3] = f3
        f[i, j, 4] = f4; f[i, j, 5] = f5; f[i, j, 6] = f6
        f[i, j, 7] = f7; f[i, j, 8] = f8; f[i, j, 9] = f9
    end
end

@kernel function _mesh_drag_physnorm_pressure_edge_2d!(f, edge_code::Int,
                                                       normal_code::Int,
                                                       rho_out, Nx::Int,
                                                       Ny::Int)
    r = @index(Global)
    T = eltype(f)
    i = edge_code == 1 ? 1  :
        edge_code == 2 ? Nx :
        r
    j = edge_code == 3 ? 1  :
        edge_code == 4 ? Ny :
        r
    rho = T(rho_out)
    @inbounds begin
        f1 = f[i, j, 1]; f2 = f[i, j, 2]; f3 = f[i, j, 3]
        f4 = f[i, j, 4]; f5 = f[i, j, 5]; f6 = f[i, j, 6]
        f7 = f[i, j, 7]; f8 = f[i, j, 8]; f9 = f[i, j, 9]
        if normal_code == 1
            u = one(T) - (f1 + f3 + f5 + T(2) * (f4 + f7 + f8)) / rho
            f2 = f4 + T(2 / 3) * rho * u
            f6 = f8 - T(0.5) * (f3 - f5) + T(1 / 6) * rho * u
            f9 = f7 + T(0.5) * (f3 - f5) + T(1 / 6) * rho * u
        elseif normal_code == 2
            u = -one(T) + (f1 + f3 + f5 + T(2) * (f2 + f6 + f9)) / rho
            f4 = f2 - T(2 / 3) * rho * u
            f7 = f9 - T(0.5) * (f3 - f5) - T(1 / 6) * rho * u
            f8 = f6 + T(0.5) * (f3 - f5) - T(1 / 6) * rho * u
        elseif normal_code == 3
            u = one(T) - (f1 + f2 + f4 + T(2) * (f5 + f8 + f9)) / rho
            f3 = f5 + T(2 / 3) * rho * u
            f6 = f8 + T(0.5) * (f4 - f2) + T(1 / 6) * rho * u
            f7 = f9 + T(0.5) * (f2 - f4) + T(1 / 6) * rho * u
        else
            u = -one(T) + (f1 + f2 + f4 + T(2) * (f3 + f6 + f7)) / rho
            f5 = f3 - T(2 / 3) * rho * u
            f8 = f6 + T(0.5) * (f2 - f4) - T(1 / 6) * rho * u
            f9 = f7 + T(0.5) * (f4 - f2) - T(1 / 6) * rho * u
        end
        f[i, j, 1] = f1; f[i, j, 2] = f2; f[i, j, 3] = f3
        f[i, j, 4] = f4; f[i, j, 5] = f5; f[i, j, 6] = f6
        f[i, j, 7] = f7; f[i, j, 8] = f8; f[i, j, 9] = f9
    end
end

function _apply_mesh_drag_physical_normal_edge_2d!(f, edge::Symbol,
                                                   bc::ZouHeVelocity,
                                                   Nx::Int, Ny::Int)
    bc.physical_dir === :auto && return nothing
    backend = KernelAbstractions.get_backend(f)
    nrun = edge in (:west, :east) ? Ny : Nx
    _mesh_drag_physnorm_velocity_edge_2d!(backend)(
        f, _edge_code_2d(edge), _normal_code_2d(bc.physical_dir),
        bc.profile, Nx, Ny; ndrange=(nrun,))
    return nothing
end

function _apply_mesh_drag_physical_normal_edge_2d!(f, edge::Symbol,
                                                   bc::ZouHePressure,
                                                   Nx::Int, Ny::Int)
    bc.physical_dir === :auto && return nothing
    backend = KernelAbstractions.get_backend(f)
    nrun = edge in (:west, :east) ? Ny : Nx
    _mesh_drag_physnorm_pressure_edge_2d!(backend)(
        f, _edge_code_2d(edge), _normal_code_2d(bc.physical_dir),
        eltype(f)(bc.ρ_out), Nx, Ny; ndrange=(nrun,))
    return nothing
end

function _apply_mesh_drag_physical_normal_edge_2d!(f, edge::Symbol,
                                                   bc::AbstractBC,
                                                   Nx::Int, Ny::Int)
    return nothing
end

function _apply_mesh_drag_physical_normal_bcs_2d!(f, bcspec::BCSpec2D,
                                                  Nx::Int, Ny::Int)
    _apply_mesh_drag_physical_normal_edge_2d!(f, :west, bcspec.west, Nx, Ny)
    _apply_mesh_drag_physical_normal_edge_2d!(f, :east, bcspec.east, Nx, Ny)
    _apply_mesh_drag_physical_normal_edge_2d!(f, :south, bcspec.south, Nx, Ny)
    _apply_mesh_drag_physical_normal_edge_2d!(f, :north, bcspec.north, Nx, Ny)
    return nothing
end

function _circle_solid_field(block::Block, cx, cy, radius)
    solid = zeros(Bool, block.mesh.Nξ, block.mesh.Nη)
    r2 = Float64(radius)^2
    tol = max(1e-14, 1e-10 * max(1.0, r2))
    for j in 1:block.mesh.Nη, i in 1:block.mesh.Nξ
        dx = Float64(block.mesh.X[i, j]) - Float64(cx)
        dy = Float64(block.mesh.Y[i, j]) - Float64(cy)
        solid[i, j] = dx * dx + dy * dy <= r2 + tol
    end
    return solid
end

function _mesh_curved_edges(block::Block)
    edges = Symbol[]
    for edge in EDGE_SYMBOLS_2D
        getproperty(block.boundary_tags, edge) === :cylinder && push!(edges, edge)
    end
    return Tuple(edges)
end

function _edge_inner_node(block::Block, edge::Symbol, r::Int)
    edge === :west  && return min(2, block.mesh.Nξ), r
    edge === :east  && return max(block.mesh.Nξ - 1, 1), r
    edge === :south && return r, min(2, block.mesh.Nη)
    edge === :north && return r, max(block.mesh.Nη - 1, 1)
    error("unknown edge $edge")
end

function _compute_bodyfit_cylinder_force_2d(mbm::MultiBlockMesh2D, states,
                                            cx, cy, radius, nu, ng::Int)
    Fx = 0.0
    Fy = 0.0
    inv_cs2_den = 1.0 / 3.0
    epsd = eps(Float64)

    for (block, st) in zip(mbm.blocks, states)
        rho_h = Array(st.ρ)
        ux_h = Array(st.ux)
        uy_h = Array(st.uy)
        for edge in EDGE_SYMBOLS_2D
            getproperty(block.boundary_tags, edge) === :cylinder || continue
            nedge = edge_length(block, edge)
            nedge < 2 && continue
            for r in 1:(nedge - 1)
                ib0, jb0 = _edge_node(block, edge, r)
                ib1, jb1 = _edge_node(block, edge, r + 1)
                ii0, ji0 = _edge_inner_node(block, edge, r)
                ii1, ji1 = _edge_inner_node(block, edge, r + 1)

                x0 = Float64(block.mesh.X[ib0, jb0])
                y0 = Float64(block.mesh.Y[ib0, jb0])
                x1 = Float64(block.mesh.X[ib1, jb1])
                y1 = Float64(block.mesh.Y[ib1, jb1])
                xm = 0.5 * (x0 + x1)
                ym = 0.5 * (y0 + y1)
                ds = hypot(x1 - x0, y1 - y0)

                nx = xm - Float64(cx)
                ny = ym - Float64(cy)
                nrm = max(hypot(nx, ny), epsd)
                nx /= nrm
                ny /= nrm
                tx = -ny
                ty = nx

                r0 = rho_h[ii0 + ng, ji0 + ng]
                r1 = rho_h[ii1 + ng, ji1 + ng]
                ux0 = ux_h[ii0 + ng, ji0 + ng]
                ux1 = ux_h[ii1 + ng, ji1 + ng]
                uy0 = uy_h[ii0 + ng, ji0 + ng]
                uy1 = uy_h[ii1 + ng, ji1 + ng]
                rho = 0.5 * (Float64(r0) + Float64(r1))
                ux = 0.5 * (Float64(ux0) + Float64(ux1))
                uy = 0.5 * (Float64(uy0) + Float64(uy1))

                xi0 = Float64(block.mesh.X[ii0, ji0])
                yi0 = Float64(block.mesh.Y[ii0, ji0])
                xi1 = Float64(block.mesh.X[ii1, ji1])
                yi1 = Float64(block.mesh.Y[ii1, ji1])
                dist0 = abs((xi0 - x0) * nx + (yi0 - y0) * ny)
                dist1 = abs((xi1 - x1) * nx + (yi1 - y1) * ny)
                wall_dist = max(0.5 * (dist0 + dist1), epsd)

                # The constant pressure part cancels on a closed boundary;
                # subtracting rho=1 reduces quadrature error on coarse O-grids.
                p = (rho - 1.0) * inv_cs2_den
                ut = ux * tx + uy * ty
                tau = rho * Float64(nu) * ut / wall_dist
                Fx += (-p * nx + tau * tx) * ds
                Fy += (-p * ny + tau * ty) * ds
            end
        end
    end
    return (; Fx, Fy)
end

function _check_block_density(states, step::Int, label::AbstractString)
    rho_min = Inf
    rho_max = -Inf
    for st in states
        rho_h = Array(st.ρ)
        ng = st.n_ghost
        phys = @view rho_h[(ng + 1):(ng + st.Nξ_phys),
                           (ng + 1):(ng + st.Nη_phys)]
        any(!isfinite, phys) && error("non-finite density in $label at step $step")
        rho_min = min(rho_min, minimum(phys))
        rho_max = max(rho_max, maximum(phys))
    end
    return Float64(rho_min), Float64(rho_max)
end

function _run_gmsh_slbm_drag(setup::SimulationSetup;
                             backend=KernelAbstractions.CPU(), T=Float64,
                             callback::Union{Nothing,Function}=nothing,
                             callback_every::Int=100)
    mesh_setup = setup.mesh
    mesh_setup === nothing && error("Gmsh SLBM drag runner requires setup.mesh")
    mesh_setup.kind === :gmsh || error("slbm_drag only supports Mesh gmsh(...); got $(mesh_setup.kind)")
    mesh_setup.multiblock || error("slbm_drag currently expects multiblock = true")
    isfile(mesh_setup.file) || error("Gmsh mesh file not found: $(mesh_setup.file)")

    mbm_raw, _ = load_gmsh_multiblock_2d(mesh_setup.file;
                                         FT=Float64,
                                         layout=mesh_setup.layout)
    mbm = autoreorient_blocks(mbm_raw; verbose=false,
                              respect_physical_tags=false)
    issues = sanity_check_multiblock(mbm; verbose=false)
    errors = filter(issue -> issue.severity === :error, issues)
    isempty(errors) || error("Gmsh multi-block mesh failed sanity checks:\n" *
                             join(string.(errors), "\n"))

    steps = setup.max_steps
    ng = max(1, round(Int, _setup_number(setup, :ng, 1.0)))
    sample_every = max(1, round(Int, _setup_number(setup, :sample_every, 10.0)))
    avg_window = max(1, min(steps, round(Int,
        _setup_number(setup, :avg_window, min(1000.0, Float64(steps))))))
    check_every = max(1, round(Int, _setup_number(setup, :check_every, 250.0)))

    Lx = _setup_number(setup, (:Lx, :lx), setup.domain.Lx)
    Ly = _setup_number(setup, (:Ly, :ly, :H), setup.domain.Ly)
    cx = _setup_number(setup, (:cx, :c_x), 0.25 * Lx)
    cy = _setup_number(setup, (:cy, :c_y), 0.5 * Ly)
    radius = _setup_number(setup, (:R, :radius), 0.05 * min(Lx, Ly))
    u_max = _setup_number(setup, (:u_max, :U, :umax), 0.04)
    u_ref = _setup_number(setup, (:u_ref, :U_ref), (2.0 / 3.0) * u_max)
    rho_out = _setup_number(setup, (:rho_out, :rho), 1.0)
    Re = _setup_number(setup, (:Re, :reynolds), NaN)
    embedded_solid = _setup_number(setup, (:embedded_solid, :use_libb_cutlinks), 0.0) > 0.5
    cylinder_reflect_ghost =
        !embedded_solid &&
        _setup_number(setup, (:cylinder_reflect_ghost, :bodyfit_reflect_ghost), 0.0) > 0.5

    dx_ref = minimum(block.mesh.dx_ref for block in mbm.blocks)
    D_eff = 2.0 * Float64(radius) / Float64(dx_ref)
    nu = if haskey(setup.physics.params, :nu)
        setup.physics.params[:nu]
    elseif !isnan(Re)
        Float64(u_ref) * D_eff / Float64(Re)
    else
        error("slbm_drag needs either Physics nu = ... or Physics Re = ...")
    end
    isnan(Re) && (Re = Float64(u_ref) * D_eff / Float64(nu))

    n_blocks = length(mbm.blocks)
    states = [_allocate_block_state_as(block, T, backend, ng)
              for block in mbm.blocks]
    geom_ext = Vector{Any}(undef, n_blocks)
    sp_ext = Vector{Any}(undef, n_blocks)
    sm_ext = Vector{Any}(undef, n_blocks)
    sp_int = Vector{Any}(undef, n_blocks)
    sm_int = Vector{Any}(undef, n_blocks)
    solid_ext = Vector{Any}(undef, n_blocks)
    qwall_ext = Vector{Any}(undef, n_blocks)
    uwx_ext = Vector{Any}(undef, n_blocks)
    uwy_ext = Vector{Any}(undef, n_blocks)
    solid_cells = 0

    for (k, block) in enumerate(mbm.blocks)
        curved_edges = _mesh_curved_edges(block)
        mesh_ext, geom_h = build_block_slbm_geometry_extended(block;
            n_ghost=ng, local_cfl=false, dx_ref=dx_ref,
            curved_edges=curved_edges,
            curved_center=(Float64(cx), Float64(cy)))
        geom_t = SLBMGeometry{T, Array{T,3}}(
            T.(geom_h.i_dep), T.(geom_h.j_dep),
            geom_h.Nξ, geom_h.Nη,
            geom_h.periodic_ξ, geom_h.periodic_η,
            T(geom_h.dx_ref))
        geom_ext[k] = transfer_slbm_geometry(geom_t, backend)

        sp_h, sm_h = compute_local_omega_2d(mesh_ext; ν=nu,
                                            scaling=:quadratic,
                                            τ_floor=0.51)
        sp_ext[k] = _copy_to_backend(backend, T, sp_h)
        sm_ext[k] = _copy_to_backend(backend, T, sm_h)
        sp_int[k] = _copy_to_backend(backend, T,
            sp_h[(ng + 1):(ng + block.mesh.Nξ),
                 (ng + 1):(ng + block.mesh.Nη)])
        sm_int[k] = _copy_to_backend(backend, T,
            sm_h[(ng + 1):(ng + block.mesh.Nξ),
                 (ng + 1):(ng + block.mesh.Nη)])

        # For a Gmsh O-grid the cylinder is the physical mesh boundary, not an
        # immersed obstacle. Marking boundary nodes as solid activates the
        # embedded LI-BB/cut-link path and corrupts the body-fitted wall model.
        solid_int = embedded_solid ? _circle_solid_field(block, cx, cy, radius) :
                    falses(block.mesh.Nξ, block.mesh.Nη)
        solid_cells += count(identity, solid_int)
        qwall_h, uwx_h, uwy_h = if embedded_solid
            precompute_q_wall_slbm_cylinder_2d(
                block.mesh, solid_int, cx, cy, radius; FT=Float64)
        else
            (zeros(Float64, block.mesh.Nξ, block.mesh.Nη, 9),
             zeros(Float64, block.mesh.Nξ, block.mesh.Nη, 9),
             zeros(Float64, block.mesh.Nξ, block.mesh.Nη, 9))
        end

        solid_ext[k] = _copy_bool_to_backend(backend,
            extend_interior_field_2d(solid_int, ng))
        qwall_ext[k] = _copy_to_backend(backend, T,
            extend_interior_field_2d(qwall_h, ng))
        uwx_ext[k] = _copy_to_backend(backend, T,
            extend_interior_field_2d(uwx_h, ng))
        uwy_ext[k] = _copy_to_backend(backend, T,
            extend_interior_field_2d(uwy_h, ng))

        f_init = zeros(T, block.mesh.Nξ, block.mesh.Nη, 9)
        for j in 1:block.mesh.Nη, i in 1:block.mesh.Nξ, q in 1:9
            u = solid_int[i, j] ? zero(T) :
                T(_parabolic_channel_u(block.mesh.Y[i, j], Ly, u_max))
            f_init[i, j, q] = T(equilibrium(D2Q9(), one(T), u, zero(T), q))
        end
        copyto!(interior_f(states[k]), f_init)
    end

    f_out = [KernelAbstractions.allocate(backend, T, size(st.f)...)
             for st in states]
    for buf in f_out
        fill!(buf, zero(T))
    end
    physical_bcspecs = [_mesh_drag_bcspec(block, setup, backend, T,
                                          Ly, u_max, rho_out)
                        for block in mbm.blocks]
    rebuild_bcspecs = [_mesh_drag_noop_bcspec(block) for block in mbm.blocks]

    cd_samples = Float64[]
    cl_samples = Float64[]
    history = NamedTuple[]
    Fx_sum = 0.0
    Fy_sum = 0.0
    n_avg = 0
    rho_min, rho_max = _check_block_density(states, 0, setup.name)
    t0 = time()

    for step in 1:steps
        exchange_ghost_shared_node_2d!(mbm, states)
        fill_ghost_corners_2d!(mbm, states)
        fill_slbm_wall_ghost_2d!(mbm, states)
        fill_physical_wall_ghost_2d!(mbm, states)
        cylinder_reflect_ghost && fill_tagged_reflection_ghost_2d!(mbm, states, :cylinder)

        for k in 1:n_blocks
            slbm_trt_libb_step_local_2d!(f_out[k], states[k].f,
                states[k].ρ, states[k].ux, states[k].uy,
                solid_ext[k], qwall_ext[k], uwx_ext[k], uwy_ext[k],
                geom_ext[k], sp_ext[k], sm_ext[k])
        end

        for k in 1:n_blocks
            nxp = states[k].Nξ_phys
            nyp = states[k].Nη_phys
            int_out = view(f_out[k], (ng + 1):(ng + nxp),
                           (ng + 1):(ng + nyp), :)
            int_in = view(states[k].f, (ng + 1):(ng + nxp),
                          (ng + 1):(ng + nyp), :)
            apply_bc_rebuild_2d!(int_out, int_in, rebuild_bcspecs[k], nu, nxp, nyp;
                                 sp_field=sp_int[k], sm_field=sm_int[k])
            _apply_mesh_drag_physical_normal_bcs_2d!(int_out, physical_bcspecs[k],
                                                     nxp, nyp)
        end

        for k in 1:n_blocks
            states[k].f, f_out[k] = f_out[k], states[k].f
        end

        if step > steps - avg_window && (step % sample_every == 0 || step == steps)
            drag = _compute_bodyfit_cylinder_force_2d(mbm, states, cx, cy,
                                                      radius, nu, ng)
            Fx = Float64(drag.Fx)
            Fy = Float64(drag.Fy)
            Cd = 2.0 * Fx / (Float64(u_ref)^2 * D_eff)
            Cl = 2.0 * Fy / (Float64(u_ref)^2 * D_eff)
            push!(cd_samples, Cd)
            push!(cl_samples, Cl)
            push!(history, (; step=step, Cd=Cd, Cl=Cl, Fx=Fx, Fy=Fy))
            Fx_sum += Fx
            Fy_sum += Fy
            n_avg += 1
        end

        if step % check_every == 0 || step == steps
            rho_min, rho_max = _check_block_density(states, step, setup.name)
        end
        if callback !== nothing && step % callback_every == 0
            Cd_now = isempty(cd_samples) ? NaN : cd_samples[end]
            Cl_now = isempty(cl_samples) ? NaN : cl_samples[end]
            callback(step, (; Cd=Cd_now, Cl=Cl_now,
                            rho_min=rho_min, rho_max=rho_max,
                            setup=setup))
        end
    end

    KernelAbstractions.synchronize(backend)
    elapsed = time() - t0
    Fx_avg = Fx_sum / max(n_avg, 1)
    Fy_avg = Fy_sum / max(n_avg, 1)
    Cd = 2.0 * Fx_avg / (Float64(u_ref)^2 * D_eff)
    Cl = 2.0 * Fy_avg / (Float64(u_ref)^2 * D_eff)
    total_nodes = sum(block.mesh.Nξ * block.mesh.Nη for block in mbm.blocks)

    return (;
        Cd=Cd, Cl=Cl, Fx=Fx_avg, Fy=Fy_avg,
        Cd_samples=cd_samples, Cl_samples=cl_samples, history=history,
        setup=setup, mesh=mbm, mesh_file=mesh_setup.file,
        blocks=n_blocks, nodes=total_nodes, solid_cells=solid_cells,
        dx_ref=Float64(dx_ref), D_eff=D_eff, Re=Float64(Re),
        u_max=Float64(u_max), u_ref=Float64(u_ref), nu=Float64(nu),
        steps=steps, avg_window=avg_window, sample_every=sample_every,
        rho_min=rho_min, rho_max=rho_max, elapsed_s=elapsed,
        MLUPs=total_nodes * steps / max(elapsed, eps()) / 1e6)
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
# Prescribed-velocity advection runner (Zalesak, reversed vortex, shear)
# ===========================================================================

"""
    _run_advection_only(setup; backend, T)

Run pure VOF advection with prescribed velocity from `Velocity { ux=... uy=... }`.
No LBM solve — used for interface transport validation tests.
"""
function _run_advection_only(setup::SimulationSetup;
                             backend=KernelAbstractions.CPU(), T=Float64)
    dom = setup.domain
    Nx, Ny = dom.Nx, dom.Ny
    dx = T(dom.Lx / Nx)
    Lx, Ly = T(dom.Lx), T(dom.Ly)

    setup.velocity_field === nothing &&
        throw(ArgumentError("Module advection_only requires a Velocity { ux=... uy=... } block"))
    setup.initial === nothing || !haskey(setup.initial.fields, :C) &&
        throw(ArgumentError("Module advection_only requires Initial { C = ... }"))

    # Build velocity function from KrakenExpr
    vf = setup.velocity_field
    ux_expr = get(vf.fields, :ux, nothing)
    uy_expr = get(vf.fields, :uy, nothing)

    function velocity_fn(x, y, t)
        kw = (; x=x, y=y, t=t, Lx=Float64(Lx), Ly=Float64(Ly),
                Nx=Float64(Nx), Ny=Float64(Ny), dx=Float64(dx))
        vx = ux_expr !== nothing ? evaluate(ux_expr; kw...) : 0.0
        vy = uy_expr !== nothing ? evaluate(uy_expr; kw...) : 0.0
        return (vx, vy)
    end

    # Build init function from Initial { C = ... }
    C_expr = setup.initial.fields[:C]
    function init_C_fn(x, y)
        kw = (; x=x, y=y, Lx=Float64(Lx), Ly=Float64(Ly),
                Nx=Float64(Nx), Ny=Float64(Ny), dx=Float64(dx))
        return evaluate(C_expr; kw...)
    end

    _adv_vtk = _find_output(setup, :vtk)
    output_interval = _adv_vtk !== nothing ? _adv_vtk.interval : 0
    output_dir = _adv_vtk !== nothing ? _adv_vtk.directory : ""

    adv_result = run_advection_2d(; Nx=Nx, Ny=Ny, max_steps=setup.max_steps,
                                   velocity_fn=velocity_fn,
                                   init_C_fn=init_C_fn,
                                   output_interval=output_interval,
                                   output_dir=output_dir,
                                   backend=backend, FT=T)

    # Wrap with setup and ρ for postprocess helper compatibility
    ρ_from_C = adv_result.C  # use C as a proxy for ρ in advection-only mode
    return merge(adv_result, (ρ=ρ_from_C, setup=setup))
end

# ===========================================================================
# Two-phase VOF runner (static droplet, capillary wave, etc.)
# ===========================================================================

"""
    _run_twophase_vof(setup; backend, T)

Run a two-phase LBM simulation with VOF interface tracking.
Generalizes `run_static_droplet_2d` for arbitrary configs.
"""
function _run_twophase_vof(setup::SimulationSetup;
                           backend=KernelAbstractions.CPU(), T=Float64)
    dom = setup.domain
    Nx, Ny = dom.Nx, dom.Ny
    dx = T(dom.Lx / Nx)
    dy = T(dom.Ly / Ny)
    Lx, Ly = T(dom.Lx), T(dom.Ly)

    params = setup.physics.params
    ν   = T(params[:nu])
    σ   = T(get(params, :sigma, 0.01))
    ρ_l = T(get(params, :rho_l, 1.0))
    ρ_g = T(get(params, :rho_g, 0.001))
    ν_l = T(get(params, :nu_l, ν))
    ν_g = T(get(params, :nu_g, ν))
    ω   = T(1.0 / (3.0 * ν + 0.5))

    # --- Initialize LBM ---
    config = LBMConfig(D2Q9(); Nx=Nx, Ny=Ny, ν=Float64(ν), u_lid=0.0,
                       max_steps=setup.max_steps, output_interval=1000)
    state = initialize_2d(config, T; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid

    # Apply geometry
    _apply_geometry!(is_solid, setup, Float64(dx), Float64(dy))

    # --- VOF arrays ---
    C     = KernelAbstractions.zeros(backend, T, Nx, Ny)
    C_new = KernelAbstractions.zeros(backend, T, Nx, Ny)
    nx_n  = KernelAbstractions.zeros(backend, T, Nx, Ny)
    ny_n  = KernelAbstractions.zeros(backend, T, Nx, Ny)
    κ     = KernelAbstractions.zeros(backend, T, Nx, Ny)
    Fx_st = KernelAbstractions.zeros(backend, T, Nx, Ny)
    Fy_st = KernelAbstractions.zeros(backend, T, Nx, Ny)

    # --- Initialize C from Initial { C = ... } ---
    if setup.initial !== nothing && haskey(setup.initial.fields, :C)
        C_expr = setup.initial.fields[:C]
        C_cpu = zeros(T, Nx, Ny)
        for j in 1:Ny, i in 1:Nx
            x = (i - T(0.5)) * dx
            y = (j - T(0.5)) * dy
            kw = (; x=Float64(x), y=Float64(y), Lx=Float64(Lx), Ly=Float64(Ly),
                    Nx=Float64(Nx), Ny=Float64(Ny), dx=Float64(dx))
            C_cpu[i, j] = clamp(T(evaluate(C_expr; kw...)), zero(T), one(T))
        end
        copyto!(C, C_cpu)

        # Initialize f to equilibrium with density from C
        w = weights(D2Q9())
        f_cpu = zeros(T, Nx, Ny, 9)
        for j in 1:Ny, i in 1:Nx
            ρ_init = C_cpu[i,j] * ρ_l + (one(T) - C_cpu[i,j]) * ρ_g
            for q in 1:9
                f_cpu[i, j, q] = T(w[q]) * ρ_init
            end
        end
        copyto!(f_in, f_cpu)
        copyto!(f_out, f_cpu)
    end

    # --- Non-Newtonian rheology for two-phase ---
    _has_vof_rheology = !isempty(setup.rheology)
    _rheo_l = nothing
    _rheo_g = nothing
    _tau_field_vof = nothing
    if _has_vof_rheology
        liq_setups = [r for r in setup.rheology if r.phase in (:liquid, :default)]
        gas_setups = [r for r in setup.rheology if r.phase == :gas]
        _rheo_l = isempty(liq_setups) ? Newtonian(T(ν_l)) : build_rheology_model(first(liq_setups); FT=T)
        _rheo_g = isempty(gas_setups) ? Newtonian(T(ν_g)) : build_rheology_model(first(gas_setups); FT=T)
        _tau_field_vof = KernelAbstractions.ones(backend, T, Nx, Ny)
    end

    # --- Select streaming kernel ---
    stream_fn! = _select_streaming_kernel(setup)

    # --- Setup output ---
    pvd = nothing
    output_dir = ""
    _vof_vtk = _find_output(setup, :vtk)
    if _vof_vtk !== nothing
        output_dir = setup_output_dir(_vof_vtk.directory)
        pvd = create_pvd(joinpath(output_dir, setup.name))
    end

    # --- Time loop ---
    for step in 1:setup.max_steps
        # 1. Stream
        stream_fn!(f_out, f_in, Nx, Ny)

        # 2. Macroscopic
        compute_macroscopic_2d!(ρ, ux, uy, f_out)

        # 3. VOF advection + clamp
        advect_vof_step!(C, C_new, ux, uy, Nx, Ny)
        copyto!(C, C_new)

        # 4. Interface normal + curvature + surface tension
        compute_vof_normal_2d!(nx_n, ny_n, C, Nx, Ny)
        compute_hf_curvature_2d!(κ, C, nx_n, ny_n, Nx, Ny)
        compute_surface_tension_2d!(Fx_st, Fy_st, κ, C, σ, Nx, Ny)

        # 5. Two-phase collision (non-Newtonian or Newtonian)
        if _has_vof_rheology
            collide_twophase_rheology_2d!(f_out, C, Fx_st, Fy_st, is_solid, _tau_field_vof;
                                          rheology_l=_rheo_l, rheology_g=_rheo_g,
                                          rho_l=Float64(ρ_l), rho_g=Float64(ρ_g))
        else
            collide_twophase_2d!(f_out, C, Fx_st, Fy_st, is_solid;
                                 ρ_l=Float64(ρ_l), ρ_g=Float64(ρ_g),
                                 ν_l=Float64(ν_l), ν_g=Float64(ν_g))
        end

        # 6. Swap
        f_in, f_out = f_out, f_in

        # 7. Output
        if _vof_vtk !== nothing && step % _vof_vtk.interval == 0
            _write_output(ρ, ux, uy, setup, _vof_vtk, pvd, output_dir, Float64(dx), step;
                          extra_fields=Dict("C" => Array(C), "kappa" => Array(κ)))
        end
    end

    # Finalize PVD
    if pvd !== nothing
        vtk_save(pvd)
    end

    # Compute diagnostics
    compute_macroscopic_2d!(ρ, ux, uy, f_in)
    ρ_cpu = Array(ρ)
    ux_cpu = Array(ux)
    uy_cpu = Array(uy)
    C_cpu = Array(C)
    max_u = sqrt(maximum(ux_cpu .^ 2 .+ uy_cpu .^ 2))

    return (ρ=ρ_cpu, ux=ux_cpu, uy=uy_cpu, C=C_cpu,
            max_u_spurious=max_u, setup=setup)
end


# ===========================================================================
# Dispatch helpers for 3D / axisymmetric / refined / thermal cases
# ===========================================================================

"""Dispatch D3Q19 cases to the appropriate 3D driver."""
function _run_d3q19(setup::SimulationSetup;
                    backend=KernelAbstractions.CPU(), T=Float64)
    name = lowercase(setup.name)
    dom  = setup.domain
    ν    = setup.physics.params[:nu]
    if occursin("cavity_3d", name) || occursin("cavity3d", name)
        # Look for a velocity BC on top/north for u_lid; default to 0.1.
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

"""Dispatch axisymmetric cases. Only Hagen-Poiseuille is supported in v0.1.0."""
function _run_axisymmetric(setup::SimulationSetup;
                           backend=KernelAbstractions.CPU(), T=Float64)
    name = lowercase(setup.name)
    dom  = setup.domain
    params = setup.physics.params
    ν  = Float64(params[:nu])
    Fz = haskey(setup.physics.body_force, :Fz) ?
         Float64(evaluate(setup.physics.body_force[:Fz])) : 1e-5
    if occursin("hagen_poiseuille", name) || occursin("hagen-poiseuille", name)
        result = run_hagen_poiseuille_2d(; Nz=dom.Nx, Nr=dom.Ny, ν=ν, Fz=Fz,
                                          max_steps=setup.max_steps,
                                          backend=backend, FT=T)
        return merge(result, (setup=setup,))
    else
        throw(ArgumentError(
            "axisymmetric dispatch: only `hagen_poiseuille` is supported in " *
            "v0.1.0 (got case name: $(setup.name))."))
    end
end

"""Dispatch grid-refined cases (2D and 3D).

Handles both D2Q9 and D3Q19 lattices with isothermal and thermal modules.
"""
function _run_refined(setup::SimulationSetup;
                      backend=KernelAbstractions.CPU(), T=Float64,
                      callback::Union{Nothing,Function}=nothing,
                      callback_every::Int=100)
    if setup.lattice === :D3Q19
        return _run_refined_3d(setup; backend=backend, T=T,
                               callback=callback, callback_every=callback_every)
    end

    is_thermal = :thermal in setup.modules

    # --- Generic isothermal refined path ---
    dom = setup.domain
    Nx, Ny = dom.Nx, dom.Ny
    dx = T(dom.Lx / Nx)
    dy = T(dom.Ly / Ny)
    ν = T(setup.physics.params[:nu])
    ω = T(1.0 / (3.0 * ν + 0.5))

    # Initialize base grid
    config = LBMConfig(D2Q9(); Nx=Nx, Ny=Ny, ν=Float64(ν), u_lid=0.0,
                       max_steps=setup.max_steps, output_interval=1000)
    state = initialize_2d(config, T; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid

    _apply_geometry!(is_solid, setup, Float64(dx), Float64(dy))

    if setup.initial !== nothing
        _apply_initial_conditions!(f_in, f_out, setup, Float64(dx), Float64(dy), T)
    end

    # Body force (Guo)
    has_body_force = !isempty(setup.physics.body_force)
    Fx_val = T(0); Fy_val = T(0)
    if has_body_force
        Fx_val = haskey(setup.physics.body_force, :Fx) ?
            T(evaluate(setup.physics.body_force[:Fx])) : T(0)
        Fy_val = haskey(setup.physics.body_force, :Fy) ?
            T(evaluate(setup.physics.body_force[:Fy])) : T(0)
    end

    # Create patches from Refine blocks
    patches = RefinementPatch{T}[]
    for rs in setup.refinements
        patch = create_patch(rs.name, 1, rs.ratio,
            (Float64(rs.region[1]), Float64(rs.region[2]),
             Float64(rs.region[3]), Float64(rs.region[4])),
            Nx, Ny, Float64(dx), Float64(ω), T; backend=backend)
        push!(patches, patch)
    end
    domain = create_refined_domain(Nx, Ny, Float64(dx), Float64(ω), patches)

    # Apply geometry on fine patches (re-evaluate at fine resolution)
    for patch in patches
        _apply_patch_geometry!(patch, setup)
    end

    # Initialize patch interiors from coarse state
    compute_macroscopic_2d!(ρ, ux, uy, f_in)
    for patch in patches
        prolongate_f_rescaled_full_2d!(
            patch.f_in, f_in, ρ, ux, uy,
            patch.ratio, patch.Nx_inner, patch.Ny_inner,
            patch.n_ghost, first(patch.parent_i_range), first(patch.parent_j_range),
            Nx, Ny, Float64(ω), Float64(patch.omega))
        copyto!(patch.f_out, patch.f_in)
        compute_macroscopic_2d!(patch.rho, patch.ux, patch.uy, patch.f_in)
    end

    # --- Thermal refined setup (if :thermal module) ---
    g_in = nothing; g_out = nothing; Temp = nothing
    thermals = nothing
    ω_T = T(0); β_g_val = T(0); T_ref_buoy = T(0)
    bc_thermal_patch_fns = nothing
    thermal_bc_face_fns = Function[]

    if is_thermal
        params = setup.physics.params
        Pr = T(get(params, :Pr, 0.71))
        α_thermal = haskey(params, :alpha) ? T(params[:alpha]) : ν / Pr
        ω_T = T(1.0 / (3.0 * Float64(α_thermal) + 0.5))

        # Detect temperature BCs from boundaries
        thermal_face_bcs = Dict{Symbol, T}()
        for bc in setup.boundaries
            if haskey(bc.values, :T)
                T_val = T(evaluate(bc.values[:T]))
                thermal_face_bcs[bc.face] = T_val
            end
        end
        T_hot = isempty(thermal_face_bcs) ? T(1) : maximum(values(thermal_face_bcs))
        T_cold = isempty(thermal_face_bcs) ? T(0) : minimum(values(thermal_face_bcs))
        ΔT = T_hot - T_cold
        if abs(ΔT) < eps(T)
            ΔT = T(1)
        end

        # Use pre-computed gbeta_DT from Setup helper when available;
        # gbeta_DT = β·g·ΔT already in lattice units (consistent with ν, α, L_ref).
        if haskey(params, :gbeta_DT)
            β_g_val = T(params[:gbeta_DT]) / ΔT
        else
            H = T(max(Nx, Ny))
            Ra = T(get(params, :Ra, 1e4))
            β_g_val = Ra * ν * α_thermal / (ΔT * H^3)
        end
        T_ref_buoy = (T_hot + T_cold) / T(2)

        # Allocate thermal arrays on base grid
        g_in  = KernelAbstractions.zeros(backend, T, Nx, Ny, 9)
        g_out = KernelAbstractions.zeros(backend, T, Nx, Ny, 9)
        Temp  = KernelAbstractions.zeros(backend, T, Nx, Ny)

        # Initialize g to linear temperature profile between hot/cold walls
        w_lat = weights(D2Q9())
        T_mid = (T_hot + T_cold) / T(2)
        g_cpu = zeros(T, Nx, Ny, 9)
        # Detect gradient direction from thermal BCs
        hot_face = :west
        for (face, tv) in thermal_face_bcs
            tv ≈ T_hot && (hot_face = face)
        end
        for j in 1:Ny, i in 1:Nx
            t_frac = if hot_face in (:west, :east)
                T(hot_face == :west ? (i - 1) / max(Nx - 1, 1) :
                                      (Nx - i) / max(Nx - 1, 1))
            else  # south/north
                T(hot_face == :south ? (j - 1) / max(Ny - 1, 1) :
                                       (Ny - j) / max(Ny - 1, 1))
            end
            T_init = T_hot - (T_hot - T_cold) * t_frac
            for q in 1:9
                g_cpu[i, j, q] = T(w_lat[q]) * T_init
            end
        end
        copyto!(g_in, g_cpu)
        copyto!(g_out, g_cpu)

        # Create thermal patch arrays
        thermals = ThermalPatchArrays{T}[
            create_thermal_patch_arrays(p, Float64(ω_T);
                T_init=Float64(T_mid), backend=backend) for p in patches]

        # Initialize patch thermal from coarse
        compute_temperature_2d!(Temp, g_in)
        for (pidx, patch) in enumerate(patches)
            fill_thermal_full!(patch, thermals[pidx], g_in, Nx, Ny)
        end

        # Build thermal patch BCs (fixed-T on faces touching domain walls)
        bc_thermal_patch_fns = _build_patch_thermal_bcs(patches, setup, T)

        # Build coarse thermal BC closures
        for (face, T_val) in thermal_face_bcs
            if face == :south
                push!(thermal_bc_face_fns, (g, nx, ny) -> apply_fixed_temp_south_2d!(g, T_val, nx))
            elseif face == :north
                push!(thermal_bc_face_fns, (g, nx, ny) -> apply_fixed_temp_north_2d!(g, T_val, nx, ny))
            elseif face == :west
                push!(thermal_bc_face_fns, (g, nx, ny) -> apply_fixed_temp_west_2d!(g, T_val, ny))
            elseif face == :east
                push!(thermal_bc_face_fns, (g, nx, ny) -> apply_fixed_temp_east_2d!(g, T_val, nx, ny))
            end
        end
    end

    # Auto-detect patch BCs
    bc_patch_fns = _build_patch_flow_bcs(patches, setup)

    # Build closures
    stream_fn! = _select_streaming_kernel(setup)
    bc_handlers = _build_boundary_handlers(setup, Float64(dx), Float64(dy), Nx, Ny, T, backend)

    collide_fn = if has_body_force
        (f, is_s) -> collide_guo_2d!(f, is_s, ω, Fx_val, Fy_val)
    else
        (f, is_s) -> collide_2d!(f, is_s, ω)
    end

    macro_fn = if has_body_force
        (r, u, v, f) -> compute_macroscopic_forced_2d!(r, u, v, f, Fx_val, Fy_val)
    else
        compute_macroscopic_2d!
    end

    bc_base_fn = (f) -> _apply_boundary_conditions!(f, bc_handlers, 0, Nx, Ny,
                                                     Float64(dx), Float64(dy), dom, T)

    # Patch collide with force scaling (F/ratio)
    patch_collide_fns = nothing
    patch_macro_fn = nothing
    if has_body_force
        patch_collide_fns = Dict{Int, Function}()
        for (pidx, patch) in enumerate(patches)
            Fx_f = Fx_val / T(patch.ratio)
            Fy_f = Fy_val / T(patch.ratio)
            ω_f = patch.omega
            patch_collide_fns[pidx] = (f, is_s) -> collide_guo_2d!(f, is_s, ω_f, Fx_f, Fy_f)
        end
        patch_macro_fn = (r, u, v, f) -> compute_macroscopic_forced_2d!(r, u, v, f,
                                                                         Fx_val, Fy_val)
    end

    # --- Thermal fused step and coarse BC closures ---
    thermal_fused_step_fn = nothing
    thermal_bc_coarse_fn = nothing
    if is_thermal
        # Collect wall faces for explicit bounce-back after restriction
        wall_faces = Symbol[bc.face for bc in setup.boundaries if bc.type == :wall]

        let sfn=stream_fn!, bch=bc_handlers, d_=dom, dx_=dx, dy_=dy,
            ρ_=ρ, ux_=ux, uy_=uy, is_s=is_solid, ω_=ω, ωT_=ω_T,
            βg_=β_g_val, Tref_=T_ref_buoy, tbcfns=thermal_bc_face_fns, T_=T,
            wf=wall_faces, Nx_=Nx, Ny_=Ny

            thermal_fused_step_fn = (f_o, f_i, g_o, g_i, Te, nx, ny) -> begin
                sfn(f_o, f_i, nx, ny)
                sfn(g_o, g_i, nx, ny)
                _apply_boundary_conditions!(f_o, bch, 0, nx, ny,
                    Float64(dx_), Float64(dy_), d_, T_)
                for bfn in tbcfns
                    bfn(g_o, nx, ny)
                end
                compute_temperature_2d!(Te, g_o)
                compute_macroscopic_2d!(ρ_, ux_, uy_, f_o)
                collide_thermal_2d!(g_o, ux_, uy_, ωT_)
                collide_boussinesq_2d!(f_o, Te, is_s, ω_, βg_, Tref_)
            end

            thermal_bc_coarse_fn = (f, g, Te, nx, ny) -> begin
                # Explicit bounce-back on wall faces (stream_2d! handles it during
                # the fused step, but restriction overwrites wall cells)
                for face in wf
                    apply_bounce_back_wall_2d!(f, Nx_, Ny_, face)
                end
                _apply_boundary_conditions!(f, bch, 0, nx, ny,
                    Float64(dx_), Float64(dy_), d_, T_)
                for bfn in tbcfns
                    bfn(g, nx, ny)
                end
            end
        end
    end

    # Output setup
    vtk_out = _find_output(setup, :vtk)
    pvd = nothing
    output_dir = ""
    if vtk_out !== nothing
        output_dir = setup_output_dir(vtk_out.directory)
        pvd = create_pvd(joinpath(output_dir, setup.name))
    end
    png_out = _find_output(setup, :png)
    gif_out = _find_output(setup, :gif)
    gif_frames = _init_gif_frames(gif_out)
    _check_image_backend(png_out, gif_out)

    # --- Time loop ---
    for step in 1:setup.max_steps
        if is_thermal
            f_in, f_out, g_in, g_out = advance_thermal_refined_step!(
                domain, thermals,
                f_in, f_out, g_in, g_out, ρ, ux, uy, Temp, is_solid;
                fused_step_fn=thermal_fused_step_fn,
                omega_T_coarse=Float64(ω_T),
                β_g=Float64(β_g_val),
                T_ref_buoy=Float64(T_ref_buoy),
                bc_thermal_patch_fns=bc_thermal_patch_fns,
                bc_flow_patch_fns=bc_patch_fns,
                bc_coarse_fn=thermal_bc_coarse_fn)
        else
            f_in, f_out = advance_refined_step!(domain, f_in, f_out, ρ, ux, uy, is_solid;
                stream_fn=stream_fn!, collide_fn=collide_fn, macro_fn=macro_fn,
                bc_base_fn=bc_base_fn, bc_patch_fns=bc_patch_fns,
                patch_collide_fns=patch_collide_fns, patch_macro_fn=patch_macro_fn)
        end

        # Re-apply coarse BCs after restriction (patches may overwrite wall cells)
        _apply_boundary_conditions!(f_in, bc_handlers, step, Nx, Ny,
                                    Float64(dx), Float64(dy), dom, T)
        compute_macroscopic_2d!(ρ, ux, uy, f_in)
        if is_thermal
            for bfn in thermal_bc_face_fns
                bfn(g_in, Nx, Ny)
            end
            compute_temperature_2d!(Temp, g_in)
        end

        # VTK output
        if vtk_out !== nothing && step % vtk_out.interval == 0
            _write_output(ρ, ux, uy, setup, vtk_out, pvd, output_dir, Float64(dx), step)
        end

        # PNG/GIF
        _maybe_save_png(png_out, ρ, ux, uy, setup, output_dir, step)
        _maybe_collect_gif(gif_out, gif_frames, ρ, ux, uy, step)

        # Callback
        if callback !== nothing && step % callback_every == 0
            cb_state = is_thermal ?
                (; rho=Array(ρ), ux=Array(ux), uy=Array(uy), Temp=Array(Temp)) :
                (; rho=Array(ρ), ux=Array(ux), uy=Array(uy))
            callback(step, cb_state)
        end
    end

    # Finalize outputs
    pvd !== nothing && vtk_save(pvd)
    _maybe_save_gif(gif_out, gif_frames, setup, output_dir)

    if is_thermal
        # Compute Nusselt from the finest patch touching a heated wall
        Nu = _compute_nusselt_from_patches(domain, thermals, Temp, setup, T)
        return (ρ=Array(ρ), ux=Array(ux), uy=Array(uy), Temp=Array(Temp),
                Nu=Nu, setup=setup)
    end
    return (ρ=Array(ρ), ux=Array(ux), uy=Array(uy), setup=setup)
end

# =========================================================================
# 3D refined runner (D3Q19 + Refine blocks)
# =========================================================================

"""Dispatch grid-refined cases for D3Q19 (3D)."""
function _run_refined_3d(setup::SimulationSetup;
                          backend=KernelAbstractions.CPU(), T=Float64,
                          callback::Union{Nothing,Function}=nothing,
                          callback_every::Int=100)
    is_thermal = :thermal in setup.modules
    dom = setup.domain
    Nx, Ny, Nz = dom.Nx, dom.Ny, dom.Nz
    Lx, Ly, Lz = dom.Lx, dom.Ly, dom.Lz
    dx = T(Lx / Nx)
    ν = T(setup.physics.params[:nu])
    ω = T(1.0 / (3.0 * ν + 0.5))

    # Initialize 3D base grid
    config = LBMConfig(D3Q19(); Nx=Nx, Ny=Ny, Nz=Nz, ν=Float64(ν), u_lid=0.0,
                       max_steps=setup.max_steps, output_interval=1000)
    state = initialize_3d(config, T; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    ρ = state.ρ; ux = state.ux; uy = state.uy; uz = state.uz
    is_solid = state.is_solid

    _apply_geometry_3d!(is_solid, setup, Float64(dx))

    if setup.initial !== nothing
        _apply_initial_conditions_3d!(f_in, f_out, setup, Float64(dx), T)
    end

    # Create 3D patches from Refine blocks
    patches = RefinementPatch3D{T}[]
    for rs in setup.refinements
        region_6 = rs.is_3d ? rs.region_3d :
            (Float64(rs.region[1]), Float64(rs.region[2]), 0.0,
             Float64(rs.region[3]), Float64(rs.region[4]), Float64(Lz))
        patch = create_patch_3d(rs.name, 1, rs.ratio, region_6,
            Nx, Ny, Nz, Float64(dx), Float64(ω), T; backend=backend)
        push!(patches, patch)
    end
    domain = create_refined_domain_3d(Nx, Ny, Nz, Float64(dx), Float64(ω), patches)

    # Apply geometry on fine patches
    for patch in patches
        _apply_patch_geometry_3d!(patch, setup)
    end

    # Initialize patch interiors from coarse state
    compute_macroscopic_3d!(ρ, ux, uy, uz, f_in)
    for patch in patches
        prolongate_f_rescaled_full_3d!(
            patch.f_in, f_in, ρ, ux, uy, uz,
            patch.ratio, patch.Nx_inner, patch.Ny_inner, patch.Nz_inner,
            patch.n_ghost,
            first(patch.parent_i_range), first(patch.parent_j_range), first(patch.parent_k_range),
            Nx, Ny, Nz, Float64(ω), Float64(patch.omega))
        copyto!(patch.f_out, patch.f_in)
        compute_macroscopic_3d!(patch.rho, patch.ux, patch.uy, patch.uz, patch.f_in)
    end

    # Build wall face list for BCs
    wall_faces = Symbol[bc.face for bc in setup.boundaries if bc.type == :wall]

    # Build patch flow BCs (bounce-back on faces touching domain walls)
    bc_patch_fns = build_patch_flow_bcs_3d(patches, Lx, Ly, Lz, Nx; wall_faces=wall_faces)

    # Build coarse boundary closure
    bc_base_fn = (f) -> begin
        for face in wall_faces
            apply_bounce_back_wall_3d!(f, Nx, Ny, Nz, face)
        end
    end

    # Collide closure
    collide_fn = (f, is_s) -> collide_3d!(f, is_s, ω)
    macro_fn = compute_macroscopic_3d!

    # --- Thermal setup ---
    g_in = nothing; g_out = nothing; Temp = nothing
    thermals = nothing
    ω_T = T(0); β_g_val = T(0); T_ref_buoy = T(0)
    bc_thermal_patch_fns = nothing
    thermal_bc_face_fns = Function[]

    if is_thermal
        params = setup.physics.params
        Pr = T(get(params, :Pr, 0.71))
        α_thermal = haskey(params, :alpha) ? T(params[:alpha]) : ν / Pr
        ω_T = T(1.0 / (3.0 * Float64(α_thermal) + 0.5))

        # Detect temperature BCs
        thermal_face_bcs = Dict{Symbol, T}()
        for bc in setup.boundaries
            if haskey(bc.values, :T)
                T_val = T(evaluate(bc.values[:T]))
                thermal_face_bcs[bc.face] = T_val
            end
        end
        T_hot = isempty(thermal_face_bcs) ? T(1) : maximum(values(thermal_face_bcs))
        T_cold = isempty(thermal_face_bcs) ? T(0) : minimum(values(thermal_face_bcs))
        ΔT = T_hot - T_cold
        abs(ΔT) < eps(T) && (ΔT = T(1))

        if haskey(params, :gbeta_DT)
            β_g_val = T(params[:gbeta_DT]) / ΔT
        else
            H = T(max(Nx, Ny, Nz))
            Ra = T(get(params, :Ra, 1e4))
            β_g_val = Ra * ν * α_thermal / (ΔT * H^3)
        end
        T_ref_buoy = (T_hot + T_cold) / T(2)

        # Allocate thermal arrays
        g_in  = KernelAbstractions.zeros(backend, T, Nx, Ny, Nz, 19)
        g_out = KernelAbstractions.zeros(backend, T, Nx, Ny, Nz, 19)
        Temp  = KernelAbstractions.zeros(backend, T, Nx, Ny, Nz)

        # Initialize g to linear temperature profile
        w_lat = weights(D3Q19())
        T_mid = (T_hot + T_cold) / T(2)
        g_cpu = zeros(T, Nx, Ny, Nz, 19)
        hot_face = :west
        for (face, tv) in thermal_face_bcs
            tv ≈ T_hot && (hot_face = face)
        end
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            t_frac = if hot_face in (:west, :east)
                T(hot_face == :west ? (i - 1) / max(Nx - 1, 1) : (Nx - i) / max(Nx - 1, 1))
            elseif hot_face in (:south, :north)
                T(hot_face == :south ? (j - 1) / max(Ny - 1, 1) : (Ny - j) / max(Ny - 1, 1))
            else  # bottom/top
                T(hot_face == :bottom ? (k - 1) / max(Nz - 1, 1) : (Nz - k) / max(Nz - 1, 1))
            end
            T_init = T_hot - (T_hot - T_cold) * t_frac
            for q in 1:19
                g_cpu[i, j, k, q] = T(w_lat[q]) * T_init
            end
        end
        copyto!(g_in, g_cpu)
        copyto!(g_out, g_cpu)

        # Create thermal patch arrays
        thermals = ThermalPatchArrays3D{T}[
            create_thermal_patch_arrays_3d(p, Float64(ω_T);
                T_init=Float64(T_mid), backend=backend) for p in patches]

        # Initialize patch thermal from coarse
        compute_temperature_3d!(Temp, g_in)
        for (pidx, patch) in enumerate(patches)
            fill_thermal_full_3d!(patch, thermals[pidx], g_in, Nx, Ny, Nz)
        end

        # Build thermal patch BCs
        bc_thermal_patch_fns = build_patch_thermal_bcs_3d(
            patches, Lx, Ly, Lz, Nx, thermal_face_bcs)

        # Build coarse thermal BC closures
        for (face, T_val) in thermal_face_bcs
            if face == :south
                push!(thermal_bc_face_fns, (g, nx, ny, nz) -> apply_fixed_temp_south_3d!(g, T_val, nx, nz))
            elseif face == :north
                push!(thermal_bc_face_fns, (g, nx, ny, nz) -> apply_fixed_temp_north_3d!(g, T_val, nx, ny, nz))
            elseif face == :west
                push!(thermal_bc_face_fns, (g, nx, ny, nz) -> apply_fixed_temp_west_3d!(g, T_val, ny, nz))
            elseif face == :east
                push!(thermal_bc_face_fns, (g, nx, ny, nz) -> apply_fixed_temp_east_3d!(g, T_val, nx, ny, nz))
            elseif face == :bottom
                push!(thermal_bc_face_fns, (g, nx, ny, nz) -> apply_fixed_temp_bottom_3d!(g, T_val, nx, ny))
            elseif face == :top
                push!(thermal_bc_face_fns, (g, nx, ny, nz) -> apply_fixed_temp_top_3d!(g, T_val, nx, ny, nz))
            end
        end
    end

    # --- Build fused thermal step and coarse BC closures ---
    thermal_fused_step_fn = nothing
    thermal_bc_coarse_fn = nothing
    if is_thermal
        let wf=wall_faces, bc_base=bc_base_fn, ρ_=ρ, ux_=ux, uy_=uy, uz_=uz,
            is_s=is_solid, ω_=ω, ωT_=ω_T, βg_=β_g_val, Tref_=T_ref_buoy,
            tbcfns=thermal_bc_face_fns, Nx_=Nx, Ny_=Ny, Nz_=Nz

            thermal_fused_step_fn = (f_o, f_i, g_o, g_i, Te, nx, ny, nz) -> begin
                stream_3d!(f_o, f_i, nx, ny, nz)
                stream_3d!(g_o, g_i, nx, ny, nz)
                bc_base(f_o)
                for bfn in tbcfns
                    bfn(g_o, nx, ny, nz)
                end
                compute_temperature_3d!(Te, g_o)
                compute_macroscopic_3d!(ρ_, ux_, uy_, uz_, f_o)
                collide_thermal_3d!(g_o, ux_, uy_, uz_, ωT_)
                collide_boussinesq_3d!(f_o, Te, is_s, ω_, βg_, Tref_)
            end

            thermal_bc_coarse_fn = (f, g, Te, nx, ny, nz) -> begin
                for face in wf
                    apply_bounce_back_wall_3d!(f, Nx_, Ny_, Nz_, face)
                end
                for bfn in tbcfns
                    bfn(g, nx, ny, nz)
                end
            end
        end
    end

    # Output setup
    vtk_out = _find_output(setup, :vtk)
    pvd = nothing; output_dir = ""
    if vtk_out !== nothing
        output_dir = setup_output_dir(vtk_out.directory)
        pvd = create_pvd(joinpath(output_dir, setup.name))
    end

    # --- Time loop ---
    for step in 1:setup.max_steps
        if is_thermal
            f_in, f_out, g_in, g_out = advance_thermal_refined_step_3d!(
                domain, thermals,
                f_in, f_out, g_in, g_out, ρ, ux, uy, uz, Temp, is_solid;
                fused_step_fn=thermal_fused_step_fn,
                omega_T_coarse=Float64(ω_T),
                β_g=Float64(β_g_val),
                T_ref_buoy=Float64(T_ref_buoy),
                bc_thermal_patch_fns=bc_thermal_patch_fns,
                bc_flow_patch_fns=bc_patch_fns,
                bc_coarse_fn=thermal_bc_coarse_fn)
        else
            f_in, f_out = advance_refined_step_3d!(
                domain, f_in, f_out, ρ, ux, uy, uz, is_solid;
                stream_fn=stream_3d!, collide_fn=collide_fn, macro_fn=macro_fn,
                bc_base_fn=bc_base_fn, bc_patch_fns=bc_patch_fns)
        end

        # Re-apply coarse BCs after restriction
        bc_base_fn(f_in)
        compute_macroscopic_3d!(ρ, ux, uy, uz, f_in)
        if is_thermal
            for bfn in thermal_bc_face_fns
                bfn(g_in, Nx, Ny, Nz)
            end
            compute_temperature_3d!(Temp, g_in)
        end

        # VTK output
        if vtk_out !== nothing && step % vtk_out.interval == 0
            _write_output_3d(ρ, ux, uy, uz, setup, vtk_out, pvd, output_dir, Float64(dx), step)
        end

        # Callback
        if callback !== nothing && step % callback_every == 0
            cb_state = is_thermal ?
                (; rho=Array(ρ), ux=Array(ux), uy=Array(uy), uz=Array(uz), Temp=Array(Temp)) :
                (; rho=Array(ρ), ux=Array(ux), uy=Array(uy), uz=Array(uz))
            callback(step, cb_state)
        end
    end

    # Finalize outputs
    pvd !== nothing && vtk_save(pvd)

    if is_thermal
        return (ρ=Array(ρ), ux=Array(ux), uy=Array(uy), uz=Array(uz),
                Temp=Array(Temp), setup=setup)
    end
    return (ρ=Array(ρ), ux=Array(ux), uy=Array(uy), uz=Array(uz), setup=setup)
end

# --- 3D geometry helpers ---

"""Evaluate obstacle geometry on 3D grid."""
function _apply_geometry_3d!(is_solid, setup::SimulationSetup, dx::Float64)
    Nx, Ny, Nz = setup.domain.Nx, setup.domain.Ny, setup.domain.Nz
    Lx, Ly, Lz = setup.domain.Lx, setup.domain.Ly, setup.domain.Lz

    isempty(setup.regions) && return

    has_fluid_region = any(r -> r.kind == :fluid, setup.regions)
    solid_cpu = has_fluid_region ? ones(Bool, Nx, Ny, Nz) : zeros(Bool, Nx, Ny, Nz)

    for region in setup.regions
        region.stl !== nothing && continue
        region.condition === nothing && continue
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            x = (i - 0.5) * dx
            y = (j - 0.5) * dx
            z = (k - 0.5) * dx
            result = evaluate(region.condition; x=x, y=y, z=z,
                             Lx=Lx, Ly=Ly, Lz=Lz, dx=dx, dy=dx, dz=dx)
            if region.kind == :fluid && result
                solid_cpu[i, j, k] = false
            elseif region.kind == :obstacle && result
                solid_cpu[i, j, k] = true
            end
        end
    end
    copyto!(is_solid, solid_cpu)
end

"""Apply initial conditions to 3D grid."""
function _apply_initial_conditions_3d!(f_in, f_out, setup::SimulationSetup,
                                        dx::Float64, ::Type{T}) where T
    setup.initial === nothing && return
    # For now, 3D initial conditions default to equilibrium (u=0, ρ=1)
    # Custom IC functions can be added later
end

"""Evaluate obstacle geometry on a 3D fine-grid patch."""
function _apply_patch_geometry_3d!(patch::RefinementPatch3D{T},
                                    setup::SimulationSetup) where T
    Nx_p, Ny_p, Nz_p = patch.Nx, patch.Ny, patch.Nz
    ng = patch.n_ghost
    dx_f = Float64(patch.dx)
    Lx, Ly, Lz = setup.domain.Lx, setup.domain.Ly, setup.domain.Lz

    isempty(setup.regions) && return

    has_fluid_region = any(r -> r.kind == :fluid, setup.regions)
    solid_cpu = has_fluid_region ? ones(Bool, Nx_p, Ny_p, Nz_p) : zeros(Bool, Nx_p, Ny_p, Nz_p)

    for region in setup.regions
        region.stl !== nothing && continue
        region.condition === nothing && continue
        for kf in 1:Nz_p, jf in 1:Ny_p, if_ in 1:Nx_p
            x = Float64(patch.x_min) + (if_ - ng - 0.5) * dx_f
            y = Float64(patch.y_min) + (jf - ng - 0.5) * dx_f
            z = Float64(patch.z_min) + (kf - ng - 0.5) * dx_f
            result = evaluate(region.condition; x=x, y=y, z=z,
                             Lx=Lx, Ly=Ly, Lz=Lz, dx=dx_f, dy=dx_f, dz=dx_f)
            if region.kind == :fluid && result
                solid_cpu[if_, jf, kf] = false
            elseif region.kind == :obstacle && result
                solid_cpu[if_, jf, kf] = true
            end
        end
    end
    copyto!(patch.is_solid, solid_cpu)
end

"""Write 3D VTK output using the existing VTK writer infrastructure."""
function _write_output_3d(ρ, ux, uy, uz, setup, vtk_out, pvd, output_dir, dx, step)
    Nx, Ny, Nz = size(ρ)
    fname = joinpath(output_dir, "$(setup.name)_$(lpad(step, 8, '0'))")
    fields = Dict{String, Array{Float64, 3}}(
        "rho" => Array(ρ),
        "ux"  => Array(ux),
        "uy"  => Array(uy),
        "uz"  => Array(uz),
    )
    write_vtk_to_pvd(pvd, fname, Nx, Ny, Nz, dx, fields, Float64(step))
end

# --- Refinement helpers ---

"""Evaluate obstacle geometry on a fine-grid patch at its native resolution."""
function _apply_patch_geometry!(patch::RefinementPatch{T},
                                setup::SimulationSetup) where T
    Nx_p, Ny_p = patch.Nx, patch.Ny
    ng = patch.n_ghost
    dx_f = Float64(patch.dx)
    Lx, Ly = setup.domain.Lx, setup.domain.Ly

    isempty(setup.regions) && return

    has_fluid_region = any(r -> r.kind == :fluid, setup.regions)
    solid_cpu = has_fluid_region ? ones(Bool, Nx_p, Ny_p) : zeros(Bool, Nx_p, Ny_p)

    for region in setup.regions
        region.stl !== nothing && continue  # TODO: STL voxelization on patches
        region.condition === nothing && continue
        for jf in 1:Ny_p, if_ in 1:Nx_p
            x = Float64(patch.x_min) + (if_ - ng - 0.5) * dx_f
            y = Float64(patch.y_min) + (jf - ng - 0.5) * dx_f
            result = evaluate(region.condition; x=x, y=y, z=0.0,
                             Lx=Lx, Ly=Ly, dx=dx_f, dy=dx_f)
            if region.kind == :fluid && result
                solid_cpu[if_, jf] = false
            elseif region.kind == :obstacle && result
                solid_cpu[if_, jf] = true
            end
        end
    end
    copyto!(patch.is_solid, solid_cpu)
end

"""Auto-detect which patch faces touch domain walls and build BC closures."""
function _build_patch_flow_bcs(patches::Vector{RefinementPatch{T}},
                                setup::SimulationSetup) where T
    Lx, Ly = setup.domain.Lx, setup.domain.Ly
    tol = Lx / setup.domain.Nx * 0.01

    bc_map = Dict{Symbol, BoundarySetup}()
    for bc in setup.boundaries
        bc.type == :periodic && continue
        bc_map[bc.face] = bc
    end

    patch_bcs = Dict{Int, Function}()
    for (pidx, patch) in enumerate(patches)
        closures = Function[]

        for (face, condition) in [
            (:west,  Float64(patch.x_min) <= tol),
            (:east,  Float64(patch.x_max) >= Lx - tol),
            (:south, Float64(patch.y_min) <= tol),
            (:north, Float64(patch.y_max) >= Ly - tol),
        ]
            condition || continue
            bc = get(bc_map, face, nothing)
            bc === nothing && continue
            cl = _make_patch_face_bc(face, bc, T)
            cl !== nothing && push!(closures, cl)
        end

        if !isempty(closures)
            # Capture closures in a local let-block to avoid closure issues
            let cls = closures
                patch_bcs[pidx] = (f, Nx_p, Ny_p) -> begin
                    for cl in cls
                        cl(f, Nx_p, Ny_p)
                    end
                end
            end
        end
    end
    return patch_bcs
end

"""Create a BC closure for one face of a patch.

Only wall (bounce-back) BCs are applied directly on patches. Velocity and
pressure BCs are handled implicitly through the ghost-fill prolongation from
the coarse grid, which already has these BCs applied. Applying Zou-He or
pressure BCs directly on the patch would target ghost rows (outside the
physical domain), producing incorrect results.
"""
function _make_patch_face_bc(face::Symbol, bc::BoundarySetup, ::Type{T}) where T
    if bc.type == :wall
        return (f, Nx_p, Ny_p) -> apply_bounce_back_wall_2d!(f, Nx_p, Ny_p, face)
    else
        # velocity / pressure / periodic: handled by ghost fill from coarse grid
        return nothing
    end
end

"""Auto-detect which patch faces touch domain walls with thermal BCs."""
function _build_patch_thermal_bcs(patches::Vector{RefinementPatch{T}},
                                   setup::SimulationSetup, ::Type{T}) where T
    Lx, Ly = setup.domain.Lx, setup.domain.Ly
    tol = Lx / setup.domain.Nx * 0.01

    # Collect faces with fixed temperature
    thermal_face_bcs = Dict{Symbol, T}()
    for bc in setup.boundaries
        if haskey(bc.values, :T)
            thermal_face_bcs[bc.face] = T(evaluate(bc.values[:T]))
        end
    end

    patch_bcs = Dict{Int, Function}()
    for (pidx, patch) in enumerate(patches)
        closures = Function[]

        for (face, condition) in [
            (:west,  Float64(patch.x_min) <= tol),
            (:east,  Float64(patch.x_max) >= Lx - tol),
            (:south, Float64(patch.y_min) <= tol),
            (:north, Float64(patch.y_max) >= Ly - tol),
        ]
            condition || continue
            T_val = get(thermal_face_bcs, face, nothing)
            T_val === nothing && continue
            cl = _make_patch_thermal_bc(face, T_val)
            push!(closures, cl)
        end

        if !isempty(closures)
            let cls = closures
                patch_bcs[pidx] = (g, Nx_p, Ny_p) -> begin
                    for cl in cls
                        cl(g, Nx_p, Ny_p)
                    end
                end
            end
        end
    end
    return patch_bcs
end

"""Create a thermal BC closure for one face of a patch (anti-bounce-back Dirichlet)."""
function _make_patch_thermal_bc(face::Symbol, T_val)
    if face == :south
        return (g, Nx_p, Ny_p) -> apply_fixed_temp_south_2d!(g, T_val, Nx_p)
    elseif face == :north
        return (g, Nx_p, Ny_p) -> apply_fixed_temp_north_2d!(g, T_val, Nx_p, Ny_p)
    elseif face == :west
        return (g, Nx_p, Ny_p) -> apply_fixed_temp_west_2d!(g, T_val, Ny_p)
    elseif face == :east
        return (g, Nx_p, Ny_p) -> apply_fixed_temp_east_2d!(g, T_val, Nx_p, Ny_p)
    end
end

"""Compute Nusselt number from the finest patch touching a heated wall."""
function _compute_nusselt_from_patches(domain::RefinedDomain{T},
                                        thermals::Vector{ThermalPatchArrays{T}},
                                        Temp_coarse, setup, ::Type{T}) where T
    # Find temperature extremes from boundary BCs
    T_hot = T(-Inf); T_cold = T(Inf)
    hot_face = :west
    for bc in setup.boundaries
        if haskey(bc.values, :T)
            T_val = T(evaluate(bc.values[:T]))
            if T_val > T_hot
                T_hot = T_val
                hot_face = bc.face
            end
            if T_val < T_cold
                T_cold = T_val
            end
        end
    end
    ΔT = T_hot - T_cold
    if abs(ΔT) < eps(T) || isinf(T_hot)
        return T(NaN)
    end

    # Use physical domain size for consistent units with patch.dx
    H = T(max(setup.domain.Lx, setup.domain.Ly))

    # Find the patch that touches the hot wall (for fine-grid gradient)
    Lx, Ly = setup.domain.Lx, setup.domain.Ly
    tol = Lx / setup.domain.Nx * 0.01

    for (pidx, patch) in enumerate(domain.patches)
        touches_hot = (hot_face == :west  && Float64(patch.x_min) <= tol) ||
                      (hot_face == :east  && Float64(patch.x_max) >= Lx - tol) ||
                      (hot_face == :south && Float64(patch.y_min) <= tol) ||
                      (hot_face == :north && Float64(patch.y_max) >= Ly - tol)
        touches_hot || continue

        T_fine = Array(thermals[pidx].Temp)
        ng = patch.n_ghost
        dx_f = T(patch.dx)

        # Compute wall-normal gradient via 2nd-order one-sided FD
        if hot_face == :west
            Ny_fine = patch.Ny_inner
            Nu_arr = zeros(T, Ny_fine)
            for jf in 1:Ny_fine
                j = jf + ng
                i1, i2, i3 = ng + 1, ng + 2, ng + 3
                dTdn = (-3 * T_fine[i1, j] + 4 * T_fine[i2, j] - T_fine[i3, j]) / (2 * dx_f)
                Nu_arr[jf] = -H * dTdn / ΔT
            end
            return sum(Nu_arr[2:end-1]) / max(Ny_fine - 2, 1)
        elseif hot_face == :east
            Ny_fine = patch.Ny_inner
            Nu_arr = zeros(T, Ny_fine)
            Nx_p = patch.Nx
            for jf in 1:Ny_fine
                j = jf + ng
                i1, i2, i3 = Nx_p - ng, Nx_p - ng - 1, Nx_p - ng - 2
                dTdn = (3 * T_fine[i1, j] - 4 * T_fine[i2, j] + T_fine[i3, j]) / (2 * dx_f)
                Nu_arr[jf] = H * dTdn / ΔT
            end
            return sum(Nu_arr[2:end-1]) / max(Ny_fine - 2, 1)
        elseif hot_face == :south
            Nx_fine = patch.Nx_inner
            Nu_arr = zeros(T, Nx_fine)
            for if_ in 1:Nx_fine
                i = if_ + ng
                j1, j2, j3 = ng + 1, ng + 2, ng + 3
                dTdn = (-3 * T_fine[i, j1] + 4 * T_fine[i, j2] - T_fine[i, j3]) / (2 * dx_f)
                Nu_arr[if_] = -H * dTdn / ΔT
            end
            return sum(Nu_arr[2:end-1]) / max(Nx_fine - 2, 1)
        elseif hot_face == :north
            Nx_fine = patch.Nx_inner
            Nu_arr = zeros(T, Nx_fine)
            Ny_p = patch.Ny
            for if_ in 1:Nx_fine
                i = if_ + ng
                j1, j2, j3 = Ny_p - ng, Ny_p - ng - 1, Ny_p - ng - 2
                dTdn = (3 * T_fine[i, j1] - 4 * T_fine[i, j2] + T_fine[i, j3]) / (2 * dx_f)
                Nu_arr[if_] = H * dTdn / ΔT
            end
            return sum(Nu_arr[2:end-1]) / max(Nx_fine - 2, 1)
        end
    end

    # No patch on the hot wall: fall back to coarse-grid Nusselt
    T_cpu = Array(Temp_coarse)
    dx_c = T(setup.domain.Lx / setup.domain.Nx)
    if hot_face == :west
        Ny_c = setup.domain.Ny
        Nu_arr = zeros(T, Ny_c)
        for j in 1:Ny_c
            dTdn = (-3*T_cpu[1,j] + 4*T_cpu[2,j] - T_cpu[3,j]) / (2*dx_c)
            Nu_arr[j] = -H * dTdn / ΔT
        end
        return sum(Nu_arr[2:end-1]) / max(Ny_c - 2, 1)
    end
    return T(NaN)
end

"""Dispatch thermal cases (non-refined) to the appropriate thermal driver."""
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
        # No dedicated conduction driver: run Rayleigh-Bénard with Ra≈0
        # so buoyancy is negligible and diffusion dominates. Documented
        # as a pragmatic fallback — the resulting temperature field
        # matches a 1D diffusive profile once steady state is reached.
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
