
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
