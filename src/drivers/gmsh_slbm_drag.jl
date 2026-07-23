
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
    cylinder_ghost_alpha = clamp(_setup_number(setup,
        (:bodyfit_cylinder_ghost_alpha, :cylinder_ghost_alpha), 0.10), 0.0, 1.0)

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
    cylinder_ghost_masks = [_mesh_drag_cylinder_ghost_masks(block, backend,
                                                            cx, cy)
                            for block in mbm.blocks]

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
        _apply_mesh_drag_cylinder_radial_ghost_bcs_2d!(mbm, states,
                                                       cylinder_ghost_masks,
                                                       cylinder_ghost_alpha)
        _apply_mesh_drag_physical_wall_ghost_bcs_2d!(mbm, states)
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
            _apply_mesh_drag_physical_wall_bcs_2d!(int_out, int_in,
                                                   mbm.blocks[k].boundary_tags,
                                                   nxp, nyp)
            _apply_mesh_drag_physical_normal_bcs_2d!(int_out, physical_bcspecs[k],
                                                     nxp, nyp)
        end

        for k in 1:n_blocks
            states[k].f, f_out[k] = f_out[k], states[k].f
        end

        if step > steps - avg_window && (step % sample_every == 0 || step == steps)
            drag = _compute_bodyfit_cylinder_force_2d(mbm, states, cx, cy,
                                                      radius, nu, dx_ref, ng)
            Fx = Float64(drag.Fx)
            Fy = Float64(drag.Fy)
            Cd = 2.0 * Fx / (Float64(u_ref)^2 * D_eff)
            Cl = 2.0 * Fy / (Float64(u_ref)^2 * D_eff)
            push!(cd_samples, Cd)
            push!(cl_samples, Cl)
            push!(history, (; step=step, Cd=Cd, Cl=Cl, Fx=Fx, Fy=Fy,
                            Fx_pressure=Float64(drag.Fx_pressure),
                            Fy_pressure=Float64(drag.Fy_pressure),
                            Fx_viscous=Float64(drag.Fx_viscous),
                            Fy_viscous=Float64(drag.Fy_viscous)))
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
                            setup=setup, mbm=mbm, states=states,
                            ng=ng, cx=cx, cy=cy, radius=radius,
                            dx_ref=dx_ref, D_eff=D_eff, u_ref=u_ref,
                            nu=nu))
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
