function _logfv_square_obstacle_mask(Nx::Int, Ny::Int, side::Int, cx::Int, cy::Int)
    side >= 2 || throw(ArgumentError("side must be >= 2"))
    3 <= cx <= Nx - 2 || throw(ArgumentError("cx must leave fluid columns around the square"))
    3 <= cy <= Ny - 2 || throw(ArgumentError("cy must leave fluid rows around the square"))
    half_lo = (side - 1) ÷ 2
    half_hi = side ÷ 2
    i1 = cx - half_lo
    i2 = cx + half_hi
    j1 = cy - half_lo
    j2 = cy + half_hi
    2 <= i1 <= i2 <= Nx - 1 || throw(ArgumentError("square obstacle must leave one fluid column on each side"))
    2 <= j1 <= j2 <= Ny - 1 || throw(ArgumentError("square obstacle must leave one fluid row on each side"))
    is_solid = fill(false, Nx, Ny)
    is_solid[i1:i2, j1:j2] .= true
    return is_solid
end

"""
    run_viscoelastic_logfv_square_periodic_2d(; kwargs...)

Run the first coarse macro-flow canary for the log-FV backend around an
axis-aligned square obstacle. The domain is periodic in `x`, has halfway walls
at `y`, and is driven by a uniform body force.

This is a stability/coupling canary, not a drag benchmark. It exercises
solid-aware velocity gradients, solid-aware `Psi` advection, local
log-conformation source update, solid-aware polymer force, and Guo coupling.
BSD uses the same solid-aware compact stencil as the polymer force path, so
low-beta checks exercise the operator used by the coarse obstacle flow.
"""
function run_viscoelastic_logfv_square_periodic_2d(;
    Nx::Integer=48,
    Ny::Integer=24,
    side::Integer=6,
    cx::Integer=Nx ÷ 3,
    cy::Integer=Ny ÷ 2,
    nu_s::Real=0.08,
    nu_p::Real=0.02,
    Fx_body::Real=1e-6,
    lambda::Real=2.0,
    bsd_fraction::Real=0.0,
    polymer_substeps=:auto,
    subcycle_relative_tolerance::Real=0.01,
    max_deformation_increment::Real=0.05,
    max_memory_deformation_increment::Real=0.07,
    max_polymer_substeps::Integer=64,
    max_steps::Integer=500,
    backend=KernelAbstractions.CPU(),
    T=Float64,
)
    Nx >= 8 || throw(ArgumentError("Nx must be >= 8"))
    Ny >= 8 || throw(ArgumentError("Ny must be >= 8"))
    nu_s > 0 || throw(ArgumentError("nu_s must be positive"))
    nu_p >= 0 || throw(ArgumentError("nu_p must be non-negative"))
    lambda > 0 || throw(ArgumentError("lambda must be positive"))
    0 <= bsd_fraction <= 1 || throw(ArgumentError("bsd_fraction must be in [0, 1]"))

    Nx_i = Int(Nx)
    Ny_i = Int(Ny)
    side_i = Int(side)
    cx_i = Int(cx)
    cy_i = Int(cy)
    is_solid_h = _logfv_square_obstacle_mask(Nx_i, Ny_i, side_i, cx_i, cy_i)

    nu_s_t = T(nu_s)
    nu_p_t = T(nu_p)
    nu_total_t = nu_s_t + nu_p_t
    bsd_t = T(bsd_fraction)
    Fx_body_t = T(Fx_body)
    lambda_t = T(lambda)
    prefactor_t = nu_p_t / lambda_t
    nu_lbm_t = nu_s_t + bsd_t * nu_p_t
    nu_lbm_t > zero(T) || throw(ArgumentError("nu_s + bsd_fraction * nu_p must be positive"))
    dx = one(T)
    dy = one(T)

    max_grad_norm_estimate = abs(Fx_body_t) * T(Ny_i) / (T(2) * max(nu_total_t, eps(T)))
    subcycle_estimate = logfv_oldroydb_subcycle_estimate(
        Float64(max_grad_norm_estimate),
        Float64(lambda_t),
        1.0;
        relative_tolerance=Float64(subcycle_relative_tolerance),
        max_deformation_increment=Float64(max_deformation_increment),
        max_memory_deformation_increment=Float64(max_memory_deformation_increment),
        min_substeps=1,
        max_substeps=max_polymer_substeps,
    )
    selected_polymer_substeps = if polymer_substeps === :auto
        subcycle_estimate.recommended
    elseif polymer_substeps isa Integer
        polymer_substeps >= 1 || throw(ArgumentError("polymer_substeps must be >= 1"))
        polymer_substeps
    else
        throw(ArgumentError("polymer_substeps must be an integer or :auto"))
    end
    dt_poly = one(T) / T(selected_polymer_substeps)

    config = LBMConfig(D2Q9(); Nx=Nx_i, Ny=Ny_i, ν=Float64(nu_lbm_t), u_lid=0.0, max_steps=max_steps)
    state = initialize_2d(config, T; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    rho, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid
    copyto!(is_solid, is_solid_h)
    omega_t = T(omega(config))

    psixx = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    psixy = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    psiyy = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    psixx_adv = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    psixy_adv = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    psiyy_adv = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    psixx_next = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    psixy_next = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    psiyy_next = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    dudx = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    dudy = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    dvdx = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    dvdy = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    ux_west = KernelAbstractions.zeros(backend, T, Ny_i)
    ux_east = KernelAbstractions.zeros(backend, T, Ny_i)
    uy_south = KernelAbstractions.zeros(backend, T, Nx_i)
    uy_north = KernelAbstractions.zeros(backend, T, Nx_i)
    west_xx = KernelAbstractions.zeros(backend, T, Ny_i)
    west_xy = KernelAbstractions.zeros(backend, T, Ny_i)
    west_yy = KernelAbstractions.zeros(backend, T, Ny_i)
    east_xx = KernelAbstractions.zeros(backend, T, Ny_i)
    east_xy = KernelAbstractions.zeros(backend, T, Ny_i)
    east_yy = KernelAbstractions.zeros(backend, T, Ny_i)
    south_xx = KernelAbstractions.zeros(backend, T, Nx_i)
    south_xy = KernelAbstractions.zeros(backend, T, Nx_i)
    south_yy = KernelAbstractions.zeros(backend, T, Nx_i)
    north_xx = KernelAbstractions.zeros(backend, T, Nx_i)
    north_xy = KernelAbstractions.zeros(backend, T, Nx_i)
    north_yy = KernelAbstractions.zeros(backend, T, Nx_i)
    ux_face = KernelAbstractions.zeros(backend, T, Nx_i + 1, Ny_i)
    uy_face = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i + 1)
    tauxx = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    tauxy = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    tauyy = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    fx_poly = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    fy_poly = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    fx_total = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    fy_total = KernelAbstractions.zeros(backend, T, Nx_i, Ny_i)
    wall_gradient_sides = WallGradientSides(uy_south, uy_north, nothing, nothing)
    logfv_bc = logfv_periodicx_wally_bcspec_2d()

    for _ in 1:max_steps
        logfv_cell_velocity_to_faces_bc_aware_2d!(
            ux_face, uy_face, ux, uy, is_solid,
            ux_west, ux_east, uy_south, uy_north, logfv_bc;
            sync=false,
        )
        logfv_advect_upwind_bc_aware_2d!(
            psixx_adv, psixy_adv, psiyy_adv,
            psixx, psixy, psiyy,
            west_xx, west_xy, west_yy,
            east_xx, east_xy, east_yy,
            south_xx, south_xy, south_yy,
            north_xx, north_xy, north_yy,
            ux_face, uy_face, is_solid, dx, dy, logfv_bc, one(T);
            sync=false,
        )
        logfv_velocity_gradient_bc_aware_2d!(
            dudx, dudy, dvdx, dvdy, ux, uy, is_solid, dx, dy, logfv_bc;
            sync=false,
        )
        # M51 wall-gradient correction REVERTED here (M53d audit): see comment near frozen_channel site.

        psixx_work, psixy_work, psiyy_work = psixx_adv, psixy_adv, psiyy_adv
        for _ in 1:selected_polymer_substeps
            logfv_step_oldroydb_log_2d!(
                psixx_next, psixy_next, psiyy_next,
                psixx_work, psixy_work, psiyy_work,
                dudx, dudy, dvdx, dvdy,
                lambda_t, dt_poly;
                sync=false,
            )
            psixx_work, psixx_next = psixx_next, psixx_work
            psixy_work, psixy_next = psixy_next, psixy_work
            psiyy_work, psiyy_next = psiyy_next, psiyy_work
        end
        psixx, psixx_adv = psixx_work, psixx
        psixy, psixy_adv = psixy_work, psixy
        psiyy, psiyy_adv = psiyy_work, psiyy

        logfv_stress_from_log_2d!(tauxx, tauxy, tauyy, psixx, psixy, psiyy, prefactor_t; sync=false)
        logfv_polymer_force_bc_aware_2d!(
            fx_poly, fy_poly, tauxx, tauxy, tauyy, is_solid, dx, dy, logfv_bc;
            sync=false,
        )
        logfv_bsd_correct_force_bc_aware_2d!(
            fx_total, fy_total, fx_poly, fy_poly, ux, uy, is_solid, bsd_t, nu_p_t, dx, dy,
            logfv_bc;
            sync=false,
        )
        logfv_add_constant_force_2d!(fx_total, fy_total, Fx_body_t, zero(T); sync=false)

        stream_periodic_x_wall_y_2d!(f_out, f_in, Nx_i, Ny_i)
        collide_guo_field_2d!(f_out, is_solid, fx_total, fy_total, omega_t)
        logfv_compute_macroscopic_forced_field_2d!(rho, ux, uy, f_out, fx_total, fy_total; sync=false)
        f_in, f_out = f_out, f_in
    end
    KernelAbstractions.synchronize(backend)

    ux_cpu = Array(ux)
    uy_cpu = Array(uy)
    rho_cpu = Array(rho)
    psixx_cpu = Array(psixx)
    psixy_cpu = Array(psixy)
    psiyy_cpu = Array(psiyy)
    min_c_eig = Inf
    max_speed = 0.0
    for j in 1:Ny_i, i in 1:Nx_i
        if !is_solid_h[i, j]
            cxx, cxy, cyy = logfv_exp_sym2_2d(psixx_cpu[i, j], psixy_cpu[i, j], psiyy_cpu[i, j])
            min_c_eig = min(min_c_eig, logfv_min_eig_sym2_2d(cxx, cxy, cyy))
            max_speed = max(max_speed, hypot(Float64(ux_cpu[i, j]), Float64(uy_cpu[i, j])))
        end
    end

    return (;
        Nx=Nx_i,
        Ny=Ny_i,
        side=side_i,
        cx=cx_i,
        cy=cy_i,
        nu_s=Float64(nu_s_t),
        nu_p=Float64(nu_p_t),
        nu_total=Float64(nu_total_t),
        nu_lbm=Float64(nu_lbm_t),
        Fx_body=Float64(Fx_body_t),
        lambda=Float64(lambda_t),
        bsd_fraction=Float64(bsd_t),
        polymer_substeps=selected_polymer_substeps,
        requested_polymer_substeps=polymer_substeps,
        subcycle_estimate,
        max_steps,
        rho=rho_cpu,
        ux=ux_cpu,
        uy=uy_cpu,
        psixx=psixx_cpu,
        psixy=psixy_cpu,
        psiyy=psiyy_cpu,
        fx_total=Array(fx_total),
        fy_total=Array(fy_total),
        is_solid=is_solid_h,
        min_c_eig,
        max_speed,
        rho_min=minimum(rho_cpu),
        rho_max=maximum(rho_cpu),
    )
end

"""
    run_viscoelastic_logfv_bfs_passive_2d(; kwargs...)

Run a passive log-FV polymer canary on a backward-facing-step geometry.

The hydrodynamic BFS field is first advanced with the modular LI-BB V2 +
Guo-field solvent step. The resulting velocity is then frozen while the
cell-centered log-conformation polymer equation is advanced with open-x
advection, solid-aware gradients, and the local Oldroyd-B source. No polymer
force is fed back into the solvent in this canary.
"""
function run_viscoelastic_logfv_bfs_passive_2d(;
    H_in::Integer=4,
    expansion_ratio::Integer=2,
    L_up::Integer=2,
    L_down::Integer=4,
    nu_s::Real=0.08,
    nu_p::Real=0.02,
    lambda::Real=5.0,
    u_mean::Real=0.01,
    Fx_body::Real=2e-7,
    hydro_steps::Integer=60,
    polymer_steps::Integer=20,
    polymer_substeps=:auto,
    max_memory_deformation_increment::Real=0.07,
    max_polymer_substeps::Integer=64,
    backend=KernelAbstractions.CPU(),
    T=Float64,
)
    H_in >= 3 || throw(ArgumentError("H_in must be >= 3"))
    expansion_ratio >= 2 || throw(ArgumentError("expansion_ratio must be >= 2"))
    nu_s > 0 || throw(ArgumentError("nu_s must be positive"))
    nu_p >= 0 || throw(ArgumentError("nu_p must be non-negative"))
    lambda > 0 || throw(ArgumentError("lambda must be positive"))
    hydro_steps >= 0 || throw(ArgumentError("hydro_steps must be non-negative"))
    polymer_steps >= 0 || throw(ArgumentError("polymer_steps must be non-negative"))

    geom_h = backward_facing_step_geometry_2d(;
        H_in=Int(H_in),
        expansion_ratio=Int(expansion_ratio),
        L_up=Int(L_up),
        L_down=Int(L_down),
        FT=T,
    )
    geom = transfer_step_geometry_2d(geom_h, backend)
    Nx, Ny = geom_h.Nx, geom_h.Ny
    is_solid = geom.is_solid
    q_wall = geom.q_wall
    is_solid_h = geom_h.is_solid

    nu_s_t = T(nu_s)
    nu_p_t = T(nu_p)
    lambda_t = T(lambda)
    prefactor_t = nu_p_t / lambda_t
    Fx_body_t = T(Fx_body)
    u_profile_h = parabolic_face_profile_2d(geom_h; face=:west, mean_velocity=T(u_mean), FT=T)
    u_profile = KernelAbstractions.allocate(backend, T, Ny)
    copyto!(u_profile, u_profile_h)
    bcspec = default_step_bcspec_2d(geom, u_profile, one(T))

    f_in = KernelAbstractions.allocate(backend, T, Nx, Ny, 9)
    f_out = KernelAbstractions.allocate(backend, T, Nx, Ny, 9)
    f_in_h = zeros(T, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        ux0 = is_solid_h[i, j] ? zero(T) : u_profile_h[j]
        f_in_h[i, j, q] = equilibrium(D2Q9(), one(T), ux0, zero(T), q)
    end
    copyto!(f_in, f_in_h)
    fill!(f_out, zero(T))

    rho = KernelAbstractions.zeros(backend, T, Nx, Ny)
    ux = KernelAbstractions.zeros(backend, T, Nx, Ny)
    uy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    uwx = KernelAbstractions.zeros(backend, T, Nx, Ny, 9)
    uwy = KernelAbstractions.zeros(backend, T, Nx, Ny, 9)
    fx_h = [is_solid_h[i, j] ? zero(T) : Fx_body_t for i in 1:Nx, j in 1:Ny]
    fy_h = zeros(T, Nx, Ny)
    fx = KernelAbstractions.allocate(backend, T, Nx, Ny)
    fy = KernelAbstractions.allocate(backend, T, Nx, Ny)
    copyto!(fx, fx_h)
    copyto!(fy, fy_h)

    for _ in 1:hydro_steps
        fused_trt_libb_v2_guo_field_step!(
            f_out, f_in, rho, ux, uy, is_solid, q_wall, uwx, uwy, fx, fy,
            Nx, Ny, nu_s_t;
        )
        apply_bc_rebuild_2d!(f_out, f_in, bcspec, nu_s_t, Nx, Ny)
        f_in, f_out = f_out, f_in
    end
    logfv_compute_macroscopic_forced_field_2d!(rho, ux, uy, f_in, fx, fy)
    KernelAbstractions.synchronize(backend)

    ux_cpu_after_hydro = Array(ux)
    ux_east_h = copy(@view ux_cpu_after_hydro[Nx, :])
    ux_east = KernelAbstractions.allocate(backend, T, Ny)
    copyto!(ux_east, ux_east_h)

    subcycle_estimate = logfv_oldroydb_subcycle_estimate(
        0.0,
        Float64(lambda_t),
        1.0;
        max_memory_deformation_increment=Float64(max_memory_deformation_increment),
        min_substeps=1,
        max_substeps=max_polymer_substeps,
    )
    selected_polymer_substeps = if polymer_substeps === :auto
        subcycle_estimate.recommended
    elseif polymer_substeps isa Integer
        polymer_substeps >= 1 || throw(ArgumentError("polymer_substeps must be >= 1"))
        polymer_substeps
    else
        throw(ArgumentError("polymer_substeps must be an integer or :auto"))
    end
    dt_poly = one(T) / T(selected_polymer_substeps)

    psixx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psiyy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixx_adv = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixy_adv = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psiyy_adv = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixx_next = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixy_next = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psiyy_next = KernelAbstractions.zeros(backend, T, Nx, Ny)
    west_xx = KernelAbstractions.zeros(backend, T, Ny)
    west_xy = KernelAbstractions.zeros(backend, T, Ny)
    west_yy = KernelAbstractions.zeros(backend, T, Ny)
    east_xx = KernelAbstractions.zeros(backend, T, Ny)
    east_xy = KernelAbstractions.zeros(backend, T, Ny)
    east_yy = KernelAbstractions.zeros(backend, T, Ny)
    south_xx = KernelAbstractions.zeros(backend, T, Nx)
    south_xy = KernelAbstractions.zeros(backend, T, Nx)
    south_yy = KernelAbstractions.zeros(backend, T, Nx)
    north_xx = KernelAbstractions.zeros(backend, T, Nx)
    north_xy = KernelAbstractions.zeros(backend, T, Nx)
    north_yy = KernelAbstractions.zeros(backend, T, Nx)
    uy_south = KernelAbstractions.zeros(backend, T, Nx)
    uy_north = KernelAbstractions.zeros(backend, T, Nx)
    ux_face = KernelAbstractions.zeros(backend, T, Nx + 1, Ny)
    uy_face = KernelAbstractions.zeros(backend, T, Nx, Ny + 1)
    dudx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    dudy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    dvdx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    dvdy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tauxx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tauxy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tauyy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    wall_gradient_sides = WallGradientSides(uy_south, uy_north, nothing, nothing)
    logfv_bc = logfv_openx_wally_bcspec_2d()

    for _ in 1:polymer_steps
        logfv_cell_velocity_to_faces_bc_aware_2d!(
            ux_face, uy_face, ux, uy, is_solid,
            u_profile, ux_east, uy_south, uy_north, logfv_bc;
            sync=false,
        )
        logfv_advect_upwind_bc_aware_2d!(
            psixx_adv, psixy_adv, psiyy_adv,
            psixx, psixy, psiyy,
            west_xx, west_xy, west_yy,
            east_xx, east_xy, east_yy,
            south_xx, south_xy, south_yy,
            north_xx, north_xy, north_yy,
            ux_face, uy_face, is_solid, one(T), one(T), logfv_bc, one(T);
            sync=false,
        )
        logfv_velocity_gradient_bc_aware_2d!(
            dudx, dudy, dvdx, dvdy, ux, uy, is_solid, one(T), one(T), logfv_bc;
            sync=false,
        )
        # M51 wall-gradient correction REVERTED here (M53d audit): see comment near frozen_channel site.

        psixx_work, psixy_work, psiyy_work = psixx_adv, psixy_adv, psiyy_adv
        for _ in 1:selected_polymer_substeps
            logfv_step_oldroydb_log_2d!(
                psixx_next, psixy_next, psiyy_next,
                psixx_work, psixy_work, psiyy_work,
                dudx, dudy, dvdx, dvdy,
                lambda_t, dt_poly;
                sync=false,
            )
            psixx_work, psixx_next = psixx_next, psixx_work
            psixy_work, psixy_next = psixy_next, psixy_work
            psiyy_work, psiyy_next = psiyy_next, psiyy_work
        end
        psixx, psixx_adv = psixx_work, psixx
        psixy, psixy_adv = psixy_work, psixy
        psiyy, psiyy_adv = psiyy_work, psiyy
    end
    logfv_stress_from_log_2d!(tauxx, tauxy, tauyy, psixx, psixy, psiyy, prefactor_t)
    KernelAbstractions.synchronize(backend)

    rho_cpu = Array(rho)
    ux_cpu = Array(ux)
    uy_cpu = Array(uy)
    psixx_cpu = Array(psixx)
    psixy_cpu = Array(psixy)
    psiyy_cpu = Array(psiyy)
    tauxx_cpu = Array(tauxx)
    tauxy_cpu = Array(tauxy)
    tauyy_cpu = Array(tauyy)

    min_c_eig = Inf
    max_speed = 0.0
    for j in 1:Ny, i in 1:Nx
        if !is_solid_h[i, j]
            cxx, cxy, cyy = logfv_exp_sym2_2d(psixx_cpu[i, j], psixy_cpu[i, j], psiyy_cpu[i, j])
            min_c_eig = min(min_c_eig, logfv_min_eig_sym2_2d(cxx, cxy, cyy))
            max_speed = max(max_speed, hypot(Float64(ux_cpu[i, j]), Float64(uy_cpu[i, j])))
        end
    end
    fluid_mask = .!is_solid_h
    max_abs_psi = max(maximum(abs, psixx_cpu), maximum(abs, psixy_cpu), maximum(abs, psiyy_cpu))
    max_abs_tau = max(maximum(abs, tauxx_cpu), maximum(abs, tauxy_cpu), maximum(abs, tauyy_cpu))

    return (;
        geometry=geom_h,
        Nx,
        Ny,
        nu_s=Float64(nu_s_t),
        nu_p=Float64(nu_p_t),
        lambda=Float64(lambda_t),
        u_mean=Float64(u_mean),
        Fx_body=Float64(Fx_body_t),
        hydro_steps,
        polymer_steps,
        polymer_substeps=selected_polymer_substeps,
        subcycle_estimate,
        rho=rho_cpu,
        ux=ux_cpu,
        uy=uy_cpu,
        psixx=psixx_cpu,
        psixy=psixy_cpu,
        psiyy=psiyy_cpu,
        tauxx=tauxx_cpu,
        tauxy=tauxy_cpu,
        tauyy=tauyy_cpu,
        is_solid=is_solid_h,
        min_c_eig,
        max_speed,
        max_abs_psi,
        max_abs_tau,
        rho_min=minimum(rho_cpu[fluid_mask]),
        rho_max=maximum(rho_cpu[fluid_mask]),
    )
end
