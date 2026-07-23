function _logfv_lbm_poiseuille_reference(Fx_body, nu_total, Ny)
    return [Fx_body / (2 * nu_total) * (j - 0.5) * (Ny + 0.5 - j) for j in 1:Ny]
end

"""
    run_viscoelastic_logfv_poiseuille_frozen_force_2d(; kwargs...)

Run the first coupled LBM/log-FV macro canary on a periodic channel.

The polymer field is frozen at the analytical Oldroyd-B Poiseuille solution,
then the production log-FV kernels reconstruct `tau_p`, `div(tau_p)`, and the
BSD-corrected force. The solvent LBM is advanced with that force field. This
isolates the momentum-coupling contract:

```text
body force + log-FV polymer force + BSD correction -> total-viscosity profile
```

It does not validate polymer advection or near-wall polymer boundary
conditions. Those stay in lower canaries before square/obstacle flows.
"""
function run_viscoelastic_logfv_poiseuille_frozen_force_2d(;
    Nx::Integer=4,
    Ny::Integer=32,
    nu_s::Real=0.04,
    nu_p::Real=0.06,
    Fx_body::Real=1e-5,
    lambda::Real=5.0,
    bsd_fraction::Real=0.0,
    force_boundary_fill::Symbol=:bc_aware,
    max_steps::Integer=12000,
    backend=KernelAbstractions.CPU(),
    T=Float64,
)
    Nx >= 3 || throw(ArgumentError("Nx must be >= 3"))
    Ny >= 5 || throw(ArgumentError("Ny must be >= 5"))
    nu_s > 0 || throw(ArgumentError("nu_s must be positive"))
    nu_p >= 0 || throw(ArgumentError("nu_p must be non-negative"))
    lambda > 0 || throw(ArgumentError("lambda must be positive"))

    nu_s_t = T(nu_s)
    nu_p_t = T(nu_p)
    nu_total_t = nu_s_t + nu_p_t
    bsd_t = T(bsd_fraction)
    Fx_body_t = T(Fx_body)
    lambda_t = T(lambda)

    nu_lbm_t = nu_s_t + bsd_t * nu_p_t
    nu_lbm_t > zero(T) || throw(ArgumentError("nu_s + bsd_fraction * nu_p must be positive"))
    force_boundary_fill in (:bc_aware, :nearest, :none) ||
        throw(ArgumentError("force_boundary_fill must be :bc_aware, :nearest, or :none"))

    height_t = T(Ny)
    width_t = T(Nx)
    umax_t = Fx_body_t * height_t * height_t / (T(8) * nu_total_t)
    prefactor_t = iszero(lambda_t) ? zero(T) : nu_p_t / lambda_t

    channel = run_viscoelastic_logfv_channel_2d(;
        Nx=Nx,
        Ny=Ny,
        flow=:poiseuille,
        height=height_t,
        width=width_t,
        umax=umax_t,
        uwall=zero(T),
        lambda=lambda_t,
        prefactor=prefactor_t,
        bsd_fraction=bsd_t,
        backend=backend,
        T=T,
    )

    fx_total_h = T.(channel.fx_total)
    fy_total_h = T.(channel.fy_total)
    for j in 1:Ny, i in 1:Nx
        fx_total_h[i, j] += Fx_body_t
    end

    fx_total = KernelAbstractions.allocate(backend, T, Nx, Ny)
    fy_total = KernelAbstractions.allocate(backend, T, Nx, Ny)
    copyto!(fx_total, fx_total_h)
    copyto!(fy_total, fy_total_h)
    if force_boundary_fill === :nearest
        logfv_fill_nearest_boundary_2d!(fx_total, fy_total)
    end

    config = LBMConfig(D2Q9(); Nx=Nx, Ny=Ny, ν=Float64(nu_lbm_t), u_lid=0.0, max_steps=max_steps)
    state = initialize_2d(config, T; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    rho, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid
    omega_t = T(omega(config))

    for _ in 1:max_steps
        stream_periodic_x_wall_y_2d!(f_out, f_in, Nx, Ny)
        collide_guo_field_2d!(f_out, is_solid, fx_total, fy_total, omega_t)
        logfv_compute_macroscopic_forced_field_2d!(rho, ux, uy, f_out, fx_total, fy_total; sync=false)
        f_in, f_out = f_out, f_in
    end
    KernelAbstractions.synchronize(backend)

    ux_cpu = Array(ux)
    uy_cpu = Array(uy)
    rho_cpu = Array(rho)
    reference_u = _logfv_lbm_poiseuille_reference(Float64(Fx_body_t), Float64(nu_total_t), Ny)
    mean_ux = [sum(@view ux_cpu[:, j]) / Nx for j in 1:Ny]
    interior = 3:(Ny - 2)
    max_abs_error = maximum(abs.(mean_ux[interior] .- reference_u[interior]))
    max_ref = maximum(abs.(reference_u[interior]))
    max_rel_error = max_abs_error / max(max_ref, eps(Float64))
    max_uy = maximum(abs, uy_cpu[:, interior])

    return (;
        Nx,
        Ny,
        nu_s=Float64(nu_s_t),
        nu_p=Float64(nu_p_t),
        nu_total=Float64(nu_total_t),
        nu_lbm=Float64(nu_lbm_t),
        Fx_body=Float64(Fx_body_t),
        lambda=Float64(lambda_t),
        bsd_fraction=Float64(bsd_t),
        force_boundary_fill,
        max_steps,
        rho=rho_cpu,
        ux=ux_cpu,
        uy=uy_cpu,
        ux_mean=mean_ux,
        reference_ux=reference_u,
        fx_total=Array(fx_total),
        fy_total=Array(fy_total),
        polymer_channel=channel,
        max_abs_error,
        max_rel_error,
        max_uy,
    )
end

"""
    run_viscoelastic_logfv_poiseuille_coupled_2d(; kwargs...)

Run a coarse coupled channel canary with dynamic log-FV polymer stress.

This keeps the flow fully developed and periodic in `x`, so polymer advection is
identically zero. The canary exercises the local coupled loop without obstacle
or curved-wall complications:

```text
LBM u -> wall-exact channel grad(u) -> log-C Oldroyd-B source
      -> tau_p -> div(tau_p) + BSD -> Guo force -> LBM u
```

`polymer_substeps` is a time-integration convergence control for the current
Lie source split. It is not a physical parameter and must not be fitted to a
benchmark. Use `:auto` to choose a global patch value from the source
subcycling estimator. The estimator limits per-step relaxation, per-step
deformation, and memory-time deformation `lambda * ||grad(u)||`; future Strang
or local affine source solves should reduce this requirement.
"""
function run_viscoelastic_logfv_poiseuille_coupled_2d(;
    Nx::Integer=6,
    Ny::Integer=20,
    nu_s::Real=0.04,
    nu_p::Real=0.06,
    Fx_body::Real=1e-5,
    lambda::Real=5.0,
    bsd_fraction::Real=1.0,
    polymer_substeps=:auto,
    subcycle_relative_tolerance::Real=0.01,
    max_deformation_increment::Real=0.05,
    max_memory_deformation_increment::Real=0.07,
    max_polymer_substeps::Integer=64,
    force_boundary_fill::Symbol=:bc_aware,
    max_steps::Integer=10000,
    backend=KernelAbstractions.CPU(),
    T=Float64,
)
    Nx >= 3 || throw(ArgumentError("Nx must be >= 3"))
    Ny >= 5 || throw(ArgumentError("Ny must be >= 5"))
    nu_s > 0 || throw(ArgumentError("nu_s must be positive"))
    nu_p >= 0 || throw(ArgumentError("nu_p must be non-negative"))
    lambda > 0 || throw(ArgumentError("lambda must be positive"))
    force_boundary_fill in (:bc_aware, :nearest, :none) ||
        throw(ArgumentError("force_boundary_fill must be :bc_aware, :nearest, or :none"))

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
    max_grad_norm_estimate = abs(Fx_body_t) * T(Ny) / (T(2) * nu_total_t)
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

    config = LBMConfig(D2Q9(); Nx=Nx, Ny=Ny, ν=Float64(nu_lbm_t), u_lid=0.0, max_steps=max_steps)
    state = initialize_2d(config, T; backend=backend)
    f_in, f_out = state.f_in, state.f_out
    rho, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid
    omega_t = T(omega(config))

    psixx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psiyy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixx_next = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixy_next = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psiyy_next = KernelAbstractions.zeros(backend, T, Nx, Ny)
    dudx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    dudy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    dvdx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    dvdy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tauxx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tauxy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tauyy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fx_poly = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fy_poly = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fx_total = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fy_total = KernelAbstractions.zeros(backend, T, Nx, Ny)
    wall_ux = KernelAbstractions.zeros(backend, T, Nx)
    wall_gradient_sides = WallGradientSides(wall_ux, wall_ux, nothing, nothing)
    logfv_bc = logfv_periodicx_wally_bcspec_2d()

    for _ in 1:max_steps
        logfv_velocity_gradient_bc_aware_2d!(
            dudx, dudy, dvdx, dvdy, ux, uy, is_solid, dx, dy, logfv_bc;
            sync=false,
        )
        # M51 wall-gradient correction REVERTED here (M53d audit): see comment near frozen_channel site.
        for _ in 1:selected_polymer_substeps
            logfv_step_oldroydb_log_2d!(
                psixx_next, psixy_next, psiyy_next,
                psixx, psixy, psiyy,
                dudx, dudy, dvdx, dvdy,
                lambda_t, dt_poly;
                sync=false,
            )
            psixx, psixx_next = psixx_next, psixx
            psixy, psixy_next = psixy_next, psixy
            psiyy, psiyy_next = psiyy_next, psiyy
        end
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
        if force_boundary_fill === :nearest
            logfv_fill_nearest_boundary_2d!(fx_total, fy_total; sync=false)
        end
        logfv_add_constant_force_2d!(fx_total, fy_total, Fx_body_t, zero(T); sync=false)

        stream_periodic_x_wall_y_2d!(f_out, f_in, Nx, Ny)
        collide_guo_field_2d!(f_out, is_solid, fx_total, fy_total, omega_t)
        logfv_compute_macroscopic_forced_field_2d!(rho, ux, uy, f_out, fx_total, fy_total; sync=false)
        f_in, f_out = f_out, f_in
    end
    KernelAbstractions.synchronize(backend)

    ux_cpu = Array(ux)
    uy_cpu = Array(uy)
    reference_u = _logfv_lbm_poiseuille_reference(Float64(Fx_body_t), Float64(nu_total_t), Ny)
    mean_ux = [sum(@view ux_cpu[:, j]) / Nx for j in 1:Ny]
    interior = 3:(Ny - 2)
    max_abs_error = maximum(abs.(mean_ux[interior] .- reference_u[interior]))
    max_ref = maximum(abs.(reference_u[interior]))
    max_rel_error = max_abs_error / max(max_ref, eps(Float64))
    max_uy = maximum(abs, uy_cpu[:, interior])

    psixx_cpu = Array(psixx)
    psixy_cpu = Array(psixy)
    psiyy_cpu = Array(psiyy)
    min_c_eig = Inf
    for j in 1:Ny, i in 1:Nx
        cxx, cxy, cyy = logfv_exp_sym2_2d(psixx_cpu[i, j], psixy_cpu[i, j], psiyy_cpu[i, j])
        min_c_eig = min(min_c_eig, logfv_min_eig_sym2_2d(cxx, cxy, cyy))
    end

    return (;
        Nx,
        Ny,
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
        max_grad_norm_estimate=Float64(max_grad_norm_estimate),
        force_boundary_fill,
        max_steps,
        rho=Array(rho),
        ux=ux_cpu,
        uy=uy_cpu,
        ux_mean=mean_ux,
        reference_ux=reference_u,
        psixx=psixx_cpu,
        psixy=psixy_cpu,
        psiyy=psiyy_cpu,
        fx_total=Array(fx_total),
        fy_total=Array(fy_total),
        min_c_eig,
        max_abs_error,
        max_rel_error,
        max_uy,
    )
end

