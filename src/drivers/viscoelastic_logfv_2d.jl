using KernelAbstractions

@inline function _logfv_channel_shear(flow::Symbol, y, height, umax, uwall)
    if flow === :poiseuille
        return 4 * umax / height * (1 - 2 * y / height)
    elseif flow === :couette
        return uwall / height
    else
        error("unsupported log-FV channel flow $(flow); expected :poiseuille or :couette")
    end
end

@inline function _logfv_channel_ux(flow::Symbol, y, height, umax, uwall)
    if flow === :poiseuille
        eta = y / height
        return 4 * umax * eta * (1 - eta)
    elseif flow === :couette
        return uwall * y / height
    else
        error("unsupported log-FV channel flow $(flow); expected :poiseuille or :couette")
    end
end

@inline function _logfv_channel_lapu(flow::Symbol, height, umax)
    if flow === :poiseuille
        return -8 * umax / (height * height)
    elseif flow === :couette
        return 0.0
    else
        error("unsupported log-FV channel flow $(flow); expected :poiseuille or :couette")
    end
end

function _logfv_channel_reference_errors(
    flow::Symbol, tauxx, tauxy, tauyy, fx_poly, fy_poly, fx_total, fy_total,
    height, umax, uwall, lambda, prefactor, bsd_fraction,
)
    Nx, Ny = size(tauxx)
    max_tau_error = 0.0
    max_poly_force_error = 0.0
    max_total_force_error = 0.0
    max_transverse_force = 0.0
    min_c_eig = Inf
    nu_p = prefactor * lambda
    lapu = _logfv_channel_lapu(flow, height, umax)
    dy = height / Ny

    for j in 1:Ny, i in 1:Nx
        y = (j - 0.5) * dy
        gamma = _logfv_channel_shear(flow, y, height, umax, uwall)
        cxx = 1 + 2 * (lambda * gamma)^2
        cxy = lambda * gamma
        cyy = 1.0
        min_c_eig = min(min_c_eig, logfv_min_eig_sym2_2d(cxx, cxy, cyy))
        expected_tau = (
            prefactor * (cxx - 1),
            prefactor * cxy,
            prefactor * (cyy - 1),
        )
        max_tau_error = max(
            max_tau_error,
            abs(tauxx[i, j] - expected_tau[1]),
            abs(tauxy[i, j] - expected_tau[2]),
            abs(tauyy[i, j] - expected_tau[3]),
        )

        if i > 1 && i < Nx && j > 1 && j < Ny
            expected_poly_fx = nu_p * lapu
            expected_total_fx = (1 - bsd_fraction) * nu_p * lapu
            max_poly_force_error = max(max_poly_force_error, abs(fx_poly[i, j] - expected_poly_fx))
            max_total_force_error = max(max_total_force_error, abs(fx_total[i, j] - expected_total_fx))
            max_transverse_force = max(
                max_transverse_force,
                abs(fy_poly[i, j]),
                abs(fy_total[i, j]),
            )
        end
    end

    return (;
        max_tau_error,
        max_poly_force_error,
        max_total_force_error,
        max_transverse_force,
        min_c_eig,
    )
end

"""
    run_viscoelastic_logfv_channel_2d(; kwargs...)

Run the first patch-local macro canary for the cell-centered log-FV
polymer backend on a prescribed channel flow.

This is not a coupled fluid solve. It exercises the macro-domain polymer
pipeline on one uniform patch:

```text
analytic channel u -> analytic steady Psi -> tau_p -> div(tau_p) -> BSD force
```

The function keeps `dx`, `dy`, and patch fields explicit so the same
kernel path can later be wrapped by Basilisk-style quadtree AMR patch
exchange/prolongation/restriction.
"""
function run_viscoelastic_logfv_channel_2d(;
    Nx::Integer=32,
    Ny::Integer=32,
    flow::Symbol=:poiseuille,
    height::Real=1.0,
    width::Real=1.0,
    umax::Real=0.05,
    uwall::Real=0.05,
    lambda::Real=5.0,
    beta::Real=0.5,
    Wi::Real=1.0,
    prefactor::Union{Nothing,Real}=nothing,
    bsd_fraction::Real=0.0,
    backend=KernelAbstractions.CPU(),
    T=Float64,
)
    Nx >= 3 || throw(ArgumentError("Nx must be >= 3 for centered force checks"))
    Ny >= 3 || throw(ArgumentError("Ny must be >= 3 for centered force checks"))
    flow in (:poiseuille, :couette) || throw(ArgumentError("flow must be :poiseuille or :couette"))

    height_t = T(height)
    width_t = T(width)
    lambda_t = T(lambda)
    umax_t = T(umax)
    uwall_t = T(uwall)
    prefactor_t = isnothing(prefactor) ? (one(T) - T(beta)) / T(Wi) : T(prefactor)
    bsd_t = T(bsd_fraction)
    dx = width_t / T(Nx)
    dy = height_t / T(Ny)
    nu_p = prefactor_t * lambda_t

    psixx_h = zeros(T, Nx, Ny)
    psixy_h = zeros(T, Nx, Ny)
    psiyy_h = zeros(T, Nx, Ny)
    ux_h = zeros(T, Nx, Ny)
    uy_h = zeros(T, Nx, Ny)

    for j in 1:Ny, i in 1:Nx
        y = (T(j) - T(0.5)) * dy
        gamma = T(_logfv_channel_shear(flow, y, height_t, umax_t, uwall_t))
        ux_h[i, j] = T(_logfv_channel_ux(flow, y, height_t, umax_t, uwall_t))
        cxx = one(T) + T(2) * (lambda_t * gamma)^2
        cxy = lambda_t * gamma
        cyy = one(T)
        psixx_h[i, j], psixy_h[i, j], psiyy_h[i, j] = logfv_log_spd_sym2_2d(cxx, cxy, cyy)
    end

    psixx = KernelAbstractions.allocate(backend, T, Nx, Ny)
    psixy = KernelAbstractions.allocate(backend, T, Nx, Ny)
    psiyy = KernelAbstractions.allocate(backend, T, Nx, Ny)
    ux = KernelAbstractions.allocate(backend, T, Nx, Ny)
    uy = KernelAbstractions.allocate(backend, T, Nx, Ny)
    copyto!(psixx, psixx_h)
    copyto!(psixy, psixy_h)
    copyto!(psiyy, psiyy_h)
    copyto!(ux, ux_h)
    copyto!(uy, uy_h)

    tauxx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tauxy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tauyy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fx_poly = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fy_poly = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fx_total = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fy_total = KernelAbstractions.zeros(backend, T, Nx, Ny)

    logfv_stress_from_log_2d!(tauxx, tauxy, tauyy, psixx, psixy, psiyy, prefactor_t)
    logfv_polymer_force_centered_2d!(fx_poly, fy_poly, tauxx, tauxy, tauyy, dx, dy)
    logfv_bsd_correct_force_centered_2d!(
        fx_total, fy_total, fx_poly, fy_poly, ux, uy, bsd_t, nu_p, dx, dy,
    )
    KernelAbstractions.synchronize(backend)

    tauxx_cpu = Array(tauxx)
    tauxy_cpu = Array(tauxy)
    tauyy_cpu = Array(tauyy)
    fx_poly_cpu = Array(fx_poly)
    fy_poly_cpu = Array(fy_poly)
    fx_total_cpu = Array(fx_total)
    fy_total_cpu = Array(fy_total)
    errors = _logfv_channel_reference_errors(
        flow, tauxx_cpu, tauxy_cpu, tauyy_cpu, fx_poly_cpu, fy_poly_cpu,
        fx_total_cpu, fy_total_cpu, Float64(height_t), Float64(umax_t),
        Float64(uwall_t), Float64(lambda_t), Float64(prefactor_t), Float64(bsd_t),
    )

    return (;
        flow,
        Nx,
        Ny,
        dx=Float64(dx),
        dy=Float64(dy),
        height=Float64(height_t),
        width=Float64(width_t),
        lambda=Float64(lambda_t),
        prefactor=Float64(prefactor_t),
        beta=Float64(beta),
        Wi=Float64(Wi),
        bsd_fraction=Float64(bsd_t),
        nu_p=Float64(nu_p),
        ux=Array(ux),
        uy=Array(uy),
        psixx=Array(psixx),
        psixy=Array(psixy),
        psiyy=Array(psiyy),
        tauxx=tauxx_cpu,
        tauxy=tauxy_cpu,
        tauyy=tauyy_cpu,
        fx_poly=fx_poly_cpu,
        fy_poly=fy_poly_cpu,
        fx_total=fx_total_cpu,
        fy_total=fy_total_cpu,
        errors...,
    )
end

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
    force_boundary_fill::Symbol=:nearest,
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
    force_boundary_fill in (:nearest, :none) ||
        throw(ArgumentError("force_boundary_fill must be :nearest or :none"))

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
subcycling estimator; future Strang or local affine source solves should
reduce this requirement.
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
    max_polymer_substeps::Integer=64,
    force_boundary_fill::Symbol=:nearest,
    max_steps::Integer=10000,
    backend=KernelAbstractions.CPU(),
    T=Float64,
)
    Nx >= 3 || throw(ArgumentError("Nx must be >= 3"))
    Ny >= 5 || throw(ArgumentError("Ny must be >= 5"))
    nu_s > 0 || throw(ArgumentError("nu_s must be positive"))
    nu_p >= 0 || throw(ArgumentError("nu_p must be non-negative"))
    lambda > 0 || throw(ArgumentError("lambda must be positive"))
    force_boundary_fill in (:nearest, :none) ||
        throw(ArgumentError("force_boundary_fill must be :nearest or :none"))

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

    for _ in 1:max_steps
        logfv_velocity_gradient_periodicx_wally_2d!(dudx, dudy, dvdx, dvdy, ux, uy, dx, dy; sync=false)
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
        logfv_polymer_force_centered_2d!(fx_poly, fy_poly, tauxx, tauxy, tauyy, dx, dy; sync=false)
        logfv_bsd_correct_force_centered_2d!(
            fx_total, fy_total, fx_poly, fy_poly, ux, uy, bsd_t, nu_p_t, dx, dy;
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
