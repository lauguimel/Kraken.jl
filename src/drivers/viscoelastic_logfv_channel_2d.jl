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
    is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny)
    copyto!(psixx, psixx_h)
    copyto!(psixy, psixy_h)
    copyto!(psiyy, psiyy_h)
    copyto!(ux, ux_h)
    copyto!(uy, uy_h)
    fill!(is_solid, false)

    tauxx = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tauxy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    tauyy = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fx_poly = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fy_poly = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fx_total = KernelAbstractions.zeros(backend, T, Nx, Ny)
    fy_total = KernelAbstractions.zeros(backend, T, Nx, Ny)
    bc = logfv_periodicx_wally_bcspec_2d()

    logfv_stress_from_log_2d!(tauxx, tauxy, tauyy, psixx, psixy, psiyy, prefactor_t)
    logfv_polymer_force_bc_aware_2d!(fx_poly, fy_poly, tauxx, tauxy, tauyy, is_solid, dx, dy, bc)
    logfv_bsd_correct_force_bc_aware_2d!(
        fx_total, fy_total, fx_poly, fy_poly, ux, uy, is_solid, bsd_t, nu_p, dx, dy, bc,
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

function _logfv_channel_reference_fields(
    flow::Symbol,
    Nx::Integer,
    Ny::Integer,
    height,
    width,
    umax,
    uwall,
    lambda,
    prefactor,
    ::Type{T},
) where {T}
    height_t = T(height)
    width_t = T(width)
    lambda_t = T(lambda)
    umax_t = T(umax)
    uwall_t = T(uwall)
    prefactor_t = T(prefactor)
    dx = width_t / T(Nx)
    dy = height_t / T(Ny)

    ux_h = zeros(T, Nx, Ny)
    uy_h = zeros(T, Nx, Ny)
    cxx_h = zeros(T, Nx, Ny)
    cxy_h = zeros(T, Nx, Ny)
    cyy_h = zeros(T, Nx, Ny)
    psixx_h = zeros(T, Nx, Ny)
    psixy_h = zeros(T, Nx, Ny)
    psiyy_h = zeros(T, Nx, Ny)
    tauxx_ref = zeros(T, Nx, Ny)
    tauxy_ref = zeros(T, Nx, Ny)
    tauyy_ref = zeros(T, Nx, Ny)
    dudx_ref = zeros(T, Nx, Ny)
    dudy_ref = zeros(T, Nx, Ny)
    dvdx_ref = zeros(T, Nx, Ny)
    dvdy_ref = zeros(T, Nx, Ny)

    for j in 1:Ny, i in 1:Nx
        y = (T(j) - T(0.5)) * dy
        gamma = T(_logfv_channel_shear(flow, y, height_t, umax_t, uwall_t))
        ux_h[i, j] = T(_logfv_channel_ux(flow, y, height_t, umax_t, uwall_t))
        dudy_ref[i, j] = gamma
        cxx = one(T) + T(2) * (lambda_t * gamma)^2
        cxy = lambda_t * gamma
        cyy = one(T)
        cxx_h[i, j] = cxx
        cxy_h[i, j] = cxy
        cyy_h[i, j] = cyy
        psixx_h[i, j], psixy_h[i, j], psiyy_h[i, j] =
            logfv_log_spd_sym2_2d(cxx, cxy, cyy)
        tauxx_ref[i, j] = prefactor_t * (cxx - one(T))
        tauxy_ref[i, j] = prefactor_t * cxy
        tauyy_ref[i, j] = prefactor_t * (cyy - one(T))
    end

    return (;
        dx,
        dy,
        ux=ux_h,
        uy=uy_h,
        cxx=cxx_h,
        cxy=cxy_h,
        cyy=cyy_h,
        psixx=psixx_h,
        psixy=psixy_h,
        psiyy=psiyy_h,
        tauxx=tauxx_ref,
        tauxy=tauxy_ref,
        tauyy=tauyy_ref,
        dudx=dudx_ref,
        dudy=dudy_ref,
        dvdx=dvdx_ref,
        dvdy=dvdy_ref,
    )
end

"""
    run_viscoelastic_logfv_frozen_channel_cde_2d(; kwargs...)

Run the FVFD/log-FV polymer CDE on a prescribed analytical channel velocity
field without feeding polymer force back into the LBM solvent.

This is the quantitative frozen-flow gate between local analytical operators
and obstacle/RheoTool comparisons:

```text
frozen u -> FVFD face velocities -> FVFD advection -> log-C source
         -> tau_p -> div(tau_p) + BSD diagnostics
```

Use `initial=:steady` to check that the production operator path preserves the
analytical Oldroyd-B channel solution up to source-splitting error. Use
`initial=:identity` to measure transient convergence toward the same reference.
"""
function run_viscoelastic_logfv_frozen_channel_cde_2d(;
    Nx::Integer=16,
    Ny::Integer=32,
    flow::Symbol=:poiseuille,
    height::Real=1.0,
    width::Real=1.0,
    umax::Real=0.02,
    uwall::Real=0.02,
    lambda::Real=2.0,
    beta::Real=0.5,
    Wi::Real=1.0,
    prefactor::Union{Nothing,Real}=nothing,
    bsd_fraction::Real=1.0,
    initial::Symbol=:steady,
    max_steps::Integer=1,
    polymer_substeps=:auto,
    subcycle_relative_tolerance::Real=0.01,
    max_deformation_increment::Real=0.05,
    max_memory_deformation_increment::Real=0.07,
    max_polymer_substeps::Integer=256,
    backend=KernelAbstractions.CPU(),
    T=Float64,
)
    Nx >= 3 || throw(ArgumentError("Nx must be >= 3"))
    Ny >= 5 || throw(ArgumentError("Ny must be >= 5"))
    flow in (:poiseuille, :couette) || throw(ArgumentError("flow must be :poiseuille or :couette"))
    lambda > 0 || throw(ArgumentError("lambda must be positive"))
    max_steps >= 0 || throw(ArgumentError("max_steps must be non-negative"))
    initial in (:steady, :identity) ||
        throw(ArgumentError("initial must be :steady or :identity"))

    prefactor_t = isnothing(prefactor) ? (one(T) - T(beta)) / T(Wi) : T(prefactor)
    bsd_t = T(bsd_fraction)
    lambda_t = T(lambda)
    height_t = T(height)
    width_t = T(width)
    umax_t = T(umax)
    uwall_t = T(uwall)
    max_grad_norm = flow === :poiseuille ?
        abs(T(4) * umax_t / height_t) :
        abs(uwall_t / height_t)
    subcycle_estimate = logfv_oldroydb_subcycle_estimate(
        Float64(max_grad_norm),
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

    ref = _logfv_channel_reference_fields(
        flow, Nx, Ny, height_t, width_t, umax_t, uwall_t, lambda_t, prefactor_t, T,
    )
    psixx_h = initial === :steady ? copy(ref.psixx) : zeros(T, Nx, Ny)
    psixy_h = initial === :steady ? copy(ref.psixy) : zeros(T, Nx, Ny)
    psiyy_h = initial === :steady ? copy(ref.psiyy) : zeros(T, Nx, Ny)

    ux = KernelAbstractions.allocate(backend, T, Nx, Ny)
    uy = KernelAbstractions.allocate(backend, T, Nx, Ny)
    is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny)
    psixx = KernelAbstractions.allocate(backend, T, Nx, Ny)
    psixy = KernelAbstractions.allocate(backend, T, Nx, Ny)
    psiyy = KernelAbstractions.allocate(backend, T, Nx, Ny)
    psixx_adv = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixy_adv = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psiyy_adv = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixx_next = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psixy_next = KernelAbstractions.zeros(backend, T, Nx, Ny)
    psiyy_next = KernelAbstractions.zeros(backend, T, Nx, Ny)
    ux_face = KernelAbstractions.zeros(backend, T, Nx + 1, Ny)
    uy_face = KernelAbstractions.zeros(backend, T, Nx, Ny + 1)
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

    copyto!(ux, ref.ux)
    copyto!(uy, ref.uy)
    fill!(is_solid, false)
    copyto!(psixx, psixx_h)
    copyto!(psixy, psixy_h)
    copyto!(psiyy, psiyy_h)

    bc = logfv_periodicx_wally_bcspec_2d()
    logfv_cell_velocity_to_faces_bc_aware_2d!(
        ux_face, uy_face, ux, uy, is_solid,
        ux, ux, uy, uy, bc; sync=false,
    )
    logfv_velocity_gradient_bc_aware_2d!(
        dudx, dudy, dvdx, dvdy, ux, uy, is_solid, ref.dx, ref.dy, bc;
        sync=false,
    )
    # M51 wall-gradient correction REVERTED here (M53d audit):
    # the helper produces wall-position gradient but the polymer chain
    # consumes cell-center gradient → semantic mismatch broke M5e
    # frozen-channel CDE steady-state assertions. Kept only on cylinder
    # + cavity where the precondition was tested.

    for _ in 1:max_steps
        logfv_advect_upwind_bc_aware_2d!(
            psixx_adv, psixy_adv, psiyy_adv,
            psixx, psixy, psiyy,
            psixx, psixy, psiyy,
            psixx, psixy, psiyy,
            psixx, psixy, psiyy,
            psixx, psixy, psiyy,
            ux_face, uy_face, is_solid, ref.dx, ref.dy, bc, one(T);
            sync=false,
        )
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
    logfv_polymer_force_bc_aware_2d!(
        fx_poly, fy_poly, tauxx, tauxy, tauyy, is_solid, ref.dx, ref.dy, bc;
        sync=false,
    )
    logfv_bsd_correct_force_bc_aware_2d!(
        fx_total, fy_total, fx_poly, fy_poly, ux, uy, is_solid,
        bsd_t, prefactor_t * lambda_t, ref.dx, ref.dy, bc;
        sync=false,
    )
    KernelAbstractions.synchronize(backend)

    psixx_cpu = Array(psixx)
    psixy_cpu = Array(psixy)
    psiyy_cpu = Array(psiyy)
    dudx_cpu = Array(dudx)
    dudy_cpu = Array(dudy)
    dvdx_cpu = Array(dvdx)
    dvdy_cpu = Array(dvdy)
    tauxx_cpu = Array(tauxx)
    tauxy_cpu = Array(tauxy)
    tauyy_cpu = Array(tauyy)
    fx_poly_cpu = Array(fx_poly)
    fy_poly_cpu = Array(fy_poly)
    fx_total_cpu = Array(fx_total)
    fy_total_cpu = Array(fy_total)

    cxx_cpu = similar(psixx_cpu)
    cxy_cpu = similar(psixy_cpu)
    cyy_cpu = similar(psiyy_cpu)
    max_c_error = 0.0
    max_psi_error = 0.0
    max_velocity_gradient_error = 0.0
    max_gradient_component_error = (dudx=0.0, dudy=0.0, dvdx=0.0, dvdy=0.0)
    min_c_eig = Inf
    for j in 1:Ny, i in 1:Nx
        cxx, cxy, cyy = logfv_exp_sym2_2d(psixx_cpu[i, j], psixy_cpu[i, j], psiyy_cpu[i, j])
        cxx_cpu[i, j] = cxx
        cxy_cpu[i, j] = cxy
        cyy_cpu[i, j] = cyy
        max_c_error = max(
            max_c_error,
            abs(cxx - ref.cxx[i, j]),
            abs(cxy - ref.cxy[i, j]),
            abs(cyy - ref.cyy[i, j]),
        )
        max_psi_error = max(
            max_psi_error,
            abs(psixx_cpu[i, j] - ref.psixx[i, j]),
            abs(psixy_cpu[i, j] - ref.psixy[i, j]),
            abs(psiyy_cpu[i, j] - ref.psiyy[i, j]),
        )
        dudx_error = abs(dudx_cpu[i, j] - ref.dudx[i, j])
        dudy_error = abs(dudy_cpu[i, j] - ref.dudy[i, j])
        dvdx_error = abs(dvdx_cpu[i, j] - ref.dvdx[i, j])
        dvdy_error = abs(dvdy_cpu[i, j] - ref.dvdy[i, j])
        max_gradient_component_error = (
            dudx=max(max_gradient_component_error.dudx, dudx_error),
            dudy=max(max_gradient_component_error.dudy, dudy_error),
            dvdx=max(max_gradient_component_error.dvdx, dvdx_error),
            dvdy=max(max_gradient_component_error.dvdy, dvdy_error),
        )
        max_velocity_gradient_error = max(
            max_velocity_gradient_error,
            dudx_error, dudy_error, dvdx_error, dvdy_error,
        )
        min_c_eig = min(min_c_eig, logfv_min_eig_sym2_2d(cxx, cxy, cyy))
    end
    errors = _logfv_channel_reference_errors(
        flow, tauxx_cpu, tauxy_cpu, tauyy_cpu, fx_poly_cpu, fy_poly_cpu,
        fx_total_cpu, fy_total_cpu, Float64(height_t), Float64(umax_t),
        Float64(uwall_t), Float64(lambda_t), Float64(prefactor_t), Float64(bsd_t),
    )

    return (;
        flow,
        Nx,
        Ny,
        dx=Float64(ref.dx),
        dy=Float64(ref.dy),
        height=Float64(height_t),
        width=Float64(width_t),
        umax=Float64(umax_t),
        uwall=Float64(uwall_t),
        lambda=Float64(lambda_t),
        prefactor=Float64(prefactor_t),
        beta=Float64(beta),
        Wi=Float64(Wi),
        bsd_fraction=Float64(bsd_t),
        initial,
        max_steps,
        polymer_substeps=selected_polymer_substeps,
        requested_polymer_substeps=polymer_substeps,
        subcycle_estimate,
        max_grad_norm_estimate=Float64(max_grad_norm),
        ux=Array(ux),
        uy=Array(uy),
        dudx=dudx_cpu,
        dudy=dudy_cpu,
        dvdx=dvdx_cpu,
        dvdy=dvdy_cpu,
        psixx=psixx_cpu,
        psixy=psixy_cpu,
        psiyy=psiyy_cpu,
        cxx=cxx_cpu,
        cxy=cxy_cpu,
        cyy=cyy_cpu,
        tauxx=tauxx_cpu,
        tauxy=tauxy_cpu,
        tauyy=tauyy_cpu,
        fx_poly=fx_poly_cpu,
        fy_poly=fy_poly_cpu,
        fx_total=fx_total_cpu,
        fy_total=fy_total_cpu,
        reference=ref,
        max_c_error,
        max_psi_error,
        max_velocity_gradient_error,
        max_gradient_component_error,
        min_c_eig,
        errors...,
    )
end

