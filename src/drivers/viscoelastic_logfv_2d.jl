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
