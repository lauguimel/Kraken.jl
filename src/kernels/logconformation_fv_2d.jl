using KernelAbstractions

# Cell-centered log-conformation FV/FD helpers for the production polymer
# backend. These functions are scalar, allocation-free, and GPU-compatible.

@inline function logfv_min_eig_sym2_2d(a, b, d)
    m = (a + d) / 2
    h = (a - d) / 2
    return m - hypot(h, b)
end

@inline function logfv_exp_sym2_2d(a, b, d)
    T = typeof(a + b + d)
    m = (a + d) / T(2)
    h = (a - d) / T(2)
    delta = hypot(h, b)
    em = exp(m)
    delta2 = delta * delta
    scale = ifelse(delta < sqrt(eps(T)), one(T) + delta2 / T(6), sinh(delta) / delta)
    ch = cosh(delta)
    return (
        em * (ch + scale * h),
        em * scale * b,
        em * (ch - scale * h),
    )
end

@inline function logfv_exp_mat2_2d(a, b, c, d)
    T = typeof(a + b + c + d)
    m = (a + d) / T(2)
    h = (a - d) / T(2)
    disc = h * h + b * c
    em = exp(m)
    small = abs(disc) < eps(T)

    ch = if small
        one(T) + disc / T(2)
    elseif disc > zero(T)
        delta = sqrt(disc)
        cosh(delta)
    else
        theta = sqrt(-disc)
        cos(theta)
    end

    scale = if small
        one(T) + disc / T(6)
    elseif disc > zero(T)
        delta = sqrt(disc)
        sinh(delta) / delta
    else
        theta = sqrt(-disc)
        sin(theta) / theta
    end

    return (
        em * (ch + scale * h),
        em * scale * b,
        em * scale * c,
        em * (ch - scale * h),
    )
end

@inline function logfv_log_spd_sym2_2d(a, b, d)
    T = typeof(a + b + d)
    m = (a + d) / T(2)
    h = (a - d) / T(2)
    delta = hypot(h, b)
    lp = log(m + delta)
    lm = log(m - delta)
    alpha = (lp + lm) / T(2)
    beta = ifelse(
        delta < sqrt(eps(T)) * max(one(T), abs(m)),
        inv(m) + delta * delta / (T(3) * m * m * m),
        (lp - lm) / (T(2) * delta),
    )
    return (
        alpha + beta * h,
        beta * b,
        alpha - beta * h,
    )
end

const LOGFV_MODEL_OLDROYDB = UInt8(1)
const LOGFV_MODEL_FENEP = UInt8(2)

const LOGFV_BC_PERIODIC = FVFD_BC_PERIODIC
const LOGFV_BC_OPEN = FVFD_BC_OPEN
const LOGFV_BC_WALL = FVFD_BC_WALL
const LogFVDomainBC2D = FVFDDomainBC2D
const LogFVFieldBC2D = FVFDFieldBC2D
const LogFVEmbeddedBoundary2D = FVFDEmbeddedBoundary2D

const logfv_domain_bc_code = fvfd_domain_bc_code
logfv_periodicx_wally_bcspec_2d() = fvfd_periodicx_wally_bcspec_2d()
logfv_openx_wally_bcspec_2d() = fvfd_openx_wally_bcspec_2d()
logfv_wallxwally_bcspec_2d() = fvfd_wallxwally_bcspec_2d()
logfv_empty_embedded_boundary_2d(args...; kwargs...) =
    fvfd_empty_embedded_boundary_2d(args...; kwargs...)
logfv_embedded_boundary_from_qwall_2d(args...; kwargs...) =
    fvfd_embedded_boundary_from_qwall_2d(args...; kwargs...)
logfv_transfer_embedded_boundary_2d(args...; kwargs...) =
    fvfd_transfer_embedded_boundary_2d(args...; kwargs...)
logfv_transfer_field_bc_2d(args...; kwargs...) =
    fvfd_transfer_field_bc_2d(args...; kwargs...)

function logfv_constitutive_model_code(model::Symbol)
    normalized = Symbol(replace(lowercase(String(model)), '-' => '_'))
    normalized in (:oldroydb, :oldroyd_b, :ob) && return LOGFV_MODEL_OLDROYDB
    normalized in (:fenep, :fene_p, :fene_peterlin) && return LOGFV_MODEL_FENEP
    throw(ArgumentError("unsupported log-FV polymer_model=$(model); expected :oldroydb or :fenep"))
end

@inline function logfv_fenep_factor_2d(cxx, cyy, L2)
    T = typeof(cxx + cyy + L2)
    return (T(L2) - T(2)) / (T(L2) - (cxx + cyy))
end

@inline function logfv_constitutive_factor_2d(cxx, cyy, model_code, L2)
    T = typeof(cxx + cyy + L2)
    return ifelse(
        model_code == LOGFV_MODEL_FENEP,
        logfv_fenep_factor_2d(cxx, cyy, L2),
        one(T),
    )
end

@inline function logfv_constitutive_relax_c_2d(cxx, cxy, cyy, lambda, dt, model_code, L2)
    if model_code == LOGFV_MODEL_FENEP
        f = logfv_fenep_factor_2d(cxx, cyy, L2)
        decay = exp(-f * dt / lambda)
        ceq = inv(f)
        return (
            ceq + (cxx - ceq) * decay,
            cxy * decay,
            ceq + (cyy - ceq) * decay,
        )
    else
        decay = exp(-dt / lambda)
        return (
            one(cxx) + (cxx - one(cxx)) * decay,
            cxy * decay,
            one(cyy) + (cyy - one(cyy)) * decay,
        )
    end
end

@inline function logfv_oldroydb_relax_c_2d(cxx, cxy, cyy, lambda, dt)
    return logfv_constitutive_relax_c_2d(cxx, cxy, cyy, lambda, dt, LOGFV_MODEL_OLDROYDB, zero(cxx))
end

@inline function logfv_constitutive_relax_log_2d(psixx, psixy, psiyy, lambda, dt, model_code, L2)
    cxx, cxy, cyy = logfv_exp_sym2_2d(psixx, psixy, psiyy)
    rxx, rxy, ryy = logfv_constitutive_relax_c_2d(cxx, cxy, cyy, lambda, dt, model_code, L2)
    return logfv_log_spd_sym2_2d(rxx, rxy, ryy)
end

@inline function logfv_oldroydb_relax_log_2d(psixx, psixy, psiyy, lambda, dt)
    return logfv_constitutive_relax_log_2d(
        psixx, psixy, psiyy, lambda, dt, LOGFV_MODEL_OLDROYDB, zero(psixx),
    )
end

@inline function logfv_constitutive_step_log_2d(
    psixx, psixy, psiyy,
    dudx, dudy, dvdx, dvdy,
    lambda, dt, model_code, L2,
)
    cxx, cxy, cyy = logfv_exp_sym2_2d(psixx, psixy, psiyy)
    a, b, c, d = logfv_exp_mat2_2d(dt * dudx, dt * dudy, dt * dvdx, dt * dvdy)

    ac_xx = a * cxx + b * cxy
    ac_xy = a * cxy + b * cyy
    ac_yx = c * cxx + d * cxy
    ac_yy = c * cxy + d * cyy

    dxx = ac_xx * a + ac_xy * b
    dxy = ac_xx * c + ac_xy * d
    dyy = ac_yx * c + ac_yy * d
    rxx, rxy, ryy = logfv_constitutive_relax_c_2d(
        dxx, dxy, dyy, lambda, dt, model_code, L2,
    )
    return logfv_log_spd_sym2_2d(rxx, rxy, ryy)
end

@inline function logfv_oldroydb_step_log_2d(
    psixx, psixy, psiyy,
    dudx, dudy, dvdx, dvdy,
    lambda, dt,
)
    return logfv_constitutive_step_log_2d(
        psixx, psixy, psiyy,
        dudx, dudy, dvdx, dvdy,
        lambda, dt, LOGFV_MODEL_OLDROYDB, zero(psixx),
    )
end

function logfv_oldroydb_split_relax_increment(relative_tolerance::Real)
    0 < relative_tolerance < 1 ||
        throw(ArgumentError("relative_tolerance must be in (0, 1)"))

    split_error(z) = 1 - z / expm1(z)
    lo = 0.0
    hi = 1.0
    while split_error(hi) <= relative_tolerance
        lo = hi
        hi *= 2
    end
    for _ in 1:80
        mid = (lo + hi) / 2
        if split_error(mid) <= relative_tolerance
            lo = mid
        else
            hi = mid
        end
    end
    return lo
end

function logfv_oldroydb_subcycle_estimate(
    max_grad_norm::Real,
    lambda::Real,
    dt::Real=1;
    relative_tolerance::Real=0.01,
    max_deformation_increment::Real=0.05,
    max_memory_deformation_increment::Real=Inf,
    min_substeps::Integer=1,
    max_substeps::Integer=64,
)
    max_grad_norm >= 0 || throw(ArgumentError("max_grad_norm must be non-negative"))
    lambda > 0 || throw(ArgumentError("lambda must be positive"))
    dt > 0 || throw(ArgumentError("dt must be positive"))
    max_deformation_increment > 0 ||
        throw(ArgumentError("max_deformation_increment must be positive"))
    max_memory_deformation_increment > 0 ||
        throw(ArgumentError("max_memory_deformation_increment must be positive"))
    min_substeps >= 1 || throw(ArgumentError("min_substeps must be >= 1"))
    max_substeps >= min_substeps ||
        throw(ArgumentError("max_substeps must be >= min_substeps"))

    relax_increment = dt / lambda
    deformation_increment = dt * max_grad_norm
    memory_deformation_increment = lambda * max_grad_norm
    max_relax_increment = logfv_oldroydb_split_relax_increment(relative_tolerance)
    relax_substeps = max(min_substeps, ceil(Int, relax_increment / max_relax_increment))
    deformation_substeps = max(min_substeps, ceil(Int, deformation_increment / max_deformation_increment))
    memory_deformation_substeps = max(
        min_substeps,
        ceil(Int, memory_deformation_increment / max_memory_deformation_increment),
    )
    raw_substeps = max(relax_substeps, deformation_substeps, memory_deformation_substeps)
    recommended = min(raw_substeps, max_substeps)

    return (;
        recommended,
        raw_substeps,
        relax_substeps,
        deformation_substeps,
        memory_deformation_substeps,
        clamped=raw_substeps > max_substeps,
        relax_increment,
        deformation_increment,
        memory_deformation_increment,
        max_relax_increment,
        max_deformation_increment,
        max_memory_deformation_increment,
        relative_tolerance,
    )
end

function logfv_recommended_oldroydb_substeps(args...; kwargs...)
    return logfv_oldroydb_subcycle_estimate(args...; kwargs...).recommended
end

@inline function logfv_oldroydb_source_c_2d(cxx, cxy, cyy, dudx, dudy, dvdx, dvdy, lambda)
    inv_lambda = inv(lambda)
    return (
        2 * (cxx * dudx + cxy * dudy) - inv_lambda * (cxx - one(cxx)),
        cxx * dvdx + cyy * dudy + cxy * (dudx + dvdy) - inv_lambda * cxy,
        2 * (cxy * dvdx + cyy * dvdy) - inv_lambda * (cyy - one(cyy)),
    )
end

@inline function logfv_constitutive_source_c_2d(
    cxx, cxy, cyy, dudx, dudy, dvdx, dvdy, lambda, model_code, L2,
)
    inv_lambda = inv(lambda)
    f = logfv_constitutive_factor_2d(cxx, cyy, model_code, L2)
    return (
        2 * (cxx * dudx + cxy * dudy) - inv_lambda * (f * cxx - one(cxx)),
        cxx * dvdx + cyy * dudy + cxy * (dudx + dvdy) - inv_lambda * f * cxy,
        2 * (cxy * dvdx + cyy * dvdy) - inv_lambda * (f * cyy - one(cyy)),
    )
end

@inline function logfv_stress_from_log_2d(psixx, psixy, psiyy, prefactor, model_code, L2)
    cxx, cxy, cyy = logfv_exp_sym2_2d(psixx, psixy, psiyy)
    f = logfv_constitutive_factor_2d(cxx, cyy, model_code, L2)
    return (
        prefactor * (f * cxx - one(cxx)),
        prefactor * f * cxy,
        prefactor * (f * cyy - one(cyy)),
    )
end

@inline function logfv_stress_from_log_2d(psixx, psixy, psiyy, prefactor)
    return logfv_stress_from_log_2d(
        psixx, psixy, psiyy, prefactor, LOGFV_MODEL_OLDROYDB, zero(psixx),
    )
end

@inline function logfv_interior_canary_upwind_scalar_advective_rhs_2d(
    phi, ux_face, uy_face, i, j,
)
    ue = ux_face[i + 1, j]
    uw = ux_face[i, j]
    vn = uy_face[i, j + 1]
    vs = uy_face[i, j]

    phie = ifelse(ue >= 0, phi[i, j], phi[i + 1, j])
    phiw = ifelse(uw >= 0, phi[i - 1, j], phi[i, j])
    phin = ifelse(vn >= 0, phi[i, j], phi[i, j + 1])
    phis = ifelse(vs >= 0, phi[i, j - 1], phi[i, j])

    flux_div = ue * phie - uw * phiw + vn * phin - vs * phis
    divu = ue - uw + vn - vs
    return -(flux_div - phi[i, j] * divu)
end

@inline function logfv_interior_canary_upwind_tensor_advective_rhs_2d(
    psixx, psixy, psiyy, ux_face, uy_face, i, j,
)
    return (
        logfv_interior_canary_upwind_scalar_advective_rhs_2d(psixx, ux_face, uy_face, i, j),
        logfv_interior_canary_upwind_scalar_advective_rhs_2d(psixy, ux_face, uy_face, i, j),
        logfv_interior_canary_upwind_scalar_advective_rhs_2d(psiyy, ux_face, uy_face, i, j),
    )
end

@kernel function logfv_relax_log_2d_kernel!(
    psixx_out, psixy_out, psiyy_out,
    @Const(psixx), @Const(psixy), @Const(psiyy),
    lambda, dt, Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            rxx, rxy, ryy = logfv_oldroydb_relax_log_2d(
                psixx[i, j], psixy[i, j], psiyy[i, j], lambda, dt,
            )
            psixx_out[i, j] = rxx
            psixy_out[i, j] = rxy
            psiyy_out[i, j] = ryy
        end
    end
end

function logfv_relax_log_2d!(
    psixx_out, psixy_out, psiyy_out,
    psixx, psixy, psiyy, lambda, dt;
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(psixx_out)
    Nx, Ny = size(psixx_out)
    kernel! = logfv_relax_log_2d_kernel!(backend)
    kernel!(
        psixx_out, psixy_out, psiyy_out,
        psixx, psixy, psiyy, lambda, dt, Nx, Ny;
        ndrange=(Nx, Ny),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function logfv_step_oldroydb_log_2d_kernel!(
    psixx_out, psixy_out, psiyy_out,
    @Const(psixx), @Const(psixy), @Const(psiyy),
    @Const(dudx), @Const(dudy), @Const(dvdx), @Const(dvdy),
    lambda, dt, Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            rxx, rxy, ryy = logfv_oldroydb_step_log_2d(
                psixx[i, j], psixy[i, j], psiyy[i, j],
                dudx[i, j], dudy[i, j], dvdx[i, j], dvdy[i, j],
                lambda, dt,
            )
            psixx_out[i, j] = rxx
            psixy_out[i, j] = rxy
            psiyy_out[i, j] = ryy
        end
    end
end

function logfv_step_oldroydb_log_2d!(
    psixx_out, psixy_out, psiyy_out,
    psixx, psixy, psiyy,
    dudx, dudy, dvdx, dvdy,
    lambda, dt;
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(psixx_out)
    Nx, Ny = size(psixx_out)
    kernel! = logfv_step_oldroydb_log_2d_kernel!(backend)
    kernel!(
        psixx_out, psixy_out, psiyy_out,
        psixx, psixy, psiyy,
        dudx, dudy, dvdx, dvdy,
        lambda, dt, Nx, Ny;
        ndrange=(Nx, Ny),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function logfv_step_constitutive_log_2d_kernel!(
    psixx_out, psixy_out, psiyy_out,
    @Const(psixx), @Const(psixy), @Const(psiyy),
    @Const(dudx), @Const(dudy), @Const(dvdx), @Const(dvdy),
    lambda, dt, model_code, L2, Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            rxx, rxy, ryy = logfv_constitutive_step_log_2d(
                psixx[i, j], psixy[i, j], psiyy[i, j],
                dudx[i, j], dudy[i, j], dvdx[i, j], dvdy[i, j],
                lambda, dt, model_code, L2,
            )
            psixx_out[i, j] = rxx
            psixy_out[i, j] = rxy
            psiyy_out[i, j] = ryy
        end
    end
end

function logfv_step_constitutive_log_2d!(
    psixx_out, psixy_out, psiyy_out,
    psixx, psixy, psiyy,
    dudx, dudy, dvdx, dvdy,
    lambda, dt, model_code, L2;
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(psixx_out)
    Nx, Ny = size(psixx_out)
    kernel! = logfv_step_constitutive_log_2d_kernel!(backend)
    kernel!(
        psixx_out, psixy_out, psiyy_out,
        psixx, psixy, psiyy,
        dudx, dudy, dvdx, dvdy,
        lambda, dt, model_code, L2, Nx, Ny;
        ndrange=(Nx, Ny),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function logfv_stress_from_log_2d_kernel!(
    tauxx, tauxy, tauyy,
    @Const(psixx), @Const(psixy), @Const(psiyy),
    prefactor, model_code, L2, Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            sxx, sxy, syy = logfv_stress_from_log_2d(
                psixx[i, j], psixy[i, j], psiyy[i, j], prefactor, model_code, L2,
            )
            tauxx[i, j] = sxx
            tauxy[i, j] = sxy
            tauyy[i, j] = syy
        end
    end
end

function logfv_stress_from_log_2d!(
    tauxx, tauxy, tauyy,
    psixx, psixy, psiyy, prefactor;
    model_code=LOGFV_MODEL_OLDROYDB,
    L2=zero(prefactor),
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(tauxx)
    Nx, Ny = size(tauxx)
    kernel! = logfv_stress_from_log_2d_kernel!(backend)
    kernel!(
        tauxx, tauxy, tauyy,
        psixx, psixy, psiyy, prefactor, model_code, L2, Nx, Ny;
        ndrange=(Nx, Ny),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function logfv_polymer_force_centered_2d_kernel!(
    fx, fy,
    @Const(tauxx), @Const(tauxy), @Const(tauyy),
    inv_dx, inv_dy, Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            if i > 1 && i < Nx && j > 1 && j < Ny
                fx[i, j] = (tauxx[i + 1, j] - tauxx[i - 1, j]) * inv_dx / 2 +
                           (tauxy[i, j + 1] - tauxy[i, j - 1]) * inv_dy / 2
                fy[i, j] = (tauxy[i + 1, j] - tauxy[i - 1, j]) * inv_dx / 2 +
                           (tauyy[i, j + 1] - tauyy[i, j - 1]) * inv_dy / 2
            else
                fx[i, j] = zero(eltype(fx))
                fy[i, j] = zero(eltype(fy))
            end
        end
    end
end

function logfv_polymer_force_centered_2d!(
    fx, fy, tauxx, tauxy, tauyy, dx, dy;
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(fx)
    Nx, Ny = size(fx)
    kernel! = logfv_polymer_force_centered_2d_kernel!(backend)
    kernel!(
        fx, fy, tauxx, tauxy, tauyy, inv(dx), inv(dy), Nx, Ny;
        ndrange=(Nx, Ny),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function logfv_polymer_force_solid_aware_2d_kernel!(
    fx, fy,
    @Const(tauxx), @Const(tauxy), @Const(tauyy), @Const(is_solid),
    inv_dx, inv_dy, inv_2dx, inv_2dy, Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            if is_solid[i, j]
                fx[i, j] = zero(eltype(fx))
                fy[i, j] = zero(eltype(fy))
            else
                fx[i, j] = _logfv_solid_aware_derivative_x_2d(tauxx, is_solid, i, j, Nx, inv_dx, inv_2dx) +
                           _logfv_solid_aware_derivative_y_2d(tauxy, is_solid, i, j, Ny, inv_dy, inv_2dy)
                fy[i, j] = _logfv_solid_aware_derivative_x_2d(tauxy, is_solid, i, j, Nx, inv_dx, inv_2dx) +
                           _logfv_solid_aware_derivative_y_2d(tauyy, is_solid, i, j, Ny, inv_dy, inv_2dy)
            end
        end
    end
end

function logfv_polymer_force_solid_aware_2d!(
    fx, fy, tauxx, tauxy, tauyy, is_solid, dx, dy;
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(fx)
    Nx, Ny = size(fx)
    kernel! = logfv_polymer_force_solid_aware_2d_kernel!(backend)
    kernel!(
        fx, fy, tauxx, tauxy, tauyy, is_solid,
        inv(dx), inv(dy), inv(2 * dx), inv(2 * dy), Nx, Ny;
        ndrange=(Nx, Ny),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function logfv_bsd_correct_force_centered_2d_kernel!(
    fx_out, fy_out,
    @Const(fx_poly), @Const(fy_poly),
    @Const(ux), @Const(uy),
    zeta_nu_p, inv_dx2, inv_dy2, Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            if i > 1 && i < Nx && j > 1 && j < Ny
                lap_ux = (ux[i + 1, j] - 2 * ux[i, j] + ux[i - 1, j]) * inv_dx2 +
                         (ux[i, j + 1] - 2 * ux[i, j] + ux[i, j - 1]) * inv_dy2
                lap_uy = (uy[i + 1, j] - 2 * uy[i, j] + uy[i - 1, j]) * inv_dx2 +
                         (uy[i, j + 1] - 2 * uy[i, j] + uy[i, j - 1]) * inv_dy2
                fx_out[i, j] = fx_poly[i, j] - zeta_nu_p * lap_ux
                fy_out[i, j] = fy_poly[i, j] - zeta_nu_p * lap_uy
            else
                fx_out[i, j] = fx_poly[i, j]
                fy_out[i, j] = fy_poly[i, j]
            end
        end
    end
end

function logfv_bsd_correct_force_centered_2d!(
    fx_out, fy_out, fx_poly, fy_poly, ux, uy, zeta, nu_p, dx, dy;
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(fx_out)
    Nx, Ny = size(fx_out)
    kernel! = logfv_bsd_correct_force_centered_2d_kernel!(backend)
    kernel!(
        fx_out, fy_out, fx_poly, fy_poly, ux, uy,
        zeta * nu_p, inv(dx * dx), inv(dy * dy), Nx, Ny;
        ndrange=(Nx, Ny),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function logfv_bsd_correct_force_solid_aware_2d_kernel!(
    fx_out, fy_out,
    @Const(fx_poly), @Const(fy_poly),
    @Const(ux), @Const(uy), @Const(is_solid),
    zeta_nu_p, inv_dx2, inv_dy2, Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            if is_solid[i, j]
                fx_out[i, j] = zero(eltype(fx_out))
                fy_out[i, j] = zero(eltype(fy_out))
            else
                lap_ux = _logfv_solid_aware_second_derivative_x_2d(ux, is_solid, i, j, Nx, inv_dx2) +
                         _logfv_solid_aware_second_derivative_y_2d(ux, is_solid, i, j, Ny, inv_dy2)
                lap_uy = _logfv_solid_aware_second_derivative_x_2d(uy, is_solid, i, j, Nx, inv_dx2) +
                         _logfv_solid_aware_second_derivative_y_2d(uy, is_solid, i, j, Ny, inv_dy2)
                fx_out[i, j] = fx_poly[i, j] - zeta_nu_p * lap_ux
                fy_out[i, j] = fy_poly[i, j] - zeta_nu_p * lap_uy
            end
        end
    end
end

function logfv_bsd_correct_force_solid_aware_2d!(
    fx_out, fy_out, fx_poly, fy_poly, ux, uy, is_solid, zeta, nu_p, dx, dy;
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(fx_out)
    Nx, Ny = size(fx_out)
    kernel! = logfv_bsd_correct_force_solid_aware_2d_kernel!(backend)
    kernel!(
        fx_out, fy_out, fx_poly, fy_poly, ux, uy, is_solid,
        zeta * nu_p, inv(dx * dx), inv(dy * dy), Nx, Ny;
        ndrange=(Nx, Ny),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

function logfv_polymer_force_bc_aware_2d!(
    fx, fy, tauxx, tauxy, tauyy, is_solid, dx, dy, bc::LogFVDomainBC2D;
    sync::Bool=true,
    polymer_wall_extrap::Symbol=:quadratic,
)
    return fvfd_tensor_divergence_2d!(
        fx, fy, tauxx, tauxy, tauyy, is_solid, dx, dy, bc;
        sync=sync, polymer_wall_extrap=polymer_wall_extrap,
    )
end

function logfv_polymer_force_embedded_bc_aware_2d!(
    fx, fy, tauxx, tauxy, tauyy, geometry::FVFDGeometry2D;
    sync::Bool=true,
)
    return fvfd_tensor_divergence_embedded_2d!(
        fx, fy, tauxx, tauxy, tauyy, geometry; sync,
    )
end

function logfv_embedded_wall_traction_2d!(
    tx, ty, tauxx, tauxy, tauyy, geometry::FVFDGeometry2D;
    sync::Bool=true,
)
    return fvfd_embedded_wall_traction_2d!(
        tx, ty, tauxx, tauxy, tauyy, geometry; sync,
    )
end

@kernel function logfv_bsd_stress_from_gradient_2d_kernel!(
    tau_bsd_xx, tau_bsd_xy, tau_bsd_yy,
    @Const(dudx), @Const(dudy), @Const(dvdx), @Const(dvdy),
    zeta_nu_p, Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            tau_bsd_xx[i, j] = 2 * zeta_nu_p * dudx[i, j]
            tau_bsd_xy[i, j] = zeta_nu_p * (dudy[i, j] + dvdx[i, j])
            tau_bsd_yy[i, j] = 2 * zeta_nu_p * dvdy[i, j]
        end
    end
end

function logfv_bsd_stress_from_gradient_2d!(
    tau_bsd_xx, tau_bsd_xy, tau_bsd_yy,
    dudx, dudy, dvdx, dvdy, zeta_nu_p;
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(tau_bsd_xx)
    Nx, Ny = size(tau_bsd_xx)
    kernel! = logfv_bsd_stress_from_gradient_2d_kernel!(backend)
    kernel!(
        tau_bsd_xx, tau_bsd_xy, tau_bsd_yy,
        dudx, dudy, dvdx, dvdy, zeta_nu_p, Nx, Ny;
        ndrange=(Nx, Ny),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

function logfv_bsd_correct_force_bc_aware_2d!(
    fx_out, fy_out, fx_poly, fy_poly, ux, uy, is_solid, zeta, nu_p, dx, dy,
    bc::LogFVDomainBC2D;
    sync::Bool=true,
)
    return fvfd_bsd_force_2d!(
        fx_out, fy_out, fx_poly, fy_poly, ux, uy, is_solid,
        zeta, nu_p, dx, dy, bc; sync,
    )
end

@kernel function logfv_velocity_gradient_centered_2d_kernel!(
    dudx, dudy, dvdx, dvdy,
    @Const(ux), @Const(uy),
    inv_2dx, inv_2dy, Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            if i > 1 && i < Nx && j > 1 && j < Ny
                dudx[i, j] = (ux[i + 1, j] - ux[i - 1, j]) * inv_2dx
                dudy[i, j] = (ux[i, j + 1] - ux[i, j - 1]) * inv_2dy
                dvdx[i, j] = (uy[i + 1, j] - uy[i - 1, j]) * inv_2dx
                dvdy[i, j] = (uy[i, j + 1] - uy[i, j - 1]) * inv_2dy
            else
                dudx[i, j] = zero(eltype(dudx))
                dudy[i, j] = zero(eltype(dudy))
                dvdx[i, j] = zero(eltype(dvdx))
                dvdy[i, j] = zero(eltype(dvdy))
            end
        end
    end
end

function logfv_velocity_gradient_centered_2d!(
    dudx, dudy, dvdx, dvdy,
    ux, uy, dx, dy;
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(ux)
    Nx, Ny = size(ux)
    kernel! = logfv_velocity_gradient_centered_2d_kernel!(backend)
    kernel!(
        dudx, dudy, dvdx, dvdy,
        ux, uy, inv(2 * dx), inv(2 * dy), Nx, Ny;
        ndrange=(Nx, Ny),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function logfv_velocity_gradient_periodicx_wally_2d_kernel!(
    dudx, dudy, dvdx, dvdy,
    @Const(ux), @Const(uy),
    inv_2dx, inv_2dy, Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            im = ifelse(i > 1, i - 1, Nx)
            ip = ifelse(i < Nx, i + 1, 1)
            dudx[i, j] = (ux[ip, j] - ux[im, j]) * inv_2dx
            dvdx[i, j] = (uy[ip, j] - uy[im, j]) * inv_2dx

            if j == 1
                dudy[i, j] = (-3 * ux[i, j] + 4 * ux[i, j + 1] - ux[i, j + 2]) * inv_2dy
                dvdy[i, j] = (-3 * uy[i, j] + 4 * uy[i, j + 1] - uy[i, j + 2]) * inv_2dy
            elseif j == Ny
                dudy[i, j] = (3 * ux[i, j] - 4 * ux[i, j - 1] + ux[i, j - 2]) * inv_2dy
                dvdy[i, j] = (3 * uy[i, j] - 4 * uy[i, j - 1] + uy[i, j - 2]) * inv_2dy
            else
                dudy[i, j] = (ux[i, j + 1] - ux[i, j - 1]) * inv_2dy
                dvdy[i, j] = (uy[i, j + 1] - uy[i, j - 1]) * inv_2dy
            end
        end
    end
end

function logfv_velocity_gradient_periodicx_wally_2d!(
    dudx, dudy, dvdx, dvdy,
    ux, uy, dx, dy;
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(ux)
    Nx, Ny = size(ux)
    Ny >= 3 || throw(ArgumentError("wall-y velocity gradient requires Ny >= 3"))
    kernel! = logfv_velocity_gradient_periodicx_wally_2d_kernel!(backend)
    kernel!(
        dudx, dudy, dvdx, dvdy,
        ux, uy, inv(2 * dx), inv(2 * dy), Nx, Ny;
        ndrange=(Nx, Ny),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

@inline function _logfv_solid_aware_derivative_x_2d(field, is_solid, i, j, Nx, inv_dx, inv_2dx)
    T = eltype(field)
    left = i > 1 && !is_solid[i - 1, j]
    right = i < Nx && !is_solid[i + 1, j]
    if left && right
        return (field[i + 1, j] - field[i - 1, j]) * inv_2dx
    elseif right
        right2 = i + 2 <= Nx && !is_solid[i + 2, j]
        return right2 ?
               (-T(3) * field[i, j] + T(4) * field[i + 1, j] - field[i + 2, j]) * inv_2dx :
               (field[i + 1, j] - field[i, j]) * inv_dx
    elseif left
        left2 = i - 2 >= 1 && !is_solid[i - 2, j]
        return left2 ?
               (T(3) * field[i, j] - T(4) * field[i - 1, j] + field[i - 2, j]) * inv_2dx :
               (field[i, j] - field[i - 1, j]) * inv_dx
    else
        return zero(T)
    end
end

@inline function _logfv_solid_aware_derivative_y_2d(field, is_solid, i, j, Ny, inv_dy, inv_2dy)
    T = eltype(field)
    down = j > 1 && !is_solid[i, j - 1]
    up = j < Ny && !is_solid[i, j + 1]
    if down && up
        return (field[i, j + 1] - field[i, j - 1]) * inv_2dy
    elseif up
        up2 = j + 2 <= Ny && !is_solid[i, j + 2]
        return up2 ?
               (-T(3) * field[i, j] + T(4) * field[i, j + 1] - field[i, j + 2]) * inv_2dy :
               (field[i, j + 1] - field[i, j]) * inv_dy
    elseif down
        down2 = j - 2 >= 1 && !is_solid[i, j - 2]
        return down2 ?
               (T(3) * field[i, j] - T(4) * field[i, j - 1] + field[i, j - 2]) * inv_2dy :
               (field[i, j] - field[i, j - 1]) * inv_dy
    else
        return zero(T)
    end
end

@inline function _logfv_solid_aware_second_derivative_x_2d(field, is_solid, i, j, Nx, inv_dx2)
    T = eltype(field)
    left = i > 1 && !is_solid[i - 1, j]
    right = i < Nx && !is_solid[i + 1, j]
    if left && right
        return (field[i + 1, j] - T(2) * field[i, j] + field[i - 1, j]) * inv_dx2
    elseif right && i + 2 <= Nx && !is_solid[i + 2, j]
        return (field[i, j] - T(2) * field[i + 1, j] + field[i + 2, j]) * inv_dx2
    elseif left && i - 2 >= 1 && !is_solid[i - 2, j]
        return (field[i, j] - T(2) * field[i - 1, j] + field[i - 2, j]) * inv_dx2
    else
        return zero(T)
    end
end

@inline function _logfv_solid_aware_second_derivative_y_2d(field, is_solid, i, j, Ny, inv_dy2)
    T = eltype(field)
    down = j > 1 && !is_solid[i, j - 1]
    up = j < Ny && !is_solid[i, j + 1]
    if down && up
        return (field[i, j + 1] - T(2) * field[i, j] + field[i, j - 1]) * inv_dy2
    elseif up && j + 2 <= Ny && !is_solid[i, j + 2]
        return (field[i, j] - T(2) * field[i, j + 1] + field[i, j + 2]) * inv_dy2
    elseif down && j - 2 >= 1 && !is_solid[i, j - 2]
        return (field[i, j] - T(2) * field[i, j - 1] + field[i, j - 2]) * inv_dy2
    else
        return zero(T)
    end
end

@kernel function logfv_velocity_gradient_solid_aware_2d_kernel!(
    dudx, dudy, dvdx, dvdy,
    @Const(ux), @Const(uy), @Const(is_solid),
    inv_dx, inv_dy, inv_2dx, inv_2dy, Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            if is_solid[i, j]
                dudx[i, j] = zero(eltype(dudx))
                dudy[i, j] = zero(eltype(dudy))
                dvdx[i, j] = zero(eltype(dvdx))
                dvdy[i, j] = zero(eltype(dvdy))
            else
                dudx[i, j] = _logfv_solid_aware_derivative_x_2d(ux, is_solid, i, j, Nx, inv_dx, inv_2dx)
                dudy[i, j] = _logfv_solid_aware_derivative_y_2d(ux, is_solid, i, j, Ny, inv_dy, inv_2dy)
                dvdx[i, j] = _logfv_solid_aware_derivative_x_2d(uy, is_solid, i, j, Nx, inv_dx, inv_2dx)
                dvdy[i, j] = _logfv_solid_aware_derivative_y_2d(uy, is_solid, i, j, Ny, inv_dy, inv_2dy)
            end
        end
    end
end

function logfv_velocity_gradient_solid_aware_2d!(
    dudx, dudy, dvdx, dvdy,
    ux, uy, is_solid, dx, dy;
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(ux)
    Nx, Ny = size(ux)
    kernel! = logfv_velocity_gradient_solid_aware_2d_kernel!(backend)
    kernel!(
        dudx, dudy, dvdx, dvdy,
        ux, uy, is_solid, inv(dx), inv(dy), inv(2 * dx), inv(2 * dy), Nx, Ny;
        ndrange=(Nx, Ny),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

function logfv_velocity_gradient_bc_aware_2d!(
    dudx, dudy, dvdx, dvdy,
    ux, uy, is_solid, dx, dy, bc::LogFVDomainBC2D;
    sync::Bool=true,
)
    return fvfd_velocity_gradient_2d!(
        dudx, dudy, dvdx, dvdy, ux, uy, is_solid, dx, dy, bc; sync,
    )
end

function logfv_velocity_gradient_embedded_bc_aware_2d!(
    dudx, dudy, dvdx, dvdy,
    ux, uy, is_solid, dx, dy, bc::LogFVDomainBC2D,
    embedded::LogFVEmbeddedBoundary2D;
    sync::Bool=true,
)
    return fvfd_velocity_gradient_embedded_2d!(
        dudx, dudy, dvdx, dvdy, ux, uy, is_solid, dx, dy, bc, embedded; sync,
    )
end

@kernel function logfv_fill_nearest_boundary_2d_kernel!(fx, fy, Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            if i == 1 || i == Nx || j == 1 || j == Ny
                ii = min(max(i, 2), Nx - 1)
                jj = min(max(j, 2), Ny - 1)
                fx[i, j] = fx[ii, jj]
                fy[i, j] = fy[ii, jj]
            end
        end
    end
end

function logfv_fill_nearest_boundary_2d!(fx, fy; sync::Bool=true)
    backend = KernelAbstractions.get_backend(fx)
    Nx, Ny = size(fx)
    Nx >= 3 && Ny >= 3 || throw(ArgumentError("nearest boundary fill requires at least 3x3 cells"))
    kernel! = logfv_fill_nearest_boundary_2d_kernel!(backend)
    kernel!(fx, fy, Nx, Ny; ndrange=(Nx, Ny))
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function logfv_add_constant_force_2d_kernel!(fx, fy, Fx, Fy, Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            T = eltype(fx)
            fx[i, j] += T(Fx)
            fy[i, j] += T(Fy)
        end
    end
end

function logfv_add_constant_force_2d!(fx, fy, Fx, Fy; sync::Bool=true)
    backend = KernelAbstractions.get_backend(fx)
    Nx, Ny = size(fx)
    kernel! = logfv_add_constant_force_2d_kernel!(backend)
    kernel!(fx, fy, Fx, Fy, Nx, Ny; ndrange=(Nx, Ny))
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function logfv_add_constant_force_fluid_2d_kernel!(
    fx, fy, @Const(is_solid), Fx, Fy, Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny && !is_solid[i, j]
            T = eltype(fx)
            fx[i, j] += T(Fx)
            fy[i, j] += T(Fy)
        end
    end
end

function logfv_add_constant_force_fluid_2d!(fx, fy, is_solid, Fx, Fy; sync::Bool=true)
    backend = KernelAbstractions.get_backend(fx)
    Nx, Ny = size(fx)
    kernel! = logfv_add_constant_force_fluid_2d_kernel!(backend)
    kernel!(fx, fy, is_solid, Fx, Fy, Nx, Ny; ndrange=(Nx, Ny))
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function logfv_copy_column_profile_2d_kernel!(profile, @Const(field), column, Nx, Ny)
    j = @index(Global)
    @inbounds begin
        if j <= Ny
            i = min(max(column, 1), Nx)
            profile[j] = field[i, j]
        end
    end
end

function logfv_copy_column_profile_2d!(profile, field, column; sync::Bool=true)
    backend = KernelAbstractions.get_backend(field)
    Nx, Ny = size(field)
    kernel! = logfv_copy_column_profile_2d_kernel!(backend)
    kernel!(profile, field, column, Nx, Ny; ndrange=Ny)
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function logfv_compute_macroscopic_forced_field_2d_kernel!(
    rho, ux, uy,
    @Const(f), @Const(fx), @Const(fy),
    Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            T = eltype(f)
            f1 = f[i, j, 1]
            f2 = f[i, j, 2]
            f3 = f[i, j, 3]
            f4 = f[i, j, 4]
            f5 = f[i, j, 5]
            f6 = f[i, j, 6]
            f7 = f[i, j, 7]
            f8 = f[i, j, 8]
            f9 = f[i, j, 9]

            rho_local = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
            inv_rho = one(T) / rho_local
            rho[i, j] = rho_local
            ux[i, j] = (f2 - f4 + f6 - f7 - f8 + f9 + fx[i, j] / T(2)) * inv_rho
            uy[i, j] = (f3 - f5 + f6 + f7 - f8 - f9 + fy[i, j] / T(2)) * inv_rho
        end
    end
end

function logfv_compute_macroscopic_forced_field_2d!(
    rho, ux, uy, f, fx, fy;
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(rho)
    kernel! = logfv_compute_macroscopic_forced_field_2d_kernel!(backend)
    kernel!(rho, ux, uy, f, fx, fy, Nx, Ny; ndrange=(Nx, Ny))
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function logfv_advect_upwind_interior_canary_2d_kernel!(
    psixx_out, psixy_out, psiyy_out,
    @Const(psixx), @Const(psixy), @Const(psiyy),
    @Const(ux_face), @Const(uy_face),
    dt, Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            if i > 1 && i < Nx && j > 1 && j < Ny
                rhs_xx, rhs_xy, rhs_yy = logfv_interior_canary_upwind_tensor_advective_rhs_2d(
                    psixx, psixy, psiyy, ux_face, uy_face, i, j,
                )
                psixx_out[i, j] = psixx[i, j] + dt * rhs_xx
                psixy_out[i, j] = psixy[i, j] + dt * rhs_xy
                psiyy_out[i, j] = psiyy[i, j] + dt * rhs_yy
            else
                psixx_out[i, j] = psixx[i, j]
                psixy_out[i, j] = psixy[i, j]
                psiyy_out[i, j] = psiyy[i, j]
            end
        end
    end
end

function logfv_advect_upwind_interior_canary_2d!(
    psixx_out, psixy_out, psiyy_out,
    psixx, psixy, psiyy, ux_face, uy_face, dt;
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(psixx_out)
    Nx, Ny = size(psixx_out)
    kernel! = logfv_advect_upwind_interior_canary_2d_kernel!(backend)
    kernel!(
        psixx_out, psixy_out, psiyy_out,
        psixx, psixy, psiyy, ux_face, uy_face, dt, Nx, Ny;
        ndrange=(Nx, Ny),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

function logfv_cell_velocity_to_faces_bc_aware_2d!(
    ux_face, uy_face, ux, uy, is_solid,
    ux_west, ux_east, uy_south, uy_north,
    bc::LogFVDomainBC2D;
    sync::Bool=true,
)
    return fvfd_cell_velocity_to_faces_2d!(
        ux_face, uy_face, ux, uy, is_solid,
        ux_west, ux_east, uy_south, uy_north,
        bc; sync,
    )
end

function logfv_cell_velocity_to_faces_embedded_2d!(
    ux_face, uy_face, ux, uy, geometry::FVFDGeometry2D,
    ux_bc::FVFDFieldBC2D, uy_bc::FVFDFieldBC2D;
    sync::Bool=true,
)
    return fvfd_cell_velocity_to_faces_embedded_2d!(
        ux_face, uy_face, ux, uy, geometry, ux_bc, uy_bc; sync,
    )
end

function logfv_cell_velocity_to_faces_solid_aware_2d!(
    ux_face, uy_face, ux, uy, is_solid;
    sync::Bool=true,
)
    return logfv_cell_velocity_to_faces_bc_aware_2d!(
        ux_face, uy_face, ux, uy, is_solid,
        ux, ux, uy, uy,
        logfv_periodicx_wally_bcspec_2d();
        sync,
    )
end

function logfv_cell_velocity_to_faces_openx_solid_aware_2d!(
    ux_face, uy_face, ux, uy, is_solid, ux_west, ux_east;
    sync::Bool=true,
)
    return logfv_cell_velocity_to_faces_bc_aware_2d!(
        ux_face, uy_face, ux, uy, is_solid,
        ux_west, ux_east, uy, uy,
        logfv_openx_wally_bcspec_2d();
        sync,
    )
end

function logfv_advect_upwind_bc_aware_2d!(
    psixx_out, psixy_out, psiyy_out,
    psixx, psixy, psiyy,
    west_xx, west_xy, west_yy,
    east_xx, east_xy, east_yy,
    south_xx, south_xy, south_yy,
    north_xx, north_xy, north_yy,
    ux_face, uy_face, is_solid,
    dx, dy, bc::LogFVDomainBC2D, dt;
    sync::Bool=true,
    advection_scheme::Symbol=:rusanov,
)
    return fvfd_sym2_advect_upwind_2d!(
        psixx_out, psixy_out, psiyy_out,
        psixx, psixy, psiyy,
        FVFDFieldBC2D(west_xx, east_xx, south_xx, north_xx),
        FVFDFieldBC2D(west_xy, east_xy, south_xy, north_xy),
        FVFDFieldBC2D(west_yy, east_yy, south_yy, north_yy),
        ux_face, uy_face, is_solid, dx, dy, bc, dt; sync, advection_scheme,
    )
end

function logfv_advect_upwind_bc_aware_2d!(
    psixx_out, psixy_out, psiyy_out,
    psixx, psixy, psiyy,
    west_xx, west_xy, west_yy,
    east_xx, east_xy, east_yy,
    south_xx, south_xy, south_yy,
    north_xx, north_xy, north_yy,
    ux_face, uy_face, geometry::FVFDGeometry2D, dt;
    sync::Bool=true,
    advection_scheme::Symbol=:rusanov,
)
    return logfv_advect_upwind_bc_aware_2d!(
        psixx_out, psixy_out, psiyy_out,
        psixx, psixy, psiyy,
        west_xx, west_xy, west_yy,
        east_xx, east_xy, east_yy,
        south_xx, south_xy, south_yy,
        north_xx, north_xy, north_yy,
        ux_face, uy_face, geometry.is_solid,
        geometry.patch.dx, geometry.patch.dy, geometry.bc, dt;
        sync,
        advection_scheme,
    )
end

function logfv_advect_upwind_bc_aware_2d!(
    psixx_out, psixy_out, psiyy_out,
    psixx, psixy, psiyy,
    west_xx, west_xy, west_yy,
    east_xx, east_xy, east_yy,
    south_xx, south_xy, south_yy,
    north_xx, north_xy, north_yy,
    ux_face, uy_face, is_solid,
    bc::LogFVDomainBC2D, dt;
    sync::Bool=true,
    advection_scheme::Symbol=:rusanov,
)
    spacing = one(eltype(psixx_out))
    return logfv_advect_upwind_bc_aware_2d!(
        psixx_out, psixy_out, psiyy_out,
        psixx, psixy, psiyy,
        west_xx, west_xy, west_yy,
        east_xx, east_xy, east_yy,
        south_xx, south_xy, south_yy,
        north_xx, north_xy, north_yy,
        ux_face, uy_face, is_solid,
        spacing, spacing, bc, dt;
        sync,
        advection_scheme,
    )
end

function logfv_advect_upwind_embedded_2d!(
    psixx_out, psixy_out, psiyy_out,
    psixx, psixy, psiyy,
    psixx_bc::FVFDFieldBC2D, psixy_bc::FVFDFieldBC2D, psiyy_bc::FVFDFieldBC2D,
    ux_face, uy_face, ux, uy,
    geometry::FVFDGeometry2D,
    ux_bc::FVFDFieldBC2D, uy_bc::FVFDFieldBC2D,
    dt;
    sync::Bool=true,
    advection_scheme::Symbol=:rusanov,
)
    return fvfd_sym2_advect_upwind_embedded_2d!(
        psixx_out, psixy_out, psiyy_out,
        psixx, psixy, psiyy,
        psixx_bc, psixy_bc, psiyy_bc,
        ux_face, uy_face, ux, uy,
        geometry, ux_bc, uy_bc, dt; sync, advection_scheme,
    )
end

function logfv_advect_upwind_solid_aware_2d!(
    psixx_out, psixy_out, psiyy_out,
    psixx, psixy, psiyy, ux_face, uy_face, is_solid, dt;
    sync::Bool=true,
    advection_scheme::Symbol=:rusanov,
)
    return logfv_advect_upwind_bc_aware_2d!(
        psixx_out, psixy_out, psiyy_out,
        psixx, psixy, psiyy,
        psixx, psixy, psiyy,
        psixx, psixy, psiyy,
        psixx, psixy, psiyy,
        psixx, psixy, psiyy,
        ux_face, uy_face, is_solid,
        logfv_periodicx_wally_bcspec_2d(), dt;
        sync,
        advection_scheme,
    )
end

function logfv_advect_upwind_openx_solid_aware_2d!(
    psixx_out, psixy_out, psiyy_out,
    psixx, psixy, psiyy,
    west_xx, west_xy, west_yy,
    east_xx, east_xy, east_yy,
    ux_face, uy_face, is_solid, dt;
    sync::Bool=true,
    advection_scheme::Symbol=:rusanov,
)
    return logfv_advect_upwind_bc_aware_2d!(
        psixx_out, psixy_out, psiyy_out,
        psixx, psixy, psiyy,
        west_xx, west_xy, west_yy,
        east_xx, east_xy, east_yy,
        psixx, psixy, psiyy,
        psixx, psixy, psiyy,
        ux_face, uy_face, is_solid,
        logfv_openx_wally_bcspec_2d(), dt;
        sync,
        advection_scheme,
    )
end
