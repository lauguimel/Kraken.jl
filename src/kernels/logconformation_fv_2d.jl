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

include("logconformation_fv_constitutive_kernels_2d.jl")
include("logconformation_fv_velocity_gradient_2d.jl")
include("logconformation_fv_forcing_utils_2d.jl")
include("logconformation_fv_advection_wrappers_2d.jl")
