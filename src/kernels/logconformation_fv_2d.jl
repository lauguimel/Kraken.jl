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

@inline function logfv_oldroydb_relax_c_2d(cxx, cxy, cyy, lambda, dt)
    decay = exp(-dt / lambda)
    return (
        one(cxx) + (cxx - one(cxx)) * decay,
        cxy * decay,
        one(cyy) + (cyy - one(cyy)) * decay,
    )
end

@inline function logfv_oldroydb_relax_log_2d(psixx, psixy, psiyy, lambda, dt)
    cxx, cxy, cyy = logfv_exp_sym2_2d(psixx, psixy, psiyy)
    rxx, rxy, ryy = logfv_oldroydb_relax_c_2d(cxx, cxy, cyy, lambda, dt)
    return logfv_log_spd_sym2_2d(rxx, rxy, ryy)
end

@inline function logfv_oldroydb_source_c_2d(cxx, cxy, cyy, dudx, dudy, dvdx, dvdy, lambda)
    inv_lambda = inv(lambda)
    return (
        2 * (cxx * dudx + cxy * dudy) - inv_lambda * (cxx - one(cxx)),
        cxx * dvdx + cyy * dudy + cxy * (dudx + dvdy) - inv_lambda * cxy,
        2 * (cxy * dvdx + cyy * dvdy) - inv_lambda * (cyy - one(cyy)),
    )
end

@inline function logfv_stress_from_log_2d(psixx, psixy, psiyy, prefactor)
    cxx, cxy, cyy = logfv_exp_sym2_2d(psixx, psixy, psiyy)
    return (
        prefactor * (cxx - one(cxx)),
        prefactor * cxy,
        prefactor * (cyy - one(cyy)),
    )
end

@inline function logfv_upwind_scalar_advective_rhs_2d(phi, ux_face, uy_face, i, j)
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

@inline function logfv_upwind_tensor_advective_rhs_2d(psixx, psixy, psiyy, ux_face, uy_face, i, j)
    return (
        logfv_upwind_scalar_advective_rhs_2d(psixx, ux_face, uy_face, i, j),
        logfv_upwind_scalar_advective_rhs_2d(psixy, ux_face, uy_face, i, j),
        logfv_upwind_scalar_advective_rhs_2d(psiyy, ux_face, uy_face, i, j),
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

@kernel function logfv_stress_from_log_2d_kernel!(
    tauxx, tauxy, tauyy,
    @Const(psixx), @Const(psixy), @Const(psiyy),
    prefactor, Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            sxx, sxy, syy = logfv_stress_from_log_2d(
                psixx[i, j], psixy[i, j], psiyy[i, j], prefactor,
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
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(tauxx)
    Nx, Ny = size(tauxx)
    kernel! = logfv_stress_from_log_2d_kernel!(backend)
    kernel!(
        tauxx, tauxy, tauyy,
        psixx, psixy, psiyy, prefactor, Nx, Ny;
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

@kernel function logfv_advect_upwind_2d_kernel!(
    psixx_out, psixy_out, psiyy_out,
    @Const(psixx), @Const(psixy), @Const(psiyy),
    @Const(ux_face), @Const(uy_face),
    dt, Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            if i > 1 && i < Nx && j > 1 && j < Ny
                rhs_xx, rhs_xy, rhs_yy = logfv_upwind_tensor_advective_rhs_2d(
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

function logfv_advect_upwind_2d!(
    psixx_out, psixy_out, psiyy_out,
    psixx, psixy, psiyy, ux_face, uy_face, dt;
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(psixx_out)
    Nx, Ny = size(psixx_out)
    kernel! = logfv_advect_upwind_2d_kernel!(backend)
    kernel!(
        psixx_out, psixy_out, psiyy_out,
        psixx, psixy, psiyy, ux_face, uy_face, dt, Nx, Ny;
        ndrange=(Nx, Ny),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end
