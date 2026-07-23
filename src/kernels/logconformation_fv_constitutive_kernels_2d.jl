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
    @trace_enter :poly_force
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

