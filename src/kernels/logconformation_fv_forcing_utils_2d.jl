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
            # Convention I (integrated): collide_guo_field_2d! already advances
            # the post-collision raw momentum by F; no +F/2 readout correction.
            ux[i, j] = (f2 - f4 + f6 - f7 - f8 + f9) * inv_rho
            uy[i, j] = (f3 - f5 + f6 + f7 - f8 - f9) * inv_rho
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

