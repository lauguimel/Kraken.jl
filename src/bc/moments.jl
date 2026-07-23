@kernel function _recompute_moments_row_2d!(f, ρ, ux, uy, i_fix::Int)
    j = @index(Global)
    @inbounds begin
        f1=f[i_fix,j,1]; f2=f[i_fix,j,2]; f3=f[i_fix,j,3]
        f4=f[i_fix,j,4]; f5=f[i_fix,j,5]; f6=f[i_fix,j,6]
        f7=f[i_fix,j,7]; f8=f[i_fix,j,8]; f9=f[i_fix,j,9]
        r = f1+f2+f3+f4+f5+f6+f7+f8+f9
        ρ[i_fix,j] = r
        inv_r = one(r) / r
        ux[i_fix,j] = (f2-f4+f6-f7-f8+f9) * inv_r
        uy[i_fix,j] = (f3-f5+f6+f7-f8-f9) * inv_r
    end
end

@kernel function _recompute_moments_col_2d!(f, ρ, ux, uy, j_fix::Int)
    i = @index(Global)
    @inbounds begin
        f1=f[i,j_fix,1]; f2=f[i,j_fix,2]; f3=f[i,j_fix,3]
        f4=f[i,j_fix,4]; f5=f[i,j_fix,5]; f6=f[i,j_fix,6]
        f7=f[i,j_fix,7]; f8=f[i,j_fix,8]; f9=f[i,j_fix,9]
        r = f1+f2+f3+f4+f5+f6+f7+f8+f9
        ρ[i,j_fix] = r
        inv_r = one(r) / r
        ux[i,j_fix] = (f2-f4+f6-f7-f8+f9) * inv_r
        uy[i,j_fix] = (f3-f5+f6+f7-f8-f9) * inv_r
    end
end

function _update_bc_moments_2d!(f_out, ρ, ux, uy, bcspec, Nx, Ny)
    backend = KernelAbstractions.get_backend(f_out)
    k_row = _recompute_moments_row_2d!(backend)
    k_col = _recompute_moments_col_2d!(backend)
    if !(bcspec.west isa HalfwayBB)
        k_row(f_out, ρ, ux, uy, 1; ndrange=Ny)
    end
    if !(bcspec.east isa HalfwayBB)
        k_row(f_out, ρ, ux, uy, Nx; ndrange=Ny)
    end
    if !(bcspec.south isa HalfwayBB)
        k_col(f_out, ρ, ux, uy, 1; ndrange=Nx)
    end
    if !(bcspec.north isa HalfwayBB)
        k_col(f_out, ρ, ux, uy, Ny; ndrange=Nx)
    end
    KernelAbstractions.synchronize(backend)
end

# ----------------------------------------------------------------------
# 3D face kernels
# ----------------------------------------------------------------------

