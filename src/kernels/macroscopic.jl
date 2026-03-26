using KernelAbstractions

# --- 2D macroscopic computation ---

@kernel function compute_macroscopic_2d_kernel!(ρ, ux, uy, @Const(f))
    i, j = @index(Global, NTuple)
    @inbounds begin
        f1 = f[i,j,1]; f2 = f[i,j,2]; f3 = f[i,j,3]
        f4 = f[i,j,4]; f5 = f[i,j,5]; f6 = f[i,j,6]
        f7 = f[i,j,7]; f8 = f[i,j,8]; f9 = f[i,j,9]

        ρ_local = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
        inv_ρ = one(ρ_local) / ρ_local
        ρ[i,j] = ρ_local
        ux[i,j] = (f2 - f4 + f6 - f7 - f8 + f9) * inv_ρ
        uy[i,j] = (f3 - f5 + f6 + f7 - f8 - f9) * inv_ρ
    end
end

function compute_macroscopic_2d!(ρ, ux, uy, f)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(ρ)
    kernel! = compute_macroscopic_2d_kernel!(backend)
    kernel!(ρ, ux, uy, f; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

# --- 3D macroscopic computation ---

@kernel function compute_macroscopic_3d_kernel!(ρ, ux, uy, uz, @Const(f))
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        f1  = f[i,j,k,1];  f2  = f[i,j,k,2];  f3  = f[i,j,k,3]
        f4  = f[i,j,k,4];  f5  = f[i,j,k,5];  f6  = f[i,j,k,6]
        f7  = f[i,j,k,7];  f8  = f[i,j,k,8];  f9  = f[i,j,k,9]
        f10 = f[i,j,k,10]; f11 = f[i,j,k,11]; f12 = f[i,j,k,12]
        f13 = f[i,j,k,13]; f14 = f[i,j,k,14]; f15 = f[i,j,k,15]
        f16 = f[i,j,k,16]; f17 = f[i,j,k,17]; f18 = f[i,j,k,18]
        f19 = f[i,j,k,19]

        ρ_local = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10 +
                  f11 + f12 + f13 + f14 + f15 + f16 + f17 + f18 + f19
        inv_ρ = one(ρ_local) / ρ_local
        ρ[i,j,k] = ρ_local
        ux[i,j,k] = (f2 - f3 + f8 - f9 + f10 - f11 + f12 - f13 + f14 - f15) * inv_ρ
        uy[i,j,k] = (f4 - f5 + f8 + f9 - f10 - f11 + f16 - f17 + f18 - f19) * inv_ρ
        uz[i,j,k] = (f6 - f7 + f12 + f13 - f14 - f15 + f16 + f17 - f18 - f19) * inv_ρ
    end
end

function compute_macroscopic_3d!(ρ, ux, uy, uz, f)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny, Nz = size(ρ)
    kernel! = compute_macroscopic_3d_kernel!(backend)
    kernel!(ρ, ux, uy, uz, f; ndrange=(Nx, Ny, Nz))
    KernelAbstractions.synchronize(backend)
end
