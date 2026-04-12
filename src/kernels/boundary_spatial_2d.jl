using KernelAbstractions

# --- Zou-He spatial velocity BC on north wall (j = Ny) ---
# Corner nodes (i=1, i=Nx) are skipped — handled by wall bounce-back in streaming.

@kernel function zou_he_velocity_north_spatial_2d_kernel!(f, @Const(ux_arr), @Const(uy_arr), Ny, Nx)
    i = @index(Global)
    j = Ny

    @inbounds if i > 1 && i < Nx
        T = eltype(f)
        u_x = T(ux_arr[i])
        u_y = T(uy_arr[i])

        f1 = f[i, j, 1]; f2 = f[i, j, 2]; f3 = f[i, j, 3]
        f4 = f[i, j, 4]; f6 = f[i, j, 6]; f7 = f[i, j, 7]

        ρ_wall = (f1 + f2 + f4 + T(2) * (f3 + f6 + f7)) / (one(T) + u_y)

        f[i, j, 5] = f3 - T(2.0/3.0) * ρ_wall * u_y
        f[i, j, 9] = f7 - T(0.5) * (f2 - f4) + T(0.5) * ρ_wall * u_x - T(1.0/6.0) * ρ_wall * u_y
        f[i, j, 8] = f6 + T(0.5) * (f2 - f4) - T(0.5) * ρ_wall * u_x - T(1.0/6.0) * ρ_wall * u_y
    end
end

"""
    apply_zou_he_north_spatial_2d!(f, ux_arr, uy_arr, Nx, Ny)

Zou-He velocity BC on north wall with per-node velocity arrays.
"""
function apply_zou_he_north_spatial_2d!(f, ux_arr, uy_arr, Nx, Ny)
    backend = KernelAbstractions.get_backend(f)
    kernel! = zou_he_velocity_north_spatial_2d_kernel!(backend)
    kernel!(f, ux_arr, uy_arr, Ny, Nx; ndrange=(Nx,))
    KernelAbstractions.synchronize(backend)
end

# --- Zou-He spatial velocity BC on south wall (j = 1) ---

@kernel function zou_he_velocity_south_spatial_2d_kernel!(f, @Const(ux_arr), @Const(uy_arr), Nx)
    i = @index(Global)
    j = 1

    @inbounds if i > 1 && i < Nx
        T = eltype(f)
        u_x = T(ux_arr[i])
        u_y = T(uy_arr[i])

        f1 = f[i, j, 1]; f2 = f[i, j, 2]; f4 = f[i, j, 4]
        f5 = f[i, j, 5]; f8 = f[i, j, 8]; f9 = f[i, j, 9]

        ρ_wall = (f1 + f2 + f4 + T(2) * (f5 + f8 + f9)) / (one(T) - u_y)

        f[i, j, 3] = f5 + T(2.0/3.0) * ρ_wall * u_y
        f[i, j, 6] = f8 - T(0.5) * (f2 - f4) + T(0.5) * ρ_wall * u_x + T(1.0/6.0) * ρ_wall * u_y
        f[i, j, 7] = f9 + T(0.5) * (f2 - f4) - T(0.5) * ρ_wall * u_x + T(1.0/6.0) * ρ_wall * u_y
    end
end

"""
    apply_zou_he_south_spatial_2d!(f, ux_arr, uy_arr, Nx)

Zou-He velocity BC on south wall with per-node velocity arrays.
"""
function apply_zou_he_south_spatial_2d!(f, ux_arr, uy_arr, Nx)
    backend = KernelAbstractions.get_backend(f)
    kernel! = zou_he_velocity_south_spatial_2d_kernel!(backend)
    kernel!(f, ux_arr, uy_arr, Nx; ndrange=(Nx,))
    KernelAbstractions.synchronize(backend)
end

# --- Zou-He spatial velocity BC on west wall (i = 1) ---

@kernel function zou_he_velocity_west_spatial_2d_kernel!(f, @Const(ux_arr), @Const(uy_arr), Ny)
    j = @index(Global)

    @inbounds if j > 1 && j < Ny
        T = eltype(f)
        u_x = T(ux_arr[j])
        u_y = T(uy_arr[j])

        f1 = f[1, j, 1]; f3 = f[1, j, 3]; f4 = f[1, j, 4]
        f5 = f[1, j, 5]; f7 = f[1, j, 7]; f8 = f[1, j, 8]

        ρ_wall = (f1 + f3 + f5 + T(2) * (f4 + f7 + f8)) / (one(T) - u_x)

        f[1, j, 2] = f4 + T(2.0/3.0) * ρ_wall * u_x
        f[1, j, 6] = f8 - T(0.5) * (f3 - f5) + T(1.0/6.0) * ρ_wall * u_x + T(0.5) * ρ_wall * u_y
        f[1, j, 9] = f7 + T(0.5) * (f3 - f5) + T(1.0/6.0) * ρ_wall * u_x - T(0.5) * ρ_wall * u_y
    end
end

"""
    apply_zou_he_west_spatial_2d!(f, ux_arr, uy_arr, Nx, Ny)

Zou-He velocity BC on west wall with per-node velocity arrays.
"""
function apply_zou_he_west_spatial_2d!(f, ux_arr, uy_arr, Nx, Ny)
    backend = KernelAbstractions.get_backend(f)
    kernel! = zou_he_velocity_west_spatial_2d_kernel!(backend)
    kernel!(f, ux_arr, uy_arr, Ny; ndrange=(Ny,))
    KernelAbstractions.synchronize(backend)
end

# --- Zou-He spatial pressure BC on east wall (i = Nx) ---

@kernel function zou_he_pressure_east_spatial_2d_kernel!(f, Nx, @Const(ρ_arr), Ny)
    j = @index(Global)

    @inbounds if j > 1 && j < Ny
        T = eltype(f)
        ρ_out = T(ρ_arr[j])

        f1 = f[Nx, j, 1]; f2 = f[Nx, j, 2]; f3 = f[Nx, j, 3]
        f5 = f[Nx, j, 5]; f6 = f[Nx, j, 6]; f9 = f[Nx, j, 9]

        ux = -one(T) + (f1 + f3 + f5 + T(2) * (f2 + f6 + f9)) / ρ_out

        f[Nx, j, 4] = f2 - T(2.0/3.0) * ρ_out * ux
        f[Nx, j, 7] = f9 - T(0.5) * (f3 - f5) - T(1.0/6.0) * ρ_out * ux
        f[Nx, j, 8] = f6 + T(0.5) * (f3 - f5) - T(1.0/6.0) * ρ_out * ux
    end
end

"""
    apply_zou_he_pressure_east_spatial_2d!(f, rho_arr, Nx, Ny)

Zou-He pressure BC on east wall with per-node density array.
"""
function apply_zou_he_pressure_east_spatial_2d!(f, rho_arr, Nx, Ny)
    backend = KernelAbstractions.get_backend(f)
    kernel! = zou_he_pressure_east_spatial_2d_kernel!(backend)
    kernel!(f, Nx, rho_arr, Ny; ndrange=(Ny,))
    KernelAbstractions.synchronize(backend)
end
