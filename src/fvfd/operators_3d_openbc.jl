@kernel function _fvfd_fill_planar_extensional_openx_bcs_3d_kernel!(
    ux_west, ux_east, u_profile_west, psi_west, psi_east,
    epsilon_dot, dx, x_center, Nx,
)
    j, k = @index(Global, NTuple)
    T = eltype(ux_west)
    @inbounds begin
        x_west = (T(0.5) - x_center) * dx
        x_east = (T(Nx) + T(0.5) - x_center) * dx
        uw = epsilon_dot * x_west
        ue = epsilon_dot * x_east
        ux_west[j, k] = uw
        ux_east[j, k] = ue
        u_profile_west[j, k] = uw
        psi_west[j, k] = zero(T)
        psi_east[j, k] = zero(T)
    end
end

"""
    fvfd_fill_planar_extensional_openx_bcs_3d!(ux_west, ux_east,
        u_profile_west, psi_west, psi_east, Nx, epsilon_dot; dx=1,
        x_center=(Nx + 1)/2)

Fill the imposed open-x FVFD face velocities and zero log-conformation ghost
values for the 3D planar-extension canary. Coordinates are cell-centered:
`x_i = (i - x_center) * dx`, so the west/east faces lie at `i = 0.5` and
`i = Nx + 0.5`.
"""
function fvfd_fill_planar_extensional_openx_bcs_3d!(
    ux_west, ux_east, u_profile_west, psi_west, psi_east,
    Nx::Int, epsilon_dot::Real;
    dx::Real=1,
    x_center::Real=(Nx + 1) / 2,
    sync::Bool=true,
)
    Ny, Nz = size(ux_west)
    _fvfd_check_boundary_size_3d(:ux_east, ux_east, (Ny, Nz))
    _fvfd_check_boundary_size_3d(:u_profile_west, u_profile_west, (Ny, Nz))
    _fvfd_check_boundary_size_3d(:psi_west, psi_west, (Ny, Nz))
    _fvfd_check_boundary_size_3d(:psi_east, psi_east, (Ny, Nz))

    backend = KernelAbstractions.get_backend(ux_west)
    T = eltype(ux_west)
    kernel! = _fvfd_fill_planar_extensional_openx_bcs_3d_kernel!(backend)
    kernel!(
        ux_west, ux_east, u_profile_west, psi_west, psi_east,
        T(epsilon_dot), T(dx), T(x_center), Nx;
        ndrange=(Ny, Nz),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function _fvfd_impose_planar_extensional_velocity_3d_kernel!(
    ux, uy, uz, epsilon_dot, dx, dy, x_center, y_center,
)
    i, j, k = @index(Global, NTuple)
    T = eltype(ux)
    @inbounds begin
        x = (T(i) - x_center) * dx
        y = (T(j) - y_center) * dy
        ux[i, j, k] = epsilon_dot * x
        uy[i, j, k] = -epsilon_dot * y
        uz[i, j, k] = zero(T)
    end
end

"""
    fvfd_impose_planar_extensional_velocity_3d!(ux, uy, uz, epsilon_dot;
        dx=1, dy=1, x_center=(Nx + 1)/2, y_center=(Ny + 1)/2)

Overwrite a cell-centered velocity field with `u=(epsilon_dot*x,
-epsilon_dot*y, 0)`. This is the explicit kinematic fallback used when the
coarse open-flow solve does not provide the analytical stagnation field.
"""
function fvfd_impose_planar_extensional_velocity_3d!(
    ux, uy, uz, epsilon_dot::Real;
    dx::Real=1,
    dy::Real=1,
    x_center::Real=(size(ux, 1) + 1) / 2,
    y_center::Real=(size(ux, 2) + 1) / 2,
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(ux)
    Nx, Ny, Nz = size(ux)
    size(uy) == (Nx, Ny, Nz) || throw(DimensionMismatch("uy must match ux"))
    size(uz) == (Nx, Ny, Nz) || throw(DimensionMismatch("uz must match ux"))
    T = eltype(ux)
    kernel! = _fvfd_impose_planar_extensional_velocity_3d_kernel!(backend)
    kernel!(
        ux, uy, uz, T(epsilon_dot), T(dx), T(dy), T(x_center), T(y_center);
        ndrange=(Nx, Ny, Nz),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

function fvfd_planar_extensional_velocity_field_host_3d(
    Nx::Int, Ny::Int, Nz::Int, epsilon_dot::Real;
    dx::Real=1,
    dy::Real=1,
    x_center::Real=(Nx + 1) / 2,
    y_center::Real=(Ny + 1) / 2,
    FT::Type{<:AbstractFloat}=Float64,
)
    ux = zeros(FT, Nx, Ny, Nz)
    uy = zeros(FT, Nx, Ny, Nz)
    uz = zeros(FT, Nx, Ny, Nz)
    epsT = FT(epsilon_dot)
    dxT = FT(dx)
    dyT = FT(dy)
    x0 = FT(x_center)
    y0 = FT(y_center)
    @inbounds for k in 1:Nz, j in 1:Ny, i in 1:Nx
        ux[i, j, k] = epsT * (FT(i) - x0) * dxT
        uy[i, j, k] = -epsT * (FT(j) - y0) * dyT
        uz[i, j, k] = zero(FT)
    end
    return (; ux, uy, uz)
end
