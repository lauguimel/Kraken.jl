@kernel function fvfd_bsd_force_2d_kernel!(
    fx_out, fy_out,
    @Const(fx_poly), @Const(fy_poly),
    @Const(ux), @Const(uy), @Const(is_solid),
    zeta_nu_p, inv_dx2, inv_dy2,
    west_bc, east_bc, south_bc, north_bc, Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            if is_solid[i, j]
                fx_out[i, j] = zero(eltype(fx_out))
                fy_out[i, j] = zero(eltype(fy_out))
            else
                lap_ux = _fvfd_solid_bc_second_derivative_x_2d(
                    ux, is_solid, i, j, Nx, inv_dx2, west_bc, east_bc,
                ) + _fvfd_solid_bc_second_derivative_y_2d(
                    ux, is_solid, i, j, Ny, inv_dy2, south_bc, north_bc,
                )
                lap_uy = _fvfd_solid_bc_second_derivative_x_2d(
                    uy, is_solid, i, j, Nx, inv_dx2, west_bc, east_bc,
                ) + _fvfd_solid_bc_second_derivative_y_2d(
                    uy, is_solid, i, j, Ny, inv_dy2, south_bc, north_bc,
                )
                fx_out[i, j] = fx_poly[i, j] - zeta_nu_p * lap_ux
                fy_out[i, j] = fy_poly[i, j] - zeta_nu_p * lap_uy
            end
        end
    end
end

function fvfd_bsd_force_2d!(
    fx_out, fy_out, fx_poly, fy_poly, ux, uy, is_solid,
    zeta, nu_p, dx, dy, bc::FVFDDomainBC2D;
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(fx_out)
    Nx, Ny = size(fx_out)
    kernel! = fvfd_bsd_force_2d_kernel!(backend)
    kernel!(
        fx_out, fy_out, fx_poly, fy_poly, ux, uy, is_solid,
        zeta * nu_p, inv(dx * dx), inv(dy * dy),
        bc.west, bc.east, bc.south, bc.north, Nx, Ny;
        ndrange=(Nx, Ny),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

function fvfd_bsd_force_2d!(
    fx_out, fy_out, fx_poly, fy_poly, ux, uy, geometry::FVFDGeometry2D,
    zeta, nu_p;
    sync::Bool=true,
)
    return fvfd_bsd_force_2d!(
        fx_out, fy_out, fx_poly, fy_poly, ux, uy,
        geometry.is_solid, zeta, nu_p,
        geometry.patch.dx, geometry.patch.dy, geometry.bc; sync,
    )
end

@kernel function fvfd_velocity_gradient_2d_kernel!(
    dudx, dudy, dvdx, dvdy,
    @Const(ux), @Const(uy), @Const(is_solid),
    inv_dx, inv_dy, inv_2dx, inv_2dy,
    west_bc, east_bc, south_bc, north_bc, Nx, Ny,
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
                dudx[i, j] = _fvfd_solid_bc_derivative_x_2d(
                    ux, is_solid, i, j, Nx, inv_dx, inv_2dx, west_bc, east_bc,
                )
                dudy[i, j] = _fvfd_solid_bc_derivative_y_2d(
                    ux, is_solid, i, j, Ny, inv_dy, inv_2dy, south_bc, north_bc,
                )
                dvdx[i, j] = _fvfd_solid_bc_derivative_x_2d(
                    uy, is_solid, i, j, Nx, inv_dx, inv_2dx, west_bc, east_bc,
                )
                dvdy[i, j] = _fvfd_solid_bc_derivative_y_2d(
                    uy, is_solid, i, j, Ny, inv_dy, inv_2dy, south_bc, north_bc,
                )
            end
        end
    end
end

@kernel function fvfd_velocity_gradient_embedded_2d_kernel!(
    dudx, dudy, dvdx, dvdy,
    @Const(ux), @Const(uy), @Const(is_solid),
    @Const(wall_nx), @Const(wall_ny), @Const(wall_inv_distance_to_center),
    inv_dx, inv_dy, inv_2dx, inv_2dy,
    west_bc, east_bc, south_bc, north_bc, Nx, Ny,
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
                ux_gx = _fvfd_solid_bc_derivative_x_2d(
                    ux, is_solid, i, j, Nx, inv_dx, inv_2dx, west_bc, east_bc,
                )
                ux_gy = _fvfd_solid_bc_derivative_y_2d(
                    ux, is_solid, i, j, Ny, inv_dy, inv_2dy, south_bc, north_bc,
                )
                uy_gx = _fvfd_solid_bc_derivative_x_2d(
                    uy, is_solid, i, j, Nx, inv_dx, inv_2dx, west_bc, east_bc,
                )
                uy_gy = _fvfd_solid_bc_derivative_y_2d(
                    uy, is_solid, i, j, Ny, inv_dy, inv_2dy, south_bc, north_bc,
                )
                ux_gx, ux_gy = _fvfd_apply_embedded_wall_gradient_2d(
                    ux_gx, ux_gy, ux, wall_nx, wall_ny, wall_inv_distance_to_center, i, j,
                )
                uy_gx, uy_gy = _fvfd_apply_embedded_wall_gradient_2d(
                    uy_gx, uy_gy, uy, wall_nx, wall_ny, wall_inv_distance_to_center, i, j,
                )
                dudx[i, j] = ux_gx
                dudy[i, j] = ux_gy
                dvdx[i, j] = uy_gx
                dvdy[i, j] = uy_gy
            end
        end
    end
end

function fvfd_velocity_gradient_2d!(
    dudx, dudy, dvdx, dvdy,
    ux, uy, is_solid, dx, dy, bc::FVFDDomainBC2D;
    sync::Bool=true,
)
    @trace_enter :vel_grad
    backend = KernelAbstractions.get_backend(ux)
    Nx, Ny = size(ux)
    kernel! = fvfd_velocity_gradient_2d_kernel!(backend)
    kernel!(
        dudx, dudy, dvdx, dvdy,
        ux, uy, is_solid,
        inv(dx), inv(dy), inv(2 * dx), inv(2 * dy),
        bc.west, bc.east, bc.south, bc.north, Nx, Ny;
        ndrange=(Nx, Ny),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

function fvfd_velocity_gradient_embedded_2d!(
    dudx, dudy, dvdx, dvdy,
    ux, uy, is_solid, dx, dy, bc::FVFDDomainBC2D,
    embedded::FVFDEmbeddedBoundary2D;
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(ux)
    Nx, Ny = size(ux)
    kernel! = fvfd_velocity_gradient_embedded_2d_kernel!(backend)
    kernel!(
        dudx, dudy, dvdx, dvdy,
        ux, uy, is_solid,
        embedded.wall_nx, embedded.wall_ny, embedded.wall_inv_distance_to_center,
        inv(dx), inv(dy), inv(2 * dx), inv(2 * dy),
        bc.west, bc.east, bc.south, bc.north, Nx, Ny;
        ndrange=(Nx, Ny),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

function fvfd_velocity_gradient_2d!(
    dudx, dudy, dvdx, dvdy,
    ux, uy, geometry::FVFDGeometry2D;
    sync::Bool=true,
)
    return fvfd_velocity_gradient_2d!(
        dudx, dudy, dvdx, dvdy,
        ux, uy, geometry.is_solid, geometry.patch.dx, geometry.patch.dy,
        geometry.bc; sync,
    )
end

function fvfd_velocity_gradient_embedded_2d!(
    dudx, dudy, dvdx, dvdy,
    ux, uy, geometry::FVFDGeometry2D;
    sync::Bool=true,
)
    return fvfd_velocity_gradient_embedded_2d!(
        dudx, dudy, dvdx, dvdy,
        ux, uy, geometry.is_solid, geometry.patch.dx, geometry.patch.dy,
        geometry.bc, geometry.embedded; sync,
    )
end
