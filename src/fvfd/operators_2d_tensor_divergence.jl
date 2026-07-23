@kernel function fvfd_tensor_divergence_2d_kernel!(
    fx, fy,
    @Const(tauxx), @Const(tauxy), @Const(tauyy), @Const(is_solid),
    inv_dx, inv_dy, inv_2dx, inv_2dy,
    west_bc, east_bc, south_bc, north_bc, Nx, Ny, polymer_wall_extrap,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            if is_solid[i, j]
                fx[i, j] = zero(eltype(fx))
                fy[i, j] = zero(eltype(fy))
            else
                fx[i, j] = _fvfd_solid_bc_derivative_x_2d(
                    tauxx, is_solid, i, j, Nx, inv_dx, inv_2dx, west_bc, east_bc,
                    polymer_wall_extrap,
                ) + _fvfd_solid_bc_derivative_y_2d(
                    tauxy, is_solid, i, j, Ny, inv_dy, inv_2dy, south_bc, north_bc,
                    polymer_wall_extrap,
                )
                fy[i, j] = _fvfd_solid_bc_derivative_x_2d(
                    tauxy, is_solid, i, j, Nx, inv_dx, inv_2dx, west_bc, east_bc,
                    polymer_wall_extrap,
                ) + _fvfd_solid_bc_derivative_y_2d(
                    tauyy, is_solid, i, j, Ny, inv_dy, inv_2dy, south_bc, north_bc,
                    polymer_wall_extrap,
                )
            end
        end
    end
end

@kernel function fvfd_tensor_divergence_embedded_2d_kernel!(
    fx, fy,
    @Const(tauxx), @Const(tauxy), @Const(tauyy), @Const(is_solid),
    @Const(cell_fraction),
    @Const(west_fraction), @Const(east_fraction),
    @Const(south_fraction), @Const(north_fraction),
    inv_dx, inv_dy,
    west_bc, east_bc, south_bc, north_bc, Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            if is_solid[i, j]
                fx[i, j] = zero(eltype(fx))
                fy[i, j] = zero(eltype(fy))
            else
                T = eltype(fx)

                if i < Nx
                    e_frac = _fvfd_xface_fraction_2d(
                        is_solid, west_fraction, east_fraction, i, i + 1, j,
                    )
                    e_xx = _fvfd_xface_scalar_average_or_zero_2d(tauxx, is_solid, i, i + 1, j)
                    e_xy = _fvfd_xface_scalar_average_or_zero_2d(tauxy, is_solid, i, i + 1, j)
                elseif east_bc == FVFD_BC_PERIODIC
                    e_frac = _fvfd_xface_fraction_2d(
                        is_solid, west_fraction, east_fraction, Nx, 1, j,
                    )
                    e_xx = _fvfd_xface_scalar_average_or_zero_2d(tauxx, is_solid, Nx, 1, j)
                    e_xy = _fvfd_xface_scalar_average_or_zero_2d(tauxy, is_solid, Nx, 1, j)
                else
                    e_frac = east_fraction[i, j]
                    e_xx = tauxx[i, j]
                    e_xy = tauxy[i, j]
                end

                if i > 1
                    w_frac = _fvfd_xface_fraction_2d(
                        is_solid, west_fraction, east_fraction, i - 1, i, j,
                    )
                    w_xx = _fvfd_xface_scalar_average_or_zero_2d(tauxx, is_solid, i - 1, i, j)
                    w_xy = _fvfd_xface_scalar_average_or_zero_2d(tauxy, is_solid, i - 1, i, j)
                elseif west_bc == FVFD_BC_PERIODIC
                    w_frac = _fvfd_xface_fraction_2d(
                        is_solid, west_fraction, east_fraction, Nx, 1, j,
                    )
                    w_xx = _fvfd_xface_scalar_average_or_zero_2d(tauxx, is_solid, Nx, 1, j)
                    w_xy = _fvfd_xface_scalar_average_or_zero_2d(tauxy, is_solid, Nx, 1, j)
                else
                    w_frac = west_fraction[i, j]
                    w_xx = tauxx[i, j]
                    w_xy = tauxy[i, j]
                end

                if j < Ny
                    n_frac = _fvfd_yface_fraction_2d(
                        is_solid, south_fraction, north_fraction, i, j, j + 1,
                    )
                    n_xy = _fvfd_yface_scalar_average_or_zero_2d(tauxy, is_solid, i, j, j + 1)
                    n_yy = _fvfd_yface_scalar_average_or_zero_2d(tauyy, is_solid, i, j, j + 1)
                elseif north_bc == FVFD_BC_PERIODIC
                    n_frac = _fvfd_yface_fraction_2d(
                        is_solid, south_fraction, north_fraction, i, Ny, 1,
                    )
                    n_xy = _fvfd_yface_scalar_average_or_zero_2d(tauxy, is_solid, i, Ny, 1)
                    n_yy = _fvfd_yface_scalar_average_or_zero_2d(tauyy, is_solid, i, Ny, 1)
                else
                    n_frac = north_fraction[i, j]
                    n_xy = tauxy[i, j]
                    n_yy = tauyy[i, j]
                end

                if j > 1
                    s_frac = _fvfd_yface_fraction_2d(
                        is_solid, south_fraction, north_fraction, i, j - 1, j,
                    )
                    s_xy = _fvfd_yface_scalar_average_or_zero_2d(tauxy, is_solid, i, j - 1, j)
                    s_yy = _fvfd_yface_scalar_average_or_zero_2d(tauyy, is_solid, i, j - 1, j)
                elseif south_bc == FVFD_BC_PERIODIC
                    s_frac = _fvfd_yface_fraction_2d(
                        is_solid, south_fraction, north_fraction, i, Ny, 1,
                    )
                    s_xy = _fvfd_yface_scalar_average_or_zero_2d(tauxy, is_solid, i, Ny, 1)
                    s_yy = _fvfd_yface_scalar_average_or_zero_2d(tauyy, is_solid, i, Ny, 1)
                else
                    s_frac = south_fraction[i, j]
                    s_xy = tauxy[i, j]
                    s_yy = tauyy[i, j]
                end

                volume_fraction = max(cell_fraction[i, j], eps(T))
                wall_x_length = west_fraction[i, j] - east_fraction[i, j]
                wall_y_length = south_fraction[i, j] - north_fraction[i, j]

                fx[i, j] = (
                    (e_frac * e_xx - w_frac * w_xx + wall_x_length * tauxx[i, j]) * inv_dx +
                    (n_frac * n_xy - s_frac * s_xy + wall_y_length * tauxy[i, j]) * inv_dy
                ) / volume_fraction
                fy[i, j] = (
                    (e_frac * e_xy - w_frac * w_xy + wall_x_length * tauxy[i, j]) * inv_dx +
                    (n_frac * n_yy - s_frac * s_yy + wall_y_length * tauyy[i, j]) * inv_dy
                ) / volume_fraction
            end
        end
    end
end

function fvfd_tensor_divergence_2d!(
    fx, fy, tauxx, tauxy, tauyy, is_solid, dx, dy, bc::FVFDDomainBC2D;
    sync::Bool=true,
    polymer_wall_extrap::Symbol=:quadratic,
)
    polymer_wall_extrap in (:quadratic, :linear) ||
        throw(ArgumentError("polymer_wall_extrap must be :quadratic or :linear"))
    backend = KernelAbstractions.get_backend(fx)
    Nx, Ny = size(fx)
    kernel! = fvfd_tensor_divergence_2d_kernel!(backend)
    kernel!(
        fx, fy, tauxx, tauxy, tauyy, is_solid,
        inv(dx), inv(dy), inv(2 * dx), inv(2 * dy),
        bc.west, bc.east, bc.south, bc.north, Nx, Ny, Val(polymer_wall_extrap);
        ndrange=(Nx, Ny),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

function fvfd_tensor_divergence_embedded_2d!(
    fx, fy, tauxx, tauxy, tauyy,
    is_solid, dx, dy, bc::FVFDDomainBC2D,
    embedded::FVFDEmbeddedBoundary2D;
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(fx)
    Nx, Ny = size(fx)
    kernel! = fvfd_tensor_divergence_embedded_2d_kernel!(backend)
    kernel!(
        fx, fy, tauxx, tauxy, tauyy, is_solid,
        embedded.cell_fraction,
        embedded.west_fraction, embedded.east_fraction,
        embedded.south_fraction, embedded.north_fraction,
        inv(dx), inv(dy),
        bc.west, bc.east, bc.south, bc.north, Nx, Ny;
        ndrange=(Nx, Ny),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

function fvfd_tensor_divergence_embedded_2d!(
    fx, fy, tauxx, tauxy, tauyy, geometry::FVFDGeometry2D;
    sync::Bool=true,
)
    return fvfd_tensor_divergence_embedded_2d!(
        fx, fy, tauxx, tauxy, tauyy,
        geometry.is_solid, geometry.patch.dx, geometry.patch.dy,
        geometry.bc, geometry.embedded; sync,
    )
end

@kernel function fvfd_scale_by_cell_fraction_2d_kernel!(
    fx, fy, @Const(cell_fraction), @Const(is_solid), Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny && !is_solid[i, j]
            c = cell_fraction[i, j]
            fx[i, j] *= c
            fy[i, j] *= c
        end
    end
end

function fvfd_scale_by_cell_fraction_2d!(
    fx, fy, embedded::FVFDEmbeddedBoundary2D, is_solid; sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(fx)
    Nx, Ny = size(fx)
    kernel! = fvfd_scale_by_cell_fraction_2d_kernel!(backend)
    kernel!(fx, fy, embedded.cell_fraction, is_solid, Nx, Ny; ndrange=(Nx, Ny))
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function fvfd_embedded_wall_traction_2d_kernel!(
    tx, ty,
    @Const(tauxx), @Const(tauxy), @Const(tauyy), @Const(is_solid),
    @Const(wall_nx), @Const(wall_ny), @Const(wall_fraction),
    Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            if is_solid[i, j]
                tx[i, j] = zero(eltype(tx))
                ty[i, j] = zero(eltype(ty))
            else
                length = wall_fraction[i, j]
                nx = wall_nx[i, j]
                ny = wall_ny[i, j]
                tx[i, j] = length * (tauxx[i, j] * nx + tauxy[i, j] * ny)
                ty[i, j] = length * (tauxy[i, j] * nx + tauyy[i, j] * ny)
            end
        end
    end
end

function fvfd_embedded_wall_traction_2d!(
    tx, ty, tauxx, tauxy, tauyy,
    is_solid, embedded::FVFDEmbeddedBoundary2D;
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(tx)
    Nx, Ny = size(tx)
    kernel! = fvfd_embedded_wall_traction_2d_kernel!(backend)
    kernel!(
        tx, ty, tauxx, tauxy, tauyy, is_solid,
        embedded.wall_nx, embedded.wall_ny, embedded.wall_fraction,
        Nx, Ny; ndrange=(Nx, Ny),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

function fvfd_embedded_wall_traction_2d!(
    tx, ty, tauxx, tauxy, tauyy, geometry::FVFDGeometry2D;
    sync::Bool=true,
)
    return fvfd_embedded_wall_traction_2d!(
        tx, ty, tauxx, tauxy, tauyy, geometry.is_solid, geometry.embedded; sync,
    )
end

function fvfd_tensor_divergence_2d!(
    fx, fy, tauxx, tauxy, tauyy, geometry::FVFDGeometry2D;
    sync::Bool=true,
    polymer_wall_extrap::Symbol=:quadratic,
)
    return fvfd_tensor_divergence_2d!(
        fx, fy, tauxx, tauxy, tauyy,
        geometry.is_solid, geometry.patch.dx, geometry.patch.dy, geometry.bc;
        sync=sync, polymer_wall_extrap=polymer_wall_extrap,
    )
end

