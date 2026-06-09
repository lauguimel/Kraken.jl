using KernelAbstractions

if !isdefined(@__MODULE__, :FVFD_BC_PERIODIC)
    const FVFD_BC_PERIODIC = UInt8(1)
    const FVFD_BC_OPEN = UInt8(2)
    const FVFD_BC_WALL = UInt8(3)
end

@inline function _gdl_xface_average_or_zero_2d(ux, is_solid, i_left, i_right, j)
    T = eltype(ux)
    return (is_solid[i_left, j] || is_solid[i_right, j]) ?
           zero(T) :
           (ux[i_left, j] + ux[i_right, j]) / T(2)
end

@inline function _gdl_yface_average_or_zero_2d(uy, is_solid, i, j_down, j_up)
    T = eltype(uy)
    return (is_solid[i, j_down] || is_solid[i, j_up]) ?
           zero(T) :
           (uy[i, j_down] + uy[i, j_up]) / T(2)
end

@inline function _gdl_xface_fraction_2d(
    is_solid, west_fraction, east_fraction, i_left, i_right, j,
)
    T = eltype(west_fraction)
    return (is_solid[i_left, j] || is_solid[i_right, j]) ?
           zero(T) :
           min(east_fraction[i_left, j], west_fraction[i_right, j])
end

@inline function _gdl_yface_fraction_2d(
    is_solid, south_fraction, north_fraction, i, j_down, j_up,
)
    T = eltype(south_fraction)
    return (is_solid[i, j_down] || is_solid[i, j_up]) ?
           zero(T) :
           min(north_fraction[i, j_down], south_fraction[i, j_up])
end

@inline function _gdl_embedded_xface_self_east(
    is_solid, west_fraction, east_fraction, i, j, Nx, east_bc,
)
    T = eltype(east_fraction)
    if i < Nx
        return _gdl_xface_fraction_2d(is_solid, west_fraction, east_fraction, i, i + 1, j) / T(2)
    elseif east_bc == FVFD_BC_PERIODIC
        return _gdl_xface_fraction_2d(is_solid, west_fraction, east_fraction, Nx, 1, j) / T(2)
    else
        return east_fraction[i, j]
    end
end

@inline function _gdl_embedded_xface_self_west(
    is_solid, west_fraction, east_fraction, i, j, Nx, west_bc,
)
    T = eltype(west_fraction)
    if i > 1
        return _gdl_xface_fraction_2d(is_solid, west_fraction, east_fraction, i - 1, i, j) / T(2)
    elseif west_bc == FVFD_BC_PERIODIC
        return _gdl_xface_fraction_2d(is_solid, west_fraction, east_fraction, Nx, 1, j) / T(2)
    else
        return west_fraction[i, j]
    end
end

@inline function _gdl_embedded_yface_self_north(
    is_solid, south_fraction, north_fraction, i, j, Ny, north_bc,
)
    T = eltype(north_fraction)
    if j < Ny
        return _gdl_yface_fraction_2d(is_solid, south_fraction, north_fraction, i, j, j + 1) / T(2)
    elseif north_bc == FVFD_BC_PERIODIC
        return _gdl_yface_fraction_2d(is_solid, south_fraction, north_fraction, i, Ny, 1) / T(2)
    else
        return north_fraction[i, j]
    end
end

@inline function _gdl_embedded_yface_self_south(
    is_solid, south_fraction, north_fraction, i, j, Ny, south_bc,
)
    T = eltype(south_fraction)
    if j > 1
        return _gdl_yface_fraction_2d(is_solid, south_fraction, north_fraction, i, j - 1, j) / T(2)
    elseif south_bc == FVFD_BC_PERIODIC
        return _gdl_yface_fraction_2d(is_solid, south_fraction, north_fraction, i, Ny, 1) / T(2)
    else
        return south_fraction[i, j]
    end
end

@kernel function gdl_divergence_2d_kernel!(
    divu, @Const(ux), @Const(uy), @Const(is_solid),
    @Const(west_fraction), @Const(east_fraction),
    @Const(south_fraction), @Const(north_fraction), @Const(cell_fraction),
    inv_dx, inv_dy, west_bc, east_bc, south_bc, north_bc, Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            if is_solid[i, j]
                divu[i, j] = zero(eltype(divu))
            else
                T = eltype(divu)

                if i < Nx
                    e_frac = _gdl_xface_fraction_2d(is_solid, west_fraction, east_fraction, i, i + 1, j)
                    e_ux = _gdl_xface_average_or_zero_2d(ux, is_solid, i, i + 1, j)
                elseif east_bc == FVFD_BC_PERIODIC
                    e_frac = _gdl_xface_fraction_2d(is_solid, west_fraction, east_fraction, Nx, 1, j)
                    e_ux = _gdl_xface_average_or_zero_2d(ux, is_solid, Nx, 1, j)
                else
                    e_frac = east_fraction[i, j]
                    e_ux = ux[i, j]
                end

                if i > 1
                    w_frac = _gdl_xface_fraction_2d(is_solid, west_fraction, east_fraction, i - 1, i, j)
                    w_ux = _gdl_xface_average_or_zero_2d(ux, is_solid, i - 1, i, j)
                elseif west_bc == FVFD_BC_PERIODIC
                    w_frac = _gdl_xface_fraction_2d(is_solid, west_fraction, east_fraction, Nx, 1, j)
                    w_ux = _gdl_xface_average_or_zero_2d(ux, is_solid, Nx, 1, j)
                else
                    w_frac = west_fraction[i, j]
                    w_ux = ux[i, j]
                end

                if j < Ny
                    n_frac = _gdl_yface_fraction_2d(is_solid, south_fraction, north_fraction, i, j, j + 1)
                    n_uy = _gdl_yface_average_or_zero_2d(uy, is_solid, i, j, j + 1)
                elseif north_bc == FVFD_BC_PERIODIC
                    n_frac = _gdl_yface_fraction_2d(is_solid, south_fraction, north_fraction, i, Ny, 1)
                    n_uy = _gdl_yface_average_or_zero_2d(uy, is_solid, i, Ny, 1)
                else
                    n_frac = north_fraction[i, j]
                    n_uy = uy[i, j]
                end

                if j > 1
                    s_frac = _gdl_yface_fraction_2d(is_solid, south_fraction, north_fraction, i, j - 1, j)
                    s_uy = _gdl_yface_average_or_zero_2d(uy, is_solid, i, j - 1, j)
                elseif south_bc == FVFD_BC_PERIODIC
                    s_frac = _gdl_yface_fraction_2d(is_solid, south_fraction, north_fraction, i, Ny, 1)
                    s_uy = _gdl_yface_average_or_zero_2d(uy, is_solid, i, Ny, 1)
                else
                    s_frac = south_fraction[i, j]
                    s_uy = uy[i, j]
                end

                volume_fraction = max(cell_fraction[i, j], eps(T))
                wall_x_length = west_fraction[i, j] - east_fraction[i, j]
                wall_y_length = south_fraction[i, j] - north_fraction[i, j]
                divu[i, j] = (
                    (e_frac * e_ux - w_frac * w_ux + wall_x_length * ux[i, j]) * inv_dx +
                    (n_frac * n_uy - s_frac * s_uy + wall_y_length * uy[i, j]) * inv_dy
                ) / volume_fraction
            end
        end
    end
end

@kernel function gdl_divergence_regular_2d_kernel!(
    divu, @Const(ux), @Const(uy), @Const(is_solid),
    inv_dx, inv_dy, west_bc, east_bc, south_bc, north_bc, Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            if is_solid[i, j]
                divu[i, j] = zero(eltype(divu))
            else
                if i < Nx
                    e_ux = _gdl_xface_average_or_zero_2d(ux, is_solid, i, i + 1, j)
                elseif east_bc == FVFD_BC_PERIODIC
                    e_ux = _gdl_xface_average_or_zero_2d(ux, is_solid, Nx, 1, j)
                else
                    e_ux = ux[i, j]
                end

                if i > 1
                    w_ux = _gdl_xface_average_or_zero_2d(ux, is_solid, i - 1, i, j)
                elseif west_bc == FVFD_BC_PERIODIC
                    w_ux = _gdl_xface_average_or_zero_2d(ux, is_solid, Nx, 1, j)
                else
                    w_ux = ux[i, j]
                end

                if j < Ny
                    n_uy = _gdl_yface_average_or_zero_2d(uy, is_solid, i, j, j + 1)
                elseif north_bc == FVFD_BC_PERIODIC
                    n_uy = _gdl_yface_average_or_zero_2d(uy, is_solid, i, Ny, 1)
                else
                    n_uy = uy[i, j]
                end

                if j > 1
                    s_uy = _gdl_yface_average_or_zero_2d(uy, is_solid, i, j - 1, j)
                elseif south_bc == FVFD_BC_PERIODIC
                    s_uy = _gdl_yface_average_or_zero_2d(uy, is_solid, i, Ny, 1)
                else
                    s_uy = uy[i, j]
                end

                divu[i, j] = (e_ux - w_ux) * inv_dx + (n_uy - s_uy) * inv_dy
            end
        end
    end
end

@kernel function gdl_pressure_gradient_2d_kernel!(
    gpx, gpy, @Const(p), @Const(is_solid),
    @Const(west_fraction), @Const(east_fraction),
    @Const(south_fraction), @Const(north_fraction), @Const(cell_fraction),
    inv_dx, inv_dy, west_bc, east_bc, south_bc, north_bc, Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            if is_solid[i, j]
                gpx[i, j] = zero(eltype(gpx))
                gpy[i, j] = zero(eltype(gpy))
            else
                T = eltype(gpx)
                half = inv(T(2))

                # Negative transpose of the embedded divergence flux form.
                # Duality convention: both scalar and vector cell products are
                # weighted by cell_fraction*h^2, so the receiving velocity cell
                # divides the transposed flux by its own open volume.
                x_flux_t = (
                    _gdl_embedded_xface_self_east(
                        is_solid, west_fraction, east_fraction, i, j, Nx, east_bc,
                    ) -
                    _gdl_embedded_xface_self_west(
                        is_solid, west_fraction, east_fraction, i, j, Nx, west_bc,
                    ) +
                    west_fraction[i, j] - east_fraction[i, j]
                ) * p[i, j]

                if i > 1
                    x_flux_t += half *
                                _gdl_xface_fraction_2d(
                                    is_solid, west_fraction, east_fraction, i - 1, i, j,
                                ) * p[i - 1, j]
                elseif east_bc == FVFD_BC_PERIODIC
                    x_flux_t += half *
                                _gdl_xface_fraction_2d(
                                    is_solid, west_fraction, east_fraction, Nx, 1, j,
                                ) * p[Nx, j]
                end

                if i < Nx
                    x_flux_t -= half *
                                _gdl_xface_fraction_2d(
                                    is_solid, west_fraction, east_fraction, i, i + 1, j,
                                ) * p[i + 1, j]
                elseif west_bc == FVFD_BC_PERIODIC
                    x_flux_t -= half *
                                _gdl_xface_fraction_2d(
                                    is_solid, west_fraction, east_fraction, Nx, 1, j,
                                ) * p[1, j]
                end

                y_flux_t = (
                    _gdl_embedded_yface_self_north(
                        is_solid, south_fraction, north_fraction, i, j, Ny, north_bc,
                    ) -
                    _gdl_embedded_yface_self_south(
                        is_solid, south_fraction, north_fraction, i, j, Ny, south_bc,
                    ) +
                    south_fraction[i, j] - north_fraction[i, j]
                ) * p[i, j]

                if j > 1
                    y_flux_t += half *
                                _gdl_yface_fraction_2d(
                                    is_solid, south_fraction, north_fraction, i, j - 1, j,
                                ) * p[i, j - 1]
                elseif north_bc == FVFD_BC_PERIODIC
                    y_flux_t += half *
                                _gdl_yface_fraction_2d(
                                    is_solid, south_fraction, north_fraction, i, Ny, 1,
                                ) * p[i, Ny]
                end

                if j < Ny
                    y_flux_t -= half *
                                _gdl_yface_fraction_2d(
                                    is_solid, south_fraction, north_fraction, i, j, j + 1,
                                ) * p[i, j + 1]
                elseif south_bc == FVFD_BC_PERIODIC
                    y_flux_t -= half *
                                _gdl_yface_fraction_2d(
                                    is_solid, south_fraction, north_fraction, i, Ny, 1,
                                ) * p[i, 1]
                end

                volume_fraction = max(cell_fraction[i, j], eps(T))
                gpx[i, j] = -x_flux_t * inv_dx / volume_fraction
                gpy[i, j] = -y_flux_t * inv_dy / volume_fraction
            end
        end
    end
end

@kernel function gdl_pressure_gradient_regular_2d_kernel!(
    gpx, gpy, @Const(p), @Const(is_solid),
    inv_dx, inv_dy, west_bc, east_bc, south_bc, north_bc, Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            if is_solid[i, j]
                gpx[i, j] = zero(eltype(gpx))
                gpy[i, j] = zero(eltype(gpy))
            else
                T = eltype(gpx)
                half = inv(T(2))
                x_flux_t = zero(T)
                y_flux_t = zero(T)

                if i == Nx && east_bc != FVFD_BC_PERIODIC
                    x_flux_t += p[i, j]
                elseif i < Nx
                    x_flux_t += half * p[i, j]
                else
                    x_flux_t += half * p[i, j]
                end

                if i == 1 && west_bc != FVFD_BC_PERIODIC
                    x_flux_t -= p[i, j]
                elseif i > 1
                    x_flux_t -= half * p[i, j]
                else
                    x_flux_t -= half * p[i, j]
                end

                if i > 1
                    x_flux_t += half * p[i - 1, j]
                elseif east_bc == FVFD_BC_PERIODIC
                    x_flux_t += half * p[Nx, j]
                end

                if i < Nx
                    x_flux_t -= half * p[i + 1, j]
                elseif west_bc == FVFD_BC_PERIODIC
                    x_flux_t -= half * p[1, j]
                end

                if j == Ny && north_bc != FVFD_BC_PERIODIC
                    y_flux_t += p[i, j]
                elseif j < Ny
                    y_flux_t += half * p[i, j]
                else
                    y_flux_t += half * p[i, j]
                end

                if j == 1 && south_bc != FVFD_BC_PERIODIC
                    y_flux_t -= p[i, j]
                elseif j > 1
                    y_flux_t -= half * p[i, j]
                else
                    y_flux_t -= half * p[i, j]
                end

                if j > 1
                    y_flux_t += half * p[i, j - 1]
                elseif north_bc == FVFD_BC_PERIODIC
                    y_flux_t += half * p[i, Ny]
                end

                if j < Ny
                    y_flux_t -= half * p[i, j + 1]
                elseif south_bc == FVFD_BC_PERIODIC
                    y_flux_t -= half * p[i, 1]
                end

                gpx[i, j] = -x_flux_t * inv_dx
                gpy[i, j] = -y_flux_t * inv_dy
            end
        end
    end
end

@kernel function gdl_laplacian_apply_2d_kernel!(
    lap, @Const(u), @Const(is_solid),
    @Const(west_fraction), @Const(east_fraction),
    @Const(south_fraction), @Const(north_fraction),
    inv_dx2, inv_dy2, west_bc, east_bc, south_bc, north_bc, Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            if is_solid[i, j]
                lap[i, j] = zero(eltype(lap))
            else
                T = eltype(lap)
                uc = u[i, j]
                acc = zero(T)

                if i < Nx
                    α = _gdl_xface_fraction_2d(is_solid, west_fraction, east_fraction, i, i + 1, j)
                    acc += α * (u[i + 1, j] - uc) * inv_dx2
                elseif east_bc == FVFD_BC_PERIODIC
                    α = _gdl_xface_fraction_2d(is_solid, west_fraction, east_fraction, Nx, 1, j)
                    acc += α * (u[1, j] - uc) * inv_dx2
                end

                if i > 1
                    α = _gdl_xface_fraction_2d(is_solid, west_fraction, east_fraction, i - 1, i, j)
                    acc += α * (u[i - 1, j] - uc) * inv_dx2
                elseif west_bc == FVFD_BC_PERIODIC
                    α = _gdl_xface_fraction_2d(is_solid, west_fraction, east_fraction, Nx, 1, j)
                    acc += α * (u[Nx, j] - uc) * inv_dx2
                end

                if j < Ny
                    α = _gdl_yface_fraction_2d(is_solid, south_fraction, north_fraction, i, j, j + 1)
                    acc += α * (u[i, j + 1] - uc) * inv_dy2
                elseif north_bc == FVFD_BC_PERIODIC
                    α = _gdl_yface_fraction_2d(is_solid, south_fraction, north_fraction, i, Ny, 1)
                    acc += α * (u[i, 1] - uc) * inv_dy2
                end

                if j > 1
                    α = _gdl_yface_fraction_2d(is_solid, south_fraction, north_fraction, i, j - 1, j)
                    acc += α * (u[i, j - 1] - uc) * inv_dy2
                elseif south_bc == FVFD_BC_PERIODIC
                    α = _gdl_yface_fraction_2d(is_solid, south_fraction, north_fraction, i, Ny, 1)
                    acc += α * (u[i, Ny] - uc) * inv_dy2
                end

                # Matrix-free apply of assemble_poisson_embedded rows: no
                # cell-fraction division is applied here.
                lap[i, j] = acc
            end
        end
    end
end

@kernel function gdl_laplacian_apply_regular_2d_kernel!(
    lap, @Const(u), @Const(is_solid),
    inv_dx2, inv_dy2, west_bc, east_bc, south_bc, north_bc, Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            if is_solid[i, j]
                lap[i, j] = zero(eltype(lap))
            else
                T = eltype(lap)
                uc = u[i, j]
                acc = zero(T)

                if i < Nx
                    !is_solid[i + 1, j] && (acc += (u[i + 1, j] - uc) * inv_dx2)
                elseif east_bc == FVFD_BC_PERIODIC
                    !is_solid[1, j] && (acc += (u[1, j] - uc) * inv_dx2)
                end

                if i > 1
                    !is_solid[i - 1, j] && (acc += (u[i - 1, j] - uc) * inv_dx2)
                elseif west_bc == FVFD_BC_PERIODIC
                    !is_solid[Nx, j] && (acc += (u[Nx, j] - uc) * inv_dx2)
                end

                if j < Ny
                    !is_solid[i, j + 1] && (acc += (u[i, j + 1] - uc) * inv_dy2)
                elseif north_bc == FVFD_BC_PERIODIC
                    !is_solid[i, 1] && (acc += (u[i, 1] - uc) * inv_dy2)
                end

                if j > 1
                    !is_solid[i, j - 1] && (acc += (u[i, j - 1] - uc) * inv_dy2)
                elseif south_bc == FVFD_BC_PERIODIC
                    !is_solid[i, Ny] && (acc += (u[i, Ny] - uc) * inv_dy2)
                end

                lap[i, j] = acc
            end
        end
    end
end

function gdl_divergence_embedded_2d!(
    divu, ux, uy, is_solid,
    west_fraction, east_fraction, south_fraction, north_fraction, cell_fraction,
    dx, dy, west_bc, east_bc, south_bc, north_bc;
    backend=KernelAbstractions.get_backend(divu), sync::Bool=true,
)
    Nx, Ny = size(divu)
    kernel! = gdl_divergence_2d_kernel!(backend)
    kernel!(
        divu, ux, uy, is_solid,
        west_fraction, east_fraction, south_fraction, north_fraction, cell_fraction,
        inv(dx), inv(dy), west_bc, east_bc, south_bc, north_bc, Nx, Ny;
        ndrange=(Nx, Ny),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

function gdl_divergence_2d!(
    divu, ux, uy, is_solid, dx, dy, west_bc, east_bc, south_bc, north_bc;
    backend=KernelAbstractions.get_backend(divu), sync::Bool=true,
)
    Nx, Ny = size(divu)
    kernel! = gdl_divergence_regular_2d_kernel!(backend)
    kernel!(
        divu, ux, uy, is_solid,
        inv(dx), inv(dy), west_bc, east_bc, south_bc, north_bc, Nx, Ny;
        ndrange=(Nx, Ny),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

function gdl_divergence_2d!(
    divu, ux, uy, is_solid,
    west_fraction, east_fraction, south_fraction, north_fraction, cell_fraction,
    dx, dy, west_bc, east_bc, south_bc, north_bc;
    backend=KernelAbstractions.get_backend(divu), sync::Bool=true,
)
    return gdl_divergence_embedded_2d!(
        divu, ux, uy, is_solid,
        west_fraction, east_fraction, south_fraction, north_fraction, cell_fraction,
        dx, dy, west_bc, east_bc, south_bc, north_bc; backend, sync,
    )
end

function gdl_pressure_gradient_embedded_2d!(
    gpx, gpy, p, is_solid,
    west_fraction, east_fraction, south_fraction, north_fraction, cell_fraction,
    dx, dy, west_bc, east_bc, south_bc, north_bc;
    backend=KernelAbstractions.get_backend(gpx), sync::Bool=true,
)
    Nx, Ny = size(gpx)
    kernel! = gdl_pressure_gradient_2d_kernel!(backend)
    kernel!(
        gpx, gpy, p, is_solid,
        west_fraction, east_fraction, south_fraction, north_fraction, cell_fraction,
        inv(dx), inv(dy), west_bc, east_bc, south_bc, north_bc, Nx, Ny;
        ndrange=(Nx, Ny),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

function gdl_pressure_gradient_2d!(
    gpx, gpy, p, is_solid, dx, dy, west_bc, east_bc, south_bc, north_bc;
    backend=KernelAbstractions.get_backend(gpx), sync::Bool=true,
)
    Nx, Ny = size(gpx)
    kernel! = gdl_pressure_gradient_regular_2d_kernel!(backend)
    kernel!(
        gpx, gpy, p, is_solid,
        inv(dx), inv(dy), west_bc, east_bc, south_bc, north_bc, Nx, Ny;
        ndrange=(Nx, Ny),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

function gdl_pressure_gradient_2d!(
    gpx, gpy, p, is_solid,
    west_fraction, east_fraction, south_fraction, north_fraction, cell_fraction,
    dx, dy, west_bc, east_bc, south_bc, north_bc;
    backend=KernelAbstractions.get_backend(gpx), sync::Bool=true,
)
    return gdl_pressure_gradient_embedded_2d!(
        gpx, gpy, p, is_solid,
        west_fraction, east_fraction, south_fraction, north_fraction, cell_fraction,
        dx, dy, west_bc, east_bc, south_bc, north_bc; backend, sync,
    )
end

function gdl_laplacian_apply_embedded_2d!(
    lap, u, is_solid,
    west_fraction, east_fraction, south_fraction, north_fraction,
    dx, dy, west_bc, east_bc, south_bc, north_bc;
    backend=KernelAbstractions.get_backend(lap), sync::Bool=true,
)
    Nx, Ny = size(lap)
    kernel! = gdl_laplacian_apply_2d_kernel!(backend)
    kernel!(
        lap, u, is_solid,
        west_fraction, east_fraction, south_fraction, north_fraction,
        inv(dx * dx), inv(dy * dy), west_bc, east_bc, south_bc, north_bc, Nx, Ny;
        ndrange=(Nx, Ny),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

function gdl_laplacian_apply_2d!(
    lap, u, is_solid, dx, dy, west_bc, east_bc, south_bc, north_bc;
    backend=KernelAbstractions.get_backend(lap), sync::Bool=true,
)
    Nx, Ny = size(lap)
    kernel! = gdl_laplacian_apply_regular_2d_kernel!(backend)
    kernel!(
        lap, u, is_solid,
        inv(dx * dx), inv(dy * dy), west_bc, east_bc, south_bc, north_bc, Nx, Ny;
        ndrange=(Nx, Ny),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

function gdl_laplacian_apply_2d!(
    lap, u, is_solid,
    west_fraction, east_fraction, south_fraction, north_fraction,
    dx, dy, west_bc, east_bc, south_bc, north_bc;
    backend=KernelAbstractions.get_backend(lap), sync::Bool=true,
)
    return gdl_laplacian_apply_embedded_2d!(
        lap, u, is_solid,
        west_fraction, east_fraction, south_fraction, north_fraction,
        dx, dy, west_bc, east_bc, south_bc, north_bc; backend, sync,
    )
end

function gdl_laplacian_apply_embedded_2d!(
    lap_ux, lap_uy, ux, uy, is_solid,
    west_fraction, east_fraction, south_fraction, north_fraction,
    dx, dy, west_bc, east_bc, south_bc, north_bc;
    backend=KernelAbstractions.get_backend(lap_ux), sync::Bool=true,
)
    gdl_laplacian_apply_embedded_2d!(
        lap_ux, ux, is_solid,
        west_fraction, east_fraction, south_fraction, north_fraction,
        dx, dy, west_bc, east_bc, south_bc, north_bc; backend, sync=false,
    )
    gdl_laplacian_apply_embedded_2d!(
        lap_uy, uy, is_solid,
        west_fraction, east_fraction, south_fraction, north_fraction,
        dx, dy, west_bc, east_bc, south_bc, north_bc; backend, sync,
    )
    return nothing
end

function gdl_laplacian_apply_2d!(
    lap_ux, lap_uy, ux, uy, is_solid,
    dx, dy, west_bc, east_bc, south_bc, north_bc;
    backend=KernelAbstractions.get_backend(lap_ux), sync::Bool=true,
)
    gdl_laplacian_apply_2d!(
        lap_ux, ux, is_solid, dx, dy, west_bc, east_bc, south_bc, north_bc;
        backend, sync=false,
    )
    gdl_laplacian_apply_2d!(
        lap_uy, uy, is_solid, dx, dy, west_bc, east_bc, south_bc, north_bc;
        backend, sync,
    )
    return nothing
end
