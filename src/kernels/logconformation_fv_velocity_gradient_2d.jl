@kernel function logfv_velocity_gradient_centered_2d_kernel!(
    dudx, dudy, dvdx, dvdy,
    @Const(ux), @Const(uy),
    inv_2dx, inv_2dy, Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            if i > 1 && i < Nx && j > 1 && j < Ny
                dudx[i, j] = (ux[i + 1, j] - ux[i - 1, j]) * inv_2dx
                dudy[i, j] = (ux[i, j + 1] - ux[i, j - 1]) * inv_2dy
                dvdx[i, j] = (uy[i + 1, j] - uy[i - 1, j]) * inv_2dx
                dvdy[i, j] = (uy[i, j + 1] - uy[i, j - 1]) * inv_2dy
            else
                dudx[i, j] = zero(eltype(dudx))
                dudy[i, j] = zero(eltype(dudy))
                dvdx[i, j] = zero(eltype(dvdx))
                dvdy[i, j] = zero(eltype(dvdy))
            end
        end
    end
end

function logfv_velocity_gradient_centered_2d!(
    dudx, dudy, dvdx, dvdy,
    ux, uy, dx, dy;
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(ux)
    Nx, Ny = size(ux)
    kernel! = logfv_velocity_gradient_centered_2d_kernel!(backend)
    kernel!(
        dudx, dudy, dvdx, dvdy,
        ux, uy, inv(2 * dx), inv(2 * dy), Nx, Ny;
        ndrange=(Nx, Ny),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

@kernel function logfv_velocity_gradient_periodicx_wally_2d_kernel!(
    dudx, dudy, dvdx, dvdy,
    @Const(ux), @Const(uy),
    inv_2dx, inv_2dy, Nx, Ny,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            im = ifelse(i > 1, i - 1, Nx)
            ip = ifelse(i < Nx, i + 1, 1)
            dudx[i, j] = (ux[ip, j] - ux[im, j]) * inv_2dx
            dvdx[i, j] = (uy[ip, j] - uy[im, j]) * inv_2dx

            if j == 1
                dudy[i, j] = (-3 * ux[i, j] + 4 * ux[i, j + 1] - ux[i, j + 2]) * inv_2dy
                dvdy[i, j] = (-3 * uy[i, j] + 4 * uy[i, j + 1] - uy[i, j + 2]) * inv_2dy
            elseif j == Ny
                dudy[i, j] = (3 * ux[i, j] - 4 * ux[i, j - 1] + ux[i, j - 2]) * inv_2dy
                dvdy[i, j] = (3 * uy[i, j] - 4 * uy[i, j - 1] + uy[i, j - 2]) * inv_2dy
            else
                dudy[i, j] = (ux[i, j + 1] - ux[i, j - 1]) * inv_2dy
                dvdy[i, j] = (uy[i, j + 1] - uy[i, j - 1]) * inv_2dy
            end
        end
    end
end

function logfv_velocity_gradient_periodicx_wally_2d!(
    dudx, dudy, dvdx, dvdy,
    ux, uy, dx, dy;
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(ux)
    Nx, Ny = size(ux)
    Ny >= 3 || throw(ArgumentError("wall-y velocity gradient requires Ny >= 3"))
    kernel! = logfv_velocity_gradient_periodicx_wally_2d_kernel!(backend)
    kernel!(
        dudx, dudy, dvdx, dvdy,
        ux, uy, inv(2 * dx), inv(2 * dy), Nx, Ny;
        ndrange=(Nx, Ny),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

@inline function _logfv_solid_aware_derivative_x_2d(field, is_solid, i, j, Nx, inv_dx, inv_2dx)
    T = eltype(field)
    left = i > 1 && !is_solid[i - 1, j]
    right = i < Nx && !is_solid[i + 1, j]
    if left && right
        return (field[i + 1, j] - field[i - 1, j]) * inv_2dx
    elseif right
        right2 = i + 2 <= Nx && !is_solid[i + 2, j]
        return right2 ?
               (-T(3) * field[i, j] + T(4) * field[i + 1, j] - field[i + 2, j]) * inv_2dx :
               (field[i + 1, j] - field[i, j]) * inv_dx
    elseif left
        left2 = i - 2 >= 1 && !is_solid[i - 2, j]
        return left2 ?
               (T(3) * field[i, j] - T(4) * field[i - 1, j] + field[i - 2, j]) * inv_2dx :
               (field[i, j] - field[i - 1, j]) * inv_dx
    else
        return zero(T)
    end
end

@inline function _logfv_solid_aware_derivative_y_2d(field, is_solid, i, j, Ny, inv_dy, inv_2dy)
    T = eltype(field)
    down = j > 1 && !is_solid[i, j - 1]
    up = j < Ny && !is_solid[i, j + 1]
    if down && up
        return (field[i, j + 1] - field[i, j - 1]) * inv_2dy
    elseif up
        up2 = j + 2 <= Ny && !is_solid[i, j + 2]
        return up2 ?
               (-T(3) * field[i, j] + T(4) * field[i, j + 1] - field[i, j + 2]) * inv_2dy :
               (field[i, j + 1] - field[i, j]) * inv_dy
    elseif down
        down2 = j - 2 >= 1 && !is_solid[i, j - 2]
        return down2 ?
               (T(3) * field[i, j] - T(4) * field[i, j - 1] + field[i, j - 2]) * inv_2dy :
               (field[i, j] - field[i, j - 1]) * inv_dy
    else
        return zero(T)
    end
end

@inline function _logfv_solid_aware_second_derivative_x_2d(field, is_solid, i, j, Nx, inv_dx2)
    T = eltype(field)
    left = i > 1 && !is_solid[i - 1, j]
    right = i < Nx && !is_solid[i + 1, j]
    if left && right
        return (field[i + 1, j] - T(2) * field[i, j] + field[i - 1, j]) * inv_dx2
    elseif right && i + 2 <= Nx && !is_solid[i + 2, j]
        return (field[i, j] - T(2) * field[i + 1, j] + field[i + 2, j]) * inv_dx2
    elseif left && i - 2 >= 1 && !is_solid[i - 2, j]
        return (field[i, j] - T(2) * field[i - 1, j] + field[i - 2, j]) * inv_dx2
    else
        return zero(T)
    end
end

@inline function _logfv_solid_aware_second_derivative_y_2d(field, is_solid, i, j, Ny, inv_dy2)
    T = eltype(field)
    down = j > 1 && !is_solid[i, j - 1]
    up = j < Ny && !is_solid[i, j + 1]
    if down && up
        return (field[i, j + 1] - T(2) * field[i, j] + field[i, j - 1]) * inv_dy2
    elseif up && j + 2 <= Ny && !is_solid[i, j + 2]
        return (field[i, j] - T(2) * field[i, j + 1] + field[i, j + 2]) * inv_dy2
    elseif down && j - 2 >= 1 && !is_solid[i, j - 2]
        return (field[i, j] - T(2) * field[i, j - 1] + field[i, j - 2]) * inv_dy2
    else
        return zero(T)
    end
end

@kernel function logfv_velocity_gradient_solid_aware_2d_kernel!(
    dudx, dudy, dvdx, dvdy,
    @Const(ux), @Const(uy), @Const(is_solid),
    inv_dx, inv_dy, inv_2dx, inv_2dy, Nx, Ny,
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
                dudx[i, j] = _logfv_solid_aware_derivative_x_2d(ux, is_solid, i, j, Nx, inv_dx, inv_2dx)
                dudy[i, j] = _logfv_solid_aware_derivative_y_2d(ux, is_solid, i, j, Ny, inv_dy, inv_2dy)
                dvdx[i, j] = _logfv_solid_aware_derivative_x_2d(uy, is_solid, i, j, Nx, inv_dx, inv_2dx)
                dvdy[i, j] = _logfv_solid_aware_derivative_y_2d(uy, is_solid, i, j, Ny, inv_dy, inv_2dy)
            end
        end
    end
end

function logfv_velocity_gradient_solid_aware_2d!(
    dudx, dudy, dvdx, dvdy,
    ux, uy, is_solid, dx, dy;
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(ux)
    Nx, Ny = size(ux)
    kernel! = logfv_velocity_gradient_solid_aware_2d_kernel!(backend)
    kernel!(
        dudx, dudy, dvdx, dvdy,
        ux, uy, is_solid, inv(dx), inv(dy), inv(2 * dx), inv(2 * dy), Nx, Ny;
        ndrange=(Nx, Ny),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

function logfv_velocity_gradient_bc_aware_2d!(
    dudx, dudy, dvdx, dvdy,
    ux, uy, is_solid, dx, dy, bc::LogFVDomainBC2D;
    sync::Bool=true,
)
    return fvfd_velocity_gradient_2d!(
        dudx, dudy, dvdx, dvdy, ux, uy, is_solid, dx, dy, bc; sync,
    )
end

function logfv_velocity_gradient_embedded_bc_aware_2d!(
    dudx, dudy, dvdx, dvdy,
    ux, uy, is_solid, dx, dy, bc::LogFVDomainBC2D,
    embedded::LogFVEmbeddedBoundary2D;
    sync::Bool=true,
)
    return fvfd_velocity_gradient_embedded_2d!(
        dudx, dudy, dvdx, dvdy, ux, uy, is_solid, dx, dy, bc, embedded; sync,
    )
end

