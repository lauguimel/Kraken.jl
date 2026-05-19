@inline function _fvfd_bc_index_1d(idx, n, lower_bc, upper_bc)
    if 1 <= idx <= n
        return idx
    elseif idx < 1 && lower_bc == FVFD_BC_PERIODIC
        return idx + n
    elseif idx > n && upper_bc == FVFD_BC_PERIODIC
        return idx - n
    else
        return 0
    end
end

@inline function _fvfd_solid_bc_derivative_x_2d(
    field, is_solid, i, j, Nx, inv_dx, inv_2dx, west_bc, east_bc,
    polymer_wall_extrap::Val=Val(:quadratic),
)
    T = eltype(field)
    li = _fvfd_bc_index_1d(i - 1, Nx, west_bc, east_bc)
    ri = _fvfd_bc_index_1d(i + 1, Nx, west_bc, east_bc)
    left = li != 0 && !is_solid[li, j]
    right = ri != 0 && !is_solid[ri, j]
    if left && right
        return (field[ri, j] - field[li, j]) * inv_2dx
    elseif right
        if polymer_wall_extrap isa Val{:linear}
            return (field[ri, j] - field[i, j]) * inv_dx
        end
        r2i = _fvfd_bc_index_1d(i + 2, Nx, west_bc, east_bc)
        return (r2i != 0 && !is_solid[r2i, j]) ?
               (-T(3) * field[i, j] + T(4) * field[ri, j] - field[r2i, j]) * inv_2dx :
               (field[ri, j] - field[i, j]) * inv_dx
    elseif left
        if polymer_wall_extrap isa Val{:linear}
            return (field[i, j] - field[li, j]) * inv_dx
        end
        l2i = _fvfd_bc_index_1d(i - 2, Nx, west_bc, east_bc)
        return (l2i != 0 && !is_solid[l2i, j]) ?
               (T(3) * field[i, j] - T(4) * field[li, j] + field[l2i, j]) * inv_2dx :
               (field[i, j] - field[li, j]) * inv_dx
    else
        return zero(T)
    end
end

@inline function _fvfd_solid_bc_derivative_y_2d(
    field, is_solid, i, j, Ny, inv_dy, inv_2dy, south_bc, north_bc,
    polymer_wall_extrap::Val=Val(:quadratic),
)
    T = eltype(field)
    dj = _fvfd_bc_index_1d(j - 1, Ny, south_bc, north_bc)
    uj = _fvfd_bc_index_1d(j + 1, Ny, south_bc, north_bc)
    down = dj != 0 && !is_solid[i, dj]
    up = uj != 0 && !is_solid[i, uj]
    if down && up
        return (field[i, uj] - field[i, dj]) * inv_2dy
    elseif up
        if polymer_wall_extrap isa Val{:linear}
            return (field[i, uj] - field[i, j]) * inv_dy
        end
        u2j = _fvfd_bc_index_1d(j + 2, Ny, south_bc, north_bc)
        return (u2j != 0 && !is_solid[i, u2j]) ?
               (-T(3) * field[i, j] + T(4) * field[i, uj] - field[i, u2j]) * inv_2dy :
               (field[i, uj] - field[i, j]) * inv_dy
    elseif down
        if polymer_wall_extrap isa Val{:linear}
            return (field[i, j] - field[i, dj]) * inv_dy
        end
        d2j = _fvfd_bc_index_1d(j - 2, Ny, south_bc, north_bc)
        return (d2j != 0 && !is_solid[i, d2j]) ?
               (T(3) * field[i, j] - T(4) * field[i, dj] + field[i, d2j]) * inv_2dy :
               (field[i, j] - field[i, dj]) * inv_dy
    else
        return zero(T)
    end
end

@inline function _fvfd_solid_bc_second_derivative_x_2d(
    field, is_solid, i, j, Nx, inv_dx2, west_bc, east_bc,
)
    T = eltype(field)
    li = _fvfd_bc_index_1d(i - 1, Nx, west_bc, east_bc)
    ri = _fvfd_bc_index_1d(i + 1, Nx, west_bc, east_bc)
    left = li != 0 && !is_solid[li, j]
    right = ri != 0 && !is_solid[ri, j]
    if left && right
        return (field[ri, j] - T(2) * field[i, j] + field[li, j]) * inv_dx2
    elseif right
        r2i = _fvfd_bc_index_1d(i + 2, Nx, west_bc, east_bc)
        return (r2i != 0 && !is_solid[r2i, j]) ?
               (field[i, j] - T(2) * field[ri, j] + field[r2i, j]) * inv_dx2 :
               zero(T)
    elseif left
        l2i = _fvfd_bc_index_1d(i - 2, Nx, west_bc, east_bc)
        return (l2i != 0 && !is_solid[l2i, j]) ?
               (field[i, j] - T(2) * field[li, j] + field[l2i, j]) * inv_dx2 :
               zero(T)
    else
        return zero(T)
    end
end

@inline function _fvfd_solid_bc_second_derivative_y_2d(
    field, is_solid, i, j, Ny, inv_dy2, south_bc, north_bc,
)
    T = eltype(field)
    dj = _fvfd_bc_index_1d(j - 1, Ny, south_bc, north_bc)
    uj = _fvfd_bc_index_1d(j + 1, Ny, south_bc, north_bc)
    down = dj != 0 && !is_solid[i, dj]
    up = uj != 0 && !is_solid[i, uj]
    if down && up
        return (field[i, uj] - T(2) * field[i, j] + field[i, dj]) * inv_dy2
    elseif up
        u2j = _fvfd_bc_index_1d(j + 2, Ny, south_bc, north_bc)
        return (u2j != 0 && !is_solid[i, u2j]) ?
               (field[i, j] - T(2) * field[i, uj] + field[i, u2j]) * inv_dy2 :
               zero(T)
    elseif down
        d2j = _fvfd_bc_index_1d(j - 2, Ny, south_bc, north_bc)
        return (d2j != 0 && !is_solid[i, d2j]) ?
               (field[i, j] - T(2) * field[i, dj] + field[i, d2j]) * inv_dy2 :
               zero(T)
    else
        return zero(T)
    end
end

@inline function _fvfd_apply_embedded_wall_gradient_2d(
    gx, gy, phi, wall_nx, wall_ny, wall_inv_distance, i, j,
)
    inv_distance = wall_inv_distance[i, j]
    if inv_distance > zero(inv_distance)
        nx = wall_nx[i, j]
        ny = wall_ny[i, j]
        target_normal_derivative = phi[i, j] * inv_distance
        current_normal_derivative = gx * nx + gy * ny
        correction = target_normal_derivative - current_normal_derivative
        return gx + correction * nx, gy + correction * ny
    end
    return gx, gy
end

@inline function _fvfd_xface_average_or_zero_2d(ux, is_solid, i_left, i_right, j)
    T = eltype(ux)
    return (is_solid[i_left, j] || is_solid[i_right, j]) ?
           zero(T) :
           (ux[i_left, j] + ux[i_right, j]) / T(2)
end

@inline function _fvfd_xface_fraction_2d(
    is_solid, west_fraction, east_fraction, i_left, i_right, j,
)
    T = eltype(west_fraction)
    return (is_solid[i_left, j] || is_solid[i_right, j]) ?
           zero(T) :
           min(east_fraction[i_left, j], west_fraction[i_right, j])
end

@inline function _fvfd_yface_average_or_zero_2d(uy, is_solid, i, j_down, j_up)
    T = eltype(uy)
    return (is_solid[i, j_down] || is_solid[i, j_up]) ?
           zero(T) :
           (uy[i, j_down] + uy[i, j_up]) / T(2)
end

@inline function _fvfd_yface_fraction_2d(
    is_solid, south_fraction, north_fraction, i, j_down, j_up,
)
    T = eltype(south_fraction)
    return (is_solid[i, j_down] || is_solid[i, j_up]) ?
           zero(T) :
           min(north_fraction[i, j_down], south_fraction[i, j_up])
end

@inline function _fvfd_xface_scalar_average_or_zero_2d(field, is_solid, i_left, i_right, j)
    T = eltype(field)
    return (is_solid[i_left, j] || is_solid[i_right, j]) ?
           zero(T) :
           (field[i_left, j] + field[i_right, j]) / T(2)
end

@inline function _fvfd_yface_scalar_average_or_zero_2d(field, is_solid, i, j_down, j_up)
    T = eltype(field)
    return (is_solid[i, j_down] || is_solid[i, j_up]) ?
           zero(T) :
           (field[i, j_down] + field[i, j_up]) / T(2)
end

@kernel function fvfd_cell_velocity_to_faces_2d_kernel!(
    ux_face, uy_face,
    @Const(ux), @Const(uy), @Const(is_solid),
    @Const(ux_west), @Const(ux_east),
    @Const(uy_south), @Const(uy_north),
    west_bc, east_bc, south_bc, north_bc, Nx, Ny,
)
    I, J = @index(Global, NTuple)
    @inbounds begin
        if I <= Nx + 1 && J <= Ny
            if I == 1
                if west_bc == FVFD_BC_PERIODIC
                    ux_face[I, J] = _fvfd_xface_average_or_zero_2d(ux, is_solid, Nx, 1, J)
                elseif west_bc == FVFD_BC_OPEN
                    ux_face[I, J] = is_solid[1, J] ? zero(eltype(ux_face)) : ux_west[J]
                else
                    ux_face[I, J] = zero(eltype(ux_face))
                end
            elseif I == Nx + 1
                if east_bc == FVFD_BC_PERIODIC
                    ux_face[I, J] = _fvfd_xface_average_or_zero_2d(ux, is_solid, Nx, 1, J)
                elseif east_bc == FVFD_BC_OPEN
                    ux_face[I, J] = is_solid[Nx, J] ? zero(eltype(ux_face)) : ux_east[J]
                else
                    ux_face[I, J] = zero(eltype(ux_face))
                end
            else
                ux_face[I, J] = _fvfd_xface_average_or_zero_2d(ux, is_solid, I - 1, I, J)
            end
        end

        if I <= Nx && J <= Ny + 1
            if J == 1
                if south_bc == FVFD_BC_PERIODIC
                    uy_face[I, J] = _fvfd_yface_average_or_zero_2d(uy, is_solid, I, Ny, 1)
                elseif south_bc == FVFD_BC_OPEN
                    uy_face[I, J] = is_solid[I, 1] ? zero(eltype(uy_face)) : uy_south[I]
                else
                    uy_face[I, J] = zero(eltype(uy_face))
                end
            elseif J == Ny + 1
                if north_bc == FVFD_BC_PERIODIC
                    uy_face[I, J] = _fvfd_yface_average_or_zero_2d(uy, is_solid, I, Ny, 1)
                elseif north_bc == FVFD_BC_OPEN
                    uy_face[I, J] = is_solid[I, Ny] ? zero(eltype(uy_face)) : uy_north[I]
                else
                    uy_face[I, J] = zero(eltype(uy_face))
                end
            else
                uy_face[I, J] = _fvfd_yface_average_or_zero_2d(uy, is_solid, I, J - 1, J)
            end
        end
    end
end

@kernel function fvfd_cell_velocity_to_faces_embedded_2d_kernel!(
    ux_face, uy_face,
    @Const(ux), @Const(uy), @Const(is_solid),
    @Const(west_fraction), @Const(east_fraction),
    @Const(south_fraction), @Const(north_fraction),
    @Const(ux_west), @Const(ux_east),
    @Const(uy_south), @Const(uy_north),
    west_bc, east_bc, south_bc, north_bc, Nx, Ny,
)
    I, J = @index(Global, NTuple)
    @inbounds begin
        if I <= Nx + 1 && J <= Ny
            if I == 1
                if west_bc == FVFD_BC_PERIODIC
                    frac = _fvfd_xface_fraction_2d(
                        is_solid, west_fraction, east_fraction, Nx, 1, J,
                    )
                    ux_face[I, J] = frac * _fvfd_xface_average_or_zero_2d(ux, is_solid, Nx, 1, J)
                elseif west_bc == FVFD_BC_OPEN
                    ux_face[I, J] = is_solid[1, J] ? zero(eltype(ux_face)) :
                                    west_fraction[1, J] * ux_west[J]
                else
                    ux_face[I, J] = zero(eltype(ux_face))
                end
            elseif I == Nx + 1
                if east_bc == FVFD_BC_PERIODIC
                    frac = _fvfd_xface_fraction_2d(
                        is_solid, west_fraction, east_fraction, Nx, 1, J,
                    )
                    ux_face[I, J] = frac * _fvfd_xface_average_or_zero_2d(ux, is_solid, Nx, 1, J)
                elseif east_bc == FVFD_BC_OPEN
                    ux_face[I, J] = is_solid[Nx, J] ? zero(eltype(ux_face)) :
                                    east_fraction[Nx, J] * ux_east[J]
                else
                    ux_face[I, J] = zero(eltype(ux_face))
                end
            else
                frac = _fvfd_xface_fraction_2d(
                    is_solid, west_fraction, east_fraction, I - 1, I, J,
                )
                ux_face[I, J] = frac * _fvfd_xface_average_or_zero_2d(
                    ux, is_solid, I - 1, I, J,
                )
            end
        end

        if I <= Nx && J <= Ny + 1
            if J == 1
                if south_bc == FVFD_BC_PERIODIC
                    frac = _fvfd_yface_fraction_2d(
                        is_solid, south_fraction, north_fraction, I, Ny, 1,
                    )
                    uy_face[I, J] = frac * _fvfd_yface_average_or_zero_2d(uy, is_solid, I, Ny, 1)
                elseif south_bc == FVFD_BC_OPEN
                    uy_face[I, J] = is_solid[I, 1] ? zero(eltype(uy_face)) :
                                    south_fraction[I, 1] * uy_south[I]
                else
                    uy_face[I, J] = zero(eltype(uy_face))
                end
            elseif J == Ny + 1
                if north_bc == FVFD_BC_PERIODIC
                    frac = _fvfd_yface_fraction_2d(
                        is_solid, south_fraction, north_fraction, I, Ny, 1,
                    )
                    uy_face[I, J] = frac * _fvfd_yface_average_or_zero_2d(uy, is_solid, I, Ny, 1)
                elseif north_bc == FVFD_BC_OPEN
                    uy_face[I, J] = is_solid[I, Ny] ? zero(eltype(uy_face)) :
                                    north_fraction[I, Ny] * uy_north[I]
                else
                    uy_face[I, J] = zero(eltype(uy_face))
                end
            else
                frac = _fvfd_yface_fraction_2d(
                    is_solid, south_fraction, north_fraction, I, J - 1, J,
                )
                uy_face[I, J] = frac * _fvfd_yface_average_or_zero_2d(
                    uy, is_solid, I, J - 1, J,
                )
            end
        end
    end
end

function fvfd_cell_velocity_to_faces_2d!(
    ux_face, uy_face, ux, uy, is_solid,
    ux_west, ux_east, uy_south, uy_north,
    bc::FVFDDomainBC2D;
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(ux_face)
    Nx, Ny = size(ux)
    bc.west == FVFD_BC_OPEN &&
        _fvfd_check_boundary_length(:ux_west, ux_west, Ny)
    bc.east == FVFD_BC_OPEN &&
        _fvfd_check_boundary_length(:ux_east, ux_east, Ny)
    bc.south == FVFD_BC_OPEN &&
        _fvfd_check_boundary_length(:uy_south, uy_south, Nx)
    bc.north == FVFD_BC_OPEN &&
        _fvfd_check_boundary_length(:uy_north, uy_north, Nx)
    kernel! = fvfd_cell_velocity_to_faces_2d_kernel!(backend)
    kernel!(
        ux_face, uy_face, ux, uy, is_solid,
        ux_west, ux_east, uy_south, uy_north,
        bc.west, bc.east, bc.south, bc.north, Nx, Ny;
        ndrange=(Nx + 1, Ny + 1),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

function fvfd_cell_velocity_to_faces_embedded_2d!(
    ux_face, uy_face, ux, uy, is_solid,
    embedded::FVFDEmbeddedBoundary2D,
    ux_west, ux_east, uy_south, uy_north,
    bc::FVFDDomainBC2D;
    sync::Bool=true,
)
    backend = KernelAbstractions.get_backend(ux_face)
    Nx, Ny = size(ux)
    bc.west == FVFD_BC_OPEN &&
        _fvfd_check_boundary_length(:ux_west, ux_west, Ny)
    bc.east == FVFD_BC_OPEN &&
        _fvfd_check_boundary_length(:ux_east, ux_east, Ny)
    bc.south == FVFD_BC_OPEN &&
        _fvfd_check_boundary_length(:uy_south, uy_south, Nx)
    bc.north == FVFD_BC_OPEN &&
        _fvfd_check_boundary_length(:uy_north, uy_north, Nx)
    kernel! = fvfd_cell_velocity_to_faces_embedded_2d_kernel!(backend)
    kernel!(
        ux_face, uy_face, ux, uy, is_solid,
        embedded.west_fraction, embedded.east_fraction,
        embedded.south_fraction, embedded.north_fraction,
        ux_west, ux_east, uy_south, uy_north,
        bc.west, bc.east, bc.south, bc.north, Nx, Ny;
        ndrange=(Nx + 1, Ny + 1),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

function fvfd_cell_velocity_to_faces_embedded_2d!(
    ux_face, uy_face, ux, uy,
    geometry::FVFDGeometry2D,
    ux_bc::FVFDFieldBC2D, uy_bc::FVFDFieldBC2D;
    sync::Bool=true,
)
    return fvfd_cell_velocity_to_faces_embedded_2d!(
        ux_face, uy_face, ux, uy,
        geometry.is_solid, geometry.embedded,
        ux_bc.west, ux_bc.east, uy_bc.south, uy_bc.north,
        geometry.bc; sync,
    )
end

function fvfd_cell_velocity_to_faces_2d!(
    ux_face, uy_face, ux, uy, is_solid,
    ux_bc::FVFDFieldBC2D, uy_bc::FVFDFieldBC2D,
    bc::FVFDDomainBC2D;
    sync::Bool=true,
)
    return fvfd_cell_velocity_to_faces_2d!(
        ux_face, uy_face, ux, uy, is_solid,
        ux_bc.west, ux_bc.east, uy_bc.south, uy_bc.north,
        bc; sync,
    )
end

function fvfd_cell_velocity_to_faces_2d!(
    ux_face, uy_face, ux, uy,
    geometry::FVFDGeometry2D,
    ux_bc::FVFDFieldBC2D, uy_bc::FVFDFieldBC2D;
    sync::Bool=true,
)
    return fvfd_cell_velocity_to_faces_2d!(
        ux_face, uy_face, ux, uy,
        geometry.is_solid, ux_bc, uy_bc, geometry.bc; sync,
    )
end

@inline function _fvfd_bc_east_scalar_2d(phi, east_phi, i, j, Nx, east_bc)
    if i < Nx
        return phi[i + 1, j]
    elseif east_bc == FVFD_BC_PERIODIC
        return phi[1, j]
    elseif east_bc == FVFD_BC_OPEN
        return east_phi[j]
    else
        return phi[i, j]
    end
end

@inline function _fvfd_bc_west_scalar_2d(phi, west_phi, i, j, Nx, west_bc)
    if i > 1
        return phi[i - 1, j]
    elseif west_bc == FVFD_BC_PERIODIC
        return phi[Nx, j]
    elseif west_bc == FVFD_BC_OPEN
        return west_phi[j]
    else
        return phi[i, j]
    end
end

@inline function _fvfd_bc_north_scalar_2d(phi, north_phi, i, j, Ny, north_bc)
    if j < Ny
        return phi[i, j + 1]
    elseif north_bc == FVFD_BC_PERIODIC
        return phi[i, 1]
    elseif north_bc == FVFD_BC_OPEN
        return north_phi[i]
    else
        return phi[i, j]
    end
end

@inline function _fvfd_bc_south_scalar_2d(phi, south_phi, i, j, Ny, south_bc)
    if j > 1
        return phi[i, j - 1]
    elseif south_bc == FVFD_BC_PERIODIC
        return phi[i, Ny]
    elseif south_bc == FVFD_BC_OPEN
        return south_phi[i]
    else
        return phi[i, j]
    end
end

function _fvfd_advection_scheme_val(advection_scheme::Symbol)
    scheme = Symbol(replace(lowercase(String(advection_scheme)), '-' => '_'))
    scheme in (:rusanov, :muscl_superbee) ||
        throw(ArgumentError("advection_scheme must be :rusanov or :muscl_superbee"))
    return Val(scheme)
end

@inline function _fvfd_superbee_limiter_2d(r)
    one_r = one(r)
    two_r = one_r + one_r
    return max(zero(r), max(min(two_r * r, one_r), min(r, two_r)))
end

@inline function _fvfd_muscl_superbee_face_value_2d(far_upwind, upwind, downwind)
    d_up = upwind - far_upwind
    d_down = downwind - upwind
    r = ifelse(d_down == zero(d_down), zero(d_down), d_up / d_down)
    return upwind + (one(r) / (one(r) + one(r))) * _fvfd_superbee_limiter_2d(r) * d_down
end

@inline function _fvfd_upwind_scalar_advective_rhs_2d(
    phi, west_phi, east_phi, south_phi, north_phi,
    ux_face, uy_face, is_solid, i, j, Nx, Ny, inv_dx, inv_dy,
    west_bc, east_bc, south_bc, north_bc,
    ::Val{:rusanov},
)
    ue = ux_face[i + 1, j]
    uw = ux_face[i, j]
    vn = uy_face[i, j + 1]
    vs = uy_face[i, j]

    east_value = _fvfd_bc_east_scalar_2d(phi, east_phi, i, j, Nx, east_bc)
    west_value = _fvfd_bc_west_scalar_2d(phi, west_phi, i, j, Nx, west_bc)
    north_value = _fvfd_bc_north_scalar_2d(phi, north_phi, i, j, Ny, north_bc)
    south_value = _fvfd_bc_south_scalar_2d(phi, south_phi, i, j, Ny, south_bc)

    phie = ifelse(ue >= 0, phi[i, j], east_value)
    phiw = ifelse(uw >= 0, west_value, phi[i, j])
    phin = ifelse(vn >= 0, phi[i, j], north_value)
    phis = ifelse(vs >= 0, south_value, phi[i, j])

    flux_div = (ue * phie - uw * phiw) * inv_dx +
               (vn * phin - vs * phis) * inv_dy
    divu = (ue - uw) * inv_dx + (vn - vs) * inv_dy
    return -(flux_div - phi[i, j] * divu)
end

@inline function _fvfd_upwind_scalar_advective_rhs_2d(
    phi, west_phi, east_phi, south_phi, north_phi,
    ux_face, uy_face, is_solid, i, j, Nx, Ny, inv_dx, inv_dy,
    west_bc, east_bc, south_bc, north_bc,
    ::Val{:muscl_superbee},
)
    if i <= 2 || i >= Nx - 1 || j <= 2 || j >= Ny - 1 ||
       is_solid[i - 2, j] || is_solid[i - 1, j] ||
       is_solid[i + 1, j] || is_solid[i + 2, j] ||
       is_solid[i, j - 2] || is_solid[i, j - 1] ||
       is_solid[i, j + 1] || is_solid[i, j + 2]
        return _fvfd_upwind_scalar_advective_rhs_2d(
            phi, west_phi, east_phi, south_phi, north_phi,
            ux_face, uy_face, is_solid, i, j, Nx, Ny, inv_dx, inv_dy,
            west_bc, east_bc, south_bc, north_bc, Val(:rusanov),
        )
    end

    ue = ux_face[i + 1, j]
    uw = ux_face[i, j]
    vn = uy_face[i, j + 1]
    vs = uy_face[i, j]

    phie = ifelse(
        ue >= 0,
        _fvfd_muscl_superbee_face_value_2d(phi[i - 1, j], phi[i, j], phi[i + 1, j]),
        _fvfd_muscl_superbee_face_value_2d(phi[i + 2, j], phi[i + 1, j], phi[i, j]),
    )
    phiw = ifelse(
        uw >= 0,
        _fvfd_muscl_superbee_face_value_2d(phi[i - 2, j], phi[i - 1, j], phi[i, j]),
        _fvfd_muscl_superbee_face_value_2d(phi[i + 1, j], phi[i, j], phi[i - 1, j]),
    )
    phin = ifelse(
        vn >= 0,
        _fvfd_muscl_superbee_face_value_2d(phi[i, j - 1], phi[i, j], phi[i, j + 1]),
        _fvfd_muscl_superbee_face_value_2d(phi[i, j + 2], phi[i, j + 1], phi[i, j]),
    )
    phis = ifelse(
        vs >= 0,
        _fvfd_muscl_superbee_face_value_2d(phi[i, j - 2], phi[i, j - 1], phi[i, j]),
        _fvfd_muscl_superbee_face_value_2d(phi[i, j + 1], phi[i, j], phi[i, j - 1]),
    )

    flux_div = (ue * phie - uw * phiw) * inv_dx +
               (vn * phin - vs * phis) * inv_dy
    divu = (ue - uw) * inv_dx + (vn - vs) * inv_dy
    return -(flux_div - phi[i, j] * divu)
end

@kernel function fvfd_advect_upwind_2d_kernel!(
    phi_out, @Const(phi),
    @Const(west_phi), @Const(east_phi), @Const(south_phi), @Const(north_phi),
    @Const(ux_face), @Const(uy_face), @Const(is_solid),
    dt, inv_dx, inv_dy,
    west_bc, east_bc, south_bc, north_bc, Nx, Ny, advection_scheme,
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if i <= Nx && j <= Ny
            if is_solid[i, j]
                phi_out[i, j] = zero(eltype(phi_out))
            else
                rhs = _fvfd_upwind_scalar_advective_rhs_2d(
                    phi, west_phi, east_phi, south_phi, north_phi,
                    ux_face, uy_face, is_solid, i, j, Nx, Ny, inv_dx, inv_dy,
                    west_bc, east_bc, south_bc, north_bc, advection_scheme,
                )
                phi_out[i, j] = phi[i, j] + dt * rhs
            end
        end
    end
end

function fvfd_advect_upwind_2d!(
    phi_out, phi, phi_bc::FVFDFieldBC2D,
    ux_face, uy_face, is_solid, dx, dy, bc::FVFDDomainBC2D, dt;
    sync::Bool=true,
    advection_scheme::Symbol=:rusanov,
)
    backend = KernelAbstractions.get_backend(phi_out)
    Nx, Ny = size(phi_out)
    fvfd_validate_field_bc_2d(phi_bc, Nx, Ny, bc; name=:phi_bc)
    scheme = _fvfd_advection_scheme_val(advection_scheme)
    kernel! = fvfd_advect_upwind_2d_kernel!(backend)
    kernel!(
        phi_out, phi,
        phi_bc.west, phi_bc.east, phi_bc.south, phi_bc.north,
        ux_face, uy_face, is_solid,
        dt, inv(dx), inv(dy),
        bc.west, bc.east, bc.south, bc.north, Nx, Ny, scheme;
        ndrange=(Nx, Ny),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end

function fvfd_advect_upwind_2d!(
    phi_out, phi, phi_bc::FVFDFieldBC2D,
    ux_face, uy_face, geometry::FVFDGeometry2D, dt;
    sync::Bool=true,
    advection_scheme::Symbol=:rusanov,
)
    return fvfd_advect_upwind_2d!(
        phi_out, phi, phi_bc,
        ux_face, uy_face, geometry.is_solid,
        geometry.patch.dx, geometry.patch.dy, geometry.bc, dt; sync, advection_scheme,
    )
end

function fvfd_advect_upwind_embedded_2d!(
    phi_out, phi, phi_bc::FVFDFieldBC2D,
    ux_face, uy_face, ux, uy,
    geometry::FVFDGeometry2D,
    ux_bc::FVFDFieldBC2D, uy_bc::FVFDFieldBC2D,
    dt;
    sync::Bool=true,
    advection_scheme::Symbol=:rusanov,
)
    fvfd_cell_velocity_to_faces_embedded_2d!(
        ux_face, uy_face, ux, uy, geometry, ux_bc, uy_bc; sync=false,
    )
    fvfd_advect_upwind_2d!(
        phi_out, phi, phi_bc, ux_face, uy_face, geometry, dt; sync=false,
        advection_scheme,
    )
    sync && KernelAbstractions.synchronize(KernelAbstractions.get_backend(phi_out))
    return nothing
end

function fvfd_sym2_advect_upwind_2d!(
    psixx_out, psixy_out, psiyy_out,
    psixx, psixy, psiyy,
    psixx_bc::FVFDFieldBC2D, psixy_bc::FVFDFieldBC2D, psiyy_bc::FVFDFieldBC2D,
    ux_face, uy_face, is_solid, dx, dy, bc::FVFDDomainBC2D, dt;
    sync::Bool=true,
    advection_scheme::Symbol=:rusanov,
)
    fvfd_advect_upwind_2d!(
        psixx_out, psixx, psixx_bc, ux_face, uy_face, is_solid, dx, dy, bc, dt;
        sync=false, advection_scheme,
    )
    fvfd_advect_upwind_2d!(
        psixy_out, psixy, psixy_bc, ux_face, uy_face, is_solid, dx, dy, bc, dt;
        sync=false, advection_scheme,
    )
    fvfd_advect_upwind_2d!(
        psiyy_out, psiyy, psiyy_bc, ux_face, uy_face, is_solid, dx, dy, bc, dt;
        sync=false, advection_scheme,
    )
    sync && KernelAbstractions.synchronize(KernelAbstractions.get_backend(psixx_out))
    return nothing
end

function fvfd_sym2_advect_upwind_2d!(
    psixx_out, psixy_out, psiyy_out,
    psixx, psixy, psiyy,
    psixx_bc::FVFDFieldBC2D, psixy_bc::FVFDFieldBC2D, psiyy_bc::FVFDFieldBC2D,
    ux_face, uy_face, geometry::FVFDGeometry2D, dt;
    sync::Bool=true,
    advection_scheme::Symbol=:rusanov,
)
    return fvfd_sym2_advect_upwind_2d!(
        psixx_out, psixy_out, psiyy_out,
        psixx, psixy, psiyy,
        psixx_bc, psixy_bc, psiyy_bc,
        ux_face, uy_face, geometry.is_solid,
        geometry.patch.dx, geometry.patch.dy, geometry.bc, dt; sync, advection_scheme,
    )
end

function fvfd_sym2_advect_upwind_embedded_2d!(
    psixx_out, psixy_out, psiyy_out,
    psixx, psixy, psiyy,
    psixx_bc::FVFDFieldBC2D, psixy_bc::FVFDFieldBC2D, psiyy_bc::FVFDFieldBC2D,
    ux_face, uy_face, ux, uy,
    geometry::FVFDGeometry2D,
    ux_bc::FVFDFieldBC2D, uy_bc::FVFDFieldBC2D,
    dt;
    sync::Bool=true,
    advection_scheme::Symbol=:rusanov,
)
    fvfd_cell_velocity_to_faces_embedded_2d!(
        ux_face, uy_face, ux, uy, geometry, ux_bc, uy_bc; sync=false,
    )
    fvfd_sym2_advect_upwind_2d!(
        psixx_out, psixy_out, psiyy_out,
        psixx, psixy, psiyy,
        psixx_bc, psixy_bc, psiyy_bc,
        ux_face, uy_face, geometry, dt; sync=false, advection_scheme,
    )
    sync && KernelAbstractions.synchronize(KernelAbstractions.get_backend(psixx_out))
    return nothing
end

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
    @Const(wall_nx), @Const(wall_ny), @Const(wall_inv_distance),
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
                    ux_gx, ux_gy, ux, wall_nx, wall_ny, wall_inv_distance, i, j,
                )
                uy_gx, uy_gy = _fvfd_apply_embedded_wall_gradient_2d(
                    uy_gx, uy_gy, uy, wall_nx, wall_ny, wall_inv_distance, i, j,
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
        embedded.wall_nx, embedded.wall_ny, embedded.wall_inv_distance,
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
