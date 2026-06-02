@enum WallGradientOrder LINEAR QUADRATIC

struct WallGradientSides
    south::Union{Nothing,AbstractVector}
    north::Union{Nothing,AbstractVector}
    east::Union{Nothing,AbstractVector}
    west::Union{Nothing,AbstractVector}
end

@kernel function _halfway_wall_gradient_correction_kernel!(
    dudx, dudy, dvdx, dvdy,
    @Const(ux), @Const(uy),
    @Const(south_profile), @Const(north_profile),
    @Const(east_profile), @Const(west_profile),
    inv_dx_half, inv_dy_half, inv_dx, inv_dy,
    Nx, Ny, order::Val, skip_top_corners,
)
    i, j = @index(Global, NTuple)
    T = eltype(dudx)
    @inbounds begin
        if i <= Nx && j <= Ny
            if j == 1 && south_profile !== nothing
                u_wall = T(south_profile[i])
                if order isa Val{:linear}
                    dudy[i, j] = (ux[i, j] - u_wall) * inv_dy_half
                    dvdy[i, j] = uy[i, j] * inv_dy_half
                elseif order isa Val{:quadratic}
                    dudy[i, j] = (T(3) * ux[i, 1] - ux[i, 2] / T(3) -
                                  (T(8) / T(3)) * u_wall) * inv_dy
                    dvdy[i, j] = (T(3) * uy[i, 1] - uy[i, 2] / T(3)) * inv_dy
                end
            end

            if j == Ny && north_profile !== nothing &&
               !(skip_top_corners && (i == 1 || i == Nx))
                u_wall = T(north_profile[i])
                if order isa Val{:linear}
                    dudy[i, j] = (u_wall - ux[i, j]) * inv_dy_half
                    dvdy[i, j] = -uy[i, j] * inv_dy_half
                elseif order isa Val{:quadratic}
                    dudy[i, j] = ((T(8) / T(3)) * u_wall -
                                  T(3) * ux[i, Ny] + ux[i, Ny - 1] / T(3)) * inv_dy
                    dvdy[i, j] = (-T(3) * uy[i, Ny] + uy[i, Ny - 1] / T(3)) * inv_dy
                end
            end

            if i == 1 && west_profile !== nothing
                v_wall = T(west_profile[j])
                if order isa Val{:linear}
                    dudx[i, j] = ux[i, j] * inv_dx_half
                    dvdx[i, j] = (uy[i, j] - v_wall) * inv_dx_half
                elseif order isa Val{:quadratic}
                    dudx[i, j] = (T(3) * ux[1, j] - ux[2, j] / T(3)) * inv_dx
                    dvdx[i, j] = (T(3) * uy[1, j] - uy[2, j] / T(3) -
                                  (T(8) / T(3)) * v_wall) * inv_dx
                end
            end

            if i == Nx && east_profile !== nothing
                v_wall = T(east_profile[j])
                if order isa Val{:linear}
                    dudx[i, j] = -ux[i, j] * inv_dx_half
                    dvdx[i, j] = (v_wall - uy[i, j]) * inv_dx_half
                elseif order isa Val{:quadratic}
                    dudx[i, j] = (-T(3) * ux[Nx, j] + ux[Nx - 1, j] / T(3)) * inv_dx
                    dvdx[i, j] = ((T(8) / T(3)) * v_wall -
                                  T(3) * uy[Nx, j] + uy[Nx - 1, j] / T(3)) * inv_dx
                end
            end
        end
    end
end

function _halfway_wall_gradient_order_val(order::Symbol)
    order === :linear && return Val(:linear)
    order === :quadratic && return Val(:quadratic)
    throw(ArgumentError("order must be :linear or :quadratic"))
end

function apply_halfway_wall_gradient_correction!(
    dudx, dudy, dvdx, dvdy, ux, uy,
    sides::WallGradientSides, dx::Real, dy::Real;
    order::Symbol=:quadratic,
    sync::Bool=true,
    skip_top_corners::Bool=false,
)
    Nx, Ny = size(dudx)
    size(dudy) == (Nx, Ny) && size(dvdx) == (Nx, Ny) && size(dvdy) == (Nx, Ny) ||
        throw(DimensionMismatch("gradient arrays must have matching sizes"))
    size(ux) == (Nx, Ny) && size(uy) == (Nx, Ny) ||
        throw(DimensionMismatch("velocity arrays must match gradient arrays"))
    if order === :quadratic
        Nx >= 2 && Ny >= 2 ||
            throw(ArgumentError("quadratic halfway wall correction requires Nx >= 2 and Ny >= 2"))
    end
    sides.south !== nothing && length(sides.south) == Nx ||
        sides.south === nothing || throw(DimensionMismatch("south profile length must equal Nx"))
    sides.north !== nothing && length(sides.north) == Nx ||
        sides.north === nothing || throw(DimensionMismatch("north profile length must equal Nx"))
    sides.east !== nothing && length(sides.east) == Ny ||
        sides.east === nothing || throw(DimensionMismatch("east profile length must equal Ny"))
    sides.west !== nothing && length(sides.west) == Ny ||
        sides.west === nothing || throw(DimensionMismatch("west profile length must equal Ny"))

    backend = KernelAbstractions.get_backend(dudx)
    T = eltype(dudx)
    kernel! = _halfway_wall_gradient_correction_kernel!(backend)
    kernel!(
        dudx, dudy, dvdx, dvdy, ux, uy,
        sides.south, sides.north, sides.east, sides.west,
        T(2) / T(dx), T(2) / T(dy), inv(T(dx)), inv(T(dy)),
        Nx, Ny, _halfway_wall_gradient_order_val(order), skip_top_corners;
        ndrange=(Nx, Ny),
    )
    sync && KernelAbstractions.synchronize(backend)
    return nothing
end
