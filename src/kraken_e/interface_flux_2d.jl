struct ScalarFluxField2D{T<:AbstractFloat,AT<:AbstractArray{T,2}}
    east::AT
    north::AT
end

function allocate_scalar_flux_field_2d(::Type{T}=Float64; Nx, Ny) where {T<:AbstractFloat}
    return ScalarFluxField2D{T,Matrix{T}}(
        zeros(T, Nx + 1, Ny),
        zeros(T, Nx, Ny + 1),
    )
end

function compute_same_level_upwind_fluxes_2d!(
    flux::ScalarFluxField2D{T}, U::AbstractArray{T,2},
    vx::T, vy::T;
    skip_east::Bool=false, skip_west::Bool=false,
    closed_x::Bool=true, periodic_y::Bool=true,
) where {T<:AbstractFloat}
    Nx, Ny = size(U)
    size(flux.east) == (Nx + 1, Ny) ||
        throw(DimensionMismatch("east flux shape must be (Nx + 1, Ny)"))
    size(flux.north) == (Nx, Ny + 1) ||
        throw(DimensionMismatch("north flux shape must be (Nx, Ny + 1)"))

    fill!(flux.east, zero(T))
    fill!(flux.north, zero(T))

    for j in 1:Ny, i_face in 2:Nx
        flux.east[i_face, j] = vx > zero(T) ?
            vx * U[i_face - 1, j] :
            vx * U[i_face, j]
    end

    if !closed_x
        throw(ArgumentError("S4 same-level fluxes support closed_x=true only"))
    end
    if skip_west
        @inbounds for j in 1:Ny
            flux.east[1, j] = zero(T)
        end
    end
    if skip_east
        @inbounds for j in 1:Ny
            flux.east[Nx + 1, j] = zero(T)
        end
    end

    if periodic_y
        for i in 1:Nx
            flux.north[i, 1] = vy > zero(T) ? vy * U[i, Ny] : vy * U[i, 1]
            flux.north[i, Ny + 1] = flux.north[i, 1]
        end
        for j_face in 2:Ny, i in 1:Nx
            flux.north[i, j_face] = vy > zero(T) ?
                vy * U[i, j_face - 1] :
                vy * U[i, j_face]
        end
    else
        for j_face in 2:Ny, i in 1:Nx
            flux.north[i, j_face] = vy > zero(T) ?
                vy * U[i, j_face - 1] :
                vy * U[i, j_face]
        end
    end

    return flux
end

@inline function reconstruct_coarse_flux_from_fine_2d(
    record::CFFaceRecord2D{T}, F_fine_1::T, F_fine_2::T,
)::T where {T<:AbstractFloat}
    w1, w2 = record.fine_to_coarse_weights
    return w1 * F_fine_1 + w2 * F_fine_2
end

@inline function cf_flux_telescoping_error(
    record::CFFaceRecord2D{T}, F_fine_1::T, F_fine_2::T,
)::T where {T<:AbstractFloat}
    F_c = reconstruct_coarse_flux_from_fine_2d(record, F_fine_1, F_fine_2)
    return abs(
        F_c * record.coarse_area -
        F_fine_1 * record.fine_areas[1] -
        F_fine_2 * record.fine_areas[2]
    )
end
