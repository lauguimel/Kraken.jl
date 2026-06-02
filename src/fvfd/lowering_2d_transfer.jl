function _fvfd_host_boundary_vector_2d(
    values, expected::Integer, ::Type{T}, name::Symbol;
    active::Bool, default,
) where {T<:AbstractFloat}
    n = Int(expected)
    out = Vector{T}(undef, n)
    if values isa Number
        fill!(out, T(values))
        return out
    end

    observed = try
        length(values)
    catch
        nothing
    end
    if observed == n
        source = values isa AbstractArray ? vec(Array(values)) : values
        @inbounds for idx in 1:n
            out[idx] = T(source[idx])
        end
        return out
    elseif active
        if observed === nothing
            throw(DimensionMismatch(
                "$(name) boundary does not provide a length; expected $(expected)",
            ))
        end
        throw(DimensionMismatch(
            "$(name) boundary length $(observed) does not match expected $(expected)",
        ))
    end

    fill!(out, T(default))
    return out
end

function _fvfd_transfer_boundary_vector_2d(
    values, backend, ::Type{T}, expected::Integer, name::Symbol;
    active::Bool, default,
) where {T<:AbstractFloat}
    host = _fvfd_host_boundary_vector_2d(
        values, expected, T, name; active, default,
    )
    dev = KernelAbstractions.allocate(backend, T, Int(expected))
    copyto!(dev, host)
    return dev
end

function fvfd_transfer_field_bc_2d(
    field_bc::FVFDFieldBC2D, backend, ::Type{T},
    Nx::Integer, Ny::Integer, bc::FVFDDomainBC2D;
    name::Symbol=:field_bc, default=zero(T),
) where {T<:AbstractFloat}
    west = _fvfd_transfer_boundary_vector_2d(
        field_bc.west, backend, T, Ny, Symbol(name, :_west);
        active=bc.west == FVFD_BC_OPEN, default,
    )
    east = _fvfd_transfer_boundary_vector_2d(
        field_bc.east, backend, T, Ny, Symbol(name, :_east);
        active=bc.east == FVFD_BC_OPEN, default,
    )
    south = _fvfd_transfer_boundary_vector_2d(
        field_bc.south, backend, T, Nx, Symbol(name, :_south);
        active=bc.south == FVFD_BC_OPEN, default,
    )
    north = _fvfd_transfer_boundary_vector_2d(
        field_bc.north, backend, T, Nx, Symbol(name, :_north);
        active=bc.north == FVFD_BC_OPEN, default,
    )
    return FVFDFieldBC2D(west, east, south, north)
end

function fvfd_transfer_field_bc_2d(
    field_bc::FVFDFieldBC2D, backend,
    Nx::Integer, Ny::Integer, bc::FVFDDomainBC2D;
    FT::Type{<:AbstractFloat}=Float64, kwargs...,
)
    return fvfd_transfer_field_bc_2d(field_bc, backend, FT, Nx, Ny, bc; kwargs...)
end

function fvfd_transfer_embedded_boundary_2d(
    embedded::FVFDEmbeddedBoundary2D, backend, ::Type{T}=eltype(embedded.wall_nx),
) where {T<:AbstractFloat}
    Nx, Ny = size(embedded.wall_inv_distance)
    wall_nx = KernelAbstractions.allocate(backend, T, Nx, Ny)
    wall_ny = KernelAbstractions.allocate(backend, T, Nx, Ny)
    wall_inv_distance = KernelAbstractions.allocate(backend, T, Nx, Ny)
    wall_distance = KernelAbstractions.allocate(backend, T, Nx, Ny)
    wall_inv_distance_to_center = KernelAbstractions.allocate(backend, T, Nx, Ny)
    cell_fraction = KernelAbstractions.allocate(backend, T, Nx, Ny)
    wall_fraction = KernelAbstractions.allocate(backend, T, Nx, Ny)
    west_fraction = KernelAbstractions.allocate(backend, T, Nx, Ny)
    east_fraction = KernelAbstractions.allocate(backend, T, Nx, Ny)
    south_fraction = KernelAbstractions.allocate(backend, T, Nx, Ny)
    north_fraction = KernelAbstractions.allocate(backend, T, Nx, Ny)
    cut_count = KernelAbstractions.allocate(backend, UInt8, Nx, Ny)
    copyto!(wall_nx, T.(embedded.wall_nx))
    copyto!(wall_ny, T.(embedded.wall_ny))
    copyto!(wall_inv_distance, T.(embedded.wall_inv_distance))
    copyto!(wall_distance, T.(embedded.wall_distance))
    copyto!(wall_inv_distance_to_center, T.(embedded.wall_inv_distance_to_center))
    copyto!(cell_fraction, T.(embedded.cell_fraction))
    copyto!(wall_fraction, T.(embedded.wall_fraction))
    copyto!(west_fraction, T.(embedded.west_fraction))
    copyto!(east_fraction, T.(embedded.east_fraction))
    copyto!(south_fraction, T.(embedded.south_fraction))
    copyto!(north_fraction, T.(embedded.north_fraction))
    copyto!(cut_count, UInt8.(embedded.cut_count))
    return FVFDEmbeddedBoundary2D(
        wall_nx, wall_ny, wall_inv_distance, wall_distance,
        wall_inv_distance_to_center, cell_fraction, wall_fraction,
        west_fraction, east_fraction, south_fraction, north_fraction, cut_count,
    )
end

function fvfd_geometry_from_lbm_2d(
    is_solid, q_wall, dx::Real, dy::Real, bc::FVFDDomainBC2D;
    FT::Type{<:AbstractFloat}=eltype(q_wall),
    level::Integer=0,
    include_axis_aligned::Bool=false,
    include_halfway::Bool=false,
)
    embedded = fvfd_embedded_boundary_from_qwall_2d(
        q_wall; FT, include_axis_aligned, include_halfway,
    )
    patch = FVFDPatch2D(FT(dx), FT(dy); level)
    return FVFDGeometry2D(is_solid, embedded, patch, bc)
end

function fvfd_transfer_geometry_2d(
    geometry::FVFDGeometry2D, backend, ::Type{T}=eltype(geometry.embedded.wall_nx),
) where {T<:AbstractFloat}
    Nx, Ny = size(geometry.is_solid)
    is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny)
    copyto!(is_solid, Matrix{Bool}(geometry.is_solid))
    embedded = fvfd_transfer_embedded_boundary_2d(geometry.embedded, backend, T)
    patch = FVFDPatch2D(T(geometry.patch.dx), T(geometry.patch.dy); level=geometry.patch.level)
    return FVFDGeometry2D(is_solid, embedded, patch, geometry.bc)
end
