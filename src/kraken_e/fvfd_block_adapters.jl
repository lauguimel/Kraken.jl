@inline function kraken_e_interior_view_2d(
    A2::AbstractArray{T,2}, block::LeafBlock2D,
) where {T}
    ng = block.ng
    return @view A2[(ng + 1):(ng + block.Nx), (ng + 1):(ng + block.Ny)]
end

function fvfd_velocity_gradient_block_2d!(
    dudx, dudy, dvdx, dvdy,
    block::LeafBlock2D, bc::FVFDDomainBC2D;
    is_solid::Union{Nothing,AbstractArray}=nothing,
    sync::Bool=true,
)
    solid = is_solid === nothing ? falses(block.Nx, block.Ny) : is_solid
    ux = kraken_e_interior_view_2d(block.ux, block)
    uy = kraken_e_interior_view_2d(block.uy, block)
    return fvfd_velocity_gradient_2d!(
        dudx, dudy, dvdx, dvdy,
        ux, uy, solid, block.dx, block.dx, bc; sync,
    )
end

function fvfd_tensor_divergence_block_2d!(
    fx, fy, tauxx, tauxy, tauyy,
    block::LeafBlock2D, bc::FVFDDomainBC2D;
    is_solid::Union{Nothing,AbstractArray}=nothing,
    sync::Bool=true,
)
    solid = is_solid === nothing ? falses(block.Nx, block.Ny) : is_solid
    return fvfd_tensor_divergence_2d!(
        fx, fy, tauxx, tauxy, tauyy, solid, block.dx, block.dx, bc; sync,
    )
end

function fvfd_cell_velocity_to_faces_block_2d!(
    ux_face, uy_face,
    block::LeafBlock2D, bc::FVFDDomainBC2D;
    is_solid::Union{Nothing,AbstractArray}=nothing,
    ux_west=nothing, ux_east=nothing,
    uy_south=nothing, uy_north=nothing,
    sync::Bool=true,
)
    T = eltype(block.f)
    solid = is_solid === nothing ? falses(block.Nx, block.Ny) : is_solid
    ux = kraken_e_interior_view_2d(block.ux, block)
    uy = kraken_e_interior_view_2d(block.uy, block)
    return fvfd_cell_velocity_to_faces_2d!(
        ux_face, uy_face, ux, uy, solid,
        ux_west === nothing ? T[] : ux_west,
        ux_east === nothing ? T[] : ux_east,
        uy_south === nothing ? T[] : uy_south,
        uy_north === nothing ? T[] : uy_north,
        bc; sync,
    )
end
