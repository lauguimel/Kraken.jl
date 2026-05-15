const KRAKEN_E_INTERIOR = Int8(0)
const KRAKEN_E_GHOST_HALO = Int8(1)
const KRAKEN_E_GHOST_CF = Int8(2)
const KRAKEN_E_WALL = Int8(3)

# Typed coarse/fine face records are defined in cf_face_2d.jl.

mutable struct LeafBlock2D{T<:AbstractFloat,AT2<:AbstractArray{T,2},AT3<:AbstractArray{T,3}}
    id::Int
    level::Int
    origin::SVector{2,T}
    dx::T
    Nx::Int
    Ny::Int
    ng::Int
    f::AT3
    f_tmp::AT3
    ρ::AT2
    ux::AT2
    uy::AT2
    cell_kind::Matrix{Int8}
    parent_id::Int
    child_ids::Vector{Int}
    same_level_neighbor_ids::NTuple{4,Int}
    cf_face_records::Vector{CFFaceRecord2D{T}}
    reflux_accumulators::Vector{T}
    epoch_remap_buffers::Vector{T}
end

"""
    allocate_leaf_block_2d([T=Float64]; Nx, Ny, id=1, level=0, origin=(0, 0), dx=1)

Allocate a single Kraken-E D2Q9 leaf block with one ghost layer and the S2
paper-only AMR fields initialized to sentinel values.
"""
function allocate_leaf_block_2d(::Type{T}=Float64; Nx::Integer, Ny::Integer,
                                id::Integer=1, level::Integer=0,
                                origin=(zero(T), zero(T)), dx=one(T),
                                ng::Integer=1) where {T<:AbstractFloat}
    ng == 1 || throw(ArgumentError("Kraken-E S2 requires ng = 1"))
    Nx > 0 || throw(ArgumentError("Nx must be positive"))
    Ny > 0 || throw(ArgumentError("Ny must be positive"))

    ni = Int(Nx) + 2
    nj = Int(Ny) + 2
    f = zeros(T, ni, nj, 9)
    f_tmp = zeros(T, ni, nj, 9)
    ρ = zeros(T, ni, nj)
    ux = zeros(T, ni, nj)
    uy = zeros(T, ni, nj)
    cell_kind = fill(KRAKEN_E_GHOST_HALO, ni, nj)
    for j in 2:(Int(Ny) + 1), i in 2:(Int(Nx) + 1)
        cell_kind[i, j] = KRAKEN_E_INTERIOR
    end

    origin_vec = SVector{2,T}(T(origin[1]), T(origin[2]))
    return LeafBlock2D{T,typeof(ρ),typeof(f)}(
        Int(id), Int(level), origin_vec, T(dx), Int(Nx), Int(Ny), Int(ng),
        f, f_tmp, ρ, ux, uy, cell_kind,
        -1, Int[], (-1, -1, -1, -1), CFFaceRecord2D{T}[], T[], T[],
    )
end

KrakenELeafBlock2D(args...; kwargs...) = allocate_leaf_block_2d(args...; kwargs...)

@inline kraken_e_i_range(block::LeafBlock2D) = 2:(block.Nx + 1)
@inline kraken_e_j_range(block::LeafBlock2D) = 2:(block.Ny + 1)

@inline function kraken_e_density_at(f, i::Int, j::Int)
    return f[i,j,1] + f[i,j,2] + f[i,j,3] + f[i,j,4] + f[i,j,5] +
           f[i,j,6] + f[i,j,7] + f[i,j,8] + f[i,j,9]
end

@inline kraken_e_west_i(block::LeafBlock2D, i::Int) = i == 2 ? block.Nx + 1 : i - 1
@inline kraken_e_east_i(block::LeafBlock2D, i::Int) = i == block.Nx + 1 ? 2 : i + 1

const CFFaceRecord = CFFaceRecord2D{Float64}
