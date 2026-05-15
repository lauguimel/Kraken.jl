@enum KrakenECFFaceAxis KRAKEN_E_CF_FACE_X KRAKEN_E_CF_FACE_Y
@enum KrakenECFFaceSide KRAKEN_E_CF_FACE_LO KRAKEN_E_CF_FACE_HI

struct CFFaceRecord2D{T<:AbstractFloat}
    coarse_block_id::Int
    fine_block_id::Int
    coarse_index::NTuple{2,Int}
    fine_indices::NTuple{2,NTuple{2,Int}}
    axis::KrakenECFFaceAxis
    side::KrakenECFFaceSide
    coarse_area::T
    fine_areas::NTuple{2,T}
    coarse_center::SVector{2,T}
    fine_centers::NTuple{2,SVector{2,T}}
    normal::SVector{2,T}
    fine_to_coarse_weights::NTuple{2,T}
    tangential_offsets_1::NTuple{3,NTuple{2,Int}}
    tangential_weights_1::NTuple{3,T}
    tangential_offsets_2::NTuple{3,NTuple{2,Int}}
    tangential_weights_2::NTuple{3,T}
    corner_lo_owned::Bool
    corner_hi_owned::Bool
end

function kraken_e_build_cf_face_record_2d(
    ::Type{T};
    coarse_block_id::Int, fine_block_id::Int,
    coarse_index::NTuple{2,Int}, fine_indices::NTuple{2,NTuple{2,Int}},
    axis::KrakenECFFaceAxis, side::KrakenECFFaceSide,
    coarse_origin::NTuple{2,T}, coarse_dx::T,
) where {T<:AbstractFloat}
    I, J = coarse_index
    half = T(1) / T(2)
    quarter = T(1) / T(4)
    three_quarter = T(3) / T(4)
    side_offset = side == KRAKEN_E_CF_FACE_LO ? zero(T) : one(T)

    if axis == KRAKEN_E_CF_FACE_X
        x_face = coarse_origin[1] + (T(I - 1) + side_offset) * coarse_dx
        y_lo = coarse_origin[2] + T(J - 1) * coarse_dx
        coarse_center = SVector{2,T}(x_face, y_lo + half * coarse_dx)
        fine_centers = (
            SVector{2,T}(x_face, y_lo + quarter * coarse_dx),
            SVector{2,T}(x_face, y_lo + three_quarter * coarse_dx),
        )
        normal = side == KRAKEN_E_CF_FACE_LO ?
            SVector{2,T}(-one(T), zero(T)) :
            SVector{2,T}(one(T), zero(T))
        tangential_offsets = ((0, 0), (0, -1), (0, 1))
    elseif axis == KRAKEN_E_CF_FACE_Y
        x_lo = coarse_origin[1] + T(I - 1) * coarse_dx
        y_face = coarse_origin[2] + (T(J - 1) + side_offset) * coarse_dx
        coarse_center = SVector{2,T}(x_lo + half * coarse_dx, y_face)
        fine_centers = (
            SVector{2,T}(x_lo + quarter * coarse_dx, y_face),
            SVector{2,T}(x_lo + three_quarter * coarse_dx, y_face),
        )
        normal = side == KRAKEN_E_CF_FACE_LO ?
            SVector{2,T}(zero(T), -one(T)) :
            SVector{2,T}(zero(T), one(T))
        tangential_offsets = ((0, 0), (-1, 0), (1, 0))
    else
        throw(ArgumentError("unsupported Kraken-E coarse/fine face axis"))
    end

    return CFFaceRecord2D{T}(
        coarse_block_id,
        fine_block_id,
        coarse_index,
        fine_indices,
        axis,
        side,
        coarse_dx,
        (half * coarse_dx, half * coarse_dx),
        coarse_center,
        fine_centers,
        normal,
        (half, half),
        tangential_offsets,
        (three_quarter, quarter, zero(T)),
        tangential_offsets,
        (three_quarter, zero(T), quarter),
        side == KRAKEN_E_CF_FACE_LO,
        side == KRAKEN_E_CF_FACE_HI,
    )
end
