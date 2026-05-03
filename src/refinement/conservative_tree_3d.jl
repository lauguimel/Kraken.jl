# Conservative cell-centered tree refinement primitives for D3Q19.
#
# These helpers manipulate integrated populations F_i = f_i * cell_volume.
# They are deliberately independent from the existing patch-based 3D
# refinement path.

const D3Q19_CX_INT = (0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0)
const D3Q19_CY_INT = (0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1)
const D3Q19_CZ_INT = (0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1)
const D3Q19_OPPOSITE_INT = (1, 3, 2, 5, 4, 7, 6, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16)

@inline function _check_d3q19_q(q::Int)
    1 <= q <= 19 || throw(ArgumentError("D3Q19 population index q must be in 1:19"))
    return q
end

@inline function d3q19_cx(q::Integer)
    qi = _check_d3q19_q(Int(q))
    return D3Q19_CX_INT[qi]
end

@inline function d3q19_cy(q::Integer)
    qi = _check_d3q19_q(Int(q))
    return D3Q19_CY_INT[qi]
end

@inline function d3q19_cz(q::Integer)
    qi = _check_d3q19_q(Int(q))
    return D3Q19_CZ_INT[qi]
end

@inline function d3q19_opposite(q::Integer)
    qi = _check_d3q19_q(Int(q))
    return D3Q19_OPPOSITE_INT[qi]
end

@inline function _check_d3q19_vector(F::AbstractVector, name::AbstractString)
    length(F) == 19 || throw(ArgumentError("$name must have length 19 for D3Q19"))
    return nothing
end

@inline function _check_child_block_3d(Fc::AbstractArray{<:Any,4}, name::AbstractString)
    size(Fc) == (2, 2, 2, 19) ||
        throw(ArgumentError("$name must have size (2, 2, 2, 19)"))
    return nothing
end

"""
    coalesce_F_3d!(Fp, Fc)

Coalesce the eight ratio-2 child cells `Fc[ix, iy, iz, q]` into one parent
population vector `Fp[q]`.
"""
function coalesce_F_3d!(Fp::AbstractVector, Fc::AbstractArray{<:Any,4})
    _check_d3q19_vector(Fp, "Fp")
    _check_child_block_3d(Fc, "Fc")

    @inbounds for q in 1:19
        Fp[q] = Fc[1, 1, 1, q] + Fc[2, 1, 1, q] +
                Fc[1, 2, 1, q] + Fc[2, 2, 1, q] +
                Fc[1, 1, 2, q] + Fc[2, 1, 2, q] +
                Fc[1, 2, 2, q] + Fc[2, 2, 2, q]
    end
    return Fp
end

"""
    explode_uniform_F_3d!(Fc, Fp)

Uniformly split a parent integrated D3Q19 population vector over its eight
ratio-2 children.
"""
function explode_uniform_F_3d!(Fc::AbstractArray{<:Any,4}, Fp::AbstractVector)
    _check_child_block_3d(Fc, "Fc")
    _check_d3q19_vector(Fp, "Fp")

    @inbounds for q in 1:19
        fq = Fp[q] / 8
        for iz in 1:2, iy in 1:2, ix in 1:2
            Fc[ix, iy, iz, q] = fq
        end
    end
    return Fc
end

function mass_F_3d(F::AbstractVector)
    _check_d3q19_vector(F, "F")
    return sum(F)
end

function momentum_F_3d(F::AbstractVector)
    _check_d3q19_vector(F, "F")

    mx = zero(F[1])
    my = zero(F[1])
    mz = zero(F[1])
    @inbounds for q in 1:19
        fq = F[q]
        mx += d3q19_cx(q) * fq
        my += d3q19_cy(q) * fq
        mz += d3q19_cz(q) * fq
    end
    return mx, my, mz
end

function moments_F_3d(F::AbstractVector)
    m = mass_F_3d(F)
    mx, my, mz = momentum_F_3d(F)
    return m, mx, my, mz
end

function mass_F_3d(F::AbstractArray{<:Any,4})
    size(F, 4) == 19 ||
        throw(ArgumentError("F must have 19 D3Q19 populations in dimension 4"))
    return sum(F)
end

function momentum_F_3d(F::AbstractArray{<:Any,4})
    size(F, 4) == 19 ||
        throw(ArgumentError("F must have 19 D3Q19 populations in dimension 4"))

    mx = zero(F[begin, begin, begin, 1])
    my = zero(F[begin, begin, begin, 1])
    mz = zero(F[begin, begin, begin, 1])
    @inbounds for q in 1:19
        cx = d3q19_cx(q)
        cy = d3q19_cy(q)
        cz = d3q19_cz(q)
        for iz in axes(F, 3), iy in axes(F, 2), ix in axes(F, 1)
            fq = F[ix, iy, iz, q]
            mx += cx * fq
            my += cy * fq
            mz += cz * fq
        end
    end
    return mx, my, mz
end

function moments_F_3d(F::AbstractArray{<:Any,4})
    m = mass_F_3d(F)
    mx, my, mz = momentum_F_3d(F)
    return m, mx, my, mz
end

function fill_equilibrium_integrated_D3Q19!(Fcell::AbstractVector,
                                            volume,
                                            rho,
                                            ux,
                                            uy,
                                            uz)
    _check_d3q19_vector(Fcell, "Fcell")
    @inbounds for q in 1:19
        Fcell[q] = volume * equilibrium(D3Q19(), rho, ux, uy, uz, q)
    end
    return Fcell
end

function fill_equilibrium_integrated_D3Q19!(F::AbstractArray{<:Any,4},
                                            volume,
                                            rho,
                                            ux,
                                            uy,
                                            uz)
    size(F, 4) == 19 ||
        throw(ArgumentError("F must have 19 D3Q19 populations in dimension 4"))
    @inbounds for iz in axes(F, 3), iy in axes(F, 2), ix in axes(F, 1)
        rho_ijk = rho isa Function ? rho(ix, iy, iz) : rho
        ux_ijk = ux isa Function ? ux(ix, iy, iz) : ux
        uy_ijk = uy isa Function ? uy(ix, iy, iz) : uy
        uz_ijk = uz isa Function ? uz(ix, iy, iz) : uz
        fill_equilibrium_integrated_D3Q19!(
            @view(F[ix, iy, iz, :]), volume, rho_ijk, ux_ijk, uy_ijk, uz_ijk)
    end
    return F
end

@inline function conservative_tree_parent_index_3d(i_f::Int, j_f::Int, k_f::Int)
    i_f >= 1 || throw(ArgumentError("i_f must be >= 1"))
    j_f >= 1 || throw(ArgumentError("j_f must be >= 1"))
    k_f >= 1 || throw(ArgumentError("k_f must be >= 1"))

    i_parent = (i_f + 1) >>> 1
    j_parent = (j_f + 1) >>> 1
    k_parent = (k_f + 1) >>> 1
    i_child = isodd(i_f) ? 1 : 2
    j_child = isodd(j_f) ? 1 : 2
    k_child = isodd(k_f) ? 1 : 2
    return i_parent, j_parent, k_parent, i_child, j_child, k_child
end

@inline function _check_conservative_tree_face_3d(face::Symbol)
    face in (:west, :east, :south, :north, :bottom, :top) ||
        throw(ArgumentError("face must be one of :west, :east, :south, :north, :bottom, :top"))
    return face
end

@inline function _face_normal_3d(face::Symbol)
    _check_conservative_tree_face_3d(face)
    if face == :west
        return -1, 0, 0
    elseif face == :east
        return 1, 0, 0
    elseif face == :south
        return 0, -1, 0
    elseif face == :north
        return 0, 1, 0
    elseif face == :bottom
        return 0, 0, -1
    else
        return 0, 0, 1
    end
end

@inline function _enters_patch_through_face_3d(q::Int, face::Symbol)
    nx, ny, nz = _face_normal_3d(face)
    return d3q19_cx(q) * nx + d3q19_cy(q) * ny + d3q19_cz(q) * nz < 0
end

@inline function _leaves_patch_through_face_3d(q::Int, face::Symbol)
    nx, ny, nz = _face_normal_3d(face)
    return d3q19_cx(q) * nx + d3q19_cy(q) * ny + d3q19_cz(q) * nz > 0
end

@inline function _face_child_indices_3d(face::Symbol)
    _check_conservative_tree_face_3d(face)
    if face == :west
        return ((1, 1, 1), (1, 2, 1), (1, 1, 2), (1, 2, 2))
    elseif face == :east
        return ((2, 1, 1), (2, 2, 1), (2, 1, 2), (2, 2, 2))
    elseif face == :south
        return ((1, 1, 1), (2, 1, 1), (1, 1, 2), (2, 1, 2))
    elseif face == :north
        return ((1, 2, 1), (2, 2, 1), (1, 2, 2), (2, 2, 2))
    elseif face == :bottom
        return ((1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1))
    else
        return ((1, 1, 2), (2, 1, 2), (1, 2, 2), (2, 2, 2))
    end
end

"""
    split_coarse_to_fine_face_F_3d!(Fc_dest, Fq, q, face)

Accumulate an integrated packet entering a refined parent through one face.
The packet is split uniformly over the four child cells adjacent to that face.
"""
function split_coarse_to_fine_face_F_3d!(Fc_dest::AbstractArray{<:Any,4},
                                         Fq,
                                         q::Int,
                                         face::Symbol)
    _check_child_block_3d(Fc_dest, "Fc_dest")
    qi = _check_d3q19_q(q)
    _check_conservative_tree_face_3d(face)
    _enters_patch_through_face_3d(qi, face) ||
        throw(ArgumentError("population q=$qi does not enter through face $face"))

    share = Fq / 4
    @inbounds for (ix, iy, iz) in _face_child_indices_3d(face)
        Fc_dest[ix, iy, iz, qi] += share
    end
    return Fc_dest
end

"""
    coalesce_fine_to_coarse_face_F_3d(Fc_src, q, face)

Return the integrated packet leaving a refined parent through one face by
summing the four child cells adjacent to that exit face.
"""
function coalesce_fine_to_coarse_face_F_3d(Fc_src::AbstractArray{<:Any,4},
                                          q::Int,
                                          face::Symbol)
    _check_child_block_3d(Fc_src, "Fc_src")
    qi = _check_d3q19_q(q)
    _check_conservative_tree_face_3d(face)
    _leaves_patch_through_face_3d(qi, face) ||
        throw(ArgumentError("population q=$qi does not leave through face $face"))

    packet = zero(Fc_src[1, 1, 1, qi])
    @inbounds for (ix, iy, iz) in _face_child_indices_3d(face)
        packet += Fc_src[ix, iy, iz, qi]
    end
    return packet
end

@inline function _check_conservative_tree_edge_3d(edge::Symbol)
    edge in (:southwest, :southeast, :northwest, :northeast,
             :bottomwest, :bottomeast, :topwest, :topeast,
             :bottomsouth, :bottomnorth, :topsouth, :topnorth) ||
        throw(ArgumentError("edge must name one of the 12 3D parent edges"))
    return edge
end

@inline function _edge_faces_3d(edge::Symbol)
    _check_conservative_tree_edge_3d(edge)
    if edge == :southwest
        return :south, :west
    elseif edge == :southeast
        return :south, :east
    elseif edge == :northwest
        return :north, :west
    elseif edge == :northeast
        return :north, :east
    elseif edge == :bottomwest
        return :bottom, :west
    elseif edge == :bottomeast
        return :bottom, :east
    elseif edge == :topwest
        return :top, :west
    elseif edge == :topeast
        return :top, :east
    elseif edge == :bottomsouth
        return :bottom, :south
    elseif edge == :bottomnorth
        return :bottom, :north
    elseif edge == :topsouth
        return :top, :south
    else
        return :top, :north
    end
end

@inline function _enters_patch_through_edge_3d(q::Int, edge::Symbol)
    f1, f2 = _edge_faces_3d(edge)
    return _enters_patch_through_face_3d(q, f1) &&
           _enters_patch_through_face_3d(q, f2)
end

@inline function _leaves_patch_through_edge_3d(q::Int, edge::Symbol)
    f1, f2 = _edge_faces_3d(edge)
    return _leaves_patch_through_face_3d(q, f1) &&
           _leaves_patch_through_face_3d(q, f2)
end

@inline function _edge_child_indices_3d(edge::Symbol)
    _check_conservative_tree_edge_3d(edge)
    if edge == :southwest
        return ((1, 1, 1), (1, 1, 2))
    elseif edge == :southeast
        return ((2, 1, 1), (2, 1, 2))
    elseif edge == :northwest
        return ((1, 2, 1), (1, 2, 2))
    elseif edge == :northeast
        return ((2, 2, 1), (2, 2, 2))
    elseif edge == :bottomwest
        return ((1, 1, 1), (1, 2, 1))
    elseif edge == :bottomeast
        return ((2, 1, 1), (2, 2, 1))
    elseif edge == :topwest
        return ((1, 1, 2), (1, 2, 2))
    elseif edge == :topeast
        return ((2, 1, 2), (2, 2, 2))
    elseif edge == :bottomsouth
        return ((1, 1, 1), (2, 1, 1))
    elseif edge == :bottomnorth
        return ((1, 2, 1), (2, 2, 1))
    elseif edge == :topsouth
        return ((1, 1, 2), (2, 1, 2))
    else
        return ((1, 2, 2), (2, 2, 2))
    end
end

"""
    split_coarse_to_fine_edge_F_3d!(Fc_dest, Fq, q, edge)

Accumulate one D3Q19 edge-aligned packet entering a refined parent through a
parent edge. The packet is split uniformly over the two child cells adjacent
to that edge.
"""
function split_coarse_to_fine_edge_F_3d!(Fc_dest::AbstractArray{<:Any,4},
                                         Fq,
                                         q::Int,
                                         edge::Symbol)
    _check_child_block_3d(Fc_dest, "Fc_dest")
    qi = _check_d3q19_q(q)
    _check_conservative_tree_edge_3d(edge)
    _enters_patch_through_edge_3d(qi, edge) ||
        throw(ArgumentError("population q=$qi does not enter through edge $edge"))

    share = Fq / 2
    @inbounds for (ix, iy, iz) in _edge_child_indices_3d(edge)
        Fc_dest[ix, iy, iz, qi] += share
    end
    return Fc_dest
end

"""
    coalesce_fine_to_coarse_edge_F_3d(Fc_src, q, edge)

Return the integrated packet leaving a refined parent through one parent edge
by summing the two child cells adjacent to that edge.
"""
function coalesce_fine_to_coarse_edge_F_3d(Fc_src::AbstractArray{<:Any,4},
                                          q::Int,
                                          edge::Symbol)
    _check_child_block_3d(Fc_src, "Fc_src")
    qi = _check_d3q19_q(q)
    _check_conservative_tree_edge_3d(edge)
    _leaves_patch_through_edge_3d(qi, edge) ||
        throw(ArgumentError("population q=$qi does not leave through edge $edge"))

    packet = zero(Fc_src[1, 1, 1, qi])
    @inbounds for (ix, iy, iz) in _edge_child_indices_3d(edge)
        packet += Fc_src[ix, iy, iz, qi]
    end
    return packet
end

@inline function _check_conservative_tree_corner_3d(corner::Symbol)
    corner in (:bottomsouthwest, :bottomsoutheast, :bottomnorthwest, :bottomnortheast,
               :topsouthwest, :topsoutheast, :topnorthwest, :topnortheast) ||
        throw(ArgumentError("corner must name one of the 8 3D parent corners"))
    return corner
end

"""
    split_coarse_to_fine_corner_F_3d!(Fc_dest, Fq, q, corner)

D3Q19 has no body-diagonal populations, so the conservative corner route set
is empty. This function exists as an explicit canary for that topology fact.
"""
function split_coarse_to_fine_corner_F_3d!(Fc_dest::AbstractArray{<:Any,4},
                                           Fq,
                                           q::Int,
                                           corner::Symbol)
    _check_child_block_3d(Fc_dest, "Fc_dest")
    qi = _check_d3q19_q(q)
    _check_conservative_tree_corner_3d(corner)
    throw(ArgumentError("population q=$qi cannot enter through corner $corner with D3Q19"))
end

"""
    coalesce_fine_to_coarse_corner_F_3d(Fc_src, q, corner)

D3Q19 has no body-diagonal populations, so the conservative corner route set
is empty. This function exists as an explicit canary for that topology fact.
"""
function coalesce_fine_to_coarse_corner_F_3d(Fc_src::AbstractArray{<:Any,4},
                                            q::Int,
                                            corner::Symbol)
    _check_child_block_3d(Fc_src, "Fc_src")
    qi = _check_d3q19_q(q)
    _check_conservative_tree_corner_3d(corner)
    throw(ArgumentError("population q=$qi cannot leave through corner $corner with D3Q19"))
end
