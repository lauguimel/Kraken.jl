# Conservative cell-centered tree refinement prototype for D2Q9.
#
# This file intentionally stays separate from the existing RefinementPatch
# implementation. The state manipulated here is the integrated population
# F_i = f_i * cell_volume, not the density population f_i.

const D2Q9_CX_INT = (0, 1, 0, -1, 0, 1, -1, -1, 1)
const D2Q9_CY_INT = (0, 0, 1, 0, -1, 1, 1, -1, -1)
const D2Q9_OPPOSITE_INT = (1, 4, 5, 2, 3, 8, 9, 6, 7)

@inline function _check_d2q9_q(q::Int)
    1 <= q <= 9 || throw(ArgumentError("D2Q9 population index q must be in 1:9"))
    return q
end

@inline function d2q9_cx(q::Integer)
    qi = _check_d2q9_q(Int(q))
    return D2Q9_CX_INT[qi]
end

@inline function d2q9_cy(q::Integer)
    qi = _check_d2q9_q(Int(q))
    return D2Q9_CY_INT[qi]
end

@inline function d2q9_opposite(q::Integer)
    qi = _check_d2q9_q(Int(q))
    return D2Q9_OPPOSITE_INT[qi]
end

@inline function _check_d2q9_vector(F::AbstractVector, name::AbstractString)
    length(F) == 9 || throw(ArgumentError("$name must have length 9 for D2Q9"))
    return nothing
end

@inline function _check_child_block_2d(Fc::AbstractArray{<:Any,3}, name::AbstractString)
    size(Fc) == (2, 2, 9) ||
        throw(ArgumentError("$name must have size (2, 2, 9)"))
    return nothing
end

"""
    coalesce_F_2d!(Fp, Fc)

Coalesce the four ratio-2 child cells `Fc[ix, iy, q]` into one parent
population vector `Fp[q]`.

Both `Fp` and `Fc` store integrated populations `F_i = f_i * volume`.
The operation preserves each oriented population exactly up to floating-point
roundoff.
"""
function coalesce_F_2d!(Fp::AbstractVector, Fc::AbstractArray{<:Any,3})
    _check_d2q9_vector(Fp, "Fp")
    _check_child_block_2d(Fc, "Fc")

    @inbounds for q in 1:9
        Fp[q] = Fc[1, 1, q] + Fc[2, 1, q] + Fc[1, 2, q] + Fc[2, 2, q]
    end
    return Fp
end

"""
    explode_uniform_F_2d!(Fc, Fp)

Uniformly split a parent integrated population vector over its four ratio-2
children. This is a conservative low-order explosion operator.
"""
function explode_uniform_F_2d!(Fc::AbstractArray{<:Any,3}, Fp::AbstractVector)
    _check_child_block_2d(Fc, "Fc")
    _check_d2q9_vector(Fp, "Fp")

    @inbounds for q in 1:9
        fq = Fp[q] / 4
        Fc[1, 1, q] = fq
        Fc[2, 1, q] = fq
        Fc[1, 2, q] = fq
        Fc[2, 2, q] = fq
    end
    return Fc
end

"""
    mass_F(F)

Mass carried by a D2Q9 integrated population vector.
"""
function mass_F(F::AbstractVector)
    _check_d2q9_vector(F, "F")
    return sum(F)
end

"""
    momentum_F(F)

Momentum carried by a D2Q9 integrated population vector.
"""
function momentum_F(F::AbstractVector)
    _check_d2q9_vector(F, "F")

    mx = zero(F[1])
    my = zero(F[1])
    @inbounds for q in 1:9
        mx += d2q9_cx(q) * F[q]
        my += d2q9_cy(q) * F[q]
    end
    return mx, my
end

"""
    moments_F(F)

Return `(mass, momentum_x, momentum_y)` for a D2Q9 integrated population vector.
"""
function moments_F(F::AbstractVector)
    m = mass_F(F)
    mx, my = momentum_F(F)
    return m, mx, my
end

function mass_F(F::AbstractArray{<:Any,3})
    size(F, 3) == 9 || throw(ArgumentError("F must have 9 D2Q9 populations in dimension 3"))
    return sum(F)
end

function momentum_F(F::AbstractArray{<:Any,3})
    size(F, 3) == 9 || throw(ArgumentError("F must have 9 D2Q9 populations in dimension 3"))

    mx = zero(F[begin, begin, 1])
    my = zero(F[begin, begin, 1])
    @inbounds for q in 1:9
        cx = d2q9_cx(q)
        cy = d2q9_cy(q)
        for j in axes(F, 2), i in axes(F, 1)
            fq = F[i, j, q]
            mx += cx * fq
            my += cy * fq
        end
    end
    return mx, my
end

function moments_F(F::AbstractArray{<:Any,3})
    m = mass_F(F)
    mx, my = momentum_F(F)
    return m, mx, my
end

function fill_equilibrium_integrated_D2Q9!(Fcell::AbstractVector,
                                           volume,
                                           rho,
                                           ux,
                                           uy)
    _check_d2q9_vector(Fcell, "Fcell")
    @inbounds for q in 1:9
        Fcell[q] = volume * equilibrium(D2Q9(), rho, ux, uy, q)
    end
    return Fcell
end

function fill_equilibrium_integrated_D2Q9!(F::AbstractArray{<:Any,3},
                                           volume,
                                           rho,
                                           ux,
                                           uy)
    size(F, 3) == 9 || throw(ArgumentError("F must have 9 D2Q9 populations in dimension 3"))
    @inbounds for j in axes(F, 2), i in axes(F, 1)
        rho_ij = rho isa Function ? rho(i, j) : rho
        ux_ij = ux isa Function ? ux(i, j) : ux
        uy_ij = uy isa Function ? uy(i, j) : uy
        fill_equilibrium_integrated_D2Q9!(@view(F[i, j, :]), volume, rho_ij, ux_ij, uy_ij)
    end
    return F
end

@inline function conservative_tree_parent_index(i_f::Int, j_f::Int)
    i_f >= 1 || throw(ArgumentError("i_f must be >= 1"))
    j_f >= 1 || throw(ArgumentError("j_f must be >= 1"))

    i_parent = (i_f + 1) >>> 1
    j_parent = (j_f + 1) >>> 1
    i_child = isodd(i_f) ? 1 : 2
    j_child = isodd(j_f) ? 1 : 2
    return i_parent, j_parent, i_child, j_child
end

"""
    split_coarse_to_fine_vertical_F_2d!(Fc_dest, Fq, q)

Accumulate one integrated packet crossing a vertical coarse/fine interface into
the two fine children adjacent to the entry face.

For east-moving populations the entry face is the west side of the refined
parent, so children `(1, 1)` and `(1, 2)` receive `Fq / 2`. For west-moving
populations the entry face is the east side, so children `(2, 1)` and `(2, 2)`
receive `Fq / 2`.
"""
function split_coarse_to_fine_vertical_F_2d!(Fc_dest::AbstractArray{<:Any,3},
                                             Fq,
                                             q::Int)
    face = d2q9_cx(q) > 0 ? :west : :east
    return split_coarse_to_fine_face_F_2d!(Fc_dest, Fq, q, face)
end

"""
    coalesce_fine_to_coarse_vertical_F(Fc_src, q)

Return the integrated packet leaving a refined parent through a vertical
interface. The function sums the two interface-adjacent fine children for the
oriented population `q`.
"""
function coalesce_fine_to_coarse_vertical_F(Fc_src::AbstractArray{<:Any,3},
                                            q::Int)
    face = d2q9_cx(q) < 0 ? :west : :east
    return coalesce_fine_to_coarse_face_F(Fc_src, q, face)
end

@inline function _check_conservative_tree_face(face::Symbol)
    face in (:west, :east, :south, :north) ||
        throw(ArgumentError("face must be one of :west, :east, :south, :north"))
    return face
end

@inline function _face_normal(face::Symbol)
    _check_conservative_tree_face(face)
    if face == :west
        return -1, 0
    elseif face == :east
        return 1, 0
    elseif face == :south
        return 0, -1
    else
        return 0, 1
    end
end

@inline function _enters_patch_through_face(q::Int, face::Symbol)
    nx, ny = _face_normal(face)
    return d2q9_cx(q) * nx + d2q9_cy(q) * ny < 0
end

@inline function _leaves_patch_through_face(q::Int, face::Symbol)
    nx, ny = _face_normal(face)
    return d2q9_cx(q) * nx + d2q9_cy(q) * ny > 0
end

"""
    split_coarse_to_fine_face_F_2d!(Fc_dest, Fq, q, face)

Accumulate an integrated packet entering a refined parent through one face.

`face` names the boundary of the refined parent (`:west`, `:east`, `:south`,
`:north`). The operator is conservative and low order: it splits the packet
uniformly over the two child cells adjacent to the entry face. Diagonal corner
routing is intentionally left to the caller; this function is a face-interface
building block.
"""
function split_coarse_to_fine_face_F_2d!(Fc_dest::AbstractArray{<:Any,3},
                                         Fq,
                                         q::Int,
                                         face::Symbol)
    _check_child_block_2d(Fc_dest, "Fc_dest")
    qi = _check_d2q9_q(q)
    _check_conservative_tree_face(face)
    _enters_patch_through_face(qi, face) ||
        throw(ArgumentError("population q=$qi does not enter through face $face"))

    half = Fq / 2
    @inbounds begin
        if face == :west
            Fc_dest[1, 1, qi] += half
            Fc_dest[1, 2, qi] += half
        elseif face == :east
            Fc_dest[2, 1, qi] += half
            Fc_dest[2, 2, qi] += half
        elseif face == :south
            Fc_dest[1, 1, qi] += half
            Fc_dest[2, 1, qi] += half
        else
            Fc_dest[1, 2, qi] += half
            Fc_dest[2, 2, qi] += half
        end
    end
    return Fc_dest
end

"""
    coalesce_fine_to_coarse_face_F(Fc_src, q, face)

Return the integrated packet leaving a refined parent through one face by
summing the two child cells adjacent to that exit face.
"""
function coalesce_fine_to_coarse_face_F(Fc_src::AbstractArray{<:Any,3},
                                        q::Int,
                                        face::Symbol)
    _check_child_block_2d(Fc_src, "Fc_src")
    qi = _check_d2q9_q(q)
    _check_conservative_tree_face(face)
    _leaves_patch_through_face(qi, face) ||
        throw(ArgumentError("population q=$qi does not leave through face $face"))

    @inbounds begin
        if face == :west
            return Fc_src[1, 1, qi] + Fc_src[1, 2, qi]
        elseif face == :east
            return Fc_src[2, 1, qi] + Fc_src[2, 2, qi]
        elseif face == :south
            return Fc_src[1, 1, qi] + Fc_src[2, 1, qi]
        else
            return Fc_src[1, 2, qi] + Fc_src[2, 2, qi]
        end
    end
end

@inline function _check_conservative_tree_corner(corner::Symbol)
    corner in (:southwest, :southeast, :northwest, :northeast) ||
        throw(ArgumentError("corner must be one of :southwest, :southeast, :northwest, :northeast"))
    return corner
end

@inline function _entry_corner_for_q(q::Int)
    if q == 6
        return :southwest
    elseif q == 7
        return :southeast
    elseif q == 8
        return :northeast
    elseif q == 9
        return :northwest
    else
        throw(ArgumentError("only diagonal populations q=6:9 enter through corners"))
    end
end

@inline function _exit_corner_for_q(q::Int)
    if q == 6
        return :northeast
    elseif q == 7
        return :northwest
    elseif q == 8
        return :southwest
    elseif q == 9
        return :southeast
    else
        throw(ArgumentError("only diagonal populations q=6:9 leave through corners"))
    end
end

@inline function _corner_child_index(corner::Symbol)
    _check_conservative_tree_corner(corner)
    if corner == :southwest
        return 1, 1
    elseif corner == :southeast
        return 2, 1
    elseif corner == :northwest
        return 1, 2
    else
        return 2, 2
    end
end

@inline function _inside_range(i::Int, j::Int,
                              irange::UnitRange{Int},
                              jrange::UnitRange{Int})
    return first(irange) <= i <= last(irange) &&
           first(jrange) <= j <= last(jrange)
end

@inline function _inside_array_2d(A::AbstractArray, i::Int, j::Int)
    return first(axes(A, 1)) <= i <= last(axes(A, 1)) &&
           first(axes(A, 2)) <= j <= last(axes(A, 2))
end

@inline function _patch_local_parent_index(patch, i_parent::Int, j_parent::Int)
    return i_parent - first(patch.parent_i_range) + 1,
           j_parent - first(patch.parent_j_range) + 1
end

@inline function _child_block_view(Ffine, ip_local::Int, jp_local::Int)
    i0 = 2 * ip_local - 1
    j0 = 2 * jp_local - 1
    return @view Ffine[i0:i0+1, j0:j0+1, :]
end

@inline function _minmod(a, b)
    if a * b <= 0
        return zero(a + b)
    end
    return abs(a) < abs(b) ? a : b
end

@inline function _interface_face_from_offset(di::Int, dj::Int)
    if di < 0 && dj == 0
        return :west
    elseif di > 0 && dj == 0
        return :east
    elseif di == 0 && dj < 0
        return :south
    elseif di == 0 && dj > 0
        return :north
    else
        throw(ArgumentError("offset does not identify a single face"))
    end
end

@inline function _interface_corner_from_offset(di::Int, dj::Int)
    if di < 0 && dj < 0
        return :southwest
    elseif di > 0 && dj < 0
        return :southeast
    elseif di < 0 && dj > 0
        return :northwest
    elseif di > 0 && dj > 0
        return :northeast
    else
        throw(ArgumentError("offset does not identify a corner"))
    end
end

"""
    split_coarse_to_fine_corner_F_2d!(Fc_dest, Fq, q, corner)

Accumulate one diagonal integrated packet entering a refined parent through a
corner. The whole packet goes to the child cell adjacent to that corner.
"""
function split_coarse_to_fine_corner_F_2d!(Fc_dest::AbstractArray{<:Any,3},
                                           Fq,
                                           q::Int,
                                           corner::Symbol)
    _check_child_block_2d(Fc_dest, "Fc_dest")
    qi = _check_d2q9_q(q)
    _check_conservative_tree_corner(corner)
    _entry_corner_for_q(qi) == corner ||
        throw(ArgumentError("population q=$qi does not enter through corner $corner"))

    ix, iy = _corner_child_index(corner)
    @inbounds Fc_dest[ix, iy, qi] += Fq
    return Fc_dest
end

"""
    coalesce_fine_to_coarse_corner_F(Fc_src, q, corner)

Return the diagonal integrated packet leaving a refined parent through a
corner. The packet is read from the child cell adjacent to that corner.
"""
function coalesce_fine_to_coarse_corner_F(Fc_src::AbstractArray{<:Any,3},
                                          q::Int,
                                          corner::Symbol)
    _check_child_block_2d(Fc_src, "Fc_src")
    qi = _check_d2q9_q(q)
    _check_conservative_tree_corner(corner)
    _exit_corner_for_q(qi) == corner ||
        throw(ArgumentError("population q=$qi does not leave through corner $corner"))

    ix, iy = _corner_child_index(corner)
    @inbounds return Fc_src[ix, iy, qi]
end

function _check_patch_boundary_transfer_layout(Ffine::AbstractArray{<:Any,3},
                                               Fcoarse::AbstractArray{<:Any,3},
                                               patch)
    _check_conservative_tree_patch_layout(patch)
    size(Ffine) == size(patch.fine_F) ||
        throw(ArgumentError("Ffine must have the same size as patch.fine_F"))
    size(Fcoarse, 3) == 9 ||
        throw(ArgumentError("Fcoarse must have 9 D2Q9 populations in dimension 3"))
    return nothing
end

"""
    coarse_to_fine_patch_boundary_F_2d!(fine_dest, coarse_src, patch)

Accumulate all coarse populations that enter the refined patch during one
coarse-grid packet transfer. Only sources outside `patch.parent_*_range` and
inside `coarse_src` are considered. The destination is the fine child block
corresponding to the inactive parent ledger cell.
"""
function coarse_to_fine_patch_boundary_F_2d!(fine_dest::AbstractArray{<:Any,3},
                                             coarse_src::AbstractArray{<:Any,3},
                                             patch)
    _check_patch_boundary_transfer_layout(fine_dest, coarse_src, patch)

    @inbounds for jp_parent in patch.parent_j_range, ip_parent in patch.parent_i_range
        ip_local, jp_local = _patch_local_parent_index(patch, ip_parent, jp_parent)
        Fc_dest = _child_block_view(fine_dest, ip_local, jp_local)
        for q in 1:9
            cx = d2q9_cx(q)
            cy = d2q9_cy(q)
            cx == 0 && cy == 0 && continue

            isrc = ip_parent - cx
            jsrc = jp_parent - cy
            _inside_array_2d(coarse_src, isrc, jsrc) || continue
            _inside_range(isrc, jsrc, patch.parent_i_range, patch.parent_j_range) && continue

            di = isrc < first(patch.parent_i_range) ? -1 :
                 isrc > last(patch.parent_i_range) ? 1 : 0
            dj = jsrc < first(patch.parent_j_range) ? -1 :
                 jsrc > last(patch.parent_j_range) ? 1 : 0

            packet = coarse_src[isrc, jsrc, q]
            if di != 0 && dj != 0
                split_coarse_to_fine_corner_F_2d!(Fc_dest, packet, q,
                                                  _interface_corner_from_offset(di, dj))
            else
                split_coarse_to_fine_face_F_2d!(Fc_dest, packet, q,
                                                _interface_face_from_offset(di, dj))
            end
        end
    end
    return fine_dest
end

"""
    fine_to_coarse_patch_boundary_F_2d!(coarse_dest, fine_src, patch)

Accumulate all fine populations that leave the refined patch during a boundary
packet transfer. Only destinations outside `patch.parent_*_range` and inside
`coarse_dest` are considered.
"""
function fine_to_coarse_patch_boundary_F_2d!(coarse_dest::AbstractArray{<:Any,3},
                                             fine_src::AbstractArray{<:Any,3},
                                             patch)
    _check_patch_boundary_transfer_layout(fine_src, coarse_dest, patch)

    @inbounds for jp_parent in patch.parent_j_range, ip_parent in patch.parent_i_range
        ip_local, jp_local = _patch_local_parent_index(patch, ip_parent, jp_parent)
        Fc_src = _child_block_view(fine_src, ip_local, jp_local)
        for q in 1:9
            cx = d2q9_cx(q)
            cy = d2q9_cy(q)
            cx == 0 && cy == 0 && continue

            idst = ip_parent + cx
            jdst = jp_parent + cy
            _inside_array_2d(coarse_dest, idst, jdst) || continue
            _inside_range(idst, jdst, patch.parent_i_range, patch.parent_j_range) && continue

            di = idst < first(patch.parent_i_range) ? -1 :
                 idst > last(patch.parent_i_range) ? 1 : 0
            dj = jdst < first(patch.parent_j_range) ? -1 :
                 jdst > last(patch.parent_j_range) ? 1 : 0

            if di != 0 && dj != 0
                coarse_dest[idst, jdst, q] += coalesce_fine_to_coarse_corner_F(
                    Fc_src, q, _interface_corner_from_offset(di, dj))
            else
                coarse_dest[idst, jdst, q] += coalesce_fine_to_coarse_face_F(
                    Fc_src, q, _interface_face_from_offset(di, dj))
            end
        end
    end
    return coarse_dest
end

"""
    collide_BGK_integrated_D2Q9!(Fcell, volume, omega)

Apply a local BGK collision to one D2Q9 cell stored as integrated populations.
The conversion is `f_i = F_i / volume`; after collision the populations are
stored again as `F_i`.
"""
function collide_BGK_integrated_D2Q9!(Fcell::AbstractVector, volume, omega)
    _check_d2q9_vector(Fcell, "Fcell")
    volume > zero(volume) || throw(ArgumentError("volume must be positive"))

    m = mass_F(Fcell)
    iszero(m) && throw(ArgumentError("Fcell mass must be nonzero"))
    mx, my = momentum_F(Fcell)
    rho = m / volume
    ux = mx / m
    uy = my / m

    @inbounds for q in 1:9
        f = Fcell[q] / volume
        feq = equilibrium(D2Q9(), rho, ux, uy, q)
        Fcell[q] = (f - omega * (f - feq)) * volume
    end
    return Fcell
end

function collide_BGK_integrated_D2Q9!(F::AbstractArray{<:Any,3}, volume, omega)
    size(F, 3) == 9 || throw(ArgumentError("F must have 9 D2Q9 populations in dimension 3"))
    @inbounds for j in axes(F, 2), i in axes(F, 1)
        collide_BGK_integrated_D2Q9!(@view(F[i, j, :]), volume, omega)
    end
    return F
end

function collide_Guo_integrated_D2Q9!(Fcell::AbstractVector,
                                      volume,
                                      omega,
                                      Fx,
                                      Fy)
    _check_d2q9_vector(Fcell, "Fcell")
    volume > zero(volume) || throw(ArgumentError("volume must be positive"))

    m = mass_F(Fcell)
    iszero(m) && throw(ArgumentError("Fcell mass must be nonzero"))
    mx, my = momentum_F(Fcell)
    rho = m / volume
    ux = (mx / volume + Fx / 2) / rho
    uy = (my / volume + Fy / 2) / rho
    guo_pref = 1 - omega / 2

    @inbounds for q in 1:9
        cx = d2q9_cx(q)
        cy = d2q9_cy(q)
        w = weights(D2Q9())[q]
        ci_dot_u = cx * ux + cy * uy
        ci_dot_F = cx * Fx + cy * Fy
        Sq = w * (3 * ((cx - ux) * Fx + (cy - uy) * Fy) + 9 * ci_dot_u * ci_dot_F)
        f = Fcell[q] / volume
        feq = equilibrium(D2Q9(), rho, ux, uy, q)
        Fcell[q] = volume * (f - omega * (f - feq) + guo_pref * Sq)
    end
    return Fcell
end

function collide_Guo_integrated_D2Q9!(F::AbstractArray{<:Any,3},
                                      volume,
                                      omega,
                                      Fx,
                                      Fy)
    size(F, 3) == 9 || throw(ArgumentError("F must have 9 D2Q9 populations in dimension 3"))
    @inbounds for j in axes(F, 2), i in axes(F, 1)
        collide_Guo_integrated_D2Q9!(@view(F[i, j, :]), volume, omega, Fx, Fy)
    end
    return F
end

function _check_d2q9_grid_pair(Fout::AbstractArray{<:Any,3},
                               Fin::AbstractArray{<:Any,3})
    size(Fout) == size(Fin) || throw(ArgumentError("Fout and Fin must have the same size"))
    size(Fin, 3) == 9 || throw(ArgumentError("Fin must have 9 D2Q9 populations in dimension 3"))
    return nothing
end

"""
    stream_fully_periodic_F_2d!(Fout, Fin)

Pure pull streaming for a leaf grid stored as integrated D2Q9 populations, with
periodic boundaries in both directions.
"""
function stream_fully_periodic_F_2d!(Fout::AbstractArray{<:Any,3},
                                     Fin::AbstractArray{<:Any,3})
    Fout === Fin && throw(ArgumentError("Fout and Fin must be distinct arrays"))
    _check_d2q9_grid_pair(Fout, Fin)

    nx = size(Fin, 1)
    ny = size(Fin, 2)
    @inbounds for q in 1:9
        cx = d2q9_cx(q)
        cy = d2q9_cy(q)
        for j in 1:ny, i in 1:nx
            isrc = mod1(i - cx, nx)
            jsrc = mod1(j - cy, ny)
            Fout[i, j, q] = Fin[isrc, jsrc, q]
        end
    end
    return Fout
end

"""
    stream_periodic_x_wall_y_F_2d!(Fout, Fin)

Pure pull streaming for a leaf grid stored as integrated D2Q9 populations, with
periodic `x` and stationary bounce-back walls at the south/north `y`
boundaries. This is a leaf-grid boundary condition; inactive parent ledgers
must not be streamed through this operator.
"""
function stream_periodic_x_wall_y_F_2d!(Fout::AbstractArray{<:Any,3},
                                        Fin::AbstractArray{<:Any,3})
    Fout === Fin && throw(ArgumentError("Fout and Fin must be distinct arrays"))
    _check_d2q9_grid_pair(Fout, Fin)

    nx = size(Fin, 1)
    ny = size(Fin, 2)
    @inbounds for j in 1:ny, i in 1:nx
        im = i > 1 ? i - 1 : nx
        ip = i < nx ? i + 1 : 1

        Fout[i, j, 1] = Fin[i, j, 1]
        Fout[i, j, 2] = Fin[im, j, 2]
        Fout[i, j, 4] = Fin[ip, j, 4]

        if j == 1
            Fout[i, j, 3] = Fin[i, j, 5]
            Fout[i, j, 5] = ny == 1 ? Fin[i, j, 3] : Fin[i, j+1, 5]
            Fout[i, j, 6] = Fin[i, j, 8]
            Fout[i, j, 7] = Fin[i, j, 9]
            Fout[i, j, 8] = ny == 1 ? Fin[i, j, 6] : Fin[ip, j+1, 8]
            Fout[i, j, 9] = ny == 1 ? Fin[i, j, 7] : Fin[im, j+1, 9]
        elseif j == ny
            Fout[i, j, 3] = Fin[i, j-1, 3]
            Fout[i, j, 5] = Fin[i, j, 3]
            Fout[i, j, 6] = Fin[im, j-1, 6]
            Fout[i, j, 7] = Fin[ip, j-1, 7]
            Fout[i, j, 8] = Fin[i, j, 6]
            Fout[i, j, 9] = Fin[i, j, 7]
        else
            Fout[i, j, 3] = Fin[i, j-1, 3]
            Fout[i, j, 5] = Fin[i, j+1, 5]
            Fout[i, j, 6] = Fin[im, j-1, 6]
            Fout[i, j, 7] = Fin[ip, j-1, 7]
            Fout[i, j, 8] = Fin[ip, j+1, 8]
            Fout[i, j, 9] = Fin[im, j+1, 9]
        end
    end
    return Fout
end

@inline function _moving_wall_delta(volume, rho_wall, wall_u, q::Int)
    return volume * 2 * (1 / 36) * rho_wall * d2q9_cx(q) * wall_u / (1 / 3)
end

"""
    stream_periodic_x_moving_wall_y_F_2d!(Fout, Fin; u_south=0, u_north=0,
                                          rho_wall=1, volume=1)

Pure pull streaming for a leaf grid with periodic `x` and bounce-back walls at
the south/north `y` boundaries. Tangential wall velocities are included through
the standard moving-wall bounce-back correction on diagonal populations.
"""
function stream_periodic_x_moving_wall_y_F_2d!(Fout::AbstractArray{<:Any,3},
                                               Fin::AbstractArray{<:Any,3};
                                               u_south=0,
                                               u_north=0,
                                               rho_wall=1,
                                               volume=1)
    Fout === Fin && throw(ArgumentError("Fout and Fin must be distinct arrays"))
    _check_d2q9_grid_pair(Fout, Fin)

    nx = size(Fin, 1)
    ny = size(Fin, 2)
    @inbounds for j in 1:ny, i in 1:nx
        im = i > 1 ? i - 1 : nx
        ip = i < nx ? i + 1 : 1

        Fout[i, j, 1] = Fin[i, j, 1]
        Fout[i, j, 2] = Fin[im, j, 2]
        Fout[i, j, 4] = Fin[ip, j, 4]

        if j == 1
            Fout[i, j, 3] = Fin[i, j, 5]
            Fout[i, j, 5] = ny == 1 ? Fin[i, j, 3] : Fin[i, j+1, 5]
            Fout[i, j, 6] = Fin[i, j, 8] + _moving_wall_delta(volume, rho_wall, u_south, 6)
            Fout[i, j, 7] = Fin[i, j, 9] + _moving_wall_delta(volume, rho_wall, u_south, 7)
            Fout[i, j, 8] = ny == 1 ? Fin[i, j, 6] : Fin[ip, j+1, 8]
            Fout[i, j, 9] = ny == 1 ? Fin[i, j, 7] : Fin[im, j+1, 9]
        elseif j == ny
            Fout[i, j, 3] = Fin[i, j-1, 3]
            Fout[i, j, 5] = Fin[i, j, 3]
            Fout[i, j, 6] = Fin[im, j-1, 6]
            Fout[i, j, 7] = Fin[ip, j-1, 7]
            Fout[i, j, 8] = Fin[i, j, 6] + _moving_wall_delta(volume, rho_wall, u_north, 8)
            Fout[i, j, 9] = Fin[i, j, 7] + _moving_wall_delta(volume, rho_wall, u_north, 9)
        else
            Fout[i, j, 3] = Fin[i, j-1, 3]
            Fout[i, j, 5] = Fin[i, j+1, 5]
            Fout[i, j, 6] = Fin[im, j-1, 6]
            Fout[i, j, 7] = Fin[ip, j-1, 7]
            Fout[i, j, 8] = Fin[ip, j+1, 8]
            Fout[i, j, 9] = Fin[im, j+1, 9]
        end
    end
    return Fout
end

function cylinder_solid_mask_leaf_2d(Nx::Int, Ny::Int, cx, cy, radius)
    mask = falses(Nx, Ny)
    r2 = radius * radius
    @inbounds for j in 1:Ny, i in 1:Nx
        mask[i, j] = (i - cx)^2 + (j - cy)^2 <= r2
    end
    return mask
end

function square_solid_mask_leaf_2d(Nx::Int, Ny::Int,
                                   i_range::AbstractUnitRange{<:Integer},
                                   j_range::AbstractUnitRange{<:Integer})
    isempty(i_range) && throw(ArgumentError("i_range must be nonempty"))
    isempty(j_range) && throw(ArgumentError("j_range must be nonempty"))
    first(i_range) >= 1 && last(i_range) <= Nx ||
        throw(ArgumentError("i_range must be inside 1:Nx"))
    first(j_range) >= 1 && last(j_range) <= Ny ||
        throw(ArgumentError("j_range must be inside 1:Ny"))

    mask = falses(Nx, Ny)
    @inbounds for j in Int(first(j_range)):Int(last(j_range)),
                  i in Int(first(i_range)):Int(last(i_range))
        mask[i, j] = true
    end
    return mask
end

function backward_facing_step_solid_mask_leaf_2d(Nx::Int, Ny::Int,
                                                 step_i::Int,
                                                 step_height::Int)
    1 <= step_i < Nx || throw(ArgumentError("step_i must be inside 1:Nx-1"))
    1 <= step_height < Ny - 1 ||
        throw(ArgumentError("step_height must leave at least two open rows"))

    mask = falses(Nx, Ny)
    @inbounds for j in 1:step_height, i in 1:step_i
        mask[i, j] = true
    end
    return mask
end

function _check_solid_mask_layout(F::AbstractArray{<:Any,3},
                                  is_solid::AbstractArray{Bool,2})
    size(is_solid) == (size(F, 1), size(F, 2)) ||
        throw(ArgumentError("is_solid must match the first two dimensions of F"))
    return nothing
end

"""
    stream_periodic_x_wall_y_solid_F_2d!(Fout, Fin, is_solid)

Pull streaming on a leaf grid with periodic x, channel walls in y, and
halfway bounce-back on solid links. Solid destination cells are filled with
their previous values and are ignored by the solid-aware collision routines.
"""
function stream_periodic_x_wall_y_solid_F_2d!(Fout::AbstractArray{<:Any,3},
                                              Fin::AbstractArray{<:Any,3},
                                              is_solid::AbstractArray{Bool,2})
    Fout === Fin && throw(ArgumentError("Fout and Fin must be distinct arrays"))
    _check_d2q9_grid_pair(Fout, Fin)
    _check_solid_mask_layout(Fin, is_solid)

    nx = size(Fin, 1)
    ny = size(Fin, 2)
    @inbounds for j in 1:ny, i in 1:nx
        if is_solid[i, j]
            for q in 1:9
                Fout[i, j, q] = Fin[i, j, q]
            end
            continue
        end

        for q in 1:9
            cx = d2q9_cx(q)
            cy = d2q9_cy(q)
            if cx == 0 && cy == 0
                Fout[i, j, q] = Fin[i, j, q]
                continue
            end

            isrc = mod1(i - cx, nx)
            jsrc = j - cy
            if jsrc < 1 || jsrc > ny
                Fout[i, j, q] = Fin[i, j, d2q9_opposite(q)]
            elseif is_solid[isrc, jsrc]
                Fout[i, j, q] = Fin[i, j, d2q9_opposite(q)]
            else
                Fout[i, j, q] = Fin[isrc, jsrc, q]
            end
        end
    end
    return Fout
end

function stream_bounceback_xy_solid_F_2d!(Fout::AbstractArray{<:Any,3},
                                          Fin::AbstractArray{<:Any,3},
                                          is_solid::AbstractArray{Bool,2})
    Fout === Fin && throw(ArgumentError("Fout and Fin must be distinct arrays"))
    _check_d2q9_grid_pair(Fout, Fin)
    _check_solid_mask_layout(Fin, is_solid)

    nx = size(Fin, 1)
    ny = size(Fin, 2)
    @inbounds for j in 1:ny, i in 1:nx
        if is_solid[i, j]
            for q in 1:9
                Fout[i, j, q] = Fin[i, j, q]
            end
            continue
        end

        for q in 1:9
            cx = d2q9_cx(q)
            cy = d2q9_cy(q)
            if cx == 0 && cy == 0
                Fout[i, j, q] = Fin[i, j, q]
                continue
            end

            isrc = i - cx
            jsrc = j - cy
            if isrc < 1 || isrc > nx || jsrc < 1 || jsrc > ny
                Fout[i, j, q] = Fin[i, j, d2q9_opposite(q)]
            elseif is_solid[isrc, jsrc]
                Fout[i, j, q] = Fin[i, j, d2q9_opposite(q)]
            else
                Fout[i, j, q] = Fin[isrc, jsrc, q]
            end
        end
    end
    return Fout
end

function apply_zou_he_west_F_2d!(F::AbstractArray{T,3}, u_in, volume) where T
    size(F, 3) == 9 || throw(ArgumentError("F must have 9 D2Q9 populations in dimension 3"))
    u = T(u_in)
    vol = T(volume)
    @inbounds for j in axes(F, 2)
        f1 = F[1, j, 1] / vol
        f3 = F[1, j, 3] / vol
        f4 = F[1, j, 4] / vol
        f5 = F[1, j, 5] / vol
        f7 = F[1, j, 7] / vol
        f8 = F[1, j, 8] / vol
        rho_wall = (f1 + f3 + f5 + T(2) * (f4 + f7 + f8)) / (one(T) - u)
        F[1, j, 2] = vol * (f4 + T(2//3) * rho_wall * u)
        F[1, j, 6] = vol * (f8 - T(0.5) * (f3 - f5) + T(1//6) * rho_wall * u)
        F[1, j, 9] = vol * (f7 + T(0.5) * (f3 - f5) + T(1//6) * rho_wall * u)
    end
    return F
end

function apply_zou_he_west_F_2d!(F::AbstractArray{T,3}, u_in, volume,
                                 is_solid::AbstractArray{Bool,2}) where T
    size(F, 3) == 9 || throw(ArgumentError("F must have 9 D2Q9 populations in dimension 3"))
    _check_solid_mask_layout(F, is_solid)
    u = T(u_in)
    vol = T(volume)
    @inbounds for j in axes(F, 2)
        is_solid[1, j] && continue
        f1 = F[1, j, 1] / vol
        f3 = F[1, j, 3] / vol
        f4 = F[1, j, 4] / vol
        f5 = F[1, j, 5] / vol
        f7 = F[1, j, 7] / vol
        f8 = F[1, j, 8] / vol
        rho_wall = (f1 + f3 + f5 + T(2) * (f4 + f7 + f8)) / (one(T) - u)
        F[1, j, 2] = vol * (f4 + T(2//3) * rho_wall * u)
        F[1, j, 6] = vol * (f8 - T(0.5) * (f3 - f5) + T(1//6) * rho_wall * u)
        F[1, j, 9] = vol * (f7 + T(0.5) * (f3 - f5) + T(1//6) * rho_wall * u)
    end
    return F
end

function apply_zou_he_pressure_east_F_2d!(F::AbstractArray{T,3}, volume;
                                          rho_out=one(T)) where T
    size(F, 3) == 9 || throw(ArgumentError("F must have 9 D2Q9 populations in dimension 3"))
    vol = T(volume)
    rho = T(rho_out)
    i = last(axes(F, 1))
    @inbounds for j in axes(F, 2)
        f1 = F[i, j, 1] / vol
        f2 = F[i, j, 2] / vol
        f3 = F[i, j, 3] / vol
        f5 = F[i, j, 5] / vol
        f6 = F[i, j, 6] / vol
        f9 = F[i, j, 9] / vol
        ux = -one(T) + (f1 + f3 + f5 + T(2) * (f2 + f6 + f9)) / rho
        F[i, j, 4] = vol * (f2 - T(2//3) * rho * ux)
        F[i, j, 7] = vol * (f9 - T(0.5) * (f3 - f5) - T(1//6) * rho * ux)
        F[i, j, 8] = vol * (f6 + T(0.5) * (f3 - f5) - T(1//6) * rho * ux)
    end
    return F
end

function apply_zou_he_pressure_east_F_2d!(F::AbstractArray{T,3}, volume,
                                          is_solid::AbstractArray{Bool,2};
                                          rho_out=one(T)) where T
    size(F, 3) == 9 || throw(ArgumentError("F must have 9 D2Q9 populations in dimension 3"))
    _check_solid_mask_layout(F, is_solid)
    vol = T(volume)
    rho = T(rho_out)
    i = last(axes(F, 1))
    @inbounds for j in axes(F, 2)
        is_solid[i, j] && continue
        f1 = F[i, j, 1] / vol
        f2 = F[i, j, 2] / vol
        f3 = F[i, j, 3] / vol
        f5 = F[i, j, 5] / vol
        f6 = F[i, j, 6] / vol
        f9 = F[i, j, 9] / vol
        ux = -one(T) + (f1 + f3 + f5 + T(2) * (f2 + f6 + f9)) / rho
        F[i, j, 4] = vol * (f2 - T(2//3) * rho * ux)
        F[i, j, 7] = vol * (f9 - T(0.5) * (f3 - f5) - T(1//6) * rho * ux)
        F[i, j, 8] = vol * (f6 + T(0.5) * (f3 - f5) - T(1//6) * rho * ux)
    end
    return F
end

function collide_BGK_integrated_D2Q9!(F::AbstractArray{<:Any,3},
                                      is_solid::AbstractArray{Bool,2},
                                      volume,
                                      omega)
    size(F, 3) == 9 || throw(ArgumentError("F must have 9 D2Q9 populations in dimension 3"))
    _check_solid_mask_layout(F, is_solid)
    @inbounds for j in axes(F, 2), i in axes(F, 1)
        is_solid[i, j] && continue
        collide_BGK_integrated_D2Q9!(@view(F[i, j, :]), volume, omega)
    end
    return F
end

function collide_Guo_integrated_D2Q9!(F::AbstractArray{<:Any,3},
                                      is_solid::AbstractArray{Bool,2},
                                      volume,
                                      omega,
                                      Fx,
                                      Fy)
    size(F, 3) == 9 || throw(ArgumentError("F must have 9 D2Q9 populations in dimension 3"))
    _check_solid_mask_layout(F, is_solid)
    @inbounds for j in axes(F, 2), i in axes(F, 1)
        is_solid[i, j] && continue
        collide_Guo_integrated_D2Q9!(@view(F[i, j, :]), volume, omega, Fx, Fy)
    end
    return F
end

function compute_drag_mea_solid_F_2d(Fpre::AbstractArray{<:Any,3},
                                     Fpost::AbstractArray{<:Any,3},
                                     is_solid::AbstractArray{Bool,2})
    _check_d2q9_grid_pair(Fpost, Fpre)
    _check_solid_mask_layout(Fpre, is_solid)

    Fx = 0.0
    Fy = 0.0
    nx = size(Fpre, 1)
    ny = size(Fpre, 2)
    @inbounds for j in 1:ny, i in 1:nx
        is_solid[i, j] && continue
        for q in 2:9
            ni = mod1(i + d2q9_cx(q), nx)
            nj = j + d2q9_cy(q)
            1 <= nj <= ny || continue
            is_solid[ni, nj] || continue
            oq = d2q9_opposite(q)
            Fx += d2q9_cx(q) * (Fpre[i, j, q] + Fpost[i, j, oq])
            Fy += d2q9_cy(q) * (Fpre[i, j, q] + Fpost[i, j, oq])
        end
    end
    return (Fx=Fx, Fy=Fy)
end

"""
    ConservativeTreePatch2D

Experimental fixed ratio-2 patch for the conservative tree route.

`fine_F` is the active fine state inside the refined region. `coarse_shadow_F`
is only a parent ledger/agregate over the same physical region; it is not an
active fluid state.
"""
struct ConservativeTreePatch2D{T}
    parent_i_range::UnitRange{Int}
    parent_j_range::UnitRange{Int}
    ratio::Int
    fine_F::Array{T,3}
    coarse_shadow_F::Array{T,3}
end

"""
    create_conservative_tree_patch_2d(parent_i_range, parent_j_range; ratio=2, T=Float64)

Allocate an experimental fixed ratio-2 conservative-tree patch. No ghost cells
are allocated.
"""
function create_conservative_tree_patch_2d(parent_i_range::AbstractUnitRange{<:Integer},
                                           parent_j_range::AbstractUnitRange{<:Integer};
                                           ratio::Int=2,
                                           T::Type{<:Real}=Float64)
    ratio == 2 || throw(ArgumentError("only ratio=2 is implemented"))
    isempty(parent_i_range) && throw(ArgumentError("parent_i_range must be nonempty"))
    isempty(parent_j_range) && throw(ArgumentError("parent_j_range must be nonempty"))

    ip = Int(first(parent_i_range)):Int(last(parent_i_range))
    jp = Int(first(parent_j_range)):Int(last(parent_j_range))
    nx_parent = length(ip)
    ny_parent = length(jp)

    fine_F = zeros(T, ratio * nx_parent, ratio * ny_parent, 9)
    coarse_shadow_F = zeros(T, nx_parent, ny_parent, 9)
    return ConservativeTreePatch2D{T}(ip, jp, ratio, fine_F, coarse_shadow_F)
end

function _check_conservative_tree_patch_layout(patch::ConservativeTreePatch2D)
    patch.ratio == 2 || throw(ArgumentError("only ratio=2 is implemented"))
    nx_parent = length(patch.parent_i_range)
    ny_parent = length(patch.parent_j_range)
    size(patch.fine_F) == (2 * nx_parent, 2 * ny_parent, 9) ||
        throw(ArgumentError("patch.fine_F has inconsistent size"))
    size(patch.coarse_shadow_F) == (nx_parent, ny_parent, 9) ||
        throw(ArgumentError("patch.coarse_shadow_F has inconsistent size"))
    return nothing
end

"""
    coalesce_patch_to_shadow_F_2d!(patch)

Fill `patch.coarse_shadow_F` by coalescing every 2x2 fine block. This is the
fine-to-parent ledger update for the conservative tree route.
"""
function coalesce_patch_to_shadow_F_2d!(patch::ConservativeTreePatch2D)
    _check_conservative_tree_patch_layout(patch)

    nx_parent = length(patch.parent_i_range)
    ny_parent = length(patch.parent_j_range)
    @inbounds for jp in 1:ny_parent, ip in 1:nx_parent
        i0 = 2 * ip - 1
        j0 = 2 * jp - 1
        Fp = @view patch.coarse_shadow_F[ip, jp, :]
        Fc = @view patch.fine_F[i0:i0+1, j0:j0+1, :]
        coalesce_F_2d!(Fp, Fc)
    end
    return patch
end

"""
    explode_shadow_to_patch_uniform_F_2d!(patch)

Fill `patch.fine_F` by uniformly exploding each parent ledger cell into its
four fine children.
"""
function explode_shadow_to_patch_uniform_F_2d!(patch::ConservativeTreePatch2D)
    _check_conservative_tree_patch_layout(patch)

    nx_parent = length(patch.parent_i_range)
    ny_parent = length(patch.parent_j_range)
    @inbounds for jp in 1:ny_parent, ip in 1:nx_parent
        i0 = 2 * ip - 1
        j0 = 2 * jp - 1
        Fc = @view patch.fine_F[i0:i0+1, j0:j0+1, :]
        Fp = @view patch.coarse_shadow_F[ip, jp, :]
        explode_uniform_F_2d!(Fc, Fp)
    end
    return patch
end

function _check_composite_coarse_layout(coarse_F::AbstractArray{<:Any,3},
                                        patch::ConservativeTreePatch2D)
    _check_conservative_tree_patch_layout(patch)
    size(coarse_F, 3) == 9 ||
        throw(ArgumentError("coarse_F must have 9 D2Q9 populations in dimension 3"))
    first(patch.parent_i_range) >= first(axes(coarse_F, 1)) ||
        throw(ArgumentError("patch.parent_i_range starts outside coarse_F"))
    last(patch.parent_i_range) <= last(axes(coarse_F, 1)) ||
        throw(ArgumentError("patch.parent_i_range ends outside coarse_F"))
    first(patch.parent_j_range) >= first(axes(coarse_F, 2)) ||
        throw(ArgumentError("patch.parent_j_range starts outside coarse_F"))
    last(patch.parent_j_range) <= last(axes(coarse_F, 2)) ||
        throw(ArgumentError("patch.parent_j_range ends outside coarse_F"))
    return nothing
end

"""
    active_mass_F(coarse_F, patch)

Mass of a composite fixed-tree state: active coarse cells outside the refined
parent range plus active fine cells inside `patch`. The inactive parent ledger
region in `coarse_F` is deliberately skipped.
"""
function active_mass_F(coarse_F::AbstractArray{<:Any,3},
                       patch::ConservativeTreePatch2D)
    _check_composite_coarse_layout(coarse_F, patch)

    total = zero(coarse_F[begin, begin, 1] + patch.fine_F[begin, begin, 1])
    @inbounds for q in 1:9, j in axes(coarse_F, 2), i in axes(coarse_F, 1)
        _inside_range(i, j, patch.parent_i_range, patch.parent_j_range) && continue
        total += coarse_F[i, j, q]
    end
    return total + mass_F(patch.fine_F)
end

"""
    active_population_sums_F(coarse_F, patch)

Return the 9 active oriented-population totals of a composite fixed-tree state.
Inactive parent cells covered by the refined patch are skipped.
"""
function active_population_sums_F(coarse_F::AbstractArray{<:Any,3},
                                  patch::ConservativeTreePatch2D)
    _check_composite_coarse_layout(coarse_F, patch)

    totals = zeros(promote_type(eltype(coarse_F), eltype(patch.fine_F)), 9)
    @inbounds for q in 1:9
        for j in axes(coarse_F, 2), i in axes(coarse_F, 1)
            _inside_range(i, j, patch.parent_i_range, patch.parent_j_range) && continue
            totals[q] += coarse_F[i, j, q]
        end
        totals[q] += sum(@view patch.fine_F[:, :, q])
    end
    return totals
end

"""
    active_momentum_F(coarse_F, patch)

Momentum of a composite fixed-tree state, counting only active coarse cells and
active fine leaves.
"""
function active_momentum_F(coarse_F::AbstractArray{<:Any,3},
                           patch::ConservativeTreePatch2D)
    _check_composite_coarse_layout(coarse_F, patch)

    mx = zero(coarse_F[begin, begin, 1] + patch.fine_F[begin, begin, 1])
    my = zero(coarse_F[begin, begin, 1] + patch.fine_F[begin, begin, 1])
    @inbounds for q in 1:9
        cx = d2q9_cx(q)
        cy = d2q9_cy(q)
        for j in axes(coarse_F, 2), i in axes(coarse_F, 1)
            _inside_range(i, j, patch.parent_i_range, patch.parent_j_range) && continue
            fq = coarse_F[i, j, q]
            mx += cx * fq
            my += cy * fq
        end
    end
    fmx, fmy = momentum_F(patch.fine_F)
    return mx + fmx, my + fmy
end

function active_moments_F(coarse_F::AbstractArray{<:Any,3},
                          patch::ConservativeTreePatch2D)
    m = active_mass_F(coarse_F, patch)
    mx, my = active_momentum_F(coarse_F, patch)
    return m, mx, my
end

@inline function _composite_parent_Fq(coarse_F::AbstractArray{<:Any,3},
                                      patch::ConservativeTreePatch2D,
                                      I::Int,
                                      J::Int,
                                      q::Int)
    if _inside_range(I, J, patch.parent_i_range, patch.parent_j_range)
        il, jl = _patch_local_parent_index(patch, I, J)
        return patch.coarse_shadow_F[il, jl, q]
    end
    return coarse_F[I, J, q]
end

@inline function _limited_parent_slope_x(coarse_F::AbstractArray{<:Any,3},
                                         patch::ConservativeTreePatch2D,
                                         I::Int,
                                         J::Int,
                                         q::Int)
    center = _composite_parent_Fq(coarse_F, patch, I, J, q)
    has_left = I > first(axes(coarse_F, 1))
    has_right = I < last(axes(coarse_F, 1))
    if has_left && has_right
        left = center - _composite_parent_Fq(coarse_F, patch, I - 1, J, q)
        right = _composite_parent_Fq(coarse_F, patch, I + 1, J, q) - center
        return _minmod(left, right)
    elseif has_left
        return center - _composite_parent_Fq(coarse_F, patch, I - 1, J, q)
    elseif has_right
        return _composite_parent_Fq(coarse_F, patch, I + 1, J, q) - center
    else
        return zero(center)
    end
end

@inline function _limited_parent_slope_y(coarse_F::AbstractArray{<:Any,3},
                                         patch::ConservativeTreePatch2D,
                                         I::Int,
                                         J::Int,
                                         q::Int)
    center = _composite_parent_Fq(coarse_F, patch, I, J, q)
    has_south = J > first(axes(coarse_F, 2))
    has_north = J < last(axes(coarse_F, 2))
    if has_south && has_north
        south = center - _composite_parent_Fq(coarse_F, patch, I, J - 1, q)
        north = _composite_parent_Fq(coarse_F, patch, I, J + 1, q) - center
        return _minmod(south, north)
    elseif has_south
        return center - _composite_parent_Fq(coarse_F, patch, I, J - 1, q)
    elseif has_north
        return _composite_parent_Fq(coarse_F, patch, I, J + 1, q) - center
    else
        return zero(center)
    end
end

function _explode_limited_linear_composite_F_2d!(
        leaf_block::AbstractArray{<:Any,3},
        coarse_F::AbstractArray{<:Any,3},
        patch::ConservativeTreePatch2D,
        I::Int,
        J::Int)
    _check_child_block_2d(leaf_block, "leaf_block")

    @inbounds for q in 1:9
        center = _composite_parent_Fq(coarse_F, patch, I, J, q)
        sx = _limited_parent_slope_x(coarse_F, patch, I, J, q)
        sy = _limited_parent_slope_y(coarse_F, patch, I, J, q)

        max_delta = (abs(sx) + abs(sy)) / 16
        base = center / 4
        if max_delta > zero(max_delta) && base < max_delta
            theta = base / max_delta
            sx *= theta
            sy *= theta
        end

        leaf_block[1, 1, q] = base - sx / 16 - sy / 16
        leaf_block[2, 1, q] = base + sx / 16 - sy / 16
        leaf_block[1, 2, q] = base - sx / 16 + sy / 16
        leaf_block[2, 2, q] = base + sx / 16 + sy / 16
    end
    return leaf_block
end

function _check_composite_pair_layout(coarse_out::AbstractArray{<:Any,3},
                                      patch_out::ConservativeTreePatch2D,
                                      coarse_in::AbstractArray{<:Any,3},
                                      patch_in::ConservativeTreePatch2D)
    size(coarse_out) == size(coarse_in) ||
        throw(ArgumentError("coarse_out and coarse_in must have the same size"))
    patch_out.parent_i_range == patch_in.parent_i_range ||
        throw(ArgumentError("patch_out and patch_in must have the same parent_i_range"))
    patch_out.parent_j_range == patch_in.parent_j_range ||
        throw(ArgumentError("patch_out and patch_in must have the same parent_j_range"))
    size(patch_out.fine_F) == size(patch_in.fine_F) ||
        throw(ArgumentError("patch_out and patch_in fine arrays must have the same size"))
    _check_composite_coarse_layout(coarse_in, patch_in)
    _check_composite_coarse_layout(coarse_out, patch_out)
    return nothing
end

function _check_leaf_layout(leaf_F::AbstractArray{<:Any,3},
                            coarse_F::AbstractArray{<:Any,3})
    size(leaf_F) == (2 * size(coarse_F, 1), 2 * size(coarse_F, 2), 9) ||
        throw(ArgumentError("leaf_F must have size (2*Nx_coarse, 2*Ny_coarse, 9)"))
    return nothing
end

"""
    composite_to_leaf_F_2d!(leaf_F, coarse_F, patch)

Expand a composite fixed-tree state to a uniform leaf grid. Active fine leaves
are copied inside `patch`; active coarse cells outside the patch are uniformly
exploded to their four ratio-2 leaves. Inactive coarse cells covered by the
patch are ignored.
"""
function composite_to_leaf_F_2d!(leaf_F::AbstractArray{<:Any,3},
                                 coarse_F::AbstractArray{<:Any,3},
                                 patch::ConservativeTreePatch2D)
    _check_composite_coarse_layout(coarse_F, patch)
    _check_leaf_layout(leaf_F, coarse_F)
    coalesce_patch_to_shadow_F_2d!(patch)

    @inbounds for J in axes(coarse_F, 2), I in axes(coarse_F, 1)
        i0 = 2 * I - 1
        j0 = 2 * J - 1
        leaf_block = @view leaf_F[i0:i0+1, j0:j0+1, :]

        if _inside_range(I, J, patch.parent_i_range, patch.parent_j_range)
            il, jl = _patch_local_parent_index(patch, I, J)
            fine_block = _child_block_view(patch.fine_F, il, jl)
            leaf_block .= fine_block
        else
            _explode_limited_linear_composite_F_2d!(leaf_block, coarse_F, patch, I, J)
        end
    end
    return leaf_F
end

"""
    leaf_to_composite_F_2d!(coarse_F, patch, leaf_F)

Restrict a uniform leaf grid back to the composite fixed-tree representation.
Outside `patch`, each 2x2 leaf block is coalesced to an active coarse cell.
Inside `patch`, the leaf values are copied to active fine leaves. The inactive
coarse parent region is zeroed in `coarse_F`.
"""
function leaf_to_composite_F_2d!(coarse_F::AbstractArray{<:Any,3},
                                 patch::ConservativeTreePatch2D,
                                 leaf_F::AbstractArray{<:Any,3})
    _check_composite_coarse_layout(coarse_F, patch)
    _check_leaf_layout(leaf_F, coarse_F)
    coarse_F .= 0

    @inbounds for J in axes(coarse_F, 2), I in axes(coarse_F, 1)
        i0 = 2 * I - 1
        j0 = 2 * J - 1
        leaf_block = @view leaf_F[i0:i0+1, j0:j0+1, :]

        if _inside_range(I, J, patch.parent_i_range, patch.parent_j_range)
            il, jl = _patch_local_parent_index(patch, I, J)
            fine_block = _child_block_view(patch.fine_F, il, jl)
            fine_block .= leaf_block
        else
            coalesce_F_2d!(@view(coarse_F[I, J, :]), leaf_block)
        end
    end

    coalesce_patch_to_shadow_F_2d!(patch)
    return coarse_F, patch
end

"""
    stream_composite_fully_periodic_leaf_F_2d!(coarse_out, patch_out,
                                               coarse_in, patch_in)

Conservative prototype stream step for a fixed-tree composite state. The state
is expanded to a uniform ratio-2 leaf grid, streamed periodically by one leaf
cell, then restricted back to coarse-outside/fine-inside ownership.

This is a topology/invariant canary, not the final physical subcycled
coarse/fine time integrator.
"""
function stream_composite_fully_periodic_leaf_F_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D)
    _check_composite_pair_layout(coarse_out, patch_out, coarse_in, patch_in)

    leaf_in = similar(coarse_in, 2 * size(coarse_in, 1), 2 * size(coarse_in, 2), 9)
    leaf_out = similar(leaf_in)
    composite_to_leaf_F_2d!(leaf_in, coarse_in, patch_in)
    stream_fully_periodic_F_2d!(leaf_out, leaf_in)
    leaf_to_composite_F_2d!(coarse_out, patch_out, leaf_out)
    return coarse_out, patch_out
end

function stream_composite_periodic_x_wall_y_leaf_F_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D)
    _check_composite_pair_layout(coarse_out, patch_out, coarse_in, patch_in)

    leaf_in = similar(coarse_in, 2 * size(coarse_in, 1), 2 * size(coarse_in, 2), 9)
    leaf_out = similar(leaf_in)
    composite_to_leaf_F_2d!(leaf_in, coarse_in, patch_in)
    stream_periodic_x_wall_y_F_2d!(leaf_out, leaf_in)
    leaf_to_composite_F_2d!(coarse_out, patch_out, leaf_out)
    return coarse_out, patch_out
end

function stream_composite_periodic_x_moving_wall_y_leaf_F_2d!(
        coarse_out::AbstractArray{<:Any,3},
        patch_out::ConservativeTreePatch2D,
        coarse_in::AbstractArray{<:Any,3},
        patch_in::ConservativeTreePatch2D;
        u_south=0,
        u_north=0,
        rho_wall=1,
        volume_leaf=1)
    _check_composite_pair_layout(coarse_out, patch_out, coarse_in, patch_in)

    leaf_in = similar(coarse_in, 2 * size(coarse_in, 1), 2 * size(coarse_in, 2), 9)
    leaf_out = similar(leaf_in)
    composite_to_leaf_F_2d!(leaf_in, coarse_in, patch_in)
    stream_periodic_x_moving_wall_y_F_2d!(leaf_out, leaf_in;
        u_south=u_south, u_north=u_north, rho_wall=rho_wall, volume=volume_leaf)
    leaf_to_composite_F_2d!(coarse_out, patch_out, leaf_out)
    return coarse_out, patch_out
end

"""
    collide_BGK_composite_F_2d!(coarse_F, patch, volume_coarse, volume_fine,
                                omega_coarse, omega_fine)

Collide only active cells of a fixed-tree composite state: coarse cells outside
the refined parent range and fine leaves inside `patch`.
"""
function collide_BGK_composite_F_2d!(coarse_F::AbstractArray{<:Any,3},
                                     patch::ConservativeTreePatch2D,
                                     volume_coarse,
                                     volume_fine,
                                     omega_coarse,
                                     omega_fine)
    _check_composite_coarse_layout(coarse_F, patch)

    @inbounds for j in axes(coarse_F, 2), i in axes(coarse_F, 1)
        _inside_range(i, j, patch.parent_i_range, patch.parent_j_range) && continue
        collide_BGK_integrated_D2Q9!(@view(coarse_F[i, j, :]), volume_coarse, omega_coarse)
    end
    collide_BGK_integrated_D2Q9!(patch.fine_F, volume_fine, omega_fine)
    coalesce_patch_to_shadow_F_2d!(patch)
    return coarse_F, patch
end

function collide_Guo_composite_F_2d!(coarse_F::AbstractArray{<:Any,3},
                                     patch::ConservativeTreePatch2D,
                                     volume_coarse,
                                     volume_fine,
                                     omega_coarse,
                                     omega_fine,
                                     Fx,
                                     Fy)
    _check_composite_coarse_layout(coarse_F, patch)

    @inbounds for j in axes(coarse_F, 2), i in axes(coarse_F, 1)
        _inside_range(i, j, patch.parent_i_range, patch.parent_j_range) && continue
        collide_Guo_integrated_D2Q9!(@view(coarse_F[i, j, :]), volume_coarse,
                                     omega_coarse, Fx, Fy)
    end
    collide_Guo_integrated_D2Q9!(patch.fine_F, volume_fine, omega_fine, Fx, Fy)
    coalesce_patch_to_shadow_F_2d!(patch)
    return coarse_F, patch
end

struct ConservativeTreeMacroFlow2D{T}
    flow::Symbol
    coarse_F::Array{T,3}
    patch::ConservativeTreePatch2D{T}
    ux_profile::Vector{T}
    analytic_ux_profile::Vector{T}
    l2_error::T
    linf_error::T
    mass_initial::T
    mass_final::T
    mass_drift::T
    steps::Int
end

struct ConservativeTreeCylinderResult2D{T}
    coarse_F::Array{T,3}
    patch::ConservativeTreePatch2D{T}
    is_solid_leaf::BitMatrix
    Fx_drag::T
    Fy_drag::T
    Cd::T
    u_ref::T
    mass_initial::T
    mass_final::T
    mass_drift::T
    steps::Int
    avg_window::Int
end

struct ConservativeTreeCylinderChannelResult2D{T}
    coarse_F::Array{T,3}
    patch::ConservativeTreePatch2D{T}
    is_solid_leaf::BitMatrix
    Fx_drag::T
    Fy_drag::T
    Cd::T
    u_in::T
    ux_mean::T
    omega::T
    mass_initial::T
    mass_final::T
    mass_drift::T
    steps::Int
    avg_window::Int
end

struct ConservativeTreeSolidFlowResult2D{T}
    flow::Symbol
    coarse_F::Array{T,3}
    patch::ConservativeTreePatch2D{T}
    is_solid_leaf::BitMatrix
    ux_mean::T
    uy_mean::T
    mass_initial::T
    mass_final::T
    mass_drift::T
    steps::Int
end

function composite_leaf_mean_ux_profile(coarse_F::AbstractArray{T,3},
                                        patch::ConservativeTreePatch2D{T};
                                        volume_leaf::T=T(0.25),
                                        force_x::T=zero(T)) where T
    leaf = zeros(T, 2 * size(coarse_F, 1), 2 * size(coarse_F, 2), 9)
    composite_to_leaf_F_2d!(leaf, coarse_F, patch)

    profile = zeros(T, size(leaf, 2))
    @inbounds for j in axes(leaf, 2)
        ux_sum = zero(T)
        for i in axes(leaf, 1)
            cell = @view leaf[i, j, :]
            rho = mass_F(cell) / volume_leaf
            ux_sum += (momentum_F(cell)[1] / volume_leaf + force_x / 2) / rho
        end
        profile[j] = ux_sum / T(size(leaf, 1))
    end
    return profile
end

function composite_leaf_velocity_field_2d(coarse_F::AbstractArray{T,3},
                                          patch::ConservativeTreePatch2D{T};
                                          volume_leaf::T=T(0.25),
                                          force_x::T=zero(T),
                                          force_y::T=zero(T)) where T
    leaf = zeros(T, 2 * size(coarse_F, 1), 2 * size(coarse_F, 2), 9)
    composite_to_leaf_F_2d!(leaf, coarse_F, patch)

    ux = zeros(T, size(leaf, 1), size(leaf, 2))
    uy = similar(ux)
    @inbounds for j in axes(leaf, 2), i in axes(leaf, 1)
        cell = @view leaf[i, j, :]
        rho = mass_F(cell) / volume_leaf
        rho > zero(T) || throw(ArgumentError("leaf cell density must be positive"))
        mx, my = momentum_F(cell)
        ux[i, j] = (mx / volume_leaf + force_x / 2) / rho
        uy[i, j] = (my / volume_leaf + force_y / 2) / rho
    end
    return (ux=ux, uy=uy)
end

function couette_analytic_profile_2d(ny::Int, U)
    ny >= 2 || throw(ArgumentError("ny must be >= 2"))
    T = typeof(float(U))
    return [T(U) * T(j - 1) / T(ny - 1) for j in 1:ny]
end

function poiseuille_analytic_profile_2d(ny::Int, Fx, omega; rho=1)
    ny >= 2 || throw(ArgumentError("ny must be >= 2"))
    T = promote_type(typeof(float(Fx)), typeof(float(omega)), typeof(float(rho)))
    nu = (one(T) / T(omega) - T(0.5)) / T(3)
    H = T(ny - 1)
    return [T(Fx) / (T(2) * T(rho) * nu) * T(j - 1) * (H - T(j - 1)) for j in 1:ny]
end

function _profile_errors(profile::AbstractVector, reference::AbstractVector)
    length(profile) == length(reference) ||
        throw(ArgumentError("profile and reference must have the same length"))
    T = promote_type(eltype(profile), eltype(reference))
    l2 = sqrt(sum((T(profile[i]) - T(reference[i]))^2 for i in eachindex(profile)) /
              T(length(profile)))
    linf = maximum(abs(T(profile[i]) - T(reference[i])) for i in eachindex(profile))
    return l2, linf
end

function _leaf_fluid_mass_F(F::AbstractArray{T,3},
                            is_solid::AbstractArray{Bool,2}) where T
    _check_solid_mask_layout(F, is_solid)
    total = zero(T)
    @inbounds for q in 1:9, j in axes(F, 2), i in axes(F, 1)
        is_solid[i, j] && continue
        total += F[i, j, q]
    end
    return total
end

function _leaf_fluid_mean_ux_F(F::AbstractArray{T,3},
                               is_solid::AbstractArray{Bool,2};
                               volume::T,
                               force_x::T=zero(T)) where T
    _check_solid_mask_layout(F, is_solid)
    ux_sum = zero(T)
    n_fluid = 0
    @inbounds for j in axes(F, 2), i in axes(F, 1)
        is_solid[i, j] && continue
        cell = @view F[i, j, :]
        rho = mass_F(cell) / volume
        ux_sum += (momentum_F(cell)[1] / volume + force_x / 2) / rho
        n_fluid += 1
    end
    n_fluid > 0 || throw(ArgumentError("solid mask leaves no fluid cells"))
    return ux_sum / T(n_fluid)
end

function _leaf_fluid_mean_velocity_F(F::AbstractArray{T,3},
                                     is_solid::AbstractArray{Bool,2};
                                     volume::T,
                                     force_x::T=zero(T),
                                     force_y::T=zero(T)) where T
    _check_solid_mask_layout(F, is_solid)
    ux_sum = zero(T)
    uy_sum = zero(T)
    n_fluid = 0
    @inbounds for j in axes(F, 2), i in axes(F, 1)
        is_solid[i, j] && continue
        cell = @view F[i, j, :]
        rho = mass_F(cell) / volume
        p = momentum_F(cell)
        ux_sum += (p[1] / volume + force_x / 2) / rho
        uy_sum += (p[2] / volume + force_y / 2) / rho
        n_fluid += 1
    end
    n_fluid > 0 || throw(ArgumentError("solid mask leaves no fluid cells"))
    return ux_sum / T(n_fluid), uy_sum / T(n_fluid)
end

function run_conservative_tree_couette_macroflow_2d(;
        Nx::Int=16,
        Ny::Int=12,
        patch_i_range::UnitRange{Int}=6:11,
        patch_j_range::UnitRange{Int}=4:9,
        U=0.04,
        omega=1.0,
        rho=1.0,
        steps::Int=3000,
        T::Type{<:Real}=Float64)
    U = T(U)
    omega = T(omega)
    rho = T(rho)
    coarse = zeros(T, Nx, Ny, 9)
    coarse_next = similar(coarse)
    patch = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    volume_coarse = one(T)
    volume_fine = T(0.25)

    fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, rho, zero(T), zero(T))
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, rho, zero(T), zero(T))
    mass_initial = active_mass_F(coarse, patch)

    for _ in 1:steps
        collide_BGK_composite_F_2d!(coarse, patch, volume_coarse, volume_fine, omega, omega)
        stream_composite_periodic_x_moving_wall_y_leaf_F_2d!(
            coarse_next, patch_next, coarse, patch;
            u_north=U, volume_leaf=volume_fine, rho_wall=rho)
        coarse, coarse_next = coarse_next, coarse
        patch, patch_next = patch_next, patch
    end

    profile = composite_leaf_mean_ux_profile(coarse, patch; volume_leaf=volume_fine)
    analytic = couette_analytic_profile_2d(length(profile), U)
    l2, linf = _profile_errors(profile, analytic)
    mass_final = active_mass_F(coarse, patch)

    return ConservativeTreeMacroFlow2D{T}(
        :couette, coarse, patch, profile, analytic, l2, linf,
        mass_initial, mass_final, mass_final - mass_initial, steps)
end

function run_conservative_tree_poiseuille_macroflow_2d(;
        Nx::Int=18,
        Ny::Int=14,
        patch_i_range::UnitRange{Int}=7:12,
        patch_j_range::UnitRange{Int}=5:10,
        Fx=5e-5,
        omega=1.0,
        rho=1.0,
        steps::Int=5000,
        T::Type{<:Real}=Float64)
    Fx = T(Fx)
    omega = T(omega)
    rho = T(rho)
    coarse = zeros(T, Nx, Ny, 9)
    coarse_next = similar(coarse)
    patch = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    volume_coarse = one(T)
    volume_fine = T(0.25)

    fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, rho, zero(T), zero(T))
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, rho, zero(T), zero(T))
    mass_initial = active_mass_F(coarse, patch)

    for _ in 1:steps
        collide_Guo_composite_F_2d!(coarse, patch, volume_coarse, volume_fine,
                                    omega, omega, Fx, zero(T))
        stream_composite_periodic_x_wall_y_leaf_F_2d!(coarse_next, patch_next,
                                                      coarse, patch)
        coarse, coarse_next = coarse_next, coarse
        patch, patch_next = patch_next, patch
    end

    profile = composite_leaf_mean_ux_profile(coarse, patch;
                                             volume_leaf=volume_fine,
                                             force_x=Fx)
    analytic = poiseuille_analytic_profile_2d(length(profile), Fx, omega; rho=rho)
    l2, linf = _profile_errors(profile, analytic)
    mass_final = active_mass_F(coarse, patch)

    return ConservativeTreeMacroFlow2D{T}(
        :poiseuille, coarse, patch, profile, analytic, l2, linf,
        mass_initial, mass_final, mass_final - mass_initial, steps)
end

function _run_conservative_tree_periodic_solid_force_macroflow_2d(
        flow::Symbol,
        is_solid::BitMatrix;
        Nx::Int,
        Ny::Int,
        patch_i_range::UnitRange{Int},
        patch_j_range::UnitRange{Int},
        Fx,
        Fy,
        omega,
        rho,
        steps::Int,
        T::Type{<:Real})
    Fx = T(Fx)
    Fy = T(Fy)
    omega = T(omega)
    rho = T(rho)
    coarse = zeros(T, Nx, Ny, 9)
    coarse_next = similar(coarse)
    patch = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    volume_coarse = one(T)
    volume_fine = T(0.25)
    leaf = zeros(T, 2 * Nx, 2 * Ny, 9)
    leaf_post = similar(leaf)

    fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, rho, zero(T), zero(T))
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, rho, zero(T), zero(T))
    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass_initial = _leaf_fluid_mass_F(leaf, is_solid)

    for _ in 1:steps
        composite_to_leaf_F_2d!(leaf, coarse, patch)
        collide_Guo_integrated_D2Q9!(leaf, is_solid, volume_fine, omega, Fx, Fy)
        stream_periodic_x_wall_y_solid_F_2d!(leaf_post, leaf, is_solid)
        leaf_to_composite_F_2d!(coarse_next, patch_next, leaf_post)
        coarse, coarse_next = coarse_next, coarse
        patch, patch_next = patch_next, patch
    end

    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass_final = _leaf_fluid_mass_F(leaf, is_solid)
    ux_mean, uy_mean = _leaf_fluid_mean_velocity_F(
        leaf, is_solid; volume=volume_fine, force_x=Fx, force_y=Fy)
    return ConservativeTreeSolidFlowResult2D{T}(
        flow, coarse, patch, is_solid, ux_mean, uy_mean,
        mass_initial, mass_final, mass_final - mass_initial, steps)
end

function _run_conservative_tree_open_solid_macroflow_2d(
        flow::Symbol,
        is_solid::BitMatrix;
        Nx::Int,
        Ny::Int,
        patch_i_range::UnitRange{Int},
        patch_j_range::UnitRange{Int},
        u_in,
        omega,
        rho,
        steps::Int,
        T::Type{<:Real})
    u_in = T(u_in)
    omega = T(omega)
    rho = T(rho)
    coarse = zeros(T, Nx, Ny, 9)
    coarse_next = similar(coarse)
    patch = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    volume_coarse = one(T)
    volume_fine = T(0.25)
    leaf = zeros(T, 2 * Nx, 2 * Ny, 9)
    leaf_post = similar(leaf)

    fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, rho, u_in, zero(T))
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, rho, u_in, zero(T))
    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass_initial = _leaf_fluid_mass_F(leaf, is_solid)

    for _ in 1:steps
        composite_to_leaf_F_2d!(leaf, coarse, patch)
        stream_bounceback_xy_solid_F_2d!(leaf_post, leaf, is_solid)
        apply_zou_he_west_F_2d!(leaf_post, u_in, volume_fine, is_solid)
        apply_zou_he_pressure_east_F_2d!(leaf_post, volume_fine, is_solid; rho_out=rho)
        collide_BGK_integrated_D2Q9!(leaf_post, is_solid, volume_fine, omega)
        leaf_to_composite_F_2d!(coarse_next, patch_next, leaf_post)
        coarse, coarse_next = coarse_next, coarse
        patch, patch_next = patch_next, patch
    end

    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass_final = _leaf_fluid_mass_F(leaf, is_solid)
    ux_mean, uy_mean = _leaf_fluid_mean_velocity_F(leaf, is_solid; volume=volume_fine)
    return ConservativeTreeSolidFlowResult2D{T}(
        flow, coarse, patch, is_solid, ux_mean, uy_mean,
        mass_initial, mass_final, mass_final - mass_initial, steps)
end

function run_conservative_tree_square_obstacle_macroflow_2d(;
        Nx::Int=24,
        Ny::Int=14,
        patch_i_range::UnitRange{Int}=8:17,
        patch_j_range::UnitRange{Int}=4:11,
        obstacle_i_range::UnitRange{Int}=22:27,
        obstacle_j_range::UnitRange{Int}=12:17,
        Fx=2e-5,
        Fy=0.0,
        omega=1.0,
        rho=1.0,
        steps::Int=1200,
        T::Type{<:Real}=Float64)
    is_solid = square_solid_mask_leaf_2d(2 * Nx, 2 * Ny,
                                         obstacle_i_range, obstacle_j_range)
    return _run_conservative_tree_periodic_solid_force_macroflow_2d(
        :square_obstacle, is_solid; Nx=Nx, Ny=Ny,
        patch_i_range=patch_i_range, patch_j_range=patch_j_range,
        Fx=Fx, Fy=Fy, omega=omega, rho=rho, steps=steps, T=T)
end

function run_conservative_tree_bfs_macroflow_2d(;
        Nx::Int=28,
        Ny::Int=14,
        patch_i_range::UnitRange{Int}=1:12,
        patch_j_range::UnitRange{Int}=1:8,
        step_i_leaf::Int=16,
        step_height_leaf::Int=8,
        u_in=0.03,
        omega=1.0,
        rho=1.0,
        steps::Int=800,
        T::Type{<:Real}=Float64)
    is_solid = backward_facing_step_solid_mask_leaf_2d(2 * Nx, 2 * Ny,
                                                       step_i_leaf,
                                                       step_height_leaf)
    return _run_conservative_tree_open_solid_macroflow_2d(
        :bfs, is_solid; Nx=Nx, Ny=Ny,
        patch_i_range=patch_i_range, patch_j_range=patch_j_range,
        u_in=u_in, omega=omega, rho=rho, steps=steps, T=T)
end

function run_conservative_tree_cylinder_macroflow_2d(;
        Nx::Int=24,
        Ny::Int=14,
        patch_i_range::UnitRange{Int}=8:17,
        patch_j_range::UnitRange{Int}=4:11,
        cx_leaf=(2 * Nx) / 2,
        cy_leaf=(2 * Ny) / 2,
        radius_leaf=3.0,
        Fx=2e-5,
        omega=1.0,
        rho=1.0,
        steps::Int=1200,
        avg_window::Int=300,
        T::Type{<:Real}=Float64)
    Fx = T(Fx)
    omega = T(omega)
    rho = T(rho)
    radius_leaf = T(radius_leaf)
    cx_leaf = T(cx_leaf)
    cy_leaf = T(cy_leaf)

    coarse = zeros(T, Nx, Ny, 9)
    coarse_next = similar(coarse)
    patch = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    volume_coarse = one(T)
    volume_fine = T(0.25)
    leaf = zeros(T, 2 * Nx, 2 * Ny, 9)
    leaf_post = similar(leaf)
    is_solid = cylinder_solid_mask_leaf_2d(size(leaf, 1), size(leaf, 2),
                                           cx_leaf, cy_leaf, radius_leaf)

    fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, rho, zero(T), zero(T))
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, rho, zero(T), zero(T))
    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass_initial = _leaf_fluid_mass_F(leaf, is_solid)

    Fx_sum = zero(T)
    Fy_sum = zero(T)
    n_avg = 0
    for step in 1:steps
        composite_to_leaf_F_2d!(leaf, coarse, patch)
        collide_Guo_integrated_D2Q9!(leaf, is_solid, volume_fine, omega, Fx, zero(T))
        stream_periodic_x_wall_y_solid_F_2d!(leaf_post, leaf, is_solid)

        if step > steps - avg_window
            drag = compute_drag_mea_solid_F_2d(leaf, leaf_post, is_solid)
            Fx_sum += T(drag.Fx)
            Fy_sum += T(drag.Fy)
            n_avg += 1
        end

        leaf_to_composite_F_2d!(coarse_next, patch_next, leaf_post)
        coarse, coarse_next = coarse_next, coarse
        patch, patch_next = patch_next, patch
    end

    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass_final = _leaf_fluid_mass_F(leaf, is_solid)
    u_ref = _leaf_fluid_mean_ux_F(leaf, is_solid; volume=volume_fine, force_x=Fx)
    Fx_drag = Fx_sum / T(n_avg)
    Fy_drag = Fy_sum / T(n_avg)
    diameter = T(2) * radius_leaf
    Cd = T(2) * Fx_drag / (rho * max(abs(u_ref), eps(T))^2 * diameter)

    return ConservativeTreeCylinderResult2D{T}(
        coarse, patch, is_solid, Fx_drag, Fy_drag, Cd, u_ref,
        mass_initial, mass_final, mass_final - mass_initial, steps, avg_window)
end

function run_conservative_tree_cylinder_channel_macroflow_2d(;
        Nx::Int=24,
        Ny::Int=14,
        patch_i_range::UnitRange{Int}=4:9,
        patch_j_range::UnitRange{Int}=4:11,
        cx_leaf=(2 * Nx) / 4,
        cy_leaf=(2 * Ny) / 2,
        radius_leaf=3.0,
        u_in=0.03,
        Re=nothing,
        omega=1.0,
        rho=1.0,
        steps::Int=2000,
        avg_window::Int=500,
        T::Type{<:Real}=Float64)
    u_in = T(u_in)
    rho = T(rho)
    radius_leaf = T(radius_leaf)
    cx_leaf = T(cx_leaf)
    cy_leaf = T(cy_leaf)
    if Re === nothing
        omega = T(omega)
    else
        nu = u_in * T(2) * radius_leaf / T(Re)
        omega = one(T) / (T(3) * nu + T(0.5))
    end

    coarse = zeros(T, Nx, Ny, 9)
    coarse_next = similar(coarse)
    patch = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    volume_coarse = one(T)
    volume_fine = T(0.25)
    leaf = zeros(T, 2 * Nx, 2 * Ny, 9)
    leaf_post = similar(leaf)
    is_solid = cylinder_solid_mask_leaf_2d(size(leaf, 1), size(leaf, 2),
                                           cx_leaf, cy_leaf, radius_leaf)

    fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, rho, u_in, zero(T))
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, rho, u_in, zero(T))
    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass_initial = _leaf_fluid_mass_F(leaf, is_solid)

    Fx_sum = zero(T)
    Fy_sum = zero(T)
    n_avg = 0
    for step in 1:steps
        composite_to_leaf_F_2d!(leaf, coarse, patch)
        stream_bounceback_xy_solid_F_2d!(leaf_post, leaf, is_solid)
        apply_zou_he_west_F_2d!(leaf_post, u_in, volume_fine, is_solid)
        apply_zou_he_pressure_east_F_2d!(leaf_post, volume_fine, is_solid; rho_out=rho)

        if step > steps - avg_window
            drag = compute_drag_mea_solid_F_2d(leaf, leaf_post, is_solid)
            Fx_sum += T(drag.Fx)
            Fy_sum += T(drag.Fy)
            n_avg += 1
        end

        collide_BGK_integrated_D2Q9!(leaf_post, is_solid, volume_fine, omega)
        leaf_to_composite_F_2d!(coarse_next, patch_next, leaf_post)
        coarse, coarse_next = coarse_next, coarse
        patch, patch_next = patch_next, patch
    end

    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass_final = _leaf_fluid_mass_F(leaf, is_solid)
    ux_mean, _ = _leaf_fluid_mean_velocity_F(leaf, is_solid; volume=volume_fine)
    Fx_drag = Fx_sum / T(n_avg)
    Fy_drag = Fy_sum / T(n_avg)
    diameter = T(2) * radius_leaf
    Cd = T(2) * Fx_drag / (rho * u_in^2 * diameter)

    return ConservativeTreeCylinderChannelResult2D{T}(
        coarse, patch, is_solid, Fx_drag, Fy_drag, Cd, u_in, ux_mean, omega,
        mass_initial, mass_final, mass_final - mass_initial, steps, avg_window)
end
