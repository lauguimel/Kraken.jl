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
    T = typeof(float(Fcell[1]))
    volume_T = T(volume)
    rho_T = T(rho)
    ux_T = T(ux)
    uy_T = T(uy)
    @inbounds for q in 1:9
        Fcell[q] = volume_T * equilibrium(D2Q9(), rho_T, ux_T, uy_T, q)
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
