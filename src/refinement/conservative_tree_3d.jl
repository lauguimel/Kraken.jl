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

"""
    ConservativeTreePatch3D

Experimental fixed ratio-2 patch for the conservative 3D tree route.

`fine_F` is the active fine state inside the refined region. `coarse_shadow_F`
is only a parent ledger/aggregate over the same physical region; it is not an
active fluid state.
"""
struct ConservativeTreePatch3D{T}
    parent_i_range::UnitRange{Int}
    parent_j_range::UnitRange{Int}
    parent_k_range::UnitRange{Int}
    ratio::Int
    fine_F::Array{T,4}
    coarse_shadow_F::Array{T,4}
end

"""
    create_conservative_tree_patch_3d(parent_i_range, parent_j_range, parent_k_range; ratio=2, T=Float64)

Allocate an experimental fixed ratio-2 conservative-tree 3D patch. No ghost
cells are allocated.
"""
function create_conservative_tree_patch_3d(parent_i_range::AbstractUnitRange{<:Integer},
                                           parent_j_range::AbstractUnitRange{<:Integer},
                                           parent_k_range::AbstractUnitRange{<:Integer};
                                           ratio::Int=2,
                                           T::Type{<:Real}=Float64)
    ratio == 2 || throw(ArgumentError("only ratio=2 is implemented"))
    isempty(parent_i_range) && throw(ArgumentError("parent_i_range must be nonempty"))
    isempty(parent_j_range) && throw(ArgumentError("parent_j_range must be nonempty"))
    isempty(parent_k_range) && throw(ArgumentError("parent_k_range must be nonempty"))

    ip = Int(first(parent_i_range)):Int(last(parent_i_range))
    jp = Int(first(parent_j_range)):Int(last(parent_j_range))
    kp = Int(first(parent_k_range)):Int(last(parent_k_range))
    nx_parent = length(ip)
    ny_parent = length(jp)
    nz_parent = length(kp)

    fine_F = zeros(T, ratio * nx_parent, ratio * ny_parent, ratio * nz_parent, 19)
    coarse_shadow_F = zeros(T, nx_parent, ny_parent, nz_parent, 19)
    return ConservativeTreePatch3D{T}(ip, jp, kp, ratio, fine_F, coarse_shadow_F)
end

function _check_conservative_tree_patch_layout(patch::ConservativeTreePatch3D)
    patch.ratio == 2 || throw(ArgumentError("only ratio=2 is implemented"))
    nx_parent = length(patch.parent_i_range)
    ny_parent = length(patch.parent_j_range)
    nz_parent = length(patch.parent_k_range)
    size(patch.fine_F) == (2 * nx_parent, 2 * ny_parent, 2 * nz_parent, 19) ||
        throw(ArgumentError("patch.fine_F has inconsistent size"))
    size(patch.coarse_shadow_F) == (nx_parent, ny_parent, nz_parent, 19) ||
        throw(ArgumentError("patch.coarse_shadow_F has inconsistent size"))
    return nothing
end

function coalesce_patch_to_shadow_F_3d!(patch::ConservativeTreePatch3D)
    _check_conservative_tree_patch_layout(patch)

    nx_parent = length(patch.parent_i_range)
    ny_parent = length(patch.parent_j_range)
    nz_parent = length(patch.parent_k_range)
    @inbounds for kp in 1:nz_parent, jp in 1:ny_parent, ip in 1:nx_parent
        i0 = 2 * ip - 1
        j0 = 2 * jp - 1
        k0 = 2 * kp - 1
        Fp = @view patch.coarse_shadow_F[ip, jp, kp, :]
        Fc = @view patch.fine_F[i0:i0+1, j0:j0+1, k0:k0+1, :]
        coalesce_F_3d!(Fp, Fc)
    end
    return patch
end

function explode_shadow_to_patch_uniform_F_3d!(patch::ConservativeTreePatch3D)
    _check_conservative_tree_patch_layout(patch)

    nx_parent = length(patch.parent_i_range)
    ny_parent = length(patch.parent_j_range)
    nz_parent = length(patch.parent_k_range)
    @inbounds for kp in 1:nz_parent, jp in 1:ny_parent, ip in 1:nx_parent
        i0 = 2 * ip - 1
        j0 = 2 * jp - 1
        k0 = 2 * kp - 1
        Fc = @view patch.fine_F[i0:i0+1, j0:j0+1, k0:k0+1, :]
        Fp = @view patch.coarse_shadow_F[ip, jp, kp, :]
        explode_uniform_F_3d!(Fc, Fp)
    end
    return patch
end

function _check_composite_coarse_layout_3d(coarse_F::AbstractArray{<:Any,4},
                                           patch::ConservativeTreePatch3D)
    _check_conservative_tree_patch_layout(patch)
    size(coarse_F, 4) == 19 ||
        throw(ArgumentError("coarse_F must have 19 D3Q19 populations in dimension 4"))
    first(patch.parent_i_range) >= first(axes(coarse_F, 1)) ||
        throw(ArgumentError("patch.parent_i_range starts outside coarse_F"))
    last(patch.parent_i_range) <= last(axes(coarse_F, 1)) ||
        throw(ArgumentError("patch.parent_i_range ends outside coarse_F"))
    first(patch.parent_j_range) >= first(axes(coarse_F, 2)) ||
        throw(ArgumentError("patch.parent_j_range starts outside coarse_F"))
    last(patch.parent_j_range) <= last(axes(coarse_F, 2)) ||
        throw(ArgumentError("patch.parent_j_range ends outside coarse_F"))
    first(patch.parent_k_range) >= first(axes(coarse_F, 3)) ||
        throw(ArgumentError("patch.parent_k_range starts outside coarse_F"))
    last(patch.parent_k_range) <= last(axes(coarse_F, 3)) ||
        throw(ArgumentError("patch.parent_k_range ends outside coarse_F"))
    return nothing
end

function _check_composite_pair_layout_3d(coarse_out::AbstractArray{<:Any,4},
                                         patch_out::ConservativeTreePatch3D,
                                         coarse_in::AbstractArray{<:Any,4},
                                         patch_in::ConservativeTreePatch3D)
    size(coarse_out) == size(coarse_in) ||
        throw(ArgumentError("coarse_out and coarse_in must have the same size"))
    patch_out.parent_i_range == patch_in.parent_i_range ||
        throw(ArgumentError("patch_out and patch_in must have the same parent_i_range"))
    patch_out.parent_j_range == patch_in.parent_j_range ||
        throw(ArgumentError("patch_out and patch_in must have the same parent_j_range"))
    patch_out.parent_k_range == patch_in.parent_k_range ||
        throw(ArgumentError("patch_out and patch_in must have the same parent_k_range"))
    size(patch_out.fine_F) == size(patch_in.fine_F) ||
        throw(ArgumentError("patch_out and patch_in fine arrays must have the same size"))
    _check_composite_coarse_layout_3d(coarse_in, patch_in)
    _check_composite_coarse_layout_3d(coarse_out, patch_out)
    return nothing
end

function active_mass_F_3d(coarse_F::AbstractArray{<:Any,4},
                          patch::ConservativeTreePatch3D)
    _check_composite_coarse_layout_3d(coarse_F, patch)

    total = zero(coarse_F[begin, begin, begin, 1] +
                 patch.fine_F[begin, begin, begin, 1])
    @inbounds for q in 1:19, k in axes(coarse_F, 3), j in axes(coarse_F, 2), i in axes(coarse_F, 1)
        _inside_range_3d(i, j, k,
                         patch.parent_i_range,
                         patch.parent_j_range,
                         patch.parent_k_range) && continue
        total += coarse_F[i, j, k, q]
    end
    return total + mass_F_3d(patch.fine_F)
end

function active_population_sums_F_3d(coarse_F::AbstractArray{<:Any,4},
                                     patch::ConservativeTreePatch3D)
    _check_composite_coarse_layout_3d(coarse_F, patch)

    totals = zeros(promote_type(eltype(coarse_F), eltype(patch.fine_F)), 19)
    @inbounds for q in 1:19
        for k in axes(coarse_F, 3), j in axes(coarse_F, 2), i in axes(coarse_F, 1)
            _inside_range_3d(i, j, k,
                             patch.parent_i_range,
                             patch.parent_j_range,
                             patch.parent_k_range) && continue
            totals[q] += coarse_F[i, j, k, q]
        end
        totals[q] += sum(@view patch.fine_F[:, :, :, q])
    end
    return totals
end

function active_momentum_F_3d(coarse_F::AbstractArray{<:Any,4},
                              patch::ConservativeTreePatch3D)
    _check_composite_coarse_layout_3d(coarse_F, patch)

    mx = zero(coarse_F[begin, begin, begin, 1] +
              patch.fine_F[begin, begin, begin, 1])
    my = zero(coarse_F[begin, begin, begin, 1] +
              patch.fine_F[begin, begin, begin, 1])
    mz = zero(coarse_F[begin, begin, begin, 1] +
              patch.fine_F[begin, begin, begin, 1])
    @inbounds for q in 1:19
        cx = d3q19_cx(q)
        cy = d3q19_cy(q)
        cz = d3q19_cz(q)
        for k in axes(coarse_F, 3), j in axes(coarse_F, 2), i in axes(coarse_F, 1)
            _inside_range_3d(i, j, k,
                             patch.parent_i_range,
                             patch.parent_j_range,
                             patch.parent_k_range) && continue
            fq = coarse_F[i, j, k, q]
            mx += cx * fq
            my += cy * fq
            mz += cz * fq
        end
    end
    fmx, fmy, fmz = momentum_F_3d(patch.fine_F)
    return mx + fmx, my + fmy, mz + fmz
end

function active_moments_F_3d(coarse_F::AbstractArray{<:Any,4},
                             patch::ConservativeTreePatch3D)
    m = active_mass_F_3d(coarse_F, patch)
    mx, my, mz = active_momentum_F_3d(coarse_F, patch)
    return m, mx, my, mz
end

function collide_BGK_integrated_D3Q19!(Fcell::AbstractVector, volume, omega)
    _check_d3q19_vector(Fcell, "Fcell")
    volume > zero(volume) || throw(ArgumentError("volume must be positive"))

    m = mass_F_3d(Fcell)
    iszero(m) && throw(ArgumentError("Fcell mass must be nonzero"))
    mx, my, mz = momentum_F_3d(Fcell)
    rho = m / volume
    ux = mx / m
    uy = my / m
    uz = mz / m

    @inbounds for q in 1:19
        f = Fcell[q] / volume
        feq = equilibrium(D3Q19(), rho, ux, uy, uz, q)
        Fcell[q] = (f - omega * (f - feq)) * volume
    end
    return Fcell
end

function collide_BGK_integrated_D3Q19!(F::AbstractArray{<:Any,4}, volume, omega)
    size(F, 4) == 19 ||
        throw(ArgumentError("F must have 19 D3Q19 populations in dimension 4"))
    @inbounds for k in axes(F, 3), j in axes(F, 2), i in axes(F, 1)
        collide_BGK_integrated_D3Q19!(@view(F[i, j, k, :]), volume, omega)
    end
    return F
end

function collide_Guo_integrated_D3Q19!(Fcell::AbstractVector,
                                       volume,
                                       omega,
                                       Fx,
                                       Fy,
                                       Fz)
    _check_d3q19_vector(Fcell, "Fcell")
    volume > zero(volume) || throw(ArgumentError("volume must be positive"))

    m = mass_F_3d(Fcell)
    iszero(m) && throw(ArgumentError("Fcell mass must be nonzero"))
    mx, my, mz = momentum_F_3d(Fcell)
    rho = m / volume
    ux = (mx / volume + Fx / 2) / rho
    uy = (my / volume + Fy / 2) / rho
    uz = (mz / volume + Fz / 2) / rho
    guo_pref = 1 - omega / 2

    @inbounds for q in 1:19
        cx = d3q19_cx(q)
        cy = d3q19_cy(q)
        cz = d3q19_cz(q)
        w = weights(D3Q19())[q]
        ci_dot_u = cx * ux + cy * uy + cz * uz
        ci_dot_F = cx * Fx + cy * Fy + cz * Fz
        Sq = w * (3 * ((cx - ux) * Fx + (cy - uy) * Fy + (cz - uz) * Fz) +
                  9 * ci_dot_u * ci_dot_F)
        f = Fcell[q] / volume
        feq = equilibrium(D3Q19(), rho, ux, uy, uz, q)
        Fcell[q] = volume * (f - omega * (f - feq) + guo_pref * Sq)
    end
    return Fcell
end

function collide_Guo_integrated_D3Q19!(F::AbstractArray{<:Any,4},
                                       volume,
                                       omega,
                                       Fx,
                                       Fy,
                                       Fz)
    size(F, 4) == 19 ||
        throw(ArgumentError("F must have 19 D3Q19 populations in dimension 4"))
    @inbounds for k in axes(F, 3), j in axes(F, 2), i in axes(F, 1)
        collide_Guo_integrated_D3Q19!(@view(F[i, j, k, :]),
                                      volume, omega, Fx, Fy, Fz)
    end
    return F
end

function collide_BGK_composite_F_3d!(coarse_F::AbstractArray{<:Any,4},
                                     patch::ConservativeTreePatch3D,
                                     volume_coarse,
                                     volume_fine,
                                     omega_coarse,
                                     omega_fine)
    _check_composite_coarse_layout_3d(coarse_F, patch)

    @inbounds for k in axes(coarse_F, 3), j in axes(coarse_F, 2), i in axes(coarse_F, 1)
        _inside_range_3d(i, j, k,
                         patch.parent_i_range,
                         patch.parent_j_range,
                         patch.parent_k_range) && continue
        collide_BGK_integrated_D3Q19!(
            @view(coarse_F[i, j, k, :]), volume_coarse, omega_coarse)
    end
    collide_BGK_integrated_D3Q19!(patch.fine_F, volume_fine, omega_fine)
    coalesce_patch_to_shadow_F_3d!(patch)
    return coarse_F, patch
end

function collide_Guo_composite_F_3d!(coarse_F::AbstractArray{<:Any,4},
                                     patch::ConservativeTreePatch3D,
                                     volume_coarse,
                                     volume_fine,
                                     omega_coarse,
                                     omega_fine,
                                     Fx,
                                     Fy,
                                     Fz)
    _check_composite_coarse_layout_3d(coarse_F, patch)

    @inbounds for k in axes(coarse_F, 3), j in axes(coarse_F, 2), i in axes(coarse_F, 1)
        _inside_range_3d(i, j, k,
                         patch.parent_i_range,
                         patch.parent_j_range,
                         patch.parent_k_range) && continue
        collide_Guo_integrated_D3Q19!(
            @view(coarse_F[i, j, k, :]), volume_coarse,
            omega_coarse, Fx, Fy, Fz)
    end
    collide_Guo_integrated_D3Q19!(patch.fine_F, volume_fine, omega_fine, Fx, Fy, Fz)
    coalesce_patch_to_shadow_F_3d!(patch)
    return coarse_F, patch
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
