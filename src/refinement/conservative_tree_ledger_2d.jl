
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

function apply_composite_zou_he_west_F_2d!(
        coarse_F::AbstractArray{T,3},
        patch::ConservativeTreePatch2D{T},
        u_in,
        volume_coarse,
        volume_fine) where T
    _check_composite_coarse_layout(coarse_F, patch)
    @inbounds for J in axes(coarse_F, 2)
        if first(patch.parent_i_range) <= 1 <= last(patch.parent_i_range) &&
                J in patch.parent_j_range
            il, jl = _patch_local_parent_index(patch, 1, J)
            for jf in (2 * jl - 1):(2 * jl)
                apply_zou_he_west_cell_F_2d!(
                    @view(patch.fine_F[2 * il - 1, jf, :]), u_in, volume_fine)
            end
        else
            apply_zou_he_west_cell_F_2d!(
                @view(coarse_F[1, J, :]), u_in, volume_coarse)
        end
    end
    coalesce_patch_to_shadow_F_2d!(patch)
    return coarse_F, patch
end

function apply_composite_zou_he_pressure_east_F_2d!(
        coarse_F::AbstractArray{T,3},
        patch::ConservativeTreePatch2D{T},
        volume_coarse,
        volume_fine;
        rho_out=one(T)) where T
    _check_composite_coarse_layout(coarse_F, patch)
    I = last(axes(coarse_F, 1))
    @inbounds for J in axes(coarse_F, 2)
        if first(patch.parent_i_range) <= I <= last(patch.parent_i_range) &&
                J in patch.parent_j_range
            il, jl = _patch_local_parent_index(patch, I, J)
            for jf in (2 * jl - 1):(2 * jl)
                apply_zou_he_pressure_east_cell_F_2d!(
                    @view(patch.fine_F[2 * il, jf, :]), volume_fine;
                    rho_out=rho_out)
            end
        else
            apply_zou_he_pressure_east_cell_F_2d!(
                @view(coarse_F[I, J, :]), volume_coarse; rho_out=rho_out)
        end
    end
    coalesce_patch_to_shadow_F_2d!(patch)
    return coarse_F, patch
end

function _check_composite_solid_mask_layout(coarse_F::AbstractArray{<:Any,3},
                                            patch::ConservativeTreePatch2D,
                                            is_solid::AbstractArray{Bool,2})
    size(is_solid) == (2 * size(coarse_F, 1), 2 * size(coarse_F, 2)) ||
        throw(ArgumentError("is_solid must have size (2*Nx, 2*Ny)"))
    @inbounds for J in axes(coarse_F, 2), I in axes(coarse_F, 1)
        _inside_range(I, J, patch.parent_i_range, patch.parent_j_range) && continue
        i0 = 2 * I - 1
        j0 = 2 * J - 1
        s11 = is_solid[i0, j0]
        s21 = is_solid[i0 + 1, j0]
        s12 = is_solid[i0, j0 + 1]
        s22 = is_solid[i0 + 1, j0 + 1]
        (s11 == s21 == s12 == s22) ||
            throw(ArgumentError("active coarse cells cannot be partially solid"))
    end
    return nothing
end

function apply_composite_zou_he_west_F_2d!(
        coarse_F::AbstractArray{T,3},
        patch::ConservativeTreePatch2D{T},
        is_solid::AbstractArray{Bool,2},
        u_in,
        volume_coarse,
        volume_fine) where T
    _check_composite_coarse_layout(coarse_F, patch)
    _check_composite_solid_mask_layout(coarse_F, patch, is_solid)
    @inbounds for J in axes(coarse_F, 2)
        if first(patch.parent_i_range) <= 1 <= last(patch.parent_i_range) &&
                J in patch.parent_j_range
            il, jl = _patch_local_parent_index(patch, 1, J)
            i_leaf = 2 * first(patch.parent_i_range) - 1
            for jf in (2 * jl - 1):(2 * jl)
                is_solid[i_leaf, jf] && continue
                apply_zou_he_west_cell_F_2d!(
                    @view(patch.fine_F[2 * il - 1, jf, :]), u_in, volume_fine)
            end
        else
            is_solid[1, 2 * J - 1] && continue
            apply_zou_he_west_cell_F_2d!(
                @view(coarse_F[1, J, :]), u_in, volume_coarse)
        end
    end
    coalesce_patch_to_shadow_F_2d!(patch)
    return coarse_F, patch
end

function apply_composite_zou_he_pressure_east_F_2d!(
        coarse_F::AbstractArray{T,3},
        patch::ConservativeTreePatch2D{T},
        is_solid::AbstractArray{Bool,2},
        volume_coarse,
        volume_fine;
        rho_out=one(T)) where T
    _check_composite_coarse_layout(coarse_F, patch)
    _check_composite_solid_mask_layout(coarse_F, patch, is_solid)
    I = last(axes(coarse_F, 1))
    @inbounds for J in axes(coarse_F, 2)
        if first(patch.parent_i_range) <= I <= last(patch.parent_i_range) &&
                J in patch.parent_j_range
            il, jl = _patch_local_parent_index(patch, I, J)
            i_leaf = 2 * last(patch.parent_i_range)
            for jf in (2 * jl - 1):(2 * jl)
                is_solid[i_leaf, jf] && continue
                apply_zou_he_pressure_east_cell_F_2d!(
                    @view(patch.fine_F[2 * il, jf, :]), volume_fine;
                    rho_out=rho_out)
            end
        else
            is_solid[2 * I - 1, 2 * J - 1] && continue
            apply_zou_he_pressure_east_cell_F_2d!(
                @view(coarse_F[I, J, :]), volume_coarse; rho_out=rho_out)
        end
    end
    coalesce_patch_to_shadow_F_2d!(patch)
    return coarse_F, patch
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
