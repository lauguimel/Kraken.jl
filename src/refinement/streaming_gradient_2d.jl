function conservative_tree_solid_mask_patch_range_2d(
        is_solid::AbstractArray{Bool,2};
        pad::Int=1)
    nx_leaf = size(is_solid, 1)
    ny_leaf = size(is_solid, 2)
    iseven(nx_leaf) && iseven(ny_leaf) ||
        throw(ArgumentError("is_solid dimensions must be even leaf-grid sizes"))
    pad >= 0 || throw(ArgumentError("pad must be nonnegative"))
    any(is_solid) || throw(ArgumentError("is_solid contains no solid cells"))

    i_min_leaf = typemax(Int)
    i_max_leaf = typemin(Int)
    j_min_leaf = typemax(Int)
    j_max_leaf = typemin(Int)
    @inbounds for j in axes(is_solid, 2), i in axes(is_solid, 1)
        if is_solid[i, j]
            i_min_leaf = min(i_min_leaf, i)
            i_max_leaf = max(i_max_leaf, i)
            j_min_leaf = min(j_min_leaf, j)
            j_max_leaf = max(j_max_leaf, j)
        end
    end

    nx = nx_leaf >>> 1
    ny = ny_leaf >>> 1
    i_min = max(1, cld(i_min_leaf, 2) - pad)
    i_max = min(nx, cld(i_max_leaf, 2) + pad)
    j_min = max(1, cld(j_min_leaf, 2) - pad)
    j_max = min(ny, cld(j_max_leaf, 2) + pad)
    return (i_range=i_min:i_max, j_range=j_min:j_max)
end

function conservative_tree_indicator_patch_range_2d(
        indicator::AbstractArray{<:Real,2};
        threshold::Real,
        pad::Int=1)
    nx = size(indicator, 1)
    ny = size(indicator, 2)
    nx > 0 && ny > 0 || throw(ArgumentError("indicator must be nonempty"))
    isfinite(threshold) || throw(ArgumentError("threshold must be finite"))
    threshold >= 0 || throw(ArgumentError("threshold must be nonnegative"))
    pad >= 0 || throw(ArgumentError("pad must be nonnegative"))

    i_min_hit = typemax(Int)
    i_max_hit = typemin(Int)
    j_min_hit = typemax(Int)
    j_max_hit = typemin(Int)
    hit = false
    @inbounds for j in axes(indicator, 2), i in axes(indicator, 1)
        value = indicator[i, j]
        isfinite(value) || throw(ArgumentError("indicator contains non-finite values"))
        if abs(value) > threshold
            hit = true
            i_min_hit = min(i_min_hit, i)
            i_max_hit = max(i_max_hit, i)
            j_min_hit = min(j_min_hit, j)
            j_max_hit = max(j_max_hit, j)
        end
    end
    hit || throw(ArgumentError("indicator has no cells above threshold"))

    i_min = max(1, i_min_hit - pad)
    i_max = min(nx, i_max_hit + pad)
    j_min = max(1, j_min_hit - pad)
    j_max = min(ny, j_max_hit + pad)
    return (i_range=i_min:i_max, j_range=j_min:j_max)
end

function conservative_tree_gradient_indicator_2d(field::AbstractArray{<:Real,2})
    nx = size(field, 1)
    ny = size(field, 2)
    nx > 0 && ny > 0 || throw(ArgumentError("field must be nonempty"))
    T = promote_type(Float64, eltype(field))
    indicator = zeros(T, nx, ny)

    @inbounds for j in axes(field, 2), i in axes(field, 1)
        isfinite(field[i, j]) || throw(ArgumentError("field contains non-finite values"))
        if nx == 1
            dx = zero(T)
        elseif i == first(axes(field, 1))
            dx = T(field[i + 1, j] - field[i, j])
        elseif i == last(axes(field, 1))
            dx = T(field[i, j] - field[i - 1, j])
        else
            dx = T(field[i + 1, j] - field[i - 1, j]) / T(2)
        end

        if ny == 1
            dy = zero(T)
        elseif j == first(axes(field, 2))
            dy = T(field[i, j + 1] - field[i, j])
        elseif j == last(axes(field, 2))
            dy = T(field[i, j] - field[i, j - 1])
        else
            dy = T(field[i, j + 1] - field[i, j - 1]) / T(2)
        end
        indicator[i, j] = sqrt(dx * dx + dy * dy)
    end
    return indicator
end

function conservative_tree_hysteresis_patch_range_2d(
        current_i_range::AbstractUnitRange{<:Integer},
        current_j_range::AbstractUnitRange{<:Integer},
        target_i_range::AbstractUnitRange{<:Integer},
        target_j_range::AbstractUnitRange{<:Integer};
        shrink_margin::Int=1)
    isempty(current_i_range) && throw(ArgumentError("current_i_range must be nonempty"))
    isempty(current_j_range) && throw(ArgumentError("current_j_range must be nonempty"))
    isempty(target_i_range) && throw(ArgumentError("target_i_range must be nonempty"))
    isempty(target_j_range) && throw(ArgumentError("target_j_range must be nonempty"))
    shrink_margin >= 0 || throw(ArgumentError("shrink_margin must be nonnegative"))

    current_i = Int(first(current_i_range)):Int(last(current_i_range))
    current_j = Int(first(current_j_range)):Int(last(current_j_range))
    target_i = Int(first(target_i_range)):Int(last(target_i_range))
    target_j = Int(first(target_j_range)):Int(last(target_j_range))

    grows = first(target_i) < first(current_i) ||
            last(target_i) > last(current_i) ||
            first(target_j) < first(current_j) ||
            last(target_j) > last(current_j)
    if grows || shrink_margin == 0
        return (i_range=target_i, j_range=target_j)
    end

    can_shrink_i = first(target_i) >= first(current_i) + shrink_margin &&
                   last(target_i) <= last(current_i) - shrink_margin
    can_shrink_j = first(target_j) >= first(current_j) + shrink_margin &&
                   last(target_j) <= last(current_j) - shrink_margin
    if can_shrink_i && can_shrink_j
        return (i_range=target_i, j_range=target_j)
    end
    return (i_range=current_i, j_range=current_j)
end

function conservative_tree_velocity_gradient_patch_range_2d(
        coarse_F::AbstractArray{T,3},
        patch::ConservativeTreePatch2D{T};
        threshold::Real,
        volume_leaf::T=T(0.25),
        force_x::T=zero(T),
        force_y::T=zero(T),
        pad_leaf::Int=0,
        pad_parent::Int=0,
        shrink_margin::Int=1) where T
    _check_composite_coarse_layout(coarse_F, patch)
    pad_leaf >= 0 || throw(ArgumentError("pad_leaf must be nonnegative"))
    pad_parent >= 0 || throw(ArgumentError("pad_parent must be nonnegative"))

    velocity = composite_leaf_velocity_field_2d(
        coarse_F, patch; volume_leaf=volume_leaf, force_x=force_x, force_y=force_y)
    indicator = conservative_tree_gradient_indicator_2d(velocity.ux)
    leaf_ranges = conservative_tree_indicator_patch_range_2d(
        indicator; threshold=threshold, pad=pad_leaf)

    nx = size(coarse_F, 1)
    ny = size(coarse_F, 2)
    i_min = max(1, cld(first(leaf_ranges.i_range), 2) - pad_parent)
    i_max = min(nx, cld(last(leaf_ranges.i_range), 2) + pad_parent)
    j_min = max(1, cld(first(leaf_ranges.j_range), 2) - pad_parent)
    j_max = min(ny, cld(last(leaf_ranges.j_range), 2) + pad_parent)
    return conservative_tree_hysteresis_patch_range_2d(
        patch.parent_i_range, patch.parent_j_range, i_min:i_max, j_min:j_max;
        shrink_margin=shrink_margin)
end

function adapt_conservative_tree_patch_to_velocity_gradient_2d(
        coarse_F::AbstractArray{T,3},
        patch::ConservativeTreePatch2D{T};
        threshold::Real,
        volume_leaf::T=T(0.25),
        force_x::T=zero(T),
        force_y::T=zero(T),
        pad_leaf::Int=0,
        pad_parent::Int=0,
        shrink_margin::Int=1) where T
    ranges = conservative_tree_velocity_gradient_patch_range_2d(
        coarse_F, patch; threshold=threshold, volume_leaf=volume_leaf,
        force_x=force_x, force_y=force_y, pad_leaf=pad_leaf,
        pad_parent=pad_parent, shrink_margin=shrink_margin)
    patch_out = create_conservative_tree_patch_2d(
        ranges.i_range, ranges.j_range; T=T)
    coarse_out = similar(coarse_F)
    regrid_conservative_tree_patch_direct_F_2d!(coarse_out, patch_out, coarse_F, patch)
    return (coarse_F=coarse_out, patch=patch_out)
end

function adapt_conservative_tree_patch_to_solid_mask_2d(
        coarse_F::AbstractArray{T,3},
        patch::ConservativeTreePatch2D{T},
        is_solid::AbstractArray{Bool,2};
        pad::Int=1) where T
    _check_composite_coarse_layout(coarse_F, patch)
    size(is_solid) == (2 * size(coarse_F, 1), 2 * size(coarse_F, 2)) ||
        throw(ArgumentError("is_solid must have size (2*Nx, 2*Ny)"))
    ranges = conservative_tree_solid_mask_patch_range_2d(is_solid; pad=pad)
    patch_out = create_conservative_tree_patch_2d(
        ranges.i_range, ranges.j_range; T=T)
    coarse_out = similar(coarse_F)
    regrid_conservative_tree_patch_direct_F_2d!(coarse_out, patch_out, coarse_F, patch)
    return (coarse_F=coarse_out, patch=patch_out)
end

function vertical_facing_step_solid_mask_leaf_2d(
        Nx::Int,
        Ny::Int,
        step_i_range::AbstractUnitRange{<:Integer},
        step_height::Int)
    Nx > 0 || throw(ArgumentError("Nx must be positive"))
    Ny > 0 || throw(ArgumentError("Ny must be positive"))
    isempty(step_i_range) && throw(ArgumentError("step_i_range must be nonempty"))
    first(step_i_range) >= 1 && last(step_i_range) <= Nx ||
        throw(ArgumentError("step_i_range must be inside 1:Nx"))
    1 <= step_height < Ny ||
        throw(ArgumentError("step_height must be inside 1:Ny-1"))

    mask = falses(Nx, Ny)
    @inbounds for j in 1:step_height, i in Int(first(step_i_range)):Int(last(step_i_range))
        mask[i, j] = true
    end
    return mask
end

