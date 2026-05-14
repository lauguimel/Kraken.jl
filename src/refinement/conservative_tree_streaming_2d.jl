include("streaming_packets_2d.jl")
include("streaming_routes_2d.jl")
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

struct ConservativeTreeAdaptiveRun2D{T}
    flow::Symbol
    coarse_F::Array{T,3}
    patch::ConservativeTreePatch2D{T}
    patch_history::Vector{Tuple{UnitRange{Int},UnitRange{Int}}}
    ux_mean::T
    mass_initial::T
    mass_final::T
    mass_drift::T
    steps::Int
    regrid_every::Int
    regrid_count::Int
end

struct ConservativeTreeSolidAdaptiveRun2D{T}
    flow::Symbol
    coarse_F::Array{T,3}
    patch::ConservativeTreePatch2D{T}
    is_solid_leaf::BitMatrix
    patch_history::Vector{Tuple{UnitRange{Int},UnitRange{Int}}}
    ux_mean::T
    uy_mean::T
    mass_initial::T
    mass_final::T
    mass_drift::T
    steps::Int
    regrid_every::Int
    regrid_count::Int
end

struct ConservativeTreeOpenChannelRun2D{T}
    flow::Symbol
    coarse_F::Array{T,3}
    patch::ConservativeTreePatch2D{T}
    ux_mean::T
    mass_initial::T
    mass_final::T
    mass_drift::T
    steps::Int
end

struct ConservativeTreeOpenChannelMassLedger2D{T}
    flow::Symbol
    coarse_F::Array{T,3}
    patch::ConservativeTreePatch2D{T}
    mass_history::Vector{T}
    ux_history::Vector{T}
    mass_initial::T
    mass_final::T
    mass_drift::T
    steps::Int
end

function _composite_open_channel_mean_ux_2d(coarse_F::AbstractArray{T,3},
                                            patch::ConservativeTreePatch2D{T},
                                            volume_fine::T) where T
    profile = composite_leaf_mean_ux_profile(coarse_F, patch;
                                             volume_leaf=volume_fine)
    return sum(profile) / T(length(profile))
end

function run_conservative_tree_couette_route_native_2d(;
        Nx::Int=18,
        Ny::Int=14,
        patch_i_range::UnitRange{Int}=7:12,
        patch_j_range::UnitRange{Int}=5:10,
        U=0.04,
        omega=1.0,
        rho=1.0,
        steps::Int=3000,
        coarse_route_mode::Symbol=:leaf_equivalent,
        T::Type{<:Real}=Float64)
    U = T(U)
    omega = T(omega)
    rho = T(rho)
    volume_coarse = one(T)
    volume_fine = T(0.25)

    coarse = zeros(T, Nx, Ny, 9)
    coarse_next = similar(coarse)
    patch = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    topology = create_conservative_tree_topology_2d(
        Nx, Ny, patch; coarse_route_mode=coarse_route_mode)

    fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, rho, zero(T), zero(T))
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, rho, zero(T), zero(T))
    mass_initial = active_mass_F(coarse, patch)

    for _ in 1:steps
        collide_BGK_composite_F_2d!(coarse, patch, volume_coarse, volume_fine,
                                    omega, omega)
        stream_composite_routes_periodic_x_moving_wall_y_F_2d!(
            coarse_next, patch_next, coarse, patch, topology;
            u_north=U, rho_wall=rho,
            volume_coarse=volume_coarse, volume_fine=volume_fine,
            coarse_prolongation=:limited_linear)
        coarse, coarse_next = coarse_next, coarse
        patch, patch_next = patch_next, patch
    end

    profile = composite_leaf_mean_ux_profile(coarse, patch; volume_leaf=volume_fine)
    analytic = couette_analytic_profile_2d(length(profile), U)
    l2, linf = _profile_errors(profile, analytic)
    mass_final = active_mass_F(coarse, patch)

    return ConservativeTreeMacroFlow2D{T}(
        :couette_route_native, coarse, patch, profile, analytic, l2, linf,
        mass_initial, mass_final, mass_final - mass_initial, steps)
end

function run_conservative_tree_poiseuille_route_native_2d(;
        Nx::Int=18,
        Ny::Int=14,
        patch_i_range::UnitRange{Int}=7:12,
        patch_j_range::UnitRange{Int}=5:10,
        Fx=5e-5,
        omega=1.0,
        rho=1.0,
        steps::Int=3000,
        coarse_route_mode::Symbol=:leaf_equivalent,
        T::Type{<:Real}=Float64)
    Fx = T(Fx)
    omega = T(omega)
    rho = T(rho)
    volume_coarse = one(T)
    volume_fine = T(0.25)

    coarse = zeros(T, Nx, Ny, 9)
    coarse_next = similar(coarse)
    patch = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    topology = create_conservative_tree_topology_2d(
        Nx, Ny, patch; coarse_route_mode=coarse_route_mode)

    fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, rho, zero(T), zero(T))
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, rho, zero(T), zero(T))
    mass_initial = active_mass_F(coarse, patch)

    for _ in 1:steps
        collide_Guo_composite_F_2d!(coarse, patch, volume_coarse, volume_fine,
                                    omega, omega, Fx, zero(T))
        stream_composite_routes_periodic_x_wall_y_F_2d!(
            coarse_next, patch_next, coarse, patch, topology;
            coarse_prolongation=:limited_linear)
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
        :poiseuille_route_native, coarse, patch, profile, analytic, l2, linf,
        mass_initial, mass_final, mass_final - mass_initial, steps)
end

function run_conservative_tree_open_channel_route_native_2d(;
        Nx::Int=18,
        Ny::Int=10,
        patch_i_range::UnitRange{Int}=7:12,
        patch_j_range::UnitRange{Int}=3:8,
        u_in=0.01,
        rho_out=1.0,
        omega=1.0,
        rho=1.0,
        steps::Int=160,
        coarse_route_mode::Symbol=:leaf_equivalent,
        T::Type{<:Real}=Float64)
    steps > 0 || throw(ArgumentError("steps must be positive"))
    u_in = T(u_in)
    rho_out = T(rho_out)
    omega = T(omega)
    rho = T(rho)
    volume_coarse = one(T)
    volume_fine = T(0.25)

    coarse = zeros(T, Nx, Ny, 9)
    coarse_next = similar(coarse)
    patch = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    topology = create_conservative_tree_topology_2d(
        Nx, Ny, patch; coarse_route_mode=coarse_route_mode)

    fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, rho, u_in, zero(T))
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, rho, u_in, zero(T))
    apply_composite_zou_he_west_F_2d!(
        coarse, patch, u_in, volume_coarse, volume_fine)
    apply_composite_zou_he_pressure_east_F_2d!(
        coarse, patch, volume_coarse, volume_fine; rho_out=rho_out)
    mass_initial = active_mass_F(coarse, patch)

    for _ in 1:steps
        collide_BGK_composite_F_2d!(coarse, patch, volume_coarse, volume_fine,
                                    omega, omega)
        stream_composite_routes_zou_he_x_wall_y_F_2d!(
            coarse_next, patch_next, coarse, patch, topology;
            u_in=u_in, rho_out=rho_out,
            volume_coarse=volume_coarse, volume_fine=volume_fine)
        coarse, coarse_next = coarse_next, coarse
        patch, patch_next = patch_next, patch
    end

    mass_final = active_mass_F(coarse, patch)
    ux_mean = sum(composite_leaf_mean_ux_profile(coarse, patch;
                                                 volume_leaf=volume_fine)) /
              T(2 * Ny)

    return ConservativeTreeOpenChannelRun2D{T}(
        :open_channel_route_native, coarse, patch, ux_mean,
        mass_initial, mass_final, mass_final - mass_initial, steps)
end

function run_conservative_tree_open_channel_mass_ledger_2d(;
        Nx::Int=18,
        Ny::Int=10,
        patch_i_range::UnitRange{Int}=7:12,
        patch_j_range::UnitRange{Int}=3:8,
        u_in=0.01,
        rho_out=1.0,
        omega=1.0,
        rho=1.0,
        steps::Int=160,
        coarse_route_mode::Symbol=:leaf_equivalent,
        T::Type{<:Real}=Float64)
    steps > 0 || throw(ArgumentError("steps must be positive"))
    u_in = T(u_in)
    rho_out = T(rho_out)
    omega = T(omega)
    rho = T(rho)
    volume_coarse = one(T)
    volume_fine = T(0.25)

    coarse = zeros(T, Nx, Ny, 9)
    coarse_next = similar(coarse)
    patch = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    topology = create_conservative_tree_topology_2d(
        Nx, Ny, patch; coarse_route_mode=coarse_route_mode)

    fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, rho, u_in, zero(T))
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, rho, u_in, zero(T))
    apply_composite_zou_he_west_F_2d!(
        coarse, patch, u_in, volume_coarse, volume_fine)
    apply_composite_zou_he_pressure_east_F_2d!(
        coarse, patch, volume_coarse, volume_fine; rho_out=rho_out)

    mass_history = Vector{T}(undef, steps + 1)
    ux_history = Vector{T}(undef, steps + 1)
    mass_history[1] = active_mass_F(coarse, patch)
    ux_history[1] = _composite_open_channel_mean_ux_2d(coarse, patch, volume_fine)

    for step in 1:steps
        collide_BGK_composite_F_2d!(coarse, patch, volume_coarse, volume_fine,
                                    omega, omega)
        stream_composite_routes_zou_he_x_wall_y_F_2d!(
            coarse_next, patch_next, coarse, patch, topology;
            u_in=u_in, rho_out=rho_out,
            volume_coarse=volume_coarse, volume_fine=volume_fine)
        coarse, coarse_next = coarse_next, coarse
        patch, patch_next = patch_next, patch

        mass_history[step + 1] = active_mass_F(coarse, patch)
        ux_history[step + 1] = _composite_open_channel_mean_ux_2d(
            coarse, patch, volume_fine)
    end

    mass_initial = mass_history[1]
    mass_final = mass_history[end]
    return ConservativeTreeOpenChannelMassLedger2D{T}(
        :open_channel_mass_ledger_route_native, coarse, patch,
        mass_history, ux_history, mass_initial, mass_final,
        mass_final - mass_initial, steps)
end

function run_conservative_tree_bfs_route_native_2d(;
        Nx::Int=28,
        Ny::Int=14,
        patch_i_range::UnitRange{Int}=1:12,
        patch_j_range::UnitRange{Int}=1:8,
        step_i_leaf::Int=16,
        step_height_leaf::Int=8,
        u_in=0.03,
        rho_out=1.0,
        omega=1.0,
        rho=1.0,
        steps::Int=240,
        coarse_route_mode::Symbol=:leaf_equivalent,
        T::Type{<:Real}=Float64)
    steps > 0 || throw(ArgumentError("steps must be positive"))
    u_in = T(u_in)
    rho_out = T(rho_out)
    omega = T(omega)
    rho = T(rho)
    volume_coarse = one(T)
    volume_fine = T(0.25)

    is_solid = backward_facing_step_solid_mask_leaf_2d(
        2 * Nx, 2 * Ny, step_i_leaf, step_height_leaf)
    coarse = zeros(T, Nx, Ny, 9)
    coarse_next = similar(coarse)
    patch = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    topology = create_conservative_tree_topology_2d(
        Nx, Ny, patch; coarse_route_mode=coarse_route_mode)
    leaf = zeros(T, 2 * Nx, 2 * Ny, 9)

    _check_route_solid_mask_layout(topology, coarse, patch, is_solid)

    fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, rho, u_in, zero(T))
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, rho, u_in, zero(T))
    apply_composite_zou_he_west_F_2d!(
        coarse, patch, is_solid, u_in, volume_coarse, volume_fine)
    apply_composite_zou_he_pressure_east_F_2d!(
        coarse, patch, is_solid, volume_coarse, volume_fine; rho_out=rho_out)
    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass_initial = _leaf_fluid_mass_F(leaf, is_solid)

    for _ in 1:steps
        collide_Guo_composite_solid_F_2d!(
            coarse, patch, topology, is_solid, volume_coarse, volume_fine,
            omega, omega, zero(T), zero(T))
        stream_composite_routes_zou_he_x_wall_y_solid_F_2d!(
            coarse_next, patch_next, coarse, patch, topology, is_solid;
            u_in=u_in, rho_out=rho_out,
            volume_coarse=volume_coarse, volume_fine=volume_fine,
            coarse_prolongation=:limited_linear)
        coarse, coarse_next = coarse_next, coarse
        patch, patch_next = patch_next, patch
    end

    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass_final = _leaf_fluid_mass_F(leaf, is_solid)
    ux_mean, uy_mean = _leaf_fluid_mean_velocity_F(leaf, is_solid; volume=volume_fine)
    return ConservativeTreeSolidFlowResult2D{T}(
        :bfs_route_native, coarse, patch, is_solid, ux_mean, uy_mean,
        mass_initial, mass_final, mass_final - mass_initial, steps)
end

function run_conservative_tree_poiseuille_adaptive_route_native_2d(;
        Nx::Int=18,
        Ny::Int=14,
        patch_schedule::Tuple=((7:12, 5:10), (6:11, 4:9), (8:13, 5:10)),
        regrid_every::Int=80,
        Fx=5e-5,
        omega=1.0,
        rho=1.0,
        steps::Int=320,
        coarse_route_mode::Symbol=:leaf_equivalent,
        T::Type{<:Real}=Float64)
    steps > 0 || throw(ArgumentError("steps must be positive"))
    regrid_every > 0 || throw(ArgumentError("regrid_every must be positive"))
    isempty(patch_schedule) && throw(ArgumentError("patch_schedule must be nonempty"))

    Fx = T(Fx)
    omega = T(omega)
    rho = T(rho)
    volume_coarse = one(T)
    volume_fine = T(0.25)

    first_ranges = patch_schedule[1]
    patch = create_conservative_tree_patch_2d(first_ranges[1], first_ranges[2]; T=T)
    patch_next = create_conservative_tree_patch_2d(first_ranges[1], first_ranges[2]; T=T)
    topology = create_conservative_tree_topology_2d(
        Nx, Ny, patch; coarse_route_mode=coarse_route_mode)
    coarse = zeros(T, Nx, Ny, 9)
    coarse_next = similar(coarse)
    fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, rho, zero(T), zero(T))
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, rho, zero(T), zero(T))
    mass_initial = active_mass_F(coarse, patch)

    patch_history = Tuple{UnitRange{Int},UnitRange{Int}}[
        (patch.parent_i_range, patch.parent_j_range)
    ]
    regrid_count = 0

    for step in 1:steps
        collide_Guo_composite_F_2d!(coarse, patch, volume_coarse, volume_fine,
                                    omega, omega, Fx, zero(T))
        stream_composite_routes_periodic_x_wall_y_F_2d!(
            coarse_next, patch_next, coarse, patch, topology;
            coarse_prolongation=:limited_linear)
        coarse, coarse_next = coarse_next, coarse
        patch, patch_next = patch_next, patch

        if step < steps && step % regrid_every == 0
            regrid_count += 1
            ranges = patch_schedule[mod1(regrid_count + 1, length(patch_schedule))]
            new_patch = create_conservative_tree_patch_2d(ranges[1], ranges[2]; T=T)
            new_coarse = similar(coarse)
            regrid_conservative_tree_patch_direct_F_2d!(
                new_coarse, new_patch, coarse, patch)
            coarse = new_coarse
            patch = new_patch
            patch_next = create_conservative_tree_patch_2d(ranges[1], ranges[2]; T=T)
            coarse_next = similar(coarse)
            topology = create_conservative_tree_topology_2d(
                Nx, Ny, patch; coarse_route_mode=coarse_route_mode)
            push!(patch_history, (patch.parent_i_range, patch.parent_j_range))
        end
    end

    mass_final = active_mass_F(coarse, patch)
    ux_mean = sum(composite_leaf_mean_ux_profile(coarse, patch;
                                                 volume_leaf=volume_fine,
                                                 force_x=Fx)) / T(2 * Ny)

    return ConservativeTreeAdaptiveRun2D{T}(
        :poiseuille_adaptive_route_native, coarse, patch, patch_history,
        ux_mean, mass_initial, mass_final, mass_final - mass_initial,
        steps, regrid_every, regrid_count)
end

function run_conservative_tree_poiseuille_gradient_adaptive_route_native_2d(;
        Nx::Int=18,
        Ny::Int=14,
        patch_i_range::UnitRange{Int}=7:12,
        patch_j_range::UnitRange{Int}=5:10,
        regrid_every::Int=80,
        gradient_threshold=5e-4,
        pad_leaf::Int=1,
        pad_parent::Int=1,
        shrink_margin::Int=1,
        Fx=5e-5,
        omega=1.0,
        rho=1.0,
        steps::Int=320,
        coarse_route_mode::Symbol=:leaf_equivalent,
        T::Type{<:Real}=Float64)
    steps > 0 || throw(ArgumentError("steps must be positive"))
    regrid_every > 0 || throw(ArgumentError("regrid_every must be positive"))

    Fx = T(Fx)
    omega = T(omega)
    rho = T(rho)
    gradient_threshold = T(gradient_threshold)
    volume_coarse = one(T)
    volume_fine = T(0.25)

    patch = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    topology = create_conservative_tree_topology_2d(
        Nx, Ny, patch; coarse_route_mode=coarse_route_mode)
    coarse = zeros(T, Nx, Ny, 9)
    coarse_next = similar(coarse)
    fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, rho, zero(T), zero(T))
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, rho, zero(T), zero(T))
    mass_initial = active_mass_F(coarse, patch)

    patch_history = Tuple{UnitRange{Int},UnitRange{Int}}[
        (patch.parent_i_range, patch.parent_j_range)
    ]
    regrid_count = 0

    for step in 1:steps
        collide_Guo_composite_F_2d!(coarse, patch, volume_coarse, volume_fine,
                                    omega, omega, Fx, zero(T))
        stream_composite_routes_periodic_x_wall_y_F_2d!(
            coarse_next, patch_next, coarse, patch, topology;
            coarse_prolongation=:limited_linear)
        coarse, coarse_next = coarse_next, coarse
        patch, patch_next = patch_next, patch

        if step < steps && step % regrid_every == 0
            ranges = conservative_tree_velocity_gradient_patch_range_2d(
                coarse, patch; threshold=gradient_threshold,
                volume_leaf=volume_fine, force_x=Fx, pad_leaf=pad_leaf,
                pad_parent=pad_parent, shrink_margin=shrink_margin)
            if ranges.i_range != patch.parent_i_range ||
                    ranges.j_range != patch.parent_j_range
                new_patch = create_conservative_tree_patch_2d(
                    ranges.i_range, ranges.j_range; T=T)
                new_coarse = similar(coarse)
                regrid_conservative_tree_patch_direct_F_2d!(
                    new_coarse, new_patch, coarse, patch)
                coarse = new_coarse
                patch = new_patch
                patch_next = create_conservative_tree_patch_2d(
                    ranges.i_range, ranges.j_range; T=T)
                coarse_next = similar(coarse)
                topology = create_conservative_tree_topology_2d(
                    Nx, Ny, patch; coarse_route_mode=coarse_route_mode)
                regrid_count += 1
                push!(patch_history, (patch.parent_i_range, patch.parent_j_range))
            end
        end
    end

    mass_final = active_mass_F(coarse, patch)
    ux_mean = sum(composite_leaf_mean_ux_profile(coarse, patch;
                                                 volume_leaf=volume_fine,
                                                 force_x=Fx)) / T(2 * Ny)

    return ConservativeTreeAdaptiveRun2D{T}(
        :poiseuille_gradient_adaptive_route_native, coarse, patch, patch_history,
        ux_mean, mass_initial, mass_final, mass_final - mass_initial,
        steps, regrid_every, regrid_count)
end

function validate_conservative_tree_route_native_phase_p_2d(;
        steps::Int=1000,
        T::Type{<:Real}=Float64)
    couette_route = run_conservative_tree_couette_route_native_2d(
        ; steps=steps, T=T)
    couette_oracle = run_conservative_tree_couette_macroflow_2d(
        ; Nx=size(couette_route.coarse_F, 1),
        Ny=size(couette_route.coarse_F, 2),
        patch_i_range=couette_route.patch.parent_i_range,
        patch_j_range=couette_route.patch.parent_j_range,
        steps=steps, T=T)
    couette_l2, couette_linf = _profile_errors(
        couette_route.ux_profile, couette_oracle.ux_profile)

    poiseuille_route = run_conservative_tree_poiseuille_route_native_2d(
        ; steps=steps, T=T)
    poiseuille_oracle = run_conservative_tree_poiseuille_macroflow_2d(
        ; Nx=size(poiseuille_route.coarse_F, 1),
        Ny=size(poiseuille_route.coarse_F, 2),
        patch_i_range=poiseuille_route.patch.parent_i_range,
        patch_j_range=poiseuille_route.patch.parent_j_range,
        steps=steps, T=T)
    poiseuille_l2, poiseuille_linf = _profile_errors(
        poiseuille_route.ux_profile, poiseuille_oracle.ux_profile)

    return (
        couette=(
            route=couette_route,
            oracle=couette_oracle,
            l2_delta=couette_l2,
            linf_delta=couette_linf,
        ),
        poiseuille=(
            route=poiseuille_route,
            oracle=poiseuille_oracle,
            l2_delta=poiseuille_l2,
            linf_delta=poiseuille_linf,
        ),
    )
end

function run_conservative_tree_square_obstacle_route_native_2d(;
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
        coarse_route_mode::Symbol=:leaf_equivalent,
        T::Type{<:Real}=Float64)
    Fx = T(Fx)
    Fy = T(Fy)
    omega = T(omega)
    rho = T(rho)
    volume_coarse = one(T)
    volume_fine = T(0.25)

    is_solid = square_solid_mask_leaf_2d(2 * Nx, 2 * Ny,
                                         obstacle_i_range, obstacle_j_range)
    coarse = zeros(T, Nx, Ny, 9)
    coarse_next = similar(coarse)
    patch = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    topology = create_conservative_tree_topology_2d(
        Nx, Ny, patch; coarse_route_mode=coarse_route_mode)
    leaf = zeros(T, 2 * Nx, 2 * Ny, 9)

    _check_route_solid_mask_layout(topology, coarse, patch, is_solid)

    fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, rho, zero(T), zero(T))
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, rho, zero(T), zero(T))
    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass_initial = _leaf_fluid_mass_F(leaf, is_solid)

    for _ in 1:steps
        collide_Guo_composite_solid_F_2d!(
            coarse, patch, topology, is_solid,
            volume_coarse, volume_fine, omega, omega, Fx, Fy)
        stream_composite_routes_periodic_x_wall_y_solid_F_2d!(
            coarse_next, patch_next, coarse, patch, topology, is_solid;
            coarse_prolongation=:limited_linear)
        coarse, coarse_next = coarse_next, coarse
        patch, patch_next = patch_next, patch
    end

    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass_final = _leaf_fluid_mass_F(leaf, is_solid)
    ux_mean, uy_mean = _leaf_fluid_mean_velocity_F(
        leaf, is_solid; volume=volume_fine, force_x=Fx, force_y=Fy)

    return ConservativeTreeSolidFlowResult2D{T}(
        :square_obstacle_route_native, coarse, patch, is_solid, ux_mean, uy_mean,
        mass_initial, mass_final, mass_final - mass_initial, steps)
end

function run_conservative_tree_cylinder_obstacle_route_native_2d(;
        Nx::Int=24,
        Ny::Int=14,
        patch_i_range::UnitRange{Int}=8:17,
        patch_j_range::UnitRange{Int}=4:11,
        cx_leaf=(2 * Nx + 1) / 2,
        cy_leaf=(2 * Ny + 1) / 2,
        radius_leaf=3.0,
        Fx=2e-5,
        omega=1.0,
        rho=1.0,
        steps::Int=1200,
        avg_window::Int=300,
        coarse_route_mode::Symbol=:leaf_equivalent,
        T::Type{<:Real}=Float64)
    steps > 0 || throw(ArgumentError("steps must be positive"))
    avg_window > 0 || throw(ArgumentError("avg_window must be positive"))
    Fx = T(Fx)
    omega = T(omega)
    rho = T(rho)
    cx_leaf = T(cx_leaf)
    cy_leaf = T(cy_leaf)
    radius_leaf = T(radius_leaf)
    volume_coarse = one(T)
    volume_fine = T(0.25)
    avg_window_i = min(avg_window, steps)

    is_solid = cylinder_solid_mask_leaf_2d(2 * Nx, 2 * Ny,
                                           cx_leaf, cy_leaf, radius_leaf)
    coarse = zeros(T, Nx, Ny, 9)
    coarse_next = similar(coarse)
    patch = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    topology = create_conservative_tree_topology_2d(
        Nx, Ny, patch; coarse_route_mode=coarse_route_mode)
    leaf = zeros(T, 2 * Nx, 2 * Ny, 9)
    leaf_next = similar(leaf)

    _check_route_solid_mask_layout(topology, coarse, patch, is_solid)

    fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, rho, zero(T), zero(T))
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, rho, zero(T), zero(T))
    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass_initial = _leaf_fluid_mass_F(leaf, is_solid)

    Fx_sum = zero(T)
    Fy_sum = zero(T)
    n_avg = 0
    for step in 1:steps
        collide_Guo_composite_solid_F_2d!(
            coarse, patch, topology, is_solid,
            volume_coarse, volume_fine, omega, omega, Fx, zero(T))
        stream_composite_routes_periodic_x_wall_y_solid_F_2d!(
            coarse_next, patch_next, coarse, patch, topology, is_solid;
            coarse_prolongation=:limited_linear)

        if step > steps - avg_window_i
            composite_to_leaf_F_2d!(leaf, coarse, patch)
            composite_to_leaf_F_2d!(leaf_next, coarse_next, patch_next)
            drag = compute_drag_mea_solid_F_2d(leaf, leaf_next, is_solid)
            Fx_sum += T(drag.Fx)
            Fy_sum += T(drag.Fy)
            n_avg += 1
        end

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
        mass_initial, mass_final, mass_final - mass_initial, steps, avg_window_i)
end

struct ConservativeTreeBenchmarkRow2D
    flow::Symbol
    method::Symbol
    Nx::Int
    Ny::Int
    steps::Int
    ux_mean::Float64
    uy_mean::Float64
    mass_rel_drift::Float64
    elapsed_s::Float64
end

function _conservative_tree_benchmark_row_2d(flow::Symbol,
                                             method::Symbol,
                                             Nx::Int,
                                             Ny::Int,
                                             result,
                                             elapsed_s::Real)
    ux = hasproperty(result, :ux_mean) ? getproperty(result, :ux_mean) :
         getproperty(result, :u_ref)
    uy = hasproperty(result, :uy_mean) ? getproperty(result, :uy_mean) : zero(ux)
    mass_initial = getproperty(result, :mass_initial)
    mass_drift = getproperty(result, :mass_drift)
    rel = abs(mass_drift) / max(abs(mass_initial), eps(typeof(float(mass_initial))))
    return ConservativeTreeBenchmarkRow2D(
        flow, method, Nx, Ny, getproperty(result, :steps),
        Float64(ux), Float64(uy), Float64(rel), Float64(elapsed_s))
end

function benchmark_conservative_tree_cartesian_vs_amr_2d(;
        flows::Tuple=(:bfs, :square, :cylinder),
        steps::Int=240,
        T::Type{<:Real}=Float64)
    steps > 0 || throw(ArgumentError("steps must be positive"))
    rows = ConservativeTreeBenchmarkRow2D[]
    for flow in flows
        if flow == :bfs
            leaf = nothing
            leaf_elapsed = @elapsed leaf = run_conservative_tree_bfs_macroflow_2d(
                ; steps=steps, T=T)
            route = nothing
            route_elapsed = @elapsed route = run_conservative_tree_bfs_route_native_2d(
                ; steps=steps, T=T)
            push!(rows, _conservative_tree_benchmark_row_2d(
                :bfs, :leaf_oracle, 28, 14, leaf, leaf_elapsed))
            push!(rows, _conservative_tree_benchmark_row_2d(
                :bfs, :amr_route_native, 28, 14, route, route_elapsed))
        elseif flow == :square
            leaf = nothing
            leaf_elapsed = @elapsed leaf = run_conservative_tree_square_obstacle_macroflow_2d(
                ; steps=steps, T=T)
            route = nothing
            route_elapsed = @elapsed route = run_conservative_tree_square_obstacle_route_native_2d(
                ; steps=steps, T=T)
            push!(rows, _conservative_tree_benchmark_row_2d(
                :square, :leaf_oracle, 24, 14, leaf, leaf_elapsed))
            push!(rows, _conservative_tree_benchmark_row_2d(
                :square, :amr_route_native, 24, 14, route, route_elapsed))
        elseif flow == :cylinder
            leaf = nothing
            leaf_elapsed = @elapsed leaf = run_conservative_tree_cylinder_macroflow_2d(
                ; steps=steps, avg_window=min(steps, 60), T=T)
            route = nothing
            route_elapsed = @elapsed route = run_conservative_tree_cylinder_obstacle_route_native_2d(
                ; steps=steps, avg_window=min(steps, 60), T=T)
            push!(rows, _conservative_tree_benchmark_row_2d(
                :cylinder, :leaf_oracle, 24, 14, leaf, leaf_elapsed))
            push!(rows, _conservative_tree_benchmark_row_2d(
                :cylinder, :amr_route_native, 24, 14, route, route_elapsed))
        else
            throw(ArgumentError("unsupported conservative-tree benchmark flow: $flow"))
        end
    end
    return rows
end

struct ConservativeTreeConvergenceRow2D
    flow::Symbol
    method::Symbol
    scale::Int
    Nx::Int
    Ny::Int
    steps::Int
    ux_mean::Float64
    uy_mean::Float64
    Fx_drag::Float64
    Fy_drag::Float64
    Cd::Float64
    mass_rel_drift::Float64
    elapsed_s::Float64
end

function _scale_parent_range_2d(range::UnitRange{Int}, scale::Int)
    scale > 0 || throw(ArgumentError("scale must be positive"))
    return ((first(range) - 1) * scale + 1):(last(range) * scale)
end

function _scale_leaf_range_2d(range::UnitRange{Int}, scale::Int)
    scale > 0 || throw(ArgumentError("scale must be positive"))
    return ((first(range) - 1) * scale + 1):(last(range) * scale)
end

function _coarsen_leaf_range_2d(range::UnitRange{Int})
    isempty(range) && throw(ArgumentError("range must be nonempty"))
    return ((first(range) + 1) >>> 1):((last(range) + 1) >>> 1)
end

function _conservative_tree_obstacle_convergence_row_2d(
        flow::Symbol,
        method::Symbol,
        scale::Int,
        Nx::Int,
        Ny::Int,
        result,
        elapsed_s::Real)
    mass_initial = getproperty(result, :mass_initial)
    mass_drift = getproperty(result, :mass_drift)
    rel = abs(mass_drift) / max(abs(mass_initial), eps(typeof(float(mass_initial))))
    if flow == :cylinder
        return ConservativeTreeConvergenceRow2D(
            flow, method, scale, Nx, Ny, getproperty(result, :steps),
            Float64(getproperty(result, :u_ref)),
            0.0,
            Float64(getproperty(result, :Fx_drag)),
            Float64(getproperty(result, :Fy_drag)),
            Float64(getproperty(result, :Cd)),
            Float64(rel),
            Float64(elapsed_s))
    end

    return ConservativeTreeConvergenceRow2D(
        flow, method, scale, Nx, Ny, getproperty(result, :steps),
        Float64(getproperty(result, :ux_mean)),
        Float64(getproperty(result, :uy_mean)),
        NaN,
        NaN,
        NaN,
        Float64(rel),
        Float64(elapsed_s))
end

function _conservative_tree_obstacle_steps(scale::Int,
                                           base_steps::Int,
                                           step_exponent::Real)
    scale > 0 || throw(ArgumentError("scale must be positive"))
    base_steps > 0 || throw(ArgumentError("base_steps must be positive"))
    step_exponent >= 0 || throw(ArgumentError("step_exponent must be nonnegative"))
    return max(1, round(Int, base_steps * scale^step_exponent))
end

function _conservative_tree_obstacle_patch_ranges_2d(flow::Symbol,
                                                     scale::Int,
                                                     Nx::Int,
                                                     Ny::Int,
                                                     patch_strategy::Symbol)
    scale > 0 || throw(ArgumentError("scale must be positive"))
    if patch_strategy == :default
        if flow == :square || flow == :cylinder
            return (
                i_range=_scale_parent_range_2d(8:17, scale),
                j_range=_scale_parent_range_2d(4:11, scale),
            )
        end
    elseif patch_strategy == :interface_buffered
        if flow == :square || flow == :cylinder
            return (
                i_range=_scale_parent_range_2d(3:22, scale),
                j_range=1:Ny,
            )
        end
    else
        throw(ArgumentError("unsupported obstacle patch_strategy: $patch_strategy"))
    end
    throw(ArgumentError("unsupported obstacle convergence flow: $flow"))
end

function convergence_conservative_tree_obstacles_2d(;
        flows::Tuple=(:square, :cylinder),
        scales::Tuple=(1, 2),
        base_steps::Int=1200,
        step_exponent::Real=1,
        avg_window::Int=300,
        patch_strategy::Symbol=:default,
        include_coarse_cartesian::Bool=false,
        T::Type{<:Real}=Float64)
    avg_window > 0 || throw(ArgumentError("avg_window must be positive"))
    rows = ConservativeTreeConvergenceRow2D[]

    for scale in scales
        scale > 0 || throw(ArgumentError("scales must contain positive integers"))
        steps = _conservative_tree_obstacle_steps(scale, base_steps, step_exponent)
        avg = min(avg_window * scale, steps)

        for flow in flows
            if flow == :square
                Nx = 24 * scale
                Ny = 14 * scale
                patch_ranges = _conservative_tree_obstacle_patch_ranges_2d(
                    :square, scale, Nx, Ny, patch_strategy)
                patch_i = patch_ranges.i_range
                patch_j = patch_ranges.j_range
                obstacle_i = _scale_leaf_range_2d(22:27, scale)
                obstacle_j = _scale_leaf_range_2d(12:17, scale)

                if include_coarse_cartesian
                    Nx_coarse = max(2, Nx >>> 1)
                    Ny_coarse = max(2, Ny >>> 1)
                    coarse_obstacle_i = _coarsen_leaf_range_2d(obstacle_i)
                    coarse_obstacle_j = _coarsen_leaf_range_2d(obstacle_j)
                    coarse_cart = nothing
                    coarse_elapsed = @elapsed coarse_cart =
                        run_conservative_tree_square_obstacle_macroflow_2d(
                            ; Nx=Nx_coarse, Ny=Ny_coarse,
                            patch_i_range=1:Nx_coarse,
                            patch_j_range=1:Ny_coarse,
                            obstacle_i_range=coarse_obstacle_i,
                            obstacle_j_range=coarse_obstacle_j,
                            steps=steps, T=T)
                    push!(rows, _conservative_tree_obstacle_convergence_row_2d(
                        :square, :cartesian_coarse, scale,
                        Nx_coarse, Ny_coarse, coarse_cart, coarse_elapsed))
                end

                leaf = nothing
                leaf_elapsed = @elapsed leaf = run_conservative_tree_square_obstacle_macroflow_2d(
                    ; Nx=Nx, Ny=Ny, patch_i_range=patch_i,
                    patch_j_range=patch_j, obstacle_i_range=obstacle_i,
                    obstacle_j_range=obstacle_j, steps=steps, T=T)
                route = nothing
                route_elapsed = @elapsed route = run_conservative_tree_square_obstacle_route_native_2d(
                    ; Nx=Nx, Ny=Ny, patch_i_range=patch_i,
                    patch_j_range=patch_j, obstacle_i_range=obstacle_i,
                    obstacle_j_range=obstacle_j, steps=steps, T=T)
                push!(rows, _conservative_tree_obstacle_convergence_row_2d(
                    :square, :leaf_oracle, scale, Nx, Ny, leaf, leaf_elapsed))
                push!(rows, _conservative_tree_obstacle_convergence_row_2d(
                    :square, :amr_route_native, scale, Nx, Ny, route, route_elapsed))
            elseif flow == :cylinder
                Nx = 24 * scale
                Ny = 14 * scale
                patch_ranges = _conservative_tree_obstacle_patch_ranges_2d(
                    :cylinder, scale, Nx, Ny, patch_strategy)
                patch_i = patch_ranges.i_range
                patch_j = patch_ranges.j_range
                cx_leaf = (2 * Nx + 1) / 2
                cy_leaf = (2 * Ny + 1) / 2
                radius_leaf = 3 * scale

                if include_coarse_cartesian
                    Nx_coarse = max(2, Nx >>> 1)
                    Ny_coarse = max(2, Ny >>> 1)
                    coarse_cart = nothing
                    coarse_elapsed = @elapsed coarse_cart =
                        run_conservative_tree_cylinder_macroflow_2d(
                            ; Nx=Nx_coarse, Ny=Ny_coarse,
                            patch_i_range=1:Nx_coarse,
                            patch_j_range=1:Ny_coarse,
                            cx_leaf=(2 * Nx_coarse + 1) / 2,
                            cy_leaf=(2 * Ny_coarse + 1) / 2,
                            radius_leaf=radius_leaf / 2,
                            steps=steps, avg_window=avg, T=T)
                    push!(rows, _conservative_tree_obstacle_convergence_row_2d(
                        :cylinder, :cartesian_coarse, scale,
                        Nx_coarse, Ny_coarse, coarse_cart, coarse_elapsed))
                end

                leaf = nothing
                leaf_elapsed = @elapsed leaf = run_conservative_tree_cylinder_macroflow_2d(
                    ; Nx=Nx, Ny=Ny, patch_i_range=patch_i,
                    patch_j_range=patch_j, cx_leaf=cx_leaf, cy_leaf=cy_leaf,
                    radius_leaf=radius_leaf, steps=steps, avg_window=avg, T=T)
                route = nothing
                route_elapsed = @elapsed route = run_conservative_tree_cylinder_obstacle_route_native_2d(
                    ; Nx=Nx, Ny=Ny, patch_i_range=patch_i,
                    patch_j_range=patch_j, cx_leaf=cx_leaf, cy_leaf=cy_leaf,
                    radius_leaf=radius_leaf, steps=steps, avg_window=avg, T=T)
                push!(rows, _conservative_tree_obstacle_convergence_row_2d(
                    :cylinder, :leaf_oracle, scale, Nx, Ny, leaf, leaf_elapsed))
                push!(rows, _conservative_tree_obstacle_convergence_row_2d(
                    :cylinder, :amr_route_native, scale, Nx, Ny, route, route_elapsed))
            else
                throw(ArgumentError("unsupported obstacle convergence flow: $flow"))
            end
        end
    end

    return rows
end

function run_conservative_tree_vfs_route_native_2d(;
        Nx::Int=28,
        Ny::Int=14,
        patch_i_range::UnitRange{Int}=5:12,
        patch_j_range::UnitRange{Int}=1:8,
        step_i_range::UnitRange{Int}=14:19,
        step_height_leaf::Int=8,
        Fx=2e-5,
        Fy=0.0,
        omega=1.0,
        rho=1.0,
        steps::Int=900,
        T::Type{<:Real}=Float64)
    Fx = T(Fx)
    Fy = T(Fy)
    omega = T(omega)
    rho = T(rho)
    volume_coarse = one(T)
    volume_fine = T(0.25)

    is_solid = vertical_facing_step_solid_mask_leaf_2d(
        2 * Nx, 2 * Ny, step_i_range, step_height_leaf)
    coarse = zeros(T, Nx, Ny, 9)
    coarse_next = similar(coarse)
    patch = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    topology = create_conservative_tree_topology_2d(Nx, Ny, patch)
    leaf = zeros(T, 2 * Nx, 2 * Ny, 9)

    _check_route_solid_mask_layout(topology, coarse, patch, is_solid)

    fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, rho, zero(T), zero(T))
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, rho, zero(T), zero(T))
    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass_initial = _leaf_fluid_mass_F(leaf, is_solid)

    for _ in 1:steps
        collide_Guo_composite_solid_F_2d!(
            coarse, patch, topology, is_solid,
            volume_coarse, volume_fine, omega, omega, Fx, Fy)
        stream_composite_routes_periodic_x_wall_y_solid_F_2d!(
            coarse_next, patch_next, coarse, patch, topology, is_solid;
            coarse_prolongation=:limited_linear)
        coarse, coarse_next = coarse_next, coarse
        patch, patch_next = patch_next, patch
    end

    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass_final = _leaf_fluid_mass_F(leaf, is_solid)
    ux_mean, uy_mean = _leaf_fluid_mean_velocity_F(
        leaf, is_solid; volume=volume_fine, force_x=Fx, force_y=Fy)

    return ConservativeTreeSolidFlowResult2D{T}(
        :vfs_route_native, coarse, patch, is_solid, ux_mean, uy_mean,
        mass_initial, mass_final, mass_final - mass_initial, steps)
end

function run_conservative_tree_vfs_mask_adaptive_route_native_2d(;
        Nx::Int=28,
        Ny::Int=14,
        patch_i_range::UnitRange{Int}=5:12,
        patch_j_range::UnitRange{Int}=1:8,
        step_i_range::UnitRange{Int}=14:19,
        step_height_leaf::Int=8,
        regrid_every::Int=120,
        pad::Int=1,
        Fx=2e-5,
        Fy=0.0,
        omega=1.0,
        rho=1.0,
        steps::Int=500,
        T::Type{<:Real}=Float64)
    steps > 0 || throw(ArgumentError("steps must be positive"))
    regrid_every > 0 || throw(ArgumentError("regrid_every must be positive"))
    Fx = T(Fx)
    Fy = T(Fy)
    omega = T(omega)
    rho = T(rho)
    volume_coarse = one(T)
    volume_fine = T(0.25)

    is_solid = vertical_facing_step_solid_mask_leaf_2d(
        2 * Nx, 2 * Ny, step_i_range, step_height_leaf)
    coarse = zeros(T, Nx, Ny, 9)
    coarse_next = similar(coarse)
    patch = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    patch_next = create_conservative_tree_patch_2d(patch_i_range, patch_j_range; T=T)
    topology = create_conservative_tree_topology_2d(Nx, Ny, patch)
    leaf = zeros(T, 2 * Nx, 2 * Ny, 9)

    _check_route_solid_mask_layout(topology, coarse, patch, is_solid)

    fill_equilibrium_integrated_D2Q9!(coarse, volume_coarse, rho, zero(T), zero(T))
    fill_equilibrium_integrated_D2Q9!(patch.fine_F, volume_fine, rho, zero(T), zero(T))
    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass_initial = _leaf_fluid_mass_F(leaf, is_solid)

    patch_history = Tuple{UnitRange{Int},UnitRange{Int}}[
        (patch.parent_i_range, patch.parent_j_range)
    ]
    regrid_count = 0

    for step in 1:steps
        collide_Guo_composite_solid_F_2d!(
            coarse, patch, topology, is_solid,
            volume_coarse, volume_fine, omega, omega, Fx, Fy)
        stream_composite_routes_periodic_x_wall_y_solid_F_2d!(
            coarse_next, patch_next, coarse, patch, topology, is_solid;
            coarse_prolongation=:limited_linear)
        coarse, coarse_next = coarse_next, coarse
        patch, patch_next = patch_next, patch

        if step < steps && step % regrid_every == 0
            ranges = conservative_tree_solid_mask_patch_range_2d(is_solid; pad=pad)
            if ranges.i_range != patch.parent_i_range ||
                    ranges.j_range != patch.parent_j_range
                adapted = adapt_conservative_tree_patch_to_solid_mask_2d(
                    coarse, patch, is_solid; pad=pad)
                coarse = adapted.coarse_F
                patch = adapted.patch
                patch_next = create_conservative_tree_patch_2d(
                    ranges.i_range, ranges.j_range; T=T)
                coarse_next = similar(coarse)
                topology = create_conservative_tree_topology_2d(Nx, Ny, patch)
                _check_route_solid_mask_layout(topology, coarse, patch, is_solid)
                regrid_count += 1
                push!(patch_history, (patch.parent_i_range, patch.parent_j_range))
            end
        end
    end

    composite_to_leaf_F_2d!(leaf, coarse, patch)
    mass_final = _leaf_fluid_mass_F(leaf, is_solid)
    ux_mean, uy_mean = _leaf_fluid_mean_velocity_F(
        leaf, is_solid; volume=volume_fine, force_x=Fx, force_y=Fy)

    return ConservativeTreeSolidAdaptiveRun2D{T}(
        :vfs_mask_adaptive_route_native, coarse, patch, is_solid, patch_history,
        ux_mean, uy_mean, mass_initial, mass_final, mass_final - mass_initial,
        steps, regrid_every, regrid_count)
end
