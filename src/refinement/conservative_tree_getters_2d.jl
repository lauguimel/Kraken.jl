
"""
    composite_leaf_mean_ux_profile(coarse_F, patch; volume_leaf, force_x)

Convention: Integrated. This getter reads integrated Guo moments directly while
building the leaf-equivalent profile.

Canonical pair member: `collide_Guo_composite_F_2d!` at
`src/refinement/conservative_tree_2d.jl:1661`.
"""
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
            ux_sum += (momentum_F(cell)[1] / volume_leaf) / rho
        end
        profile[j] = ux_sum / T(size(leaf, 1))
    end
    return profile
end

"""
    composite_leaf_velocity_field_2d(coarse_F, patch; volume_leaf,
                                     force_x, force_y)

Convention: Integrated. This getter reads integrated Guo moments directly while
forming leaf-equivalent fields.

Canonical pair member: `collide_Guo_composite_F_2d!` at
`src/refinement/conservative_tree_2d.jl:1661`.
"""
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
        ux[i, j] = (mx / volume_leaf) / rho
        uy[i, j] = (my / volume_leaf) / rho
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
    H = T(ny)
    return [
        T(Fx) / (T(2) * T(rho) * nu) *
        (T(j) - T(0.5)) * (H + T(0.5) - T(j))
        for j in 1:ny
    ]
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

"""
    _leaf_fluid_mean_ux_F(F, is_solid; volume, force_x)

Convention: Integrated. This getter reads integrated Guo moments directly for
the fluid-cell mean `ux`.

Canonical pair members: `collide_Guo_integrated_D2Q9!` at
`src/refinement/conservative_tree_2d.jl:1042` and
`collide_Guo_composite_solid_F_2d!` at
`src/refinement/conservative_tree_streaming_2d.jl:792`.
"""
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
        ux_sum += (momentum_F(cell)[1] / volume) / rho
        n_fluid += 1
    end
    n_fluid > 0 || throw(ArgumentError("solid mask leaves no fluid cells"))
    return ux_sum / T(n_fluid)
end

"""
    _leaf_fluid_mean_velocity_F(F, is_solid; volume, force_x, force_y)

Convention: Integrated. This getter reads integrated Guo moments directly for
the fluid-cell mean velocity.

Canonical pair member: `collide_Guo_integrated_D2Q9!` at
`src/refinement/conservative_tree_2d.jl:1042`.
"""
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
        ux_sum += (p[1] / volume) / rho
        uy_sum += (p[2] / volume) / rho
        n_fluid += 1
    end
    n_fluid > 0 || throw(ArgumentError("solid mask leaves no fluid cells"))
    return ux_sum / T(n_fluid), uy_sum / T(n_fluid)
end
