"""
    collide_BGK_integrated_D2Q9!(Fcell, volume, omega)

Apply a local BGK collision to one D2Q9 cell stored as integrated populations.
The conversion is `f_i = F_i / volume`; after collision the populations are
stored again as `F_i`.
"""
function collide_BGK_integrated_D2Q9!(Fcell::AbstractVector, volume, omega)
    _check_d2q9_vector(Fcell, "Fcell")
    volume > zero(volume) || throw(ArgumentError("volume must be positive"))

    T = typeof(float(Fcell[1]))
    volume_T = T(volume)
    omega_T = T(omega)
    m = T(mass_F(Fcell))
    iszero(m) && throw(ArgumentError("Fcell mass must be nonzero"))
    mx, my = momentum_F(Fcell)
    mx_T = T(mx)
    my_T = T(my)
    rho = m / volume_T
    ux = mx_T / m
    uy = my_T / m

    @inbounds for q in 1:9
        f = T(Fcell[q]) / volume_T
        feq = equilibrium(D2Q9(), rho, ux, uy, q)
        Fcell[q] = (f - omega_T * (f - feq)) * volume_T
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

"""
    collide_Guo_integrated_D2Q9!(Fcell, volume, omega, Fx, Fy)

Convention: Integrated. This collision integrates the Guo source into
integrated populations so raw moments read as physical velocity moments.

Canonical pair member: `compute_macroscopic_2d!` at
`src/kernels/macroscopic.jl:20`.
"""
function collide_Guo_integrated_D2Q9!(Fcell::AbstractVector,
                                      volume,
                                      omega,
                                      Fx,
                                      Fy)
    _check_d2q9_vector(Fcell, "Fcell")
    volume > zero(volume) || throw(ArgumentError("volume must be positive"))

    T = typeof(float(Fcell[1]))
    volume_T = T(volume)
    omega_T = T(omega)
    Fx_T = T(Fx)
    Fy_T = T(Fy)
    m = T(mass_F(Fcell))
    iszero(m) && throw(ArgumentError("Fcell mass must be nonzero"))
    mx, my = momentum_F(Fcell)
    mx_T = T(mx)
    my_T = T(my)
    rho = m / volume_T
    ux = (mx_T / volume_T + Fx_T / T(2)) / rho
    uy = (my_T / volume_T + Fy_T / T(2)) / rho
    guo_pref = T(1) - omega_T / T(2)

    @inbounds for q in 1:9
        cx = T(d2q9_cx(q))
        cy = T(d2q9_cy(q))
        w = T(weights(D2Q9())[q])
        ci_dot_u = cx * ux + cy * uy
        ci_dot_F = cx * Fx_T + cy * Fy_T
        Sq = w * (T(3) * ((cx - ux) * Fx_T + (cy - uy) * Fy_T) +
                  T(9) * ci_dot_u * ci_dot_F)
        f = T(Fcell[q]) / volume_T
        feq = equilibrium(D2Q9(), rho, ux, uy, q)
        Fcell[q] = volume_T * (f - omega_T * (f - feq) + guo_pref * Sq)
    end
    return Fcell
end

"""
    collide_Guo_integrated_D2Q9!(F, volume, omega, Fx, Fy)

Convention: Integrated. This array wrapper applies the integrated Guo collision
cellwise and guarantees raw moments are the physical velocity readout.

Canonical pair member: `compute_macroscopic_2d!` at
`src/kernels/macroscopic.jl:20`.
"""
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

"""
    collide_Guo_integrated_D2Q9!(F, is_solid, volume, omega, Fx, Fy)

Convention: Integrated. This solid-mask wrapper skips solid cells and applies
the integrated Guo collision to fluid cells.

Canonical pair member: `_leaf_fluid_mean_velocity_F` at
`src/refinement/conservative_tree_2d.jl:1800`.
"""
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

"""
    collide_Guo_composite_F_2d!(coarse_F, patch, volume_coarse,
                                volume_fine, omega_coarse, omega_fine,
                                Fx, Fy)

Convention: Integrated. This composite collision applies integrated Guo updates
to active coarse cells and fine patch leaves.

Canonical pair member: `composite_leaf_mean_ux_profile` at
`src/refinement/conservative_tree_2d.jl:1702`.
"""
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
