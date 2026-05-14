include("conservative_tree_types_2d.jl")
include("conservative_tree_transfer_2d.jl")
include("conservative_tree_streaming_oracle_2d.jl")
include("conservative_tree_ledger_2d.jl")
include("conservative_tree_collide_guo_2d.jl")

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
        cx_leaf=(2 * Nx + 1) / 2,
        cy_leaf=(2 * Ny + 1) / 2,
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
        cy_leaf=(2 * Ny + 1) / 2,
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
