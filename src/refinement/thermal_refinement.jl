# ===========================================================================
# Thermal extension for patch-based grid refinement
#
# Adds thermal DDF (g populations) support to the existing refinement system.
# The thermal populations g use the same D2Q9 lattice and the same
# prolongation/restriction kernels as f (Filippova-Hanel rescaling applies
# identically to the thermal non-equilibrium part).
#
# The thermal relaxation ω_T is rescaled the same way as ω_f:
#   tau_T_fine = ratio * (tau_T_coarse - 0.5) + 0.5
# ===========================================================================

"""
    ThermalPatchArrays{T}

Extra arrays for thermal DDF on a refinement patch. Stored alongside
the existing RefinementPatch (which holds f arrays).
"""
struct ThermalPatchArrays{T}
    g_in::AbstractArray{T, 3}
    g_out::AbstractArray{T, 3}
    Temp::AbstractMatrix{T}
    omega_T::T
    # Temporal interpolation buffers for thermal (same size as parent region)
    Temp_prev::AbstractMatrix{T}
    g_prev::AbstractArray{T, 3}
end

"""
    create_thermal_patch_arrays(patch, omega_T_parent; backend)

Allocate thermal arrays matching an existing RefinementPatch.
Initializes g to equilibrium at T=0.5 (mid-range).
"""
function create_thermal_patch_arrays(patch::RefinementPatch{T},
                                      omega_T_parent::Real;
                                      T_init::Real=0.5,
                                      backend=KernelAbstractions.CPU()) where T
    Nx, Ny = patch.Nx, patch.Ny

    # Rescale thermal omega the same way as flow omega
    omega_T_fine = T(rescaled_omega(omega_T_parent, patch.ratio))

    g_in  = KernelAbstractions.zeros(backend, T, Nx, Ny, 9)
    g_out = KernelAbstractions.zeros(backend, T, Nx, Ny, 9)
    Temp  = KernelAbstractions.zeros(backend, T, Nx, Ny)

    # Initialize g to equilibrium at T_init
    w = weights(D2Q9())
    g_cpu = zeros(T, Nx, Ny, 9)
    for q in 1:9
        g_cpu[:, :, q] .= T(w[q]) * T(T_init)
    end
    copyto!(g_in, g_cpu)
    copyto!(g_out, g_cpu)

    # Temporal interpolation buffers (same parent-region size as patch)
    n_parent_i = size(patch.rho_prev, 1)
    n_parent_j = size(patch.rho_prev, 2)
    Temp_prev = KernelAbstractions.zeros(backend, T, n_parent_i, n_parent_j)
    g_prev    = KernelAbstractions.zeros(backend, T, n_parent_i, n_parent_j, 9)

    return ThermalPatchArrays{T}(g_in, g_out, Temp, omega_T_fine,
                                  Temp_prev, g_prev)
end

"""
    save_thermal_coarse_state!(patch, thermal, g_coarse, Temp_c)

Save coarse thermal state at time n for temporal interpolation.
"""
function save_thermal_coarse_state!(patch::RefinementPatch{T},
                                    thermal::ThermalPatchArrays{T},
                                    g_coarse, Temp_c) where T
    i_range = patch.parent_i_range
    j_range = patch.parent_j_range
    Nx_c = size(Temp_c, 1)
    Ny_c = size(Temp_c, 2)

    i_lo = max(first(i_range) - 1, 1)
    i_hi = min(last(i_range) + 1, Nx_c)
    j_lo = max(first(j_range) - 1, 1)
    j_hi = min(last(j_range) + 1, Ny_c)

    n_i = i_hi - i_lo + 1
    n_j = j_hi - j_lo + 1

    copyto!(@view(thermal.Temp_prev[1:n_i, 1:n_j]),
            @view(Temp_c[i_lo:i_hi, j_lo:j_hi]))
    copyto!(@view(thermal.g_prev[1:n_i, 1:n_j, :]),
            @view(g_coarse[i_lo:i_hi, j_lo:j_hi, :]))
end

"""
    fill_thermal_ghost!(patch, thermal, g_coarse, Nx_c, Ny_c)

Fill fine-grid thermal ghost cells using simple injection from coarse g.
Unlike flow populations, thermal g uses simple bilinear interpolation
(no Filippova-Hanel rescaling — g_eq is linear in T, not quadratic in u).
"""
function fill_thermal_ghost!(patch::RefinementPatch{T},
                              thermal::ThermalPatchArrays{T},
                              g_coarse,
                              Nx_c::Int, Ny_c::Int) where T
    _fill_thermal_ghost_simple!(
        thermal.g_in, g_coarse,
        patch.ratio, patch.Nx_inner, patch.Ny_inner,
        patch.n_ghost, first(patch.parent_i_range), first(patch.parent_j_range),
        Nx_c, Ny_c
    )
end

"""Simple injection of coarse g into fine ghost cells (nearest-neighbor)."""
function _fill_thermal_ghost_simple!(g_fine, g_coarse,
                                      ratio, Nx_inner, Ny_inner, n_ghost,
                                      i_offset, j_offset, Nx_c, Ny_c)
    backend = KernelAbstractions.get_backend(g_fine)
    Nx_f = Nx_inner + 2 * n_ghost
    Ny_f = Ny_inner + 2 * n_ghost
    kernel! = _fill_thermal_ghost_simple_kernel!(backend)
    kernel!(g_fine, g_coarse, ratio, Nx_inner, Ny_inner, n_ghost,
            i_offset, j_offset, Nx_c, Ny_c; ndrange=(Nx_f, Ny_f))
    KernelAbstractions.synchronize(backend)
end

@kernel function _fill_thermal_ghost_simple_kernel!(g_fine, @Const(g_coarse),
                                                     ratio, Nx_inner, Ny_inner, n_ghost,
                                                     i_offset, j_offset, Nx_c, Ny_c)
    i_f, j_f = @index(Global, NTuple)

    @inbounds begin
        # Only write ghost cells
        is_ghost = (i_f <= n_ghost) || (i_f > n_ghost + Nx_inner) ||
                   (j_f <= n_ghost) || (j_f > n_ghost + Ny_inner)
        if is_ghost
            T = eltype(g_fine)
            # Map fine cell center to coarse index
            x_f = T(i_f - n_ghost - 0.5) / T(ratio)
            y_f = T(j_f - n_ghost - 0.5) / T(ratio)

            i_c = clamp(round(Int, x_f) + i_offset, 1, Nx_c)
            j_c = clamp(round(Int, y_f) + j_offset, 1, Ny_c)

            for q in 1:9
                g_fine[i_f, j_f, q] = g_coarse[i_c, j_c, q]
            end
        end
    end
end

"""
    restrict_thermal_to_coarse!(patch, thermal, g_coarse, Temp_c)

Restrict fine thermal populations back to coarse overlap (block average).
"""
function restrict_thermal_to_coarse!(patch::RefinementPatch{T},
                                      thermal::ThermalPatchArrays{T},
                                      g_coarse, Temp_c) where T
    Nx_overlap = length(patch.parent_i_range)
    Ny_overlap = length(patch.parent_j_range)
    _restrict_thermal_simple!(
        g_coarse, Temp_c, thermal.g_in, thermal.Temp,
        patch.ratio, patch.n_ghost,
        first(patch.parent_i_range), first(patch.parent_j_range),
        Nx_overlap, Ny_overlap
    )
end

function _restrict_thermal_simple!(g_coarse, Temp_c, g_fine, Temp_f,
                                    ratio, n_ghost, i_offset, j_offset,
                                    Nx_overlap, Ny_overlap)
    backend = KernelAbstractions.get_backend(g_coarse)
    kernel! = _restrict_thermal_simple_kernel!(backend)
    kernel!(g_coarse, Temp_c, g_fine, Temp_f,
            ratio, n_ghost, i_offset, j_offset;
            ndrange=(Nx_overlap, Ny_overlap))
    KernelAbstractions.synchronize(backend)
end

@kernel function _restrict_thermal_simple_kernel!(g_coarse, Temp_c, @Const(g_fine), @Const(Temp_f),
                                                   ratio, n_ghost, i_offset, j_offset)
    ic, jc = @index(Global, NTuple)

    @inbounds begin
        T = eltype(g_coarse)
        i_c = ic + i_offset - 1
        j_c = jc + j_offset - 1
        inv_r2 = one(T) / T(ratio * ratio)

        # Block-average temperature
        Temp_avg = zero(T)
        for dj in 0:ratio-1, di in 0:ratio-1
            i_f = n_ghost + (ic - 1) * ratio + di + 1
            j_f = n_ghost + (jc - 1) * ratio + dj + 1
            Temp_avg += Temp_f[i_f, j_f]
        end
        Temp_c[i_c, j_c] = Temp_avg * inv_r2

        # Block-average g populations
        for q in 1:9
            g_avg = zero(T)
            for dj in 0:ratio-1, di in 0:ratio-1
                i_f = n_ghost + (ic - 1) * ratio + di + 1
                j_f = n_ghost + (jc - 1) * ratio + dj + 1
                g_avg += g_fine[i_f, j_f, q]
            end
            g_coarse[i_c, j_c, q] = g_avg * inv_r2
        end
    end
end

"""
    advance_thermal_refined_step!(domain, thermals,
        f_in, f_out, g_in, g_out, rho, ux, uy, Temp, is_solid;
        fused_step_fn, bc_thermal_patch_fns)

Advance one coarse timestep for natural convection with thermal refinement.
Uses the fused kernel on the base grid and sub-cycles both f and g on patches.

# Arguments
- `thermals`: Vector{ThermalPatchArrays} matching domain.patches
- `fused_step_fn`: closure `(f_out, f_in, g_out, g_in, Temp, Nx, Ny) -> ...` for base grid
- `bc_thermal_patch_fns`: Dict{Int, Function} for patch thermal BCs
"""
function advance_thermal_refined_step!(domain::RefinedDomain{T},
                                        thermals::Vector{ThermalPatchArrays{T}},
                                        f_in, f_out, g_in, g_out,
                                        rho, ux, uy, Temp, is_solid;
                                        fused_step_fn,
                                        omega_T_coarse::Real,
                                        β_g::Real=zero(T),
                                        T_ref_buoy::Real=zero(T),
                                        bc_thermal_patch_fns=nothing) where T
    Nx = domain.base_Nx
    Ny = domain.base_Ny

    # 1. Save coarse state at time n for all patches (f + g)
    for (pidx, patch) in enumerate(domain.patches)
        save_coarse_state!(patch, f_in, rho, ux, uy)
        save_thermal_coarse_state!(patch, thermals[pidx], g_in, Temp)
    end

    # 2. Advance coarse grid one step (fused kernel)
    fused_step_fn(f_out, f_in, g_out, g_in, Temp, Nx, Ny)
    f_in, f_out = f_out, f_in
    g_in, g_out = g_out, g_in
    # Recover macroscopic on coarse for ghost fill
    compute_macroscopic_2d!(rho, ux, uy, f_in)
    compute_temperature_2d!(Temp, g_in)

    # 3. Sub-cycle each patch
    for (pidx, patch) in enumerate(domain.patches)
        thermal = thermals[pidx]
        ratio = patch.ratio

        for sub_step in 1:ratio
            t_frac = T((sub_step - 1) / ratio)

            # Flow ghost fill (with temporal interpolation for sub-steps)
            if t_frac > zero(T)
                _fill_ghost_interpolated!(patch, f_in, rho, ux, uy,
                                          Float64(domain.base_omega),
                                          Nx, Ny, t_frac)
            else
                fill_ghost_from_coarse!(patch, f_in, rho, ux, uy,
                                         Float64(domain.base_omega), Nx, Ny)
            end
            # Thermal ghost: simple injection from coarse
            fill_thermal_ghost!(patch, thermal, g_in, Nx, Ny)

            # Stream both f and g on patch
            stream_2d!(patch.f_out, patch.f_in, patch.Nx, patch.Ny)
            stream_2d!(thermal.g_out, thermal.g_in, patch.Nx, patch.Ny)

            # Thermal BCs on patch (if patch edge touches domain wall)
            if bc_thermal_patch_fns !== nothing
                bc_fn = get(bc_thermal_patch_fns, pidx, nothing)
                if bc_fn !== nothing
                    bc_fn(thermal.g_out, patch.Nx, patch.Ny)
                end
            end

            # Recover temperature
            compute_temperature_2d!(thermal.Temp, thermal.g_out)
            # Recover macroscopic flow
            compute_macroscopic_2d!(patch.rho, patch.ux, patch.uy, patch.f_out)

            # Collide thermal
            collide_thermal_2d!(thermal.g_out, patch.ux, patch.uy, thermal.omega_T)
            # Collide flow with Boussinesq
            collide_boussinesq_2d!(patch.f_out, thermal.Temp, patch.is_solid,
                                    patch.omega, T(β_g), T(T_ref_buoy))

            # Swap for next sub-step
            copyto!(patch.f_in, patch.f_out)
            copyto!(thermal.g_in, thermal.g_out)
        end
    end

    # 4. Restrict fine results back to coarse
    for (pidx, patch) in enumerate(domain.patches)
        thermal = thermals[pidx]
        restrict_to_coarse!(patch, f_in, rho, ux, uy,
                           Float64(domain.base_omega))
        restrict_thermal_to_coarse!(patch, thermal, g_in, Temp)
    end

    return f_in, f_out, g_in, g_out
end
