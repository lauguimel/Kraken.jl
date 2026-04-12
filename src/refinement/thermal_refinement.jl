# ===========================================================================
# Thermal extension for patch-based grid refinement
#
# Adds thermal DDF (g populations) support to the existing refinement system.
# The thermal populations g use the same D2Q9 lattice but simpler
# prolongation (bilinear interpolation without Filippova-Hanel rescaling,
# since g_eq is linear in T) and restriction (block averaging).
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
    fill_thermal_ghost!(patch, thermal, g_coarse, Nx_c, Ny_c; t_frac=0)

Fill fine-grid thermal ghost cells using bilinear interpolation of coarse g.
Unlike flow populations, no Filippova-Hanel rescaling (g_eq is linear in T).
When `t_frac > 0`, temporally interpolates between saved g_prev (time n)
and current g_coarse (time n+1) for sub-cycling consistency.
"""
function fill_thermal_ghost!(patch::RefinementPatch{T},
                              thermal::ThermalPatchArrays{T},
                              g_coarse,
                              Nx_c::Int, Ny_c::Int;
                              t_frac::Real=zero(T)) where T
    # Always use the temporal path: at t_frac=0 it reads from g_prev (saved at
    # coarse time n) which is what we want, since at sub_step=1 the current
    # coarse g_in has already been advanced to time n+1 by the fused base step.
    _fill_thermal_ghost_temporal!(
        thermal.g_in, g_coarse, thermal.g_prev,
        patch.ratio, patch.Nx_inner, patch.Ny_inner,
        patch.n_ghost, first(patch.parent_i_range), first(patch.parent_j_range),
        Nx_c, Ny_c, T(t_frac))
end

"""Bilinear interpolation of coarse g into fine ghost cells."""
function _fill_thermal_ghost_simple!(g_fine, g_coarse,
                                      ratio, Nx_inner, Ny_inner, n_ghost,
                                      i_offset, j_offset, Nx_c, Ny_c)
    backend = KernelAbstractions.get_backend(g_fine)
    Nx_f = Nx_inner + 2 * n_ghost
    Ny_f = Ny_inner + 2 * n_ghost
    kernel! = _fill_thermal_ghost_bilinear_kernel!(backend)
    kernel!(g_fine, g_coarse, ratio, Nx_inner, Ny_inner, n_ghost,
            i_offset, j_offset, Nx_c, Ny_c; ndrange=(Nx_f, Ny_f))
    KernelAbstractions.synchronize(backend)
end

@kernel function _fill_thermal_ghost_bilinear_kernel!(g_fine, @Const(g_coarse),
                                                       ratio, Nx_inner, Ny_inner, n_ghost,
                                                       i_offset, j_offset, Nx_c, Ny_c)
    i_f, j_f = @index(Global, NTuple)

    @inbounds begin
        is_ghost = (i_f <= n_ghost) || (i_f > n_ghost + Nx_inner) ||
                   (j_f <= n_ghost) || (j_f > n_ghost + Ny_inner)
        if is_ghost
            T = eltype(g_fine)

            # Fine cell center → coarse continuous index (same mapping as flow)
            xc = T(i_f - n_ghost - 1) / T(ratio) + T(i_offset) + T(0.5) / T(ratio)
            yc = T(j_f - n_ghost - 1) / T(ratio) + T(j_offset) + T(0.5) / T(ratio)

            # Bilinear stencil
            i0_raw = trunc(Int, xc)
            j0_raw = trunc(Int, yc)
            tx = xc - T(i0_raw)
            ty = yc - T(j0_raw)

            i0 = clamp(i0_raw, 1, Nx_c)
            i1 = clamp(i0_raw + 1, 1, Nx_c)
            j0 = clamp(j0_raw, 1, Ny_c)
            j1 = clamp(j0_raw + 1, 1, Ny_c)
            tx = clamp(tx, zero(T), one(T))
            ty = clamp(ty, zero(T), one(T))

            w00 = (one(T) - tx) * (one(T) - ty)
            w10 = tx * (one(T) - ty)
            w01 = (one(T) - tx) * ty
            w11 = tx * ty

            # Interpolate g populations directly (g_eq linear in T → no rescaling)
            for q in 1:9
                g_fine[i_f, j_f, q] = w00 * g_coarse[i0, j0, q] +
                                       w10 * g_coarse[i1, j0, q] +
                                       w01 * g_coarse[i0, j1, q] +
                                       w11 * g_coarse[i1, j1, q]
            end
        end
    end
end

"""Temporal + bilinear interpolation of coarse g into fine ghost cells."""
function _fill_thermal_ghost_temporal!(g_fine, g_coarse, g_prev,
                                       ratio, Nx_inner, Ny_inner, n_ghost,
                                       i_offset, j_offset, Nx_c, Ny_c,
                                       t_frac)
    backend = KernelAbstractions.get_backend(g_fine)
    Nx_f = Nx_inner + 2 * n_ghost
    Ny_f = Ny_inner + 2 * n_ghost
    # g_prev local origin: coarse index i_lo maps to g_prev[1,:]
    i_lo = max(i_offset - 1, 1)
    j_lo = max(j_offset - 1, 1)
    kernel! = _fill_thermal_ghost_temporal_kernel!(backend)
    kernel!(g_fine, g_coarse, g_prev, ratio, Nx_inner, Ny_inner, n_ghost,
            i_offset, j_offset, Nx_c, Ny_c, t_frac,
            i_lo, j_lo, Int(size(g_prev, 1)), Int(size(g_prev, 2));
            ndrange=(Nx_f, Ny_f))
    KernelAbstractions.synchronize(backend)
end

@kernel function _fill_thermal_ghost_temporal_kernel!(
        g_fine, @Const(g_coarse), @Const(g_prev),
        ratio, Nx_inner, Ny_inner, n_ghost,
        i_offset, j_offset, Nx_c, Ny_c, t_frac,
        i_lo, j_lo, Ni_prev, Nj_prev)
    i_f, j_f = @index(Global, NTuple)

    @inbounds begin
        is_ghost = (i_f <= n_ghost) || (i_f > n_ghost + Nx_inner) ||
                   (j_f <= n_ghost) || (j_f > n_ghost + Ny_inner)
        if is_ghost
            T = eltype(g_fine)
            t = T(t_frac)

            xc = T(i_f - n_ghost - 1) / T(ratio) + T(i_offset) + T(0.5) / T(ratio)
            yc = T(j_f - n_ghost - 1) / T(ratio) + T(j_offset) + T(0.5) / T(ratio)

            i0_raw = trunc(Int, xc)
            j0_raw = trunc(Int, yc)
            tx = xc - T(i0_raw)
            ty = yc - T(j0_raw)

            i0 = clamp(i0_raw, 1, Nx_c)
            i1 = clamp(i0_raw + 1, 1, Nx_c)
            j0 = clamp(j0_raw, 1, Ny_c)
            j1 = clamp(j0_raw + 1, 1, Ny_c)
            tx = clamp(tx, zero(T), one(T))
            ty = clamp(ty, zero(T), one(T))

            w00 = (one(T) - tx) * (one(T) - ty)
            w10 = tx * (one(T) - ty)
            w01 = (one(T) - tx) * ty
            w11 = tx * ty

            # Local indices into g_prev buffer
            ip0 = clamp(i0 - i_lo + 1, 1, Ni_prev)
            ip1 = clamp(i1 - i_lo + 1, 1, Ni_prev)
            jp0 = clamp(j0 - j_lo + 1, 1, Nj_prev)
            jp1 = clamp(j1 - j_lo + 1, 1, Nj_prev)

            for q in 1:9
                # Temporal blend at each stencil point
                g00 = (one(T) - t) * g_prev[ip0, jp0, q] + t * g_coarse[i0, j0, q]
                g10 = (one(T) - t) * g_prev[ip1, jp0, q] + t * g_coarse[i1, j0, q]
                g01 = (one(T) - t) * g_prev[ip0, jp1, q] + t * g_coarse[i0, j1, q]
                g11 = (one(T) - t) * g_prev[ip1, jp1, q] + t * g_coarse[i1, j1, q]

                # Spatial bilinear interpolation
                g_fine[i_f, j_f, q] = w00 * g00 + w10 * g10 + w01 * g01 + w11 * g11
            end
        end
    end
end

"""
    fill_thermal_full!(patch, thermal, g_coarse, Nx_c, Ny_c)

Bilinearly prolongate coarse g into ALL cells (interior + ghost) of the
fine patch. Used once at initialization so the patch state matches the
coarse state instead of starting from a uniform T_init.
"""
function fill_thermal_full!(patch::RefinementPatch{T},
                             thermal::ThermalPatchArrays{T},
                             g_coarse, Nx_c::Int, Ny_c::Int) where T
    backend = KernelAbstractions.get_backend(thermal.g_in)
    Nx_f = patch.Nx_inner + 2 * patch.n_ghost
    Ny_f = patch.Ny_inner + 2 * patch.n_ghost
    kernel! = _fill_thermal_full_kernel!(backend)
    kernel!(thermal.g_in, g_coarse, patch.ratio,
            patch.n_ghost, first(patch.parent_i_range), first(patch.parent_j_range),
            Nx_c, Ny_c; ndrange=(Nx_f, Ny_f))
    KernelAbstractions.synchronize(backend)
    copyto!(thermal.g_out, thermal.g_in)
end

@kernel function _fill_thermal_full_kernel!(g_fine, @Const(g_coarse),
                                             ratio, n_ghost,
                                             i_offset, j_offset, Nx_c, Ny_c)
    i_f, j_f = @index(Global, NTuple)

    @inbounds begin
        T = eltype(g_fine)
        xc = T(i_f - n_ghost - 1) / T(ratio) + T(i_offset) + T(0.5) / T(ratio)
        yc = T(j_f - n_ghost - 1) / T(ratio) + T(j_offset) + T(0.5) / T(ratio)

        i0_raw = trunc(Int, xc)
        j0_raw = trunc(Int, yc)
        tx = xc - T(i0_raw)
        ty = yc - T(j0_raw)

        i0 = clamp(i0_raw, 1, Nx_c)
        i1 = clamp(i0_raw + 1, 1, Nx_c)
        j0 = clamp(j0_raw, 1, Ny_c)
        j1 = clamp(j0_raw + 1, 1, Ny_c)
        tx = clamp(tx, zero(T), one(T))
        ty = clamp(ty, zero(T), one(T))

        w00 = (one(T) - tx) * (one(T) - ty)
        w10 = tx * (one(T) - ty)
        w01 = (one(T) - tx) * ty
        w11 = tx * ty

        for q in 1:9
            g_fine[i_f, j_f, q] = w00 * g_coarse[i0, j0, q] +
                                   w10 * g_coarse[i1, j0, q] +
                                   w01 * g_coarse[i0, j1, q] +
                                   w11 * g_coarse[i1, j1, q]
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
                                        bc_thermal_patch_fns=nothing,
                                        bc_flow_patch_fns=nothing) where T
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

            # Flow ghost fill: temporal interpolation between *_prev (n) and f_in (n+1)
            fill_ghost_temporal!(patch, f_in, rho, ux, uy,
                                Float64(domain.base_omega), Nx, Ny, t_frac)
            # Thermal ghost: bilinear with temporal interpolation
            fill_thermal_ghost!(patch, thermal, g_in, Nx, Ny;
                                t_frac=t_frac)

            # Stream both f and g on patch
            stream_2d!(patch.f_out, patch.f_in, patch.Nx, patch.Ny)
            stream_2d!(thermal.g_out, thermal.g_in, patch.Nx, patch.Ny)

            # Flow BCs on patch (if patch edge touches a domain wall)
            if bc_flow_patch_fns !== nothing
                bc_f_fn = get(bc_flow_patch_fns, pidx, nothing)
                if bc_f_fn !== nothing
                    bc_f_fn(patch.f_out, patch.Nx, patch.Ny)
                end
            end

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
            # Collide flow with Boussinesq.
            # Body force scales with the local lattice spacing: in fine units
            # the acceleration must be multiplied by `ratio` so it produces
            # the same physical acceleration as on the coarse grid.
            collide_boussinesq_2d!(patch.f_out, thermal.Temp, patch.is_solid,
                                    patch.omega, T(β_g) * T(patch.ratio),
                                    T(T_ref_buoy))

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
