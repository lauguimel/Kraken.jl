# ===========================================================================
# Multi-level time stepping with temporal sub-cycling
#
# Algorithm (for each coarse time step):
#   1. Save coarse state at time n (for temporal interpolation)
#   2. Advance coarse grid: stream -> BC -> collide -> macro -> swap
#   3. For sub_step = 1:ratio:
#      a. Compute t_frac for temporal interpolation
#      b. Fill fine ghost layers (prolongate with rescaling)
#      c. Stream fine patch (existing kernel)
#      d. Collide fine patch (existing kernel, omega_fine)
#      e. Macro fine patch
#      f. Swap fine
#   4. Restrict fine -> coarse in overlap region
#
# References:
# - Dupuis & Chopard (2003) doi:10.1016/S0378-4371(03)00281-4
# ===========================================================================

"""
    save_coarse_state!(patch, f_coarse, rho_c, ux_c, uy_c)

Copy the coarse-grid state covering this patch (+ bilinear margin) into the
patch's temporal interpolation buffers (rho_prev, ux_prev, uy_prev, f_prev).
"""
function save_coarse_state!(patch::RefinementPatch{T},
                            f_coarse, rho_c, ux_c, uy_c) where T
    i_range = patch.parent_i_range
    j_range = patch.parent_j_range
    Nx_c = size(rho_c, 1)
    Ny_c = size(rho_c, 2)

    # Extract region with 1-cell margin for bilinear interpolation
    i_lo = max(first(i_range) - 1, 1)
    i_hi = min(last(i_range) + 1, Nx_c)
    j_lo = max(first(j_range) - 1, 1)
    j_hi = min(last(j_range) + 1, Ny_c)

    n_i = i_hi - i_lo + 1
    n_j = j_hi - j_lo + 1

    # Copy from coarse arrays to patch buffers (GPU-to-GPU)
    src_rho = @view rho_c[i_lo:i_hi, j_lo:j_hi]
    src_ux  = @view ux_c[i_lo:i_hi, j_lo:j_hi]
    src_uy  = @view uy_c[i_lo:i_hi, j_lo:j_hi]
    src_f   = @view f_coarse[i_lo:i_hi, j_lo:j_hi, :]

    dst_rho = @view patch.rho_prev[1:n_i, 1:n_j]
    dst_ux  = @view patch.ux_prev[1:n_i, 1:n_j]
    dst_uy  = @view patch.uy_prev[1:n_i, 1:n_j]
    dst_f   = @view patch.f_prev[1:n_i, 1:n_j, :]

    copyto!(dst_rho, src_rho)
    copyto!(dst_ux, src_ux)
    copyto!(dst_uy, src_uy)
    copyto!(dst_f, src_f)
end

"""
    advance_patch!(patch; stream_fn, collide_fn, macro_fn, bc_fn)

Advance a single refinement patch by one fine timestep using existing kernels.

# Arguments
- `stream_fn`: streaming kernel (e.g., `stream_2d!`)
- `collide_fn`: collision closure `(f_out, is_solid) -> ...`
- `macro_fn`: macroscopic closure `(rho, ux, uy, f_out) -> ...`
- `bc_fn`: optional boundary condition closure, or `nothing`
"""
function advance_patch!(patch::RefinementPatch{T};
                        stream_fn,
                        collide_fn,
                        macro_fn,
                        bc_fn=nothing) where T
    Nx, Ny = patch.Nx, patch.Ny

    # Stream
    stream_fn(patch.f_out, patch.f_in, Nx, Ny)

    # Boundary conditions (if patch edge coincides with domain boundary)
    if bc_fn !== nothing
        bc_fn(patch.f_out, Nx, Ny)
    end

    # Collide
    collide_fn(patch.f_out, patch.is_solid)

    # Macroscopic
    macro_fn(patch.rho, patch.ux, patch.uy, patch.f_out)
end

"""
    fill_ghost_from_coarse!(patch, f_coarse, rho_c, ux_c, uy_c,
                            omega_coarse, Nx_c, Ny_c)

Fill the ghost cells of a fine patch using coarse-grid data with
Filippova-Hanel rescaling.
"""
function fill_ghost_from_coarse!(patch::RefinementPatch{T},
                                 f_coarse, rho_c, ux_c, uy_c,
                                 omega_coarse::Real,
                                 Nx_c::Int, Ny_c::Int) where T
    prolongate_f_rescaled_2d!(
        patch.f_in, f_coarse, rho_c, ux_c, uy_c,
        patch.ratio, patch.Nx_inner, patch.Ny_inner,
        patch.n_ghost, first(patch.parent_i_range), first(patch.parent_j_range),
        Nx_c, Ny_c, omega_coarse, Float64(patch.omega)
    )
end

"""
    restrict_to_coarse!(patch, f_coarse, rho_c, ux_c, uy_c, omega_coarse)

Restrict fine-grid interior back to coarse overlap region with inverse
Filippova-Hanel rescaling.
"""
function restrict_to_coarse!(patch::RefinementPatch{T},
                             f_coarse, rho_c, ux_c, uy_c,
                             omega_coarse::Real) where T
    Nx_overlap = length(patch.parent_i_range)
    Ny_overlap = length(patch.parent_j_range)
    restrict_f_rescaled_2d!(
        f_coarse, rho_c, ux_c, uy_c,
        patch.f_in, patch.rho, patch.ux, patch.uy,
        patch.ratio, patch.n_ghost,
        first(patch.parent_i_range), first(patch.parent_j_range),
        Nx_overlap, Ny_overlap,
        omega_coarse, Float64(patch.omega)
    )
end

"""
    advance_refined_step!(domain, f_in, f_out, rho, ux, uy, is_solid;
                          stream_fn, collide_fn, macro_fn, bc_base_fn,
                          bc_patch_fns)

Advance the full refined domain by one coarse timestep with sub-cycling.

# Algorithm
1. Save coarse state at time n for each patch
2. Advance coarse grid one step
3. For each patch, sub-cycle `ratio` fine steps with temporal interpolation
4. Restrict fine results back to coarse overlap

Returns updated (f_in, f_out) for the base grid (swapped).
"""
function advance_refined_step!(domain::RefinedDomain{T},
                               f_in, f_out, rho, ux, uy, is_solid;
                               stream_fn,
                               collide_fn,
                               macro_fn,
                               bc_base_fn=nothing,
                               bc_patch_fns=nothing) where T
    Nx = domain.base_Nx
    Ny = domain.base_Ny

    # 1. Save coarse state at time n for all patches
    for patch in domain.patches
        save_coarse_state!(patch, f_in, rho, ux, uy)
    end

    # 2. Advance coarse grid one step
    stream_fn(f_out, f_in, Nx, Ny)
    if bc_base_fn !== nothing
        bc_base_fn(f_out)
    end
    collide_fn(f_out, is_solid)
    macro_fn(rho, ux, uy, f_out)
    f_in, f_out = f_out, f_in

    # 3. Sub-cycle each patch
    for (pidx, patch) in enumerate(domain.patches)
        ratio = patch.ratio

        for sub_step in 1:ratio
            # Temporal interpolation: sub_step=1 -> t_frac=0, sub_step=ratio -> (ratio-1)/ratio
            t_frac = T((sub_step - 1) / ratio)

            if t_frac > zero(T)
                _fill_ghost_interpolated!(patch, f_in, rho, ux, uy,
                                         Float64(domain.base_omega),
                                         Nx, Ny, t_frac)
            else
                # First sub-step: use coarse state at time n
                fill_ghost_from_coarse!(patch, f_in, rho, ux, uy,
                                        Float64(domain.base_omega), Nx, Ny)
            end

            # Advance patch one fine step
            bc_fn = bc_patch_fns !== nothing ? get(bc_patch_fns, pidx, nothing) : nothing
            advance_patch!(patch;
                          stream_fn=stream_fn,
                          collide_fn=(f, is_s) -> collide_2d!(f, is_s, patch.omega),
                          macro_fn=compute_macroscopic_2d!,
                          bc_fn=bc_fn)

            # Copy f_out -> f_in for next sub-step
            copyto!(patch.f_in, patch.f_out)
        end
    end

    # 4. Restrict fine results back to coarse
    for patch in domain.patches
        restrict_to_coarse!(patch, f_in, rho, ux, uy,
                           Float64(domain.base_omega))
    end

    return f_in, f_out
end

"""
    _fill_ghost_interpolated!(patch, f_curr, rho_curr, ux_curr, uy_curr,
                              omega_coarse, Nx_c, Ny_c, t_frac)

Fill ghost cells using temporally interpolated coarse data.
"""
function _fill_ghost_interpolated!(patch::RefinementPatch{T},
                                   f_curr, rho_curr, ux_curr, uy_curr,
                                   omega_coarse::Float64,
                                   Nx_c::Int, Ny_c::Int,
                                   t_frac::Real) where T
    i_range = patch.parent_i_range
    j_range = patch.parent_j_range

    i_lo = max(first(i_range) - 1, 1)
    i_hi = min(last(i_range) + 1, Nx_c)
    j_lo = max(first(j_range) - 1, 1)
    j_hi = min(last(j_range) + 1, Ny_c)
    n_i = i_hi - i_lo + 1
    n_j = j_hi - j_lo + 1

    # Extract current coarse region
    src_rho = @view rho_curr[i_lo:i_hi, j_lo:j_hi]
    src_ux  = @view ux_curr[i_lo:i_hi, j_lo:j_hi]
    src_uy  = @view uy_curr[i_lo:i_hi, j_lo:j_hi]
    src_f   = @view f_curr[i_lo:i_hi, j_lo:j_hi, :]

    # Temporally interpolate prev/curr into prev buffers (in-place reuse)
    prev_rho = @view patch.rho_prev[1:n_i, 1:n_j]
    prev_ux  = @view patch.ux_prev[1:n_i, 1:n_j]
    prev_uy  = @view patch.uy_prev[1:n_i, 1:n_j]
    prev_f   = @view patch.f_prev[1:n_i, 1:n_j, :]

    temporal_interpolate_2d!(prev_rho, prev_ux, prev_uy, prev_f,
                             patch.rho_prev, patch.ux_prev, patch.uy_prev, patch.f_prev,
                             src_rho, src_ux, src_uy, src_f,
                             t_frac)

    # Fill ghost from interpolated data
    fill_ghost_from_coarse!(patch, f_curr, rho_curr, ux_curr, uy_curr,
                           omega_coarse, Nx_c, Ny_c)
end
