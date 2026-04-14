using KernelAbstractions

# ===========================================================================
# Ghost-layer exchange kernels for patch-based grid refinement
#
# Filippova-Hanel (1998) non-equilibrium rescaling at coarse-fine interfaces.
# Each kernel operates on thin ghost strips — lightweight vs. interior work.
#
# References:
# - Filippova & Hanel (1998) doi:10.1006/jcph.1998.6057
# - Dupuis & Chopard (2003) doi:10.1016/S0378-4371(03)00281-4
# ===========================================================================

# --- Inline equilibrium for use inside kernels ---

@inline function _feq_d2q9(q::Int, rho::T, ux::T, uy::T) where T
    # D2Q9 weights
    w = ifelse(q == 1, T(4)/T(9),
        ifelse(q <= 5, T(1)/T(9), T(1)/T(36)))
    # D2Q9 velocities
    cx = ifelse(q == 1, T(0),
         ifelse(q == 2, T(1),
         ifelse(q == 3, T(0),
         ifelse(q == 4, T(-1),
         ifelse(q == 5, T(0),
         ifelse(q == 6, T(1),
         ifelse(q == 7, T(-1),
         ifelse(q == 8, T(-1), T(1)))))))))
    cy = ifelse(q == 1, T(0),
         ifelse(q == 2, T(0),
         ifelse(q == 3, T(1),
         ifelse(q == 4, T(0),
         ifelse(q == 5, T(-1),
         ifelse(q == 6, T(1),
         ifelse(q == 7, T(1),
         ifelse(q == 8, T(-1), T(-1)))))))))
    cu = cx * ux + cy * uy
    usq = ux * ux + uy * uy
    return w * rho * (one(T) + T(3) * cu + T(4.5) * cu * cu - T(1.5) * usq)
end

# --- Prolongation: coarse → fine ghost fill with rescaling ---

"""
Kernel: fill fine-grid ghost cells from coarse-grid data using bilinear
interpolation of macroscopic fields + Filippova-Hanel f_neq rescaling.

Operates on the FULL fine grid (Nx_fine × Ny_fine) but only writes to ghost
cells (i_f <= n_ghost or i_f > n_ghost + Nx_inner, same for j).
Interior cells are left untouched.

Arguments:
- `f_fine`: fine grid distributions [Nx_f, Ny_f, 9]
- `f_coarse`: coarse grid distributions [Nx_c, Ny_c, 9] (or local buffer)
- `rho_c, ux_c, uy_c`: coarse macroscopic fields
- `ratio`: refinement ratio
- `Nx_inner, Ny_inner`: fine inner grid dimensions (without ghosts)
- `n_ghost`: ghost width
- `i_c_start, j_c_start`: coarse indices where this patch starts (1-based)
- `Nx_c, Ny_c`: coarse grid dimensions (for clamping)
- `alpha_c2f`: non-equilibrium rescaling factor
"""
@kernel function prolongate_f_rescaled_2d_kernel!(
        f_fine, @Const(f_coarse), @Const(rho_c), @Const(ux_c), @Const(uy_c),
        ratio, Nx_inner, Ny_inner, n_ghost,
        i_c_start, j_c_start, Nx_c, Ny_c, alpha_c2f)

    i_f, j_f = @index(Global, NTuple)

    @inbounds begin
        T = eltype(f_fine)
        Nx_f = Nx_inner + 2 * n_ghost
        Ny_f = Ny_inner + 2 * n_ghost

        # Only process ghost cells (skip interior)
        is_ghost = (i_f <= n_ghost) || (i_f > n_ghost + Nx_inner) ||
                   (j_f <= n_ghost) || (j_f > n_ghost + Ny_inner)

        if is_ghost
            # Fine cell center in coarse continuous-index space
            # Fine inner cell 1 maps to coarse cell i_c_start, etc.
            # Ghost cells extend beyond the covered coarse region
            xc = T(i_f - n_ghost - 1) / T(ratio) + T(i_c_start) + T(0.5) / T(ratio)
            yc = T(j_f - n_ghost - 1) / T(ratio) + T(j_c_start) + T(0.5) / T(ratio)

            # Bilinear stencil: floor indices
            i0_raw = unsafe_trunc(Int, xc)
            j0_raw = unsafe_trunc(Int, yc)
            tx = xc - T(i0_raw)
            ty = yc - T(j0_raw)

            # Clamp to valid coarse range
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

            # Interpolate macroscopic fields
            rho_f = w00 * rho_c[i0, j0] + w10 * rho_c[i1, j0] +
                    w01 * rho_c[i0, j1] + w11 * rho_c[i1, j1]
            ux_f  = w00 * ux_c[i0, j0] + w10 * ux_c[i1, j0] +
                    w01 * ux_c[i0, j1] + w11 * ux_c[i1, j1]
            uy_f  = w00 * uy_c[i0, j0] + w10 * uy_c[i1, j0] +
                    w01 * uy_c[i0, j1] + w11 * uy_c[i1, j1]

            # For each population: interpolate f_neq and assemble
            for q in 1:9
                feq_f = _feq_d2q9(q, rho_f, ux_f, uy_f)

                # Compute f_neq at each coarse stencil node
                fneq_00 = f_coarse[i0, j0, q] - _feq_d2q9(q, rho_c[i0, j0], ux_c[i0, j0], uy_c[i0, j0])
                fneq_10 = f_coarse[i1, j0, q] - _feq_d2q9(q, rho_c[i1, j0], ux_c[i1, j0], uy_c[i1, j0])
                fneq_01 = f_coarse[i0, j1, q] - _feq_d2q9(q, rho_c[i0, j1], ux_c[i0, j1], uy_c[i0, j1])
                fneq_11 = f_coarse[i1, j1, q] - _feq_d2q9(q, rho_c[i1, j1], ux_c[i1, j1], uy_c[i1, j1])

                fneq_interp = w00 * fneq_00 + w10 * fneq_10 +
                              w01 * fneq_01 + w11 * fneq_11

                f_fine[i_f, j_f, q] = feq_f + T(alpha_c2f) * fneq_interp
            end
        end
    end
end

"""
    prolongate_f_rescaled_2d!(f_fine, f_coarse, rho_c, ux_c, uy_c, patch)

Fill ghost cells of fine patch from coarse grid data with Filippova-Hanel
non-equilibrium rescaling.
"""
function prolongate_f_rescaled_2d!(f_fine, f_coarse, rho_c, ux_c, uy_c,
                                   ratio::Int, Nx_inner::Int, Ny_inner::Int,
                                   n_ghost::Int, i_c_start::Int, j_c_start::Int,
                                   Nx_c::Int, Ny_c::Int,
                                   omega_coarse::Real, omega_fine::Real)
    backend = KernelAbstractions.get_backend(f_fine)
    T = eltype(f_fine)
    alpha = T(rescaling_factor_c2f(omega_coarse, omega_fine, ratio))
    Nx_f = Nx_inner + 2 * n_ghost
    Ny_f = Ny_inner + 2 * n_ghost
    kernel! = prolongate_f_rescaled_2d_kernel!(backend)
    kernel!(f_fine, f_coarse, rho_c, ux_c, uy_c,
            ratio, Nx_inner, Ny_inner, n_ghost,
            i_c_start, j_c_start, Nx_c, Ny_c, alpha;
            ndrange=(Nx_f, Ny_f))
end

# --- Full prolongation (interior + ghost) — used once at init ---

@kernel function prolongate_f_rescaled_full_2d_kernel!(
        f_fine, @Const(f_coarse), @Const(rho_c), @Const(ux_c), @Const(uy_c),
        ratio, Nx_inner, Ny_inner, n_ghost,
        i_c_start, j_c_start, Nx_c, Ny_c, alpha_c2f)

    i_f, j_f = @index(Global, NTuple)

    @inbounds begin
        T = eltype(f_fine)

        xc = T(i_f - n_ghost - 1) / T(ratio) + T(i_c_start) + T(0.5) / T(ratio)
        yc = T(j_f - n_ghost - 1) / T(ratio) + T(j_c_start) + T(0.5) / T(ratio)

        i0_raw = unsafe_trunc(Int, xc)
        j0_raw = unsafe_trunc(Int, yc)
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

        rho_f = w00 * rho_c[i0, j0] + w10 * rho_c[i1, j0] +
                w01 * rho_c[i0, j1] + w11 * rho_c[i1, j1]
        ux_f  = w00 * ux_c[i0, j0] + w10 * ux_c[i1, j0] +
                w01 * ux_c[i0, j1] + w11 * ux_c[i1, j1]
        uy_f  = w00 * uy_c[i0, j0] + w10 * uy_c[i1, j0] +
                w01 * uy_c[i0, j1] + w11 * uy_c[i1, j1]

        for q in 1:9
            feq_f = _feq_d2q9(q, rho_f, ux_f, uy_f)
            fneq_00 = f_coarse[i0, j0, q] - _feq_d2q9(q, rho_c[i0, j0], ux_c[i0, j0], uy_c[i0, j0])
            fneq_10 = f_coarse[i1, j0, q] - _feq_d2q9(q, rho_c[i1, j0], ux_c[i1, j0], uy_c[i1, j0])
            fneq_01 = f_coarse[i0, j1, q] - _feq_d2q9(q, rho_c[i0, j1], ux_c[i0, j1], uy_c[i0, j1])
            fneq_11 = f_coarse[i1, j1, q] - _feq_d2q9(q, rho_c[i1, j1], ux_c[i1, j1], uy_c[i1, j1])
            fneq_interp = w00 * fneq_00 + w10 * fneq_10 + w01 * fneq_01 + w11 * fneq_11
            f_fine[i_f, j_f, q] = feq_f + T(alpha_c2f) * fneq_interp
        end
    end
end

"""
    prolongate_f_rescaled_full_2d!(f_fine, f_coarse, rho_c, ux_c, uy_c, ...)

Like `prolongate_f_rescaled_2d!` but fills BOTH interior and ghost cells.
Used to initialize a fresh patch from the coarse state.
"""
function prolongate_f_rescaled_full_2d!(f_fine, f_coarse, rho_c, ux_c, uy_c,
                                        ratio::Int, Nx_inner::Int, Ny_inner::Int,
                                        n_ghost::Int, i_c_start::Int, j_c_start::Int,
                                        Nx_c::Int, Ny_c::Int,
                                        omega_coarse::Real, omega_fine::Real)
    backend = KernelAbstractions.get_backend(f_fine)
    T = eltype(f_fine)
    alpha = T(rescaling_factor_c2f(omega_coarse, omega_fine, ratio))
    Nx_f = Nx_inner + 2 * n_ghost
    Ny_f = Ny_inner + 2 * n_ghost
    kernel! = prolongate_f_rescaled_full_2d_kernel!(backend)
    kernel!(f_fine, f_coarse, rho_c, ux_c, uy_c,
            ratio, Nx_inner, Ny_inner, n_ghost,
            i_c_start, j_c_start, Nx_c, Ny_c, alpha;
            ndrange=(Nx_f, Ny_f))
end

# --- Temporal prolongation: ghost fill with time-interpolated coarse state ---

"""
Kernel: fill fine-grid ghost cells from temporally interpolated coarse data.
Blends between `f_prev` (local buffer, saved at time n) and `f_curr` (global,
at time n+1) using weight `t_frac`. Then applies bilinear spatial interpolation
and Filippova-Hanel non-equilibrium rescaling.

At t_frac=0, reads purely from f_prev (correct state at time n).
At t_frac>0, blends toward current state f_curr.
"""
@kernel function prolongate_f_rescaled_temporal_2d_kernel!(
        f_fine, @Const(f_curr), @Const(rho_curr), @Const(ux_curr), @Const(uy_curr),
        @Const(f_prev), @Const(rho_prev), @Const(ux_prev), @Const(uy_prev),
        ratio, Nx_inner, Ny_inner, n_ghost,
        i_c_start, j_c_start, Nx_c, Ny_c, alpha_c2f, t_frac,
        i_lo, j_lo, Ni_prev, Nj_prev)

    i_f, j_f = @index(Global, NTuple)

    @inbounds begin
        T = eltype(f_fine)
        Nx_f = Nx_inner + 2 * n_ghost
        Ny_f = Ny_inner + 2 * n_ghost

        is_ghost = (i_f <= n_ghost) || (i_f > n_ghost + Nx_inner) ||
                   (j_f <= n_ghost) || (j_f > n_ghost + Ny_inner)

        if is_ghost
            t = T(t_frac)
            omt = one(T) - t  # weight for prev state

            xc = T(i_f - n_ghost - 1) / T(ratio) + T(i_c_start) + T(0.5) / T(ratio)
            yc = T(j_f - n_ghost - 1) / T(ratio) + T(j_c_start) + T(0.5) / T(ratio)

            i0_raw = unsafe_trunc(Int, xc)
            j0_raw = unsafe_trunc(Int, yc)
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

            # Local indices into prev buffers
            ip0 = clamp(i0 - i_lo + 1, 1, Ni_prev)
            ip1 = clamp(i1 - i_lo + 1, 1, Ni_prev)
            jp0 = clamp(j0 - j_lo + 1, 1, Nj_prev)
            jp1 = clamp(j1 - j_lo + 1, 1, Nj_prev)

            # Temporally blend macroscopic fields at stencil points
            rho00 = omt * rho_prev[ip0, jp0] + t * rho_curr[i0, j0]
            rho10 = omt * rho_prev[ip1, jp0] + t * rho_curr[i1, j0]
            rho01 = omt * rho_prev[ip0, jp1] + t * rho_curr[i0, j1]
            rho11 = omt * rho_prev[ip1, jp1] + t * rho_curr[i1, j1]

            ux00 = omt * ux_prev[ip0, jp0] + t * ux_curr[i0, j0]
            ux10 = omt * ux_prev[ip1, jp0] + t * ux_curr[i1, j0]
            ux01 = omt * ux_prev[ip0, jp1] + t * ux_curr[i0, j1]
            ux11 = omt * ux_prev[ip1, jp1] + t * ux_curr[i1, j1]

            uy00 = omt * uy_prev[ip0, jp0] + t * uy_curr[i0, j0]
            uy10 = omt * uy_prev[ip1, jp0] + t * uy_curr[i1, j0]
            uy01 = omt * uy_prev[ip0, jp1] + t * uy_curr[i0, j1]
            uy11 = omt * uy_prev[ip1, jp1] + t * uy_curr[i1, j1]

            # Spatial bilinear interpolation of macroscopic
            rho_f = w00 * rho00 + w10 * rho10 + w01 * rho01 + w11 * rho11
            ux_f  = w00 * ux00  + w10 * ux10  + w01 * ux01  + w11 * ux11
            uy_f  = w00 * uy00  + w10 * uy10  + w01 * uy01  + w11 * uy11

            for q in 1:9
                feq_f = _feq_d2q9(q, rho_f, ux_f, uy_f)

                # Temporal blend of f at stencil points, then compute f_neq
                f00 = omt * f_prev[ip0, jp0, q] + t * f_curr[i0, j0, q]
                f10 = omt * f_prev[ip1, jp0, q] + t * f_curr[i1, j0, q]
                f01 = omt * f_prev[ip0, jp1, q] + t * f_curr[i0, j1, q]
                f11 = omt * f_prev[ip1, jp1, q] + t * f_curr[i1, j1, q]

                fneq_00 = f00 - _feq_d2q9(q, rho00, ux00, uy00)
                fneq_10 = f10 - _feq_d2q9(q, rho10, ux10, uy10)
                fneq_01 = f01 - _feq_d2q9(q, rho01, ux01, uy01)
                fneq_11 = f11 - _feq_d2q9(q, rho11, ux11, uy11)

                fneq_interp = w00 * fneq_00 + w10 * fneq_10 +
                              w01 * fneq_01 + w11 * fneq_11

                f_fine[i_f, j_f, q] = feq_f + T(alpha_c2f) * fneq_interp
            end
        end
    end
end

"""
    prolongate_f_rescaled_temporal_2d!(f_fine, f_curr, rho_curr, ux_curr, uy_curr,
                                       f_prev, rho_prev, ux_prev, uy_prev, ...)

Fill ghost cells with temporally interpolated Filippova-Hanel rescaling.
At t_frac=0, reads from `*_prev` buffers (time n). At t_frac>0, blends
toward current state `*_curr` (time n+1).
"""
function prolongate_f_rescaled_temporal_2d!(
        f_fine, f_curr, rho_curr, ux_curr, uy_curr,
        f_prev, rho_prev, ux_prev, uy_prev,
        ratio::Int, Nx_inner::Int, Ny_inner::Int,
        n_ghost::Int, i_c_start::Int, j_c_start::Int,
        Nx_c::Int, Ny_c::Int,
        omega_coarse::Real, omega_fine::Real, t_frac::Real)
    backend = KernelAbstractions.get_backend(f_fine)
    T = eltype(f_fine)
    alpha = T(rescaling_factor_c2f(omega_coarse, omega_fine, ratio))
    Nx_f = Nx_inner + 2 * n_ghost
    Ny_f = Ny_inner + 2 * n_ghost
    i_lo = max(i_c_start - 1, 1)
    j_lo = max(j_c_start - 1, 1)
    # Compute actual filled size (matching save_coarse_state! logic).
    # Patches at domain boundaries have a shorter margin, so the filled
    # region can be smaller than the allocated buffer.
    i_c_end = i_c_start + Nx_inner ÷ ratio - 1
    j_c_end = j_c_start + Ny_inner ÷ ratio - 1
    Ni_prev = min(i_c_end + 1, Nx_c) - i_lo + 1
    Nj_prev = min(j_c_end + 1, Ny_c) - j_lo + 1
    kernel! = prolongate_f_rescaled_temporal_2d_kernel!(backend)
    kernel!(f_fine, f_curr, rho_curr, ux_curr, uy_curr,
            f_prev, rho_prev, ux_prev, uy_prev,
            ratio, Nx_inner, Ny_inner, n_ghost,
            i_c_start, j_c_start, Nx_c, Ny_c, alpha, T(t_frac),
            i_lo, j_lo, Ni_prev, Nj_prev;
            ndrange=(Nx_f, Ny_f))
end

# --- Restriction: fine → coarse with inverse rescaling ---

"""
Kernel: block-average fine interior cells back to coarse overlap region
with inverse Filippova-Hanel rescaling.

Each coarse cell (ic, jc) collects the mean of its ratio×ratio fine children.
Only the overlap region is written (not the full coarse grid).

Arguments:
- `f_coarse`: coarse distributions [Nx_c, Ny_c, 9] — modified in overlap
- `rho_c, ux_c, uy_c`: coarse macroscopic — updated in overlap
- `f_fine`: fine distributions [Nx_f, Ny_f, 9]
- `rho_f, ux_f, uy_f`: fine macroscopic fields
- `ratio`: refinement ratio
- `n_ghost`: fine grid ghost width
- `i_c_start, j_c_start`: coarse indices for the start of the overlap
- `Nx_overlap, Ny_overlap`: size of the overlap in coarse cells
- `alpha_f2c`: non-equilibrium rescaling factor (fine→coarse)
"""
@kernel function restrict_f_rescaled_2d_kernel!(
        f_coarse, rho_c, ux_c, uy_c,
        @Const(f_fine), @Const(rho_f), @Const(ux_f), @Const(uy_f),
        ratio, n_ghost, i_c_start, j_c_start,
        Nx_overlap, Ny_overlap, alpha_f2c)

    ic_local, jc_local = @index(Global, NTuple)

    @inbounds begin
        T = eltype(f_coarse)
        inv_r2 = one(T) / T(ratio * ratio)

        # Global coarse index
        ic = i_c_start + ic_local - 1
        jc = j_c_start + jc_local - 1

        # Block-average macroscopic from fine children
        rho_avg = zero(T)
        ux_avg = zero(T)
        uy_avg = zero(T)
        for dj in 0:(ratio - 1)
            j_f = n_ghost + (jc_local - 1) * ratio + 1 + dj
            for di in 0:(ratio - 1)
                i_f = n_ghost + (ic_local - 1) * ratio + 1 + di
                rho_avg += rho_f[i_f, j_f]
                ux_avg += ux_f[i_f, j_f]
                uy_avg += uy_f[i_f, j_f]
            end
        end
        rho_avg *= inv_r2
        ux_avg *= inv_r2
        uy_avg *= inv_r2

        # Update coarse macroscopic
        rho_c[ic, jc] = rho_avg
        ux_c[ic, jc] = ux_avg
        uy_c[ic, jc] = uy_avg

        # Block-average f_neq from fine and rescale
        for q in 1:9
            feq_c = _feq_d2q9(q, rho_avg, ux_avg, uy_avg)

            fneq_avg = zero(T)
            for dj in 0:(ratio - 1)
                j_f = n_ghost + (jc_local - 1) * ratio + 1 + dj
                for di in 0:(ratio - 1)
                    i_f = n_ghost + (ic_local - 1) * ratio + 1 + di
                    fneq_avg += f_fine[i_f, j_f, q] -
                        _feq_d2q9(q, rho_f[i_f, j_f], ux_f[i_f, j_f], uy_f[i_f, j_f])
                end
            end
            fneq_avg *= inv_r2

            f_coarse[ic, jc, q] = feq_c + T(alpha_f2c) * fneq_avg
        end
    end
end

"""
    restrict_f_rescaled_2d!(f_coarse, rho_c, ux_c, uy_c,
                            f_fine, rho_f, ux_f, uy_f, patch)

Block-average fine interior back to coarse overlap with inverse rescaling.
"""
function restrict_f_rescaled_2d!(f_coarse, rho_c, ux_c, uy_c,
                                 f_fine, rho_f, ux_f, uy_f,
                                 ratio::Int, n_ghost::Int,
                                 i_c_start::Int, j_c_start::Int,
                                 Nx_overlap::Int, Ny_overlap::Int,
                                 omega_coarse::Real, omega_fine::Real)
    backend = KernelAbstractions.get_backend(f_coarse)
    T = eltype(f_coarse)
    alpha = T(rescaling_factor_f2c(omega_coarse, omega_fine, ratio))
    kernel! = restrict_f_rescaled_2d_kernel!(backend)
    kernel!(f_coarse, rho_c, ux_c, uy_c,
            f_fine, rho_f, ux_f, uy_f,
            ratio, n_ghost, i_c_start, j_c_start,
            Nx_overlap, Ny_overlap, alpha;
            ndrange=(Nx_overlap, Ny_overlap))
end

# --- Temporal interpolation of coarse data ---

"""
Kernel: linearly interpolate coarse fields between time levels n and n+1.

    field_interp = (1 - t_frac) * field_prev + t_frac * field_curr
"""
@kernel function temporal_interpolate_2d_kernel!(
        rho_interp, ux_interp, uy_interp, f_interp,
        @Const(rho_prev), @Const(ux_prev), @Const(uy_prev), @Const(f_prev),
        @Const(rho_curr), @Const(ux_curr), @Const(uy_curr), @Const(f_curr),
        t_frac)

    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(rho_interp)
        t = T(t_frac)
        omt = one(T) - t

        rho_interp[i, j] = omt * rho_prev[i, j] + t * rho_curr[i, j]
        ux_interp[i, j]  = omt * ux_prev[i, j]  + t * ux_curr[i, j]
        uy_interp[i, j]  = omt * uy_prev[i, j]  + t * uy_curr[i, j]

        for q in 1:9
            f_interp[i, j, q] = omt * f_prev[i, j, q] + t * f_curr[i, j, q]
        end
    end
end

"""
    temporal_interpolate_2d!(dst_rho, dst_ux, dst_uy, dst_f,
                             prev_rho, prev_ux, prev_uy, prev_f,
                             curr_rho, curr_ux, curr_uy, curr_f,
                             t_frac)

Linearly interpolate coarse data between two time levels.
`t_frac = 0` gives prev, `t_frac = 1` gives curr.
"""
function temporal_interpolate_2d!(rho_i, ux_i, uy_i, f_i,
                                  rho_p, ux_p, uy_p, f_p,
                                  rho_c, ux_c, uy_c, f_c,
                                  t_frac::Real)
    backend = KernelAbstractions.get_backend(rho_i)
    Ni, Nj = size(rho_i)
    kernel! = temporal_interpolate_2d_kernel!(backend)
    kernel!(rho_i, ux_i, uy_i, f_i,
            rho_p, ux_p, uy_p, f_p,
            rho_c, ux_c, uy_c, f_c,
            t_frac; ndrange=(Ni, Nj))
end

# --- Copy macroscopic overlap for VTK output ---

"""
Kernel: block-average fine macroscopic fields to coarse overlap for output.
"""
@kernel function copy_macroscopic_overlap_2d_kernel!(
        rho_c, ux_c, uy_c,
        @Const(rho_f), @Const(ux_f), @Const(uy_f),
        ratio, n_ghost, i_c_start, j_c_start)

    ic_local, jc_local = @index(Global, NTuple)

    @inbounds begin
        T = eltype(rho_c)
        inv_r2 = one(T) / T(ratio * ratio)
        ic = i_c_start + ic_local - 1
        jc = j_c_start + jc_local - 1

        rho_avg = zero(T)
        ux_avg = zero(T)
        uy_avg = zero(T)
        for dj in 0:(ratio - 1)
            j_f = n_ghost + (jc_local - 1) * ratio + 1 + dj
            for di in 0:(ratio - 1)
                i_f = n_ghost + (ic_local - 1) * ratio + 1 + di
                rho_avg += rho_f[i_f, j_f]
                ux_avg += ux_f[i_f, j_f]
                uy_avg += uy_f[i_f, j_f]
            end
        end
        rho_c[ic, jc] = rho_avg * inv_r2
        ux_c[ic, jc] = ux_avg * inv_r2
        uy_c[ic, jc] = uy_avg * inv_r2
    end
end

"""
    copy_macroscopic_overlap_2d!(rho_c, ux_c, uy_c, rho_f, ux_f, uy_f, patch)

Block-average fine macroscopic fields back to coarse for visualization.
"""
function copy_macroscopic_overlap_2d!(rho_c, ux_c, uy_c,
                                      rho_f, ux_f, uy_f,
                                      ratio::Int, n_ghost::Int,
                                      i_c_start::Int, j_c_start::Int,
                                      Nx_overlap::Int, Ny_overlap::Int)
    backend = KernelAbstractions.get_backend(rho_c)
    kernel! = copy_macroscopic_overlap_2d_kernel!(backend)
    kernel!(rho_c, ux_c, uy_c, rho_f, ux_f, uy_f,
            ratio, n_ghost, i_c_start, j_c_start;
            ndrange=(Nx_overlap, Ny_overlap))
end
