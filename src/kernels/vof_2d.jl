using KernelAbstractions

# ===========================================================================
# VOF PLIC 2D — Sharp interface tracking for LBM
#
# Volume fraction C ∈ [0,1]: C=1 liquid, C=0 gas
# Interface reconstructed as a line n·x = d in each mixed cell (0 < C < 1)
#
# References:
# - Scardovelli & Zaleski (1999) doi:10.1146/annurev.fluid.31.1.567
# - Rider & Kothe (1998) doi:10.1006/jcph.1998.6029
# ===========================================================================

# ===================================================================
# Part 1: Interface normals (MYC — Mixed Youngs-Centered, GPU kernel)
# ===================================================================
#
# Exact port of Basilisk myc2d.h: compute both central and Youngs
# normals, keep the one with the largest min-component ratio (better
# aligned with the interface). Convention: n = -∇C (outward from
# liquid, matching Basilisk geometry.h).
#
# Reference: Aulisa et al. (2007) doi:10.1016/j.jcp.2007.03.015

@kernel function compute_vof_normal_2d_kernel!(nx, ny, @Const(C), Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(C)
        ip = ifelse(i < Nx, i + 1, 1)
        im = ifelse(i > 1,  i - 1, Nx)
        jp = ifelse(j < Ny, j + 1, 1)
        jm = ifelse(j > 1,  j - 1, Ny)

        # --- Central differences (3-cell sums) ---
        c_l = C[im,jm] + C[im,j] + C[im,jp]   # left column
        c_r = C[ip,jm] + C[ip,j] + C[ip,jp]   # right column
        c_b = C[im,jm] + C[i,jm] + C[ip,jm]   # bottom row
        c_t = C[im,jp] + C[i,jp] + C[ip,jp]   # top row

        # Central normal (sign: -∇C = left-right, bottom-top)
        mx0 = (c_l - c_r) / T(2)
        my0 = (c_b - c_t) / T(2)

        # Determine which component is smaller (ix=1 → |mx|≤|my|)
        ix = abs(mx0) <= abs(my0)
        # Clamp the dominant component to ±1
        mx0 = ifelse(ix, mx0, ifelse(mx0 > zero(T), one(T), -one(T)))
        my0 = ifelse(ix, ifelse(my0 > zero(T), one(T), -one(T)), my0)

        # --- Youngs stencil (weighted 1-2-1) ---
        mm1 = C[im,jm] + T(2)*C[im,j] + C[im,jp]
        mm2 = C[ip,jm] + T(2)*C[ip,j] + C[ip,jp]
        mx1 = (mm1 - mm2) + T(1e-30)

        mm1 = C[im,jm] + T(2)*C[i,jm] + C[ip,jm]
        mm2 = C[im,jp] + T(2)*C[i,jp] + C[ip,jp]
        my1 = (mm1 - mm2) + T(1e-30)

        # --- Pick the better normal ---
        # For the smaller-component axis (ix), compare the ratio
        # |minor/major| — larger ratio means better alignment
        if ix
            # |mx| was smaller: compare |mx/my| for central vs Youngs
            ratio_y = abs(mx1) / abs(my1)
            mx_out = ifelse(ratio_y > abs(mx0), mx1, mx0)
            my_out = ifelse(ratio_y > abs(mx0), my1, my0)
        else
            # |my| was smaller: compare |my/mx| for central vs Youngs
            ratio_y = abs(my1) / abs(mx1)
            mx_out = ifelse(ratio_y > abs(my0), mx1, mx0)
            my_out = ifelse(ratio_y > abs(my0), my1, my0)
        end

        # Normalize: |mx| + |my| = 1 (L1 norm, matching Basilisk)
        mm = abs(mx_out) + abs(my_out)
        nx[i,j] = mx_out / mm
        ny[i,j] = my_out / mm
    end
end

function compute_vof_normal_2d!(nx, ny, C, Nx, Ny)
    backend = KernelAbstractions.get_backend(C)
    kernel! = compute_vof_normal_2d_kernel!(backend)
    kernel!(nx, ny, C, Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

# ===================================================================
# Part 1b: ELVIRA reconstruction (second-order accurate normals)
# ===================================================================
#
# ELVIRA: Efficient Least-squares VOF Interface Reconstruction Algorithm
# For each interface cell, test 6 candidate normals (3 column-wise slopes
# giving ny/nx ratios, 3 row-wise slopes giving nx/ny ratios) and keep
# the one that minimizes the L2 reconstruction error on the 3×3 stencil.
#
# This achieves O(Δx²) interface reconstruction vs O(Δx) for Youngs/MYC.
#
# Reference: Pilliod & Puckett (2004) doi:10.1016/j.jcp.2003.12.023

# L2 reconstruction error of a candidate normal on a 3×3 stencil
@inline function _elvira_l2(cnx::T, cny::T, s22::T,
                             s11::T, s21::T, s31::T,
                             s12::T, s32::T,
                             s13::T, s23::T, s33::T) where T
    alpha = _line_alpha(s22, cnx, cny)
    err = zero(T)
    err += (_line_area(cnx, cny, alpha - cnx - cny) - s11)^2
    err += (_line_area(cnx, cny, alpha       - cny) - s21)^2
    err += (_line_area(cnx, cny, alpha + cnx - cny) - s31)^2
    err += (_line_area(cnx, cny, alpha - cnx      ) - s12)^2
    # center cell exact by construction — skip
    err += (_line_area(cnx, cny, alpha + cnx      ) - s32)^2
    err += (_line_area(cnx, cny, alpha - cnx + cny) - s13)^2
    err += (_line_area(cnx, cny, alpha       + cny) - s23)^2
    err += (_line_area(cnx, cny, alpha + cnx + cny) - s33)^2
    err
end

# Normalize candidate and compute error; update best if better
@inline function _elvira_try(cnx::T, cny::T, s22::T,
                              s11::T, s21::T, s31::T,
                              s12::T, s32::T,
                              s13::T, s23::T, s33::T,
                              best_err::T, best_nx::T, best_ny::T) where T
    mm = abs(cnx) + abs(cny) + T(1e-30)
    cnx_n = cnx / mm; cny_n = cny / mm
    err = _elvira_l2(cnx_n, cny_n, s22, s11, s21, s31, s12, s32, s13, s23, s33)
    do_update = err < best_err
    best_err = ifelse(do_update, err, best_err)
    best_nx  = ifelse(do_update, cnx_n, best_nx)
    best_ny  = ifelse(do_update, cny_n, best_ny)
    best_err, best_nx, best_ny
end

@kernel function compute_vof_normal_elvira_2d_kernel!(nx_out, ny_out, @Const(C), Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(C)
        c = C[i,j]

        ip = ifelse(i < Nx, i + 1, 1)
        im = ifelse(i > 1,  i - 1, Nx)
        jp = ifelse(j < Ny, j + 1, 1)
        jm = ifelse(j > 1,  j - 1, Ny)

        # Non-interface cells: zero normal
        is_interface = c > T(1e-6) && c < one(T) - T(1e-6)

        # Load 3×3 stencil
        s11 = C[im,jm]; s21 = C[i,jm]; s31 = C[ip,jm]
        s12 = C[im,j];  s22 = c;       s32 = C[ip,j]
        s13 = C[im,jp]; s23 = C[i,jp]; s33 = C[ip,jp]

        # 6 candidate slopes (-∇C convention)
        # 3 with ny dominant: nx = row-wise central diff, ny = ±1
        slope_b = (s11 - s31) / T(2)
        slope_m = (s12 - s32) / T(2)
        slope_t = (s13 - s33) / T(2)
        # 3 with nx dominant: ny = column-wise central diff, nx = ±1
        slope_l = (s11 - s13) / T(2)
        slope_c = (s21 - s23) / T(2)
        slope_r = (s31 - s33) / T(2)

        # Sign of dominant direction (from Youngs stencil)
        youngs_mx = s11 + T(2)*s12 + s13 - s31 - T(2)*s32 - s33
        youngs_my = s11 + T(2)*s21 + s31 - s13 - T(2)*s23 - s33
        sign_nx = ifelse(youngs_mx > zero(T), one(T), -one(T))
        sign_ny = ifelse(youngs_my > zero(T), one(T), -one(T))

        best_err = T(1e30)
        best_nx = zero(T)
        best_ny = zero(T)

        # Test 6 candidates × 2 signs = 12 total (brute force, GPU-safe)
        # ny dominant: (slope, +1) and (slope, -1)
        best_err, best_nx, best_ny = _elvira_try(
            slope_b, one(T), s22, s11, s21, s31, s12, s32, s13, s23, s33,
            best_err, best_nx, best_ny)
        best_err, best_nx, best_ny = _elvira_try(
            slope_b, -one(T), s22, s11, s21, s31, s12, s32, s13, s23, s33,
            best_err, best_nx, best_ny)
        best_err, best_nx, best_ny = _elvira_try(
            slope_m, one(T), s22, s11, s21, s31, s12, s32, s13, s23, s33,
            best_err, best_nx, best_ny)
        best_err, best_nx, best_ny = _elvira_try(
            slope_m, -one(T), s22, s11, s21, s31, s12, s32, s13, s23, s33,
            best_err, best_nx, best_ny)
        best_err, best_nx, best_ny = _elvira_try(
            slope_t, one(T), s22, s11, s21, s31, s12, s32, s13, s23, s33,
            best_err, best_nx, best_ny)
        best_err, best_nx, best_ny = _elvira_try(
            slope_t, -one(T), s22, s11, s21, s31, s12, s32, s13, s23, s33,
            best_err, best_nx, best_ny)

        # nx dominant: (+1, slope) and (-1, slope)
        best_err, best_nx, best_ny = _elvira_try(
            one(T), slope_l, s22, s11, s21, s31, s12, s32, s13, s23, s33,
            best_err, best_nx, best_ny)
        best_err, best_nx, best_ny = _elvira_try(
            -one(T), slope_l, s22, s11, s21, s31, s12, s32, s13, s23, s33,
            best_err, best_nx, best_ny)
        best_err, best_nx, best_ny = _elvira_try(
            one(T), slope_c, s22, s11, s21, s31, s12, s32, s13, s23, s33,
            best_err, best_nx, best_ny)
        best_err, best_nx, best_ny = _elvira_try(
            -one(T), slope_c, s22, s11, s21, s31, s12, s32, s13, s23, s33,
            best_err, best_nx, best_ny)
        best_err, best_nx, best_ny = _elvira_try(
            one(T), slope_r, s22, s11, s21, s31, s12, s32, s13, s23, s33,
            best_err, best_nx, best_ny)
        best_err, best_nx, best_ny = _elvira_try(
            -one(T), slope_r, s22, s11, s21, s31, s12, s32, s13, s23, s33,
            best_err, best_nx, best_ny)

        # Fix sign: _line_area is symmetric in sign, so we must orient
        # the normal using the Youngs gradient direction (-∇C = outward)
        youngs_dot = best_nx * youngs_mx + best_ny * youngs_my
        flip = youngs_dot < zero(T)
        best_nx = ifelse(flip, -best_nx, best_nx)
        best_ny = ifelse(flip, -best_ny, best_ny)

        nx_out[i,j] = ifelse(is_interface, best_nx, zero(T))
        ny_out[i,j] = ifelse(is_interface, best_ny, zero(T))
    end
end

function compute_vof_normal_elvira_2d!(nx, ny, C, Nx, Ny)
    backend = KernelAbstractions.get_backend(C)
    kernel! = compute_vof_normal_elvira_2d_kernel!(backend)
    kernel!(nx, ny, C, Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

# ===================================================================
# Part 2: PLIC geometry (exact Basilisk formulas)
# ===================================================================
#
# Cell coordinates: [-0.5, 0.5]² (cell-centered)
# Normal: n = (nx, ny), pointing from liquid to gas (Youngs)
# Alpha: line parameter such that n·x = alpha defines the interface
#
# References:
# - Scardovelli & Zaleski (2000) doi:10.1006/jcph.2000.6567
# - Weymouth & Yue (2010) doi:10.1016/j.jcp.2010.04.024
# - Basilisk: src/geometry.h, src/vof.h (Popinet 2009)

# --- _line_area: exact Basilisk line_area() ---
# Area of fluid below line n·x = α in unit cell [0,1]²

@inline function _line_area(nx::T, ny::T, alpha::T) where T
    a = alpha + (nx + ny) / T(2)
    n1 = nx; n2 = ny
    if n1 < zero(T); a -= n1; n1 = -n1; end
    if n2 < zero(T); a -= n2; n2 = -n2; end
    if a <= zero(T); return zero(T); end
    if a >= n1 + n2; return one(T); end
    if n1 < T(1e-10); return clamp(a / n2, zero(T), one(T)); end
    if n2 < T(1e-10); return clamp(a / n1, zero(T), one(T)); end
    v = a * a
    a1 = a - n1; if a1 > zero(T); v -= a1 * a1; end
    a2 = a - n2; if a2 > zero(T); v -= a2 * a2; end
    return clamp(v / (T(2) * n1 * n2), zero(T), one(T))
end

# --- _line_alpha: exact Basilisk line_alpha() ---
# Given C and (nx, ny), find α such that area below n·x = α equals C
# in cell [-0.5, 0.5]²

@inline function _line_alpha(c::T, nx::T, ny::T) where T
    n1 = abs(nx); n2 = abs(ny)
    if n1 > n2; n1, n2 = n2, n1; end
    cc = clamp(c, zero(T), one(T))
    v1 = n1 / T(2)
    if n2 < T(1e-10); return zero(T); end  # degenerate
    if cc <= v1 / n2
        alpha = sqrt(T(2) * cc * n1 * n2)
    elseif cc <= one(T) - v1 / n2
        alpha = cc * n2 + v1
    else
        alpha = n1 + n2 - sqrt(T(2) * n1 * n2 * (one(T) - cc))
    end
    if nx < zero(T); alpha += nx; end
    if ny < zero(T); alpha += ny; end
    return alpha - (nx + ny) / T(2)
end

# --- _rectangle_fraction: exact Basilisk rectangle_fraction() ---
# Fraction of fluid in rectangle [ax,bx] × [ay,by] given line n·x = α

@inline function _rectangle_fraction(nx::T, ny::T, alpha::T, ax::T, ay::T, bx::T, by::T) where T
    alpha_p = alpha - nx*(bx+ax)/T(2) - ny*(by+ay)/T(2)
    nx_p = nx * (bx - ax)
    ny_p = ny * (by - ay)
    return _line_area(nx_p, ny_p, alpha_p)
end

# --- plic_line_position: backward-compat wrapper ---

@inline plic_line_position(nx::T, ny::T, C::T) where T = _line_alpha(C, nx, ny)

# ===================================================================
# Part 3: PLIC advection kernels (Basilisk algorithm)
# ===================================================================
#
# Geometric PLIC advection with Weymouth-Yue (2010) correction.
# Follows the exact Basilisk algorithm from src/vof.h:
# - Face velocity = average of cell-centered velocities
# - Donor cell PLIC reconstruction → rectangle_fraction for strip flux
# - Weymouth-Yue: cc = (C > 0.5) ? 1 : 0 for compression term

@kernel function advect_vof_plic_x_2d_kernel!(C_new, @Const(C), @Const(cc_field),
                                                @Const(nx_n), @Const(ny_n),
                                                @Const(ux), dt_factor, Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(C)
        ip = ifelse(i < Nx, i + 1, 1)
        im = ifelse(i > 1,  i - 1, Nx)

        # --- Right face velocity (scaled by dt_factor for sub-stepping) ---
        u_right = (ux[i, j] + ux[ip, j]) / T(2) * dt_factor

        # --- Right face flux (geometric PLIC) ---
        if u_right > zero(T)
            c_donor = C[i,j]
            if c_donor <= zero(T) || c_donor >= one(T)
                cf_right = c_donor
            else
                s = one(T)
                alpha_d = _line_alpha(c_donor, nx_n[i,j], ny_n[i,j])
                cf_right = _rectangle_fraction(-s*nx_n[i,j], ny_n[i,j], alpha_d,
                                               -T(0.5), -T(0.5), s*u_right - T(0.5), T(0.5))
            end
            flux_right = cf_right * u_right
        else
            c_donor = C[ip,j]
            if c_donor <= zero(T) || c_donor >= one(T)
                cf_right = c_donor
            else
                s = -one(T)
                alpha_d = _line_alpha(c_donor, nx_n[ip,j], ny_n[ip,j])
                cf_right = _rectangle_fraction(-s*nx_n[ip,j], ny_n[ip,j], alpha_d,
                                               -T(0.5), -T(0.5), s*u_right - T(0.5), T(0.5))
            end
            flux_right = cf_right * u_right
        end

        # --- Left face velocity ---
        u_left = (ux[im, j] + ux[i, j]) / T(2) * dt_factor

        # --- Left face flux (geometric PLIC) ---
        if u_left > zero(T)
            c_donor = C[im,j]
            if c_donor <= zero(T) || c_donor >= one(T)
                cf_left = c_donor
            else
                s = one(T)
                alpha_d = _line_alpha(c_donor, nx_n[im,j], ny_n[im,j])
                cf_left = _rectangle_fraction(-s*nx_n[im,j], ny_n[im,j], alpha_d,
                                              -T(0.5), -T(0.5), s*u_left - T(0.5), T(0.5))
            end
            flux_left = cf_left * u_left
        else
            c_donor = C[i,j]
            if c_donor <= zero(T) || c_donor >= one(T)
                cf_left = c_donor
            else
                s = -one(T)
                alpha_d = _line_alpha(c_donor, nx_n[i,j], ny_n[i,j])
                cf_left = _rectangle_fraction(-s*nx_n[i,j], ny_n[i,j], alpha_d,
                                              -T(0.5), -T(0.5), s*u_left - T(0.5), T(0.5))
            end
            flux_left = cf_left * u_left
        end

        # Weymouth-Yue (2010) correction — cc frozen before all sweeps
        C_new[i, j] = C[i, j] - (flux_right - flux_left) + cc_field[i,j] * (u_right - u_left)
    end
end

@kernel function advect_vof_plic_y_2d_kernel!(C_new, @Const(C), @Const(cc_field),
                                                @Const(nx_n), @Const(ny_n),
                                                @Const(uy), dt_factor, Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(C)
        jp = ifelse(j < Ny, j + 1, 1)
        jm = ifelse(j > 1,  j - 1, Ny)

        # --- Top face velocity (scaled by dt_factor for sub-stepping) ---
        u_top = (uy[i, j] + uy[i, jp]) / T(2) * dt_factor

        # --- Top face flux (geometric PLIC) ---
        if u_top > zero(T)
            c_donor = C[i,j]
            if c_donor <= zero(T) || c_donor >= one(T)
                cf_top = c_donor
            else
                s = one(T)
                alpha_d = _line_alpha(c_donor, nx_n[i,j], ny_n[i,j])
                cf_top = _rectangle_fraction(nx_n[i,j], -s*ny_n[i,j], alpha_d,
                                             -T(0.5), -T(0.5), T(0.5), s*u_top - T(0.5))
            end
            flux_top = cf_top * u_top
        else
            c_donor = C[i,jp]
            if c_donor <= zero(T) || c_donor >= one(T)
                cf_top = c_donor
            else
                s = -one(T)
                alpha_d = _line_alpha(c_donor, nx_n[i,jp], ny_n[i,jp])
                cf_top = _rectangle_fraction(nx_n[i,jp], -s*ny_n[i,jp], alpha_d,
                                             -T(0.5), -T(0.5), T(0.5), s*u_top - T(0.5))
            end
            flux_top = cf_top * u_top
        end

        # --- Bottom face velocity ---
        u_bot = (uy[i, jm] + uy[i, j]) / T(2) * dt_factor

        # --- Bottom face flux (geometric PLIC) ---
        if u_bot > zero(T)
            c_donor = C[i,jm]
            if c_donor <= zero(T) || c_donor >= one(T)
                cf_bot = c_donor
            else
                s = one(T)
                alpha_d = _line_alpha(c_donor, nx_n[i,jm], ny_n[i,jm])
                cf_bot = _rectangle_fraction(nx_n[i,jm], -s*ny_n[i,jm], alpha_d,
                                             -T(0.5), -T(0.5), T(0.5), s*u_bot - T(0.5))
            end
            flux_bot = cf_bot * u_bot
        else
            c_donor = C[i,j]
            if c_donor <= zero(T) || c_donor >= one(T)
                cf_bot = c_donor
            else
                s = -one(T)
                alpha_d = _line_alpha(c_donor, nx_n[i,j], ny_n[i,j])
                cf_bot = _rectangle_fraction(nx_n[i,j], -s*ny_n[i,j], alpha_d,
                                             -T(0.5), -T(0.5), T(0.5), s*u_bot - T(0.5))
            end
            flux_bot = cf_bot * u_bot
        end

        # Weymouth-Yue (2010) correction — cc frozen before all sweeps
        C_new[i, j] = C[i, j] - (flux_top - flux_bot) + cc_field[i,j] * (u_top - u_bot)
    end
end

"""
    advect_vof_plic_2d!(C_new, C, nx_n, ny_n, cc_field, ux, uy, Nx, Ny; step=1)

Geometric PLIC advection of volume fraction C using directional splitting
with Basilisk-style `rectangle_fraction` flux computation and Weymouth-Yue
compression correction.

Automatic CFL sub-stepping: if max|u|·dt/Δx > 0.5, the step is split
into N sub-steps so that each sub-step satisfies CFL ≤ 0.5.

Requires pre-computed interface normals `(nx_n, ny_n)` from `compute_vof_normal_2d!`
and a pre-allocated `cc_field` work array (same size as C).
Strang alternating splitting: x→y on odd steps, y→x on even steps.
"""
function advect_vof_plic_2d!(C_new, C, nx_n, ny_n, cc_field, ux, uy, Nx, Ny;
                              step::Int=1, recon::Symbol=:myc)
    backend = KernelAbstractions.get_backend(C)
    T = eltype(C)

    # Select reconstruction function
    recon_fn! = recon === :elvira ? compute_vof_normal_elvira_2d! : compute_vof_normal_2d!

    # Compute CFL and determine sub-stepping
    u_max = max(maximum(abs, Array(ux)), maximum(abs, Array(uy)))
    cfl = u_max  # dt/dx = 1 in lattice units
    n_sub = max(1, ceil(Int, cfl / T(0.45)))  # target CFL ≤ 0.45 per sub-step
    dt_sub = one(T) / T(n_sub)

    for sub in 1:n_sub
        # Weymouth-Yue: freeze cc = (C > 0.5) ONCE before all sweeps
        _compute_cc_field!(cc_field, C, backend, Nx, Ny)

        # Strang alternating: alternate sweep order based on (step + sub)
        sweep_idx = step + sub - 1
        if isodd(sweep_idx)
            _plic_sweep_xy!(C_new, C, cc_field, nx_n, ny_n, ux, uy, dt_sub,
                            recon_fn!, backend, Nx, Ny)
        else
            _plic_sweep_yx!(C_new, C, cc_field, nx_n, ny_n, ux, uy, dt_sub,
                            recon_fn!, backend, Nx, Ny)
        end
        copyto!(C, C_new)
    end
end

function _plic_sweep_xy!(C_new, C, cc_field, nx_n, ny_n, ux, uy, dt_sub,
                          recon_fn!, backend, Nx, Ny)
    kernel_x! = advect_vof_plic_x_2d_kernel!(backend)
    kernel_x!(C_new, C, cc_field, nx_n, ny_n, ux, dt_sub, Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
    copyto!(C, C_new)
    recon_fn!(nx_n, ny_n, C, Nx, Ny)
    kernel_y! = advect_vof_plic_y_2d_kernel!(backend)
    kernel_y!(C_new, C, cc_field, nx_n, ny_n, uy, dt_sub, Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

function _plic_sweep_yx!(C_new, C, cc_field, nx_n, ny_n, ux, uy, dt_sub,
                          recon_fn!, backend, Nx, Ny)
    kernel_y! = advect_vof_plic_y_2d_kernel!(backend)
    kernel_y!(C_new, C, cc_field, nx_n, ny_n, uy, dt_sub, Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
    copyto!(C, C_new)
    recon_fn!(nx_n, ny_n, C, Nx, Ny)
    kernel_x! = advect_vof_plic_x_2d_kernel!(backend)
    kernel_x!(C_new, C, cc_field, nx_n, ny_n, ux, dt_sub, Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

# --- Helper: compute Weymouth-Yue cc field on GPU ---

@kernel function _cc_field_kernel!(cc, @Const(C))
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(C)
        cc[i,j] = ifelse(C[i,j] > T(0.5), one(T), zero(T))
    end
end

function _compute_cc_field!(cc, C, backend, Nx, Ny)
    kernel! = _cc_field_kernel!(backend)
    kernel!(cc, C; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

# ===================================================================
# Part 4: MUSCL-Superbee advection (algebraic VOF advection)
# ===================================================================

# --- Superbee flux limiter (GPU-safe inline) ---

@inline function _superbee(r::T) where T
    # Superbee limiter: max(0, min(2r,1), min(r,2))
    # Most compressive TVD limiter — ideal for sharp VOF interfaces
    ifelse(r <= zero(T), zero(T),
        max(min(T(2) * r, one(T)), min(r, T(2))))
end

# --- MUSCL-Superbee VOF advection (directional splitting) ---
#
# Second-order TVD scheme with superbee flux limiter for sharp interfaces.
# Split advection: first x-direction, then y-direction.
# Conservative: total C preserved to machine precision.

@kernel function advect_vof_x_2d_kernel!(C_new, @Const(C), @Const(ux), Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(C)
        ip  = ifelse(i < Nx, i + 1, 1)
        im  = ifelse(i > 1,  i - 1, Nx)
        ipp = ifelse(ip < Nx, ip + 1, 1)
        imm = ifelse(im > 1,  im - 1, Nx)

        # Face velocities (arithmetic mean)
        u_right = (ux[i, j] + ux[ip, j]) / T(2)
        u_left  = (ux[im, j] + ux[i, j]) / T(2)

        # --- Right face flux (MUSCL-Superbee) ---
        # Upwind value and gradient ratio
        if u_right > zero(T)
            C_up = C[i, j]
            dC_down = C[ip, j] - C[i, j]
            dC_up   = C[i, j] - C[im, j]
        else
            C_up = C[ip, j]
            dC_down = C[i, j] - C[ip, j]
            dC_up   = C[ip, j] - C[ipp, j]
        end
        r_right = ifelse(abs(dC_down) > T(1e-30),
                         dC_up / dC_down, zero(T))
        phi_r = _superbee(r_right)
        flux_right = u_right * (C_up + T(0.5) * phi_r * dC_down * max(zero(T), one(T) - abs(u_right)))

        # --- Left face flux (MUSCL-Superbee) ---
        if u_left > zero(T)
            C_up = C[im, j]
            dC_down = C[i, j] - C[im, j]
            dC_up   = C[im, j] - C[imm, j]
        else
            C_up = C[i, j]
            dC_down = C[im, j] - C[i, j]
            dC_up   = C[i, j] - C[ip, j]
        end
        r_left = ifelse(abs(dC_down) > T(1e-30),
                        dC_up / dC_down, zero(T))
        phi_l = _superbee(r_left)
        flux_left = u_left * (C_up + T(0.5) * phi_l * dC_down * max(zero(T), one(T) - abs(u_left)))

        C_new[i, j] = C[i, j] - (flux_right - flux_left)
    end
end

@kernel function advect_vof_y_2d_kernel!(C_new, @Const(C), @Const(uy), Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(C)
        jp  = ifelse(j < Ny, j + 1, 1)
        jm  = ifelse(j > 1,  j - 1, Ny)
        jpp = ifelse(jp < Ny, jp + 1, 1)
        jmm = ifelse(jm > 1,  jm - 1, Ny)

        u_top = (uy[i, j] + uy[i, jp]) / T(2)
        u_bot = (uy[i, jm] + uy[i, j]) / T(2)

        # --- Top face flux ---
        if u_top > zero(T)
            C_up = C[i, j]
            dC_down = C[i, jp] - C[i, j]
            dC_up   = C[i, j] - C[i, jm]
        else
            C_up = C[i, jp]
            dC_down = C[i, j] - C[i, jp]
            dC_up   = C[i, jp] - C[i, jpp]
        end
        r_top = ifelse(abs(dC_down) > T(1e-30),
                       dC_up / dC_down, zero(T))
        phi_t = _superbee(r_top)
        flux_top = u_top * (C_up + T(0.5) * phi_t * dC_down * max(zero(T), one(T) - abs(u_top)))

        # --- Bottom face flux ---
        if u_bot > zero(T)
            C_up = C[i, jm]
            dC_down = C[i, j] - C[i, jm]
            dC_up   = C[i, jm] - C[i, jmm]
        else
            C_up = C[i, j]
            dC_down = C[i, jm] - C[i, j]
            dC_up   = C[i, j] - C[i, jp]
        end
        r_bot = ifelse(abs(dC_down) > T(1e-30),
                       dC_up / dC_down, zero(T))
        phi_b = _superbee(r_bot)
        flux_bot = u_bot * (C_up + T(0.5) * phi_b * dC_down * max(zero(T), one(T) - abs(u_bot)))

        C_new[i, j] = C[i, j] - (flux_top - flux_bot)
    end
end

function advect_vof_2d!(C_new, C, ux, uy, Nx, Ny)
    backend = KernelAbstractions.get_backend(C)
    # Strang splitting: x then y
    kernel_x! = advect_vof_x_2d_kernel!(backend)
    kernel_x!(C_new, C, ux, Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)

    # Copy C_new → C for y-pass
    copyto!(C, C_new)

    kernel_y! = advect_vof_y_2d_kernel!(backend)
    kernel_y!(C_new, C, uy, Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

# ===================================================================
# Part 5: Height-function curvature
# ===================================================================
#
# For each interface cell, compute curvature from height function:
# κ = -h''/(1 + h'²)^{3/2}
# where h(x) = Σⱼ C(i, j_stencil) for a vertical column (if |ny| > |nx|)
# or h(y) = Σᵢ C(i_stencil, j) for a horizontal column

@kernel function compute_hf_curvature_2d_kernel!(κ, @Const(C), @Const(nx_n), @Const(ny_n),
                                                   Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(C)
        c = C[i,j]

        # Only compute curvature at interface cells
        if c > T(0.01) && c < T(0.99)
            if abs(ny_n[i,j]) >= abs(nx_n[i,j])
                # Interface more horizontal → use vertical columns h(x)
                # h(x) = Σ_j C(x, j) over a stencil of ±3 cells in y
                # h at i-1, i, i+1
                im = ifelse(i > 1, i - 1, Nx)
                ip = ifelse(i < Nx, i + 1, 1)

                h_m = zero(T); h_0 = zero(T); h_p = zero(T)
                for dj in -2:2
                    jj = j + dj
                    jj = ifelse(jj < 1, jj + Ny, ifelse(jj > Ny, jj - Ny, jj))
                    h_m += C[im, jj]
                    h_0 += C[i,  jj]
                    h_p += C[ip, jj]
                end

                # h' ≈ (h_p - h_m) / 2
                hp = (h_p - h_m) / T(2)
                # h'' ≈ h_p - 2h_0 + h_m
                hpp = h_p - T(2)*h_0 + h_m

                κ[i,j] = -hpp / (one(T) + hp^2)^T(1.5)
            else
                # Interface more vertical → use horizontal rows h(y)
                jm = ifelse(j > 1, j - 1, Ny)
                jp = ifelse(j < Ny, j + 1, 1)

                h_m = zero(T); h_0 = zero(T); h_p = zero(T)
                for di in -2:2
                    ii = i + di
                    ii = ifelse(ii < 1, ii + Nx, ifelse(ii > Nx, ii - Nx, ii))
                    h_m += C[ii, jm]
                    h_0 += C[ii, j]
                    h_p += C[ii, jp]
                end

                hp = (h_p - h_m) / T(2)
                hpp = h_p - T(2)*h_0 + h_m

                κ[i,j] = -hpp / (one(T) + hp^2)^T(1.5)
            end
        else
            κ[i,j] = zero(T)
        end
    end
end

function compute_hf_curvature_2d!(κ, C, nx, ny, Nx, Ny)
    backend = KernelAbstractions.get_backend(C)
    kernel! = compute_hf_curvature_2d_kernel!(backend)
    kernel!(κ, C, nx, ny, Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

# ===================================================================
# Part 6: Surface tension, azimuthal curvature, two-phase collision
# ===================================================================

# --- CSF surface tension force: F = σ·κ·∇C ---

@kernel function compute_surface_tension_2d_kernel!(Fx, Fy, @Const(κ), @Const(C),
                                                      σ, Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(C)
        ip = ifelse(i < Nx, i + 1, 1)
        im = ifelse(i > 1,  i - 1, Nx)
        jp = ifelse(j < Ny, j + 1, 1)
        jm = ifelse(j > 1,  j - 1, Ny)

        # ∇C (central differences)
        dCdx = (C[ip,j] - C[im,j]) / T(2)
        dCdy = (C[i,jp] - C[i,jm]) / T(2)

        # CSF: F = σ·κ·∇C
        Fx[i,j] = σ * κ[i,j] * dCdx
        Fy[i,j] = σ * κ[i,j] * dCdy
    end
end

# --- Axisymmetric curvature: add azimuthal component κ₂ = n_r / r ---

@kernel function add_azimuthal_curvature_2d_kernel!(κ, @Const(C), @Const(ny_n), Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(κ)
        c = C[i,j]
        if c > T(0.01) && c < T(0.99)
            r = max(T(j) - T(0.5), one(T))  # clamp r ≥ 1 to avoid singularity
            # Azimuthal curvature: κ₂ = n_r / r where n_r = ny_n
            # Normal points outward (Basilisk convention), so n_r > 0 above axis
            κ_axi = ny_n[i,j] / r
            # Clamp total curvature to avoid instability
            κ_total = κ[i,j] + κ_axi
            κ[i,j] = clamp(κ_total, -T(0.5), T(0.5))
        end
    end
end

function add_azimuthal_curvature_2d!(κ, C, ny_n, Ny)
    backend = KernelAbstractions.get_backend(κ)
    Nx = size(κ, 1)
    kernel! = add_azimuthal_curvature_2d_kernel!(backend)
    kernel!(κ, C, ny_n, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

function compute_surface_tension_2d!(Fx, Fy, κ, C, σ, Nx, Ny)
    backend = KernelAbstractions.get_backend(C)
    T = eltype(C)
    kernel! = compute_surface_tension_2d_kernel!(backend)
    kernel!(Fx, Fy, κ, C, T(σ), Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

# --- Two-phase collision: variable density and viscosity ---

@kernel function collide_twophase_2d_kernel!(f, @Const(C), @Const(Fx_st), @Const(Fy_st),
                                               @Const(is_solid), ρ_l, ρ_g, ν_l, ν_g)
    i, j = @index(Global, NTuple)

    @inbounds begin
        if is_solid[i, j]
            tmp = f[i,j,2]; f[i,j,2] = f[i,j,4]; f[i,j,4] = tmp
            tmp = f[i,j,3]; f[i,j,3] = f[i,j,5]; f[i,j,5] = tmp
            tmp = f[i,j,6]; f[i,j,6] = f[i,j,8]; f[i,j,8] = tmp
            tmp = f[i,j,7]; f[i,j,7] = f[i,j,9]; f[i,j,9] = tmp
        else
            T = eltype(f)
            f1=f[i,j,1]; f2=f[i,j,2]; f3=f[i,j,3]; f4=f[i,j,4]
            f5=f[i,j,5]; f6=f[i,j,6]; f7=f[i,j,7]; f8=f[i,j,8]; f9=f[i,j,9]

            c = C[i,j]
            # Interpolated density and viscosity
            ρ_local = c * ρ_l + (one(T) - c) * ρ_g
            ν_local = c * ν_l + (one(T) - c) * ν_g
            ω_local = one(T) / (T(3) * ν_local + T(0.5))

            # Surface tension force
            fx = Fx_st[i,j]
            fy = Fy_st[i,j]

            ρ_f = f1+f2+f3+f4+f5+f6+f7+f8+f9
            inv_ρ = one(T) / ρ_f
            ux = ((f2-f4+f6-f7-f8+f9) + fx/T(2)) * inv_ρ
            uy = ((f3-f5+f6+f7-f8-f9) + fy/T(2)) * inv_ρ
            usq = ux*ux + uy*uy

            guo_pref = one(T) - ω_local / T(2)
            t3=T(3); t45=T(4.5); t15=T(1.5)

            # BGK + Guo surface tension (same pattern as collide_sc_2d)
            feq=T(4.0/9.0)*ρ_f*(one(T)-t15*usq)
            Sq=T(4.0/9.0)*((-ux)*fx+(-uy)*fy)*t3
            f[i,j,1]=f1-ω_local*(f1-feq)+guo_pref*Sq

            cu=ux; feq=T(1.0/9.0)*ρ_f*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            Sq=T(1.0/9.0)*((one(T)-ux)*fx+(-uy)*fy)*t3+T(1.0/9.0)*ux*fx*T(9)
            f[i,j,2]=f2-ω_local*(f2-feq)+guo_pref*Sq

            cu=uy; feq=T(1.0/9.0)*ρ_f*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            Sq=T(1.0/9.0)*((-ux)*fx+(one(T)-uy)*fy)*t3+T(1.0/9.0)*uy*fy*T(9)
            f[i,j,3]=f3-ω_local*(f3-feq)+guo_pref*Sq

            cu=-ux; feq=T(1.0/9.0)*ρ_f*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            Sq=T(1.0/9.0)*((-one(T)-ux)*fx+(-uy)*fy)*t3+T(1.0/9.0)*ux*fx*T(9)
            f[i,j,4]=f4-ω_local*(f4-feq)+guo_pref*Sq

            cu=-uy; feq=T(1.0/9.0)*ρ_f*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            Sq=T(1.0/9.0)*((-ux)*fx+(-one(T)-uy)*fy)*t3+T(1.0/9.0)*uy*fy*T(9)
            f[i,j,5]=f5-ω_local*(f5-feq)+guo_pref*Sq

            cu=ux+uy; feq=T(1.0/36.0)*ρ_f*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            Sq=T(1.0/36.0)*((one(T)-ux)*fx+(one(T)-uy)*fy)*t3+T(1.0/36.0)*cu*(fx+fy)*T(9)
            f[i,j,6]=f6-ω_local*(f6-feq)+guo_pref*Sq

            cu=-ux+uy; feq=T(1.0/36.0)*ρ_f*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            Sq=T(1.0/36.0)*((-one(T)-ux)*fx+(one(T)-uy)*fy)*t3+T(1.0/36.0)*cu*(-fx+fy)*T(9)
            f[i,j,7]=f7-ω_local*(f7-feq)+guo_pref*Sq

            cu=-ux-uy; feq=T(1.0/36.0)*ρ_f*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            Sq=T(1.0/36.0)*((-one(T)-ux)*fx+(-one(T)-uy)*fy)*t3+T(1.0/36.0)*cu*(-fx-fy)*T(9)
            f[i,j,8]=f8-ω_local*(f8-feq)+guo_pref*Sq

            cu=ux-uy; feq=T(1.0/36.0)*ρ_f*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            Sq=T(1.0/36.0)*((one(T)-ux)*fx+(-one(T)-uy)*fy)*t3+T(1.0/36.0)*cu*(fx-fy)*T(9)
            f[i,j,9]=f9-ω_local*(f9-feq)+guo_pref*Sq
        end
    end
end

function collide_twophase_2d!(f, C, Fx_st, Fy_st, is_solid; ρ_l=1.0, ρ_g=0.001, ν_l=0.1, ν_g=0.1)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    T = eltype(f)
    kernel! = collide_twophase_2d_kernel!(backend)
    kernel!(f, C, Fx_st, Fy_st, is_solid, T(ρ_l), T(ρ_g), T(ν_l), T(ν_g); ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end
