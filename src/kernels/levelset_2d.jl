using KernelAbstractions

# ===========================================================================
# Level-set 2D — Signed distance function for interface tracking
#
# φ > 0: liquid, φ < 0: gas, φ = 0: interface
# Convention: for a droplet of radius R, φ = R - r (positive inside)
#
# References:
# - Sussman, Smereka & Osher (1994) doi:10.1006/jcph.1994.1155
# - Sussman & Fatemi (1999) doi:10.1137/S1064827596298245
# ===========================================================================

# --- Minmod limiter (for smooth fields like φ) ---

@inline function _minmod(a::T, b::T) where T
    ifelse(a * b <= zero(T), zero(T),
        ifelse(abs(a) < abs(b), a, b))
end

# --- Level-set advection (MUSCL-Minmod, 2nd order TVD) ---
#
# Uses the same MUSCL framework as VOF but with minmod limiter (less
# compressive, better suited for smooth signed-distance fields).

@kernel function advect_ls_2d_kernel!(phi_new, @Const(phi), @Const(ux), @Const(uy), Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(phi)
        ip  = ifelse(i < Nx, i + 1, 1)
        im  = ifelse(i > 1,  i - 1, Nx)
        ipp = ifelse(ip < Nx, ip + 1, 1)
        imm = ifelse(im > 1,  im - 1, Nx)
        jp  = ifelse(j < Ny, j + 1, 1)
        jm  = ifelse(j > 1,  j - 1, Ny)
        jpp = ifelse(jp < Ny, jp + 1, 1)
        jmm = ifelse(jm > 1,  jm - 1, Ny)

        p = phi[i, j]
        u = ux[i, j]
        v = uy[i, j]

        # --- x-direction MUSCL ---
        if u > zero(T)
            grad_up   = p - phi[im, j]
            grad_down = phi[ip, j] - p
            slope = _minmod(grad_up, grad_down)
            phi_face = p + T(0.5) * slope * max(zero(T), one(T) - abs(u))
            flux_x = u * phi_face
        else
            grad_up   = phi[ip, j] - phi[ipp, j]
            grad_down = p - phi[ip, j]
            slope = _minmod(grad_up, grad_down)
            phi_face = phi[ip, j] - T(0.5) * slope * max(zero(T), one(T) - abs(u))
            flux_x = u * phi_face
        end

        if u > zero(T)
            # Left face
            grad_up_l   = phi[im, j] - phi[imm, j]
            grad_down_l = p - phi[im, j]
            slope_l = _minmod(grad_up_l, grad_down_l)
            phi_face_l = phi[im, j] + T(0.5) * slope_l * max(zero(T), one(T) - abs(u))
        else
            grad_up_l   = p - phi[ip, j]
            grad_down_l = phi[im, j] - p
            slope_l = _minmod(grad_up_l, grad_down_l)
            phi_face_l = p - T(0.5) * slope_l * max(zero(T), one(T) - abs(u))
        end
        u_left = (ux[im, j] + u) / T(2)
        flux_x_left = u_left * phi_face_l

        # --- y-direction MUSCL ---
        if v > zero(T)
            grad_up_y   = p - phi[i, jm]
            grad_down_y = phi[i, jp] - p
            slope_y = _minmod(grad_up_y, grad_down_y)
            phi_face_y = p + T(0.5) * slope_y * max(zero(T), one(T) - abs(v))
            flux_y = v * phi_face_y
        else
            grad_up_y   = phi[i, jp] - phi[i, jpp]
            grad_down_y = p - phi[i, jp]
            slope_y = _minmod(grad_up_y, grad_down_y)
            phi_face_y = phi[i, jp] - T(0.5) * slope_y * max(zero(T), one(T) - abs(v))
            flux_y = v * phi_face_y
        end

        if v > zero(T)
            grad_up_yb   = phi[i, jm] - phi[i, jmm]
            grad_down_yb = p - phi[i, jm]
            slope_yb = _minmod(grad_up_yb, grad_down_yb)
            phi_face_yb = phi[i, jm] + T(0.5) * slope_yb * max(zero(T), one(T) - abs(v))
        else
            grad_up_yb   = p - phi[i, jp]
            grad_down_yb = phi[i, jm] - p
            slope_yb = _minmod(grad_up_yb, grad_down_yb)
            phi_face_yb = p - T(0.5) * slope_yb * max(zero(T), one(T) - abs(v))
        end
        v_bot = (uy[i, jm] + v) / T(2)
        flux_y_bot = v_bot * phi_face_yb

        # Update: unsplit advection
        phi_new[i, j] = p - (flux_x - flux_x_left) - (flux_y - flux_y_bot)
    end
end

function advect_ls_2d!(phi_new, phi, ux, uy, Nx, Ny)
    backend = KernelAbstractions.get_backend(phi)
    kernel! = advect_ls_2d_kernel!(backend)
    kernel!(phi_new, phi, ux, uy, Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

# --- Redistanciation: restore |∇φ| = 1 ---
#
# PDE: ∂φ/∂τ + S(φ₀)(|∇φ| - 1) = 0
# S(φ₀) = φ₀ / √(φ₀² + dx²)  smoothed sign function
# Godunov upwind scheme for |∇φ|

@kernel function reinit_ls_2d_kernel!(phi_new, @Const(phi), @Const(phi0), dtau, Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(phi)
        ip = ifelse(i < Nx, i + 1, 1)
        im = ifelse(i > 1,  i - 1, Nx)
        jp = ifelse(j < Ny, j + 1, 1)
        jm = ifelse(j > 1,  j - 1, Ny)

        p = phi[i, j]

        # One-sided differences (dx = 1)
        a = p - phi[im, j]   # D⁻x
        b = phi[ip, j] - p   # D⁺x
        c = p - phi[i, jm]   # D⁻y
        d = phi[i, jp] - p   # D⁺y

        # Smoothed sign from original φ₀
        p0 = phi0[i, j]
        S = p0 / sqrt(p0^2 + one(T))

        # Godunov upwind gradient magnitude
        if S >= zero(T)
            ap = max(a, zero(T))
            bm = min(b, zero(T))
            cp = max(c, zero(T))
            dm = min(d, zero(T))
            G = sqrt(max(ap^2, bm^2) + max(cp^2, dm^2))
        else
            am = min(a, zero(T))
            bp = max(b, zero(T))
            cm = min(c, zero(T))
            dp = max(d, zero(T))
            G = sqrt(max(am^2, bp^2) + max(cm^2, dp^2))
        end

        phi_new[i, j] = p - dtau * S * (G - one(T))
    end
end

"""
    reinit_ls_2d!(phi, phi_work, phi0, Nx, Ny; n_iter=5, dtau=0.5)

Redistance φ to restore |∇φ| ≈ 1 using iterative PDE method.
`phi_work` and `phi0` are pre-allocated work arrays of same size.
"""
function reinit_ls_2d!(phi, phi_work, phi0, Nx, Ny; n_iter=5, dtau=0.5)
    backend = KernelAbstractions.get_backend(phi)
    T = eltype(phi)
    kernel! = reinit_ls_2d_kernel!(backend)
    copyto!(phi0, phi)  # save original for sign function
    for _ in 1:n_iter
        kernel!(phi_work, phi, phi0, T(dtau), Nx, Ny; ndrange=(Nx, Ny))
        KernelAbstractions.synchronize(backend)
        copyto!(phi, phi_work)
    end
end

# --- Curvature from level-set: κ = -∇·(∇φ/|∇φ|) ---

@kernel function curvature_ls_2d_kernel!(κ, @Const(phi), Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(phi)
        ip = ifelse(i < Nx, i + 1, 1)
        im = ifelse(i > 1,  i - 1, Nx)
        jp = ifelse(j < Ny, j + 1, 1)
        jm = ifelse(j > 1,  j - 1, Ny)

        # Periodic corner indices
        ip_jp = ifelse(i < Nx, i + 1, 1)
        im_jp = ifelse(i > 1,  i - 1, Nx)
        ip_jm = ifelse(i < Nx, i + 1, 1)
        im_jm = ifelse(i > 1,  i - 1, Nx)

        # First derivatives (central)
        phi_x  = (phi[ip, j] - phi[im, j]) / T(2)
        phi_y  = (phi[i, jp] - phi[i, jm]) / T(2)

        # Second derivatives
        phi_xx = phi[ip, j] - T(2) * phi[i, j] + phi[im, j]
        phi_yy = phi[i, jp] - T(2) * phi[i, j] + phi[i, jm]
        phi_xy = (phi[ip_jp, jp] - phi[im_jp, jp] -
                  phi[ip_jm, jm] + phi[im_jm, jm]) / T(4)

        grad_sq = phi_x^2 + phi_y^2 + T(1e-30)
        grad_mag3 = grad_sq * sqrt(grad_sq)

        # κ = -div(∇φ/|∇φ|) = -(φ_xx·φ_y² - 2·φ_xy·φ_x·φ_y + φ_yy·φ_x²) / |∇φ|³
        κ[i, j] = -(phi_xx * phi_y^2 - T(2) * phi_xy * phi_x * phi_y +
                     phi_yy * phi_x^2) / grad_mag3
    end
end

function curvature_ls_2d!(κ, phi, Nx, Ny)
    backend = KernelAbstractions.get_backend(phi)
    kernel! = curvature_ls_2d_kernel!(backend)
    kernel!(κ, phi, Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

# --- Azimuthal curvature for axisymmetric LS ---
#
# In axisymmetric coords (x=z, y=r), the total curvature is:
#   κ_total = κ_meridional + κ_azimuthal
# where κ_meridional = -div(∇φ/|∇φ|) (from curvature_ls_2d!)
# and   κ_azimuthal  = -(1/r)·(φ_r / |∇φ|) = -(1/r)·n_r
#
# This kernel ADDS the azimuthal component to an existing κ field.

@kernel function add_azimuthal_curvature_ls_2d_kernel!(κ, @Const(phi), Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(phi)
        Nx = size(phi, 1)
        ip = ifelse(i < Nx, i + 1, 1)
        im = ifelse(i > 1,  i - 1, Nx)
        jp = ifelse(j < Ny, j + 1, 1)
        jm = ifelse(j > 1,  j - 1, Ny)

        phi_x = (phi[ip, j] - phi[im, j]) / T(2)
        phi_y = (phi[i, jp] - phi[i, jm]) / T(2)
        grad_mag = sqrt(phi_x^2 + phi_y^2) + T(1e-30)

        # n_r = φ_y / |∇φ|  (radial component of normal)
        n_r = phi_y / grad_mag

        # Radial position (y=r, cell center at j-0.5)
        r = max(T(j) - T(0.5), one(T))  # clamp r ≥ 1 near axis

        # κ_azimuthal = -n_r / r
        κ_axi = -n_r / r

        # Add and clamp total curvature
        κ[i, j] = clamp(κ[i, j] + κ_axi, -T(0.5), T(0.5))
    end
end

"""
    add_azimuthal_curvature_ls_2d!(κ, phi, Ny)

Add azimuthal curvature κ₂ = -n_r/r to existing meridional curvature.
For axisymmetric geometry with y = r (radial coordinate).
"""
function add_azimuthal_curvature_ls_2d!(κ, phi, Ny)
    backend = KernelAbstractions.get_backend(κ)
    Nx = size(κ, 1)
    kernel! = add_azimuthal_curvature_ls_2d_kernel!(backend)
    kernel!(κ, phi, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end
