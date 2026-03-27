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

# --- Level-set advection (first-order upwind) ---

@kernel function advect_ls_2d_kernel!(phi_new, @Const(phi), @Const(ux), @Const(uy), Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(phi)
        ip = ifelse(i < Nx, i + 1, 1)
        im = ifelse(i > 1,  i - 1, Nx)
        jp = ifelse(j < Ny, j + 1, 1)
        jm = ifelse(j > 1,  j - 1, Ny)

        p = phi[i, j]
        u = ux[i, j]
        v = uy[i, j]

        # Upwind differences
        dphi_dx = ifelse(u > zero(T), p - phi[im, j], phi[ip, j] - p)
        dphi_dy = ifelse(v > zero(T), p - phi[i, jm], phi[i, jp] - p)

        # dt = dx = 1 in lattice units
        phi_new[i, j] = p - (u * dphi_dx + v * dphi_dy)
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
