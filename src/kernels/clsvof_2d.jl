using KernelAbstractions

# ===========================================================================
# CLSVOF 2D — Coupled Level-Set / Volume-of-Fluid
#
# VOF (C): conservative mass transport
# LS  (φ): smooth curvature via κ = -∇·(∇φ/|∇φ|)
# Coupling: reconstruct φ from C, then redistance
#
# References:
# - Sussman & Puckett (2000) doi:10.1006/jcph.2000.6537
# - Son & Hur (2002) doi:10.1006/jcph.2002.7118
# ===========================================================================

# --- Reconstruct φ from VOF volume fraction C ---
#
# Simple mapping: φ = (2C - 1) · Δ where Δ is a length scale.
# φ > 0 where C > 0.5 (liquid), φ < 0 where C < 0.5 (gas).
# Must be followed by redistanciation to obtain a proper signed distance.

@kernel function ls_from_vof_2d_kernel!(phi, @Const(C), half_width, Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(phi)
        phi[i, j] = (T(2) * C[i, j] - one(T)) * half_width
    end
end

"""
    ls_from_vof_2d!(phi, C, Nx, Ny; half_width=1.0)

Reconstruct level-set φ from volume fraction C.
Sets φ = (2C-1)·half_width. Must be followed by `reinit_ls_2d!`.
"""
function ls_from_vof_2d!(phi, C, Nx, Ny; half_width=1.0)
    backend = KernelAbstractions.get_backend(C)
    T = eltype(C)
    kernel! = ls_from_vof_2d_kernel!(backend)
    kernel!(phi, C, T(half_width), Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

# --- Surface tension from CLSVOF: F = σ·κ·δ_ε(φ)·∇φ ---
#
# δ_ε(φ) = (1 + cos(πφ/ε)) / (2ε)  for |φ| < ε, else 0
# Localizes the force to a band of width 2ε around the interface.

@kernel function surface_tension_clsvof_2d_kernel!(Fx, Fy, @Const(κ), @Const(phi),
                                                      σ, ε, Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(phi)
        ip = ifelse(i < Nx, i + 1, 1)
        im = ifelse(i > 1,  i - 1, Nx)
        jp = ifelse(j < Ny, j + 1, 1)
        jm = ifelse(j > 1,  j - 1, Ny)

        # ∇φ (central differences)
        phi_x = (phi[ip, j] - phi[im, j]) / T(2)
        phi_y = (phi[i, jp] - phi[i, jm]) / T(2)

        # Smoothed delta function
        abs_phi = abs(phi[i, j])
        delta_e = ifelse(abs_phi < ε,
                         (one(T) + cos(T(π) * phi[i, j] / ε)) / (T(2) * ε),
                         zero(T))

        # CSF force: F = σ·κ·δ_ε(φ)·∇φ
        Fx[i, j] = σ * κ[i, j] * delta_e * phi_x
        Fy[i, j] = σ * κ[i, j] * delta_e * phi_y
    end
end

"""
    surface_tension_clsvof_2d!(Fx, Fy, κ, phi, σ, Nx, Ny; epsilon=1.5)

Compute surface tension force using level-set curvature and smoothed
delta function. `epsilon` controls the support width (default 1.5 dx).
"""
function surface_tension_clsvof_2d!(Fx, Fy, κ, phi, σ, Nx, Ny; epsilon=1.5)
    backend = KernelAbstractions.get_backend(phi)
    T = eltype(phi)
    kernel! = surface_tension_clsvof_2d_kernel!(backend)
    kernel!(Fx, Fy, κ, phi, T(σ), T(epsilon), Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end
