# Phase-field (Allen-Cahn + pressure-based LBM) for two-phase flows
# with density ratios up to 1000:1.
#
# Two D2Q9 distributions:
#   f_q : pressure-based Navier-Stokes (MRT collision)
#   g_q : Allen-Cahn equation for order parameter φ ∈ [-1, 1]
#
# Reference: Fakhari, Mitchell, Bolster & Leonardi (2017) JCP 334:620-638
#
# Convention:
#   φ = +1 : liquid,  φ = -1 : gas
#   C = (1 + φ)/2 : volume fraction
#   ρ(φ) = (ρ_l + ρ_g)/2 + (ρ_l - ρ_g)/2 · φ
#
# Free energy: f(φ) = β(φ² - 1)²/4
# Chemical potential: μ = β·φ(φ² - 1) - κ·∇²φ
# Parameters: κ = 3σW/4, β = 3σ/(2W)
# Equilibrium profile: φ = tanh(x/W)
# Surface tension: σ = (2√2/3)·√(κβ)
# Force: F_σ = μ·∇φ

using KernelAbstractions

# --- Parameter helper ---

"""
    phasefield_params(σ, W)

Compute Cahn-Hilliard parameters (β, κ) from surface tension σ and interface width W.

Returns named tuple `(β, κ)` where:
- β = 3σ/(2W): double-well height
- κ = 3σW/4: gradient energy coefficient
- Equilibrium profile: φ = tanh(x/W)
"""
function phasefield_params(σ, W)
    return (β = 3σ / (2W), κ = 3σ * W / 4)
end

# --- Compute φ from g distributions ---

@kernel function compute_phi_2d_kernel!(φ, @Const(g))
    i, j = @index(Global, NTuple)
    @inbounds begin
        φ[i,j] = g[i,j,1] + g[i,j,2] + g[i,j,3] + g[i,j,4] + g[i,j,5] +
                  g[i,j,6] + g[i,j,7] + g[i,j,8] + g[i,j,9]
    end
end

"""
    compute_phi_2d!(φ, g)

Compute order parameter φ = Σg_q from Allen-Cahn distributions.
"""
function compute_phi_2d!(φ, g)
    backend = KernelAbstractions.get_backend(g)
    Nx, Ny = size(φ)
    kernel! = compute_phi_2d_kernel!(backend)
    kernel!(φ, g; ndrange=(Nx, Ny))
end

# --- Compute VOF C = (1+φ)/2 from order parameter ---

@kernel function compute_vof_from_phi_2d_kernel!(C, @Const(φ))
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(φ)
        C[i,j] = (one(T) + φ[i,j]) / T(2)
    end
end

"""
    compute_vof_from_phi_2d!(C, φ)

Compute volume fraction C = (1+φ)/2 from order parameter φ ∈ [-1,1].
"""
function compute_vof_from_phi_2d!(C, φ)
    backend = KernelAbstractions.get_backend(φ)
    Nx, Ny = size(φ)
    kernel! = compute_vof_from_phi_2d_kernel!(backend)
    kernel!(C, φ; ndrange=(Nx, Ny))
end

# --- Chemical potential ---

@kernel function compute_chemical_potential_2d_kernel!(μ, @Const(φ), β, κ, Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(φ)
        φ_c = φ[i,j]
        # 5-point Laplacian (zero-gradient BC at domain boundaries)
        ip = min(i + 1, Nx); im = max(i - 1, 1)
        jp = min(j + 1, Ny); jm = max(j - 1, 1)
        lap_φ = φ[ip,j] + φ[im,j] + φ[i,jp] + φ[i,jm] - T(4) * φ_c
        # μ = β·φ(φ²-1) - κ·∇²φ
        μ[i,j] = β * φ_c * (φ_c * φ_c - one(T)) - κ * lap_φ
    end
end

"""
    compute_chemical_potential_2d!(μ, φ, β, κ)

Compute chemical potential μ = β·φ(φ²-1) - κ·∇²φ from order parameter φ.
Uses 5-point Laplacian with zero-gradient boundary conditions.
"""
function compute_chemical_potential_2d!(μ, φ, β, κ)
    backend = KernelAbstractions.get_backend(φ)
    Nx, Ny = size(φ)
    T = eltype(φ)
    kernel! = compute_chemical_potential_2d_kernel!(backend)
    kernel!(μ, φ, T(β), T(κ), Int32(Nx), Int32(Ny); ndrange=(Nx, Ny))
end

# --- Azimuthal correction to chemical potential (axisymmetric) ---
# In cylindrical coordinates: ∇²φ = ∂²φ/∂z² + ∂²φ/∂r² + (1/r)·∂φ/∂r
# The 2D Cartesian Laplacian misses (1/r)·∂φ/∂r → correct μ:
# μ_axisym = μ_cart - κ·(1/r)·∂φ/∂r

@kernel function add_azimuthal_chemical_potential_2d_kernel!(μ, @Const(φ), κ, Ny)
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(φ)
        if j == 1
            # Axis: L'Hôpital rule, (1/r)∂φ/∂r → ∂²φ/∂r² ≈ 2(φ[i,2]-φ[i,1])
            correction = κ * T(2) * (φ[i,2] - φ[i,1])
        else
            r = T(j) - T(0.5)
            jp = min(j + 1, Int(Ny)); jm = j - 1
            dφ_dr = (φ[i,jp] - φ[i,jm]) / T(2)
            correction = κ * dφ_dr / r
        end
        μ[i,j] -= correction
    end
end

"""
    add_azimuthal_chemical_potential_2d!(μ, φ, κ, Ny)

Add azimuthal curvature correction to chemical potential for axisymmetric flows.
Subtracts κ·(1/r)·∂φ/∂r from μ (cylindrical Laplacian correction).
y-coordinate = radial (j=1 is axis).
"""
function add_azimuthal_chemical_potential_2d!(μ, φ, κ, Ny)
    backend = KernelAbstractions.get_backend(φ)
    Nx = size(φ, 1)
    T = eltype(φ)
    kernel! = add_azimuthal_chemical_potential_2d_kernel!(backend)
    kernel!(μ, φ, T(κ), Int32(Ny); ndrange=(Nx, Ny))
end

# --- Surface tension force from phase-field ---

@kernel function compute_phasefield_force_2d_kernel!(Fx, Fy, @Const(μ), @Const(φ), Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(φ)
        ip = min(i + 1, Nx); im = max(i - 1, 1)
        jp = min(j + 1, Ny); jm = max(j - 1, 1)
        dφ_dx = (φ[ip,j] - φ[im,j]) / T(2)
        dφ_dy = (φ[i,jp] - φ[i,jm]) / T(2)
        μ_c = μ[i,j]
        Fx[i,j] = μ_c * dφ_dx
        Fy[i,j] = μ_c * dφ_dy
    end
end

"""
    compute_phasefield_force_2d!(Fx, Fy, μ, φ)

Compute surface tension force F = μ·∇φ from chemical potential and order parameter.
Central differences for ∇φ, zero-gradient at domain boundaries.
"""
function compute_phasefield_force_2d!(Fx, Fy, μ, φ)
    backend = KernelAbstractions.get_backend(φ)
    Nx, Ny = size(φ)
    kernel! = compute_phasefield_force_2d_kernel!(backend)
    kernel!(Fx, Fy, μ, φ, Int32(Nx), Int32(Ny); ndrange=(Nx, Ny))
end

# --- Antidiffusion flux for conservative Allen-Cahn ---
# A = (1-φ²) · ∇φ / (|∇φ| + ε)   (= (1-φ²) · n̂)

@kernel function compute_antidiffusion_flux_2d_kernel!(Ax, Ay, @Const(φ), Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(φ)
        eps = T(1e-8)
        ip = min(i + 1, Nx); im = max(i - 1, 1)
        jp = min(j + 1, Ny); jm = max(j - 1, 1)

        dφx = (φ[ip,j] - φ[im,j]) / T(2)
        dφy = (φ[i,jp] - φ[i,jm]) / T(2)
        grad_mag = sqrt(dφx * dφx + dφy * dφy) + eps

        coeff = (one(T) - φ[i,j] * φ[i,j]) / grad_mag
        Ax[i,j] = coeff * dφx
        Ay[i,j] = coeff * dφy
    end
end

"""
    compute_antidiffusion_flux_2d!(Ax, Ay, φ)

Compute antidiffusion flux A = (1-φ²)·∇φ/|∇φ| for the conservative Allen-Cahn equation.
"""
function compute_antidiffusion_flux_2d!(Ax, Ay, φ)
    backend = KernelAbstractions.get_backend(φ)
    Nx, Ny = size(φ)
    kernel! = compute_antidiffusion_flux_2d_kernel!(backend)
    kernel!(Ax, Ay, φ, Int32(Nx), Int32(Ny); ndrange=(Nx, Ny))
end

# --- Conservative Allen-Cahn BGK collision for g_q ---
#
# Conservative Allen-Cahn (Chiu & Lin 2011):
#   ∂φ/∂t + ∇·(φu) = ∇·[D(∇φ + (1/W)(1-φ²)·n̂)]
#                    = D∇²φ + (D/W)·∇·[(1-φ²)·n̂]
#
# The LBM diffusion provides D∇²φ.
# Source: R = -(D/W)·∇·A  where A = (1-φ²)·n̂  (precomputed)
# NOTE: minus sign because ∇·A gives the NEGATIVE of the sharpening term
# for the tanh profile.
#
# For a flat interface this reduces to R = (2D/W²)·φ(1-φ²).
# For curved interfaces it includes curvature correction that conserves mass.
#
# Equilibrium: g_q^eq = w_q · φ · (1 + 3·c_q·u)

@kernel function collide_allen_cahn_2d_kernel!(g, @Const(ux), @Const(uy),
                                                @Const(Ax), @Const(Ay),
                                                inv_tau, src_pref, neg_D_over_W, Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(g)

        g1 = g[i,j,1]; g2 = g[i,j,2]; g3 = g[i,j,3]; g4 = g[i,j,4]
        g5 = g[i,j,5]; g6 = g[i,j,6]; g7 = g[i,j,7]; g8 = g[i,j,8]; g9 = g[i,j,9]

        phi = g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8 + g9
        u = ux[i,j]
        v = uy[i,j]

        # Equilibrium: g_eq = w_q · φ · (1 + 3·c_q·u)
        g1_eq = T(4/9)  * phi
        g2_eq = T(1/9)  * phi * (one(T) + T(3) * u)
        g3_eq = T(1/9)  * phi * (one(T) + T(3) * v)
        g4_eq = T(1/9)  * phi * (one(T) - T(3) * u)
        g5_eq = T(1/9)  * phi * (one(T) - T(3) * v)
        g6_eq = T(1/36) * phi * (one(T) + T(3) * (u + v))
        g7_eq = T(1/36) * phi * (one(T) + T(3) * (-u + v))
        g8_eq = T(1/36) * phi * (one(T) + T(3) * (-u - v))
        g9_eq = T(1/36) * phi * (one(T) + T(3) * (u - v))

        # Conservative source: R = -(D/W) · ∇·A
        ip = min(i + 1, Int(Nx)); im = max(i - 1, 1)
        jp = min(j + 1, Int(Ny)); jm = max(j - 1, 1)
        div_A = (Ax[ip,j] - Ax[im,j]) / T(2) + (Ay[i,jp] - Ay[i,jm]) / T(2)
        R = neg_D_over_W * div_A

        # Collision + source
        g[i,j,1] = g1 - inv_tau * (g1 - g1_eq) + T(4/9)  * src_pref * R
        g[i,j,2] = g2 - inv_tau * (g2 - g2_eq) + T(1/9)  * src_pref * R
        g[i,j,3] = g3 - inv_tau * (g3 - g3_eq) + T(1/9)  * src_pref * R
        g[i,j,4] = g4 - inv_tau * (g4 - g4_eq) + T(1/9)  * src_pref * R
        g[i,j,5] = g5 - inv_tau * (g5 - g5_eq) + T(1/9)  * src_pref * R
        g[i,j,6] = g6 - inv_tau * (g6 - g6_eq) + T(1/36) * src_pref * R
        g[i,j,7] = g7 - inv_tau * (g7 - g7_eq) + T(1/36) * src_pref * R
        g[i,j,8] = g8 - inv_tau * (g8 - g8_eq) + T(1/36) * src_pref * R
        g[i,j,9] = g9 - inv_tau * (g9 - g9_eq) + T(1/36) * src_pref * R
    end
end

"""
    collide_allen_cahn_2d!(g, ux, uy, Ax, Ay; τ_g=0.7, W=5.0)

Conservative Allen-Cahn BGK collision for D2Q9 distributions g_q.

Uses the conservative formulation (Chiu & Lin 2011) with precomputed
antidiffusion flux A = (1-φ²)·n̂. Call `compute_antidiffusion_flux_2d!`
before this to fill Ax, Ay.

The conservative source R = -(D/W)·∇·A preserves mass for curved interfaces,
unlike the non-conservative R = (2D/W²)·φ(1-φ²) which causes droplet shrinkage.
"""
function collide_allen_cahn_2d!(g, ux, uy, Ax, Ay; τ_g=0.7, W=5.0)
    backend = KernelAbstractions.get_backend(g)
    Nx, Ny = size(g, 1), size(g, 2)
    T = eltype(g)
    tau = T(τ_g)
    inv_tau = one(T) / tau
    src_pref = one(T) - inv_tau / T(2)
    D_g = (tau - T(0.5)) / T(3)
    neg_D_over_W = -D_g / T(W)
    kernel! = collide_allen_cahn_2d_kernel!(backend)
    kernel!(g, ux, uy, Ax, Ay, inv_tau, src_pref, neg_D_over_W,
            Int32(Nx), Int32(Ny); ndrange=(Nx, Ny))
end

# --- Azimuthal diffusion source for Allen-Cahn (axisymmetric) ---
# Missing cylindrical term: D·(1/r)·∂φ/∂r
# Added as post-collision source correction.

@kernel function add_azimuthal_allen_cahn_source_2d_kernel!(g, @Const(φ), src_pref, D_g, Ny)
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(g)
        if j == 1
            # Axis: L'Hôpital, (1/r)∂φ/∂r → ∂²φ/∂r² ≈ 2(φ[i,2]-φ[i,1])
            S = src_pref * D_g * T(2) * (φ[i,2] - φ[i,1])
        else
            r = T(j) - T(0.5)
            jp = min(j + 1, Int(Ny)); jm = j - 1
            dφ_dr = (φ[i,jp] - φ[i,jm]) / T(2)
            S = src_pref * D_g * dφ_dr / r
        end
        g[i,j,1] += T(4/9)  * S
        g[i,j,2] += T(1/9)  * S
        g[i,j,3] += T(1/9)  * S
        g[i,j,4] += T(1/9)  * S
        g[i,j,5] += T(1/9)  * S
        g[i,j,6] += T(1/36) * S
        g[i,j,7] += T(1/36) * S
        g[i,j,8] += T(1/36) * S
        g[i,j,9] += T(1/36) * S
    end
end

"""
    add_azimuthal_allen_cahn_source_2d!(g, φ; τ_g=0.7)

Add cylindrical diffusion correction D·(1/r)·∂φ/∂r to Allen-Cahn distributions.
Call after `collide_allen_cahn_2d!` for axisymmetric simulations.
y-coordinate = radial (j=1 is axis).
"""
function add_azimuthal_allen_cahn_source_2d!(g, φ; τ_g=0.7)
    backend = KernelAbstractions.get_backend(g)
    Nx, Ny = size(g, 1), size(g, 2)
    T = eltype(g)
    tau = T(τ_g)
    inv_tau = one(T) / tau
    src_pref = one(T) - inv_tau / T(2)
    D_g = (tau - T(0.5)) / T(3)
    kernel! = add_azimuthal_allen_cahn_source_2d_kernel!(backend)
    kernel!(g, φ, src_pref, D_g, Int32(Ny); ndrange=(Nx, Ny))
end

# --- Pressure-based MRT collision for two-phase phase-field ---
#
# Modified equilibrium (Fakhari et al. 2017):
#   f_q^eq = w_q · [ρ_lbm + ρ(φ)·(3c·u + 4.5(c·u)² - 1.5u²)]
#
# Key properties:
#   Σf_eq = ρ_lbm (≈1, distributions stay O(1) for any density ratio)
#   Σf_eq·c = ρ(φ)·u (first moment = physical momentum)
#
# Velocity: u = (j + F/2)/ρ(φ) using physical density
# Viscosity: ν(φ) interpolated, s_ν = 1/(3ν + 0.5)
# Force: F = F_σ (physical surface tension, added to j moments)

@kernel function collide_pressure_phasefield_mrt_2d_kernel!(f, @Const(φ),
                                                              @Const(Fx), @Const(Fy),
                                                              @Const(is_solid),
                                                              ρ_l, ρ_g, ν_l, ν_g,
                                                              s_e, s_eps, s_q)
    i, j = @index(Global, NTuple)
    @inbounds begin
        if is_solid[i,j]
            tmp = f[i,j,2]; f[i,j,2] = f[i,j,4]; f[i,j,4] = tmp
            tmp = f[i,j,3]; f[i,j,3] = f[i,j,5]; f[i,j,5] = tmp
            tmp = f[i,j,6]; f[i,j,6] = f[i,j,8]; f[i,j,8] = tmp
            tmp = f[i,j,7]; f[i,j,7] = f[i,j,9]; f[i,j,9] = tmp
        else
            T = eltype(f)
            f1=f[i,j,1]; f2=f[i,j,2]; f3=f[i,j,3]; f4=f[i,j,4]
            f5=f[i,j,5]; f6=f[i,j,6]; f7=f[i,j,7]; f8=f[i,j,8]; f9=f[i,j,9]

            # Physical density from phase-field (with floor for stability)
            phi_c = φ[i,j]
            C_local = (one(T) + phi_c) / T(2)
            ρ_raw = C_local * ρ_l + (one(T) - C_local) * ρ_g
            ρ_phys = max(ρ_raw, ρ_l * T(0.01))

            # Per-node kinematic viscosity
            ν_local = C_local * ν_l + (one(T) - C_local) * ν_g
            s_nu = one(T) / (T(3) * ν_local + T(0.5))

            fx = Fx[i,j]
            fy = Fy[i,j]

            # Transform to moment space (same M as standard MRT)
            ρ_lbm = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
            e   = -T(4)*f1 - f2 - f3 - f4 - f5 + T(2)*(f6+f7+f8+f9)
            eps = T(4)*f1 - T(2)*(f2+f3+f4+f5) + f6+f7+f8+f9
            jx  = f2 - f4 + f6 - f7 - f8 + f9
            qx  = -T(2)*f2 + T(2)*f4 + f6 - f7 - f8 + f9
            jy  = f3 - f5 + f6 + f7 - f8 - f9
            qy  = -T(2)*f3 + T(2)*f5 + f6 + f7 - f8 - f9
            pxx = f2 - f3 + f4 - f5
            pxy = f6 - f7 + f8 - f9

            # Physical velocity: u = (j + F/2) / ρ(φ)
            inv_ρ_phys = one(T) / ρ_phys
            ux = (jx + fx / T(2)) * inv_ρ_phys
            uy = (jy + fy / T(2)) * inv_ρ_phys
            usq = ux * ux + uy * uy

            # Modified equilibrium moments (pressure-based):
            # ρ-moment uses ρ_lbm, velocity terms use ρ_phys
            e_eq   = -T(2) * ρ_lbm + T(3) * ρ_phys * usq
            eps_eq = ρ_lbm - T(3) * ρ_phys * usq
            qx_eq  = -ρ_phys * ux
            qy_eq  = -ρ_phys * uy
            pxx_eq = ρ_phys * (ux * ux - uy * uy)
            pxy_eq = ρ_phys * ux * uy

            # Force: add to momentum moments (Guo scheme)
            jx_force = fx
            jy_force = fy

            # Relax non-conserved moments
            e_star   = e   - s_e   * (e   - e_eq)
            eps_star = eps - s_eps * (eps - eps_eq)
            jx_star  = jx + jx_force
            jy_star  = jy + jy_force
            qx_star  = qx  - s_q  * (qx  - qx_eq)
            qy_star  = qy  - s_q  * (qy  - qy_eq)
            pxx_star = pxx - s_nu * (pxx - pxx_eq)
            pxy_star = pxy - s_nu * (pxy - pxy_eq)

            # Transform back: f* = M⁻¹·m*
            r = ρ_lbm; es = e_star; ep = eps_star
            jxs = jx_star; qxs = qx_star; jys = jy_star; qys = qy_star
            ps = pxx_star; pxys = pxy_star

            f[i,j,1] = T(1/9)*r  - T(1/9)*es  + T(1/9)*ep
            f[i,j,2] = T(1/9)*r  - T(1/36)*es - T(1/18)*ep + T(1/6)*jxs - T(1/6)*qxs + T(1/4)*ps
            f[i,j,3] = T(1/9)*r  - T(1/36)*es - T(1/18)*ep + T(1/6)*jys - T(1/6)*qys - T(1/4)*ps
            f[i,j,4] = T(1/9)*r  - T(1/36)*es - T(1/18)*ep - T(1/6)*jxs + T(1/6)*qxs + T(1/4)*ps
            f[i,j,5] = T(1/9)*r  - T(1/36)*es - T(1/18)*ep - T(1/6)*jys + T(1/6)*qys - T(1/4)*ps
            f[i,j,6] = T(1/9)*r  + T(1/18)*es + T(1/36)*ep + T(1/6)*jxs + T(1/12)*qxs + T(1/6)*jys + T(1/12)*qys + T(1/4)*pxys
            f[i,j,7] = T(1/9)*r  + T(1/18)*es + T(1/36)*ep - T(1/6)*jxs - T(1/12)*qxs + T(1/6)*jys + T(1/12)*qys - T(1/4)*pxys
            f[i,j,8] = T(1/9)*r  + T(1/18)*es + T(1/36)*ep - T(1/6)*jxs - T(1/12)*qxs - T(1/6)*jys - T(1/12)*qys + T(1/4)*pxys
            f[i,j,9] = T(1/9)*r  + T(1/18)*es + T(1/36)*ep + T(1/6)*jxs + T(1/12)*qxs - T(1/6)*jys - T(1/12)*qys - T(1/4)*pxys
        end
    end
end

"""
    collide_pressure_phasefield_mrt_2d!(f, φ, Fx, Fy, is_solid;
        ρ_l=1.0, ρ_g=0.001, ν_l=0.1, ν_g=0.1, s_e=1.4, s_eps=1.4, s_q=1.2)

Pressure-based MRT collision for two-phase phase-field LBM.

Modified equilibrium: velocity terms weighted by ρ(φ), density term by ρ_lbm.
This keeps distributions O(1) for any density ratio (up to 1000:1).
Physical velocity u = (j + F/2)/ρ(φ) computed from physical density.
"""
function collide_pressure_phasefield_mrt_2d!(f, φ, Fx, Fy, is_solid;
                                              ρ_l=1.0, ρ_g=0.001, ν_l=0.1, ν_g=0.1,
                                              s_e=1.4, s_eps=1.4, s_q=1.2)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    T = eltype(f)
    kernel! = collide_pressure_phasefield_mrt_2d_kernel!(backend)
    kernel!(f, φ, Fx, Fy, is_solid,
            T(ρ_l), T(ρ_g), T(ν_l), T(ν_g), T(s_e), T(s_eps), T(s_q);
            ndrange=(Nx, Ny))
end

# --- Macroscopic quantities (pressure-based) ---

@kernel function compute_macroscopic_phasefield_2d_kernel!(p, ux, uy,
                                                            @Const(f), @Const(φ),
                                                            @Const(Fx), @Const(Fy),
                                                            ρ_l, ρ_g)
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(f)
        f1=f[i,j,1]; f2=f[i,j,2]; f3=f[i,j,3]; f4=f[i,j,4]
        f5=f[i,j,5]; f6=f[i,j,6]; f7=f[i,j,7]; f8=f[i,j,8]; f9=f[i,j,9]

        # LBM density (≈1) and pressure
        ρ_lbm = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
        p[i,j] = ρ_lbm / T(3)

        # Physical density from φ (with floor for stability at high ρ_ratio)
        C_local = (one(T) + φ[i,j]) / T(2)
        ρ_raw = C_local * ρ_l + (one(T) - C_local) * ρ_g
        ρ_phys = max(ρ_raw, ρ_l * T(0.01))

        # Momentum and physical velocity
        jx = f2 - f4 + f6 - f7 - f8 + f9
        jy = f3 - f5 + f6 + f7 - f8 - f9
        inv_ρ = one(T) / ρ_phys
        ux[i,j] = (jx + Fx[i,j] / T(2)) * inv_ρ
        uy[i,j] = (jy + Fy[i,j] / T(2)) * inv_ρ
    end
end

"""
    compute_macroscopic_phasefield_2d!(p, ux, uy, f, φ, Fx, Fy; ρ_l=1.0, ρ_g=0.001)

Pressure-based macroscopic computation for phase-field two-phase flows.

- p = cs²·Σf (LBM pressure)
- u = (j + F/2)/ρ(φ) (physical velocity from physical density)
"""
function compute_macroscopic_phasefield_2d!(p, ux, uy, f, φ, Fx, Fy;
                                             ρ_l=1.0, ρ_g=0.001)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(p)
    T = eltype(f)
    kernel! = compute_macroscopic_phasefield_2d_kernel!(backend)
    kernel!(p, ux, uy, f, φ, Fx, Fy, T(ρ_l), T(ρ_g); ndrange=(Nx, Ny))
end

# --- Inlet BC for Allen-Cahn distributions ---

@kernel function set_phasefield_west_2d_kernel!(g, @Const(φ_inlet), @Const(ux_inlet))
    j = @index(Global)
    @inbounds begin
        T = eltype(g)
        phi = φ_inlet[j]
        u = ux_inlet[j]
        # g_eq = w_q · φ · (1 + 3·c_qx·u)  (uy=0 at inlet)
        g[1,j,1] = T(4/9)  * phi
        g[1,j,2] = T(1/9)  * phi * (one(T) + T(3) * u)
        g[1,j,3] = T(1/9)  * phi
        g[1,j,4] = T(1/9)  * phi * (one(T) - T(3) * u)
        g[1,j,5] = T(1/9)  * phi
        g[1,j,6] = T(1/36) * phi * (one(T) + T(3) * u)
        g[1,j,7] = T(1/36) * phi * (one(T) - T(3) * u)
        g[1,j,8] = T(1/36) * phi * (one(T) - T(3) * u)
        g[1,j,9] = T(1/36) * phi * (one(T) + T(3) * u)
    end
end

"""
    set_phasefield_west_2d!(g, φ_inlet, ux_inlet)

Set Allen-Cahn distributions at west (i=1) boundary to equilibrium
with prescribed φ and velocity profiles.
"""
function set_phasefield_west_2d!(g, φ_inlet, ux_inlet)
    backend = KernelAbstractions.get_backend(g)
    Ny = size(g, 2)
    kernel! = set_phasefield_west_2d_kernel!(backend)
    kernel!(g, φ_inlet, ux_inlet; ndrange=(Ny,))
end

# --- Outlet BC: zero-gradient extrapolation for g ---

@kernel function extrapolate_phasefield_east_2d_kernel!(g, Nx)
    j = @index(Global)
    @inbounds begin
        for q in 1:9
            g[Nx,j,q] = g[Nx-1,j,q]
        end
    end
end

"""
    extrapolate_phasefield_east_2d!(g, Nx, Ny)

Zero-gradient (Neumann) extrapolation of Allen-Cahn distributions at east boundary.
"""
function extrapolate_phasefield_east_2d!(g, Nx, Ny)
    backend = KernelAbstractions.get_backend(g)
    kernel! = extrapolate_phasefield_east_2d_kernel!(backend)
    kernel!(g, Int32(Nx); ndrange=(Ny,))
end

# --- CPU initializers ---

"""
    init_phasefield_equilibrium(φ, ux, uy, FT=Float64)

Initialize Allen-Cahn distributions g_q at equilibrium from φ and velocity fields (CPU).
Returns 3D array g[Nx, Ny, 9].
"""
function init_phasefield_equilibrium(φ, ux, uy, FT=Float64)
    Nx, Ny = size(φ)
    g = zeros(FT, Nx, Ny, 9)
    w = (FT(4/9), FT(1/9), FT(1/9), FT(1/9), FT(1/9),
         FT(1/36), FT(1/36), FT(1/36), FT(1/36))
    cx = (FT(0), FT(1), FT(0), FT(-1), FT(0), FT(1), FT(-1), FT(-1), FT(1))
    cy = (FT(0), FT(0), FT(1), FT(0), FT(-1), FT(1), FT(1), FT(-1), FT(-1))
    for j in 1:Ny, i in 1:Nx
        phi = φ[i,j]
        u = ux[i,j]; v = uy[i,j]
        for q in 1:9
            cu = cx[q] * u + cy[q] * v
            g[i,j,q] = w[q] * phi * (one(FT) + FT(3) * cu)
        end
    end
    return g
end

"""
    init_pressure_equilibrium(φ, ux, uy, ρ_l, ρ_g, FT=Float64)

Initialize pressure-based distributions f_q at equilibrium (CPU).

Equilibrium: f_q = w_q · [ρ_lbm + ρ(φ)·(3c·u + 4.5(c·u)² - 1.5u²)]
with ρ_lbm = 1 (constant reference density).
Returns 3D array f[Nx, Ny, 9].
"""
function init_pressure_equilibrium(φ, ux, uy, ρ_l, ρ_g, FT=Float64)
    Nx, Ny = size(φ)
    f = zeros(FT, Nx, Ny, 9)
    w = (FT(4/9), FT(1/9), FT(1/9), FT(1/9), FT(1/9),
         FT(1/36), FT(1/36), FT(1/36), FT(1/36))
    cx = (FT(0), FT(1), FT(0), FT(-1), FT(0), FT(1), FT(-1), FT(-1), FT(1))
    cy = (FT(0), FT(0), FT(1), FT(0), FT(-1), FT(1), FT(1), FT(-1), FT(-1))
    ρ_lbm = one(FT)
    for j in 1:Ny, i in 1:Nx
        phi = φ[i,j]
        C = (one(FT) + phi) / FT(2)
        ρ_phys = C * FT(ρ_l) + (one(FT) - C) * FT(ρ_g)
        u = ux[i,j]; v = uy[i,j]
        usq = u^2 + v^2
        for q in 1:9
            cu = cx[q] * u + cy[q] * v
            f[i,j,q] = w[q] * (ρ_lbm + ρ_phys * (FT(3) * cu + FT(4.5) * cu^2 - FT(1.5) * usq))
        end
    end
    return f
end
