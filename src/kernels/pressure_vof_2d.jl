# Pressure-based MRT + VOF-PLIC for two-phase flows at high density ratio.
#
# Architecture: reuse existing PLIC (sharp, mass-conserving) + new collision
# where ρ_lbm ≈ 1 everywhere. The density ratio enters only through ρ(C).
#
# Key fix for ρ_ratio > 100: density-weighted CSF (Tryggvason 2011)
#   F = σ·κ·∇C · 2ρ(C)/(ρ_l+ρ_g)
# ensures F/ρ is bounded → no supersonic velocities in the gas phase.

using KernelAbstractions

# --- Density-weighted CSF surface tension force ---

@kernel function compute_surface_tension_weighted_2d_kernel!(Fx, Fy, @Const(κ), @Const(C),
                                                              σ, ρ_l, ρ_g, Nx, Ny)
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(C)
        ip = min(i + 1, Nx); im = max(i - 1, 1)
        jp = min(j + 1, Ny); jm = max(j - 1, 1)

        dCdx = (C[ip,j] - C[im,j]) / T(2)
        dCdy = (C[i,jp] - C[i,jm]) / T(2)

        # Density weighting: F ∝ ρ(C) → F/ρ bounded for any ρ_ratio
        c = C[i,j]
        ρ_local = c * ρ_l + (one(T) - c) * ρ_g
        ρ_mean = (ρ_l + ρ_g) / T(2)
        weight = ρ_local / ρ_mean

        Fx[i,j] = σ * κ[i,j] * dCdx * weight
        Fy[i,j] = σ * κ[i,j] * dCdy * weight
    end
end

"""
    compute_surface_tension_weighted_2d!(Fx, Fy, κ, C, σ, Nx, Ny; ρ_l=1.0, ρ_g=0.001)

Density-weighted CSF surface tension force (Tryggvason et al., 2011).

F = σ·κ·∇C · 2ρ(C)/(ρ_l+ρ_g)

This ensures that the force per unit mass (F/ρ) is bounded for any density
ratio, preventing supersonic velocities in the light phase. Essential for
ρ_ratio > 100 with sharp VOF interfaces.
"""
function compute_surface_tension_weighted_2d!(Fx, Fy, κ, C, σ, Nx, Ny;
                                               ρ_l=1.0, ρ_g=0.001)
    backend = KernelAbstractions.get_backend(C)
    T = eltype(C)
    kernel! = compute_surface_tension_weighted_2d_kernel!(backend)
    kernel!(Fx, Fy, κ, C, T(σ), T(ρ_l), T(ρ_g), Int32(Nx), Int32(Ny);
            ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

# --- Pressure-based MRT collision with VOF field C ---

@kernel function collide_pressure_vof_mrt_2d_kernel!(f, @Const(C),
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

            # Physical density from VOF (with floor)
            c = C[i,j]
            ρ_raw = c * ρ_l + (one(T) - c) * ρ_g
            ρ_phys = max(ρ_raw, ρ_l * T(0.01))
            ν_local = c * ν_l + (one(T) - c) * ν_g
            s_nu = one(T) / (T(3) * ν_local + T(0.5))

            fx = Fx[i,j]; fy = Fy[i,j]

            # Moment space
            ρ_lbm = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
            e   = -T(4)*f1 - f2 - f3 - f4 - f5 + T(2)*(f6+f7+f8+f9)
            eps = T(4)*f1 - T(2)*(f2+f3+f4+f5) + f6+f7+f8+f9
            jx  = f2 - f4 + f6 - f7 - f8 + f9
            qx  = -T(2)*f2 + T(2)*f4 + f6 - f7 - f8 + f9
            jy  = f3 - f5 + f6 + f7 - f8 - f9
            qy  = -T(2)*f3 + T(2)*f5 + f6 + f7 - f8 - f9
            pxx = f2 - f3 + f4 - f5
            pxy = f6 - f7 + f8 - f9

            # Physical velocity: u = (j + F/2) / ρ(C)
            inv_ρ = one(T) / ρ_phys
            ux = (jx + fx / T(2)) * inv_ρ
            uy = (jy + fy / T(2)) * inv_ρ
            usq = ux * ux + uy * uy

            # Pressure-based equilibrium: density terms → ρ_lbm, velocity → ρ_phys
            e_eq   = -T(2) * ρ_lbm + T(3) * ρ_phys * usq
            eps_eq = ρ_lbm - T(3) * ρ_phys * usq
            qx_eq  = -ρ_phys * ux
            qy_eq  = -ρ_phys * uy
            pxx_eq = ρ_phys * (ux * ux - uy * uy)
            pxy_eq = ρ_phys * ux * uy

            # Relax
            e_star   = e   - s_e   * (e   - e_eq)
            eps_star = eps - s_eps * (eps - eps_eq)
            jx_star  = jx + fx
            jy_star  = jy + fy
            qx_star  = qx  - s_q  * (qx  - qx_eq)
            qy_star  = qy  - s_q  * (qy  - qy_eq)
            pxx_star = pxx - s_nu * (pxx - pxx_eq)
            pxy_star = pxy - s_nu * (pxy - pxy_eq)

            # Back-transform
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
    collide_pressure_vof_mrt_2d!(f, C, Fx, Fy, is_solid;
        ρ_l=1.0, ρ_g=0.001, ν_l=0.1, ν_g=0.1, s_e=1.4, s_eps=1.4, s_q=1.2)

Pressure-based MRT collision for VOF-tracked two-phase flows.

Modified equilibrium keeps ρ_lbm = Σf ≈ 1 (distributions O(1) for any density
ratio). Physical density ρ(C) enters only through velocity and stress terms.
Use with `compute_surface_tension_weighted_2d!` for F/ρ-bounded forcing.
"""
function collide_pressure_vof_mrt_2d!(f, C, Fx, Fy, is_solid;
                                       ρ_l=1.0, ρ_g=0.001, ν_l=0.1, ν_g=0.1,
                                       s_e=1.4, s_eps=1.4, s_q=1.2)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(C)
    T = eltype(f)
    kernel! = collide_pressure_vof_mrt_2d_kernel!(backend)
    kernel!(f, C, Fx, Fy, is_solid,
            T(ρ_l), T(ρ_g), T(ν_l), T(ν_g), T(s_e), T(s_eps), T(s_q);
            ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

# --- Initialize pressure-based distributions from VOF ---

"""
    init_pressure_vof_equilibrium(C, ux, uy, ρ_l, ρ_g, FT=Float64)

Initialize pressure-based D2Q9 distributions from VOF field (CPU).
Returns f[Nx, Ny, 9] with ρ_lbm = 1 everywhere.
"""
function init_pressure_vof_equilibrium(C, ux, uy, ρ_l, ρ_g, FT=Float64)
    Nx, Ny = size(C)
    f = zeros(FT, Nx, Ny, 9)
    w = (FT(4/9), FT(1/9), FT(1/9), FT(1/9), FT(1/9),
         FT(1/36), FT(1/36), FT(1/36), FT(1/36))
    cx = (FT(0), FT(1), FT(0), FT(-1), FT(0), FT(1), FT(-1), FT(-1), FT(1))
    cy = (FT(0), FT(0), FT(1), FT(0), FT(-1), FT(1), FT(1), FT(-1), FT(-1))
    for j in 1:Ny, i in 1:Nx
        c = C[i,j]
        ρ_phys = c * FT(ρ_l) + (one(FT) - c) * FT(ρ_g)
        u = ux[i,j]; v = uy[i,j]
        usq = u^2 + v^2
        for q in 1:9
            cu = cx[q] * u + cy[q] * v
            f[i,j,q] = w[q] * (one(FT) + ρ_phys * (FT(3) * cu + FT(4.5) * cu^2 - FT(1.5) * usq))
        end
    end
    return f
end
