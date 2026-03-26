using KernelAbstractions

# --- MRT (Multiple Relaxation Time) collision for D2Q9 ---
#
# Transform to moment space: m = M·f
# Relax: m* = m - S·(m - m^eq)
# Transform back: f* = M⁻¹·m*
#
# Moments: (ρ, e, ε, jx, qx, jy, qy, pxx, pxy)
# S = diag(s_ρ, s_e, s_ε, s_j, s_q, s_j, s_q, s_ν, s_ν)
# s_ν = 1/(3ν + 0.5) controls viscosity (= standard ω)
# Other rates (s_e, s_ε, s_q) tuned for stability

# Transformation matrix M for D2Q9 (Lallemand & Luo, 2000)
# Row order: ρ, e, ε, jx, qx, jy, qy, pxx, pxy
# Column order: q = 1..9 (rest, E, N, W, S, NE, NW, SW, SE)
const _MRT_M = Float64[
#   rest  E   N   W   S   NE  NW  SW  SE
    1     1   1   1   1   1   1   1   1;    # ρ
   -4    -1  -1  -1  -1   2   2   2   2;    # e
    4    -2  -2  -2  -2   1   1   1   1;    # ε
    0     1   0  -1   0   1  -1  -1   1;    # jx
    0    -2   0   2   0   1  -1  -1   1;    # qx
    0     0   1   0  -1   1   1  -1  -1;    # jy
    0     0  -2   0   2   1   1  -1  -1;    # qy
    0     1  -1   1  -1   0   0   0   0;    # pxx
    0     0   0   0   0   1  -1   1  -1     # pxy
]

# Inverse of M (precomputed)
const _MRT_Minv = inv(_MRT_M)

# Equilibrium moments (functions of ρ, ux, uy)
# m^eq = (ρ, e^eq, ε^eq, jx, qx^eq, jy, qy^eq, pxx^eq, pxy^eq)
# where:
#   e^eq = -2ρ + 3ρ(ux²+uy²)
#   ε^eq = ρ - 3ρ(ux²+uy²)
#   qx^eq = -ux·ρ
#   qy^eq = -uy·ρ
#   pxx^eq = ρ(ux²-uy²)
#   pxy^eq = ρ·ux·uy

@kernel function collide_mrt_2d_kernel!(f, @Const(is_solid),
                                         s_e, s_eps, s_q, s_nu)
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

            # Transform to moment space: m = M·f (inline, unrolled)
            ρ   = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
            e   = -T(4)*f1 - f2 - f3 - f4 - f5 + T(2)*(f6+f7+f8+f9)
            eps = T(4)*f1 - T(2)*(f2+f3+f4+f5) + f6+f7+f8+f9
            jx  = f2 - f4 + f6 - f7 - f8 + f9
            qx  = -T(2)*f2 + T(2)*f4 + f6 - f7 - f8 + f9
            jy  = f3 - f5 + f6 + f7 - f8 - f9
            qy  = -T(2)*f3 + T(2)*f5 + f6 + f7 - f8 - f9
            pxx = f2 - f3 + f4 - f5
            pxy = f6 - f7 + f8 - f9

            # Macroscopic velocity
            inv_ρ = one(T) / ρ
            ux = jx * inv_ρ
            uy = jy * inv_ρ
            usq = ux*ux + uy*uy

            # Equilibrium moments
            e_eq   = -T(2)*ρ + T(3)*ρ*usq
            eps_eq = ρ - T(3)*ρ*usq
            qx_eq  = -ρ*ux
            qy_eq  = -ρ*uy
            pxx_eq = ρ*(ux*ux - uy*uy)
            pxy_eq = ρ*ux*uy

            # Relax moments: m* = m - s·(m - m^eq)
            # ρ and j are conserved (s_ρ = s_j = 0 effectively)
            e_star   = e   - s_e   * (e   - e_eq)
            eps_star = eps - s_eps * (eps - eps_eq)
            # jx, jy unchanged (conserved)
            qx_star  = qx  - s_q   * (qx  - qx_eq)
            qy_star  = qy  - s_q   * (qy  - qy_eq)
            pxx_star = pxx - s_nu  * (pxx - pxx_eq)
            pxy_star = pxy - s_nu  * (pxy - pxy_eq)

            # Transform back: f* = M⁻¹·m* (verified coefficients from inv(M))
            r = ρ; es = e_star; ep = eps_star
            jxs = jx; qxs = qx_star; jys = jy; qys = qy_star
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
    collide_mrt_2d!(f, is_solid, ν; s_e=1.4, s_eps=1.4, s_q=1.2)

MRT collision for D2Q9 (Lallemand & Luo, 2000).
The stress relaxation rate s_ν = 1/(3ν + 0.5) is computed from viscosity.
Other rates (s_e, s_eps, s_q) can be tuned for stability (default values from literature).
"""
function collide_mrt_2d!(f, is_solid, ν; s_e=1.4, s_eps=1.4, s_q=1.2)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    T = eltype(f)
    s_nu = T(1.0 / (3.0 * ν + 0.5))
    kernel! = collide_mrt_2d_kernel!(backend)
    kernel!(f, is_solid, T(s_e), T(s_eps), T(s_q), s_nu; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end
