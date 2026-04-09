using KernelAbstractions

# --- BGK collision with viscoelastic stress source (Liu et al. 2025) ---
#
# Reference: Liu et al. arxiv 2508.16997, Eq. 25.
#
# The polymeric stress is injected directly into the post-collision
# distribution as a Hermite source:
#
#   T_i = -w_i (H_{i,αβ}) / (2 cs⁴) · τ_αβ
#
# where H_{i,αβ} = c_iα·c_iβ - cs²·δ_{αβ} is the 2nd-order Hermite
# polynomial. This embeds the stress contribution exactly at the LBM
# moment level, avoiding spatial gradients of τ_p.
#
# In 2D D2Q9 with cs² = 1/3 and 1/(2·cs⁴) = 9/2:
#   T_i = -(9/2)·w_i·[(c_ix² - 1/3)·τxx + (c_iy² - 1/3)·τyy + 2·c_ix·c_iy·τxy]

@kernel function collide_viscoelastic_source_2d_kernel!(f, @Const(is_solid), ω,
                                                         @Const(tau_p_xx),
                                                         @Const(tau_p_xy),
                                                         @Const(tau_p_yy))
    i, j = @index(Global, NTuple)

    @inbounds begin
        if is_solid[i, j]
            bounce_back_2d!(f, i, j)
        else
            T = eltype(f)
            f1=f[i,j,1]; f2=f[i,j,2]; f3=f[i,j,3]; f4=f[i,j,4]
            f5=f[i,j,5]; f6=f[i,j,6]; f7=f[i,j,7]; f8=f[i,j,8]; f9=f[i,j,9]

            ρ, ux, uy = moments_2d(f1, f2, f3, f4, f5, f6, f7, f8, f9)
            usq = ux*ux + uy*uy

            feq1 = feq_2d(Val(1), ρ, ux, uy, usq)
            feq2 = feq_2d(Val(2), ρ, ux, uy, usq)
            feq3 = feq_2d(Val(3), ρ, ux, uy, usq)
            feq4 = feq_2d(Val(4), ρ, ux, uy, usq)
            feq5 = feq_2d(Val(5), ρ, ux, uy, usq)
            feq6 = feq_2d(Val(6), ρ, ux, uy, usq)
            feq7 = feq_2d(Val(7), ρ, ux, uy, usq)
            feq8 = feq_2d(Val(8), ρ, ux, uy, usq)
            feq9 = feq_2d(Val(9), ρ, ux, uy, usq)

            txx = tau_p_xx[i,j]; txy = tau_p_xy[i,j]; tyy = tau_p_yy[i,j]
            pre = -T(9.0/2.0)
            cs2 = T(1/3)
            wr = T(4/9); wa = T(1/9); we = T(1/36)

            # T_q = pre * w_q * ((cx²-cs²)·txx + (cy²-cs²)·tyy + 2·cx·cy·txy)
            # q=1 (rest, cx=cy=0): pre * 4/9 * (-cs2·txx - cs2·tyy)
            T1 = pre * wr * (-cs2*(txx + tyy))
            # q=2 (E, cx=1, cy=0): pre * 1/9 * ((1-1/3)·txx + (0-1/3)·tyy)
            T2 = pre * wa * ((one(T)-cs2)*txx - cs2*tyy)
            # q=3 (N, cx=0, cy=1)
            T3 = pre * wa * (-cs2*txx + (one(T)-cs2)*tyy)
            # q=4 (W, cx=-1, cy=0): same as q=2
            T4 = T2
            # q=5 (S, cx=0, cy=-1): same as q=3
            T5 = T3
            # q=6 (NE, cx=1, cy=1): pre * 1/36 * (2/3·txx + 2/3·tyy + 2·txy)
            T6 = pre * we * ((one(T)-cs2)*txx + (one(T)-cs2)*tyy + T(2)*txy)
            # q=7 (NW, cx=-1, cy=1)
            T7 = pre * we * ((one(T)-cs2)*txx + (one(T)-cs2)*tyy - T(2)*txy)
            # q=8 (SW, cx=-1, cy=-1)
            T8 = pre * we * ((one(T)-cs2)*txx + (one(T)-cs2)*tyy + T(2)*txy)
            # q=9 (SE, cx=1, cy=-1)
            T9 = pre * we * ((one(T)-cs2)*txx + (one(T)-cs2)*tyy - T(2)*txy)

            f[i,j,1] = f1 - ω*(f1 - feq1) + T1
            f[i,j,2] = f2 - ω*(f2 - feq2) + T2
            f[i,j,3] = f3 - ω*(f3 - feq3) + T3
            f[i,j,4] = f4 - ω*(f4 - feq4) + T4
            f[i,j,5] = f5 - ω*(f5 - feq5) + T5
            f[i,j,6] = f6 - ω*(f6 - feq6) + T6
            f[i,j,7] = f7 - ω*(f7 - feq7) + T7
            f[i,j,8] = f8 - ω*(f8 - feq8) + T8
            f[i,j,9] = f9 - ω*(f9 - feq9) + T9
        end
    end
end

"""
    collide_viscoelastic_source_2d!(f, is_solid, ω, tau_p_xx, tau_p_xy, tau_p_yy)

BGK collision with viscoelastic stress injected as Hermite source
(Liu et al. 2025, Eq. 25). No explicit ∇·τ_p needed.
"""
function collide_viscoelastic_source_2d!(f, is_solid, ω, tau_p_xx, tau_p_xy, tau_p_yy)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    T = eltype(f)
    kernel! = collide_viscoelastic_source_2d_kernel!(backend)
    kernel!(f, is_solid, T(ω), tau_p_xx, tau_p_xy, tau_p_yy; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end
