using KernelAbstractions

# --- BGK + Guo body force + viscoelastic Hermite stress source ---
#
# This kernel combines:
#   1. Standard BGK collision with relaxation rate ω
#   2. Guo (2002) body force scheme for a uniform Fx, Fy
#   3. Liu et al. (2025) Hermite source T_i for the polymer stress τ_p
#
# Used for the Poiseuille viscoelastic channel benchmark and any other
# case driven by both a body force and a coupled polymer stress.

@kernel function _collide_visco_guo_2d_kernel!(f, @Const(is_solid), ω,
                                                 Fx, Fy,
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

            ρ = f1+f2+f3+f4+f5+f6+f7+f8+f9
            inv_ρ = one(T) / ρ
            ux = ((f2-f4+f6-f7-f8+f9) + Fx/T(2)) * inv_ρ
            uy = ((f3-f5+f6+f7-f8-f9) + Fy/T(2)) * inv_ρ
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
            # Liu et al. 2025 Eq. 25:
            #   T_i = -w_i · H_iαβ / (2·cs⁴ · τ_s,1) · τ_αβ
            # with τ_s,1 = 1/ω (BGK relaxation time). For D2Q9 cs² = 1/3,
            # so 1/(2·cs⁴) = 9/2 and:
            #   pre = -ω·9/2
            pre = -ω * T(9.0/2.0)
            cs2 = T(1/3)
            wr = T(4/9); wa = T(1/9); we = T(1/36)

            T1 = pre * wr * (-cs2*(txx + tyy))
            T2 = pre * wa * ((one(T)-cs2)*txx - cs2*tyy)
            T3 = pre * wa * (-cs2*txx + (one(T)-cs2)*tyy)
            T4 = T2
            T5 = T3
            T6 = pre * we * ((one(T)-cs2)*txx + (one(T)-cs2)*tyy + T(2)*txy)
            T7 = pre * we * ((one(T)-cs2)*txx + (one(T)-cs2)*tyy - T(2)*txy)
            T8 = pre * we * ((one(T)-cs2)*txx + (one(T)-cs2)*tyy + T(2)*txy)
            T9 = pre * we * ((one(T)-cs2)*txx + (one(T)-cs2)*tyy - T(2)*txy)

            guo_pref = one(T) - ω / T(2)
            t3 = T(3); t9 = T(9)

            S1 = wr * ((-ux)*Fx + (-uy)*Fy) * t3
            S2 = wa * ((one(T)-ux)*Fx + (-uy)*Fy) * t3 + wa * ux*Fx * t9
            S3 = wa * ((-ux)*Fx + (one(T)-uy)*Fy) * t3 + wa * uy*Fy * t9
            S4 = wa * ((-one(T)-ux)*Fx + (-uy)*Fy) * t3 + wa * ux*Fx * t9
            S5 = wa * ((-ux)*Fx + (-one(T)-uy)*Fy) * t3 + wa * uy*Fy * t9
            S6 = we * ((one(T)-ux)*Fx + (one(T)-uy)*Fy) * t3 + we * (ux+uy)*(Fx+Fy) * t9
            S7 = we * ((-one(T)-ux)*Fx + (one(T)-uy)*Fy) * t3 + we * (-ux+uy)*(-Fx+Fy) * t9
            S8 = we * ((-one(T)-ux)*Fx + (-one(T)-uy)*Fy) * t3 + we * (-ux-uy)*(-Fx-Fy) * t9
            S9 = we * ((one(T)-ux)*Fx + (-one(T)-uy)*Fy) * t3 + we * (ux-uy)*(Fx-Fy) * t9

            f[i,j,1] = f1 - ω*(f1 - feq1) + guo_pref*S1 + T1
            f[i,j,2] = f2 - ω*(f2 - feq2) + guo_pref*S2 + T2
            f[i,j,3] = f3 - ω*(f3 - feq3) + guo_pref*S3 + T3
            f[i,j,4] = f4 - ω*(f4 - feq4) + guo_pref*S4 + T4
            f[i,j,5] = f5 - ω*(f5 - feq5) + guo_pref*S5 + T5
            f[i,j,6] = f6 - ω*(f6 - feq6) + guo_pref*S6 + T6
            f[i,j,7] = f7 - ω*(f7 - feq7) + guo_pref*S7 + T7
            f[i,j,8] = f8 - ω*(f8 - feq8) + guo_pref*S8 + T8
            f[i,j,9] = f9 - ω*(f9 - feq9) + guo_pref*S9 + T9
        end
    end
end

"""
    collide_viscoelastic_source_guo_2d!(f, is_solid, ω, Fx, Fy,
                                          tau_p_xx, tau_p_xy, tau_p_yy)

BGK collision with uniform Guo body force (Fx, Fy) AND viscoelastic
Hermite stress source. Both forces are added to the post-collision
distributions in the same step.
"""
function collide_viscoelastic_source_guo_2d!(f, is_solid, ω, Fx, Fy,
                                                tau_p_xx, tau_p_xy, tau_p_yy)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    T = eltype(f)
    kernel! = _collide_visco_guo_2d_kernel!(backend)
    kernel!(f, is_solid, T(ω), T(Fx), T(Fy),
            tau_p_xx, tau_p_xy, tau_p_yy; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end
