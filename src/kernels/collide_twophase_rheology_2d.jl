using KernelAbstractions

# --- Two-phase collision with per-phase non-Newtonian rheology ---
#
# Each phase can have a different rheology model (e.g., power-law liquid
# with Newtonian gas). The viscosity is interpolated between phases using
# the VOF field C. Follows the same pattern as collide_twophase_2d.

@kernel function collide_twophase_rheology_2d_kernel!(f, @Const(C), @Const(Fx_st), @Const(Fy_st),
                                                        @Const(is_solid), rheology_l, rheology_g,
                                                        ρ_l, ρ_g, tau_field)
    i, j = @index(Global, NTuple)

    @inbounds begin
        if is_solid[i, j]
            bounce_back_2d!(f, i, j)
        else
            T = eltype(f)
            f1=f[i,j,1]; f2=f[i,j,2]; f3=f[i,j,3]; f4=f[i,j,4]
            f5=f[i,j,5]; f6=f[i,j,6]; f7=f[i,j,7]; f8=f[i,j,8]; f9=f[i,j,9]

            c = C[i,j]

            # Interpolated density
            ρ_local = c * ρ_l + (one(T) - c) * ρ_g

            # Surface tension + body forces
            fx = Fx_st[i,j]
            fy = Fy_st[i,j]

            # Macroscopic with Guo half-force correction
            ρ_f = f1+f2+f3+f4+f5+f6+f7+f8+f9
            inv_ρ = one(T) / ρ_f
            ux = ((f2-f4+f6-f7-f8+f9) + fx/T(2)) * inv_ρ
            uy = ((f3-f5+f6+f7-f8-f9) + fy/T(2)) * inv_ρ
            usq = ux*ux + uy*uy

            # Equilibrium
            feq1 = feq_2d(Val(1), ρ_f, ux, uy, usq)
            feq2 = feq_2d(Val(2), ρ_f, ux, uy, usq)
            feq3 = feq_2d(Val(3), ρ_f, ux, uy, usq)
            feq4 = feq_2d(Val(4), ρ_f, ux, uy, usq)
            feq5 = feq_2d(Val(5), ρ_f, ux, uy, usq)
            feq6 = feq_2d(Val(6), ρ_f, ux, uy, usq)
            feq7 = feq_2d(Val(7), ρ_f, ux, uy, usq)
            feq8 = feq_2d(Val(8), ρ_f, ux, uy, usq)
            feq9 = feq_2d(Val(9), ρ_f, ux, uy, usq)

            # Strain rate from non-equilibrium distributions
            tau_prev = tau_field[i,j]
            gamma_dot = strain_rate_magnitude_2d(
                f1,f2,f3,f4,f5,f6,f7,f8,f9,
                feq1,feq2,feq3,feq4,feq5,feq6,feq7,feq8,feq9,
                ρ_f, tau_prev)

            # Per-phase effective viscosity
            ν_l = effective_viscosity(rheology_l, gamma_dot)
            ν_g = effective_viscosity(rheology_g, gamma_dot)

            # Interpolated viscosity (arithmetic average)
            ν_local = c * ν_l + (one(T) - c) * ν_g
            ω_local = one(T) / (T(3) * ν_local + T(0.5))

            tau_field[i,j] = one(T) / ω_local

            # BGK + Guo surface tension forcing
            guo_pref = one(T) - ω_local / T(2)

            Sq=T(4.0/9.0)*((-ux)*fx+(-uy)*fy)*T(3)
            f[i,j,1]=f1-ω_local*(f1-feq1)+guo_pref*Sq

            Sq=T(1.0/9.0)*((one(T)-ux)*fx+(-uy)*fy)*T(3)+T(1.0/9.0)*ux*fx*T(9)
            f[i,j,2]=f2-ω_local*(f2-feq2)+guo_pref*Sq

            Sq=T(1.0/9.0)*((-ux)*fx+(one(T)-uy)*fy)*T(3)+T(1.0/9.0)*uy*fy*T(9)
            f[i,j,3]=f3-ω_local*(f3-feq3)+guo_pref*Sq

            Sq=T(1.0/9.0)*((-one(T)-ux)*fx+(-uy)*fy)*T(3)+T(1.0/9.0)*ux*fx*T(9)
            f[i,j,4]=f4-ω_local*(f4-feq4)+guo_pref*Sq

            Sq=T(1.0/9.0)*((-ux)*fx+(-one(T)-uy)*fy)*T(3)+T(1.0/9.0)*uy*fy*T(9)
            f[i,j,5]=f5-ω_local*(f5-feq5)+guo_pref*Sq

            Sq=T(1.0/36.0)*((one(T)-ux)*fx+(one(T)-uy)*fy)*T(3)+T(1.0/36.0)*(ux+uy)*(fx+fy)*T(9)
            f[i,j,6]=f6-ω_local*(f6-feq6)+guo_pref*Sq

            Sq=T(1.0/36.0)*((-one(T)-ux)*fx+(one(T)-uy)*fy)*T(3)+T(1.0/36.0)*(-ux+uy)*(-fx+fy)*T(9)
            f[i,j,7]=f7-ω_local*(f7-feq7)+guo_pref*Sq

            Sq=T(1.0/36.0)*((-one(T)-ux)*fx+(-one(T)-uy)*fy)*T(3)+T(1.0/36.0)*(-ux-uy)*(-fx-fy)*T(9)
            f[i,j,8]=f8-ω_local*(f8-feq8)+guo_pref*Sq

            Sq=T(1.0/36.0)*((one(T)-ux)*fx+(-one(T)-uy)*fy)*T(3)+T(1.0/36.0)*(ux-uy)*(fx-fy)*T(9)
            f[i,j,9]=f9-ω_local*(f9-feq9)+guo_pref*Sq
        end
    end
end

# ============================================================
# Public API
# ============================================================

"""
    collide_twophase_rheology_2d!(f, C, Fx_st, Fy_st, is_solid, tau_field;
                                  rheology_l, rheology_g, rho_l=1.0, rho_g=0.001)

Two-phase BGK collision with per-phase non-Newtonian rheology.
Each phase has its own `GeneralizedNewtonian` model; viscosity is
interpolated using the VOF field `C`.
"""
function collide_twophase_rheology_2d!(f, C, Fx_st, Fy_st, is_solid, tau_field;
                                        rheology_l::GeneralizedNewtonian=Newtonian(0.1),
                                        rheology_g::GeneralizedNewtonian=Newtonian(0.1),
                                        rho_l=1.0, rho_g=0.001)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    T = eltype(f)
    kernel! = collide_twophase_rheology_2d_kernel!(backend)
    kernel!(f, C, Fx_st, Fy_st, is_solid, rheology_l, rheology_g,
            T(rho_l), T(rho_g), tau_field; ndrange=(Nx, Ny))
end
