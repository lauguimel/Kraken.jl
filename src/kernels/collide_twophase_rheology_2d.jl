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
            tmp = f[i,j,2]; f[i,j,2] = f[i,j,4]; f[i,j,4] = tmp
            tmp = f[i,j,3]; f[i,j,3] = f[i,j,5]; f[i,j,5] = tmp
            tmp = f[i,j,6]; f[i,j,6] = f[i,j,8]; f[i,j,8] = tmp
            tmp = f[i,j,7]; f[i,j,7] = f[i,j,9]; f[i,j,9] = tmp
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
            t3=T(3); t45=T(4.5); t15=T(1.5)
            feq1 = T(4.0/9.0)*ρ_f*(one(T) - t15*usq)
            cu=ux;    feq2 = T(1.0/9.0)*ρ_f*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            cu=uy;    feq3 = T(1.0/9.0)*ρ_f*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            cu=-ux;   feq4 = T(1.0/9.0)*ρ_f*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            cu=-uy;   feq5 = T(1.0/9.0)*ρ_f*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            cu=ux+uy; feq6 = T(1.0/36.0)*ρ_f*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            cu=-ux+uy;feq7 = T(1.0/36.0)*ρ_f*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            cu=-ux-uy;feq8 = T(1.0/36.0)*ρ_f*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            cu=ux-uy; feq9 = T(1.0/36.0)*ρ_f*(one(T)+t3*cu+t45*cu*cu-t15*usq)

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

            feq=feq1
            Sq=T(4.0/9.0)*((-ux)*fx+(-uy)*fy)*t3
            f[i,j,1]=f1-ω_local*(f1-feq)+guo_pref*Sq

            cu=ux; feq=feq2
            Sq=T(1.0/9.0)*((one(T)-ux)*fx+(-uy)*fy)*t3+T(1.0/9.0)*ux*fx*T(9)
            f[i,j,2]=f2-ω_local*(f2-feq)+guo_pref*Sq

            cu=uy; feq=feq3
            Sq=T(1.0/9.0)*((-ux)*fx+(one(T)-uy)*fy)*t3+T(1.0/9.0)*uy*fy*T(9)
            f[i,j,3]=f3-ω_local*(f3-feq)+guo_pref*Sq

            cu=-ux; feq=feq4
            Sq=T(1.0/9.0)*((-one(T)-ux)*fx+(-uy)*fy)*t3+T(1.0/9.0)*ux*fx*T(9)
            f[i,j,4]=f4-ω_local*(f4-feq)+guo_pref*Sq

            cu=-uy; feq=feq5
            Sq=T(1.0/9.0)*((-ux)*fx+(-one(T)-uy)*fy)*t3+T(1.0/9.0)*uy*fy*T(9)
            f[i,j,5]=f5-ω_local*(f5-feq)+guo_pref*Sq

            cu=ux+uy; feq=feq6
            Sq=T(1.0/36.0)*((one(T)-ux)*fx+(one(T)-uy)*fy)*t3+T(1.0/36.0)*cu*(fx+fy)*T(9)
            f[i,j,6]=f6-ω_local*(f6-feq)+guo_pref*Sq

            cu=-ux+uy; feq=feq7
            Sq=T(1.0/36.0)*((-one(T)-ux)*fx+(one(T)-uy)*fy)*t3+T(1.0/36.0)*cu*(-fx+fy)*T(9)
            f[i,j,7]=f7-ω_local*(f7-feq)+guo_pref*Sq

            cu=-ux-uy; feq=feq8
            Sq=T(1.0/36.0)*((-one(T)-ux)*fx+(-one(T)-uy)*fy)*t3+T(1.0/36.0)*cu*(-fx-fy)*T(9)
            f[i,j,8]=f8-ω_local*(f8-feq)+guo_pref*Sq

            cu=ux-uy; feq=feq9
            Sq=T(1.0/36.0)*((one(T)-ux)*fx+(-one(T)-uy)*fy)*t3+T(1.0/36.0)*cu*(fx-fy)*T(9)
            f[i,j,9]=f9-ω_local*(f9-feq)+guo_pref*Sq
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
    KernelAbstractions.synchronize(backend)
end
