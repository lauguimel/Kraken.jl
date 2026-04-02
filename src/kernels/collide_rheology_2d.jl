using KernelAbstractions

# --- GNF collision kernel: BGK with shear-rate-dependent viscosity ---
#
# Single parametric kernel — Julia's JIT specializes for each concrete
# rheology type, producing the same code as hand-written kernels.
#
# The strain rate is computed from non-equilibrium distributions (local,
# no neighbor reads). The relaxation time from the previous step is
# stored in tau_field to resolve the implicit γ̇ ↔ τ coupling.

# ============================================================
# Without body forces
# ============================================================

@kernel function collide_rheology_2d_kernel!(f, @Const(is_solid), rheology, tau_field)
    i, j = @index(Global, NTuple)

    @inbounds begin
        if is_solid[i, j]
            tmp2 = f[i,j,2]; f[i,j,2] = f[i,j,4]; f[i,j,4] = tmp2
            tmp3 = f[i,j,3]; f[i,j,3] = f[i,j,5]; f[i,j,5] = tmp3
            tmp6 = f[i,j,6]; f[i,j,6] = f[i,j,8]; f[i,j,8] = tmp6
            tmp7 = f[i,j,7]; f[i,j,7] = f[i,j,9]; f[i,j,9] = tmp7
        else
            T = eltype(f)
            f1=f[i,j,1]; f2=f[i,j,2]; f3=f[i,j,3]; f4=f[i,j,4]
            f5=f[i,j,5]; f6=f[i,j,6]; f7=f[i,j,7]; f8=f[i,j,8]; f9=f[i,j,9]

            # Macroscopic quantities
            ρ = f1+f2+f3+f4+f5+f6+f7+f8+f9
            inv_ρ = one(T) / ρ
            ux = (f2-f4+f6-f7-f8+f9) * inv_ρ
            uy = (f3-f5+f6+f7-f8-f9) * inv_ρ
            usq = ux*ux + uy*uy

            # Equilibrium distributions (unrolled)
            t3=T(3); t45=T(4.5); t15=T(1.5)
            cu = zero(T)
            feq1 = T(4.0/9.0)*ρ*(one(T) - t15*usq)
            cu=ux;    feq2 = T(1.0/9.0)*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            cu=uy;    feq3 = T(1.0/9.0)*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            cu=-ux;   feq4 = T(1.0/9.0)*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            cu=-uy;   feq5 = T(1.0/9.0)*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            cu=ux+uy; feq6 = T(1.0/36.0)*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            cu=-ux+uy;feq7 = T(1.0/36.0)*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            cu=-ux-uy;feq8 = T(1.0/36.0)*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            cu=ux-uy; feq9 = T(1.0/36.0)*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)

            # Strain rate from non-equilibrium distributions (purely local)
            tau_prev = tau_field[i,j]
            gamma_dot = strain_rate_magnitude_2d(
                f1,f2,f3,f4,f5,f6,f7,f8,f9,
                feq1,feq2,feq3,feq4,feq5,feq6,feq7,feq8,feq9,
                ρ, tau_prev)

            # Effective viscosity from rheology model (compile-time dispatch)
            ν_local = effective_viscosity(rheology, gamma_dot)
            ω = one(T) / (T(3) * ν_local + T(0.5))

            # Store tau for next step (also useful as output field)
            tau_field[i,j] = one(T) / ω

            # BGK collision
            f[i,j,1] = f1 - ω*(f1 - feq1)
            f[i,j,2] = f2 - ω*(f2 - feq2)
            f[i,j,3] = f3 - ω*(f3 - feq3)
            f[i,j,4] = f4 - ω*(f4 - feq4)
            f[i,j,5] = f5 - ω*(f5 - feq5)
            f[i,j,6] = f6 - ω*(f6 - feq6)
            f[i,j,7] = f7 - ω*(f7 - feq7)
            f[i,j,8] = f8 - ω*(f8 - feq8)
            f[i,j,9] = f9 - ω*(f9 - feq9)
        end
    end
end

# ============================================================
# With body forces (Guo forcing scheme)
# ============================================================

@kernel function collide_rheology_guo_2d_kernel!(f, @Const(is_solid), rheology,
                                                   tau_field, Fx, Fy)
    i, j = @index(Global, NTuple)

    @inbounds begin
        if is_solid[i, j]
            tmp2 = f[i,j,2]; f[i,j,2] = f[i,j,4]; f[i,j,4] = tmp2
            tmp3 = f[i,j,3]; f[i,j,3] = f[i,j,5]; f[i,j,5] = tmp3
            tmp6 = f[i,j,6]; f[i,j,6] = f[i,j,8]; f[i,j,8] = tmp6
            tmp7 = f[i,j,7]; f[i,j,7] = f[i,j,9]; f[i,j,9] = tmp7
        else
            T = eltype(f)
            f1=f[i,j,1]; f2=f[i,j,2]; f3=f[i,j,3]; f4=f[i,j,4]
            f5=f[i,j,5]; f6=f[i,j,6]; f7=f[i,j,7]; f8=f[i,j,8]; f9=f[i,j,9]

            # Force at this node
            fx = Fx[i,j]
            fy = Fy[i,j]

            # Macroscopic with half-force correction (Guo et al. 2002)
            ρ = f1+f2+f3+f4+f5+f6+f7+f8+f9
            inv_ρ = one(T) / ρ
            ux = ((f2-f4+f6-f7-f8+f9) + fx/T(2)) * inv_ρ
            uy = ((f3-f5+f6+f7-f8-f9) + fy/T(2)) * inv_ρ
            usq = ux*ux + uy*uy

            # Equilibrium
            t3=T(3); t45=T(4.5); t15=T(1.5)
            feq1 = T(4.0/9.0)*ρ*(one(T) - t15*usq)
            cu=ux;    feq2 = T(1.0/9.0)*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            cu=uy;    feq3 = T(1.0/9.0)*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            cu=-ux;   feq4 = T(1.0/9.0)*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            cu=-uy;   feq5 = T(1.0/9.0)*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            cu=ux+uy; feq6 = T(1.0/36.0)*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            cu=-ux+uy;feq7 = T(1.0/36.0)*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            cu=-ux-uy;feq8 = T(1.0/36.0)*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            cu=ux-uy; feq9 = T(1.0/36.0)*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)

            # Strain rate and effective viscosity
            tau_prev = tau_field[i,j]
            gamma_dot = strain_rate_magnitude_2d(
                f1,f2,f3,f4,f5,f6,f7,f8,f9,
                feq1,feq2,feq3,feq4,feq5,feq6,feq7,feq8,feq9,
                ρ, tau_prev)

            ν_local = effective_viscosity(rheology, gamma_dot)
            ω = one(T) / (T(3) * ν_local + T(0.5))
            tau_field[i,j] = one(T) / ω

            # Guo forcing prefactor
            guo_pref = one(T) - ω / T(2)

            # BGK + Guo forcing (same pattern as collide_twophase_2d)
            feq=feq1
            Sq=T(4.0/9.0)*((-ux)*fx+(-uy)*fy)*t3
            f[i,j,1]=f1-ω*(f1-feq)+guo_pref*Sq

            cu=ux; feq=feq2
            Sq=T(1.0/9.0)*((one(T)-ux)*fx+(-uy)*fy)*t3+T(1.0/9.0)*ux*fx*T(9)
            f[i,j,2]=f2-ω*(f2-feq)+guo_pref*Sq

            cu=uy; feq=feq3
            Sq=T(1.0/9.0)*((-ux)*fx+(one(T)-uy)*fy)*t3+T(1.0/9.0)*uy*fy*T(9)
            f[i,j,3]=f3-ω*(f3-feq)+guo_pref*Sq

            cu=-ux; feq=feq4
            Sq=T(1.0/9.0)*((-one(T)-ux)*fx+(-uy)*fy)*t3+T(1.0/9.0)*ux*fx*T(9)
            f[i,j,4]=f4-ω*(f4-feq)+guo_pref*Sq

            cu=-uy; feq=feq5
            Sq=T(1.0/9.0)*((-ux)*fx+(-one(T)-uy)*fy)*t3+T(1.0/9.0)*uy*fy*T(9)
            f[i,j,5]=f5-ω*(f5-feq)+guo_pref*Sq

            cu=ux+uy; feq=feq6
            Sq=T(1.0/36.0)*((one(T)-ux)*fx+(one(T)-uy)*fy)*t3+T(1.0/36.0)*cu*(fx+fy)*T(9)
            f[i,j,6]=f6-ω*(f6-feq)+guo_pref*Sq

            cu=-ux+uy; feq=feq7
            Sq=T(1.0/36.0)*((-one(T)-ux)*fx+(one(T)-uy)*fy)*t3+T(1.0/36.0)*cu*(-fx+fy)*T(9)
            f[i,j,7]=f7-ω*(f7-feq)+guo_pref*Sq

            cu=-ux-uy; feq=feq8
            Sq=T(1.0/36.0)*((-one(T)-ux)*fx+(-one(T)-uy)*fy)*t3+T(1.0/36.0)*cu*(-fx-fy)*T(9)
            f[i,j,8]=f8-ω*(f8-feq)+guo_pref*Sq

            cu=ux-uy; feq=feq9
            Sq=T(1.0/36.0)*((one(T)-ux)*fx+(-one(T)-uy)*fy)*t3+T(1.0/36.0)*cu*(fx-fy)*T(9)
            f[i,j,9]=f9-ω*(f9-feq)+guo_pref*Sq
        end
    end
end

# ============================================================
# With body forces + thermal coupling
# ============================================================

@kernel function collide_rheology_thermal_2d_kernel!(f, @Const(is_solid), rheology,
                                                       tau_field, Fx, Fy, @Const(Temp))
    i, j = @index(Global, NTuple)

    @inbounds begin
        if is_solid[i, j]
            tmp2 = f[i,j,2]; f[i,j,2] = f[i,j,4]; f[i,j,4] = tmp2
            tmp3 = f[i,j,3]; f[i,j,3] = f[i,j,5]; f[i,j,5] = tmp3
            tmp6 = f[i,j,6]; f[i,j,6] = f[i,j,8]; f[i,j,8] = tmp6
            tmp7 = f[i,j,7]; f[i,j,7] = f[i,j,9]; f[i,j,9] = tmp7
        else
            T = eltype(f)
            f1=f[i,j,1]; f2=f[i,j,2]; f3=f[i,j,3]; f4=f[i,j,4]
            f5=f[i,j,5]; f6=f[i,j,6]; f7=f[i,j,7]; f8=f[i,j,8]; f9=f[i,j,9]

            fx = Fx[i,j]
            fy = Fy[i,j]

            ρ = f1+f2+f3+f4+f5+f6+f7+f8+f9
            inv_ρ = one(T) / ρ
            ux = ((f2-f4+f6-f7-f8+f9) + fx/T(2)) * inv_ρ
            uy = ((f3-f5+f6+f7-f8-f9) + fy/T(2)) * inv_ρ
            usq = ux*ux + uy*uy

            t3=T(3); t45=T(4.5); t15=T(1.5)
            feq1 = T(4.0/9.0)*ρ*(one(T) - t15*usq)
            cu=ux;    feq2 = T(1.0/9.0)*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            cu=uy;    feq3 = T(1.0/9.0)*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            cu=-ux;   feq4 = T(1.0/9.0)*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            cu=-uy;   feq5 = T(1.0/9.0)*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            cu=ux+uy; feq6 = T(1.0/36.0)*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            cu=-ux+uy;feq7 = T(1.0/36.0)*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            cu=-ux-uy;feq8 = T(1.0/36.0)*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            cu=ux-uy; feq9 = T(1.0/36.0)*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)

            tau_prev = tau_field[i,j]
            gamma_dot = strain_rate_magnitude_2d(
                f1,f2,f3,f4,f5,f6,f7,f8,f9,
                feq1,feq2,feq3,feq4,feq5,feq6,feq7,feq8,feq9,
                ρ, tau_prev)

            # Thermal shift: viscosity depends on local temperature
            T_local = Temp[i,j]
            ν_local = effective_viscosity_thermal(rheology, gamma_dot, T_local)
            ω = one(T) / (T(3) * ν_local + T(0.5))
            tau_field[i,j] = one(T) / ω

            guo_pref = one(T) - ω / T(2)

            feq=feq1
            Sq=T(4.0/9.0)*((-ux)*fx+(-uy)*fy)*t3
            f[i,j,1]=f1-ω*(f1-feq)+guo_pref*Sq

            cu=ux; feq=feq2
            Sq=T(1.0/9.0)*((one(T)-ux)*fx+(-uy)*fy)*t3+T(1.0/9.0)*ux*fx*T(9)
            f[i,j,2]=f2-ω*(f2-feq)+guo_pref*Sq

            cu=uy; feq=feq3
            Sq=T(1.0/9.0)*((-ux)*fx+(one(T)-uy)*fy)*t3+T(1.0/9.0)*uy*fy*T(9)
            f[i,j,3]=f3-ω*(f3-feq)+guo_pref*Sq

            cu=-ux; feq=feq4
            Sq=T(1.0/9.0)*((-one(T)-ux)*fx+(-uy)*fy)*t3+T(1.0/9.0)*ux*fx*T(9)
            f[i,j,4]=f4-ω*(f4-feq)+guo_pref*Sq

            cu=-uy; feq=feq5
            Sq=T(1.0/9.0)*((-ux)*fx+(-one(T)-uy)*fy)*t3+T(1.0/9.0)*uy*fy*T(9)
            f[i,j,5]=f5-ω*(f5-feq)+guo_pref*Sq

            cu=ux+uy; feq=feq6
            Sq=T(1.0/36.0)*((one(T)-ux)*fx+(one(T)-uy)*fy)*t3+T(1.0/36.0)*cu*(fx+fy)*T(9)
            f[i,j,6]=f6-ω*(f6-feq)+guo_pref*Sq

            cu=-ux+uy; feq=feq7
            Sq=T(1.0/36.0)*((-one(T)-ux)*fx+(one(T)-uy)*fy)*t3+T(1.0/36.0)*cu*(-fx+fy)*T(9)
            f[i,j,7]=f7-ω*(f7-feq)+guo_pref*Sq

            cu=-ux-uy; feq=feq8
            Sq=T(1.0/36.0)*((-one(T)-ux)*fx+(-one(T)-uy)*fy)*t3+T(1.0/36.0)*cu*(-fx-fy)*T(9)
            f[i,j,8]=f8-ω*(f8-feq)+guo_pref*Sq

            cu=ux-uy; feq=feq9
            Sq=T(1.0/36.0)*((one(T)-ux)*fx+(-one(T)-uy)*fy)*t3+T(1.0/36.0)*cu*(fx-fy)*T(9)
            f[i,j,9]=f9-ω*(f9-feq)+guo_pref*Sq
        end
    end
end

# ============================================================
# Public API wrappers
# ============================================================

"""
    collide_rheology_2d!(f, is_solid, rheology, tau_field)

BGK collision with shear-rate-dependent viscosity (no body forces).
The `rheology` argument is a `GeneralizedNewtonian` model struct.
`tau_field` stores the relaxation time from the previous step.
"""
function collide_rheology_2d!(f, is_solid, rheology::GeneralizedNewtonian, tau_field)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    kernel! = collide_rheology_2d_kernel!(backend)
    kernel!(f, is_solid, rheology, tau_field; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

"""
    collide_rheology_guo_2d!(f, is_solid, rheology, tau_field, Fx, Fy)

BGK collision with shear-rate-dependent viscosity and Guo body forcing.
"""
function collide_rheology_guo_2d!(f, is_solid, rheology::GeneralizedNewtonian,
                                   tau_field, Fx, Fy)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    kernel! = collide_rheology_guo_2d_kernel!(backend)
    kernel!(f, is_solid, rheology, tau_field, Fx, Fy; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

"""
    collide_rheology_thermal_2d!(f, is_solid, rheology, tau_field, Fx, Fy, Temp)

BGK collision with thermo-rheological coupling: viscosity depends on
both shear rate γ̇ and local temperature T.
"""
function collide_rheology_thermal_2d!(f, is_solid, rheology::GeneralizedNewtonian,
                                       tau_field, Fx, Fy, Temp)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    kernel! = collide_rheology_thermal_2d_kernel!(backend)
    kernel!(f, is_solid, rheology, tau_field, Fx, Fy, Temp; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end
