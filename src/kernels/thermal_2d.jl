using KernelAbstractions

# --- Thermal D2Q9: Double Distribution Function for temperature ---
#
# Temperature field T is tracked by a separate population g[i,j,q] on D2Q9.
# Equilibrium: g_eq_q = w_q * T * (1 + c_q·u / cs²)
# Collision:   g_q = g_q - ω_T * (g_q - g_eq_q)
# where ω_T = 1 / (3α + 0.5), α = thermal diffusivity
# Macroscopic: T = Σ g_q

# --- Stream thermal populations (reuses same stream kernels as flow) ---
# For walls: adiabatic (bounce-back) or fixed T (anti-bounce-back)

# --- Collide thermal (BGK on temperature populations) ---

@kernel function collide_thermal_2d_kernel!(g, @Const(ux), @Const(uy), ω_T)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(g)
        g1=g[i,j,1]; g2=g[i,j,2]; g3=g[i,j,3]; g4=g[i,j,4]
        g5=g[i,j,5]; g6=g[i,j,6]; g7=g[i,j,7]; g8=g[i,j,8]; g9=g[i,j,9]

        # Macroscopic temperature
        Temp = g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8 + g9

        # Local velocity (from flow solver)
        u_x = ux[i,j]
        u_y = uy[i,j]

        # Thermal equilibrium: g_eq = w_q * T * (1 + 3 * c_q · u)
        # (simplified: no u² terms for passive scalar transport)
        t3 = T(3)

        geq = T(4.0/9.0) * Temp
        g[i,j,1] = g1 - ω_T * (g1 - geq)

        geq = T(1.0/9.0) * Temp * (one(T) + t3 * u_x)
        g[i,j,2] = g2 - ω_T * (g2 - geq)

        geq = T(1.0/9.0) * Temp * (one(T) + t3 * u_y)
        g[i,j,3] = g3 - ω_T * (g3 - geq)

        geq = T(1.0/9.0) * Temp * (one(T) - t3 * u_x)
        g[i,j,4] = g4 - ω_T * (g4 - geq)

        geq = T(1.0/9.0) * Temp * (one(T) - t3 * u_y)
        g[i,j,5] = g5 - ω_T * (g5 - geq)

        geq = T(1.0/36.0) * Temp * (one(T) + t3 * (u_x + u_y))
        g[i,j,6] = g6 - ω_T * (g6 - geq)

        geq = T(1.0/36.0) * Temp * (one(T) + t3 * (-u_x + u_y))
        g[i,j,7] = g7 - ω_T * (g7 - geq)

        geq = T(1.0/36.0) * Temp * (one(T) + t3 * (-u_x - u_y))
        g[i,j,8] = g8 - ω_T * (g8 - geq)

        geq = T(1.0/36.0) * Temp * (one(T) + t3 * (u_x - u_y))
        g[i,j,9] = g9 - ω_T * (g9 - geq)
    end
end

function collide_thermal_2d!(g, ux, uy, ω_T)
    backend = KernelAbstractions.get_backend(g)
    Nx, Ny = size(g, 1), size(g, 2)
    kernel! = collide_thermal_2d_kernel!(backend)
    kernel!(g, ux, uy, ω_T; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

# --- Compute temperature from thermal populations ---

@kernel function compute_temperature_2d_kernel!(Temp, @Const(g))
    i, j = @index(Global, NTuple)
    @inbounds begin
        Temp[i,j] = g[i,j,1] + g[i,j,2] + g[i,j,3] + g[i,j,4] +
                    g[i,j,5] + g[i,j,6] + g[i,j,7] + g[i,j,8] + g[i,j,9]
    end
end

function compute_temperature_2d!(Temp, g)
    backend = KernelAbstractions.get_backend(g)
    Nx, Ny = size(Temp)
    kernel! = compute_temperature_2d_kernel!(backend)
    kernel!(Temp, g; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

# --- Fixed temperature BC (anti-bounce-back for Dirichlet) ---

@kernel function apply_fixed_temp_south_2d_kernel!(g, T_wall)
    i = @index(Global)
    j = 1
    @inbounds begin
        TT = eltype(g)
        # Anti-bounce-back: g_q(wall) = -g_opp(wall) + 2*w_q*T_wall
        # For south wall (j=1), unknown populations pointing north: 3, 6, 7
        g[i,j,3] = -g[i,j,5] + TT(2) * TT(1.0/9.0)  * T_wall
        g[i,j,6] = -g[i,j,8] + TT(2) * TT(1.0/36.0) * T_wall
        g[i,j,7] = -g[i,j,9] + TT(2) * TT(1.0/36.0) * T_wall
    end
end

@kernel function apply_fixed_temp_north_2d_kernel!(g, T_wall, Ny)
    i = @index(Global)
    j = Ny
    @inbounds begin
        TT = eltype(g)
        # Unknown populations pointing south: 5, 8, 9
        g[i,j,5] = -g[i,j,3] + TT(2) * TT(1.0/9.0)  * T_wall
        g[i,j,8] = -g[i,j,6] + TT(2) * TT(1.0/36.0) * T_wall
        g[i,j,9] = -g[i,j,7] + TT(2) * TT(1.0/36.0) * T_wall
    end
end

function apply_fixed_temp_south_2d!(g, T_wall, Nx)
    backend = KernelAbstractions.get_backend(g)
    kernel! = apply_fixed_temp_south_2d_kernel!(backend)
    kernel!(g, eltype(g)(T_wall); ndrange=(Nx,))
    KernelAbstractions.synchronize(backend)
end

function apply_fixed_temp_north_2d!(g, T_wall, Nx, Ny)
    backend = KernelAbstractions.get_backend(g)
    kernel! = apply_fixed_temp_north_2d_kernel!(backend)
    kernel!(g, eltype(g)(T_wall), Ny; ndrange=(Nx,))
    KernelAbstractions.synchronize(backend)
end

# --- BGK collision with per-node Boussinesq body force (Guo scheme) ---

@kernel function collide_boussinesq_2d_kernel!(f, @Const(Temp), @Const(is_solid),
                                                ω, β_g, T_ref)
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

            # Per-node buoyancy force: Fy = β·g·(T - T_ref)
            fy = β_g * (Temp[i,j] - T_ref)
            fx = zero(T)

            ρ = f1+f2+f3+f4+f5+f6+f7+f8+f9
            inv_ρ = one(T) / ρ
            ux = ((f2-f4+f6-f7-f8+f9) + fx/T(2)) * inv_ρ
            uy = ((f3-f5+f6+f7-f8-f9) + fy/T(2)) * inv_ρ
            usq = ux*ux + uy*uy

            guo_pref = one(T) - ω / T(2)
            t3 = T(3); t45 = T(4.5); t15 = T(1.5)

            # Rest
            feq = T(4.0/9.0)*ρ*(one(T)-t15*usq)
            Sq = T(4.0/9.0) * ((-ux)*fx + (-uy)*fy)*t3
            f[i,j,1] = f1 - ω*(f1-feq) + guo_pref*Sq

            # E (+1,0)
            cu=ux; feq=T(1.0/9.0)*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            Sq = T(1.0/9.0)*((one(T)-ux)*fx+(-uy)*fy)*t3 + T(1.0/9.0)*ux*fx*T(9)
            f[i,j,2] = f2 - ω*(f2-feq) + guo_pref*Sq

            # N (0,+1)
            cu=uy; feq=T(1.0/9.0)*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            Sq = T(1.0/9.0)*((-ux)*fx+(one(T)-uy)*fy)*t3 + T(1.0/9.0)*uy*fy*T(9)
            f[i,j,3] = f3 - ω*(f3-feq) + guo_pref*Sq

            # W (-1,0)
            cu=-ux; feq=T(1.0/9.0)*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            Sq = T(1.0/9.0)*((-one(T)-ux)*fx+(-uy)*fy)*t3 + T(1.0/9.0)*ux*fx*T(9)
            f[i,j,4] = f4 - ω*(f4-feq) + guo_pref*Sq

            # S (0,-1)
            cu=-uy; feq=T(1.0/9.0)*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            Sq = T(1.0/9.0)*((-ux)*fx+(-one(T)-uy)*fy)*t3 + T(1.0/9.0)*uy*fy*T(9)
            f[i,j,5] = f5 - ω*(f5-feq) + guo_pref*Sq

            # NE (+1,+1)
            cu=ux+uy; feq=T(1.0/36.0)*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            Sq = T(1.0/36.0)*((one(T)-ux)*fx+(one(T)-uy)*fy)*t3 + T(1.0/36.0)*(ux+uy)*(fx+fy)*T(9)
            f[i,j,6] = f6 - ω*(f6-feq) + guo_pref*Sq

            # NW (-1,+1)
            cu=-ux+uy; feq=T(1.0/36.0)*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            Sq = T(1.0/36.0)*((-one(T)-ux)*fx+(one(T)-uy)*fy)*t3 + T(1.0/36.0)*(-ux+uy)*(-fx+fy)*T(9)
            f[i,j,7] = f7 - ω*(f7-feq) + guo_pref*Sq

            # SW (-1,-1)
            cu=-ux-uy; feq=T(1.0/36.0)*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            Sq = T(1.0/36.0)*((-one(T)-ux)*fx+(-one(T)-uy)*fy)*t3 + T(1.0/36.0)*(-ux-uy)*(-fx-fy)*T(9)
            f[i,j,8] = f8 - ω*(f8-feq) + guo_pref*Sq

            # SE (+1,-1)
            cu=ux-uy; feq=T(1.0/36.0)*ρ*(one(T)+t3*cu+t45*cu*cu-t15*usq)
            Sq = T(1.0/36.0)*((one(T)-ux)*fx+(-one(T)-uy)*fy)*t3 + T(1.0/36.0)*(ux-uy)*(fx-fy)*T(9)
            f[i,j,9] = f9 - ω*(f9-feq) + guo_pref*Sq
        end
    end
end

function collide_boussinesq_2d!(f, Temp, is_solid, ω, β_g, T_ref)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    T = eltype(f)
    kernel! = collide_boussinesq_2d_kernel!(backend)
    kernel!(f, Temp, is_solid, ω, T(β_g), T(T_ref); ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end
