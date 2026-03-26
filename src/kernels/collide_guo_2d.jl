using KernelAbstractions

# --- BGK collision with Guo body-force term ---

@kernel function collide_guo_2d_kernel!(f, @Const(is_solid), ω, Fx, Fy)
    i, j = @index(Global, NTuple)

    @inbounds begin
        if is_solid[i, j]
            # Solid obstacle: full bounce-back (swap opposite directions)
            tmp2 = f[i,j,2]; f[i,j,2] = f[i,j,4]; f[i,j,4] = tmp2
            tmp3 = f[i,j,3]; f[i,j,3] = f[i,j,5]; f[i,j,5] = tmp3
            tmp6 = f[i,j,6]; f[i,j,6] = f[i,j,8]; f[i,j,8] = tmp6
            tmp7 = f[i,j,7]; f[i,j,7] = f[i,j,9]; f[i,j,9] = tmp7
        else
            T = eltype(f)
            f1 = f[i,j,1]; f2 = f[i,j,2]; f3 = f[i,j,3]
            f4 = f[i,j,4]; f5 = f[i,j,5]; f6 = f[i,j,6]
            f7 = f[i,j,7]; f8 = f[i,j,8]; f9 = f[i,j,9]

            # Force components cast to element type
            fx = T(Fx)
            fy = T(Fy)

            ρ = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
            inv_ρ = one(T) / ρ
            # Force-corrected velocity: u = (Σf·c + F/2) / ρ  (Guo et al. 2002)
            ux = ((f2 - f4 + f6 - f7 - f8 + f9) + fx / T(2)) * inv_ρ
            uy = ((f3 - f5 + f6 + f7 - f8 - f9) + fy / T(2)) * inv_ρ
            usq = ux * ux + uy * uy

            # Guo forcing prefactor
            guo_pref = one(T) - ω / T(2)

            # q=1: rest (cx=0, cy=0), w=4/9
            cu = zero(T)
            feq = T(4.0/9.0) * ρ * (one(T) - T(1.5) * usq)
            Sq = T(4.0/9.0) * ((-ux) * fx + (-uy) * fy) * T(3)
            f[i,j,1] = f1 - ω * (f1 - feq) + guo_pref * Sq

            # q=2: E (cx=1, cy=0), w=1/9
            cu = ux
            feq = T(1.0/9.0) * ρ * (one(T) + T(3)*cu + T(4.5)*cu*cu - T(1.5)*usq)
            Sq = T(1.0/9.0) * ((one(T) - ux) * fx + (-uy) * fy) * T(3) +
                 T(1.0/9.0) * (ux) * (fx) * T(9)
            f[i,j,2] = f2 - ω * (f2 - feq) + guo_pref * Sq

            # q=3: N (cx=0, cy=1), w=1/9
            cu = uy
            feq = T(1.0/9.0) * ρ * (one(T) + T(3)*cu + T(4.5)*cu*cu - T(1.5)*usq)
            Sq = T(1.0/9.0) * ((-ux) * fx + (one(T) - uy) * fy) * T(3) +
                 T(1.0/9.0) * (uy) * (fy) * T(9)
            f[i,j,3] = f3 - ω * (f3 - feq) + guo_pref * Sq

            # q=4: W (cx=-1, cy=0), w=1/9
            cu = -ux
            feq = T(1.0/9.0) * ρ * (one(T) + T(3)*cu + T(4.5)*cu*cu - T(1.5)*usq)
            Sq = T(1.0/9.0) * ((-one(T) - ux) * fx + (-uy) * fy) * T(3) +
                 T(1.0/9.0) * (-ux) * (-fx) * T(9)
            f[i,j,4] = f4 - ω * (f4 - feq) + guo_pref * Sq

            # q=5: S (cx=0, cy=-1), w=1/9
            cu = -uy
            feq = T(1.0/9.0) * ρ * (one(T) + T(3)*cu + T(4.5)*cu*cu - T(1.5)*usq)
            Sq = T(1.0/9.0) * ((-ux) * fx + (-one(T) - uy) * fy) * T(3) +
                 T(1.0/9.0) * (-uy) * (-fy) * T(9)
            f[i,j,5] = f5 - ω * (f5 - feq) + guo_pref * Sq

            # q=6: NE (cx=1, cy=1), w=1/36
            cu = ux + uy
            feq = T(1.0/36.0) * ρ * (one(T) + T(3)*cu + T(4.5)*cu*cu - T(1.5)*usq)
            Sq = T(1.0/36.0) * ((one(T) - ux) * fx + (one(T) - uy) * fy) * T(3) +
                 T(1.0/36.0) * (ux + uy) * (fx + fy) * T(9)
            f[i,j,6] = f6 - ω * (f6 - feq) + guo_pref * Sq

            # q=7: NW (cx=-1, cy=1), w=1/36
            cu = -ux + uy
            feq = T(1.0/36.0) * ρ * (one(T) + T(3)*cu + T(4.5)*cu*cu - T(1.5)*usq)
            Sq = T(1.0/36.0) * ((-one(T) - ux) * fx + (one(T) - uy) * fy) * T(3) +
                 T(1.0/36.0) * (-ux + uy) * (-fx + fy) * T(9)
            f[i,j,7] = f7 - ω * (f7 - feq) + guo_pref * Sq

            # q=8: SW (cx=-1, cy=-1), w=1/36
            cu = -ux - uy
            feq = T(1.0/36.0) * ρ * (one(T) + T(3)*cu + T(4.5)*cu*cu - T(1.5)*usq)
            Sq = T(1.0/36.0) * ((-one(T) - ux) * fx + (-one(T) - uy) * fy) * T(3) +
                 T(1.0/36.0) * (-ux - uy) * (-fx - fy) * T(9)
            f[i,j,8] = f8 - ω * (f8 - feq) + guo_pref * Sq

            # q=9: SE (cx=1, cy=-1), w=1/36
            cu = ux - uy
            feq = T(1.0/36.0) * ρ * (one(T) + T(3)*cu + T(4.5)*cu*cu - T(1.5)*usq)
            Sq = T(1.0/36.0) * ((one(T) - ux) * fx + (-one(T) - uy) * fy) * T(3) +
                 T(1.0/36.0) * (ux - uy) * (fx - fy) * T(9)
            f[i,j,9] = f9 - ω * (f9 - feq) + guo_pref * Sq
        end
    end
end

# --- Public API ---

function collide_guo_2d!(f, is_solid, ω, Fx, Fy)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    kernel! = collide_guo_2d_kernel!(backend)
    kernel!(f, is_solid, ω, Fx, Fy; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end
