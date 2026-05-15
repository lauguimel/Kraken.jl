function kraken_e_compute_macroscopic_2d!(block::LeafBlock2D; Fx=0.0, Fy=0.0)
    f = block.f
    ρ = block.ρ
    ux = block.ux
    uy = block.uy
    T = eltype(f)
    fx = T(Fx)
    fy = T(Fy)
    for j in kraken_e_j_range(block), i in kraken_e_i_range(block)
        f1 = f[i,j,1]; f2 = f[i,j,2]; f3 = f[i,j,3]
        f4 = f[i,j,4]; f5 = f[i,j,5]; f6 = f[i,j,6]
        f7 = f[i,j,7]; f8 = f[i,j,8]; f9 = f[i,j,9]
        ρ_local = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
        inv_ρ = one(T) / ρ_local
        ρ[i,j] = ρ_local
        ux[i,j] = ((f2 - f4 + f6 - f7 - f8 + f9) + fx / T(2)) * inv_ρ
        uy[i,j] = ((f3 - f5 + f6 + f7 - f8 - f9) + fy / T(2)) * inv_ρ
    end
    return block
end

function kraken_e_collide_2d!(block::LeafBlock2D, ω; Fx=0.0, Fy=0.0)
    f = block.f
    T = eltype(f)
    omega = T(ω)
    fx = T(Fx)
    fy = T(Fy)
    guo_pref = one(T) - omega / T(2)
    for j in kraken_e_j_range(block), i in kraken_e_i_range(block)
        f1 = f[i,j,1]; f2 = f[i,j,2]; f3 = f[i,j,3]
        f4 = f[i,j,4]; f5 = f[i,j,5]; f6 = f[i,j,6]
        f7 = f[i,j,7]; f8 = f[i,j,8]; f9 = f[i,j,9]

        ρ = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9
        inv_ρ = one(T) / ρ
        ux = ((f2 - f4 + f6 - f7 - f8 + f9) + fx / T(2)) * inv_ρ
        uy = ((f3 - f5 + f6 + f7 - f8 - f9) + fy / T(2)) * inv_ρ
        usq = ux * ux + uy * uy

        Sq = T(4.0/9.0) * ((-ux) * fx + (-uy) * fy) * T(3)
        f[i,j,1] = f1 - omega * (f1 - feq_2d(Val(1), ρ, ux, uy, usq)) + guo_pref * Sq

        Sq = T(1.0/9.0) * ((one(T) - ux) * fx + (-uy) * fy) * T(3) +
             T(1.0/9.0) * ux * fx * T(9)
        f[i,j,2] = f2 - omega * (f2 - feq_2d(Val(2), ρ, ux, uy, usq)) + guo_pref * Sq

        Sq = T(1.0/9.0) * ((-ux) * fx + (one(T) - uy) * fy) * T(3) +
             T(1.0/9.0) * uy * fy * T(9)
        f[i,j,3] = f3 - omega * (f3 - feq_2d(Val(3), ρ, ux, uy, usq)) + guo_pref * Sq

        Sq = T(1.0/9.0) * ((-one(T) - ux) * fx + (-uy) * fy) * T(3) +
             T(1.0/9.0) * ux * fx * T(9)
        f[i,j,4] = f4 - omega * (f4 - feq_2d(Val(4), ρ, ux, uy, usq)) + guo_pref * Sq

        Sq = T(1.0/9.0) * ((-ux) * fx + (-one(T) - uy) * fy) * T(3) +
             T(1.0/9.0) * uy * fy * T(9)
        f[i,j,5] = f5 - omega * (f5 - feq_2d(Val(5), ρ, ux, uy, usq)) + guo_pref * Sq

        Sq = T(1.0/36.0) * ((one(T) - ux) * fx + (one(T) - uy) * fy) * T(3) +
             T(1.0/36.0) * (ux + uy) * (fx + fy) * T(9)
        f[i,j,6] = f6 - omega * (f6 - feq_2d(Val(6), ρ, ux, uy, usq)) + guo_pref * Sq

        Sq = T(1.0/36.0) * ((-one(T) - ux) * fx + (one(T) - uy) * fy) * T(3) +
             T(1.0/36.0) * (-ux + uy) * (-fx + fy) * T(9)
        f[i,j,7] = f7 - omega * (f7 - feq_2d(Val(7), ρ, ux, uy, usq)) + guo_pref * Sq

        Sq = T(1.0/36.0) * ((-one(T) - ux) * fx + (-one(T) - uy) * fy) * T(3) +
             T(1.0/36.0) * (-ux - uy) * (-fx - fy) * T(9)
        f[i,j,8] = f8 - omega * (f8 - feq_2d(Val(8), ρ, ux, uy, usq)) + guo_pref * Sq

        Sq = T(1.0/36.0) * ((one(T) - ux) * fx + (-one(T) - uy) * fy) * T(3) +
             T(1.0/36.0) * (ux - uy) * (fx - fy) * T(9)
        f[i,j,9] = f9 - omega * (f9 - feq_2d(Val(9), ρ, ux, uy, usq)) + guo_pref * Sq
    end
    return block
end

function kraken_e_stream_2d!(block::LeafBlock2D)
    f = block.f
    out = block.f_tmp
    for j in kraken_e_j_range(block), i in kraken_e_i_range(block)
        im = i - 1
        ip = i + 1
        jm = j - 1
        jp = j + 1
        out[i,j,1] = f[i,j,1]
        out[i,j,2] = f[im,j,2]
        out[i,j,3] = f[i,jm,3]
        out[i,j,4] = f[ip,j,4]
        out[i,j,5] = f[i,jp,5]
        out[i,j,6] = f[im,jm,6]
        out[i,j,7] = f[ip,jm,7]
        out[i,j,8] = f[ip,jp,8]
        out[i,j,9] = f[im,jp,9]
    end
    block.f, block.f_tmp = block.f_tmp, block.f
    return block
end

function kraken_e_step!(block::LeafBlock2D, ω; bc::Symbol=:none,
                        exchange::Symbol=:none, Fx=0.0, Fy=0.0, u_top=0.0)
    kraken_e_apply_bcs!(block; kind=bc, u_top=u_top)
    kraken_e_exchange_halo!(block; kind=exchange)
    kraken_e_compute_macroscopic_2d!(block; Fx=Fx, Fy=Fy)
    kraken_e_collide_2d!(block, ω; Fx=Fx, Fy=Fy)
    kraken_e_apply_bcs!(block; kind=bc, u_top=u_top)
    kraken_e_exchange_halo!(block; kind=exchange)
    kraken_e_stream_2d!(block)
    return block
end
