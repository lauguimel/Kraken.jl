using KernelAbstractions

# --- 3D Refinement exchange kernels (D3Q19, trilinear + Filippova-Hanel) ---

# Inline D3Q19 equilibrium for GPU kernels
@inline function _feq_3d(q::Int, ρ, ux, uy, uz)
    TT = typeof(ρ)
    w  = TT(_D3Q19_W[q])
    cx = TT(_D3Q19_CX[q])
    cy = TT(_D3Q19_CY[q])
    cz = TT(_D3Q19_CZ[q])
    cu = cx * ux + cy * uy + cz * uz
    usq = ux * ux + uy * uy + uz * uz
    return w * ρ * (one(TT) + TT(3) * cu + TT(4.5) * cu * cu - TT(1.5) * usq)
end

# =====================================================================
# Prolongation: coarse → fine ghost fill with Filippova-Hanel (D3Q19)
# =====================================================================

@kernel function prolongate_f_rescaled_3d_kernel!(
        f_fine, @Const(f_coarse), @Const(rho_c), @Const(ux_c), @Const(uy_c), @Const(uz_c),
        ratio, Nx_inner, Ny_inner, Nz_inner, n_ghost,
        i_c_start, j_c_start, k_c_start,
        Nx_c, Ny_c, Nz_c, omega_c, omega_f)

    i_f, j_f, k_f = @index(Global, NTuple)

    @inbounds begin
        is_ghost = (i_f <= n_ghost) || (i_f > n_ghost + Nx_inner) ||
                   (j_f <= n_ghost) || (j_f > n_ghost + Ny_inner) ||
                   (k_f <= n_ghost) || (k_f > n_ghost + Nz_inner)
        if is_ghost
            T = eltype(f_fine)
            alpha = (T(omega_f) - T(0.5)) * (one(T) / (T(omega_c) - T(0.5)))

            # Fine cell center → coarse continuous coords
            xc = T(i_f - n_ghost - 1) / T(ratio) + T(i_c_start) + T(0.5) / T(ratio)
            yc = T(j_f - n_ghost - 1) / T(ratio) + T(j_c_start) + T(0.5) / T(ratio)
            zc = T(k_f - n_ghost - 1) / T(ratio) + T(k_c_start) + T(0.5) / T(ratio)

            # Trilinear stencil
            i0r = trunc(Int, xc); j0r = trunc(Int, yc); k0r = trunc(Int, zc)
            tx = xc - T(i0r); ty = yc - T(j0r); tz = zc - T(k0r)
            i0 = clamp(i0r, 1, Nx_c); i1 = clamp(i0r+1, 1, Nx_c)
            j0 = clamp(j0r, 1, Ny_c); j1 = clamp(j0r+1, 1, Ny_c)
            k0 = clamp(k0r, 1, Nz_c); k1 = clamp(k0r+1, 1, Nz_c)
            tx = clamp(tx, zero(T), one(T))
            ty = clamp(ty, zero(T), one(T))
            tz = clamp(tz, zero(T), one(T))

            w000=(one(T)-tx)*(one(T)-ty)*(one(T)-tz)
            w100=tx*(one(T)-ty)*(one(T)-tz)
            w010=(one(T)-tx)*ty*(one(T)-tz)
            w110=tx*ty*(one(T)-tz)
            w001=(one(T)-tx)*(one(T)-ty)*tz
            w101=tx*(one(T)-ty)*tz
            w011=(one(T)-tx)*ty*tz
            w111=tx*ty*tz

            # Interpolate macroscopic
            rho_f = w000*rho_c[i0,j0,k0]+w100*rho_c[i1,j0,k0]+w010*rho_c[i0,j1,k0]+w110*rho_c[i1,j1,k0]+
                    w001*rho_c[i0,j0,k1]+w101*rho_c[i1,j0,k1]+w011*rho_c[i0,j1,k1]+w111*rho_c[i1,j1,k1]
            ux_f = w000*ux_c[i0,j0,k0]+w100*ux_c[i1,j0,k0]+w010*ux_c[i0,j1,k0]+w110*ux_c[i1,j1,k0]+
                   w001*ux_c[i0,j0,k1]+w101*ux_c[i1,j0,k1]+w011*ux_c[i0,j1,k1]+w111*ux_c[i1,j1,k1]
            uy_f = w000*uy_c[i0,j0,k0]+w100*uy_c[i1,j0,k0]+w010*uy_c[i0,j1,k0]+w110*uy_c[i1,j1,k0]+
                   w001*uy_c[i0,j0,k1]+w101*uy_c[i1,j0,k1]+w011*uy_c[i0,j1,k1]+w111*uy_c[i1,j1,k1]
            uz_f = w000*uz_c[i0,j0,k0]+w100*uz_c[i1,j0,k0]+w010*uz_c[i0,j1,k0]+w110*uz_c[i1,j1,k0]+
                   w001*uz_c[i0,j0,k1]+w101*uz_c[i1,j0,k1]+w011*uz_c[i0,j1,k1]+w111*uz_c[i1,j1,k1]

            for q in 1:19
                # Equilibrium at interpolated fine state
                feq_fine = _feq_3d(q, rho_f, ux_f, uy_f, uz_f)

                # Interpolated f_neq from 8 stencil nodes
                fneq = w000*(f_coarse[i0,j0,k0,q]-_feq_3d(q,rho_c[i0,j0,k0],ux_c[i0,j0,k0],uy_c[i0,j0,k0],uz_c[i0,j0,k0]))+
                       w100*(f_coarse[i1,j0,k0,q]-_feq_3d(q,rho_c[i1,j0,k0],ux_c[i1,j0,k0],uy_c[i1,j0,k0],uz_c[i1,j0,k0]))+
                       w010*(f_coarse[i0,j1,k0,q]-_feq_3d(q,rho_c[i0,j1,k0],ux_c[i0,j1,k0],uy_c[i0,j1,k0],uz_c[i0,j1,k0]))+
                       w110*(f_coarse[i1,j1,k0,q]-_feq_3d(q,rho_c[i1,j1,k0],ux_c[i1,j1,k0],uy_c[i1,j1,k0],uz_c[i1,j1,k0]))+
                       w001*(f_coarse[i0,j0,k1,q]-_feq_3d(q,rho_c[i0,j0,k1],ux_c[i0,j0,k1],uy_c[i0,j0,k1],uz_c[i0,j0,k1]))+
                       w101*(f_coarse[i1,j0,k1,q]-_feq_3d(q,rho_c[i1,j0,k1],ux_c[i1,j0,k1],uy_c[i1,j0,k1],uz_c[i1,j0,k1]))+
                       w011*(f_coarse[i0,j1,k1,q]-_feq_3d(q,rho_c[i0,j1,k1],ux_c[i0,j1,k1],uy_c[i0,j1,k1],uz_c[i0,j1,k1]))+
                       w111*(f_coarse[i1,j1,k1,q]-_feq_3d(q,rho_c[i1,j1,k1],ux_c[i1,j1,k1],uy_c[i1,j1,k1],uz_c[i1,j1,k1]))

                f_fine[i_f, j_f, k_f, q] = feq_fine + alpha * fneq
            end
        end
    end
end

function prolongate_f_rescaled_3d!(f_fine, f_coarse, rho_c, ux_c, uy_c, uz_c,
                                    ratio, Nx_inner, Ny_inner, Nz_inner, n_ghost,
                                    i_c_start, j_c_start, k_c_start,
                                    Nx_c, Ny_c, Nz_c, omega_c, omega_f)
    backend = KernelAbstractions.get_backend(f_fine)
    Nx_f = Nx_inner + 2 * n_ghost
    Ny_f = Ny_inner + 2 * n_ghost
    Nz_f = Nz_inner + 2 * n_ghost
    kernel! = prolongate_f_rescaled_3d_kernel!(backend)
    kernel!(f_fine, f_coarse, rho_c, ux_c, uy_c, uz_c,
            ratio, Nx_inner, Ny_inner, Nz_inner, n_ghost,
            i_c_start, j_c_start, k_c_start,
            Nx_c, Ny_c, Nz_c, omega_c, omega_f;
            ndrange=(Nx_f, Ny_f, Nz_f))
    KernelAbstractions.synchronize(backend)
end

# =====================================================================
# Full prolongation (init time — operates on ALL cells, not just ghosts)
# =====================================================================

@kernel function prolongate_f_rescaled_full_3d_kernel!(
        f_fine, @Const(f_coarse), @Const(rho_c), @Const(ux_c), @Const(uy_c), @Const(uz_c),
        ratio, Nx_inner, Ny_inner, Nz_inner, n_ghost,
        i_c_start, j_c_start, k_c_start,
        Nx_c, Ny_c, Nz_c, omega_c, omega_f)

    i_f, j_f, k_f = @index(Global, NTuple)

    @inbounds begin
        T = eltype(f_fine)
        alpha = (T(omega_f) - T(0.5)) * (one(T) / (T(omega_c) - T(0.5)))

        xc = T(i_f - n_ghost - 1) / T(ratio) + T(i_c_start) + T(0.5) / T(ratio)
        yc = T(j_f - n_ghost - 1) / T(ratio) + T(j_c_start) + T(0.5) / T(ratio)
        zc = T(k_f - n_ghost - 1) / T(ratio) + T(k_c_start) + T(0.5) / T(ratio)

        i0r = trunc(Int, xc); j0r = trunc(Int, yc); k0r = trunc(Int, zc)
        tx = clamp(xc - T(i0r), zero(T), one(T))
        ty = clamp(yc - T(j0r), zero(T), one(T))
        tz = clamp(zc - T(k0r), zero(T), one(T))
        i0 = clamp(i0r, 1, Nx_c); i1 = clamp(i0r+1, 1, Nx_c)
        j0 = clamp(j0r, 1, Ny_c); j1 = clamp(j0r+1, 1, Ny_c)
        k0 = clamp(k0r, 1, Nz_c); k1 = clamp(k0r+1, 1, Nz_c)

        w000=(one(T)-tx)*(one(T)-ty)*(one(T)-tz); w100=tx*(one(T)-ty)*(one(T)-tz)
        w010=(one(T)-tx)*ty*(one(T)-tz);           w110=tx*ty*(one(T)-tz)
        w001=(one(T)-tx)*(one(T)-ty)*tz;           w101=tx*(one(T)-ty)*tz
        w011=(one(T)-tx)*ty*tz;                     w111=tx*ty*tz

        rho_f = w000*rho_c[i0,j0,k0]+w100*rho_c[i1,j0,k0]+w010*rho_c[i0,j1,k0]+w110*rho_c[i1,j1,k0]+
                w001*rho_c[i0,j0,k1]+w101*rho_c[i1,j0,k1]+w011*rho_c[i0,j1,k1]+w111*rho_c[i1,j1,k1]
        ux_f = w000*ux_c[i0,j0,k0]+w100*ux_c[i1,j0,k0]+w010*ux_c[i0,j1,k0]+w110*ux_c[i1,j1,k0]+
               w001*ux_c[i0,j0,k1]+w101*ux_c[i1,j0,k1]+w011*ux_c[i0,j1,k1]+w111*ux_c[i1,j1,k1]
        uy_f = w000*uy_c[i0,j0,k0]+w100*uy_c[i1,j0,k0]+w010*uy_c[i0,j1,k0]+w110*uy_c[i1,j1,k0]+
               w001*uy_c[i0,j0,k1]+w101*uy_c[i1,j0,k1]+w011*uy_c[i0,j1,k1]+w111*uy_c[i1,j1,k1]
        uz_f = w000*uz_c[i0,j0,k0]+w100*uz_c[i1,j0,k0]+w010*uz_c[i0,j1,k0]+w110*uz_c[i1,j1,k0]+
               w001*uz_c[i0,j0,k1]+w101*uz_c[i1,j0,k1]+w011*uz_c[i0,j1,k1]+w111*uz_c[i1,j1,k1]

        for q in 1:19
            feq_fine = _feq_3d(q, rho_f, ux_f, uy_f, uz_f)
            fneq = w000*(f_coarse[i0,j0,k0,q]-_feq_3d(q,rho_c[i0,j0,k0],ux_c[i0,j0,k0],uy_c[i0,j0,k0],uz_c[i0,j0,k0]))+
                   w100*(f_coarse[i1,j0,k0,q]-_feq_3d(q,rho_c[i1,j0,k0],ux_c[i1,j0,k0],uy_c[i1,j0,k0],uz_c[i1,j0,k0]))+
                   w010*(f_coarse[i0,j1,k0,q]-_feq_3d(q,rho_c[i0,j1,k0],ux_c[i0,j1,k0],uy_c[i0,j1,k0],uz_c[i0,j1,k0]))+
                   w110*(f_coarse[i1,j1,k0,q]-_feq_3d(q,rho_c[i1,j1,k0],ux_c[i1,j1,k0],uy_c[i1,j1,k0],uz_c[i1,j1,k0]))+
                   w001*(f_coarse[i0,j0,k1,q]-_feq_3d(q,rho_c[i0,j0,k1],ux_c[i0,j0,k1],uy_c[i0,j0,k1],uz_c[i0,j0,k1]))+
                   w101*(f_coarse[i1,j0,k1,q]-_feq_3d(q,rho_c[i1,j0,k1],ux_c[i1,j0,k1],uy_c[i1,j0,k1],uz_c[i1,j0,k1]))+
                   w011*(f_coarse[i0,j1,k1,q]-_feq_3d(q,rho_c[i0,j1,k1],ux_c[i0,j1,k1],uy_c[i0,j1,k1],uz_c[i0,j1,k1]))+
                   w111*(f_coarse[i1,j1,k1,q]-_feq_3d(q,rho_c[i1,j1,k1],ux_c[i1,j1,k1],uy_c[i1,j1,k1],uz_c[i1,j1,k1]))
            f_fine[i_f, j_f, k_f, q] = feq_fine + alpha * fneq
        end
    end
end

function prolongate_f_rescaled_full_3d!(f_fine, f_coarse, rho_c, ux_c, uy_c, uz_c,
                                         ratio, Nx_inner, Ny_inner, Nz_inner, n_ghost,
                                         i_c_start, j_c_start, k_c_start,
                                         Nx_c, Ny_c, Nz_c, omega_c, omega_f)
    backend = KernelAbstractions.get_backend(f_fine)
    Nx_f = Nx_inner + 2 * n_ghost
    Ny_f = Ny_inner + 2 * n_ghost
    Nz_f = Nz_inner + 2 * n_ghost
    kernel! = prolongate_f_rescaled_full_3d_kernel!(backend)
    kernel!(f_fine, f_coarse, rho_c, ux_c, uy_c, uz_c,
            ratio, Nx_inner, Ny_inner, Nz_inner, n_ghost,
            i_c_start, j_c_start, k_c_start,
            Nx_c, Ny_c, Nz_c, omega_c, omega_f;
            ndrange=(Nx_f, Ny_f, Nz_f))
    KernelAbstractions.synchronize(backend)
end

# =====================================================================
# Temporal ghost fill (sub-cycling interpolation between time n and n+1)
# =====================================================================

@kernel function prolongate_f_rescaled_temporal_3d_kernel!(
        f_fine,
        @Const(f_curr), @Const(rho_curr), @Const(ux_curr), @Const(uy_curr), @Const(uz_curr),
        @Const(f_prev), @Const(rho_prev), @Const(ux_prev), @Const(uy_prev), @Const(uz_prev),
        ratio, Nx_inner, Ny_inner, Nz_inner, n_ghost,
        i_c_start, j_c_start, k_c_start,
        Nx_c, Ny_c, Nz_c, omega_c, omega_f, t_frac)

    i_f, j_f, k_f = @index(Global, NTuple)

    @inbounds begin
        is_ghost = (i_f <= n_ghost) || (i_f > n_ghost + Nx_inner) ||
                   (j_f <= n_ghost) || (j_f > n_ghost + Ny_inner) ||
                   (k_f <= n_ghost) || (k_f > n_ghost + Nz_inner)
        if is_ghost
            T = eltype(f_fine)
            alpha = (T(omega_f) - T(0.5)) * (one(T) / (T(omega_c) - T(0.5)))
            tf = T(t_frac)
            omtf = one(T) - tf

            xc = T(i_f - n_ghost - 1) / T(ratio) + T(i_c_start) + T(0.5) / T(ratio)
            yc = T(j_f - n_ghost - 1) / T(ratio) + T(j_c_start) + T(0.5) / T(ratio)
            zc = T(k_f - n_ghost - 1) / T(ratio) + T(k_c_start) + T(0.5) / T(ratio)

            i0r = trunc(Int, xc); j0r = trunc(Int, yc); k0r = trunc(Int, zc)
            tx = clamp(xc - T(i0r), zero(T), one(T))
            ty = clamp(yc - T(j0r), zero(T), one(T))
            tz = clamp(zc - T(k0r), zero(T), one(T))

            # Coarse grid indices (current state at n+1)
            ic0 = clamp(i0r, 1, Nx_c); ic1 = clamp(i0r+1, 1, Nx_c)
            jc0 = clamp(j0r, 1, Ny_c); jc1 = clamp(j0r+1, 1, Ny_c)
            kc0 = clamp(k0r, 1, Nz_c); kc1 = clamp(k0r+1, 1, Nz_c)

            # Prev buffer indices (local, offset from i_lo = max(i_c_start-1, 1))
            i_lo = max(i_c_start - 1, 1)
            j_lo = max(j_c_start - 1, 1)
            k_lo = max(k_c_start - 1, 1)
            Ni_p = size(rho_prev, 1); Nj_p = size(rho_prev, 2); Nk_p = size(rho_prev, 3)
            ip0 = clamp(i0r - i_lo + 1, 1, Ni_p); ip1 = clamp(i0r+1 - i_lo + 1, 1, Ni_p)
            jp0 = clamp(j0r - j_lo + 1, 1, Nj_p); jp1 = clamp(j0r+1 - j_lo + 1, 1, Nj_p)
            kp0 = clamp(k0r - k_lo + 1, 1, Nk_p); kp1 = clamp(k0r+1 - k_lo + 1, 1, Nk_p)

            w000=(one(T)-tx)*(one(T)-ty)*(one(T)-tz); w100=tx*(one(T)-ty)*(one(T)-tz)
            w010=(one(T)-tx)*ty*(one(T)-tz);           w110=tx*ty*(one(T)-tz)
            w001=(one(T)-tx)*(one(T)-ty)*tz;           w101=tx*(one(T)-ty)*tz
            w011=(one(T)-tx)*ty*tz;                     w111=tx*ty*tz

            # Temporally interpolated macroscopic at each stencil node, then spatial interp
            # For efficiency, blend the 8 stencil nodes after temporal interp
            rho_f = zero(T); uxf = zero(T); uyf = zero(T); uzf = zero(T)
            for (ws, ic, jc, kc, ipp, jpp, kpp) in (
                (w000, ic0, jc0, kc0, ip0, jp0, kp0), (w100, ic1, jc0, kc0, ip1, jp0, kp0),
                (w010, ic0, jc1, kc0, ip0, jp1, kp0), (w110, ic1, jc1, kc0, ip1, jp1, kp0),
                (w001, ic0, jc0, kc1, ip0, jp0, kp1), (w101, ic1, jc0, kc1, ip1, jp0, kp1),
                (w011, ic0, jc1, kc1, ip0, jp1, kp1), (w111, ic1, jc1, kc1, ip1, jp1, kp1))
                r = omtf * rho_prev[ipp, jpp, kpp] + tf * rho_curr[ic, jc, kc]
                u = omtf * ux_prev[ipp, jpp, kpp]  + tf * ux_curr[ic, jc, kc]
                v = omtf * uy_prev[ipp, jpp, kpp]  + tf * uy_curr[ic, jc, kc]
                w = omtf * uz_prev[ipp, jpp, kpp]  + tf * uz_curr[ic, jc, kc]
                rho_f += ws * r; uxf += ws * u; uyf += ws * v; uzf += ws * w
            end

            for q in 1:19
                feq_fine = _feq_3d(q, rho_f, uxf, uyf, uzf)

                fneq = zero(T)
                for (ws, ic, jc, kc, ipp, jpp, kpp) in (
                    (w000, ic0, jc0, kc0, ip0, jp0, kp0), (w100, ic1, jc0, kc0, ip1, jp0, kp0),
                    (w010, ic0, jc1, kc0, ip0, jp1, kp0), (w110, ic1, jc1, kc0, ip1, jp1, kp0),
                    (w001, ic0, jc0, kc1, ip0, jp0, kp1), (w101, ic1, jc0, kc1, ip1, jp0, kp1),
                    (w011, ic0, jc1, kc1, ip0, jp1, kp1), (w111, ic1, jc1, kc1, ip1, jp1, kp1))
                    r = omtf * rho_prev[ipp, jpp, kpp] + tf * rho_curr[ic, jc, kc]
                    u = omtf * ux_prev[ipp, jpp, kpp]  + tf * ux_curr[ic, jc, kc]
                    v = omtf * uy_prev[ipp, jpp, kpp]  + tf * uy_curr[ic, jc, kc]
                    ww = omtf * uz_prev[ipp, jpp, kpp] + tf * uz_curr[ic, jc, kc]
                    f_interp = omtf * f_prev[ipp, jpp, kpp, q] + tf * f_curr[ic, jc, kc, q]
                    fneq += ws * (f_interp - _feq_3d(q, r, u, v, ww))
                end

                f_fine[i_f, j_f, k_f, q] = feq_fine + alpha * fneq
            end
        end
    end
end

function prolongate_f_rescaled_temporal_3d!(f_fine,
        f_curr, rho_curr, ux_curr, uy_curr, uz_curr,
        f_prev, rho_prev, ux_prev, uy_prev, uz_prev,
        ratio, Nx_inner, Ny_inner, Nz_inner, n_ghost,
        i_c_start, j_c_start, k_c_start,
        Nx_c, Ny_c, Nz_c, omega_c, omega_f, t_frac)
    backend = KernelAbstractions.get_backend(f_fine)
    Nx_f = Nx_inner + 2 * n_ghost
    Ny_f = Ny_inner + 2 * n_ghost
    Nz_f = Nz_inner + 2 * n_ghost
    kernel! = prolongate_f_rescaled_temporal_3d_kernel!(backend)
    kernel!(f_fine,
            f_curr, rho_curr, ux_curr, uy_curr, uz_curr,
            f_prev, rho_prev, ux_prev, uy_prev, uz_prev,
            ratio, Nx_inner, Ny_inner, Nz_inner, n_ghost,
            i_c_start, j_c_start, k_c_start,
            Nx_c, Ny_c, Nz_c, omega_c, omega_f, eltype(f_fine)(t_frac);
            ndrange=(Nx_f, Ny_f, Nz_f))
    KernelAbstractions.synchronize(backend)
end

# =====================================================================
# Restriction: fine → coarse (block-average + inverse Filippova-Hanel)
# =====================================================================

@kernel function restrict_f_rescaled_3d_kernel!(
        f_coarse, rho_c, ux_c, uy_c, uz_c,
        @Const(f_fine), @Const(rho_f), @Const(ux_f), @Const(uy_f), @Const(uz_f),
        ratio, n_ghost, i_c_start, j_c_start, k_c_start,
        omega_c, omega_f)

    ic, jc, kc = @index(Global, NTuple)

    @inbounds begin
        T = eltype(f_coarse)
        alpha_f2c = (T(omega_c) - T(0.5)) * (one(T) / (T(omega_f) - T(0.5)))
        inv_r3 = one(T) / T(ratio * ratio * ratio)

        i_c = ic + i_c_start - 1
        j_c = jc + j_c_start - 1
        k_c = kc + k_c_start - 1

        # Block-average macroscopic
        rho_avg = zero(T); ux_avg = zero(T); uy_avg = zero(T); uz_avg = zero(T)
        for dk in 0:ratio-1, dj in 0:ratio-1, di in 0:ratio-1
            i_ff = n_ghost + (ic - 1) * ratio + di + 1
            j_ff = n_ghost + (jc - 1) * ratio + dj + 1
            k_ff = n_ghost + (kc - 1) * ratio + dk + 1
            rho_avg += rho_f[i_ff, j_ff, k_ff]
            ux_avg  += ux_f[i_ff, j_ff, k_ff]
            uy_avg  += uy_f[i_ff, j_ff, k_ff]
            uz_avg  += uz_f[i_ff, j_ff, k_ff]
        end
        rho_avg *= inv_r3; ux_avg *= inv_r3; uy_avg *= inv_r3; uz_avg *= inv_r3

        rho_c[i_c, j_c, k_c] = rho_avg
        ux_c[i_c, j_c, k_c] = ux_avg
        uy_c[i_c, j_c, k_c] = uy_avg
        uz_c[i_c, j_c, k_c] = uz_avg

        # Block-average f_neq + inverse rescale
        for q in 1:19
            feq_c = _feq_3d(q, rho_avg, ux_avg, uy_avg, uz_avg)
            fneq_avg = zero(T)
            for dk in 0:ratio-1, dj in 0:ratio-1, di in 0:ratio-1
                i_ff = n_ghost + (ic - 1) * ratio + di + 1
                j_ff = n_ghost + (jc - 1) * ratio + dj + 1
                k_ff = n_ghost + (kc - 1) * ratio + dk + 1
                fneq_avg += f_fine[i_ff, j_ff, k_ff, q] -
                    _feq_3d(q, rho_f[i_ff,j_ff,k_ff], ux_f[i_ff,j_ff,k_ff],
                            uy_f[i_ff,j_ff,k_ff], uz_f[i_ff,j_ff,k_ff])
            end
            fneq_avg *= inv_r3
            f_coarse[i_c, j_c, k_c, q] = feq_c + alpha_f2c * fneq_avg
        end
    end
end

function restrict_f_rescaled_3d!(f_coarse, rho_c, ux_c, uy_c, uz_c,
                                  f_fine, rho_f, ux_f, uy_f, uz_f,
                                  ratio, n_ghost, i_c_start, j_c_start, k_c_start,
                                  Nx_overlap, Ny_overlap, Nz_overlap,
                                  omega_c, omega_f)
    backend = KernelAbstractions.get_backend(f_coarse)
    kernel! = restrict_f_rescaled_3d_kernel!(backend)
    kernel!(f_coarse, rho_c, ux_c, uy_c, uz_c,
            f_fine, rho_f, ux_f, uy_f, uz_f,
            ratio, n_ghost, i_c_start, j_c_start, k_c_start,
            omega_c, omega_f;
            ndrange=(Nx_overlap, Ny_overlap, Nz_overlap))
    KernelAbstractions.synchronize(backend)
end
