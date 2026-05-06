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
            # Hermite stress source fused into the BGK collision.
            pre = -ω * T(9.0/2.0)
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

# --- Standalone Hermite stress source (post-collision addition) ---
# Used when the collision step has already been performed separately
# (e.g., by `fused_trt_libb_v2_step!`) and we only need to inject τ_p
# into f_out. Uses the TRT symmetric rate s_plus in place of ω.

@kernel function apply_hermite_source_2d_kernel!(f, @Const(is_solid), s_plus,
                                                   @Const(tau_p_xx),
                                                   @Const(tau_p_xy),
                                                   @Const(tau_p_yy),
                                                   source_scale,
                                                   apply_y_domain_walls,
                                                   Ny)
    i, j = @index(Global, NTuple)
    @inbounds if !is_solid[i, j] &&
                 (apply_y_domain_walls || (j != 1 && j != Ny))
        T = eltype(f)
        txx = tau_p_xx[i,j]; txy = tau_p_xy[i,j]; tyy = tau_p_yy[i,j]
        pre = -s_plus * T(9.0/2.0) * source_scale
        cs2 = T(1/3)
        wr = T(4/9); wa = T(1/9); we = T(1/36)

        T1 = pre * wr * (-cs2*(txx + tyy))
        T2 = pre * wa * ((one(T)-cs2)*txx - cs2*tyy)
        T3 = pre * wa * (-cs2*txx + (one(T)-cs2)*tyy)
        T6 = pre * we * ((one(T)-cs2)*txx + (one(T)-cs2)*tyy + T(2)*txy)
        T7 = pre * we * ((one(T)-cs2)*txx + (one(T)-cs2)*tyy - T(2)*txy)

        f[i,j,1] += T1
        f[i,j,2] += T2; f[i,j,4] += T2
        f[i,j,3] += T3; f[i,j,5] += T3
        f[i,j,6] += T6; f[i,j,8] += T6
        f[i,j,7] += T7; f[i,j,9] += T7
    end
end

"""
    apply_hermite_source_2d!(f, is_solid, s_plus, tau_p_xx, tau_p_xy, tau_p_yy;
                             ce_correction=true)

Post-collision injection of the viscoelastic Hermite stress source T_q.
Use when the BGK/TRT collision has already been performed by a separate
kernel (e.g. `fused_trt_libb_v2_step!`) and only the τ_p contribution
remains to be added.

`ce_correction=true` applies the standard post-collision half-step scaling
`1/(1-s_plus/2)`, giving the bulk moment closure
`ΔΠ_αβ = -s_plus τ_αβ/(1-s_plus/2)`. `ce_correction=false` matches the
local in-collision Liu/Yu source amplitude, `ΔΠ_αβ = -s_plus τ_αβ`, and is
kept as a diagnostic for separating source-amplitude and boundary-MEA effects.

For TRT, pass `s_plus = 1/(3ν+0.5)` (symmetric rate, controls viscosity).
For BGK, pass `s_plus = ω`.
"""
function apply_hermite_source_2d!(f, is_solid, s_plus,
                                    tau_p_xx, tau_p_xy, tau_p_yy;
                                    ce_correction::Bool=true,
                                    source_scale::Real=1,
                                    apply_y_domain_walls::Bool=true)
    backend = KernelAbstractions.get_backend(f)
    Nx, Ny = size(f, 1), size(f, 2)
    T = eltype(f)
    source_scale_t = T(source_scale) *
        (ce_correction ? inv(one(T) - T(s_plus) / T(2)) : one(T))
    kernel! = apply_hermite_source_2d_kernel!(backend)
    kernel!(f, is_solid, T(s_plus), tau_p_xx, tau_p_xy, tau_p_yy,
            source_scale_t, apply_y_domain_walls, Ny;
            ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

@inline function _has_wall_cut_link_2d(q_wall, i, j)
    for q in 2:9
        q_wall[i, j, q] > zero(eltype(q_wall)) && return true
    end
    return false
end

@inline function _clean_stress_sample_2d(is_solid, q_wall, i, j, Nx, Ny)
    return 1 <= i <= Nx && 1 <= j <= Ny &&
           !is_solid[i, j] &&
           !_has_wall_cut_link_2d(q_wall, i, j)
end

@inline function _lagrange_extrapolate_to_wall_cell_2d(y1::T, s1::T,
                                                       y2::T, s2::T) where {T}
    return (s2 * y1 - s1 * y2) / (s2 - s1)
end

@inline function _lagrange_extrapolate_to_wall_cell_2d(y1::T, s1::T,
                                                       y2::T, s2::T,
                                                       y3::T, s3::T) where {T}
    l1 = (s2 * s3) / ((s1 - s2) * (s1 - s3))
    l2 = (s1 * s3) / ((s2 - s1) * (s2 - s3))
    l3 = (s1 * s2) / ((s3 - s1) * (s3 - s2))
    return l1 * y1 + l2 * y2 + l3 * y3
end

@inline function _reconstruct_wall_cell_from_interior_2d(a, is_solid, q_wall,
                                                         i, j, q, Nx, Ny,
                                                         order)
    T = eltype(a)
    cxq = _cx_q(q)
    cyq = _cy_q(q)

    y1 = zero(T); y2 = zero(T); y3 = zero(T)
    s1 = zero(T); s2 = zero(T); s3 = zero(T)
    n = 0
    max_steps = max(Nx, Ny)
    for step in 1:max_steps
        ii = i - step * cxq
        jj = j - step * cyq
        if _clean_stress_sample_2d(is_solid, q_wall, ii, jj, Nx, Ny)
            n += 1
            if n == 1
                y1 = a[ii, jj]; s1 = T(step)
            elseif n == 2
                y2 = a[ii, jj]; s2 = T(step)
                order < 2 && break
            else
                y3 = a[ii, jj]; s3 = T(step)
                break
            end
        elseif !(1 <= ii <= Nx && 1 <= jj <= Ny) || is_solid[ii, jj]
            break
        end
    end

    if n >= 3 && order >= 2
        return _lagrange_extrapolate_to_wall_cell_2d(y1, s1, y2, s2, y3, s3), true
    elseif n >= 2
        return _lagrange_extrapolate_to_wall_cell_2d(y1, s1, y2, s2), true
    elseif n == 1
        return y1, true
    end
    return a[i, j], false
end

@kernel function reconstruct_wall_cell_stress_from_interior_2d_kernel!(
        tau_xx_out, tau_xy_out, tau_yy_out,
        @Const(tau_xx), @Const(tau_xy), @Const(tau_yy),
        @Const(q_wall), @Const(is_solid), order, Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds begin
        T = eltype(tau_xx_out)
        if is_solid[i, j]
            tau_xx_out[i, j] = tau_xx[i, j]
            tau_xy_out[i, j] = tau_xy[i, j]
            tau_yy_out[i, j] = tau_yy[i, j]
        else
            sum_xx = zero(T)
            sum_xy = zero(T)
            sum_yy = zero(T)
            n = zero(T)
            for q in 2:9
                if q_wall[i, j, q] > zero(T)
                    vxx, ok_xx = _reconstruct_wall_cell_from_interior_2d(
                        tau_xx, is_solid, q_wall, i, j, q, Nx, Ny, order)
                    vxy, ok_xy = _reconstruct_wall_cell_from_interior_2d(
                        tau_xy, is_solid, q_wall, i, j, q, Nx, Ny, order)
                    vyy, ok_yy = _reconstruct_wall_cell_from_interior_2d(
                        tau_yy, is_solid, q_wall, i, j, q, Nx, Ny, order)
                    if ok_xx && ok_xy && ok_yy
                        sum_xx += vxx
                        sum_xy += vxy
                        sum_yy += vyy
                        n += one(T)
                    end
                end
            end
            if n > zero(T)
                tau_xx_out[i, j] = sum_xx / n
                tau_xy_out[i, j] = sum_xy / n
                tau_yy_out[i, j] = sum_yy / n
            else
                tau_xx_out[i, j] = tau_xx[i, j]
                tau_xy_out[i, j] = tau_xy[i, j]
                tau_yy_out[i, j] = tau_yy[i, j]
            end
        end
    end
end

"""
    reconstruct_wall_cell_stress_from_interior_2d!(out_xx, out_xy, out_yy,
        tau_xx, tau_xy, tau_yy, q_wall, is_solid; order=1)

Build a cell-centered physical stress field for Hermite source evaluation near
curved walls. Boundary-adjacent cells are reconstructed from interior samples
along cut-links; other fluid cells are copied unchanged.
"""
function reconstruct_wall_cell_stress_from_interior_2d!(
        out_xx, out_xy, out_yy, tau_xx, tau_xy, tau_yy, q_wall, is_solid;
        order::Integer=1)
    backend = KernelAbstractions.get_backend(out_xx)
    Nx, Ny = size(out_xx)
    kernel! = reconstruct_wall_cell_stress_from_interior_2d_kernel!(backend)
    kernel!(out_xx, out_xy, out_yy, tau_xx, tau_xy, tau_yy, q_wall, is_solid,
            Int(order), Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end
