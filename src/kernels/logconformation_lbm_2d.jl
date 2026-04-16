using KernelAbstractions

# =====================================================================
# Log-conformation TRT-LBM for viscoelastic flows (Fattal & Kupferman
# 2004; Afonso et al. 2009, Hulsen et al. 2005).
#
# Evolves Ψ = log(C) instead of C. The log transformation preserves the
# positive-definiteness of C at the discrete level, which is required
# for stability at high Weissenberg number (HWNP).
#
# For Oldroyd-B:
#   ∂_t Ψ + u·∇Ψ = 2·B + [Ω, Ψ] - (I - exp(-Ψ))/λ
#
# where in the eigenframe of Ψ (eigenvalues Λ_1, Λ_2 and rotation R):
#   ∇u' = R^T · ∇u · R
#   Ω'_12 = (Λ_2 · ∇u'_12 + Λ_1 · ∇u'_21) / (Λ_2 - Λ_1),  Ω'_11=Ω'_22=0
#   B'   = diag(∇u'_11, ∇u'_22)
#
# All 2×2 operations are analytical (eigendecomposition of symmetric
# Ψ uses atan2).
#
# The scheme reuses the same 9-population D2Q9 TRT-LBM infrastructure
# as collide_conformation_2d!, swapping only the source term.
# =====================================================================

# Degenerate-eigenvalue tolerance: |Λ₁ − Λ₂| below this falls back
# to vorticity-only source for Ω (avoids 0/0).
const _LOGCONF_DEGEN_TOL = 1e-10

"""
    logconf_source_2d(Ψxx, Ψxy, Ψyy, dudx, dudy, dvdx, dvdy, λ, component)

Return the source term for the log-conformation Oldroyd-B evolution of
Ψ = log(C) in the lab frame, for one of the three independent
components (1=xx, 2=xy, 3=yy).

Uses analytical 2×2 eigendecomposition of the symmetric Ψ matrix.
"""
@inline function logconf_source_2d(Ψxx::T, Ψxy::T, Ψyy::T,
                                     dudx::T, dudy::T, dvdx::T, dvdy::T,
                                     λ::T, component::Int) where {T<:AbstractFloat}
    # Eigendecomposition of symmetric Ψ = [Ψxx Ψxy; Ψxy Ψyy]
    tr = Ψxx + Ψyy
    diff = Ψxx - Ψyy
    disc = sqrt(diff*diff + T(4)*Ψxy*Ψxy)
    Λ1 = T(0.5) * (tr + disc)
    Λ2 = T(0.5) * (tr - disc)
    # Rotation angle: tan(2θ) = 2Ψxy / (Ψxx - Ψyy)
    θ = T(0.5) * atan(T(2)*Ψxy, diff)
    c = cos(θ); s = sin(θ)

    # ∇u in eigenframe: ∇u' = R^T · ∇u · R  (R = [c -s; s c])
    # ∇u'_11 = c²·dudx + cs·(dudy+dvdx) + s²·dvdy
    # ∇u'_12 = -cs·dudx + c²·dudy - s²·dvdx + cs·dvdy
    # ∇u'_21 = -cs·dudx - s²·dudy + c²·dvdx + cs·dvdy
    # ∇u'_22 = s²·dudx - cs·(dudy+dvdx) + c²·dvdy
    cs = c * s
    c2 = c * c
    s2 = s * s
    g11 = c2*dudx + cs*(dudy + dvdx) + s2*dvdy
    g12 = -cs*dudx + c2*dudy - s2*dvdx + cs*dvdy
    g21 = -cs*dudx - s2*dudy + c2*dvdx + cs*dvdy
    g22 = s2*dudx - cs*(dudy + dvdx) + c2*dvdy

    # Source in eigenframe: S' = 2·B̃' + [Ω̃', Ψ'] − (I − exp(−Ψ'))/λ
    # In the eigenframe of Ψ:
    #   - Non-degenerate (Λ_1 ≠ Λ_2): B̃' is diagonal (Fattal-Kupferman
    #     decomposition: off-diagonal strain absorbed into Ω̃'); the
    #     off-diagonal source is the commutator [Ω̃', Ψ'] = Λ_2 g'_12
    #     + Λ_1 g'_21.
    #   - Degenerate (Λ_1 = Λ_2): Ψ' is a scalar multiple of I; the
    #     eigenframe is arbitrary; B̃' = full symmetric ∇u' and
    #     [Ω̃', Ψ'] = 0. The off-diagonal source is then 2·B'_12 =
    #     g'_12 + g'_21 (symmetric strain).
    ΔΛ = Λ1 - Λ2
    inv_λ = one(T) / λ
    S11p = T(2)*g11 - (one(T) - exp(-Λ1)) * inv_λ
    S22p = T(2)*g22 - (one(T) - exp(-Λ2)) * inv_λ
    # Off-diagonal source: Loewner derivative formula for f = exp.
    # (d log C / dt)'_12 = [(Λ_1 − Λ_2) / (exp Λ_1 − exp Λ_2)]
    #                     · [exp(Λ_2)·g'_12 + exp(Λ_1)·g'_21]
    # At Λ_1 = Λ_2 the limit is g'_12 + g'_21.
    eΛ1 = exp(Λ1); eΛ2 = exp(Λ2)
    Δe = eΛ1 - eΛ2
    S12p = if abs(Δe) > T(_LOGCONF_DEGEN_TOL)
        (ΔΛ / Δe) * (eΛ2 * g12 + eΛ1 * g21)
    else
        g12 + g21
    end

    # Rotate back: S = R · S' · R^T
    # S_xx = c²·S'_11 - 2·cs·S'_12 + s²·S'_22
    # S_xy = cs·S'_11 + (c²-s²)·S'_12 - cs·S'_22
    # S_yy = s²·S'_11 + 2·cs·S'_12 + c²·S'_22
    # (S'_12 = S'_21 in eigenframe; Ω contribution is antisymmetric but
    #  [Ω, Ψ'] is symmetric after commutation since Ψ' is diagonal)
    if component == 1
        return c2*S11p - T(2)*cs*S12p + s2*S22p
    elseif component == 2
        return cs*S11p + (c2 - s2)*S12p - cs*S22p
    else
        return s2*S11p + T(2)*cs*S12p + c2*S22p
    end
end

# =====================================================================
# Collision kernel for log-conformation (TRT, D2Q9).
# Identical to collide_conformation_2d_kernel! except the source `S`
# is replaced by logconf_source_2d (Fattal-Kupferman).
# =====================================================================

@kernel function collide_logconf_2d_kernel!(g, @Const(Ψ_field), @Const(ux), @Const(uy),
                                              @Const(Ψ_xx_f), @Const(Ψ_xy_f), @Const(Ψ_yy_f),
                                              @Const(is_solid),
                                              tau_plus, tau_minus, lambda,
                                              component, Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds if !is_solid[i, j]
        T = eltype(g)
        φ = Ψ_field[i, j]
        u = ux[i, j]
        v = uy[i, j]
        usq = u*u + v*v

        ip = ifelse(i < Nx, i + 1, 1)
        im = ifelse(i > 1,  i - 1, Nx)
        jp = min(j + 1, Ny)
        jm = max(j - 1, 1)

        dudx = (ux[ip,j] - ux[im,j]) / T(2)
        dudy = (ux[i,jp] - ux[i,jm]) / T(2)
        dvdx = (uy[ip,j] - uy[im,j]) / T(2)
        dvdy = (uy[i,jp] - uy[i,jm]) / T(2)

        Ψxx = Ψ_xx_f[i, j]; Ψxy = Ψ_xy_f[i, j]; Ψyy = Ψ_yy_f[i, j]
        S = logconf_source_2d(Ψxx, Ψxy, Ψyy, dudx, dudy, dvdx, dvdy,
                                T(lambda), component)

        g1 = g[i,j,1]; g2 = g[i,j,2]; g3 = g[i,j,3]; g4 = g[i,j,4]; g5 = g[i,j,5]
        g6 = g[i,j,6]; g7 = g[i,j,7]; g8 = g[i,j,8]; g9 = g[i,j,9]

        ge1 = feq_2d(Val(1), φ, u, v, usq)
        ge2 = feq_2d(Val(2), φ, u, v, usq)
        ge3 = feq_2d(Val(3), φ, u, v, usq)
        ge4 = feq_2d(Val(4), φ, u, v, usq)
        ge5 = feq_2d(Val(5), φ, u, v, usq)
        ge6 = feq_2d(Val(6), φ, u, v, usq)
        ge7 = feq_2d(Val(7), φ, u, v, usq)
        ge8 = feq_2d(Val(8), φ, u, v, usq)
        ge9 = feq_2d(Val(9), φ, u, v, usq)

        ωp = one(T) / T(tau_plus)
        ωm = one(T) / T(tau_minus)
        half = T(0.5)
        wr = T(4/9); wa = T(1/9); we = T(1/36)

        nq1 = g1 - ge1
        g[i,j,1] = g1 - ωp * nq1 + wr * S

        gp24 = (g2 + g4) * half;  gm24 = (g2 - g4) * half
        ep24 = (ge2 + ge4) * half; em24 = (ge2 - ge4) * half
        post2 = g2 - ωp*(gp24 - ep24) - ωm*(gm24 - em24)
        post4 = g4 - ωp*(gp24 - ep24) - ωm*(-(gm24 - em24))
        g[i,j,2] = post2 + wa * S
        g[i,j,4] = post4 + wa * S

        gp35 = (g3 + g5) * half;  gm35 = (g3 - g5) * half
        ep35 = (ge3 + ge5) * half; em35 = (ge3 - ge5) * half
        post3 = g3 - ωp*(gp35 - ep35) - ωm*(gm35 - em35)
        post5 = g5 - ωp*(gp35 - ep35) - ωm*(-(gm35 - em35))
        g[i,j,3] = post3 + wa * S
        g[i,j,5] = post5 + wa * S

        gp68 = (g6 + g8) * half;  gm68 = (g6 - g8) * half
        ep68 = (ge6 + ge8) * half; em68 = (ge6 - ge8) * half
        post6 = g6 - ωp*(gp68 - ep68) - ωm*(gm68 - em68)
        post8 = g8 - ωp*(gp68 - ep68) - ωm*(-(gm68 - em68))
        g[i,j,6] = post6 + we * S
        g[i,j,8] = post8 + we * S

        gp79 = (g7 + g9) * half;  gm79 = (g7 - g9) * half
        ep79 = (ge7 + ge9) * half; em79 = (ge7 - ge9) * half
        post7 = g7 - ωp*(gp79 - ep79) - ωm*(gm79 - em79)
        post9 = g9 - ωp*(gp79 - ep79) - ωm*(-(gm79 - em79))
        g[i,j,7] = post7 + we * S
        g[i,j,9] = post9 + we * S
    end
end

"""
    collide_logconf_2d!(g, Ψ_field, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid,
                         tau_plus, lambda; magic=0.25, component=1)

TRT collision + Fattal-Kupferman source for one scalar component of
Ψ = log(C). Drop-in replacement for `collide_conformation_2d!` when
evolving Ψ rather than C directly. Use together with
`update_polymer_stress!(..., ::LogConfOldroydB)` which reconstructs
C = exp(Ψ) via 2×2 eigendecomposition at each cell.
"""
function collide_logconf_2d!(g, Ψ_field, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid,
                                tau_plus, lambda; magic=0.25, component=1)
    backend = KernelAbstractions.get_backend(g)
    Nx, Ny = size(Ψ_field)
    T = eltype(g)
    tau_minus = magic / (tau_plus - 0.5) + 0.5
    kernel! = collide_logconf_2d_kernel!(backend)
    kernel!(g, Ψ_field, ux, uy, Ψ_xx, Ψ_xy, Ψ_yy, is_solid,
            T(tau_plus), T(tau_minus), T(lambda),
            Int(component), Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

# =====================================================================
# Ψ ↔ C conversion (analytical 2×2 symmetric matrix log/exp)
# =====================================================================

@kernel function psi_to_C_2d_kernel!(C_xx, C_xy, C_yy,
                                       @Const(Ψ_xx), @Const(Ψ_xy), @Const(Ψ_yy))
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(C_xx)
        Ψxx = Ψ_xx[i,j]; Ψxy = Ψ_xy[i,j]; Ψyy = Ψ_yy[i,j]
        tr = Ψxx + Ψyy
        diff = Ψxx - Ψyy
        disc = sqrt(diff*diff + T(4)*Ψxy*Ψxy)
        Λ1 = T(0.5) * (tr + disc)
        Λ2 = T(0.5) * (tr - disc)
        e1 = exp(Λ1); e2 = exp(Λ2)
        θ = T(0.5) * atan(T(2)*Ψxy, diff)
        c = cos(θ); s = sin(θ)
        C_xx[i,j] = c*c*e1 + s*s*e2
        C_xy[i,j] = c*s*(e1 - e2)
        C_yy[i,j] = s*s*e1 + c*c*e2
    end
end

"""
    psi_to_C_2d!(C_xx, C_xy, C_yy, Ψ_xx, Ψ_xy, Ψ_yy)

Element-wise analytical C = exp(Ψ) for symmetric 2×2 Ψ via
eigendecomposition. In-place on `C_*`.
"""
function psi_to_C_2d!(C_xx, C_xy, C_yy, Ψ_xx, Ψ_xy, Ψ_yy)
    backend = KernelAbstractions.get_backend(C_xx)
    Nx, Ny = size(C_xx)
    kernel! = psi_to_C_2d_kernel!(backend)
    kernel!(C_xx, C_xy, C_yy, Ψ_xx, Ψ_xy, Ψ_yy; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

@kernel function C_to_psi_2d_kernel!(Ψ_xx, Ψ_xy, Ψ_yy,
                                       @Const(C_xx), @Const(C_xy), @Const(C_yy))
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(Ψ_xx)
        Cxx = C_xx[i,j]; Cxy = C_xy[i,j]; Cyy = C_yy[i,j]
        tr = Cxx + Cyy
        diff = Cxx - Cyy
        disc = sqrt(diff*diff + T(4)*Cxy*Cxy)
        μ1 = T(0.5) * (tr + disc)
        μ2 = T(0.5) * (tr - disc)
        # C must be positive definite (μ_i > 0)
        l1 = log(max(μ1, T(1e-30)))
        l2 = log(max(μ2, T(1e-30)))
        θ = T(0.5) * atan(T(2)*Cxy, diff)
        c = cos(θ); s = sin(θ)
        Ψ_xx[i,j] = c*c*l1 + s*s*l2
        Ψ_xy[i,j] = c*s*(l1 - l2)
        Ψ_yy[i,j] = s*s*l1 + c*c*l2
    end
end

"""
    C_to_psi_2d!(Ψ_xx, Ψ_xy, Ψ_yy, C_xx, C_xy, C_yy)

Element-wise analytical Ψ = log(C) for symmetric positive-definite 2×2 C.
In-place on `Ψ_*`. Guards against numerical underflow in the log.
"""
function C_to_psi_2d!(Ψ_xx, Ψ_xy, Ψ_yy, C_xx, C_xy, C_yy)
    backend = KernelAbstractions.get_backend(Ψ_xx)
    Nx, Ny = size(Ψ_xx)
    kernel! = C_to_psi_2d_kernel!(backend)
    kernel!(Ψ_xx, Ψ_xy, Ψ_yy, C_xx, C_xy, C_yy; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end
