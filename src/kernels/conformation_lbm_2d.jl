using KernelAbstractions

# --- Conformation tensor LBM (TRT, D2Q9) for viscoelastic flows ---
#
# Reference: Liu et al., "An improved lattice Boltzmann method with a
# novel conservative boundary scheme for viscoelastic fluid flows",
# arxiv 2508.16997 (Aug 2025).
#
# Three independent scalar D2Q9 advection-diffusion-reaction LBMs evolve
# C_xx, C_xy, C_yy. The diffusion is built into the LBM via the relaxation
# time: κ = (τp,1 - 0.5)/3. TRT (two-relaxation-time) gives better stability.
# Source S = Φ_αβ contains the upper-convected derivative + relaxation.

# D2Q9 indexing (Kraken convention):
#   1: rest (0,0)        opp = 1
#   2: E   (+1, 0)       opp = 4
#   3: N   ( 0,+1)       opp = 5
#   4: W   (-1, 0)       opp = 2
#   5: S   ( 0,-1)       opp = 3
#   6: NE  (+1,+1)       opp = 8
#   7: NW  (-1,+1)       opp = 9
#   8: SW  (-1,-1)       opp = 6
#   9: SE  (+1,-1)       opp = 7

@kernel function collide_conformation_2d_kernel!(g, @Const(C_field), @Const(ux), @Const(uy),
                                                   @Const(C_xx_f), @Const(C_xy_f), @Const(C_yy_f),
                                                   @Const(is_solid),
                                                   tau_plus, tau_minus, lambda,
                                                   component, Nx, Ny)
    i, j = @index(Global, NTuple)

    @inbounds if !is_solid[i, j]
        T = eltype(g)
        φ = C_field[i, j]
        u = ux[i, j]
        v = uy[i, j]
        usq = u*u + v*v

        # Velocity gradient (central differences)
        ip = ifelse(i < Nx, i + 1, 1)
        im = ifelse(i > 1,  i - 1, Nx)
        jp = min(j + 1, Ny)
        jm = max(j - 1, 1)

        dudx = (ux[ip,j] - ux[im,j]) / T(2)
        dudy = (ux[i,jp] - ux[i,jm]) / T(2)
        dvdx = (uy[ip,j] - uy[im,j]) / T(2)
        dvdy = (uy[i,jp] - uy[i,jm]) / T(2)

        # Source term Φ_αβ
        cxx = C_xx_f[i, j]; cxy = C_xy_f[i, j]; cyy = C_yy_f[i, j]
        inv_λ = one(T) / T(lambda)
        S = zero(T)
        if component == 1   # xx
            S = -inv_λ * (cxx - one(T)) + T(2) * (cxx*dudx + cxy*dudy)
        elseif component == 2   # xy
            S = -inv_λ * cxy + (cxx*dvdx + cyy*dudy + cxy*(dudx + dvdy))
        else                # yy
            S = -inv_λ * (cyy - one(T)) + T(2) * (cxy*dvdx + cyy*dvdy)
        end

        # Pre-load all 9 populations
        g1 = g[i,j,1]; g2 = g[i,j,2]; g3 = g[i,j,3]; g4 = g[i,j,4]; g5 = g[i,j,5]
        g6 = g[i,j,6]; g7 = g[i,j,7]; g8 = g[i,j,8]; g9 = g[i,j,9]

        # Equilibria (reuse feq_2d with φ instead of ρ)
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

        # TRT: each pair (q, opp) is collided together.
        # For q=1 (rest, self-opposite), only the symmetric part exists.
        nq1 = g1 - ge1   # already symmetric (opp=1)
        g[i,j,1] = g1 - ωp * nq1 + wr * S

        # Pair (2, 4) — E and W
        gp24 = (g2 + g4) * half;  gm24 = (g2 - g4) * half
        ep24 = (ge2 + ge4) * half; em24 = (ge2 - ge4) * half
        post2 = g2 - ωp*(gp24 - ep24) - ωm*(gm24 - em24)
        post4 = g4 - ωp*(gp24 - ep24) - ωm*(-(gm24 - em24))
        g[i,j,2] = post2 + wa * S
        g[i,j,4] = post4 + wa * S

        # Pair (3, 5) — N and S
        gp35 = (g3 + g5) * half;  gm35 = (g3 - g5) * half
        ep35 = (ge3 + ge5) * half; em35 = (ge3 - ge5) * half
        post3 = g3 - ωp*(gp35 - ep35) - ωm*(gm35 - em35)
        post5 = g5 - ωp*(gp35 - ep35) - ωm*(-(gm35 - em35))
        g[i,j,3] = post3 + wa * S
        g[i,j,5] = post5 + wa * S

        # Pair (6, 8) — NE and SW
        gp68 = (g6 + g8) * half;  gm68 = (g6 - g8) * half
        ep68 = (ge6 + ge8) * half; em68 = (ge6 - ge8) * half
        post6 = g6 - ωp*(gp68 - ep68) - ωm*(gm68 - em68)
        post8 = g8 - ωp*(gp68 - ep68) - ωm*(-(gm68 - em68))
        g[i,j,6] = post6 + we * S
        g[i,j,8] = post8 + we * S

        # Pair (7, 9) — NW and SE
        gp79 = (g7 + g9) * half;  gm79 = (g7 - g9) * half
        ep79 = (ge7 + ge9) * half; em79 = (ge7 - ge9) * half
        post7 = g7 - ωp*(gp79 - ep79) - ωm*(gm79 - em79)
        post9 = g9 - ωp*(gp79 - ep79) - ωm*(-(gm79 - em79))
        g[i,j,7] = post7 + we * S
        g[i,j,9] = post9 + we * S
    end
end

"""
    collide_conformation_2d!(g, C_field, ux, uy, C_xx, C_xy, C_yy, is_solid,
                              tau_plus, lambda; magic=0.25, component=1)

TRT collision + source for one scalar component of the conformation tensor.
`g` : D2Q9 distributions for that component, shape (Nx, Ny, 9)
`C_field` : the macroscopic value being evolved (= moments of g)
`C_xx, C_xy, C_yy` : all 3 components needed for the source Φ
`component` : 1=xx, 2=xy, 3=yy
`tau_plus = τp,1` — sets diffusion: κ = (tau_plus - 0.5)/3
`tau_minus = magic/(tau_plus - 0.5) + 0.5` — TRT magic-parameter relation
"""
function collide_conformation_2d!(g, C_field, ux, uy, C_xx, C_xy, C_yy, is_solid,
                                   tau_plus, lambda; magic=0.25, component=1)
    backend = KernelAbstractions.get_backend(g)
    Nx, Ny = size(C_field)
    T = eltype(g)
    tau_minus = magic / (tau_plus - 0.5) + 0.5
    kernel! = collide_conformation_2d_kernel!(backend)
    kernel!(g, C_field, ux, uy, C_xx, C_xy, C_yy, is_solid,
            T(tau_plus), T(tau_minus), T(lambda),
            Int(component), Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

# ============================================================
# Initialization: g = g^eq with given C field and velocity field
# ============================================================

@kernel function init_conformation_field_2d_kernel!(g, @Const(C_field), @Const(ux), @Const(uy))
    i, j = @index(Global, NTuple)
    @inbounds begin
        T = eltype(g)
        φ = C_field[i, j]
        u = ux[i, j]
        v = uy[i, j]
        usq = u*u + v*v
        g[i,j,1] = feq_2d(Val(1), φ, u, v, usq)
        g[i,j,2] = feq_2d(Val(2), φ, u, v, usq)
        g[i,j,3] = feq_2d(Val(3), φ, u, v, usq)
        g[i,j,4] = feq_2d(Val(4), φ, u, v, usq)
        g[i,j,5] = feq_2d(Val(5), φ, u, v, usq)
        g[i,j,6] = feq_2d(Val(6), φ, u, v, usq)
        g[i,j,7] = feq_2d(Val(7), φ, u, v, usq)
        g[i,j,8] = feq_2d(Val(8), φ, u, v, usq)
        g[i,j,9] = feq_2d(Val(9), φ, u, v, usq)
    end
end

function init_conformation_field_2d!(g, C_field, ux, uy)
    backend = KernelAbstractions.get_backend(g)
    Nx, Ny = size(C_field)
    kernel! = init_conformation_field_2d_kernel!(backend)
    kernel!(g, C_field, ux, uy; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

# ============================================================
# Macroscopic recovery: φ = sum_q g_q
# ============================================================

@kernel function compute_conformation_macro_2d_kernel!(C_field, @Const(g))
    i, j = @index(Global, NTuple)
    @inbounds begin
        C_field[i,j] = g[i,j,1] + g[i,j,2] + g[i,j,3] + g[i,j,4] + g[i,j,5] +
                       g[i,j,6] + g[i,j,7] + g[i,j,8] + g[i,j,9]
    end
end

function compute_conformation_macro_2d!(C_field, g)
    backend = KernelAbstractions.get_backend(g)
    Nx, Ny = size(C_field)
    kernel! = compute_conformation_macro_2d_kernel!(backend)
    kernel!(C_field, g; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end

# ============================================================
# Conservative non-equilibrium bounce-back (CNEBB) at solid walls
# ============================================================

@kernel function apply_cnebb_2d_kernel!(g, @Const(is_solid), @Const(ux), @Const(uy),
                                          @Const(C_field))
    i, j = @index(Global, NTuple)
    @inbounds if is_solid[i, j]
        T = eltype(g)
        φ = C_field[i, j]
        u = ux[i, j]
        v = uy[i, j]
        usq = u*u + v*v
        g[i,j,1] = feq_2d(Val(1), φ, u, v, usq)
        g[i,j,2] = feq_2d(Val(2), φ, u, v, usq)
        g[i,j,3] = feq_2d(Val(3), φ, u, v, usq)
        g[i,j,4] = feq_2d(Val(4), φ, u, v, usq)
        g[i,j,5] = feq_2d(Val(5), φ, u, v, usq)
        g[i,j,6] = feq_2d(Val(6), φ, u, v, usq)
        g[i,j,7] = feq_2d(Val(7), φ, u, v, usq)
        g[i,j,8] = feq_2d(Val(8), φ, u, v, usq)
        g[i,j,9] = feq_2d(Val(9), φ, u, v, usq)
    end
end

function apply_cnebb_conformation_2d!(g, is_solid, ux, uy, C_field)
    backend = KernelAbstractions.get_backend(g)
    Nx, Ny = size(C_field)
    kernel! = apply_cnebb_2d_kernel!(backend)
    kernel!(g, is_solid, ux, uy, C_field; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
end
