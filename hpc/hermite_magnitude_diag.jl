# Hermite source magnitude test — 2D vs 3D pure isolation.
#
# Setup: uniform f at rest equilibrium (ρ=1, u=v=w=0). Apply Hermite
# source ONCE with a known τ_p (e.g. τ_xy = 1e-3, others = 0). Compute
# the 2nd-moment Π_αβ = sum_q c_q,α · c_q,β · f_q on the post-source
# populations. Compare to the expected closure.
#
# Expected closure (Liu 2025 form, post-collision injection on standard
# BGK with the (1 − s/2) division in pre):
#   T_q = -s_plus · 9/2 · w_q · H_{q,αβ} · τ_αβ / (1 − s_plus/2)
# Plugging into Π and using <H_αβ · H_γδ>_w = (1/9)·(δ_αγδ_βδ + δ_αδδ_βγ −
#   2/3·δ_αβ·δ_γδ) (deviatoric projector for D2Q9 / D3Q19 isotropic 4th-order):
#   ΔΠ_αβ = sum_q c·c·T_q = -s · τ_αβ / (1 − s/2)
#
# Then σ_p = (1 − s/2) · ΔΠ / s = -τ_αβ. Sign chosen so that adding the
# polymer Hermite source recovers the polymer stress in the LBM moment.
#
# Diagnostic: ratio σ_p_xy_recovered / τ_xy_input should be the SAME in 2D
# and 3D. Any difference reveals a magnitude/normalisation bug specific
# to one stencil.

using Kraken, Printf, KernelAbstractions

backend = KernelAbstractions.CPU()
FT = Float64

println("="^70)
println("Hermite source magnitude — 2D vs 3D pure isolation")
println("="^70)

τ_xy = 1e-3
ν    = 0.1                    # solvent viscosity
s    = 1.0 / (3.0 * ν + 0.5)  # TRT s_plus = BGK ω

println("Setup: τ_xy_input = $(τ_xy), ν_s = $(ν), s_plus = $(round(s, digits=4))")
println()

# ----------------------------------------------------------------------
# 2D test
# ----------------------------------------------------------------------
let
    Nx, Ny = 4, 4
    is_solid = KernelAbstractions.zeros(backend, Bool, Nx, Ny)
    f = KernelAbstractions.allocate(backend, FT, Nx, Ny, 9)
    # Initialize at rest equilibrium: ρ=1, u=v=0
    f_h = zeros(FT, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        f_h[i,j,q] = Kraken.equilibrium(D2Q9(), one(FT), zero(FT), zero(FT), q)
    end
    copyto!(f, f_h)

    txx = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    txy = KernelAbstractions.zeros(backend, FT, Nx, Ny); fill!(txy, FT(τ_xy))
    tyy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    apply_hermite_source_2d!(f, is_solid, FT(s), txx, txy, tyy)

    # Recovered Π_xy at centre cell
    cxv = [0, 1, 0, -1,  0, 1, -1, -1,  1]
    cyv = [0, 0, 1,  0, -1, 1,  1, -1, -1]
    fh = Array(f)
    Π_xy = 0.0
    for q in 1:9
        Π_xy += cxv[q] * cyv[q] * fh[2, 2, q]
    end
    # Subtract eq part: at u=0 ρ=1, Π^eq_xy = 0
    Π_xy_neq = Π_xy
    σ_p_xy_recovered = (1 - s/2) * Π_xy_neq / s
    @printf("2D: ΔΠ_xy = %.6e   σ_p_xy_recovered = (1-s/2)·ΔΠ/s = %.6e\n",
            Π_xy_neq, σ_p_xy_recovered)
    @printf("   ratio σ_p_recovered / τ_xy_input = %.6f  (target: -1.0)\n",
            σ_p_xy_recovered / τ_xy)
end

# ----------------------------------------------------------------------
# 3D test
# ----------------------------------------------------------------------
let
    Nx, Ny, Nz = 4, 4, 4
    is_solid = KernelAbstractions.zeros(backend, Bool, Nx, Ny, Nz)
    f = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz, 19)
    f_h = zeros(FT, Nx, Ny, Nz, 19)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx, q in 1:19
        f_h[i,j,k,q] = Kraken.equilibrium(D3Q19(), one(FT), zero(FT), zero(FT), zero(FT), q)
    end
    copyto!(f, f_h)

    txx = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    txy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz); fill!(txy, FT(τ_xy))
    txz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    tyy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    tyz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    tzz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    apply_hermite_source_3d!(f, is_solid, FT(s), txx, txy, txz, tyy, tyz, tzz)

    cxv = [0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0]
    cyv = [0,  0,  0,  1, -1,  0,  0,  1,  1, -1, -1,  0,  0,  0,  0,  1, -1,  1, -1]
    czv = [0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1,  1, -1, -1,  1,  1, -1, -1]
    fh = Array(f)
    Π_xy = 0.0; Π_xz = 0.0; Π_yz = 0.0
    Π_xx = 0.0; Π_yy = 0.0; Π_zz = 0.0
    for q in 1:19
        Π_xy += cxv[q] * cyv[q] * fh[2, 2, 2, q]
        Π_xz += cxv[q] * czv[q] * fh[2, 2, 2, q]
        Π_yz += cyv[q] * czv[q] * fh[2, 2, 2, q]
        Π_xx += cxv[q]^2 * fh[2, 2, 2, q]
        Π_yy += cyv[q]^2 * fh[2, 2, 2, q]
        Π_zz += czv[q]^2 * fh[2, 2, 2, q]
    end
    # Subtract Π^eq at u=0, ρ=1: Π^eq_αβ = ρ·cs²·δ_αβ = 1/3·δ
    Π_xx_neq = Π_xx - 1/3
    Π_yy_neq = Π_yy - 1/3
    Π_zz_neq = Π_zz - 1/3
    Π_xy_neq = Π_xy
    σ_p_xy_recovered = (1 - s/2) * Π_xy_neq / s
    σ_p_xx_recovered = (1 - s/2) * Π_xx_neq / s
    σ_p_yy_recovered = (1 - s/2) * Π_yy_neq / s
    σ_p_zz_recovered = (1 - s/2) * Π_zz_neq / s
    @printf("3D: ΔΠ_xy = %.6e   σ_p_xy_recovered = %.6e\n",
            Π_xy_neq, σ_p_xy_recovered)
    @printf("   ratio σ_p_recovered / τ_xy_input = %.6f  (target: -1.0)\n",
            σ_p_xy_recovered / τ_xy)
    @printf("3D crosstalk: σ_p_xx_recovered = %.6e  σ_p_yy_recovered = %.6e  σ_p_zz_recovered = %.6e\n",
            σ_p_xx_recovered, σ_p_yy_recovered, σ_p_zz_recovered)
    @printf("3D leak: ΔΠ_xz = %.6e   ΔΠ_yz = %.6e   (target: 0)\n", Π_xz, Π_yz)
end

# ----------------------------------------------------------------------
# Now repeat with τ_xx ≠ 0 (axial / first normal stress)
# ----------------------------------------------------------------------
println("\n--- τ_xx = 1e-3 only (axial / normal stress test) ---")
τ_xx_in = 1e-3

let
    Nx, Ny = 4, 4
    is_solid = KernelAbstractions.zeros(backend, Bool, Nx, Ny)
    f = KernelAbstractions.allocate(backend, FT, Nx, Ny, 9)
    f_h = zeros(FT, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        f_h[i,j,q] = Kraken.equilibrium(D2Q9(), one(FT), zero(FT), zero(FT), q)
    end
    copyto!(f, f_h)
    txx = KernelAbstractions.zeros(backend, FT, Nx, Ny); fill!(txx, FT(τ_xx_in))
    txy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    tyy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    apply_hermite_source_2d!(f, is_solid, FT(s), txx, txy, tyy)

    cxv = [0, 1, 0, -1,  0, 1, -1, -1,  1]
    cyv = [0, 0, 1,  0, -1, 1,  1, -1, -1]
    fh = Array(f)
    Π_xx = sum(cxv[q]^2 * fh[2, 2, q] for q in 1:9)
    Π_yy = sum(cyv[q]^2 * fh[2, 2, q] for q in 1:9)
    Π_xx_neq = Π_xx - 1/3
    Π_yy_neq = Π_yy - 1/3
    σ_p_xx = (1 - s/2) * Π_xx_neq / s
    σ_p_yy = (1 - s/2) * Π_yy_neq / s
    @printf("2D: σ_p_xx = %.6e (ratio %.4f)   σ_p_yy = %.6e (ratio %.4f)\n",
            σ_p_xx, σ_p_xx/τ_xx_in, σ_p_yy, σ_p_yy/τ_xx_in)
end

let
    Nx, Ny, Nz = 4, 4, 4
    is_solid = KernelAbstractions.zeros(backend, Bool, Nx, Ny, Nz)
    f = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz, 19)
    f_h = zeros(FT, Nx, Ny, Nz, 19)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx, q in 1:19
        f_h[i,j,k,q] = Kraken.equilibrium(D3Q19(), one(FT), zero(FT), zero(FT), zero(FT), q)
    end
    copyto!(f, f_h)
    txx = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz); fill!(txx, FT(τ_xx_in))
    txy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    txz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    tyy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    tyz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    tzz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
    apply_hermite_source_3d!(f, is_solid, FT(s), txx, txy, txz, tyy, tyz, tzz)

    cxv = [0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0]
    cyv = [0,  0,  0,  1, -1,  0,  0,  1,  1, -1, -1,  0,  0,  0,  0,  1, -1,  1, -1]
    czv = [0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1,  1, -1, -1,  1,  1, -1, -1]
    fh = Array(f)
    Π_xx = sum(cxv[q]^2 * fh[2,2,2,q] for q in 1:19)
    Π_yy = sum(cyv[q]^2 * fh[2,2,2,q] for q in 1:19)
    Π_zz = sum(czv[q]^2 * fh[2,2,2,q] for q in 1:19)
    Π_xx_neq = Π_xx - 1/3
    Π_yy_neq = Π_yy - 1/3
    Π_zz_neq = Π_zz - 1/3
    σ_p_xx = (1 - s/2) * Π_xx_neq / s
    σ_p_yy = (1 - s/2) * Π_yy_neq / s
    σ_p_zz = (1 - s/2) * Π_zz_neq / s
    @printf("3D: σ_p_xx = %.6e (ratio %.4f)   σ_p_yy = %.6e (ratio %.4f)   σ_p_zz = %.6e (ratio %.4f)\n",
            σ_p_xx, σ_p_xx/τ_xx_in, σ_p_yy, σ_p_yy/τ_xx_in, σ_p_zz, σ_p_zz/τ_xx_in)
end

println("\nDone.")
