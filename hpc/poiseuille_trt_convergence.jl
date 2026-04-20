# Constant-Wi convergence using the PRODUCTION pipeline:
# TRT + LI-BB V2 + apply_hermite_source_2d! + reset_conformation_* + CNEBB
#
# This is what `run_conformation_cylinder_libb_2d` uses internally (validated
# to 0.32% Cd at R=48 vs Liu Table 3). The previous BGK-based convergence
# test gave only order ~0.5 on Cxy_max — an artefact of BGK halfway-BB wall
# accuracy, NOT of the production pipeline.
#
# Expected with TRT+LIBB: order ~2 on u, Cxy, N1 (Ginzburg-exact at Λ=3/16).
# If we see <1% at Ny=128 → scheme is validated for publication.
# If we still see saturation → real issue in TRT+LIBB + Hermite at fine mesh.

using Kraken, Printf, CUDA, KernelAbstractions
import Kraken: D2Q9, equilibrium,
               fused_trt_libb_v2_step!, apply_bc_rebuild_2d!,
               apply_hermite_source_2d!,
               stream_2d!, compute_macroscopic_2d!,
               init_conformation_field_2d!, compute_conformation_macro_2d!,
               apply_polymer_wall_bc!, collide_conformation_2d!,
               reset_conformation_inlet_2d!, reset_conformation_outlet_2d!,
               BCSpec2D, ZouHeVelocity, ZouHePressure, CNEBB, OldroydB

backend = CUDABackend()
FT = Float64

const Re_target = 1.0
const Wi_target = 0.1
const β = 0.59
const u_max_target = 0.02
const tau_plus = 1.0

println("="^70)
println("TRT + LI-BB V2 constant-Wi convergence (= production pipeline)")
println("Re=$Re_target, Wi=$Wi_target, β=$β, u_max=$u_max_target")
println("="^70)

function run_ny(Ny::Int; max_steps=200_000)
    Nx = 6 * (Ny ÷ 10)  # 6× streamwise for Poiseuille to develop from inlet
    H = Float64(Ny)
    ν_total = u_max_target * (H/2) / Re_target
    ν_s = β * ν_total
    ν_p = (1 - β) * ν_total
    λ_visc = Wi_target * (H/2) / u_max_target
    G = ν_p / λ_visc
    s_plus_s = 1.0 / (3.0 * ν_s + 0.5)
    @printf("  [Ny=%d Nx=%d] ν_s=%.4f λ=%.2f s_plus=%.4f\n",
            Ny, Nx, ν_s, λ_visc, s_plus_s)

    # Allocations
    q_wall = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    is_solid = KernelAbstractions.zeros(backend, Bool, Nx, Ny)
    uw_x = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    uw_y = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    f_in  = KernelAbstractions.allocate(backend, FT, Nx, Ny, 9)
    f_out = KernelAbstractions.allocate(backend, FT, Nx, Ny, 9)
    ρ  = KernelAbstractions.zeros(backend, FT, Nx, Ny); fill!(ρ, FT(1))
    ux = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    uy = KernelAbstractions.zeros(backend, FT, Nx, Ny)

    # Inlet parabolic profile (Schäfer-Turek: u_max at centreline)
    u_prof_h = [4 * u_max_target * (j - 0.5) * (H - (j - 0.5)) / H^2 for j in 1:Ny]
    u_profile = KernelAbstractions.allocate(backend, FT, Ny); copyto!(u_profile, FT.(u_prof_h))
    bcspec = BCSpec2D(; west=ZouHeVelocity(u_profile), east=ZouHePressure(FT(1.0)))

    # Init f at equilibrium with inlet profile (helps convergence)
    f_h = zeros(FT, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        f_h[i,j,q] = equilibrium(D2Q9(), one(FT), FT(u_prof_h[j]), zero(FT), q)
    end
    copyto!(f_in, f_h); fill!(f_out, zero(FT))

    # Conformation fields
    Cxx = KernelAbstractions.zeros(backend, FT, Nx, Ny); fill!(Cxx, FT(1))
    Cxy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Cyy = KernelAbstractions.zeros(backend, FT, Nx, Ny); fill!(Cyy, FT(1))
    g_xx = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9); init_conformation_field_2d!(g_xx, Cxx, ux, uy)
    g_xy = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9); init_conformation_field_2d!(g_xy, Cxy, ux, uy)
    g_yy = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9); init_conformation_field_2d!(g_yy, Cyy, ux, uy)
    g_xx_buf = similar(g_xx); g_xy_buf = similar(g_xy); g_yy_buf = similar(g_yy)

    # Inlet C profile (analytical Oldroyd-B at y-parabolic shear)
    Cxy_in_h = zeros(FT, Ny); Cxx_in_h = ones(FT, Ny); Cyy_in_h = ones(FT, Ny)
    for j in 1:Ny
        y = FT(j) - FT(0.5)
        dudy = 4 * u_max_target * (H - 2*y) / H^2
        Cxy_in_h[j] = FT(λ_visc) * FT(dudy)
        Cxx_in_h[j] = FT(1) + FT(2) * (FT(λ_visc) * FT(dudy))^2
    end
    Cxx_in_d = KernelAbstractions.allocate(backend, FT, Ny); copyto!(Cxx_in_d, Cxx_in_h)
    Cxy_in_d = KernelAbstractions.allocate(backend, FT, Ny); copyto!(Cxy_in_d, Cxy_in_h)
    Cyy_in_d = KernelAbstractions.allocate(backend, FT, Ny); copyto!(Cyy_in_d, Cyy_in_h)

    txx = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    txy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    tyy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    G_FT = FT(G); λ_FT = FT(λ_visc); τp_FT = FT(tau_plus)

    t0 = time()
    for step in 1:max_steps
        # Solvent: TRT + LI-BB V2 + Zou-He + Hermite source
        fused_trt_libb_v2_step!(f_out, f_in, ρ, ux, uy, is_solid,
                                  q_wall, uw_x, uw_y, Nx, Ny, FT(ν_s))
        apply_bc_rebuild_2d!(f_out, f_in, bcspec, ν_s, Nx, Ny)
        apply_hermite_source_2d!(f_out, is_solid, FT(s_plus_s), txx, txy, tyy)
        f_in, f_out = f_out, f_in

        # Conformation: stream + CNEBB + reset + collide
        stream_2d!(g_xx_buf, g_xx, Nx, Ny)
        stream_2d!(g_xy_buf, g_xy, Nx, Ny)
        stream_2d!(g_yy_buf, g_yy, Nx, Ny)
        apply_polymer_wall_bc!(g_xx_buf, g_xx, is_solid, Cxx, CNEBB())
        apply_polymer_wall_bc!(g_xy_buf, g_xy, is_solid, Cxy, CNEBB())
        apply_polymer_wall_bc!(g_yy_buf, g_yy, is_solid, Cyy, CNEBB())
        reset_conformation_inlet_2d!(g_xx_buf, Cxx_in_d, u_profile, Ny)
        reset_conformation_inlet_2d!(g_xy_buf, Cxy_in_d, u_profile, Ny)
        reset_conformation_inlet_2d!(g_yy_buf, Cyy_in_d, u_profile, Ny)
        reset_conformation_outlet_2d!(g_xx_buf, Nx, Ny)
        reset_conformation_outlet_2d!(g_xy_buf, Nx, Ny)
        reset_conformation_outlet_2d!(g_yy_buf, Nx, Ny)
        g_xx, g_xx_buf = g_xx_buf, g_xx
        g_xy, g_xy_buf = g_xy_buf, g_xy
        g_yy, g_yy_buf = g_yy_buf, g_yy

        compute_conformation_macro_2d!(Cxx, g_xx)
        compute_conformation_macro_2d!(Cxy, g_xy)
        compute_conformation_macro_2d!(Cyy, g_yy)

        collide_conformation_2d!(g_xx, Cxx, ux, uy, Cxx, Cxy, Cyy, is_solid, τp_FT, λ_FT; component=1)
        collide_conformation_2d!(g_xy, Cxy, ux, uy, Cxx, Cxy, Cyy, is_solid, τp_FT, λ_FT; component=2)
        collide_conformation_2d!(g_yy, Cyy, ux, uy, Cxx, Cxy, Cyy, is_solid, τp_FT, λ_FT; component=3)

        @. txx = G_FT * (Cxx - one(FT))
        @. txy = G_FT * Cxy
        @. tyy = G_FT * (Cyy - one(FT))
    end
    elapsed = time() - t0

    # Sample profile at downstream i = 3*Nx/4
    i_s = 3 * Nx ÷ 4
    ux_h = Array(ux)[i_s, :]
    Cxy_h = Array(Cxy)[i_s, :]
    txx_h = Array(txx)[i_s, :]
    tyy_h = Array(tyy)[i_s, :]

    ux_ana = [4 * u_max_target * (j - 0.5) * (H - (j - 0.5)) / H^2 for j in 1:Ny]
    γ̇_ana = [4 * u_max_target * (H - 2*(j - 0.5)) / H^2 for j in 1:Ny]
    Cxy_ana = λ_visc .* γ̇_ana
    N1_ana = 2 * ν_p * λ_visc .* γ̇_ana.^2
    N1_num = txx_h .- tyy_h

    u_err   = abs(maximum(abs.(ux_h)) - maximum(abs.(ux_ana))) / maximum(abs.(ux_ana))
    Cxy_err = abs(maximum(abs.(Cxy_h)) - maximum(abs.(Cxy_ana))) / maximum(abs.(Cxy_ana))
    N1_err  = abs(N1_num[Ny÷4] - N1_ana[Ny÷4]) / abs(N1_ana[Ny÷4])

    @printf("Ny=%-4d  u_err=%.4f  Cxy_err=%.4f  N1_err=%.4f  time=%.0fs\n",
            Ny, u_err, Cxy_err, N1_err, elapsed)
    return (; Ny, u_err, Cxy_err, N1_err)
end

results = [run_ny(Ny) for Ny in (30, 60, 120, 240)]

println("\n--- Convergence orders (err ~ Ny^(-p)) ---")
for k in 2:length(results)
    rk = results[k]; rp = results[k-1]
    p_u = log(rp.u_err/rk.u_err) / log(rk.Ny/rp.Ny)
    p_Cxy = log(rp.Cxy_err/rk.Cxy_err) / log(rk.Ny/rp.Ny)
    p_N1 = log(rp.N1_err/rk.N1_err) / log(rk.Ny/rp.Ny)
    @printf("%d→%-3d        p_u=%-8.3f p_Cxy=%-8.3f p_N1=%-8.3f\n",
            rp.Ny, rk.Ny, p_u, p_Cxy, p_N1)
end
println("\nDone.")
