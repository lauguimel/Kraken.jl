# Convergence study at CONSTANT Wi — proper LBM physical-problem refinement.
#
# Physical problem: fixed u_max, fixed H_phys, fixed Wi (λ·u_max/H_phys)
# Mesh refinement: Δx = H_phys/Ny, Δt chosen for convective scaling.
#
# Practical LBM implementation with unit-lattice-spacing convention:
#   - u_max_lat = const (say 0.02, Mach-safe at all Ny)
#   - ν_lat = ν_total·H_lat/Re  (fixed Re; if Re fixed, ν_lat ∝ H_lat = Ny)
#   - λ_lat = Wi·H_lat/u_max = Wi·Ny/u_max  (λ scales ∝ Ny)
#   - Fx_lat = 2·ν_lat·u_max/(H/2)²  (body force for Poiseuille)
#
# With ν_lat ∝ Ny, ω_s = 1/(3ν+0.5) varies across Ny. ω → 0 as Ny → ∞.
# This is unavoidable in "acoustic" scaling. Alternative: keep ω fixed,
# but then physical time scales nontrivially. Tradeoff accepted for now.
#
# If N1_err/Cxy_err converge to 0 (order ≥ 1) → the previous saturation
# was from varying Wi, scheme is fine.
# If N1_err/Cxy_err saturate even at fixed Wi → real bug in the scheme.

using Kraken, Printf, CUDA, KernelAbstractions
import Kraken: D2Q9, equilibrium,
               stream_periodic_x_wall_y_2d!,
               compute_macroscopic_2d!,
               collide_viscoelastic_source_guo_2d!,
               init_conformation_field_2d!, compute_conformation_macro_2d!,
               collide_conformation_2d!

backend = CUDABackend()
FT = Float64

const Re_target = 1.0
const Wi_target = 0.1
const β = 0.59
const u_max_target = 0.02
const tau_plus = 1.0

println("="^70)
println("Constant-Wi convergence: Re=$Re_target, Wi=$Wi_target, β=$β, u_max=$u_max_target")
println("="^70)

function run_ny(Ny::Int; max_steps=200_000)
    Nx = 4
    H = Float64(Ny)
    # Physical scaling: Re fixed → ν scales with Ny. Wi fixed → λ ∝ Ny.
    ν_total = u_max_target * (H/2) / Re_target
    ν_s = β * ν_total
    ν_p = (1 - β) * ν_total
    λ_visc = Wi_target * (H/2) / u_max_target
    G = ν_p / λ_visc
    ω_s = 1.0 / (3.0 * ν_s + 0.5)
    Fx_val = 2 * ν_total * u_max_target / (H/2)^2

    @printf("  [Ny=%d] ν_s=%.4f ν_p=%.4f λ=%.2f ω_s=%.4f Fx=%.4e\n",
            Ny, ν_s, ν_p, λ_visc, ω_s, Fx_val)

    f_in  = KernelAbstractions.allocate(backend, FT, Nx, Ny, 9)
    f_out = KernelAbstractions.allocate(backend, FT, Nx, Ny, 9)
    is_solid = KernelAbstractions.zeros(backend, Bool, Nx, Ny)
    ux = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    uy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    ρ  = KernelAbstractions.zeros(backend, FT, Nx, Ny); fill!(ρ, FT(1))
    f_h = zeros(FT, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        f_h[i,j,q] = equilibrium(D2Q9(), one(FT), zero(FT), zero(FT), q)
    end
    copyto!(f_in, f_h); copyto!(f_out, f_h)

    Cxx = KernelAbstractions.zeros(backend, FT, Nx, Ny); fill!(Cxx, FT(1))
    Cxy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    Cyy = KernelAbstractions.zeros(backend, FT, Nx, Ny); fill!(Cyy, FT(1))
    g_xx = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9); init_conformation_field_2d!(g_xx, Cxx, ux, uy)
    g_xy = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9); init_conformation_field_2d!(g_xy, Cxy, ux, uy)
    g_yy = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9); init_conformation_field_2d!(g_yy, Cyy, ux, uy)
    g_xx_buf = similar(g_xx); g_xy_buf = similar(g_xy); g_yy_buf = similar(g_yy)

    txx = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    txy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    tyy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    G_FT = FT(G)

    t0 = time()
    for step in 1:max_steps
        stream_periodic_x_wall_y_2d!(f_out, f_in, Nx, Ny)
        collide_viscoelastic_source_guo_2d!(f_out, is_solid, FT(ω_s),
                                              FT(Fx_val), zero(FT),
                                              txx, txy, tyy)
        f_in, f_out = f_out, f_in
        compute_macroscopic_2d!(ρ, ux, uy, f_in)

        stream_periodic_x_wall_y_2d!(g_xx_buf, g_xx, Nx, Ny)
        stream_periodic_x_wall_y_2d!(g_xy_buf, g_xy, Nx, Ny)
        stream_periodic_x_wall_y_2d!(g_yy_buf, g_yy, Nx, Ny)
        g_xx, g_xx_buf = g_xx_buf, g_xx
        g_xy, g_xy_buf = g_xy_buf, g_xy
        g_yy, g_yy_buf = g_yy_buf, g_yy

        compute_conformation_macro_2d!(Cxx, g_xx)
        compute_conformation_macro_2d!(Cxy, g_xy)
        compute_conformation_macro_2d!(Cyy, g_yy)

        collide_conformation_2d!(g_xx, Cxx, ux, uy, Cxx, Cxy, Cyy, is_solid, FT(tau_plus), FT(λ_visc); component=1)
        collide_conformation_2d!(g_xy, Cxy, ux, uy, Cxx, Cxy, Cyy, is_solid, FT(tau_plus), FT(λ_visc); component=2)
        collide_conformation_2d!(g_yy, Cyy, ux, uy, Cxx, Cxy, Cyy, is_solid, FT(tau_plus), FT(λ_visc); component=3)

        @. txx = G_FT * (Cxx - one(FT))
        @. txy = G_FT * Cxy
        @. tyy = G_FT * (Cyy - one(FT))
    end
    elapsed = time() - t0

    ux_ana = [Fx_val / (2 * ν_total) * (j - 0.5) * (H + 0.5 - j) for j in 1:Ny]
    γ̇_ana = [Fx_val / ν_total * (H/2 - (j - 0.5)) for j in 1:Ny]
    Cxy_ana = λ_visc .* γ̇_ana
    N1_ana = 2 .* ν_p .* λ_visc .* γ̇_ana.^2

    ux_h = Array(ux)[2, :]
    Cxy_h = Array(Cxy)[2, :]
    txx_h = Array(txx)[2, :]
    tyy_h = Array(tyy)[2, :]
    N1_num = txx_h .- tyy_h

    u_max_num = maximum(abs.(ux_h))
    Cxy_max_num = maximum(abs.(Cxy_h))
    N1_quart_num = N1_num[Ny ÷ 4]
    N1_quart_ana = N1_ana[Ny ÷ 4]
    u_err   = abs(u_max_num - maximum(abs.(ux_ana))) / maximum(abs.(ux_ana))
    Cxy_err = abs(Cxy_max_num - maximum(abs.(Cxy_ana))) / maximum(abs.(Cxy_ana))
    N1_err  = abs(N1_quart_num - N1_quart_ana) / abs(N1_quart_ana)

    Wi_effective = λ_visc * u_max_target / (H/2)
    @printf("Ny=%-4d  Wi=%.4f  u_err=%.4f  Cxy_err=%.4f  N1_err=%.4f  time=%.0fs\n",
            Ny, Wi_effective, u_err, Cxy_err, N1_err, elapsed)
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
