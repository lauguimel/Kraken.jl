# Convergence study: 2D Poiseuille viscoelastic, EXACT replica of
# test_conformation_lbm.jl Poiseuille setup which already shows:
#   Ny=32:  u_err=5.32%, N1_err=10.7%
#   Ny=64:  u_err=1.63%, N1_err=5.63%
#   Ny=128: u_err=0.28%, N1_err=0.98%   ← demonstrates order ~2
#
# We REPLICATE this here on Metal Float32 to:
# (1) verify our infrastructure reproduces the validated convergence
# (2) check if the convergence rate is uniform (real order 2) or
#     if there's a saturation/floor that suggests a non-discretization bug
# (3) extrapolate to find the resolution needed for <1% on local C_xy
#
# Then we'll know whether the apparent 5-15% deficits we've been
# seeing are TRUE discretization (vanishing as Ny²) or a real bug.

using Kraken, Printf, CUDA, KernelAbstractions
import Kraken: D2Q9, equilibrium,
               stream_periodic_x_wall_y_2d!,
               compute_macroscopic_2d!,
               collide_viscoelastic_source_guo_2d!,
               init_conformation_field_2d!, compute_conformation_macro_2d!,
               collide_conformation_2d!

backend = CUDABackend()
FT = Float64

# Validated test parameters (test_conformation_lbm.jl Poiseuille)
const ν_s = 0.04
const ν_p = 0.06
const ν_total = ν_s + ν_p
const lambda_visc = 50.0
const u_max_target = 0.02      # Mach-safe; Fx scaled per Ny to keep this constant
const max_steps = 200_000
const G = ν_p / lambda_visc
const ω_s = 1.0 / (3.0 * ν_s + 0.5)
const tau_plus = 1.0
# Wi_target = λ·u_max_target/(H/2) — varies with Ny too if H/2 = Ny/2 changes
# To keep Wi constant, also scale λ proportional to Ny... but the validated
# test keeps λ const and lets Wi vary slightly with Ny. We do the same.

println("="^70)
println("Poiseuille viscoelastic convergence study (replica of test_conformation_lbm)")
println("ν_s=$ν_s, ν_p=$ν_p, λ=$lambda_visc, u_max_target=$u_max_target (Fx scaled per Ny)")
println("ω_s = $ω_s   τ_s = $(1/ω_s)")
println("Backend: Metal Float32")
println("="^70)

function run_one(Ny::Int)
    Nx = 4
    H = Float64(Ny)
    n_steps = max_steps
    # Fx scaled to keep u_max_ana = u_max_target
    Fx_val = 2 * ν_total * u_max_target / (H/2)^2
    @printf("  [Ny=%d] Fx=%.4e → expected u_max=%.4f\n", Ny, Fx_val, u_max_target)

    f_in   = KernelAbstractions.allocate(backend, FT, Nx, Ny, 9)
    f_out  = KernelAbstractions.allocate(backend, FT, Nx, Ny, 9)
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

    G_FT = FT(G); λ_FT = FT(lambda_visc); ω_FT = FT(ω_s); τp_FT = FT(tau_plus)

    t0 = time()
    for step in 1:n_steps
        stream_periodic_x_wall_y_2d!(f_out, f_in, Nx, Ny)
        collide_viscoelastic_source_guo_2d!(f_out, is_solid, ω_FT,
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

        collide_conformation_2d!(g_xx, Cxx, ux, uy, Cxx, Cxy, Cyy, is_solid, τp_FT, λ_FT; component=1)
        collide_conformation_2d!(g_xy, Cxy, ux, uy, Cxx, Cxy, Cyy, is_solid, τp_FT, λ_FT; component=2)
        collide_conformation_2d!(g_yy, Cyy, ux, uy, Cxx, Cxy, Cyy, is_solid, τp_FT, λ_FT; component=3)

        @. txx = G_FT * (Cxx - one(FT))
        @. txy = G_FT * Cxy
        @. tyy = G_FT * (Cyy - one(FT))
    end
    elapsed = time() - t0

    # Analytical
    ux_ana  = [Fx_val / (2 * ν_total) * (j - 0.5) * (H + 0.5 - j) for j in 1:Ny]
    γ̇_ana   = [Fx_val / ν_total * (H/2 - (j - 0.5)) for j in 1:Ny]
    Cxy_ana = lambda_visc .* γ̇_ana
    Cxx_ana = 1 .+ 2 .* (lambda_visc .* γ̇_ana).^2

    ux_h = Float64.(Array(ux)[Nx÷2+1, :])
    Cxy_h = Float64.(Array(Cxy)[Nx÷2+1, :])
    Cxx_h = Float64.(Array(Cxx)[Nx÷2+1, :])
    txx_h = Float64.(Array(txx)[Nx÷2+1, :])
    tyy_h = Float64.(Array(tyy)[Nx÷2+1, :])
    N1_num = txx_h .- tyy_h
    N1_ana = 2 .* ν_p .* lambda_visc .* γ̇_ana.^2

    u_max_num = maximum(abs.(ux_h))
    u_max_ana = maximum(abs.(ux_ana))
    Cxy_max_num = maximum(abs.(Cxy_h))
    Cxy_max_ana = maximum(abs.(Cxy_ana))
    N1_max_num = maximum(abs.(N1_num))
    N1_max_ana = maximum(abs.(N1_ana))

    u_err   = abs(u_max_num - u_max_ana) / u_max_ana
    Cxy_err = abs(Cxy_max_num - Cxy_max_ana) / Cxy_max_ana
    N1_err  = abs(N1_max_num - N1_max_ana) / N1_max_ana

    return (; Ny, elapsed, u_err, Cxy_err, N1_err,
            u_max_num, u_max_ana, Cxy_max_num, Cxy_max_ana, N1_max_num, N1_max_ana)
end

@printf("\n%-6s %-10s %-10s %-10s %-12s %-12s %-12s %-8s\n",
        "Ny", "u_max", "Cxy_max", "N1_max", "u_err", "Cxy_err", "N1_err", "time")
@printf("%-6s %-10s %-10s %-10s %-12s %-12s %-12s %-8s\n",
        "--", "------", "-------", "------", "------", "-------", "------", "----")

results = []
for Ny in (30, 60, 120, 240)
    r = run_one(Ny)
    push!(results, r)
    @printf("%-6d %-10.6f %-10.4e %-10.4e %-12.4f %-12.4f %-12.4f %-8.0fs\n",
            r.Ny, r.u_max_num, r.Cxy_max_num, r.N1_max_num,
            r.u_err, r.Cxy_err, r.N1_err, r.elapsed)
end

println("\n--- Convergence rates (order p such that err ~ Ny^(-p)) ---")
@printf("%-12s %-15s %-15s %-15s\n", "step", "u order", "Cxy order", "N1 order")
for k in 2:length(results)
    rk = results[k]; rkm1 = results[k-1]
    p_u   = log(rkm1.u_err   / rk.u_err)   / log(rk.Ny / rkm1.Ny)
    p_Cxy = log(rkm1.Cxy_err / rk.Cxy_err) / log(rk.Ny / rkm1.Ny)
    p_N1  = log(rkm1.N1_err  / rk.N1_err)  / log(rk.Ny / rkm1.Ny)
    @printf("%d→%-3d        p_u=%-10.3f p_Cxy=%-10.3f p_N1=%-10.3f\n",
            rkm1.Ny, rk.Ny, p_u, p_Cxy, p_N1)
end

println("\nDone.")
