# EXACT replica of test_conformation_lbm.jl Poiseuille — Float64 Aqua H100.
# This should reproduce the documented N1_err=5.63% at Ny=64.
# If it doesn't, the documented convergence is stale OR a regression has
# been introduced since that test was certified.

using Kraken, Printf, CUDA, KernelAbstractions
import Kraken: D2Q9, equilibrium,
               stream_periodic_x_wall_y_2d!,
               compute_macroscopic_2d!,
               collide_viscoelastic_source_guo_2d!,
               init_conformation_field_2d!, compute_conformation_macro_2d!,
               collide_conformation_2d!

backend = CUDABackend()
FT = Float64

function run_ny(Ny::Int; max_steps=100_000)
    Nx = 4
    ν_s = 0.04; ν_p = 0.06; ν_total = ν_s + ν_p
    lambda = 50.0
    Fx_val = 1e-5
    G = ν_p / lambda
    ω_s = 1.0 / (3.0 * ν_s + 0.5)
    tau_plus = 1.0

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

        collide_conformation_2d!(g_xx, Cxx, ux, uy, Cxx, Cxy, Cyy, is_solid, FT(tau_plus), FT(lambda); component=1)
        collide_conformation_2d!(g_xy, Cxy, ux, uy, Cxx, Cxy, Cyy, is_solid, FT(tau_plus), FT(lambda); component=2)
        collide_conformation_2d!(g_yy, Cyy, ux, uy, Cxx, Cxy, Cyy, is_solid, FT(tau_plus), FT(lambda); component=3)

        @. txx = G_FT * (Cxx - one(FT))
        @. txy = G_FT * Cxy
        @. tyy = G_FT * (Cyy - one(FT))
    end
    elapsed = time() - t0

    H = Float64(Ny)
    ux_ana = [Fx_val / (2 * ν_total) * (j - 0.5) * (H + 0.5 - j) for j in 1:Ny]
    γ̇_ana = [Fx_val / ν_total * (H/2 - (j - 0.5)) for j in 1:Ny]
    Cxy_ana = lambda .* γ̇_ana
    N1_ana = 2 .* ν_p .* lambda .* γ̇_ana.^2

    ux_h = Array(ux)[2, :]
    Cxy_h = Array(Cxy)[2, :]
    txx_h = Array(txx)[2, :]
    tyy_h = Array(tyy)[2, :]
    N1_num = txx_h .- tyy_h

    u_max_num = maximum(abs.(ux_h)); u_max_ana = maximum(abs.(ux_ana))
    Cxy_max_num = maximum(abs.(Cxy_h)); Cxy_max_ana = maximum(abs.(Cxy_ana))
    N1_quart_num = N1_num[Ny ÷ 4]
    N1_quart_ana = N1_ana[Ny ÷ 4]

    u_err   = abs(u_max_num - u_max_ana) / u_max_ana
    Cxy_err = abs(Cxy_max_num - Cxy_max_ana) / Cxy_max_ana
    N1_err_quart = abs(N1_quart_num - N1_quart_ana) / abs(N1_quart_ana)

    @printf("Ny=%-4d  u_max=%.5f (ana=%.5f, err=%.4f)  Cxy_max=%.4e (ana=%.4e, err=%.4f)  N1@quart=%.4e (ana=%.4e, err=%.4f)  time=%.0fs\n",
            Ny, u_max_num, u_max_ana, u_err,
            Cxy_max_num, Cxy_max_ana, Cxy_err,
            N1_quart_num, N1_quart_ana, N1_err_quart, elapsed)
    return (; Ny, u_err, Cxy_err, N1_err_quart)
end

println("="^70)
println("EXACT replica of test_conformation_lbm.jl Poiseuille")
println("Expected (from test comment at Ny=64): N1_err ≈ 5.63%")
println("="^70)
run_ny(32)
run_ny(64)
run_ny(128)
println("\nIf N1_err @ Ny=64 ≈ 5.6%, scheme is consistent with documented validation.")
println("If N1_err @ Ny=64 > 10%, regression detected.")
