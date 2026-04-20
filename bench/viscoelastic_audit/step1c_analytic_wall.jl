# Step 1c — diagnostic : reset g to g^eq(C_analytic) at the wall-adjacent
# rows (j=1, j=Ny) at every step. Uses knowledge of the analytic solution,
# so NOT a production-ready BC, but it answers the question:
#
#   "If we had the correct BC on g, would the bulk scheme give O(2) ?"
#
# If yes → HWBB-on-g is confirmed as the single convergence-order killer.
# If no → there are other bias sources (collision term, source prefactor).

using Kraken, Printf, KernelAbstractions
include("common.jl")
using .ViscoAudit

backend = KernelAbstractions.CPU()
FT = Float64

ν_total = 0.1; β = 0.59; u_max = 0.02; Wi = 0.1
ν_s = β * ν_total; ν_p = (1 - β) * ν_total

Ny_list = [30, 60, 120, 240]
u_err = Float64[]; Cxy_err = Float64[]; N1_err = Float64[]
u_err_b = Float64[]; Cxy_err_b = Float64[]; N1_err_b = Float64[]; times = Float64[]

println("="^78)
println("Step 1c — BGK+Guo+Hermite + analytic g-reset at j=1, Ny (diag)")
println("="^78)

for Ny in Ny_list
    Nx = 16
    H = Float64(Ny)
    Fx_val = 8 * ν_total * u_max / H^2
    λ = Wi * H / (4 * u_max)
    ω_s = 1.0 / (3 * ν_s + 0.5); tau_plus = 1.0; G = ν_p / λ
    max_steps = 30_000

    @printf("\n[Ny=%d]  Fx=%.3e  λ=%.2f\n", Ny, Fx_val, λ)

    ref = ViscoAudit.poiseuille_ref(Ny, Fx_val, ν_total, ν_p, λ)

    f = zeros(FT, Nx, Ny, 9); is_solid = falses(Nx, Ny)
    ρ  = ones(FT, Nx, Ny); ux = zeros(FT, Nx, Ny); uy = zeros(FT, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        ux[i, j] = ref.u[j]
    end
    for j in 1:Ny, i in 1:Nx, q in 1:9
        f[i, j, q] = Kraken.equilibrium(D2Q9(), 1.0, ref.u[j], 0.0, q)
    end
    f_buf = similar(f)

    C_xx = [1.0 + 2.0 * (λ * ref.γ̇[j])^2 for i in 1:Nx, j in 1:Ny]
    C_xy = [λ * ref.γ̇[j] for i in 1:Nx, j in 1:Ny]
    C_yy = ones(FT, Nx, Ny)
    g_xx = zeros(FT, Nx, Ny, 9); g_xy = zeros(FT, Nx, Ny, 9); g_yy = zeros(FT, Nx, Ny, 9)
    init_conformation_field_2d!(g_xx, C_xx, ux, uy)
    init_conformation_field_2d!(g_xy, C_xy, ux, uy)
    init_conformation_field_2d!(g_yy, C_yy, ux, uy)
    g_xx_buf = similar(g_xx); g_xy_buf = similar(g_xy); g_yy_buf = similar(g_yy)

    τ_p_xx = G .* (C_xx .- 1.0); τ_p_xy = G .* C_xy; τ_p_yy = G .* (C_yy .- 1.0)

    # Pre-compute analytic g^eq at wall rows (u_wall=0 because no-slip)
    function feq_g(C_val, u_x, component_val, q)
        Kraken.equilibrium(D2Q9(), C_val, u_x, 0.0, q)
    end

    t0 = time()
    let f=f, f_buf=f_buf, g_xx=g_xx, g_xy=g_xy, g_yy=g_yy,
        g_xx_buf=g_xx_buf, g_xy_buf=g_xy_buf, g_yy_buf=g_yy_buf
    for step in 1:max_steps
        stream_periodic_x_wall_y_2d!(f_buf, f, Nx, Ny)
        collide_viscoelastic_source_guo_2d!(f_buf, is_solid, ω_s,
                                              Fx_val, 0.0,
                                              τ_p_xx, τ_p_xy, τ_p_yy)
        f, f_buf = f_buf, f
        compute_macroscopic_2d!(ρ, ux, uy, f)

        stream_periodic_x_wall_y_2d!(g_xx_buf, g_xx, Nx, Ny)
        stream_periodic_x_wall_y_2d!(g_xy_buf, g_xy, Nx, Ny)
        stream_periodic_x_wall_y_2d!(g_yy_buf, g_yy, Nx, Ny)
        g_xx, g_xx_buf = g_xx_buf, g_xx
        g_xy, g_xy_buf = g_xy_buf, g_xy
        g_yy, g_yy_buf = g_yy_buf, g_yy

        # --- Analytic reset at j=1 and j=Ny ---
        @inbounds for i in 1:Nx, q in 1:9
            g_xx[i,1,q]  = Kraken.equilibrium(D2Q9(), ref.Cxx[1],  ref.u[1],  0.0, q)
            g_xy[i,1,q]  = Kraken.equilibrium(D2Q9(), ref.Cxy[1],  ref.u[1],  0.0, q)
            g_yy[i,1,q]  = Kraken.equilibrium(D2Q9(), 1.0,          ref.u[1],  0.0, q)
            g_xx[i,Ny,q] = Kraken.equilibrium(D2Q9(), ref.Cxx[Ny], ref.u[Ny], 0.0, q)
            g_xy[i,Ny,q] = Kraken.equilibrium(D2Q9(), ref.Cxy[Ny], ref.u[Ny], 0.0, q)
            g_yy[i,Ny,q] = Kraken.equilibrium(D2Q9(), 1.0,          ref.u[Ny], 0.0, q)
        end

        compute_conformation_macro_2d!(C_xx, g_xx)
        compute_conformation_macro_2d!(C_xy, g_xy)
        compute_conformation_macro_2d!(C_yy, g_yy)
        collide_conformation_2d!(g_xx, C_xx, ux, uy, C_xx, C_xy, C_yy, is_solid, tau_plus, λ; component=1)
        collide_conformation_2d!(g_xy, C_xy, ux, uy, C_xx, C_xy, C_yy, is_solid, tau_plus, λ; component=2)
        collide_conformation_2d!(g_yy, C_yy, ux, uy, C_xx, C_xy, C_yy, is_solid, tau_plus, λ; component=3)
        @. τ_p_xx = G * (C_xx - 1.0); @. τ_p_xy = G * C_xy; @. τ_p_yy = G * (C_yy - 1.0)
    end
    end
    dt = time() - t0

    ic = Nx ÷ 2
    j_bulk_lo = max(2, round(Int, 0.1*Ny)); j_bulk_hi = min(Ny-1, round(Int, 0.9*Ny))
    rel_err(a,b) = maximum(abs, a .- b) / maximum(abs, b)

    eu   = rel_err([ux[ic,j]    for j in 2:Ny-1], ref.u[2:Ny-1])
    ecxy = rel_err([C_xy[ic,j]  for j in 2:Ny-1], ref.Cxy[2:Ny-1])
    en1  = rel_err([τ_p_xx[ic,j]-τ_p_yy[ic,j] for j in 2:Ny-1], ref.N1[2:Ny-1])
    eu_b   = rel_err([ux[ic,j]    for j in j_bulk_lo:j_bulk_hi], ref.u[j_bulk_lo:j_bulk_hi])
    ecxy_b = rel_err([C_xy[ic,j]  for j in j_bulk_lo:j_bulk_hi], ref.Cxy[j_bulk_lo:j_bulk_hi])
    en1_b  = rel_err([τ_p_xx[ic,j]-τ_p_yy[ic,j] for j in j_bulk_lo:j_bulk_hi], ref.N1[j_bulk_lo:j_bulk_hi])

    push!(u_err, eu); push!(Cxy_err, ecxy); push!(N1_err, en1); push!(times, dt)
    push!(u_err_b, eu_b); push!(Cxy_err_b, ecxy_b); push!(N1_err_b, en1_b)

    @printf("   ALL  err_u=%.3e err_Cxy=%.3e err_N1=%.3e\n", eu, ecxy, en1)
    @printf("   BULK err_u=%.3e err_Cxy=%.3e err_N1=%.3e  time=%.0fs\n",
            eu_b, ecxy_b, en1_b, dt)
end

ViscoAudit.print_convergence("Step 1c ALL — analytic g-reset at wall",
    Ny_list, Dict("u"=>u_err, "Cxy"=>Cxy_err, "N1"=>N1_err))
ViscoAudit.print_convergence("Step 1c BULK — analytic g-reset at wall",
    Ny_list, Dict("u_b"=>u_err_b, "Cxy_b"=>Cxy_err_b, "N1_b"=>N1_err_b))
@printf("Total time: %.0fs\n", sum(times))
