# Step 1b — diagnostic : dump Cxy(j) and N1(j) profiles at Ny=240 to see
# where the wall bias lives. Same setup as step 1, single Ny, output as
# columnar text for plotting.

using Kraken, Printf, KernelAbstractions
include("common.jl")
using .ViscoAudit

backend = KernelAbstractions.CPU()
FT = Float64

ν_total = 0.1; β = 0.59; u_max = 0.02; Wi = 0.1
ν_s = β * ν_total; ν_p = (1 - β) * ν_total
Ny = 240; Nx = 16
H = Float64(Ny)
Fx_val = 8 * ν_total * u_max / H^2
λ = Wi * H / (4 * u_max)
ω_s = 1.0 / (3 * ν_s + 0.5); tau_plus = 1.0; G = ν_p / λ
max_steps = 30_000

@printf("Ny=%d  Fx=%.3e  λ=%.2f  ω_s=%.4f\n", Ny, Fx_val, λ, ω_s)

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

let f=f, f_buf=f_buf, g_xx=g_xx, g_xy=g_xy, g_yy=g_yy,
    g_xx_buf=g_xx_buf, g_xy_buf=g_xy_buf, g_yy_buf=g_yy_buf
global C_xx_out, C_xy_out, C_yy_out, ux_out, τ_p_xx_out, τ_p_xy_out, τ_p_yy_out
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
    compute_conformation_macro_2d!(C_xx, g_xx)
    compute_conformation_macro_2d!(C_xy, g_xy)
    compute_conformation_macro_2d!(C_yy, g_yy)
    collide_conformation_2d!(g_xx, C_xx, ux, uy, C_xx, C_xy, C_yy, is_solid, tau_plus, λ; component=1)
    collide_conformation_2d!(g_xy, C_xy, ux, uy, C_xx, C_xy, C_yy, is_solid, tau_plus, λ; component=2)
    collide_conformation_2d!(g_yy, C_yy, ux, uy, C_xx, C_xy, C_yy, is_solid, tau_plus, λ; component=3)
    @. τ_p_xx = G * (C_xx - 1.0); @. τ_p_xy = G * C_xy; @. τ_p_yy = G * (C_yy - 1.0)
end
end

# --- Dump profile at i = Nx÷2 ---
ic = Nx ÷ 2
println("\n# j  y/H  u_num        u_ana        Cxy_num      Cxy_ana      N1_num       N1_ana       Cxy_err%     N1_err%")
println("#", "-"^130)
for j in 1:Ny
    u_num_j = ux[ic, j]; u_ana_j = ref.u[j]
    Cxy_num = C_xy[ic, j]; Cxy_ana = ref.Cxy[j]
    N1_num = τ_p_xx[ic, j] - τ_p_yy[ic, j]; N1_ana = ref.N1[j]
    Cxy_err = Cxy_ana == 0 ? 0.0 : (Cxy_num - Cxy_ana) / Cxy_ana * 100
    N1_err = N1_ana == 0 ? 0.0 : (N1_num - N1_ana) / N1_ana * 100
    @printf("%4d %.4f  %.4e %.4e %.4e %.4e %.4e %.4e %+8.3f  %+8.3f\n",
            j, (j-0.5)/H, u_num_j, u_ana_j, Cxy_num, Cxy_ana,
            N1_num, N1_ana, Cxy_err, N1_err)
end
