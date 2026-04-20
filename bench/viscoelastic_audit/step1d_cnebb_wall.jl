# Step 1d — ABANDONED : canal with solid-cell walls + CNEBB on g.
#
# Tried on 2026-04-21. Flow didn't develop (oscillates and decays toward 0
# instead of settling at analytic Poiseuille). Root cause : stream_2d!
# applies HWBB at sim borders regardless of `is_solid`, then the collision
# kernel does bounce_back_2d! on solid cells — double BC on f at solid
# rows j=1 and j=Ny.
#
# This is NOT the correct way to test CNEBB on a canal. The production
# pipeline only uses CNEBB on CURVED walls (cylinder, sphere) where the
# geometry naturally matches stream_2d's assumption that solid cells are
# just fluid cells marked as blocked.
#
# Decision : do not pursue canal+CNEBB. The canal result is informational
# (HWBB-on-g is bad) and the cylinder production run 20184519 on Aqua
# will tell us whether CNEBB-on-curved works properly.

using Kraken, Printf, KernelAbstractions
include("common.jl")
using .ViscoAudit

backend = KernelAbstractions.CPU()
FT = Float64

ν_total = 0.1; β = 0.59; u_max = 0.02; Wi = 0.1
ν_s = β * ν_total; ν_p = (1 - β) * ν_total

# Sim domain sizes (Ny_sim = fluid_Ny + 2 solid rows)
Ny_fluid_list = [30, 60, 120, 240]
u_err = Float64[]; Cxy_err = Float64[]; N1_err = Float64[]; times = Float64[]
u_err_b = Float64[]; Cxy_err_b = Float64[]; N1_err_b = Float64[]

println("="^78)
println("Step 1d — BGK+Guo+Hermite, solid-cell walls + CNEBB on g")
println("="^78)

for Ny_fluid in Ny_fluid_list
    Ny_sim = Ny_fluid + 2
    Nx = 16
    H = Float64(Ny_fluid)           # fluid wall-to-wall distance
    Fx_val = 8 * ν_total * u_max / H^2
    λ = Wi * H / (4 * u_max)
    ω_s = 1.0 / (3 * ν_s + 0.5); tau_plus = 1.0; G = ν_p / λ
    max_steps = 30_000

    @printf("\n[Ny_fluid=%d, Ny_sim=%d]  Fx=%.3e  λ=%.2f\n",
            Ny_fluid, Ny_sim, Fx_val, λ)

    # Analytic profile for fluid rows 2..Ny_sim-1 : y = j - 1.5
    ref_fluid = ViscoAudit.poiseuille_ref(Ny_fluid, Fx_val, ν_total, ν_p, λ)

    f = zeros(FT, Nx, Ny_sim, 9); is_solid = falses(Nx, Ny_sim)
    @inbounds for i in 1:Nx
        is_solid[i, 1]     = true
        is_solid[i, Ny_sim]= true
    end

    ρ  = ones(FT, Nx, Ny_sim); ux = zeros(FT, Nx, Ny_sim); uy = zeros(FT, Nx, Ny_sim)
    for j in 2:Ny_sim-1, i in 1:Nx
        ux[i, j] = ref_fluid.u[j-1]
    end
    for j in 1:Ny_sim, i in 1:Nx, q in 1:9
        u_init = is_solid[i,j] ? 0.0 : ref_fluid.u[j-1]
        f[i, j, q] = Kraken.equilibrium(D2Q9(), 1.0, u_init, 0.0, q)
    end
    f_buf = similar(f)

    C_xx = ones(FT, Nx, Ny_sim); C_xy = zeros(FT, Nx, Ny_sim); C_yy = ones(FT, Nx, Ny_sim)
    for j in 2:Ny_sim-1, i in 1:Nx
        jf = j - 1
        C_xx[i, j] = 1.0 + 2.0 * (λ * ref_fluid.γ̇[jf])^2
        C_xy[i, j] = λ * ref_fluid.γ̇[jf]
    end
    g_xx = zeros(FT, Nx, Ny_sim, 9); g_xy = zeros(FT, Nx, Ny_sim, 9); g_yy = zeros(FT, Nx, Ny_sim, 9)
    init_conformation_field_2d!(g_xx, C_xx, ux, uy)
    init_conformation_field_2d!(g_xy, C_xy, ux, uy)
    init_conformation_field_2d!(g_yy, C_yy, ux, uy)
    g_xx_buf = similar(g_xx); g_xy_buf = similar(g_xy); g_yy_buf = similar(g_yy)

    τ_p_xx = G .* (C_xx .- 1.0); τ_p_xy = G .* C_xy; τ_p_yy = G .* (C_yy .- 1.0)

    t0 = time()
    let f=f, f_buf=f_buf, g_xx=g_xx, g_xy=g_xy, g_yy=g_yy,
        g_xx_buf=g_xx_buf, g_xy_buf=g_xy_buf, g_yy_buf=g_yy_buf
    for step in 1:max_steps
        # Solvent: stream_2d (HWBB at all borders) + BGK+Guo+Hermite
        stream_2d!(f_buf, f, Nx, Ny_sim)
        collide_viscoelastic_source_guo_2d!(f_buf, is_solid, ω_s,
                                              Fx_val, 0.0,
                                              τ_p_xx, τ_p_xy, τ_p_yy)
        f, f_buf = f_buf, f
        compute_macroscopic_2d!(ρ, ux, uy, f)

        # Conformation: stream_2d + CNEBB at wall-adjacent cells
        stream_2d!(g_xx_buf, g_xx, Nx, Ny_sim)
        stream_2d!(g_xy_buf, g_xy, Nx, Ny_sim)
        stream_2d!(g_yy_buf, g_yy, Nx, Ny_sim)

        apply_cnebb_conformation_2d!(g_xx_buf, g_xx, is_solid, C_xx)
        apply_cnebb_conformation_2d!(g_xy_buf, g_xy, is_solid, C_xy)
        apply_cnebb_conformation_2d!(g_yy_buf, g_yy, is_solid, C_yy)

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
    dt = time() - t0

    # Errors on fluid rows 2..Ny_sim-1 (mapped to ref index 1..Ny_fluid)
    ic = Nx ÷ 2
    jf_range = 1:Ny_fluid            # fluid rows
    j_bulk_lo = max(1, round(Int, 0.1*Ny_fluid))
    j_bulk_hi = min(Ny_fluid, round(Int, 0.9*Ny_fluid))

    u_num_all = [ux[ic, jf+1]    for jf in jf_range]
    Cxy_num_all = [C_xy[ic, jf+1] for jf in jf_range]
    N1_num_all  = [τ_p_xx[ic, jf+1] - τ_p_yy[ic, jf+1] for jf in jf_range]
    u_num_b = [ux[ic, jf+1]    for jf in j_bulk_lo:j_bulk_hi]
    Cxy_num_b = [C_xy[ic, jf+1] for jf in j_bulk_lo:j_bulk_hi]
    N1_num_b  = [τ_p_xx[ic, jf+1] - τ_p_yy[ic, jf+1] for jf in j_bulk_lo:j_bulk_hi]

    rel_err(a,b) = maximum(abs, a .- b) / maximum(abs, b)
    eu = rel_err(u_num_all, ref_fluid.u); ecxy = rel_err(Cxy_num_all, ref_fluid.Cxy)
    en1 = rel_err(N1_num_all, ref_fluid.N1)
    eu_b = rel_err(u_num_b, ref_fluid.u[j_bulk_lo:j_bulk_hi])
    ecxy_b = rel_err(Cxy_num_b, ref_fluid.Cxy[j_bulk_lo:j_bulk_hi])
    en1_b = rel_err(N1_num_b, ref_fluid.N1[j_bulk_lo:j_bulk_hi])

    push!(u_err, eu); push!(Cxy_err, ecxy); push!(N1_err, en1); push!(times, dt)
    push!(u_err_b, eu_b); push!(Cxy_err_b, ecxy_b); push!(N1_err_b, en1_b)

    @printf("   ALL  err_u=%.3e err_Cxy=%.3e err_N1=%.3e\n", eu, ecxy, en1)
    @printf("   BULK err_u=%.3e err_Cxy=%.3e err_N1=%.3e  time=%.0fs\n",
            eu_b, ecxy_b, en1_b, dt)
end

ViscoAudit.print_convergence("Step 1d ALL — CNEBB wall on canal",
    Ny_fluid_list, Dict("u"=>u_err, "Cxy"=>Cxy_err, "N1"=>N1_err))
ViscoAudit.print_convergence("Step 1d BULK — CNEBB wall on canal",
    Ny_fluid_list, Dict("u_b"=>u_err_b, "Cxy_b"=>Cxy_err_b, "N1_b"=>N1_err_b))
@printf("Total time: %.0fs\n", sum(times))
