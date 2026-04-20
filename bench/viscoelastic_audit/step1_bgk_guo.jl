# Step 1 — BGK + Guo + Hermite source, HWBB + x-periodic, flat channel.
#
# Baseline pipeline (reproduces test_conformation_lbm.jl:138 extended to
# multiple Ny). Reference: ../../REFERENCES.md (analytic Poiseuille OB).
#
# Wi = 0.1 held constant (λ scaled with Ny). ν_total fixed, u_max fixed.
# Initialization from analytic u/C fields → no transient waiting needed.

using Kraken, Printf, KernelAbstractions
include("common.jl")
using .ViscoAudit

backend = KernelAbstractions.CPU()
FT = Float64

# --- Physical targets (held constant across Ny) ---
ν_total = 0.1
β = 0.59
u_max = 0.02
Wi = 0.1
ν_s = β * ν_total
ν_p = (1 - β) * ν_total

Ny_list = [30, 60, 120, 240]
u_err = Float64[]; Cxy_err = Float64[]; N1_err = Float64[]; times = Float64[]
u_err_b = Float64[]; Cxy_err_b = Float64[]; N1_err_b = Float64[]

println("="^78)
println("Step 1 — BGK + Guo + Hermite source, HWBB walls, x-periodic")
println("ν_total=$ν_total  β=$β  u_max=$u_max  Wi=$Wi")
println("="^78)

for Ny in Ny_list
    Nx = 16                    # thin strip; uniform in x under periodicity
    H = Float64(Ny)
    Fx_val = 8 * ν_total * u_max / H^2
    λ = Wi * H / (4 * u_max)
    ω_s = 1.0 / (3 * ν_s + 0.5)
    tau_plus = 1.0
    G = ν_p / λ
    # Init is at analytic → only polymer/f relaxation needed: ~ 10·λ steps.
    max_steps = 30_000

    @printf("\n[Ny=%d]  Nx=%d  Fx=%.3e  λ=%.2f  ω_s=%.4f  max_steps=%d\n",
            Ny, Nx, Fx_val, λ, ω_s, max_steps)

    # --- Analytic reference ---
    ref = ViscoAudit.poiseuille_ref(Ny, Fx_val, ν_total, ν_p, λ)

    # --- Allocate fields (CPU arrays for reproducibility) ---
    f = zeros(FT, Nx, Ny, 9)
    is_solid = falses(Nx, Ny)
    ρ  = ones(FT, Nx, Ny)
    ux = zeros(FT, Nx, Ny); uy = zeros(FT, Nx, Ny)
    # Initialize at analytic Poiseuille (no transient wait)
    for j in 1:Ny, i in 1:Nx
        ux[i, j] = ref.u[j]
    end
    for j in 1:Ny, i in 1:Nx, q in 1:9
        f[i, j, q] = Kraken.equilibrium(D2Q9(), 1.0, ref.u[j], 0.0, q)
    end
    f_buf = similar(f)

    C_xx = [1.0 + 2.0 * (λ * ref.γ̇[j])^2 for i in 1:Nx, j in 1:Ny]
    C_xy = [λ * ref.γ̇[j]                  for i in 1:Nx, j in 1:Ny]
    C_yy = ones(FT, Nx, Ny)

    g_xx = zeros(FT, Nx, Ny, 9)
    g_xy = zeros(FT, Nx, Ny, 9)
    g_yy = zeros(FT, Nx, Ny, 9)
    init_conformation_field_2d!(g_xx, C_xx, ux, uy)
    init_conformation_field_2d!(g_xy, C_xy, ux, uy)
    init_conformation_field_2d!(g_yy, C_yy, ux, uy)
    g_xx_buf = similar(g_xx); g_xy_buf = similar(g_xy); g_yy_buf = similar(g_yy)

    τ_p_xx = G .* (C_xx .- 1.0)
    τ_p_xy = G .* C_xy
    τ_p_yy = G .* (C_yy .- 1.0)

    t0 = time()
    for step in 1:max_steps
        # Solvent: stream + BGK(Guo) + Hermite (fused kernel)
        stream_periodic_x_wall_y_2d!(f_buf, f, Nx, Ny)
        collide_viscoelastic_source_guo_2d!(f_buf, is_solid, ω_s,
                                              Fx_val, 0.0,
                                              τ_p_xx, τ_p_xy, τ_p_yy)
        f, f_buf = f_buf, f
        compute_macroscopic_2d!(ρ, ux, uy, f)

        # Conformation: stream + TRT relax
        stream_periodic_x_wall_y_2d!(g_xx_buf, g_xx, Nx, Ny)
        stream_periodic_x_wall_y_2d!(g_xy_buf, g_xy, Nx, Ny)
        stream_periodic_x_wall_y_2d!(g_yy_buf, g_yy, Nx, Ny)
        g_xx, g_xx_buf = g_xx_buf, g_xx
        g_xy, g_xy_buf = g_xy_buf, g_xy
        g_yy, g_yy_buf = g_yy_buf, g_yy

        compute_conformation_macro_2d!(C_xx, g_xx)
        compute_conformation_macro_2d!(C_xy, g_xy)
        compute_conformation_macro_2d!(C_yy, g_yy)

        collide_conformation_2d!(g_xx, C_xx, ux, uy, C_xx, C_xy, C_yy,
                                  is_solid, tau_plus, λ; component=1)
        collide_conformation_2d!(g_xy, C_xy, ux, uy, C_xx, C_xy, C_yy,
                                  is_solid, tau_plus, λ; component=2)
        collide_conformation_2d!(g_yy, C_yy, ux, uy, C_xx, C_xy, C_yy,
                                  is_solid, tau_plus, λ; component=3)

        @. τ_p_xx = G * (C_xx - 1.0)
        @. τ_p_xy = G * C_xy
        @. τ_p_yy = G * (C_yy - 1.0)
    end
    dt = time() - t0

    # --- Error norms: two slices — ALL (2..Ny-1) and BULK (10%..90% of H) ---
    ic = Nx ÷ 2
    j_bulk_lo = max(2, round(Int, 0.1 * Ny))
    j_bulk_hi = min(Ny-1, round(Int, 0.9 * Ny))

    function rel_err(num_vec, ref_vec)
        maximum(abs, num_vec .- ref_vec) / maximum(abs, ref_vec)
    end

    u_all  = [ux[ic, j]    for j in 2:Ny-1]
    Cxy_all = [C_xy[ic, j] for j in 2:Ny-1]
    N1_all  = [τ_p_xx[ic,j] - τ_p_yy[ic,j] for j in 2:Ny-1]
    u_bulk  = [ux[ic, j]    for j in j_bulk_lo:j_bulk_hi]
    Cxy_bulk = [C_xy[ic, j] for j in j_bulk_lo:j_bulk_hi]
    N1_bulk  = [τ_p_xx[ic,j] - τ_p_yy[ic,j] for j in j_bulk_lo:j_bulk_hi]

    eu   = rel_err(u_all,   ref.u[2:Ny-1])
    ecxy = rel_err(Cxy_all, ref.Cxy[2:Ny-1])
    en1  = rel_err(N1_all,  ref.N1[2:Ny-1])
    eu_b   = rel_err(u_bulk,   ref.u[j_bulk_lo:j_bulk_hi])
    ecxy_b = rel_err(Cxy_bulk, ref.Cxy[j_bulk_lo:j_bulk_hi])
    en1_b  = rel_err(N1_bulk,  ref.N1[j_bulk_lo:j_bulk_hi])

    push!(u_err, eu); push!(Cxy_err, ecxy); push!(N1_err, en1); push!(times, dt)
    push!(u_err_b, eu_b); push!(Cxy_err_b, ecxy_b); push!(N1_err_b, en1_b)

    @printf("   ALL  err_u=%.3e err_Cxy=%.3e err_N1=%.3e\n", eu, ecxy, en1)
    @printf("   BULK err_u=%.3e err_Cxy=%.3e err_N1=%.3e  time=%.0fs\n",
            eu_b, ecxy_b, en1_b, dt)
end

ViscoAudit.print_convergence(
    "Step 1 ALL — BGK+Guo+Hermite (canal) — Linf relative error on j∈[2,Ny-1]",
    Ny_list,
    Dict("u" => u_err, "Cxy" => Cxy_err, "N1" => N1_err))

ViscoAudit.print_convergence(
    "Step 1 BULK — BGK+Guo+Hermite (canal) — Linf on j∈[10%H, 90%H] (wall excluded)",
    Ny_list,
    Dict("u_b" => u_err_b, "Cxy_b" => Cxy_err_b, "N1_b" => N1_err_b))

@printf("Total time: %.0fs\n", sum(times))
