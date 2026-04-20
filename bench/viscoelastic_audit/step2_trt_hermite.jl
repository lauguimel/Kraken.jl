# Step 2 — TRT (solvent) + Guo body force + Hermite source, HWBB + x-periodic.
#
# Same setup as step 1 but the solvent collision uses TRT instead of BGK.
# This is closer to the production pipeline (fused_trt_libb_v2) minus the
# LI-BB cut-link handling (flat walls → q_wall identically zero).
#
# Pass criterion: bulk order should match step 1 or improve.
# Discrimination: if step 2 bulk order drops, TRT implementation is the
# culprit.

using Kraken, Printf, KernelAbstractions
include("common.jl")
using .ViscoAudit

backend = KernelAbstractions.CPU()
FT = Float64

# --- Physical targets (same as step 1) ---
ν_total = 0.1; β = 0.59; u_max = 0.02; Wi = 0.1
ν_s = β * ν_total; ν_p = (1 - β) * ν_total
Λ_magic = 3/16                         # TRT magic number (BB error viscosity-independent)

Ny_list = [30, 60, 120, 240]
u_err = Float64[]; Cxy_err = Float64[]; N1_err = Float64[]; times = Float64[]
u_err_b = Float64[]; Cxy_err_b = Float64[]; N1_err_b = Float64[]

println("="^78)
println("Step 2 — TRT (solvent) + Guo + Hermite, HWBB walls, x-periodic")
println("ν_total=$ν_total  β=$β  u_max=$u_max  Wi=$Wi  Λ=$Λ_magic")
println("="^78)

# TRT collision + Guo body force + Hermite source on f_out, in place.
# Reads moments, writes collided populations. `f_pulled` is the post-
# stream distribution at (i,j) for all q.
function trt_guo_hermite_step!(f_out, f_pulled, is_solid,
                                 ρ, ux_arr, uy_arr,
                                 Fx, Fy, τxx, τxy, τyy,
                                 s_plus, s_minus, Nx, Ny)
    # Hermite pre-factor: matches collide_viscoelastic_source_guo_2d!
    # (BGK+Guo fused) which uses -ω·9/2 WITHOUT the (1-ω/2) division.
    # That factor is absorbed into the Guo (1-ω/2)·S prefactor for the
    # body force AND into the standard CE recovery — when the source is
    # injected DURING collision (not post-hoc), -ω·9/2 gives σ_source = -τ.
    pre = -s_plus * 9.0/2.0
    cs2 = 1.0/3.0
    wr = 4/9; wa = 1/9; we = 1/36
    a = (s_plus + s_minus) * 0.5
    b = (s_plus - s_minus) * 0.5

    @inbounds for j in 1:Ny, i in 1:Nx
        f1 = f_pulled[i,j,1]; f2 = f_pulled[i,j,2]; f3 = f_pulled[i,j,3]
        f4 = f_pulled[i,j,4]; f5 = f_pulled[i,j,5]; f6 = f_pulled[i,j,6]
        f7 = f_pulled[i,j,7]; f8 = f_pulled[i,j,8]; f9 = f_pulled[i,j,9]

        if is_solid[i,j]
            # Bounce-back on solid (swap opposites)
            f_out[i,j,1] = f1
            f_out[i,j,2] = f4; f_out[i,j,4] = f2
            f_out[i,j,3] = f5; f_out[i,j,5] = f3
            f_out[i,j,6] = f8; f_out[i,j,8] = f6
            f_out[i,j,7] = f9; f_out[i,j,9] = f7
            continue
        end

        ρl = f1+f2+f3+f4+f5+f6+f7+f8+f9
        inv_ρ = 1.0/ρl
        # Guo-corrected velocities
        ux = ((f2-f4+f6-f7-f8+f9) + Fx*0.5) * inv_ρ
        uy = ((f3-f5+f6+f7-f8-f9) + Fy*0.5) * inv_ρ
        usq = ux*ux + uy*uy
        ρ[i,j] = ρl; ux_arr[i,j] = ux; uy_arr[i,j] = uy

        feq1 = Kraken.equilibrium(D2Q9(), ρl, ux, uy, 1)
        feq2 = Kraken.equilibrium(D2Q9(), ρl, ux, uy, 2)
        feq3 = Kraken.equilibrium(D2Q9(), ρl, ux, uy, 3)
        feq4 = Kraken.equilibrium(D2Q9(), ρl, ux, uy, 4)
        feq5 = Kraken.equilibrium(D2Q9(), ρl, ux, uy, 5)
        feq6 = Kraken.equilibrium(D2Q9(), ρl, ux, uy, 6)
        feq7 = Kraken.equilibrium(D2Q9(), ρl, ux, uy, 7)
        feq8 = Kraken.equilibrium(D2Q9(), ρl, ux, uy, 8)
        feq9 = Kraken.equilibrium(D2Q9(), ρl, ux, uy, 9)

        txx = τxx[i,j]; txy = τxy[i,j]; tyy = τyy[i,j]
        # Hermite τ_p source (per-q, symmetric pairs share value)
        T1 = pre * wr * (-cs2*(txx + tyy))
        T2 = pre * wa * ((1-cs2)*txx - cs2*tyy);   T4 = T2
        T3 = pre * wa * (-cs2*txx + (1-cs2)*tyy);  T5 = T3
        T6 = pre * we * ((1-cs2)*txx + (1-cs2)*tyy + 2*txy); T8 = T6
        T7 = pre * we * ((1-cs2)*txx + (1-cs2)*tyy - 2*txy); T9 = T7

        # Guo body force source (standard D2Q9 — cf. Guo 2002 Eq. 19)
        # For TRT: body force is in the ANTISYMMETRIC (momentum) mode,
        # so the correction factor uses s_minus (not s_plus).
        gpref = 1.0 - s_minus*0.5
        t3 = 3.0; t9 = 9.0
        S1 = wr * ((-ux)*Fx + (-uy)*Fy) * t3
        S2 = wa * ((1-ux)*Fx + (-uy)*Fy) * t3 + wa * ux*Fx * t9
        S3 = wa * ((-ux)*Fx + (1-uy)*Fy) * t3 + wa * uy*Fy * t9
        S4 = wa * ((-1-ux)*Fx + (-uy)*Fy) * t3 + wa * ux*Fx * t9
        S5 = wa * ((-ux)*Fx + (-1-uy)*Fy) * t3 + wa * uy*Fy * t9
        S6 = we * ((1-ux)*Fx + (1-uy)*Fy) * t3 + we * (ux+uy)*(Fx+Fy) * t9
        S7 = we * ((-1-ux)*Fx + (1-uy)*Fy) * t3 + we * (-ux+uy)*(-Fx+Fy) * t9
        S8 = we * ((-1-ux)*Fx + (-1-uy)*Fy) * t3 + we * (-ux-uy)*(-Fx-Fy) * t9
        S9 = we * ((1-ux)*Fx + (-1-uy)*Fy) * t3 + we * (ux-uy)*(Fx-Fy) * t9

        # TRT relaxation with a, b (BGK recovered at s_plus = s_minus)
        f_out[i,j,1] = f1 - s_plus*(f1 - feq1) + gpref*S1 + T1
        f_out[i,j,2] = f2 - a*(f2 - feq2) - b*(f4 - feq4) + gpref*S2 + T2
        f_out[i,j,4] = f4 - a*(f4 - feq4) - b*(f2 - feq2) + gpref*S4 + T4
        f_out[i,j,3] = f3 - a*(f3 - feq3) - b*(f5 - feq5) + gpref*S3 + T3
        f_out[i,j,5] = f5 - a*(f5 - feq5) - b*(f3 - feq3) + gpref*S5 + T5
        f_out[i,j,6] = f6 - a*(f6 - feq6) - b*(f8 - feq8) + gpref*S6 + T6
        f_out[i,j,8] = f8 - a*(f8 - feq8) - b*(f6 - feq6) + gpref*S8 + T8
        f_out[i,j,7] = f7 - a*(f7 - feq7) - b*(f9 - feq9) + gpref*S7 + T7
        f_out[i,j,9] = f9 - a*(f9 - feq9) - b*(f7 - feq7) + gpref*S9 + T9
    end
end

for Ny in Ny_list
    Nx = 16
    H = Float64(Ny)
    Fx_val = 8 * ν_total * u_max / H^2
    λ = Wi * H / (4 * u_max)
    # Debug mode: set s_plus=s_minus=ω_s → BGK equivalent. Should match step 1.
    debug_bgk = get(ENV, "AUDIT_DEBUG_BGK", "0") == "1"
    ω_s = 1.0 / (3*ν_s + 0.5)
    s_plus, s_minus = debug_bgk ? (ω_s, ω_s) : trt_rates(ν_s; Λ=Λ_magic)
    tau_plus = 1.0; G = ν_p / λ
    max_steps = 30_000

    @printf("\n[Ny=%d]  Nx=%d  Fx=%.3e  λ=%.2f  s+=%.4f s-=%.4f  max_steps=%d\n",
            Ny, Nx, Fx_val, λ, s_plus, s_minus, max_steps)

    ref = ViscoAudit.poiseuille_ref(Ny, Fx_val, ν_total, ν_p, λ)

    f_pre = zeros(FT, Nx, Ny, 9); f_out = zeros(FT, Nx, Ny, 9)
    is_solid = falses(Nx, Ny)
    ρ  = ones(FT, Nx, Ny); ux = zeros(FT, Nx, Ny); uy = zeros(FT, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        ux[i,j] = ref.u[j]
    end
    for j in 1:Ny, i in 1:Nx, q in 1:9
        f_pre[i,j,q] = Kraken.equilibrium(D2Q9(), 1.0, ref.u[j], 0.0, q)
    end

    C_xx = [1.0 + 2.0*(λ*ref.γ̇[j])^2 for i in 1:Nx, j in 1:Ny]
    C_xy = [λ*ref.γ̇[j] for i in 1:Nx, j in 1:Ny]
    C_yy = ones(FT, Nx, Ny)
    g_xx = zeros(FT, Nx, Ny, 9); g_xy = zeros(FT, Nx, Ny, 9); g_yy = zeros(FT, Nx, Ny, 9)
    init_conformation_field_2d!(g_xx, C_xx, ux, uy)
    init_conformation_field_2d!(g_xy, C_xy, ux, uy)
    init_conformation_field_2d!(g_yy, C_yy, ux, uy)
    g_xx_buf = similar(g_xx); g_xy_buf = similar(g_xy); g_yy_buf = similar(g_yy)

    τ_p_xx = G .* (C_xx .- 1.0); τ_p_xy = G .* C_xy; τ_p_yy = G .* (C_yy .- 1.0)
    f_pulled = similar(f_pre)

    t0 = time()
    for step in 1:max_steps
        stream_periodic_x_wall_y_2d!(f_pulled, f_pre, Nx, Ny)
        trt_guo_hermite_step!(f_out, f_pulled, is_solid, ρ, ux, uy,
                              Fx_val, 0.0, τ_p_xx, τ_p_xy, τ_p_yy,
                              s_plus, s_minus, Nx, Ny)
        f_pre, f_out = f_out, f_pre

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

ViscoAudit.print_convergence("Step 2 ALL — TRT+Guo+Hermite (canal) — Linf on j∈[2,Ny-1]",
    Ny_list, Dict("u"=>u_err, "Cxy"=>Cxy_err, "N1"=>N1_err))
ViscoAudit.print_convergence("Step 2 BULK — TRT+Guo+Hermite (canal) — Linf on j∈[10%H,90%H]",
    Ny_list, Dict("u_b"=>u_err_b, "Cxy_b"=>Cxy_err_b, "N1_b"=>N1_err_b))

@printf("Total time: %.0fs\n", sum(times))
