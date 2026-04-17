# Systematic bottom-up debug of viscoelastic cylinder benchmark.
# Each test is wrapped in a function to avoid Julia 1.12 scope issues.

using Kraken, Printf, KernelAbstractions, CUDA

const backend = CUDABackend()
const FT = Float64

println("="^70)
println("VISCOELASTIC DEBUG — SYSTEMATIC BOTTOM-UP")
println("GPU: ", CUDA.name(CUDA.device()))
println("="^70)

function test1_hermite_prescribed()
    println("\n>>> TEST 1: Hermite source with prescribed linear τ_p")
    Nx, Ny = 4, 64; ν_s = 0.1; Fx = 1e-5; A = 5e-6
    ω_s = FT(1.0 / (3*ν_s + 0.5))
    H = Float64(Ny)

    f_in  = zeros(FT, Nx, Ny, 9); f_out = zeros(FT, Nx, Ny, 9)
    is_solid = falses(Nx, Ny)
    ux = zeros(FT, Nx, Ny); uy = zeros(FT, Nx, Ny); ρ = ones(FT, Nx, Ny)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        f_in[i,j,q] = Kraken.equilibrium(D2Q9(), 1.0, 0.0, 0.0, q)
    end
    copy!(f_out, f_in)

    tp_xx = zeros(FT, Nx, Ny); tp_xy = zeros(FT, Nx, Ny); tp_yy = zeros(FT, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        tp_xy[i,j] = A * (H/2 - (j - 0.5))
    end

    for step in 1:50_000
        stream_periodic_x_wall_y_2d!(f_out, f_in, Nx, Ny)
        collide_viscoelastic_source_guo_2d!(f_out, is_solid, ω_s, FT(Fx), FT(0), tp_xx, tp_xy, tp_yy)
        f_in, f_out = f_out, f_in
        compute_macroscopic_2d!(ρ, ux, uy, f_in)
    end

    u_ana = [FT(Fx - A) / (2*ν_s) * (j - 0.5) * (H + 0.5 - j) for j in 1:Ny]
    ratio = maximum(ux[2,:]) / maximum(u_ana)
    @printf("  u_max num=%.6f ana=%.6f ratio=%.5f %s\n",
            maximum(ux[2,:]), maximum(u_ana), ratio, abs(ratio-1)<0.01 ? "✓" : "✗")
end

function test2_coupled_poiseuille()
    println("\n>>> TEST 2: Coupled Poiseuille effective viscosity")
    Nx, Ny = 4, 64; ν_s = 0.06; ν_p = 0.04; λ = 50.0
    ν_total = ν_s + ν_p; G = FT(ν_p / λ)
    ω_s = FT(1.0 / (3*ν_s + 0.5)); τp = FT(1.0); Fx = FT(1e-5)

    f_in = zeros(FT, Nx, Ny, 9); f_out = zeros(FT, Nx, Ny, 9)
    sol = falses(Nx, Ny)
    ux = zeros(FT, Nx, Ny); uy = zeros(FT, Nx, Ny); ρ = ones(FT, Nx, Ny)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        f_in[i,j,q] = Kraken.equilibrium(D2Q9(), 1.0, 0.0, 0.0, q)
    end
    copy!(f_out, f_in)

    Cxx = ones(FT, Nx, Ny); Cxy = zeros(FT, Nx, Ny); Cyy = ones(FT, Nx, Ny)
    gxx = zeros(FT, Nx, Ny, 9); gxy = zeros(FT, Nx, Ny, 9); gyy = zeros(FT, Nx, Ny, 9)
    init_conformation_field_2d!(gxx, Cxx, ux, uy)
    init_conformation_field_2d!(gxy, Cxy, ux, uy)
    init_conformation_field_2d!(gyy, Cyy, ux, uy)
    gxx_b = similar(gxx); gxy_b = similar(gxy); gyy_b = similar(gyy)
    txx = zeros(FT, Nx, Ny); txy = zeros(FT, Nx, Ny); tyy = zeros(FT, Nx, Ny)

    for step in 1:100_000
        stream_periodic_x_wall_y_2d!(f_out, f_in, Nx, Ny)
        collide_viscoelastic_source_guo_2d!(f_out, sol, ω_s, Fx, FT(0), txx, txy, tyy)
        f_in, f_out = f_out, f_in
        compute_macroscopic_2d!(ρ, ux, uy, f_in)

        stream_periodic_x_wall_y_2d!(gxx_b, gxx, Nx, Ny)
        stream_periodic_x_wall_y_2d!(gxy_b, gxy, Nx, Ny)
        stream_periodic_x_wall_y_2d!(gyy_b, gyy, Nx, Ny)
        gxx, gxx_b = gxx_b, gxx; gxy, gxy_b = gxy_b, gxy; gyy, gyy_b = gyy_b, gyy
        compute_conformation_macro_2d!(Cxx, gxx)
        compute_conformation_macro_2d!(Cxy, gxy)
        compute_conformation_macro_2d!(Cyy, gyy)
        collide_conformation_2d!(gxx, Cxx, ux, uy, Cxx, Cxy, Cyy, sol, τp, λ; component=1)
        collide_conformation_2d!(gxy, Cxy, ux, uy, Cxx, Cxy, Cyy, sol, τp, λ; component=2)
        collide_conformation_2d!(gyy, Cyy, ux, uy, Cxx, Cxy, Cyy, sol, τp, λ; component=3)
        @. txx = G * (Cxx - 1); @. txy = G * Cxy; @. tyy = G * (Cyy - 1)
    end

    H = Float64(Ny)
    u_ana = [Float64(Fx) / (2*ν_total) * (j - 0.5) * (H + 0.5 - j) for j in 1:Ny]
    ratio_u = maximum(ux[2,:]) / maximum(u_ana)
    jq = Ny ÷ 4
    γ_dot = Float64(Fx) / ν_total * (H/2 - (jq - 0.5))
    ratio_C = Cxy[2, jq] / (λ * γ_dot)
    @printf("  u_max ratio=%.4f %s    C_xy(Ny/4) ratio=%.4f %s\n",
            ratio_u, abs(ratio_u-1)<0.05 ? "✓" : "✗",
            ratio_C, abs(ratio_C-1)<0.1 ? "✓" : "✗")
end

function test3_newtonian_cylinder()
    println("\n>>> TEST 3: Newtonian cylinder LI-BB V2")
    # Liu convention: Re_Liu = U_avg · R / ν, Cd = Fx / (0.5·ρ·U_avg²·D)
    # K = Cd · Re_Liu / 2 = Fx / (ρ · U_avg · ν)
    # We use run_cylinder_libb_2d with parabolic inlet: u_ref = 2/3·u_in.
    for R in [20, 30]
        u_mean = 0.02
        ν = u_mean * R / 1.0  # Re_Liu = U_avg·R/ν = 1.0
        r = run_cylinder_libb_2d(; Nx=30R, Ny=4R, radius=R,
                                    u_in=FT(1.5*u_mean), ν=FT(ν),
                                    max_steps=100_000, avg_window=20_000,
                                    inlet=:parabolic, backend=backend, T=FT)
        Re_liu = r.u_ref * R / ν
        K = r.Cd * Re_liu / 2
        @printf("  R=%d: Cd=%.3f  Re_Liu=%.2f  u_ref=%.4f  K=%.2f\n",
                R, r.Cd, Re_liu, r.u_ref, K)
    end
end

function test4_viscoelastic_cylinder()
    println("\n>>> TEST 4: Viscoelastic cylinder Wi=0.1, R=30")
    # Liu convention: Re_Liu = U_avg · R / ν_total = 1
    # Wi = λ · U_avg / R
    # Cd = Fx / (0.5 ρ U_avg² D)
    R = 30; u_mean = 0.02
    ν_total = u_mean * R / 1.0  # Re_Liu = 1
    ν_s = 0.59 * ν_total; ν_p = 0.41 * ν_total
    Wi = 0.1; λ = Wi * R / u_mean

    @printf("  Setup: R=%d, u_mean=%.3f, ν_total=%.3f, ν_s=%.3f, ν_p=%.3f, λ=%.1f\n",
            R, u_mean, ν_total, ν_s, ν_p, λ)
    Re_liu = u_mean * R / ν_total
    Re_D   = u_mean * 2R / ν_total
    @printf("  Re_Liu(R)=%.2f  Re(D)=%.2f  Wi=%.3f\n", Re_liu, Re_D, Wi)

    for (label, model) in [("direct-C", OldroydB(G=ν_p/λ, λ=λ)),
                            ("log-conf", LogConfOldroydB(G=ν_p/λ, λ=λ))]
        r = run_conformation_cylinder_libb_2d(;
                Nx=30R, Ny=4R, radius=R, cx=15R, cy=2R,
                u_mean=u_mean, ν_s=ν_s,
                polymer_model=model, polymer_bc=CNEBB(),
                inlet=:parabolic, tau_plus=1.0,
                max_steps=200_000, avg_window=40_000,
                backend=backend, FT=FT)

        # The driver reports Re = u_ref·D/ν_total (Re based on D).
        # Liu Cd = Fx/(0.5·ρ·u_mean²·D). Our Cd is already normalized by u_ref=u_mean.
        # K_Liu = Cd · Re_Liu / 2 where Re_Liu = u_mean·R/ν_total.
        K = r.Cd * Re_liu / 2
        err_Cd = (r.Cd - 130.36) / 130.36 * 100
        @printf("  %s: Cd=%.3f  K=%.2f  Cd_p=%.3f  err_Cd=%.2f%%\n",
                label, r.Cd, K, r.Cd_p, err_Cd)

        # C along centerline behind cylinder
        jc = 2R
        print("    C_xx(centerline, x/R=15..25): ")
        for xr in 15:5:25
            ix = xr * R
            ix > size(r.C_xx, 1) && continue
            @printf("%.3f  ", r.C_xx[ix, jc])
        end
        println()
        max_txy = maximum(abs.(r.tau_p_xy))
        @printf("    max|τ_p_xy|=%.6f\n", max_txy)
    end

    # Compare: what is Liu expecting?
    println("  --- Liu Table 3 ref (R=30, Wi=0.1, CNEBB, Sc=10⁴): Cd ≈ 130.36 ---")
    println("  Note: Liu uses Re_Liu=1 (same as us). But Sc differs: ours ~1.4 vs Liu 10⁴")
end

function test5_halfway_bb()
    println("\n>>> TEST 5: Viscoelastic cylinder Wi=0.1, R=30 — halfway-BB (no LI-BB)")
    R = 30; u_mean = 0.02
    ν_total = u_mean * R / 1.0
    ν_s = 0.59 * ν_total; ν_p = 0.41 * ν_total
    λ = 0.1 * R / u_mean

    r = run_conformation_cylinder_2d(;
            Nx=30R, Ny=4R, radius=R, cx=15R, cy=2R,
            u_in=u_mean, ν_s=ν_s, ν_p=ν_p, lambda=λ,
            tau_plus=1.0, max_steps=200_000, avg_window=40_000,
            backend=backend, FT=FT)
    err = (r.Cd - 130.36) / 130.36 * 100
    @printf("  halfway-BB: Cd=%.3f  Cd_p=%.3f  err=%.2f%%\n", r.Cd, r.Cd_p, err)
end

function test6_inlet_profile()
    println("\n>>> TEST 6: Poiseuille inlet u_mean verification")
    Ny = 120; u_mean = 0.02; u_max = 1.5 * u_mean
    u_prof = [FT(4) * u_max * FT(j-1) * FT(Ny-j) / FT(Ny-1)^2 for j in 1:Ny]
    u_mean_c = sum(u_prof) / Ny
    ratio = u_mean_c / u_mean
    @printf("  u_mean target=%.4f computed=%.6f ratio=%.5f %s\n",
            u_mean, u_mean_c, ratio, abs(ratio-1)<0.02 ? "✓" : "✗ WRONG PROFILE")
end

test1_hermite_prescribed()
test2_coupled_poiseuille()
test3_newtonian_cylinder()
test4_viscoelastic_cylinder()
test5_halfway_bb()
test6_inlet_profile()

println("\n", "="^70)
println("DEBUG COMPLETE")
println("="^70)
