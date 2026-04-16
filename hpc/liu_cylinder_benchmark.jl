# Reproduce Liu et al. 2025 (arxiv 2508.16997) cylinder benchmark exactly.
#
# Setup (p.37-39):
#   - Domain 30R × 4R, cylinder at (15R, 2R), B = 0.5
#   - Re = 1 (based on U_c = U_avg = 2/3 U_max, L_c = R)
#   - Ma = 0.01 → U_c = Ma · cs = 0.01/√3 ≈ 0.00577
#   - Inlet: fully developed Poiseuille profile + analytical C (Eq 61-62)
#   - Outlet: zero-gradient (convective)
#   - Walls + cylinder: no-slip (bounce-back)
#   - Cd = Fx / (0.5 ρ U_avg² D)   (Eq 64)
#   - β = 0.59, Sc = 10⁴
#
# Reference values (Table 3, CNEBB, Sc=10⁴):
#   R=30, Wi=0.1 → Cd ≈ 130.36
#   R=30, Wi=0.5 → Cd ≈ 126.31
#   R=30, Wi=1.0 → Cd ≈ 151.31  (note: Cd increases at Wi=1!)
#
# Usage: julia --project=. hpc/liu_cylinder_benchmark.jl

using Kraken, Printf, CUDA, KernelAbstractions

backend = CUDABackend()
FT = Float64

function run_liu_cylinder(; R, Wi, β=0.59, Re=1.0, U_c=0.02,
                            max_steps=500_000, avg_frac=0.2,
                            backend, FT)
    U_c = FT(U_c)                   # characteristic velocity = U_avg
    U_max = FT(1.5 * U_c)          # peak Poiseuille velocity

    D = 2R
    Ny = 4R                        # B = D/(2·Ny/2) = D/Ny = 0.5... wait
    # Actually B = R / (Ny/2) = R/(2R) = 0.5 ✓ (half-channel = 2R)
    Nx = 30R
    cx_cyl = 15R                    # cylinder center x
    cy_cyl = 2R                     # cylinder center y (= Ny/2)

    ν_total = U_c * R / Re          # Re = U_c · R / ν_total (L_c = R)
    ν_s = β * ν_total
    ν_p = ν_total - ν_s
    ω_s = FT(1 / (3 * ν_s + 0.5))

    λ = Wi * R / U_c                # Wi = λ · U_c / R
    G = FT(ν_p / λ)
    tau_plus = FT(1.0)

    # Schmidt number Sc = ν_s / κ, κ = (tau_plus - 0.5)/3
    κ = (tau_plus - 0.5) / 3.0
    Sc = ν_s / κ

    avg_window = round(Int, max_steps * avg_frac)

    Ma = U_c / cs
    @info "Liu cylinder" R Nx Ny Wi Re β Ma U_c U_max ν_total ν_s ν_p λ G Sc ω_s

    # --- Initialize flow field ---
    state, _ = initialize_cylinder_2d(; Nx=Nx, Ny=Ny, cx=cx_cyl, cy=cy_cyl,
                                        radius=R, u_in=U_c, ν=ν_s,
                                        backend=backend, T=FT)
    f_in, f_out = state.f_in, state.f_out
    ρ, ux, uy = state.ρ, state.ux, state.uy
    is_solid = state.is_solid

    # --- Poiseuille inlet profile (Eq 61) ---
    # u_x(y) = 1.5 × 4·U_c·y·(Ny - y) / Ny² = U_max · 4y(Ny-y)/Ny²
    H = FT(Ny)
    u_profile_cpu = zeros(FT, Ny)
    for j in 1:Ny
        y = FT(j) - FT(0.5)   # half-way BB convention
        u_profile_cpu[j] = U_max * FT(4) * y * (H - y) / (H * H)
    end
    u_profile = KernelAbstractions.allocate(backend, FT, Ny)
    copyto!(u_profile, u_profile_cpu)

    # Re-init f to equilibrium with Poiseuille profile
    f_cpu = zeros(FT, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx
        u_loc = u_profile_cpu[j]
        for q in 1:9
            f_cpu[i,j,q] = Kraken.equilibrium(D2Q9(), one(FT), FT(u_loc), zero(FT), q)
        end
    end
    copyto!(f_in, f_cpu); copyto!(f_out, f_cpu)

    # --- Conformation fields ---
    C_xx = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    C_xy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    C_yy = KernelAbstractions.zeros(backend, FT, Nx, Ny)

    # Initialize C with analytical Poiseuille values (Eq 62)
    C_xx_cpu = ones(FT, Nx, Ny)
    C_xy_cpu = zeros(FT, Nx, Ny)
    C_yy_cpu = ones(FT, Nx, Ny)
    for j in 1:Ny, i in 1:Nx
        y = FT(j) - FT(0.5)
        dudy = U_max * FT(4) * (H - FT(2)*y) / (H * H)
        # Wi_local = λ/Ny · dudy  (but in lattice units, λ and dudy are already in l.u.)
        C_xy_cpu[i,j] = λ * dudy
        C_xx_cpu[i,j] = one(FT) + FT(2) * (λ * dudy)^2
        # C_yy = 1 (Eq 62)
    end
    copyto!(C_xx, C_xx_cpu); copyto!(C_xy, C_xy_cpu); copyto!(C_yy, C_yy_cpu)

    g_xx = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    g_xy = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    g_yy = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    init_conformation_field_2d!(g_xx, C_xx, ux, uy)
    init_conformation_field_2d!(g_xy, C_xy, ux, uy)
    init_conformation_field_2d!(g_yy, C_yy, ux, uy)
    g_xx_buf = similar(g_xx); g_xy_buf = similar(g_xy); g_yy_buf = similar(g_yy)

    tau_p_xx = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    tau_p_xy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    tau_p_yy = KernelAbstractions.zeros(backend, FT, Nx, Ny)

    # Initialize tau_p from analytical C
    @. tau_p_xx = G * (C_xx - 1)
    @. tau_p_xy = G * C_xy
    @. tau_p_yy = G * (C_yy - 1)

    Fx_sum = 0.0; n_avg = 0

    for step in 1:max_steps
        # --- Solvent LBM ---
        stream_2d!(f_out, f_in, Nx, Ny)

        # Inlet: Poiseuille velocity profile (Eq 61)
        apply_zou_he_west_profile_2d!(f_out, u_profile, Nx, Ny)
        # Outlet: zero-gradient (convective), NOT Zou-He pressure
        apply_extrapolate_east_2d!(f_out, Nx, Ny)

        # MEA drag
        if step > max_steps - avg_window
            drag = compute_drag_mea_2d(f_in, f_out, is_solid, Nx, Ny)
            Fx_sum += drag.Fx
            n_avg += 1
        end

        collide_viscoelastic_source_guo_2d!(f_out, is_solid, ω_s,
                                              FT(0), FT(0),
                                              tau_p_xx, tau_p_xy, tau_p_yy)
        compute_macroscopic_2d!(ρ, ux, uy, f_out)

        # --- Conformation LBM (TRT) ---
        stream_2d!(g_xx_buf, g_xx, Nx, Ny)
        stream_2d!(g_xy_buf, g_xy, Nx, Ny)
        stream_2d!(g_yy_buf, g_yy, Nx, Ny)

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

        @. tau_p_xx = G * (C_xx - 1)
        @. tau_p_xy = G * C_xy
        @. tau_p_yy = G * (C_yy - 1)

        f_in, f_out = f_out, f_in
    end

    Fx_avg = Fx_sum / n_avg
    # Cd = Fx / (0.5·ρ·U_avg²·D), Eq 64
    Cd = Fx_avg / (0.5 * 1.0 * U_c^2 * D)

    @info "Liu result" Wi R Cd Fx=Fx_avg

    return (; Cd, Wi, R, Fx=Fx_avg, U_c, Re)
end

# ============================================================
# Main sweep
# ============================================================

println("="^70)
println("Liu et al. 2025 cylinder benchmark reproduction")
println("Backend: $(typeof(backend)), GPU: $(CUDA.name(CUDA.device()))")
println("="^70)

# First: Newtonian reference (Wi ≈ 0, use tiny Wi=0.001 to avoid λ=∞)
# Then: Wi sweep at R=20 and R=30

for R in [20, 30]
    println("\n>>> R = $R")
    @printf("%-6s %-10s %-10s\n", "Wi", "Cd", "Cd_Liu_ref")
    println("-"^30)

    # Literature values (Table 3, CNEBB, Sc=10⁴) for R=30
    liu_ref = Dict(0.1 => 130.36, 0.5 => 126.31, 1.0 => 151.31)

    for Wi in [0.001, 0.1, 0.5]
        steps = Wi < 0.01 ? 300_000 : 500_000
        t0 = time()
        r = run_liu_cylinder(; R=R, Wi=Wi, max_steps=steps, backend=backend, FT=FT)
        dt = time() - t0
        ref = get(liu_ref, Wi, NaN)
        @printf("%-6.3f %-10.3f %-10.3f  (%.0fs)\n", Wi, r.Cd, ref, dt)
    end
end

println("\nDone.")
