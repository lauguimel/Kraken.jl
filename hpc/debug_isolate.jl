# Isolate which component of the NEW driver causes the 1.21 ratio
# (vs 0.947 of OLD driver) at Wi=0.001, R=20, β=0.59.
#
# We build a HYBRID driver that mixes OLD and NEW components:
#  - Solver: TRT+LI-BB (fused) OR plain BGK+stream+bounce
#  - MEA: Mei-with-Bouzidi OR standard halfway-BB
#  - Source: in-collision OR post-collision
# and scan all 8 combinations.

using Kraken, Printf, CUDA, KernelAbstractions
const backend = CUDABackend()
const FT = Float64

# Fixed setup
const R = 20; const Nx = 30R; const Ny = 4R
const cx_c = 15R; const cy_c = 2R
const u_mean = FT(0.02)
const ν_total = u_mean * R / 1.0
const β = FT(0.59)
const ν_s = β * ν_total
const ν_p = ν_total - ν_s
const Wi = FT(0.001); const λ = Wi * R / u_mean
const G = FT(ν_p / λ)
const tau_plus = FT(1.0)
const ω_s = FT(1.0 / (3*ν_s + 0.5))
const max_steps = 100_000
const avg_window = 20_000

# Reference Cd_Newt via run_cylinder_libb_2d (parabolic) + run_cylinder_2d (plug)
rNp = run_cylinder_libb_2d(; Nx=Nx, Ny=Ny, radius=R,
        u_in=FT(1.5*u_mean), ν=FT(ν_total),
        max_steps=max_steps, avg_window=avg_window,
        inlet=:parabolic, backend=backend, T=FT)
rNu = run_cylinder_2d(; Nx=Nx, Ny=Ny, radius=R,
        u_in=Float64(u_mean), ν=Float64(ν_total),
        max_steps=max_steps, avg_window=avg_window,
        backend=backend, T=FT)
println("="^66)
@printf("Cd_Newt LI-BB parabolic = %.3f\n", rNp.Cd)
@printf("Cd_Newt halfway plug    = %.3f\n", rNu.Cd)
println("="^66)

function run_hybrid(; solver::Symbol, mea::Symbol, source_timing::Symbol,
                      inlet::Symbol=:parabolic)
    q_wall_h, is_solid_h = precompute_q_wall_cylinder(Nx, Ny, cx_c, cy_c, R; FT=FT)
    q_wall = KernelAbstractions.allocate(backend, FT, Nx, Ny, 9)
    is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny)
    uw_x = KernelAbstractions.allocate(backend, FT, Nx, Ny, 9)
    uw_y = KernelAbstractions.allocate(backend, FT, Nx, Ny, 9)
    copyto!(q_wall, q_wall_h); copyto!(is_solid, is_solid_h)
    fill!(uw_x, zero(FT)); fill!(uw_y, zero(FT))

    u_max = inlet === :parabolic ? FT(1.5 * u_mean) : FT(u_mean)
    u_prof_h = inlet === :parabolic ?
        [FT(4) * u_max * FT(j - 1) * FT(Ny - j) / FT(Ny - 1)^2 for j in 1:Ny] :
        fill(FT(u_mean), Ny)
    u_profile = KernelAbstractions.allocate(backend, FT, Ny)
    copyto!(u_profile, u_prof_h)

    f_in = KernelAbstractions.allocate(backend, FT, Nx, Ny, 9)
    f_out = KernelAbstractions.allocate(backend, FT, Nx, Ny, 9)
    f_cpu = zeros(FT, Nx, Ny, 9)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        f_cpu[i,j,q] = Kraken.equilibrium(D2Q9(), one(FT), u_prof_h[j], zero(FT), q)
    end
    copyto!(f_in, f_cpu); fill!(f_out, zero(FT))
    ρ = KernelAbstractions.allocate(backend, FT, Nx, Ny); fill!(ρ, one(FT))
    ux = KernelAbstractions.allocate(backend, FT, Nx, Ny); fill!(ux, zero(FT))
    uy = KernelAbstractions.allocate(backend, FT, Nx, Ny); fill!(uy, zero(FT))

    C_xx = KernelAbstractions.zeros(backend, FT, Nx, Ny); fill!(C_xx, FT(1))
    C_xy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    C_yy = KernelAbstractions.zeros(backend, FT, Nx, Ny); fill!(C_yy, FT(1))
    g_xx = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    g_xy = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    g_yy = KernelAbstractions.zeros(backend, FT, Nx, Ny, 9)
    init_conformation_field_2d!(g_xx, C_xx, ux, uy)
    init_conformation_field_2d!(g_xy, C_xy, ux, uy)
    init_conformation_field_2d!(g_yy, C_yy, ux, uy)
    g_xx_b = similar(g_xx); g_xy_b = similar(g_xy); g_yy_b = similar(g_yy)
    txx = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    txy = KernelAbstractions.zeros(backend, FT, Nx, Ny)
    tyy = KernelAbstractions.zeros(backend, FT, Nx, Ny)

    bcspec = BCSpec2D(; west=ZouHeVelocity(u_profile), east=ZouHePressure(FT(1.0)))

    Fx_sum = 0.0; n = 0
    for step in 1:max_steps
        # --- SOLVER ---
        if solver === :trt_libb
            fused_trt_libb_v2_step!(f_out, f_in, ρ, ux, uy, is_solid,
                                     q_wall, uw_x, uw_y, Nx, Ny, FT(ν_s))
            apply_bc_rebuild_2d!(f_out, f_in, bcspec, ν_s, Nx, Ny)
        else  # :bgk
            stream_2d!(f_out, f_in, Nx, Ny)
            if inlet === :parabolic
                apply_zou_he_west_profile_2d!(f_out, u_profile, Nx, Ny)
            else
                apply_zou_he_west_2d!(f_out, FT(u_mean), Nx, Ny)
            end
            apply_zou_he_pressure_east_2d!(f_out, Nx, Ny)
        end

        # --- SOURCE (timing + collision for BGK) ---
        if solver === :bgk
            if source_timing === :incol
                collide_viscoelastic_source_guo_2d!(f_out, is_solid, ω_s,
                    FT(0), FT(0), txx, txy, tyy)
            else  # :postcol
                collide_2d!(f_out, is_solid, ω_s)
                apply_hermite_source_2d!(f_out, is_solid, ω_s, txx, txy, tyy)
            end
        else  # :trt_libb (always post-coll in current impl)
            apply_hermite_source_2d!(f_out, is_solid, ω_s, txx, txy, tyy)
        end

        # --- MEA ---
        if step > max_steps - avg_window
            if mea === :mei
                drag = compute_drag_libb_mei_2d(f_out, q_wall, uw_x, uw_y, Nx, Ny)
            else  # :std (halfway-BB)
                drag = compute_drag_mea_2d(f_in, f_out, is_solid, Nx, Ny)
            end
            Fx_sum += drag.Fx; n += 1
        end

        compute_macroscopic_2d!(ρ, ux, uy, f_out)

        # --- Conformation update ---
        stream_2d!(g_xx_b, g_xx, Nx, Ny)
        stream_2d!(g_xy_b, g_xy, Nx, Ny)
        stream_2d!(g_yy_b, g_yy, Nx, Ny)
        apply_cnebb_conformation_2d!(g_xx_b, g_xx, is_solid, C_xx)
        apply_cnebb_conformation_2d!(g_xy_b, g_xy, is_solid, C_xy)
        apply_cnebb_conformation_2d!(g_yy_b, g_yy, is_solid, C_yy)
        g_xx, g_xx_b = g_xx_b, g_xx; g_xy, g_xy_b = g_xy_b, g_xy; g_yy, g_yy_b = g_yy_b, g_yy
        compute_conformation_macro_2d!(C_xx, g_xx)
        compute_conformation_macro_2d!(C_xy, g_xy)
        compute_conformation_macro_2d!(C_yy, g_yy)
        collide_conformation_2d!(g_xx, C_xx, ux, uy, C_xx, C_xy, C_yy, is_solid, tau_plus, Float64(λ); component=1)
        collide_conformation_2d!(g_xy, C_xy, ux, uy, C_xx, C_xy, C_yy, is_solid, tau_plus, Float64(λ); component=2)
        collide_conformation_2d!(g_yy, C_yy, ux, uy, C_xx, C_xy, C_yy, is_solid, tau_plus, Float64(λ); component=3)
        @. txx = G * (C_xx - one(FT)); @. txy = G * C_xy; @. tyy = G * (C_yy - one(FT))

        f_in, f_out = f_out, f_in
    end
    KernelAbstractions.synchronize(backend)
    Cd = 2.0 * (Fx_sum/n) / (Float64(u_mean)^2 * 2R)
    return Cd
end

@printf("\n%-12s %-6s %-8s %-6s %-10s %-8s\n",
        "solver", "mea", "source", "inlet", "Cd", "ratio")
println("-"^62)
for solver in [:bgk, :trt_libb], mea in [:std, :mei], timing in [:incol, :postcol]
    # LI-BB only supports post-coll in our impl
    if solver === :trt_libb && timing === :incol
        continue
    end
    inlet = solver === :trt_libb ? :parabolic : :uniform
    ref_Cd = inlet === :parabolic ? rNp.Cd : rNu.Cd
    Cd = run_hybrid(; solver=solver, mea=mea, source_timing=timing, inlet=inlet)
    @printf("%-12s %-6s %-8s %-6s %-10.3f %-8.4f\n",
            string(solver), string(mea), string(timing),
            string(inlet), Cd, Cd/ref_Cd)
end
