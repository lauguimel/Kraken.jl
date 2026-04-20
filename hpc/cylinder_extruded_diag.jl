# Cylinder extruded 3D vs cylinder 2D — diagnostic for u↔C coupling.
#
# Most discriminating test for the 3D viscoelastic port. The 2D Liu
# cylinder is validated to 0.32% at R=48, Wi=0.1 (Cd=130.78 vs Liu
# 130.36). The 3D port must reproduce this benchmark when the geometry
# is z-extruded (= infinite cylinder, periodic in z, u_z = 0). Any
# deviation localises the bug:
#
#  Cd_3D ≈ Cd_2D, fields match → kernel + drag OK on extruded; sphere
#                                bug = TRUE 3D curvature (CNEBB on
#                                sphere surface diagonal links).
#  Cd_3D differs but C_xy(y) bulk OK → drag integration on 3D cut links
#                                       differs from 2D version.
#  C_xy(y) differs in bulk → Hermite source × wall (LI-BB cut links)
#                              with full feedback bug.
#  C_xz, C_yz, C_zz - 1 grow in 3D → indexing/coupling bug in
#                                      collide_conformation_3d! that
#                                      doesn't show in homogeneous shear.
#
# Output: results/cylinder_extruded_diag.txt

using Kraken, Printf, CUDA, KernelAbstractions
import Kraken: D3Q19, equilibrium, fused_trt_libb_v2_step_3d_periodic_z!,
               apply_bc_rebuild_3d!, apply_hermite_source_3d!,
               compute_drag_libb_3d, init_conformation_field_3d!,
               compute_conformation_macro_3d!, apply_polymer_wall_bc!,
               collide_conformation_3d!, update_polymer_stress_3d!,
               reset_conformation_inlet_3d!, reset_conformation_outlet_3d!,
               precompute_q_wall_cylinder_extruded_3d, BCSpec3D,
               ZouHeVelocity, ZouHePressure, CNEBB, OldroydB,
               stream_fully_periodic_3d!

backend = CUDABackend()
FT = Float64

println("="^70)
println("Cylinder 2D vs cylinder 3D extruded (LI-BB V2 + Hermite)")
println("Backend: $(typeof(backend)), GPU: $(CUDA.name(CUDA.device()))")
println("="^70)

# Match the 2D benchmark (Liu Wi=0.1 sweet spot, R=30 to keep cost low)
R     = 30
Nx    = 30 * R
Ny    = 4  * R
Nz_3d = 12              # thin in z, periodic
cx    = 15 * R
cy    = 2  * R
β     = 0.59
u_mean = 0.02
ν_total = u_mean * R / 1.0
ν_s = β * ν_total
ν_p = (1 - β) * ν_total
Wi  = 0.1
λ   = Wi * R / u_mean
G   = ν_p / λ
max_steps  = 100_000
avg_window = max_steps ÷ 5

@printf("Geometry  : Nx=%d, Ny=%d, Nz_3d=%d, R=%d, (cx,cy)=(%d,%d)\n",
        Nx, Ny, Nz_3d, R, cx, cy)
@printf("Flow      : u_mean=%.3g, β=%.2f, λ=%.3g, Wi=%.2f\n",
        u_mean, β, λ, Wi)
@printf("LBM       : steps=%d, avg_window=%d\n\n", max_steps, avg_window)

# ----------------------------------------------------------------------
# 2D reference
# ----------------------------------------------------------------------
println("--- Running 2D cylinder Oldroyd-B (reference) ---")
t0 = time()
m_OB = OldroydB(G=FT(G), λ=FT(λ))
r2d = run_conformation_cylinder_libb_2d(;
        Nx=Nx, Ny=Ny, radius=R, cx=cx, cy=cy,
        u_mean=u_mean, ν_s=ν_s,
        polymer_model=m_OB, polymer_bc=CNEBB(),
        inlet=:parabolic, ρ_out=1.0, tau_plus=1.0,
        max_steps=max_steps, avg_window=avg_window,
        backend=backend, FT=FT)
@printf("2D done in %.0f s — Cd = %.4f\n", time() - t0, r2d.Cd)

# Sample at the centerline x = cx + 2R (downstream of cylinder, in wake)
i_sample = cx + 2*R
Cxy2d_wake = r2d.C_xy[i_sample, :]
Cxx2d_wake = r2d.C_xx[i_sample, :]
ux2d_wake  = r2d.ux[i_sample, :]

# ----------------------------------------------------------------------
# 3D extruded
# ----------------------------------------------------------------------
println("\n--- Running 3D extruded cylinder Oldroyd-B ---")
t0 = time()
# Build the 3D extruded geometry manually + use a custom run loop.
# For now, hack: use the existing run_conformation_sphere_libb_3d by
# substituting precompute_q_wall_sphere_3d via Julia method-replacement
# is unreliable. Cleaner: write inline.

# --- inline 3D extruded driver ---
let
    Nx_l = Nx; Ny_l = Ny; Nz_l = Nz_3d
    s_plus_s = 1.0 / (3.0 * ν_s + 0.5)

    q_wall_h, is_solid_h = precompute_q_wall_cylinder_extruded_3d(
        Nx_l, Ny_l, Nz_l, Float64(cx), Float64(cy), Float64(R); FT=FT)

    # Inlet velocity profile parabolic_y, uniform z (Schäfer-Turek)
    u_max = FT(1.5) * FT(u_mean)
    u_prof_h = zeros(FT, Ny_l, Nz_l)
    for k in 1:Nz_l, j in 1:Ny_l
        y = FT(j) - FT(0.5)
        u_prof_h[j, k] = FT(4) * u_max * y * (FT(Ny_l) - y) / FT(Ny_l)^2
    end

    # Inlet conformation profile (analytical Oldroyd-B, y-only shear)
    Cxx_in_h = ones(FT, Ny_l, Nz_l)
    Cxy_in_h = zeros(FT, Ny_l, Nz_l)
    for k in 1:Nz_l, j in 1:Ny_l
        y = FT(j) - FT(0.5)
        dudy = FT(4) * u_max * (FT(Ny_l) - 2*y) / FT(Ny_l)^2
        Cxy_in_h[j, k] = FT(λ) * dudy
        Cxx_in_h[j, k] = FT(1) + FT(2) * (FT(λ) * dudy)^2
    end
    Cident_h = ones(FT, Ny_l, Nz_l)
    Czero_h  = zeros(FT, Ny_l, Nz_l)

    q_wall   = KernelAbstractions.allocate(backend, FT,   Nx_l, Ny_l, Nz_l, 19); copyto!(q_wall, q_wall_h)
    is_solid = KernelAbstractions.allocate(backend, Bool, Nx_l, Ny_l, Nz_l);     copyto!(is_solid, is_solid_h)
    uw_x = KernelAbstractions.zeros(backend, FT, Nx_l, Ny_l, Nz_l, 19)
    uw_y = KernelAbstractions.zeros(backend, FT, Nx_l, Ny_l, Nz_l, 19)
    uw_z = KernelAbstractions.zeros(backend, FT, Nx_l, Ny_l, Nz_l, 19)
    f_in  = KernelAbstractions.allocate(backend, FT, Nx_l, Ny_l, Nz_l, 19)
    f_out = KernelAbstractions.allocate(backend, FT, Nx_l, Ny_l, Nz_l, 19)
    ρ  = KernelAbstractions.zeros(backend, FT, Nx_l, Ny_l, Nz_l); fill!(ρ, FT(1))
    ux = KernelAbstractions.zeros(backend, FT, Nx_l, Ny_l, Nz_l)
    uy = KernelAbstractions.zeros(backend, FT, Nx_l, Ny_l, Nz_l)
    uz = KernelAbstractions.zeros(backend, FT, Nx_l, Ny_l, Nz_l)

    u_profile  = KernelAbstractions.allocate(backend, FT, Ny_l, Nz_l); copyto!(u_profile, u_prof_h)
    Cxx_in_d   = KernelAbstractions.allocate(backend, FT, Ny_l, Nz_l); copyto!(Cxx_in_d, Cxx_in_h)
    Cxy_in_d   = KernelAbstractions.allocate(backend, FT, Ny_l, Nz_l); copyto!(Cxy_in_d, Cxy_in_h)
    Cident_d   = KernelAbstractions.allocate(backend, FT, Ny_l, Nz_l); copyto!(Cident_d, Cident_h)
    Czero_d    = KernelAbstractions.allocate(backend, FT, Ny_l, Nz_l); copyto!(Czero_d, Czero_h)

    bcspec = BCSpec3D(; west=ZouHeVelocity(u_profile), east=ZouHePressure(FT(1.0)))

    # Init f to equilibrium at inlet profile
    f_in_h = zeros(FT, Nx_l, Ny_l, Nz_l, 19)
    for k in 1:Nz_l, j in 1:Ny_l, i in 1:Nx_l, q in 1:19
        u0 = is_solid_h[i,j,k] ? zero(FT) : u_prof_h[j,k]
        f_in_h[i,j,k,q] = equilibrium(D3Q19(), one(FT), u0, zero(FT), zero(FT), q)
    end
    copyto!(f_in, f_in_h); fill!(f_out, zero(FT))

    Cxx = KernelAbstractions.zeros(backend, FT, Nx_l, Ny_l, Nz_l); fill!(Cxx, FT(1))
    Cyy = KernelAbstractions.zeros(backend, FT, Nx_l, Ny_l, Nz_l); fill!(Cyy, FT(1))
    Czz = KernelAbstractions.zeros(backend, FT, Nx_l, Ny_l, Nz_l); fill!(Czz, FT(1))
    Cxy = KernelAbstractions.zeros(backend, FT, Nx_l, Ny_l, Nz_l)
    Cxz = KernelAbstractions.zeros(backend, FT, Nx_l, Ny_l, Nz_l)
    Cyz = KernelAbstractions.zeros(backend, FT, Nx_l, Ny_l, Nz_l)

    g_xx = KernelAbstractions.zeros(backend, FT, Nx_l, Ny_l, Nz_l, 19); init_conformation_field_3d!(g_xx, Cxx, ux, uy, uz)
    g_xy = KernelAbstractions.zeros(backend, FT, Nx_l, Ny_l, Nz_l, 19); init_conformation_field_3d!(g_xy, Cxy, ux, uy, uz)
    g_xz = KernelAbstractions.zeros(backend, FT, Nx_l, Ny_l, Nz_l, 19); init_conformation_field_3d!(g_xz, Cxz, ux, uy, uz)
    g_yy = KernelAbstractions.zeros(backend, FT, Nx_l, Ny_l, Nz_l, 19); init_conformation_field_3d!(g_yy, Cyy, ux, uy, uz)
    g_yz = KernelAbstractions.zeros(backend, FT, Nx_l, Ny_l, Nz_l, 19); init_conformation_field_3d!(g_yz, Cyz, ux, uy, uz)
    g_zz = KernelAbstractions.zeros(backend, FT, Nx_l, Ny_l, Nz_l, 19); init_conformation_field_3d!(g_zz, Czz, ux, uy, uz)
    g_xx_buf = similar(g_xx); g_xy_buf = similar(g_xy); g_xz_buf = similar(g_xz)
    g_yy_buf = similar(g_yy); g_yz_buf = similar(g_yz); g_zz_buf = similar(g_zz)

    txx = KernelAbstractions.zeros(backend, FT, Nx_l, Ny_l, Nz_l)
    txy = similar(txx); txz = similar(txx)
    tyy = similar(txx); tyz = similar(txx); tzz = similar(txx)

    polymer_model = OldroydB(G=FT(G), λ=FT(λ))
    Fx_sum = 0.0; n_avg = 0
    for step in 1:max_steps
        fused_trt_libb_v2_step_3d_periodic_z!(f_out, f_in, ρ, ux, uy, uz, is_solid,
                                                q_wall, uw_x, uw_y, uw_z,
                                                Nx_l, Ny_l, Nz_l, FT(ν_s))
        apply_bc_rebuild_3d!(f_out, f_in, bcspec, ν_s, Nx_l, Ny_l, Nz_l)
        apply_hermite_source_3d!(f_out, is_solid, s_plus_s, txx, txy, txz, tyy, tyz, tzz)
        if step > max_steps - avg_window
            d = compute_drag_libb_3d(f_out, q_wall, Nx_l, Ny_l, Nz_l)
            Fx_sum += d.Fx; n_avg += 1
        end
        # Conformation
        stream_fully_periodic_3d!(g_xx_buf, g_xx, Nx_l, Ny_l, Nz_l)
        stream_fully_periodic_3d!(g_xy_buf, g_xy, Nx_l, Ny_l, Nz_l)
        stream_fully_periodic_3d!(g_xz_buf, g_xz, Nx_l, Ny_l, Nz_l)
        stream_fully_periodic_3d!(g_yy_buf, g_yy, Nx_l, Ny_l, Nz_l)
        stream_fully_periodic_3d!(g_yz_buf, g_yz, Nx_l, Ny_l, Nz_l)
        stream_fully_periodic_3d!(g_zz_buf, g_zz, Nx_l, Ny_l, Nz_l)
        apply_polymer_wall_bc!(g_xx_buf, g_xx, is_solid, Cxx, CNEBB())
        apply_polymer_wall_bc!(g_xy_buf, g_xy, is_solid, Cxy, CNEBB())
        apply_polymer_wall_bc!(g_xz_buf, g_xz, is_solid, Cxz, CNEBB())
        apply_polymer_wall_bc!(g_yy_buf, g_yy, is_solid, Cyy, CNEBB())
        apply_polymer_wall_bc!(g_yz_buf, g_yz, is_solid, Cyz, CNEBB())
        apply_polymer_wall_bc!(g_zz_buf, g_zz, is_solid, Czz, CNEBB())
        reset_conformation_inlet_3d!(g_xx_buf, Cxx_in_d, u_profile, Ny_l, Nz_l)
        reset_conformation_inlet_3d!(g_xy_buf, Cxy_in_d, u_profile, Ny_l, Nz_l)
        reset_conformation_inlet_3d!(g_xz_buf, Czero_d,  u_profile, Ny_l, Nz_l)
        reset_conformation_inlet_3d!(g_yy_buf, Cident_d, u_profile, Ny_l, Nz_l)
        reset_conformation_inlet_3d!(g_yz_buf, Czero_d,  u_profile, Ny_l, Nz_l)
        reset_conformation_inlet_3d!(g_zz_buf, Cident_d, u_profile, Ny_l, Nz_l)
        reset_conformation_outlet_3d!(g_xx_buf, Nx_l, Ny_l, Nz_l)
        reset_conformation_outlet_3d!(g_xy_buf, Nx_l, Ny_l, Nz_l)
        reset_conformation_outlet_3d!(g_xz_buf, Nx_l, Ny_l, Nz_l)
        reset_conformation_outlet_3d!(g_yy_buf, Nx_l, Ny_l, Nz_l)
        reset_conformation_outlet_3d!(g_yz_buf, Nx_l, Ny_l, Nz_l)
        reset_conformation_outlet_3d!(g_zz_buf, Nx_l, Ny_l, Nz_l)
        g_xx, g_xx_buf = g_xx_buf, g_xx; g_xy, g_xy_buf = g_xy_buf, g_xy
        g_xz, g_xz_buf = g_xz_buf, g_xz; g_yy, g_yy_buf = g_yy_buf, g_yy
        g_yz, g_yz_buf = g_yz_buf, g_yz; g_zz, g_zz_buf = g_zz_buf, g_zz
        compute_conformation_macro_3d!(Cxx, g_xx); compute_conformation_macro_3d!(Cxy, g_xy)
        compute_conformation_macro_3d!(Cxz, g_xz); compute_conformation_macro_3d!(Cyy, g_yy)
        compute_conformation_macro_3d!(Cyz, g_yz); compute_conformation_macro_3d!(Czz, g_zz)
        for (g, Cf, comp) in ((g_xx, Cxx, 1), (g_xy, Cxy, 2), (g_xz, Cxz, 3),
                                (g_yy, Cyy, 4), (g_yz, Cyz, 5), (g_zz, Czz, 6))
            collide_conformation_3d!(g, Cf, ux, uy, uz, Cxx, Cxy, Cxz, Cyy, Cyz, Czz, is_solid, 1.0, FT(λ); component=comp)
        end
        update_polymer_stress_3d!(txx, txy, txz, tyy, tyz, tzz,
                                    Cxx, Cxy, Cxz, Cyy, Cyz, Czz, polymer_model)
        f_in, f_out = f_out, f_in
    end
    KernelAbstractions.synchronize(backend)

    Fx = Fx_sum / n_avg
    D_diam = 2 * R
    # 3D extruded drag normalisation: project to per-unit-z (divide by Nz)
    Cd_3d = 2.0 * (Fx / Nz_l) / (u_mean^2 * D_diam)
    @printf("3D extruded done — Fx_total=%.4e (over Nz=%d), Cd_per_z=%.4f\n",
            Fx, Nz_l, Cd_3d)

    # Sample at i_sample, k = Nz/2
    Cxy3d_wake = Array(Cxy)[i_sample, :, Nz_l ÷ 2]
    Cxx3d_wake = Array(Cxx)[i_sample, :, Nz_l ÷ 2]
    ux3d_wake  = Array(ux)[i_sample, :, Nz_l ÷ 2]
    Cxz3d      = Array(Cxz)
    Cyz3d      = Array(Cyz)
    Czz3d      = Array(Czz)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------
    println("\n--- Profile comparison (wake centerline) ---")
    @printf("%-4s %-10s %-10s %-10s %-10s %-10s %-10s\n",
            "j", "u_2d", "u_3d", "Cxy_2d", "Cxy_3d", "Cxx_2d", "Cxx_3d")
    for j in 1:4:Ny
        @printf("%-4d %-10.5f %-10.5f %-10.4f %-10.4f %-10.4f %-10.4f\n",
                j, ux2d_wake[j], ux3d_wake[j],
                Cxy2d_wake[j], Cxy3d_wake[j],
                Cxx2d_wake[j], Cxx3d_wake[j])
    end

    # Bulk metric: skip wall cells
    function compare(num, ref, label)
        j1, j2 = 4, length(num) - 3
        err = maximum(abs.(num[j1:j2] .- ref[j1:j2]))
        norm = max(maximum(abs.(ref[j1:j2])), 1e-12)
        @printf("  %-12s : max_diff=%.4e  rel=%.4f\n", label, err, err/norm)
    end

    println("\n--- 3D extruded vs 2D reference (wake) ---")
    compare(ux3d_wake,  ux2d_wake,  "u_x")
    compare(Cxy3d_wake, Cxy2d_wake, "C_xy")
    compare(Cxx3d_wake, Cxx2d_wake, "C_xx")

    @printf("\nCd ratio 3D/2D = %.4f  (target ≈ 1.00)\n", Cd_3d / r2d.Cd)

    println("\n--- 3D out-of-plane (max over whole 3D domain) ---")
    @printf("  max |C_xz|     = %.4e\n", maximum(abs.(Cxz3d)))
    @printf("  max |C_yz|     = %.4e\n", maximum(abs.(Cyz3d)))
    @printf("  max |C_zz - 1| = %.4e\n", maximum(abs.(Czz3d .- 1)))
end

println("\nDone.")
