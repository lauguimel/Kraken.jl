# 3D viscoelastic Poiseuille flow in a SQUARE DUCT — Option C diagnostic.
#
# Cross-section: square Ny × Nz with halfway-BB walls on all 4 transverse
# faces. Streamwise: pressure-driven via Zou-He inlet (parabolic in y AND
# z, doubly parabolic) and pressure outlet. NO internal solid → no LI-BB
# Bouzidi cut links → walls are pure axis-aligned halfway-BB.
#
# This isolates whether the Aqua sphere bug (Cd_visco / Cd_Newt = 0.89 at
# Wi=0.1, decreasing in Wi) comes from CNEBB-on-curved-walls (which only
# the sphere has) or from the 3D Hermite source × wall interaction in
# general (which the duct also has).
#
# Expected results:
# - Newtonian Poiseuille square duct: well-known reference (e.g. White
#   Viscous Fluid Flow, Tab 3-2: Q vs Δp/L analytical series solution).
# - Viscoelastic at low Wi: same flow as Newtonian at ν_total (since
#   ∂τ_p/∂x ≈ 0 in fully developed channel — but the corner regions
#   have non-trivial stress feedback).
#
# Diagnostic:
# - Compare bulk centerline velocity to Newtonian reference
# - Check C_xy(y, k=Nz/2), C_xz(j=Ny/2, z), C_yz, C_xx, C_yy, C_zz fields
# - All should be smooth, finite, magnitude consistent with Wi
# - NO field should explode if the kernels are correct with axis-aligned walls
#
# If the duct works correctly (no field explosion, sensible Cd correction
# vs Newtonian) → bug is specific to CNEBB on curved walls (sphere)
# If the duct also explodes → bug is in Hermite × axis-aligned wall too
#
# Output: results/duct_visco_diag.txt

using Kraken, Printf, CUDA, KernelAbstractions
import Kraken: D3Q19, equilibrium, fused_trt_libb_v2_step_3d!,
               apply_bc_rebuild_3d!, apply_hermite_source_3d!,
               init_conformation_field_3d!, compute_conformation_macro_3d!,
               apply_polymer_wall_bc!, collide_conformation_3d!,
               update_polymer_stress_3d!, reset_conformation_inlet_3d!,
               reset_conformation_outlet_3d!, BCSpec3D,
               ZouHeVelocity, ZouHePressure, CNEBB, OldroydB, stream_3d!

backend = CUDABackend()
FT = Float64

println("="^70)
println("3D viscoelastic Poiseuille square duct — diagnostic for Hermite × wall")
println("Backend: $(typeof(backend)), GPU: $(CUDA.name(CUDA.device()))")
println("="^70)

H     = 30                      # half-height of duct (R from Liu convention)
Ny    = 2 * H                   # full y extent
Nz    = 2 * H                   # full z extent (square cross-section)
Nx    = 6 * H                   # streamwise length (~6 H suffices for fully developed)
β     = 0.59
u_max = 0.04                    # peak velocity at duct centre (j=Nz=H)
ν_total = u_max * H / 1.0       # Re_local = u_max·H/ν_total ≈ 1
ν_s   = β * ν_total
ν_p   = (1 - β) * ν_total
λ_visc = 0.1 * H / u_max        # Wi = λ·u_max/H = 0.1
G     = ν_p / λ_visc
max_steps  = 100_000
avg_window = max_steps ÷ 5
s_plus_s   = 1.0 / (3.0 * ν_s + 0.5)

@printf("Geometry  : Nx=%d, Ny=%d, Nz=%d, H=%d (square duct)\n", Nx, Ny, Nz, H)
@printf("Flow      : u_max=%.3g, β=%.2f, ν_s=%.3g, ν_p=%.3g, λ=%.3g, Wi=0.1\n",
        u_max, β, ν_s, ν_p, λ_visc)
@printf("LBM       : steps=%d, avg_window=%d\n\n", max_steps, avg_window)

# --- All-fluid mask, q_wall=0 (no LI-BB cut links — pure halfway-BB walls)
q_wall_h = zeros(FT, Nx, Ny, Nz, 19)
is_solid_h = zeros(Bool, Nx, Ny, Nz)

# --- Doubly-parabolic inlet u(y, k) — analog of 2D Poiseuille
u_prof_h = zeros(FT, Ny, Nz)
Hy = FT(Ny - 1); Hz = FT(Nz - 1)
for k in 1:Nz, j in 1:Ny
    yy = FT(j - 1); zz = FT(k - 1)
    u_prof_h[j, k] = FT(16) * FT(u_max) *
                      yy * (Hy - yy) * zz * (Hz - zz) / (Hy^2 * Hz^2)
end

# --- Inlet conformation profile: analytical Oldroyd-B at fully-developed
# square duct flow. For this 3D inlet we use the y-shear and z-shear
# contributions separately; Cxx_in includes both.
Cxx_in_h = ones(FT, Ny, Nz); Cxy_in_h = zeros(FT, Ny, Nz)
Cxz_in_h = zeros(FT, Ny, Nz); Cyy_in_h = ones(FT, Ny, Nz)
Cyz_in_h = zeros(FT, Ny, Nz); Czz_in_h = ones(FT, Ny, Nz)
for k in 1:Nz, j in 1:Ny
    y = FT(j) - FT(0.5); z = FT(k) - FT(0.5)
    dudy = FT(16) * FT(u_max) * (Hy - 2*y) * z * (Hz - z) / (Hy^2 * Hz^2)
    dudz = FT(16) * FT(u_max) * y * (Hy - y) * (Hz - 2*z) / (Hy^2 * Hz^2)
    Cxy_in_h[j, k] = FT(λ_visc) * dudy
    Cxz_in_h[j, k] = FT(λ_visc) * dudz
    Cxx_in_h[j, k] = FT(1) + FT(2) * (FT(λ_visc) * dudy)^2 + FT(2) * (FT(λ_visc) * dudz)^2
end

# --- Device allocations + main loop wrapped in `let` (Julia 1.12 soft-scope)
let
q_wall   = KernelAbstractions.allocate(backend, FT,   Nx, Ny, Nz, 19); copyto!(q_wall, q_wall_h)
is_solid = KernelAbstractions.allocate(backend, Bool, Nx, Ny, Nz);     copyto!(is_solid, is_solid_h)
uw_x = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
uw_y = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
uw_z = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19)
f_in  = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz, 19)
f_out = KernelAbstractions.allocate(backend, FT, Nx, Ny, Nz, 19)
ρ  = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz); fill!(ρ, FT(1))
ux = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
uy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
uz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)

u_profile = KernelAbstractions.allocate(backend, FT, Ny, Nz); copyto!(u_profile, u_prof_h)
Cxx_in_d = KernelAbstractions.allocate(backend, FT, Ny, Nz); copyto!(Cxx_in_d, Cxx_in_h)
Cxy_in_d = KernelAbstractions.allocate(backend, FT, Ny, Nz); copyto!(Cxy_in_d, Cxy_in_h)
Cxz_in_d = KernelAbstractions.allocate(backend, FT, Ny, Nz); copyto!(Cxz_in_d, Cxz_in_h)
Cident_d = KernelAbstractions.allocate(backend, FT, Ny, Nz); fill!(Cident_d, FT(1))
Czero_d  = KernelAbstractions.allocate(backend, FT, Ny, Nz); fill!(Czero_d,  zero(FT))

bcspec = BCSpec3D(; west=ZouHeVelocity(u_profile), east=ZouHePressure(FT(1.0)))

# --- Init f to equilibrium at inlet profile
f_in_h = zeros(FT, Nx, Ny, Nz, 19)
for k in 1:Nz, j in 1:Ny, i in 1:Nx, q in 1:19
    f_in_h[i,j,k,q] = equilibrium(D3Q19(), one(FT), u_prof_h[j,k], zero(FT), zero(FT), q)
end
copyto!(f_in, f_in_h); fill!(f_out, zero(FT))

Cxx = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz); fill!(Cxx, FT(1))
Cyy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz); fill!(Cyy, FT(1))
Czz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz); fill!(Czz, FT(1))
Cxy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
Cxz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
Cyz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)

g_xx = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19); init_conformation_field_3d!(g_xx, Cxx, ux, uy, uz)
g_xy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19); init_conformation_field_3d!(g_xy, Cxy, ux, uy, uz)
g_xz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19); init_conformation_field_3d!(g_xz, Cxz, ux, uy, uz)
g_yy = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19); init_conformation_field_3d!(g_yy, Cyy, ux, uy, uz)
g_yz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19); init_conformation_field_3d!(g_yz, Cyz, ux, uy, uz)
g_zz = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz, 19); init_conformation_field_3d!(g_zz, Czz, ux, uy, uz)
g_xx_buf = similar(g_xx); g_xy_buf = similar(g_xy); g_xz_buf = similar(g_xz)
g_yy_buf = similar(g_yy); g_yz_buf = similar(g_yz); g_zz_buf = similar(g_zz)

txx = KernelAbstractions.zeros(backend, FT, Nx, Ny, Nz)
txy = similar(txx); txz = similar(txx)
tyy = similar(txx); tyz = similar(txx); tzz = similar(txx)

polymer_model = OldroydB(G=FT(G), λ=FT(λ_visc))

println("--- Running 3D viscoelastic square duct ---")
t0 = time()
for step in 1:max_steps
    fused_trt_libb_v2_step_3d!(f_out, f_in, ρ, ux, uy, uz, is_solid,
                                 q_wall, uw_x, uw_y, uw_z,
                                 Nx, Ny, Nz, FT(ν_s))
    apply_bc_rebuild_3d!(f_out, f_in, bcspec, ν_s, Nx, Ny, Nz)
    apply_hermite_source_3d!(f_out, is_solid, s_plus_s, txx, txy, txz, tyy, tyz, tzz)

    stream_3d!(g_xx_buf, g_xx, Nx, Ny, Nz)
    stream_3d!(g_xy_buf, g_xy, Nx, Ny, Nz)
    stream_3d!(g_xz_buf, g_xz, Nx, Ny, Nz)
    stream_3d!(g_yy_buf, g_yy, Nx, Ny, Nz)
    stream_3d!(g_yz_buf, g_yz, Nx, Ny, Nz)
    stream_3d!(g_zz_buf, g_zz, Nx, Ny, Nz)
    apply_polymer_wall_bc!(g_xx_buf, g_xx, is_solid, Cxx, CNEBB())
    apply_polymer_wall_bc!(g_xy_buf, g_xy, is_solid, Cxy, CNEBB())
    apply_polymer_wall_bc!(g_xz_buf, g_xz, is_solid, Cxz, CNEBB())
    apply_polymer_wall_bc!(g_yy_buf, g_yy, is_solid, Cyy, CNEBB())
    apply_polymer_wall_bc!(g_yz_buf, g_yz, is_solid, Cyz, CNEBB())
    apply_polymer_wall_bc!(g_zz_buf, g_zz, is_solid, Czz, CNEBB())
    reset_conformation_inlet_3d!(g_xx_buf, Cxx_in_d, u_profile, Ny, Nz)
    reset_conformation_inlet_3d!(g_xy_buf, Cxy_in_d, u_profile, Ny, Nz)
    reset_conformation_inlet_3d!(g_xz_buf, Cxz_in_d, u_profile, Ny, Nz)
    reset_conformation_inlet_3d!(g_yy_buf, Cident_d, u_profile, Ny, Nz)
    reset_conformation_inlet_3d!(g_yz_buf, Czero_d,  u_profile, Ny, Nz)
    reset_conformation_inlet_3d!(g_zz_buf, Cident_d, u_profile, Ny, Nz)
    reset_conformation_outlet_3d!(g_xx_buf, Nx, Ny, Nz)
    reset_conformation_outlet_3d!(g_xy_buf, Nx, Ny, Nz)
    reset_conformation_outlet_3d!(g_xz_buf, Nx, Ny, Nz)
    reset_conformation_outlet_3d!(g_yy_buf, Nx, Ny, Nz)
    reset_conformation_outlet_3d!(g_yz_buf, Nx, Ny, Nz)
    reset_conformation_outlet_3d!(g_zz_buf, Nx, Ny, Nz)
    g_xx, g_xx_buf = g_xx_buf, g_xx; g_xy, g_xy_buf = g_xy_buf, g_xy
    g_xz, g_xz_buf = g_xz_buf, g_xz; g_yy, g_yy_buf = g_yy_buf, g_yy
    g_yz, g_yz_buf = g_yz_buf, g_yz; g_zz, g_zz_buf = g_zz_buf, g_zz
    compute_conformation_macro_3d!(Cxx, g_xx); compute_conformation_macro_3d!(Cxy, g_xy)
    compute_conformation_macro_3d!(Cxz, g_xz); compute_conformation_macro_3d!(Cyy, g_yy)
    compute_conformation_macro_3d!(Cyz, g_yz); compute_conformation_macro_3d!(Czz, g_zz)
    for (g, Cf, comp) in ((g_xx, Cxx, 1), (g_xy, Cxy, 2), (g_xz, Cxz, 3),
                            (g_yy, Cyy, 4), (g_yz, Cyz, 5), (g_zz, Czz, 6))
        collide_conformation_3d!(g, Cf, ux, uy, uz, Cxx, Cxy, Cxz, Cyy, Cyz, Czz, is_solid, 1.0, FT(λ_visc); component=comp)
    end
    update_polymer_stress_3d!(txx, txy, txz, tyy, tyz, tzz,
                                Cxx, Cxy, Cxz, Cyy, Cyz, Czz, polymer_model)
    f_in, f_out = f_out, f_in
end
KernelAbstractions.synchronize(backend)
@printf("3D duct run done in %.0f s\n\n", time() - t0)

# --- Diagnostics
ux_h = Array(ux); uy_h = Array(uy); uz_h = Array(uz)
Cxx_h = Array(Cxx); Cxy_h = Array(Cxy); Cxz_h = Array(Cxz)
Cyy_h = Array(Cyy); Cyz_h = Array(Cyz); Czz_h = Array(Czz)

i_s = 3 * Nx ÷ 4   # downstream sample
jc = Ny ÷ 2; kc = Nz ÷ 2

println("--- Centerline (i=$i_s, j=$jc, k=$kc) values ---")
@printf("u_x = %.5f  uy=%.4e  uz=%.4e\n", ux_h[i_s,jc,kc], uy_h[i_s,jc,kc], uz_h[i_s,jc,kc])
@printf("C_xx=%.4f  C_yy=%.4f  C_zz=%.4f\n", Cxx_h[i_s,jc,kc], Cyy_h[i_s,jc,kc], Czz_h[i_s,jc,kc])
@printf("C_xy=%.4e C_xz=%.4e C_yz=%.4e\n", Cxy_h[i_s,jc,kc], Cxz_h[i_s,jc,kc], Cyz_h[i_s,jc,kc])

# Symmetry check: at center y=Ny/2, ∂u/∂y = 0 → C_xy ≈ 0
# At center z=Nz/2, ∂u/∂z = 0 → C_xz ≈ 0
println("\n--- Field-magnitude sanity (skip 3 cells from each wall) ---")
function bulk_max(arr)
    a = arr[3:end-2, 3:end-2, 3:end-2]
    return maximum(abs.(a))
end
@printf("max |u_x|     = %.4e  (target ≈ %.4e)\n", bulk_max(ux_h), u_max)
@printf("max |u_y|     = %.4e  (target ≈ 0; allow 1e-3)\n", bulk_max(uy_h))
@printf("max |u_z|     = %.4e  (target ≈ 0; allow 1e-3)\n", bulk_max(uz_h))
@printf("max |C_xy|    = %.4e  (target ≈ %.4e = λ·γ̇_max)\n",
        bulk_max(Cxy_h), λ_visc * 4 * u_max / (Ny - 1))
@printf("max |C_xz|    = %.4e  (target ≈ %.4e = λ·γ̇_max)\n",
        bulk_max(Cxz_h), λ_visc * 4 * u_max / (Nz - 1))
@printf("max |C_yz|    = %.4e  (target ≈ 0)\n", bulk_max(Cyz_h))
@printf("max |C_zz - 1| = %.4e  (target ≈ 0)\n", bulk_max(Czz_h .- 1))
@printf("max |C_xx|    = %.4e  (target ≈ %.4e)\n",
        bulk_max(Cxx_h), 1 + 2 * (λ_visc * 4 * u_max / (Ny - 1))^2)

# Profile of C_xy(j) at i=i_s, k=kc — should be antisymmetric about j=Ny/2
println("\n--- C_xy(j) profile @ i=$i_s, k=$kc (should be antisymmetric, ≈ λγ̇) ---")
for j in 1:4:Ny
    @printf("  j=%-3d  u_x=%.5f  C_xy=%.4e  C_xz=%.4e\n",
            j, ux_h[i_s,j,kc], Cxy_h[i_s,j,kc], Cxz_h[i_s,j,kc])
end

# Profile of C_xz(k) at i=i_s, j=jc — should be antisymmetric about k=Nz/2
println("\n--- C_xz(k) profile @ i=$i_s, j=$jc (should be antisymmetric, ≈ λγ̇) ---")
for k in 1:4:Nz
    @printf("  k=%-3d  u_x=%.5f  C_xz=%.4e  C_xy=%.4e\n",
            k, ux_h[i_s,jc,k], Cxz_h[i_s,jc,k], Cxy_h[i_s,jc,k])
end

end  # close `let` block

println("\nDone.")
