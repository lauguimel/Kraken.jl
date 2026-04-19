# Poiseuille viscoelastic — 2D vs 3D vs analytical (Oldroyd-B).
#
# Diagnostic test for the 3D conformation/Hermite kernels with WALLS,
# but no u↔C feedback (Poiseuille has ∂τ_p/∂x = 0 → no body force on u).
# This is intermediate between the pure-shear-no-walls test (test_simple_
# shear_3d.jl, validated) and the full coupled cylinder test
# (cylinder_extruded_diag.jl). It tests stream + CNEBB at walls + collide
# + Hermite source effect on f at non-uniform shear.
#
# Setup: pressure-driven Poiseuille via inlet/outlet BC.
# - 2D: Nx=Ny=parameterised, walls halfway-BB at j=1, j=Ny.
# - 3D: extra Nz, periodic in z (so it's truly z-uniform Poiseuille).
#       Use the existing run_conformation_sphere_libb_3d driver with
#       sphere placed FAR outside the domain (cx=cy=cz=−Nx) so the
#       is_solid mask is empty and only the channel walls remain.
#
# Analytical (Oldroyd-B steady):
#   u(y) = u_max · 4·y·(H − y) / H², where H = Ny − 1 (centre of the
#         halfway-BB walls so the distance between walls is Ny − 1)
#   γ̇(y) = du/dy = u_max · 4·(H − 2y) / H²
#   C_xy(y) = λ · γ̇(y)              — linear in y, antisymmetric
#   C_xx(y) = 1 + 2·(λ·γ̇(y))²       — parabolic
#   C_yy = C_zz = 1, C_xz = C_yz = 0
#
# Output: results/poiseuille_visco_diag.txt with side-by-side profiles.

using Kraken, Printf, CUDA

backend = CUDABackend()
FT = Float64

println("="^70)
println("Poiseuille viscoelastic 2D vs 3D vs analytical (Oldroyd-B)")
println("Backend: $(typeof(backend)), GPU: $(CUDA.name(CUDA.device()))")
println("="^70)

R     = 30          # half-channel height (matches 2D Liu Wi sweep convention)
Ny    = 4 * R       # full channel
Nx    = 6 * R       # short streamwise (no obstacle)
Nz_3d = 8           # thin in z (we want z-uniform)
β     = 0.59
u_mean = 0.02
ν_total = u_mean * R / 1.0      # Re_local = u_mean·R/ν = 1
ν_s = β * ν_total
ν_p = (1 - β) * ν_total
Wi  = 0.1
λ   = Wi * R / u_mean
G   = ν_p / λ
max_steps = 50_000
avg_window = max_steps ÷ 5

@printf("Geometry  : Nx=%d, Ny=%d, Nz_3d=%d, R=%d\n", Nx, Ny, Nz_3d, R)
@printf("Flow       : u_mean=%.3g, β=%.2f, ν_s=%.3g, ν_p=%.3g, λ=%.3g, Wi=%.2f\n",
        u_mean, β, ν_s, ν_p, λ, Wi)
@printf("LBM        : steps=%d, avg_window=%d\n\n", max_steps, avg_window)

# ----------------------------------------------------------------------
# 2D Poiseuille via run_conformation_cylinder_libb_2d with cylinder OUT
# ----------------------------------------------------------------------
println("--- Running 2D Poiseuille viscoelastic ---")
t0 = time()
m_OB = OldroydB(G=FT(G), λ=FT(λ))
r2d = run_conformation_cylinder_libb_2d(;
        Nx=Nx, Ny=Ny, radius=1, cx=-100, cy=Ny÷2,    # cylinder far outside
        u_mean=u_mean, ν_s=ν_s,
        polymer_model=m_OB, polymer_bc=CNEBB(),
        inlet=:parabolic, ρ_out=1.0, tau_plus=1.0,
        max_steps=max_steps, avg_window=avg_window,
        backend=backend, FT=FT)
@printf("2D done in %.0f s\n", time() - t0)

# Sample at i = 3*Nx/4 (downstream, no obstacle effect)
i_sample_2d = 3 * Nx ÷ 4
ux2d = r2d.ux[i_sample_2d, :]
Cxy2d = r2d.C_xy[i_sample_2d, :]
Cxx2d = r2d.C_xx[i_sample_2d, :]
Cyy2d = r2d.C_yy[i_sample_2d, :]

# ----------------------------------------------------------------------
# 3D Poiseuille via run_conformation_sphere_libb_3d with sphere OUT
# ----------------------------------------------------------------------
println("\n--- Running 3D Poiseuille viscoelastic ---")
t0 = time()
m_OB3 = OldroydB(G=FT(G), λ=FT(λ))
r3d = run_conformation_sphere_libb_3d(;
        Nx=Nx, Ny=Ny, Nz=Nz_3d, radius=1, cx=-100, cy=Ny÷2, cz=Nz_3d÷2,
        u_in=1.5*u_mean,                          # parabolic_y centerline = u_max
        ν_s=ν_s,
        polymer_model=m_OB3, polymer_bc=CNEBB(),
        inlet=:parabolic_y, ρ_out=1.0, tau_plus=1.0,
        max_steps=max_steps, avg_window=avg_window,
        backend=backend, FT=FT)
@printf("3D done in %.0f s\n", time() - t0)

# Sample at i = 3*Nx/4, k = Nz_3d/2 (mid-z plane)
i_sample_3d = 3 * Nx ÷ 4
k_sample_3d = Nz_3d ÷ 2
ux3d = r3d.ux[i_sample_3d, :, k_sample_3d]
Cxy3d = r3d.C_xy[i_sample_3d, :, k_sample_3d]
Cxx3d = r3d.C_xx[i_sample_3d, :, k_sample_3d]
Cyy3d = r3d.C_yy[i_sample_3d, :, k_sample_3d]
Cxz3d = r3d.C_xz[i_sample_3d, :, k_sample_3d]
Cyz3d = r3d.C_yz[i_sample_3d, :, k_sample_3d]
Czz3d = r3d.C_zz[i_sample_3d, :, k_sample_3d]

# ----------------------------------------------------------------------
# Analytical (Poiseuille Oldroyd-B), use H = Ny − 1 (cell-centre wall
# convention with halfway-BB at j=1 and j=Ny → centres at y=0 and y=Ny−1
# bisected by walls at y=−0.5, y=Ny−0.5, so flow domain has H=Ny). Use
# u(y_c) where y_c = j − 0.5 to centre on cell midpoints.
# ----------------------------------------------------------------------
H_chan = FT(Ny)
u_max = FT(1.5) * u_mean   # u_max = (3/2) · u_mean for parabolic
ux_an  = zeros(Float64, Ny)
γ̇_an  = zeros(Float64, Ny)
Cxy_an = zeros(Float64, Ny)
Cxx_an = zeros(Float64, Ny)
for j in 1:Ny
    y = FT(j) - FT(0.5)
    ux_an[j]  = 4 * u_max * y * (H_chan - y) / H_chan^2
    γ̇_an[j]   = 4 * u_max * (H_chan - 2*y) / H_chan^2
    Cxy_an[j] = λ * γ̇_an[j]
    Cxx_an[j] = 1 + 2 * (λ * γ̇_an[j])^2
end

# ----------------------------------------------------------------------
# Print profiles + compute errors (skip the 2 rows next to walls — CNEBB
# discretisation makes those slightly off as expected).
# ----------------------------------------------------------------------
println("\n--- Profile comparison @ i_sample (centerline of obstacle-free channel) ---")
@printf("%-4s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n",
        "j", "u_an", "u_2d", "u_3d", "Cxy_an", "Cxy_2d", "Cxy_3d",
        "Cxx_an", "Cxx_2d", "Cxx_3d")
for j in 1:Ny
    @printf("%-4d %-10.5f %-10.5f %-10.5f %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f\n",
            j, ux_an[j], ux2d[j], ux3d[j],
            Cxy_an[j], Cxy2d[j], Cxy3d[j],
            Cxx_an[j], Cxx2d[j], Cxx3d[j])
end

# Bulk-only metrics: skip 3 cells near each wall
function bulk_metrics(num, ana, label)
    j1, j2 = 4, length(num) - 3
    err_max = maximum(abs.(num[j1:j2] .- ana[j1:j2]))
    norm_ana = max(maximum(abs.(ana[j1:j2])), 1e-12)
    @printf("  %-12s : max_err=%.4e  rel=%.4f\n", label, err_max, err_max/norm_ana)
end

println("\n--- 2D bulk error vs analytical ---")
bulk_metrics(ux2d, ux_an, "u_x")
bulk_metrics(Cxy2d, Cxy_an, "C_xy")
bulk_metrics(Cxx2d, Cxx_an, "C_xx")

println("--- 3D bulk error vs analytical ---")
bulk_metrics(ux3d, ux_an, "u_x")
bulk_metrics(Cxy3d, Cxy_an, "C_xy")
bulk_metrics(Cxx3d, Cxx_an, "C_xx")

println("--- 3D out-of-plane (should be ≈ 0 or ≈ 1) ---")
@printf("  max |C_xz|     = %.4e\n", maximum(abs.(Cxz3d)))
@printf("  max |C_yz|     = %.4e\n", maximum(abs.(Cyz3d)))
@printf("  max |C_zz - 1| = %.4e\n", maximum(abs.(Czz3d .- 1)))

println("\n--- 3D vs 2D (should match) ---")
bulk_metrics(ux3d, ux2d, "u_x diff")
bulk_metrics(Cxy3d, Cxy2d, "C_xy diff")
bulk_metrics(Cxx3d, Cxx2d, "C_xx diff")

println("\nDone.")
