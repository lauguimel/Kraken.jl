# 2D analog of duct_visco_diag.jl — same physical parameters, same BC
# topology (axis-aligned walls + Poiseuille inlet/outlet). Discriminates:
#
# - If 2D gives accurate C_xy(y) (≤5% deficit at the wall) → the duct
#   3D 22% deficit is a 3D-SPECIFIC bug in the streaming + collide loop
#   for non-uniform velocity in 3D.
# - If 2D gives ~20% deficit too → the deficit is intrinsic to the
#   conformation TRT scheme at this resolution + Wi for non-uniform
#   shear (not a 3D bug). Liu-cylinder Cd validation may be misleading
#   because Cd is integrated and averages out local C profile errors.
#
# Output: results/poiseuille_2d_analog_diag.txt

using Kraken, Printf, CUDA

backend = CUDABackend()
FT = Float64

println("="^70)
println("2D analog of duct viscoelastic test — Poiseuille channel, walls HBB")
println("Backend: $(typeof(backend)), GPU: $(CUDA.name(CUDA.device()))")
println("="^70)

H     = 30
Ny    = 2 * H
Nx    = 6 * H
β     = 0.59
u_max = 0.04
ν_total = u_max * H / 1.0
ν_s   = β * ν_total
ν_p   = (1 - β) * ν_total
Wi    = 0.1
λ_visc = Wi * H / u_max
G     = ν_p / λ_visc

# Use u_mean = (2/3)·u_max (Schäfer-Turek convention) so the
# `:parabolic` inlet of run_conformation_cylinder_libb_2d gives
# centerline u = u_max.
u_mean = (2/3) * u_max

max_steps = 100_000
avg_window = max_steps ÷ 5

@printf("Geometry  : Nx=%d, Ny=%d, H=%d (channel walls j=1, j=%d)\n", Nx, Ny, H, Ny)
@printf("Flow      : u_max=%.3g (centerline), β=%.2f, λ=%.3g, Wi=0.1\n",
        u_max, β, λ_visc)
@printf("LBM       : steps=%d, avg_window=%d\n\n", max_steps, avg_window)

println("--- Running 2D Poiseuille viscoelastic ---")
t0 = time()
m_OB = OldroydB(G=FT(G), λ=FT(λ_visc))
r2d = run_conformation_cylinder_libb_2d(;
        Nx=Nx, Ny=Ny, radius=1, cx=-100, cy=Ny÷2,    # cylinder far outside domain
        u_mean=u_mean, ν_s=ν_s,
        polymer_model=m_OB, polymer_bc=CNEBB(),
        inlet=:parabolic, ρ_out=1.0, tau_plus=1.0,
        max_steps=max_steps, avg_window=avg_window,
        backend=backend, FT=FT)
@printf("2D done in %.0f s\n", time() - t0)

# Sample at i_sample = 3*Nx/4 (downstream, no obstacle effect)
i_sample = 3 * Nx ÷ 4
ux2d  = r2d.ux[i_sample, :]
Cxy2d = r2d.C_xy[i_sample, :]
Cxx2d = r2d.C_xx[i_sample, :]
Cyy2d = r2d.C_yy[i_sample, :]

# Analytical Poiseuille Oldroyd-B
H_chan = FT(Ny)
ux_an  = zeros(Float64, Ny)
γ̇_an  = zeros(Float64, Ny)
Cxy_an = zeros(Float64, Ny)
Cxx_an = zeros(Float64, Ny)
for j in 1:Ny
    y = FT(j) - FT(0.5)
    ux_an[j]  = 4 * u_max * y * (H_chan - y) / H_chan^2
    γ̇_an[j]   = 4 * u_max * (H_chan - 2*y) / H_chan^2
    Cxy_an[j] = λ_visc * γ̇_an[j]
    Cxx_an[j] = 1 + 2 * (λ_visc * γ̇_an[j])^2
end

println("\n--- Profile @ i=$(i_sample) ---")
@printf("%-4s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n",
        "j", "ux_an", "ux_2d", "Cxy_an", "Cxy_2d", "Cxx_an", "Cxx_2d", "rel_Cxy")
for j in 1:4:Ny
    rel = Cxy_an[j] != 0 ? abs(Cxy2d[j] - Cxy_an[j]) / abs(Cxy_an[j]) : 0.0
    @printf("%-4d %-10.5f %-10.5f %-10.4e %-10.4e %-10.4f %-10.4f %-10.4f\n",
            j, ux_an[j], ux2d[j], Cxy_an[j], Cxy2d[j],
            Cxx_an[j], Cxx2d[j], rel)
end

# Bulk metrics: skip 3 cells from each wall
function bulk_metrics(num, ana, label)
    j1, j2 = 4, length(num) - 3
    err_max = maximum(abs.(num[j1:j2] .- ana[j1:j2]))
    norm_ana = max(maximum(abs.(ana[j1:j2])), 1e-12)
    @printf("  %-12s : max_err=%.4e  rel=%.4f\n", label, err_max, err_max/norm_ana)
end

println("\n--- 2D bulk error vs analytical (skip 3 wall cells) ---")
bulk_metrics(ux2d, ux_an, "u_x")
bulk_metrics(Cxy2d, Cxy_an, "C_xy")
bulk_metrics(Cxx2d, Cxx_an, "C_xx")

# Wall-row C_xy comparison (the key quantity from duct test)
println("\n--- Wall-row C_xy comparison (j=1 + j=2) ---")
for j in (1, 2, Ny-1, Ny)
    rel = Cxy_an[j] != 0 ? abs(Cxy2d[j] - Cxy_an[j]) / abs(Cxy_an[j]) : 0.0
    @printf("  j=%-3d  C_xy_2d = %.4e  C_xy_an = %.4e  rel_err = %.3f\n",
            j, Cxy2d[j], Cxy_an[j], rel)
end

# Maximum |C_xy| anywhere in the bulk
max_Cxy_2d = maximum(abs.(Cxy2d))
max_Cxy_an = maximum(abs.(Cxy_an))
@printf("\n  max|C_xy_2d| = %.4e  max|C_xy_an| = %.4e  ratio = %.4f\n",
        max_Cxy_2d, max_Cxy_an, max_Cxy_2d/max_Cxy_an)
@printf("  (3D duct gave: max|C_xy| = 0.174 vs target 0.203, ratio = 0.86)\n")

println("\nDone.")
