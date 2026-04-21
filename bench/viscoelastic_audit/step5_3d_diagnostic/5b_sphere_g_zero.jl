# Step 5b — sphere 3D at Wi=0.1, R=16, with G=0 (so τ_p = G·(C−I) ≡ 0).
# This decouples the polymer stress from the flow entirely : the
# Hermite source is zero, the conformation LBM evolves purely passively.
# Cd MUST equal Cd_Newtonian to machine precision (or at least within
# 0.1%, accounting for f initialization / init-C differences).
#
# If ratio ≠ 1 here → bug in the solvent-side coupling, NOT in τ_p :
#   - polymer LBM still allocating memory and potentially corrupting f
#   - kernel-launch ordering issue
#   - Hermite source kernel being called even with τ_p = 0 (should be
#     no-op) but with a side effect
#
# If ratio = 1 (expected) → the bug is downstream, in the τ_p production.

using Kraken, Printf, CUDA

backend = CUDABackend()
FT = Float64

println("="^75)
println("Step 5b — sphere 3D R=16, Wi=0.1, G=0 (τ_p ≡ 0)")
println("Backend: $(typeof(backend)), GPU: $(CUDA.name(CUDA.device()))")
println("="^75)

R_s = 16
Nx = 24 * R_s; Ny = 4 * R_s; Nz = 4 * R_s
cx = 8 * R_s; cy = Ny ÷ 2; cz = Nz ÷ 2
Wi = 0.1; β = 0.5; u_in = 0.02
ν_total = u_in * (2 * R_s) / 1.0
ν_s = β * ν_total
λ = Wi * R_s / u_in
# G=0 → τ_p = G·(C−I) ≡ 0, no feedback on flow
m_OB_zero = OldroydB(G=0.0, λ=λ)

t0 = time()
ref = run_sphere_libb_3d(; Nx=Nx, Ny=Ny, Nz=Nz, radius=R_s,
                          cx=cx, cy=cy, cz=cz,
                          u_in=u_in, ν=ν_s, inlet=:uniform,
                          max_steps=20_000, avg_window=4_000,
                          backend=backend, T=FT)
t_newt = time() - t0
@printf("Newtonian (ν=ν_s=%.4f): Cd=%.4f (time=%.0fs)\n", ν_s, ref.Cd, t_newt)

t0 = time()
r = run_conformation_sphere_libb_3d(;
        Nx=Nx, Ny=Ny, Nz=Nz, radius=R_s, cx=cx, cy=cy, cz=cz,
        u_in=u_in, ν_s=ν_s,
        inlet=:uniform, ρ_out=1.0, tau_plus=1.0,
        polymer_bc=CNEBB(),
        polymer_model=m_OB_zero,
        max_steps=30_000, avg_window=6_000,
        backend=backend, FT=FT)
t_vis = time() - t0

ratio = r.Cd / ref.Cd
delta_pct = 100 * (ratio - 1)
@printf("G=0 pseudo-visco: Cd=%.4f ratio=%.4f delta=%.3f%% (time=%.0fs)\n",
        r.Cd, ratio, delta_pct, t_vis)
println()
println("EXPECTED : ratio ≈ 1.000 (within ~0.1%)")
println("  If |ratio−1| < 0.2% → coupling loop is clean")
println("  If |ratio−1| > 0.5% → bug in coupling unrelated to τ_p")
println()
println("NOTE : Newtonian reference uses ν=ν_s (solvent only, NOT ν_total),")
println("       matching the solvent viscosity in the visco run with G=0.")
