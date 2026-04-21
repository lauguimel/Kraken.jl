# Step 5a — sphere 3D at Wi=0.1, R=16, with NoPolymerWallBC() in place
# of CNEBB. Isolates whether CNEBB 3D is the source of the order-0.19
# convergence failure.
#
# Comparison baseline (step 4, CNEBB): ratio = 0.892 (deficit 10.8%).
# If 5a gives ratio closer to 1.0 → CNEBB is the culprit.
# If 5a gives same ratio ~0.89 → CNEBB is NOT the main cause; look at
# Hermite source or other 3D-specific code.

using Kraken, Printf, CUDA

backend = CUDABackend()
FT = Float64

println("="^75)
println("Step 5a — sphere 3D R=16, Wi=0.1, NoPolymerWallBC()")
println("Backend: $(typeof(backend)), GPU: $(CUDA.name(CUDA.device()))")
println("="^75)

R_s = 16
Nx = 24 * R_s; Ny = 4 * R_s; Nz = 4 * R_s
cx = 8 * R_s; cy = Ny ÷ 2; cz = Nz ÷ 2
Wi = 0.1; β = 0.5; u_in = 0.02
ν_total = u_in * (2 * R_s) / 1.0
ν_s = β * ν_total; ν_p = (1 - β) * ν_total
λ = Wi * R_s / u_in
m_OB = OldroydB(G=ν_p/λ, λ=λ)

# Newtonian baseline
t0 = time()
ref = run_sphere_libb_3d(; Nx=Nx, Ny=Ny, Nz=Nz, radius=R_s,
                          cx=cx, cy=cy, cz=cz,
                          u_in=u_in, ν=ν_total, inlet=:uniform,
                          max_steps=20_000, avg_window=4_000,
                          backend=backend, T=FT)
t_newt = time() - t0
@printf("Newtonian: Cd=%.3f (time=%.0fs)\n", ref.Cd, t_newt)

# Viscoelastic with NoPolymerWallBC
t0 = time()
r = run_conformation_sphere_libb_3d(;
        Nx=Nx, Ny=Ny, Nz=Nz, radius=R_s, cx=cx, cy=cy, cz=cz,
        u_in=u_in, ν_s=ν_s,
        inlet=:uniform, ρ_out=1.0, tau_plus=1.0,
        polymer_bc=NoPolymerWallBC(),
        polymer_model=m_OB,
        max_steps=30_000, avg_window=6_000,
        backend=backend, FT=FT)
t_vis = time() - t0

ratio = r.Cd / ref.Cd
@printf("Viscoelastic (NoPolymerWallBC): Cd=%.3f ratio=%.4f deficit=%.2f%% (time=%.0fs)\n",
        r.Cd, ratio, 100*(1-ratio), t_vis)
@printf("\nBaseline step4 (CNEBB): ratio=0.8920 deficit=10.80%%\n")
@printf("If this ratio is closer to 1.0 → CNEBB 3D is the main bias source.\n")
@printf("If same ratio ~0.89 → bug is NOT in CNEBB; investigate Hermite source 3D.\n")
