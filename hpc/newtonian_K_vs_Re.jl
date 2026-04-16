# Newtonian K = Cd·Re/2 vs Re for confined cylinder (B=0.5).
# Quick diagnostic to check whether our K(Re→0) matches Hulsen 2005: 132.36.
#
# Usage: julia --project=. hpc/newtonian_K_vs_Re.jl

using Kraken, Printf, CUDA

backend = CUDABackend()
FT = Float64

radius = 32; D = 2*radius; Ny = 4*radius; u_in = 0.02
Nx = 20*D

println("="^60)
println("Newtonian K vs Re (B=0.5, R=$radius, Nx=$Nx, Ny=$Ny)")
println("Backend: $(typeof(backend))")
println("="^60)

for Re_target in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 3.2]
    ν = u_in * D / Re_target
    max_steps = min(500_000, max(100_000, round(Int, 200_000 * (ν / 0.1))))
    avg_window = max_steps ÷ 5
    t0 = time()
    r = run_cylinder_2d(; Nx=Nx, Ny=Ny, radius=radius, u_in=u_in, ν=ν,
                          max_steps=max_steps, avg_window=avg_window,
                          backend=backend, T=FT)
    dt = time() - t0
    K = r.Cd * Re_target / 2
    @printf("Re=%5.2f  Cd=%10.3f  K=%8.3f  (ν=%.4f, steps=%d, %.1fs)\n",
            Re_target, r.Cd, K, ν, max_steps, dt)
end

println("\nLiterature: K(Re→0, B=0.5) = 132.36 (Hulsen 2005)")
println("Done.")
