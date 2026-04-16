# Grid convergence of K = Cd·Re/2 for Newtonian cylinder at B=0.5, Re=1
# Verifies basic cylinder drag against Hulsen 2005: K = 132.36
#
# Usage: julia --project=. hpc/newtonian_K_convergence.jl

using Kraken, Printf, CUDA
backend = CUDABackend()
FT = Float64

u_in = 0.02; Re = 1.0

println("="^60)
println("Newtonian K convergence (B=0.5, Re=$Re, u_in=$u_in)")
println("Backend: $(typeof(backend)), GPU: $(CUDA.name(CUDA.device()))")
println("="^60)

for R in [8, 16, 24, 32, 48, 64]
    D = 2R; Ny = 4R; Nx = 20*D
    ν = u_in * D / Re
    ω = 1 / (3ν + 0.5)
    steps = max(100_000, round(Int, 100_000 * (R/8)^2))
    steps = min(steps, 500_000)
    avg = steps ÷ 5
    t0 = time()
    r = run_cylinder_2d(; Nx=Nx, Ny=Ny, radius=R, u_in=u_in, ν=ν,
                          max_steps=steps, avg_window=avg,
                          backend=backend, T=FT)
    dt = time() - t0
    K = r.Cd * Re / 2
    @printf("R=%3d  Ny=%4d  Nx=%5d  ν=%.4f  ω=%.6f  Cd=%8.3f  K=%8.3f  (%.1fs)\n",
            R, Ny, Nx, ν, ω, r.Cd, K, dt)
end

println("\nHulsen 2005 ref: K(Re→0, B=0.5) = 132.36")
println("Done.")
