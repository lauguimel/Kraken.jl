# Mesh convergence at Wi = 0.1 for confined cylinder Oldroyd-B.
# Extends the Liu Table 3 comparison from R ∈ {20, 30} to {20, 30, 40, 48}.
# Expected: err% decreases monotonically → 0 as R → ∞.
#
# Liu Table 3 (CNEBB, Sc=10⁴):
#   R=20  Cd = 129.42
#   R=30  Cd = 130.36
#   R=40  Cd = 129.42  (Liu Table 3 Wi=0.1 col, R=40 row)
#   R=48  Cd ~ 130.8 (converged)
#
# Usage: julia --project=. hpc/liu_R_convergence.jl

using Kraken, Printf, CUDA

backend = CUDABackend()
FT = Float64

println("="^70)
println("Mesh convergence at Wi = 0.1 (β=0.59, Re=1)")
println("Backend: $(typeof(backend)), GPU: $(CUDA.name(CUDA.device()))")
println("="^70)

β = 0.59; u_mean = 0.02
liu_ref_R = Dict(20 => 129.42, 25 => 129.61, 30 => 130.36, 35 => 130.77,
                  40 => 130.79, 48 => 130.83)

@printf("%-6s %-6s %-6s %-10s %-10s %-8s %-8s\n",
        "R", "Nx", "Ny", "Cd_sim", "Cd_Liu", "err%", "time")
println("-"^60)

for R in [20, 30, 40, 48]
    Nx = 30 * R; Ny = 4 * R; cx = 15 * R; cy = 2 * R
    ν_total = u_mean * R / 1.0
    ν_s = β * ν_total
    ν_p = (1 - β) * ν_total
    λ = 0.1 * R / u_mean          # Wi = 0.1
    max_steps = round(Int, 200_000 * (R / 30)^2)
    max_steps = min(max_steps, 500_000)
    avg_window = max_steps ÷ 5

    model = OldroydB(G=ν_p/λ, λ=λ)

    t0 = time()
    r = run_conformation_cylinder_libb_2d(;
            Nx=Nx, Ny=Ny, radius=R, cx=cx, cy=cy,
            u_mean=u_mean, ν_s=ν_s,
            polymer_model=model, polymer_bc=CNEBB(),
            inlet=:parabolic, ρ_out=1.0, tau_plus=1.0,
            max_steps=max_steps, avg_window=avg_window,
            backend=backend, FT=FT)
    dt = time() - t0
    ref = get(liu_ref_R, R, NaN)
    err = isnan(ref) ? NaN : (r.Cd - ref) / ref * 100
    @printf("%-6d %-6d %-6d %-10.3f %-10.3f %-8.2f %-8.0fs\n",
            R, Nx, Ny, r.Cd, ref, err, dt)
end

println("\nDone.")
