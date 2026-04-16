# Liu et al. 2025 cylinder benchmark — reproduction with Kraken.jl.
#
# Uses `run_conformation_cylinder_libb_2d`:
#   - Fused TRT + Bouzidi LI-BB V2 for the solvent flow (curved cylinder)
#   - Modular BCSpec: ZouHeVelocity(Poiseuille) inlet + ZouHePressure outlet
#   - Mei-consistent MEA drag on cut-link q_wall
#   - TRT conformation LBM + CNEBB + Hermite stress source
#
# Liu setup (Table 3, CNEBB, Sc=10⁴):
#   - Domain 30R × 4R, cylinder at (15R, 2R), B = 0.5
#   - Re = 1, β = 0.59
#   - Re = U_avg · R / ν_total (L_c = R, NOT D)
#   - Cd = Fx / (0.5 ρ U_avg² D)   — Liu Eq 64
#
# Reference values (R=30, CNEBB, Sc=10⁴):
#   Wi=0.1 → Cd ≈ 130.36
#   Wi=0.5 → Cd ≈ 126.31
#   Wi=1.0 → Cd ≈ 151.31
#
# Usage: julia --project=. hpc/liu_cylinder_benchmark.jl

using Kraken, Printf, CUDA

backend = CUDABackend()
FT = Float64

println("="^70)
println("Liu et al. 2025 cylinder benchmark — LI-BB V2 driver")
println("Backend: $(typeof(backend)), GPU: $(CUDA.name(CUDA.device()))")
println("="^70)

# Liu Table 3 reference values (CNEBB, Sc=10⁴)
liu_ref = Dict(
    (20, 0.1) => 129.42, (20, 0.5) => 125.17, (20, 1.0) => 164.26,
    (30, 0.1) => 130.36, (30, 0.5) => 126.31, (30, 1.0) => 151.31,
)

for R in [20, 30]
    # Liu uses Re = U_avg · R / ν → solve for ν_total given u_mean, R, Re
    # Then split ν_s, ν_p with β = 0.59
    Re_target = 1.0
    β = 0.59
    u_mean = 0.02   # keeps ω_s in [0.5, 1.5] safely
    ν_total = u_mean * R / Re_target
    ν_s = β * ν_total
    ν_p = (1 - β) * ν_total
    Nx = 30 * R
    Ny = 4 * R
    cx = 15 * R
    cy = 2 * R

    println("\n>>> R = $R  (Nx=$Nx, Ny=$Ny, ν_s=$ν_s, ν_p=$ν_p)")
    @printf("%-6s %-10s %-10s %-8s\n", "Wi", "Cd_sim", "Cd_Liu", "err%")
    println("-"^38)

    for Wi in [0.001, 0.1, 0.5, 1.0]
        # Liu: Wi = λ · U_c / R with U_c = U_avg = u_mean
        λ = Wi * R / u_mean
        max_steps = Wi < 0.01 ? 100_000 : 200_000
        avg_window = max_steps ÷ 5

        t0 = time()
        r = run_conformation_cylinder_libb_2d(;
                Nx=Nx, Ny=Ny, radius=R, cx=cx, cy=cy,
                u_mean=u_mean, ν_s=ν_s, ν_p=ν_p, lambda=λ, tau_plus=1.0,
                inlet=:parabolic, ρ_out=1.0,
                max_steps=max_steps, avg_window=avg_window,
                backend=backend, FT=FT)
        dt = time() - t0

        ref = get(liu_ref, (R, Wi), NaN)
        err = isnan(ref) ? NaN : (r.Cd - ref) / ref * 100
        @printf("%-6.3f %-10.3f %-10.3f %-8.2f  (%.0fs)\n",
                Wi, r.Cd, ref, err, dt)
    end
end

println("\nDone.")
