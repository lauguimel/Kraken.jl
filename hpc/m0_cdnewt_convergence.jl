# M0 — Cd_Newt convergence for the LI-BB cylinder driver at B=0.5, Re=1.
#
# Mirrors step5d's EXACT setup so the Cd_Newt numbers are directly comparable:
#   * run_cylinder_libb_2d (NOT run_cylinder_2d staircase)
#   * Nx = 30R, Ny = 4R, cx = 15R, cy = 2R
#   * u_mean = 0.02  (u_in = 3*u_mean/2 = 0.03 for :parabolic inlet)
#   * Re = u_mean * R / ν = 1  (Liu convention, char length = R)
#   * Cd normalisation computed in compute_drag_libb (Liu Eq 64)
#
# Canonical Newtonian reference at B=0.5 creeping:
#   Hulsen 2005 (K = Cd·Re/2) : K = 132.36
#   Kumar 2024 (arxiv 2403.05904) Table 2 at De=0.025 : Cd = 132.09
#   Dou & Phan-Thien 1999 : Cd_Newt = 131.5
#
# Kraken step5d gave Cd_Newt=142.87 at R=30, +8% above canonical.
# This script tests whether that is grid bias (R too small) by going
# to R=60 and R=120.

using Kraken, Printf, CUDA, KernelAbstractions

backend = CUDABackend()
FT = Float64

u_mean = 0.02
Re_target = 1.0   # Liu convention : Re = u_mean * R / ν_total

println("="^75)
println("M0 — Cd_Newt convergence, LI-BB cylinder, B=0.5, Re=1, Float64")
@printf("Backend=%s  GPU=%s  FT=%s\n", typeof(backend), CUDA.name(CUDA.device()), FT)
println("="^75)
@printf("%-5s %-6s %-6s %-10s %-10s %-10s %-10s %-10s\n",
        "R", "Nx", "Ny", "ν_total", "Cd_Newt", "K", "steps", "time")
println("-"^75)

results = Tuple{Int, Float64, Float64, Float64}[]

for R in [30, 60, 120]
    Nx = 30*R; Ny = 4*R; cx = 15*R; cy = 2*R
    ν_total = u_mean * R / Re_target
    # Advection time ≈ Nx/u_mean = 30R/0.02 = 1500R steps.
    # Run at least 3× advection + 1× diffusion (R²/ν = 1/0.02 * R = 50R).
    adv = round(Int, Nx / u_mean)
    max_steps = max(60_000, 3 * adv)
    avg_window = max_steps ÷ 4

    t0 = time()
    r = try
        run_cylinder_libb_2d(; Nx=Nx, Ny=Ny, cx=cx, cy=cy, radius=R,
                               u_in=3*u_mean/2, ν=ν_total,
                               inlet=:parabolic,
                               max_steps=max_steps, avg_window=avg_window,
                               backend=backend, T=FT)
    catch err
        @warn "R=$R failed" err
        (; Cd=NaN)
    end
    dt = time() - t0

    K = r.Cd * Re_target / 2
    @printf("%-5d %-6d %-6d %-10.4f %-10.4f %-10.4f %-10d %-8.0fs\n",
            R, Nx, Ny, ν_total, r.Cd, K, max_steps, dt)
    flush(stdout)
    push!(results, (R, ν_total, r.Cd, K))
end

println()
println("Canonical Newtonian refs at B=0.5, Re→0:")
println("  Hulsen 2005           K = 132.36  (so Cd ≈ 264.72 at Re=1 in Hulsen's conv;")
println("                                     i.e. Cd = 132.36 if K = Cd·Re/2 with Re=1)")
println("  Kumar 2024 (De=0.025) Cd = 132.09")
println("  Dou & Phan-Thien 1999 Cd = 131.50")
println("  Liu 2025 (β=0.59 Wi=0.1 ≈ quasi-Newton) Cd = 130.83 (R=48)")
println()

println("Richardson extrapolation Cd_Newt(R→∞) (assuming 2nd-order convergence)")
if length(results) >= 3
    R1, ν1, Cd1, K1 = results[1]
    R2, ν2, Cd2, K2 = results[2]
    R3, ν3, Cd3, K3 = results[3]
    # p = log((Cd1 - Cd2)/(Cd2 - Cd3)) / log(R2/R1)  -- R-ratio based
    denom1 = Cd1 - Cd2
    denom2 = Cd2 - Cd3
    if abs(denom2) > 1e-9
        ratio = denom1 / denom2
        p = log(abs(ratio)) / log(R2/R1)
        # Richardson for R3→∞: Cd_inf ≈ Cd3 - (Cd2 - Cd3) / ((R3/R2)^p - 1)
        Rratio = R3/R2
        Cd_inf = Cd3 - (Cd2 - Cd3) / (Rratio^p - 1)
        @printf("  measured order p ≈ %.2f,  Cd_Newt(R→∞) ≈ %.3f\n", p, Cd_inf)
    else
        println("  cannot extrapolate (Cd2 ≈ Cd3)")
    end
end

println()
println("Decision (M0 per DEBUG_PLAN.md):")
println("  Cd_Newt(R=120) ∈ [130, 134] → reference is fine, proceed to M2 β-sweep")
println("  Cd_Newt(R=120) > 135 OR < 128 → Newtonian reference broken, root-cause")
println("="^75)
