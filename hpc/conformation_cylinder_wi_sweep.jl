# Wi sweep — conformation TRT-LBM cylinder vs Alves (2001) / Hulsen (2005).
#
# Standard Oldroyd-B confined cylinder benchmark:
#   β = 0.59, B = D/(2H) = 0.5, creeping flow (Re ≈ 0.32)
# Fixed lattice (R=32), sweep Wi by varying relaxation time λ = Wi·R/u_in.
# Computes drag coefficient Cd and the dimensionless group K = Cd·Re/2.
#
# Usage: julia --project=. hpc/conformation_cylinder_wi_sweep.jl

using Kraken, Printf
using CUDA

const backend = CUDABackend()
const FT      = Float64

# --- Fixed geometry & physics ---
const R       = 32
const D       = 2R
const Nx      = 20 * D          # = 1280
const Ny      = 4 * R           # = 128  (B = D/(2H) = 0.5)
const u_in    = 0.005           # creeping flow
const ν_total = 0.1
const β       = 0.59
const ν_s     = β * ν_total
const ν_p     = (1 - β) * ν_total
const Re      = u_in * D / ν_total

const max_steps  = 200_000
const avg_window = 50_000

const Wi_values = (0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7)

println("=" ^ 70)
println("Conformation cylinder Wi sweep (Oldroyd-B, β=$β, B=0.5)")
println("  Nx=$Nx, Ny=$Ny, radius=$R, D=$D")
println("  Re = $Re,  u_in = $u_in")
println("  ν_s = $ν_s,  ν_p = $ν_p,  ν_total = $ν_total")
println("  max_steps = $max_steps, avg_window = $avg_window")
println("  Backend: $(typeof(backend))")
println("=" ^ 70)

if backend isa CUDABackend
    println("GPU: ", CUDA.name(CUDA.device()))
end

results = []
diverged = false

for Wi in Wi_values
    global diverged
    diverged && break

    λ = Wi * R / u_in
    println("\n>>> Wi = $Wi  (λ = $(round(λ, digits=1))) ...")

    t0 = time()
    Cd = if Wi == 0.0
        ref = run_cylinder_2d(; Nx=Nx, Ny=Ny, radius=R, u_in=u_in, ν=ν_total,
                                max_steps=max_steps, avg_window=avg_window,
                                backend=backend, T=FT)
        ref.Cd
    else
        res = run_conformation_cylinder_2d(; Nx=Nx, Ny=Ny, radius=R, u_in=u_in,
                                             ν_s=ν_s, ν_p=ν_p, lambda=λ, tau_plus=1.0,
                                             max_steps=max_steps, avg_window=avg_window,
                                             backend=backend, FT=FT)
        res.Cd_s
    end
    dt = time() - t0
    K = Cd * Re / 2
    @printf("    Cd = %.4f   K = %.4f   time = %.1fs\n", Cd, K, dt)
    push!(results, (Wi=Wi, Cd=Cd, K=K, time=dt))

    if isnan(Cd) || isinf(Cd)
        println("  ⚠ DIVERGED — stopping sweep")
        diverged = true
    end
end

# --- Summary table ---
Cd0 = isempty(results) ? NaN : results[1].Cd
println("\n" * "=" ^ 70)
println("RESULTS: Cd vs Wi (Oldroyd-B, β=$β, B=0.5, Re=$Re)")
println("  Nx=$Nx, Ny=$Ny, radius=$R")
println("-" ^ 70)
@printf("%-6s %-12s %-12s %-12s %-10s\n", "Wi", "Cd", "K=Cd·Re/2", "Cd/Cd(0)", "time(s)")
println("-" ^ 70)
for r in results
    ratio = r.Cd / Cd0
    @printf("%-6.3f %-12.4f %-12.4f %-12.4f %-10.1f\n", r.Wi, r.Cd, r.K, ratio, r.time)
end
println("=" ^ 70)

# --- Literature reference (Hulsen et al. 2005, Alves et al. 2001) ---
# Oldroyd-B, β = 0.59, B = 0.5, creeping flow
println("\nLiterature (Hulsen et al. 2005, B=0.5, β=0.59):")
println("-" ^ 70)
@printf("%-6s %-12s %-12s %-12s\n", "Wi", "K_lit", "K_sim", "rel.err%")
println("-" ^ 70)
const K_lit = Dict(0.0 => 132.36, 0.5 => 130.36, 0.6 => 129.72, 0.7 => 129.15)
for r in results
    if haskey(K_lit, r.Wi)
        klit = K_lit[r.Wi]
        err = 100 * (r.K - klit) / klit
        @printf("%-6.3f %-12.4f %-12.4f %-+12.3f\n", r.Wi, klit, r.K, err)
    end
end
println("=" ^ 70)

println("\nExpected: K decreases monotonically with Wi (drag reduction).")
println("Done.")
