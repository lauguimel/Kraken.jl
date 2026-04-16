# Spatial convergence — conformation TRT-LBM cylinder vs Newtonian reference.
#
# Same physical setup (Re, Wi, β) — only the lattice resolution changes.
# Reports |Cd_s - Cd_ref| / Cd_ref and the observed order.
#
# Validates the Newtonian limit of the Liu et al. 2025 TRT-LBM scheme:
# at Wi → 0, MEA drag on f should converge to the Newtonian Cd at ν_total.
#
# Usage: julia --project=. hpc/conformation_cylinder_convergence.jl

using Kraken, Printf
using CUDA

const backend = CUDABackend()
const FT      = Float64

# --- Physical constants (fixed across resolutions) ---
const Re_target = 3.2
const Wi_target = 0.0125
const β         = 0.6
const aspect_x  = 30      # Nx / R
const aspect_y  = 7.5     # Ny / R
const u_in      = 0.02

function run_one(R)
    Nx = round(Int, aspect_x * R)
    Ny = round(Int, aspect_y * R)
    D  = 2R
    ν_total = u_in * D / Re_target
    ν_s     = β * ν_total
    ν_p     = ν_total - ν_s
    λ       = Wi_target * R / u_in
    # Steps ∝ R² to keep diffusive time fixed
    max_steps  = max(20_000, round(Int, 20_000 * (R/8)^2))
    avg_window = max(2_000,  round(Int, max_steps ÷ 10))

    println("\n>>> R = $R  (Nx=$Nx, Ny=$Ny, steps=$max_steps)")

    t0 = time()
    ref = run_cylinder_2d(; Nx=Nx, Ny=Ny, radius=R, u_in=u_in, ν=ν_total,
                            max_steps=max_steps, avg_window=avg_window,
                            backend=backend, T=FT)
    t_ref = time() - t0

    t0 = time()
    res = run_conformation_cylinder_2d(; Nx=Nx, Ny=Ny, radius=R, u_in=u_in,
                                         ν_s=ν_s, ν_p=ν_p, lambda=λ, tau_plus=1.0,
                                         max_steps=max_steps, avg_window=avg_window,
                                         backend=backend, FT=FT)
    t_ve = time() - t0

    err = abs(res.Cd_s - ref.Cd) / ref.Cd
    @printf("    Cd_ref=%.4f  Cd_s=%.4f  Cd_p=%.4f  err=%.3f%%  (ref %.1fs / ve %.1fs)\n",
            ref.Cd, res.Cd_s, res.Cd_p, 100*err, t_ref, t_ve)

    return (R=R, Nx=Nx, Ny=Ny, steps=max_steps,
            Cd_ref=ref.Cd, Cd_s=res.Cd_s, Cd_p=res.Cd_p, err=err,
            t_ref=t_ref, t_ve=t_ve)
end

println("=" ^ 70)
println("Conformation cylinder convergence (Re=$Re_target, Wi=$Wi_target, β=$β)")
println("  Backend: $(typeof(backend))")
println("=" ^ 70)

# Pre-flight: nvidia-smi
if backend isa CUDABackend
    println("GPU: ", CUDA.name(CUDA.device()))
end

results = []
for R in (8, 16, 24, 32, 48)
    push!(results, run_one(R))
end

println("\n" * "=" ^ 70)
println("# Convergence summary")
@printf("%-4s %-5s %-5s %-9s %-10s %-10s %-10s %-8s\n",
        "R", "Nx", "Ny", "steps", "Cd_ref", "Cd_s", "Cd_p", "err%")
println("-" ^ 70)
for r in results
    @printf("%-4d %-5d %-5d %-9d %-10.4f %-10.4f %-10.4f %-8.3f\n",
            r.R, r.Nx, r.Ny, r.steps, r.Cd_ref, r.Cd_s, r.Cd_p, 100*r.err)
end

println("\n# Observed convergence order")
for k in 2:length(results)
    ratio  = results[k-1].err / results[k].err
    factor = results[k].R / results[k-1].R
    order  = log(ratio) / log(factor)
    @printf("R %2d → %2d : err ratio = %.2f, order ≈ %.2f\n",
            results[k-1].R, results[k].R, ratio, order)
end

println("\nDone.")
