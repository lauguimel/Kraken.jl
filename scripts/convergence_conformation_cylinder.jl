# Spatial convergence: conformation TRT-LBM cylinder vs Newtonian reference.
# Same physical setup (Re, Wi, β) — only the lattice resolution changes.
# Reports |Cd_s - Cd_ref| / Cd_ref and the observed order.

using Kraken, Printf

# Physical constants (kept fixed across resolutions)
const Re_target = 3.2
const Wi_target = 0.0125
const β         = 0.6           # ν_s / ν_total
const aspect_x  = 30            # Nx / R
const aspect_y  = 7.5           # Ny / R   (240/8 / 60/8)
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

    ref = run_cylinder_2d(; Nx=Nx, Ny=Ny, radius=R, u_in=u_in, ν=ν_total,
                            max_steps=max_steps, avg_window=avg_window)
    res = run_conformation_cylinder_2d(; Nx=Nx, Ny=Ny, radius=R, u_in=u_in,
                                         ν_s=ν_s, ν_p=ν_p, lambda=λ, tau_plus=1.0,
                                         max_steps=max_steps, avg_window=avg_window)
    err = abs(res.Cd_s - ref.Cd) / ref.Cd
    return (R=R, Nx=Nx, Ny=Ny, steps=max_steps,
            Cd_ref=ref.Cd, Cd_s=res.Cd_s, Cd_p=res.Cd_p, err=err)
end

results = []
for R in (8, 12, 16, 24)
    @info "==== R = $R ===="
    push!(results, run_one(R))
end

println("\n# Convergence summary (Re=$Re_target, Wi=$Wi_target, β=$β)")
@printf("%-4s %-5s %-5s %-9s %-9s %-9s %-9s %-7s\n",
        "R", "Nx", "Ny", "steps", "Cd_ref", "Cd_s", "Cd_p", "err%")
for r in results
    @printf("%-4d %-5d %-5d %-9d %-9.4f %-9.4f %-9.4f %-7.3f\n",
            r.R, r.Nx, r.Ny, r.steps, r.Cd_ref, r.Cd_s, r.Cd_p, 100*r.err)
end

# Observed order between consecutive refinements
println("\n# Observed order (log2 of error ratio for R doubled)")
for k in 2:length(results)
    ratio = results[k-1].err / results[k].err
    factor = results[k].R / results[k-1].R
    order  = log(ratio) / log(factor)
    @printf("R %d → %d : err ratio = %.2f, order ≈ %.2f\n",
            results[k-1].R, results[k].R, ratio, order)
end
