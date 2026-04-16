# τ_plus sweep: stabilize Wi=0.5, 1.0 via higher artificial diffusion κ.
#
# κ = (τ_plus - 0.5) / 3  (TRT ADE diffusivity)
# Sc = ν_s / κ  (Schmidt number; Liu uses Sc = 10⁴ by default)
#
# Default τ_plus = 1.0 → κ = 0.167 → Sc ≈ 1.4  (way too low)
# τ_plus = 2.0 → κ = 0.5 → Sc ≈ 0.5
# τ_plus = 5.0 → κ = 1.5 → Sc ≈ 0.17
# τ_plus = 0.6 → κ = 0.033 → Sc ≈ 7  (closer to Liu, lower diff)
#
# We want low Sc (high κ) for Wi stability, but high Sc gives physical
# accuracy. Liu fixes Sc=10⁴ via specific TRT + regularised scheme (Λ_p=10⁻⁶).
# Our simple TRT does κ = (τ_p,1 - 0.5)/3 only, so we sweep τ_plus.
#
# Usage: julia --project=. hpc/liu_tau_plus_sweep.jl

using Kraken, Printf, CUDA

backend = CUDABackend()
FT = Float64

println("="^70)
println("τ_plus sweep for high-Wi stability (R=30, β=0.59)")
println("Backend: $(typeof(backend)), GPU: $(CUDA.name(CUDA.device()))")
println("="^70)

R = 30; Nx = 30*R; Ny = 4*R; cx = 15*R; cy = 2*R
β = 0.59; u_mean = 0.02
ν_total = u_mean * R / 1.0         # Re_Liu=1 ⇒ ν_total = 0.6
ν_s = β * ν_total
ν_p = (1 - β) * ν_total

liu_ref = Dict(0.1 => 130.36, 0.5 => 126.31, 1.0 => 151.31)

for τp in [0.6, 1.0, 2.0, 5.0]
    κ = (τp - 0.5) / 3
    Sc = ν_s / κ
    println("\n>>> τ_plus = $τp  (κ = $(round(κ, digits=3)), Sc ≈ $(round(Sc, digits=2)))")
    @printf("%-6s %-10s %-10s %-8s\n", "Wi", "Cd_sim", "Cd_Liu", "err%")
    println("-"^40)

    for Wi in [0.1, 0.5, 1.0]
        λ = Wi * R / u_mean
        model = OldroydB(G=ν_p/λ, λ=λ)
        max_steps = 200_000
        avg_window = 40_000
        t0 = time()
        r = try
            run_conformation_cylinder_libb_2d(;
                Nx=Nx, Ny=Ny, radius=R, cx=cx, cy=cy,
                u_mean=u_mean, ν_s=ν_s,
                polymer_model=model, polymer_bc=CNEBB(),
                inlet=:parabolic, ρ_out=1.0, tau_plus=τp,
                max_steps=max_steps, avg_window=avg_window,
                backend=backend, FT=FT)
        catch e
            (Cd = NaN, Wi = Wi)
        end
        dt = time() - t0
        ref = get(liu_ref, Wi, NaN)
        err = isnan(ref) || isnan(r.Cd) ? NaN : (r.Cd - ref) / ref * 100
        @printf("%-6.3f %-10.3f %-10.3f %-8.2f  (%.0fs)\n",
                Wi, r.Cd, ref, err, dt)
    end
end

println("\nDone.")
