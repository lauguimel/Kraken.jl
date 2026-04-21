# Step 5c — Wi sweep at R=16 to check if the sign of the effect is
# consistent (i.e. ratio monotone with Wi) or sign-reversing. Lunsmann
# 1993 predicts ratio monotone INCREASING with Wi. Apr 18 Aqua data
# (from VISCOELASTIC_FINDINGS §1) had :
#   Wi=0.1 → 0.892,  Wi=0.5 → 0.673,  Wi=1.0 → 0.611
# — monotone DECREASING, opposite sign.
#
# Here we extend to very small Wi = {0.01, 0.05, 0.1, 0.25} to check
# the low-Wi asymptote. At Wi → 0, τ_p ≈ 2·ν_p·S (quasi-Newtonian) so
# ratio should tend to 1 smoothly. If ratio at Wi=0.01 is still
# significantly < 1 → global bias, not a high-Wi numerical issue.

using Kraken, Printf, CUDA

backend = CUDABackend()
FT = Float64

println("="^75)
println("Step 5c — sphere 3D R=16, Wi sweep {0.01, 0.05, 0.1, 0.25}")
println("Backend: $(typeof(backend)), GPU: $(CUDA.name(CUDA.device()))")
println("="^75)

R_s = 16
Nx = 24 * R_s; Ny = 4 * R_s; Nz = 4 * R_s
cx = 8 * R_s; cy = Ny ÷ 2; cz = Nz ÷ 2
β = 0.5; u_in = 0.02
ν_total = u_in * (2 * R_s) / 1.0
ν_s = β * ν_total; ν_p = (1 - β) * ν_total

# Reference Newtonian once
t0 = time()
ref = run_sphere_libb_3d(; Nx=Nx, Ny=Ny, Nz=Nz, radius=R_s,
                          cx=cx, cy=cy, cz=cz,
                          u_in=u_in, ν=ν_total, inlet=:uniform,
                          max_steps=20_000, avg_window=4_000,
                          backend=backend, T=FT)
t_newt = time() - t0
@printf("Newtonian (ν_total): Cd=%.4f (%.0fs)\n", ref.Cd, t_newt)
println()

@printf("%-6s %-8s %-10s %-10s %-10s %-10s\n",
        "Wi", "λ", "Cd_visco", "ratio", "shift%", "time")
println("-"^60)
for Wi in [0.01, 0.05, 0.1, 0.25]
    λ = Wi * R_s / u_in
    m_OB = OldroydB(G=ν_p/λ, λ=λ)
    # Scale max_steps with λ for polymer relaxation (5λ floor, 30k minimum)
    max_steps = max(30_000, Int(ceil(8 * λ)))
    avg_window = max_steps ÷ 5

    t0 = time()
    r = try
        run_conformation_sphere_libb_3d(;
            Nx=Nx, Ny=Ny, Nz=Nz, radius=R_s, cx=cx, cy=cy, cz=cz,
            u_in=u_in, ν_s=ν_s,
            inlet=:uniform, ρ_out=1.0, tau_plus=1.0,
            polymer_bc=CNEBB(), polymer_model=m_OB,
            max_steps=max_steps, avg_window=avg_window,
            backend=backend, FT=FT)
    catch err
        @warn "Wi=$Wi failed" err
        (; Cd=NaN)
    end
    dt = time() - t0
    ratio = isnan(r.Cd) ? NaN : r.Cd / ref.Cd
    shift = isnan(ratio) ? NaN : 100*(ratio - 1)
    @printf("%-6.3f %-8.1f %-10.4f %-10.4f %+-10.3f %-8.0fs\n",
            Wi, λ, r.Cd, ratio, shift, dt)
    flush(stdout)
end

println()
println("Interpretation :")
println("  Monotone decrease of ratio with Wi → wrong sign persists at all Wi")
println("  ratio(Wi=0.01) > 0.99 → low-Wi limit clean, issue is strictly Wi-scaling")
println("  ratio(Wi=0.01) < 0.95 → global bias, not just an HWNP issue")
