# Step 5d — 2D cylinder at Wi ∈ {0.001, 0.01, 0.1} at R=30, same β=0.5
# as the sphere, to check if the Wi-enhancement anomaly at low Wi is
# 3D-specific or present in 2D too.
#
# At Wi=0.01 Oldroyd-B should give enhancement << 1% (of order Wi² ≈
# 1e-4). Getting +8% at Wi=0.01 in the sphere is ~800× too much.
#
# If the 2D cylinder at Wi=0.01 gives Cd/Cd_Newt ≈ 1.0001 → the bug is
# 3D-specific (look at Hermite 3D or 3D conformation source).
# If 2D also gives +5-10% at Wi=0.01 → the bug is in the shared code
# path (prefactor or drag formula).

using Kraken, Printf, CUDA

backend = CUDABackend()
FT = Float64

println("="^75)
println("Step 5d — 2D cylinder Wi sweep at R=30, β=0.5 (same as sphere)")
println("="^75)

R = 30; β = 0.5; u_mean = 0.02
Nx = 30 * R; Ny = 4 * R; cx = 15 * R; cy = 2 * R
Re_target = 1.0
ν_total = u_mean * R / Re_target
ν_s = β * ν_total; ν_p = (1 - β) * ν_total

# Newtonian reference at ν_total
t0 = time()
ref_Cd = NaN
try
    ref = run_cylinder_libb_2d(; Nx=Nx, Ny=Ny, cx=cx, cy=cy, radius=R,
                                u_mean=u_mean, ν=ν_total,
                                inlet=:parabolic,
                                max_steps=60_000, avg_window=12_000,
                                backend=backend, FT=FT)
    global ref_Cd = ref.Cd
catch err
    @warn "Newtonian ref failed — falling back to Liu table value 130.36" err
    global ref_Cd = 130.36
end
t_newt = time() - t0
@printf("Newtonian (ν_total=%.4f): Cd=%.4f (time=%.0fs)\n", ν_total, ref_Cd, t_newt)
println()

@printf("%-6s %-8s %-10s %-10s %-10s %-10s\n",
        "Wi", "λ", "Cd_visco", "ratio", "shift%", "time")
println("-"^60)

for Wi in [0.001, 0.01, 0.1]
    λ = Wi * R / u_mean
    max_steps = max(60_000, Int(ceil(10 * λ)))
    avg_window = max_steps ÷ 5
    m_OB = OldroydB(G=ν_p/λ, λ=λ)

    t0 = time()
    r = try
        run_conformation_cylinder_libb_2d(;
            Nx=Nx, Ny=Ny, radius=R, cx=cx, cy=cy,
            u_mean=u_mean, ν_s=ν_s,
            polymer_model=m_OB, polymer_bc=CNEBB(),
            inlet=:parabolic, ρ_out=1.0, tau_plus=1.0,
            max_steps=max_steps, avg_window=avg_window,
            backend=backend, FT=FT)
    catch err
        @warn "Wi=$Wi failed" err
        (; Cd=NaN)
    end
    dt = time() - t0
    ratio = isnan(r.Cd) ? NaN : r.Cd / ref_Cd
    shift = isnan(ratio) ? NaN : 100*(ratio - 1)
    @printf("%-6.3f %-8.1f %-10.4f %-10.4f %+-10.3f %-8.0fs\n",
            Wi, λ, r.Cd, ratio, shift, dt)
    flush(stdout)
end

println()
println("Interpretation :")
println("  Wi=0.001 ratio > 1.02 or Wi=0.01 ratio > 1.01 →")
println("    anomalous low-Wi enhancement present in 2D too :")
println("    bug is in a shared code path (Hermite prefactor, drag formula).")
println("  Wi=0.01 ratio < 1.001 → 2D is clean, sphere 3D bug is 3D-specific.")
