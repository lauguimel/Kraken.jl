# High-resolution Wi sweep: direct-C vs log-conformation at R=48 on Liu setup.
#
# Rationale (2026-04-18): the validated R-convergence study showed R=48 brings
# Cd at Wi=0.1 to within 0.35 % of Liu Table 3. At R=30 the log-conformation
# driver stays stable up to Wi=1.0 but the residual error at Wi=0.5/1.0 is
# −26 %/−44 %, dominated by HWNP (high Weissenberg number problem) being only
# partially cured at that resolution. Extrapolating the R-convergence slope
# suggests R=48 should bring Wi=0.5 to <10 % error, which would constitute a
# publishable 2D Oldroyd-B validation across the Liu Wi range.
#
# Outputs: results/liu_logconf_sweep_R48.txt
# Usage  : julia --project=. hpc/liu_logconf_sweep_R48.jl

using Kraken, Printf, CUDA

backend = CUDABackend()
FT = Float64

println("="^75)
println("Liu cylinder: direct-C vs log-conformation (R=48, β=0.59, Re=1)")
println("Backend: $(typeof(backend)), GPU: $(CUDA.name(CUDA.device()))")
println("="^75)

R = 48; Nx = 30*R; Ny = 4*R; cx = 15*R; cy = 2*R
β = 0.59; u_mean = 0.02
ν_total = u_mean * R / 1.0   # Re_Liu = 1
ν_s = β * ν_total
ν_p = (1 - β) * ν_total

liu_ref = Dict(0.1 => 130.36, 0.5 => 126.31, 1.0 => 151.31)

@printf("%-6s %-10s %-10s %-10s %-10s %-8s\n",
        "Wi", "Cd_direct", "Cd_logconf", "Cd_Liu", "err_log%", "time")
println("-"^60)

for Wi in [0.1, 0.5, 1.0]
    λ = Wi * R / u_mean
    # R=48 → ~2.5× cells vs R=30 → scale steps so residence time remains
    # comparable. Keep avg_window = max_steps/5.
    max_steps = 300_000
    avg_window = max_steps ÷ 5
    common = (; Nx=Nx, Ny=Ny, radius=R, cx=cx, cy=cy,
                u_mean=u_mean, ν_s=ν_s, polymer_bc=CNEBB(),
                inlet=:parabolic, ρ_out=1.0, tau_plus=1.0,
                max_steps=max_steps, avg_window=avg_window,
                backend=backend, FT=FT)

    m_direct = OldroydB(G=ν_p/λ, λ=λ)
    m_logc   = LogConfOldroydB(G=ν_p/λ, λ=λ)

    t0 = time()
    r_d = try run_conformation_cylinder_libb_2d(; common..., polymer_model=m_direct)
          catch; (; Cd=NaN) end
    r_l = try run_conformation_cylinder_libb_2d(; common..., polymer_model=m_logc)
          catch; (; Cd=NaN) end
    dt = time() - t0

    ref = get(liu_ref, Wi, NaN)
    err_log = isnan(r_l.Cd) ? NaN : (r_l.Cd - ref)/ref*100
    @printf("%-6.2f %-10.3f %-10.3f %-10.3f %-10.2f %-8.0fs\n",
            Wi, r_d.Cd, r_l.Cd, ref, err_log, dt)
end

println("\nDone.")
