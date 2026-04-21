# Step 4 — sphere 3D R-sweep at Wi=0.1 on Metal M3 Max (Float32).
#
# Goal: does Cd_visco(Wi=0.1) / Cd_Newt converge toward 1.0 as R grows ?
# If yes → -10.8% at R=16 was discretization, driver valid.
# If plateau → 3D-specific bug.
#
# Same setup as hpc/sphere_R_sweep.jl (uniform inlet, β=0.5, blockage 0.5)
# but Float32 Metal local.

using Kraken, Printf, KernelAbstractions, Metal

backend = MetalBackend()
FT = Float32

println("="^75)
println("Step 4 — sphere R-sweep (Metal M3 Max, Float32)")
println("="^75)

Wi = 0.1f0; β = 0.5f0; u_in = 0.02f0
@printf("%-5s %-9s %-10s %-10s %-10s %-10s %-8s\n",
        "R", "cells", "Cd_Newt", "Cd_visco", "ratio", "deficit", "time")
println("-"^70)

for R_s in [16, 32]
    Nx = 24 * R_s; Ny = 4 * R_s; Nz = 4 * R_s
    cx = 8 * R_s; cy = Ny ÷ 2; cz = Nz ÷ 2
    cells = Nx * Ny * Nz
    ν_total = u_in * (2 * R_s) / 1.0f0    # Re=1
    ν_s = β * ν_total; ν_p = (1 - β) * ν_total
    λ = Wi * R_s / u_in
    steps_newt  = 20_000
    steps_visco = max(30_000, Int(ceil(8f0 * λ)))
    avg_newt  = steps_newt  ÷ 4
    avg_visco = steps_visco ÷ 5

    flush(stdout)
    t0 = time()
    ref = try
        run_sphere_libb_3d(; Nx=Nx, Ny=Ny, Nz=Nz, radius=R_s,
                             cx=cx, cy=cy, cz=cz,
                             u_in=u_in, ν=ν_total, inlet=:uniform,
                             max_steps=steps_newt, avg_window=avg_newt,
                             backend=backend, T=FT)
    catch err
        @warn "R=$R_s Newtonian failed" err
        (; Cd=NaN)
    end
    t_newt = time() - t0

    m_OB = OldroydB(G=ν_p/λ, λ=λ)
    t0 = time()
    r = try
        run_conformation_sphere_libb_3d(;
            Nx=Nx, Ny=Ny, Nz=Nz, radius=R_s, cx=cx, cy=cy, cz=cz,
            u_in=u_in, ν_s=ν_s,
            inlet=:uniform, ρ_out=1.0, tau_plus=1.0,
            polymer_bc=CNEBB(), polymer_model=m_OB,
            max_steps=steps_visco, avg_window=avg_visco,
            backend=backend, FT=FT)
    catch err
        @warn "R=$R_s Wi=$Wi failed" err
        (; Cd=NaN)
    end
    t_vis = time() - t0

    ratio   = isnan(r.Cd) || isnan(ref.Cd) ? NaN : r.Cd / ref.Cd
    deficit = isnan(ratio) ? NaN : 1.0 - ratio
    @printf("%-5d %-9d %-10.3f %-10.3f %-10.4f %-10.4f %-8.0fs\n",
            R_s, cells, ref.Cd, r.Cd, ratio, deficit, t_newt + t_vis)
    flush(stdout)
end

println("\nInterpretation:")
println("  Ratio → 1.0 with R → discretization only, driver OK.")
println("  Ratio plateau ≠ 1 → 3D bug in CNEBB or Hermite source.")
