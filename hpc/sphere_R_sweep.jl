# 3D Oldroyd-B sphere — R-convergence at fixed Wi=0.1.
#
# Question: does Cd(R) converge toward the Newtonian reference at low Wi?
# If yes, driver is validated (the R=16 deficit is pure discretization).
# If no, real bug to find (CNEBB × LI-BB on curved surface at small R).
#
# Baseline from hpc/sphere_oldroyd_3d.jl (R=16):
#   Wi=0   Cd_Newt = 215.3
#   Wi=0.1 Cd_visco = 192.0  → deficit 10.8%
#
# Setup identical to sphere_oldroyd_3d.jl (β=0.5, Re=1, uniform inlet,
# blockage R/H = 0.5). Only R varies.
#
# Output: results/sphere_R_sweep.txt

using Kraken, Printf, CUDA

backend = CUDABackend()
FT = Float64

println("="^75)
println("3D Oldroyd-B sphere — R-sweep at Wi=0.1 (β=0.5, Re=1)")
println("Backend: $(typeof(backend)), GPU: $(CUDA.name(CUDA.device()))")
println("="^75)

Wi = 0.1
β  = 0.5
u_in = 0.02

@printf("%-5s %-8s %-10s %-10s %-10s %-10s %-8s\n",
        "R", "cells", "Cd_Newt", "Cd_visco", "ratio", "deficit", "time")
println("-"^70)

for R_s in [16, 32, 48]
    Nx = 24 * R_s
    Ny = 4  * R_s
    Nz = 4  * R_s
    cx = 8  * R_s
    cy = Ny ÷ 2
    cz = Nz ÷ 2
    cells = Nx * Ny * Nz

    ν_total = u_in * (2 * R_s) / 1.0       # Re=1
    ν_s = β * ν_total
    ν_p = (1 - β) * ν_total
    λ   = Wi * R_s / u_in

    # Scale steps ~ λ (polymer build-up) with floor of 20k / 30k.
    steps_newt  = max(20_000, 10 * R_s * 125)           # diffusive-time scaling
    steps_visco = max(30_000, Int(ceil(8 * λ)))
    avg_newt    = steps_newt  ÷ 4
    avg_visco   = steps_visco ÷ 5

    # Newtonian reference
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

    # Viscoelastic run
    m_OB = OldroydB(G=ν_p/λ, λ=λ)
    t0 = time()
    r = try
        run_conformation_sphere_libb_3d(;
            Nx=Nx, Ny=Ny, Nz=Nz, radius=R_s, cx=cx, cy=cy, cz=cz,
            u_in=u_in, ν_s=ν_s,
            inlet=:uniform, ρ_out=1.0, tau_plus=1.0,
            polymer_bc=CNEBB(),
            polymer_model=m_OB,
            max_steps=steps_visco, avg_window=avg_visco,
            backend=backend, FT=FT)
    catch err
        @warn "R=$R_s Wi=$Wi failed" err
        (; Cd=NaN)
    end
    t_vis = time() - t0

    ratio   = isnan(r.Cd) || isnan(ref.Cd) ? NaN : r.Cd / ref.Cd
    deficit = isnan(ratio) ? NaN : 1.0 - ratio
    @printf("%-5d %-8d %-10.3f %-10.3f %-10.4f %-10.4f %-8.0fs\n",
            R_s, cells, ref.Cd, r.Cd, ratio, deficit, t_newt + t_vis)
    flush(stdout)
end

println("\nInterpretation:")
println("  - Deficit shrinking with R (order ~2-3) → discretization; driver OK.")
println("  - Deficit plateaus → real bug (CNEBB × LI-BB at curved surface).")
println("\nDone.")
