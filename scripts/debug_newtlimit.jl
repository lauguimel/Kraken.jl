using Kraken, Metal, KernelAbstractions
backend = MetalBackend(); FT = Float32
R = 20; Ny = 4R; Nx = 800; u = FT(0.02)
ν_t = FT(u * R); ν_s = FT(0.59) * ν_t; ν_p = ν_t - ν_s

# A: Newtonian at ν_total
rN = run_cylinder_2d(; Nx=Nx, Ny=Ny, radius=R, u_in=Float64(u), ν=Float64(ν_t),
        max_steps=20000, avg_window=5000, backend=backend, T=FT)

# B: Solvent-only at ν_s (τ_p=0, no conformation)
# Use collide_viscoelastic_source_guo_2d! with zero τ_p → same as collide_2d at ω_s
rS = run_cylinder_2d(; Nx=Nx, Ny=Ny, radius=R, u_in=Float64(u), ν=Float64(ν_s),
        max_steps=20000, avg_window=5000, backend=backend, T=FT)

# C: Full coupling Wi=0.001
Wi = FT(0.001); λ = Float64(Wi * R / u)
rV = run_conformation_cylinder_2d(; Nx=Nx, Ny=Ny, radius=R,
        u_in=Float64(u), ν_s=Float64(ν_s), ν_p=Float64(ν_p), lambda=λ, tau_plus=1.0,
        max_steps=20000, avg_window=5000, backend=backend, FT=FT)

println("A: Newtonian ν_total  Cd = ", round(rN.Cd, digits=3))
println("B: Solvent-only ν_s   Cd = ", round(rS.Cd, digits=3))
println("C: VE Wi=0.001        Cd = ", round(rV.Cd, digits=3))
println()
println("C/A ratio = ", round(rV.Cd / rN.Cd, digits=4), "  (should → 1.0)")
println("C/B ratio = ", round(rV.Cd / rS.Cd, digits=4), "  (should > 1, polymer adds viscosity)")
println("A/B ratio = ", round(rN.Cd / rS.Cd, digits=4), "  (expected: ν_total/ν_s effect on Cd)")
