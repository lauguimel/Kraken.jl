using Kraken, Metal, KernelAbstractions
backend = MetalBackend(); FT = Float32

# Fix ν_total = 0.4, vary β (viscosity split) → changes ω_s
# If the bug depends on ω, the ratio C/A will change with β.
# If it's constant, the bug is ω-independent.
R = 20; Ny = 4R; Nx = 800; u = 0.02
ν_total = u * R  # = 0.4

rN = run_cylinder_2d(; Nx=Nx, Ny=Ny, radius=R, u_in=u, ν=ν_total,
        max_steps=20000, avg_window=5000, backend=backend, T=FT)
println("Newtonien ref Cd = ", round(rN.Cd, digits=3))

println("\nβ        ν_s      ω_s      Cd_VE     C/A")
for β in [0.9, 0.7, 0.59, 0.4, 0.2]
    ν_s = β * ν_total; ν_p = ν_total - ν_s
    Wi = 0.001; λ = Wi * R / u
    r = run_conformation_cylinder_2d(; Nx=Nx, Ny=Ny, radius=R,
            u_in=u, ν_s=ν_s, ν_p=ν_p, lambda=λ, tau_plus=1.0,
            max_steps=20000, avg_window=5000, backend=backend, FT=FT)
    ω_s = 1.0 / (3*ν_s + 0.5)
    println(round(β, digits=2), "      ",
            round(ν_s, digits=3), "    ",
            round(ω_s, digits=4), "    ",
            round(r.Cd, digits=3), "     ",
            round(r.Cd/rN.Cd, digits=4))
end
