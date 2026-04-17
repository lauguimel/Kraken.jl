using Kraken, Metal
backend = MetalBackend(); FT = Float32
R = 20; u = 0.02; Ny = 4R; Nx = 800
ν_t = u * R; ν_s = 0.59 * ν_t; ν_p = ν_t - ν_s
Wi = 0.001; λ = Wi * R / u

rN = run_cylinder_2d(; Nx=Nx, Ny=Ny, radius=R, u_in=u, ν=ν_t,
        max_steps=20000, avg_window=5000, backend=backend, T=FT)
println("Newtonien ref Cd = ", round(rN.Cd, digits=3))

for (label, pbc) in [("CNEBB", CNEBB()), ("NoPBC", NoPolymerWallBC())]
    r = run_conformation_cylinder_libb_2d(;
            Nx=30R, Ny=Ny, radius=R, cx=15R, cy=2R,
            u_mean=u, ν_s=ν_s, ν_p=ν_p, lambda=λ, tau_plus=1.0,
            polymer_bc=pbc, inlet=:parabolic,
            max_steps=20000, avg_window=5000, backend=backend, FT=FT)
    # Adjust ref for parabolic Cd convention (u_ref=u_mean=u)
    println(label, ": Cd=", round(r.Cd, digits=3),
            "  ratio_vs_plug_Newt=", round(r.Cd/rN.Cd, digits=4))
end

# Also test OLD driver (halfway-BB, no LI-BB) without CNEBB
# by calling with a modified driver that skips CNEBB
r_old = run_conformation_cylinder_2d(; Nx=Nx, Ny=Ny, radius=R,
        u_in=u, ν_s=ν_s, ν_p=ν_p, lambda=λ, tau_plus=1.0,
        max_steps=20000, avg_window=5000, backend=backend, FT=FT)
println("OLD_hwBB: Cd=", round(r_old.Cd, digits=3),
        "  ratio=", round(r_old.Cd/rN.Cd, digits=4))
