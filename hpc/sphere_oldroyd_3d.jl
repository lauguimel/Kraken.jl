# 3D Oldroyd-B sphere benchmark — Re=1, Wi sweep, β=0.5.
#
# References:
# - Lunsmann et al., J. Non-Newt. Fluid Mech. 50 (1993) 135–155 (ducted sphere)
# - Alves & Pinho, J. Non-Newt. Fluid Mech. 110 (2003) 45–75 (4:1:1 contraction
#   and sphere benchmark recap)
#
# Setup: cylindrical pipe approximated by a square duct (Schäfer-Turek
# convention extended to 3D). H = duct half-height, R_s = sphere radius.
# β_blockage = R_s/H = 0.5 (standard Lunsmann ratio). β_visco = 0.5.
#
# Domain: pipe length 24·R_s, cross-section 4·R_s × 4·R_s. With R_s = 16
# we get 384 × 64 × 64 = 1.6e6 cells. Comfortable on a single H100.
#
# Outputs: results/sphere_oldroyd_3d.txt with Cd vs Wi.

using Kraken, Printf, CUDA

backend = CUDABackend()
FT = Float64

println("="^75)
println("3D Oldroyd-B sphere (β=0.5, Re=1, doubly-parabolic duct)")
println("Backend: $(typeof(backend)), GPU: $(CUDA.name(CUDA.device()))")
println("="^75)

# Geometry
R_s = 16
Nx = 24 * R_s        # 384
Ny = 4  * R_s        # 64 — gives R_s/H = 0.5 (tight blockage)
Nz = 4  * R_s        # 64
cx = 8 * R_s         # 128 — sphere ~1/3 from inlet
cy = Ny ÷ 2
cz = Nz ÷ 2

# Flow scales: Re_Liu = 1 with u_ref = u_in (uniform inlet)
β  = 0.5
u_in = 0.02
ν_total = u_in * (2 * R_s) / 1.0       # Re=1
ν_s = β * ν_total
ν_p = (1 - β) * ν_total

@printf("%-6s %-10s %-10s %-10s %-8s\n", "Wi", "Cd_visco", "Cd_Newt", "rel_diff", "time")
println("-"^55)

# Newtonian reference (Wi → 0 limit)
t0 = time()
ref = run_sphere_libb_3d(; Nx=Nx, Ny=Ny, Nz=Nz, radius=R_s, cx=cx, cy=cy, cz=cz,
                          u_in=u_in, ν=ν_total, inlet=:uniform,
                          max_steps=20_000, avg_window=4_000,
                          backend=backend, T=FT)
@printf("%-6s %-10.3f %-10s %-10s %-8.0fs\n",
        "Newt", ref.Cd, "—", "—", time() - t0)

for Wi in [0.1, 0.5, 1.0]
    λ = Wi * R_s / u_in
    max_steps = 30_000
    avg_window = max_steps ÷ 5
    common = (; Nx=Nx, Ny=Ny, Nz=Nz, radius=R_s, cx=cx, cy=cy, cz=cz,
                u_in=u_in, ν_s=ν_s,
                inlet=:uniform, ρ_out=1.0, tau_plus=1.0,
                polymer_bc=CNEBB(),
                max_steps=max_steps, avg_window=avg_window,
                backend=backend, FT=FT)

    m_OB = OldroydB(G=ν_p/λ, λ=λ)
    t0 = time()
    r = try run_conformation_sphere_libb_3d(; common..., polymer_model=m_OB)
        catch err
            @warn "Wi=$Wi run failed" err
            (; Cd=NaN)
        end
    dt = time() - t0
    rel = isnan(r.Cd) ? NaN : abs(r.Cd - ref.Cd) / ref.Cd
    @printf("%-6.2f %-10.3f %-10.3f %-10.4f %-8.0fs\n", Wi, r.Cd, ref.Cd, rel, dt)
end

println("\nDone.")
