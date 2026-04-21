# Smoke test: does sphere_libb_3d segfault on Metal M3 at R=16 (1.6M cells) ?
# 500 steps only — ~10s if it runs, crashes fast if not.

using Kraken, Printf, KernelAbstractions
try
    using Metal
    global backend = MetalBackend()
    println("Metal backend OK")
catch e
    println("Metal not available : $e")
    global backend = KernelAbstractions.CPU()
end

FT = Float32
R = 16
Nx = 24 * R; Ny = 4 * R; Nz = 4 * R
cx = 8 * R; cy = Ny ÷ 2; cz = Nz ÷ 2
u_in = 0.02f0
ν_total = u_in * 2 * R / 1.0f0       # Re = 1

@printf("Grid %d×%d×%d = %.2fM cells. ν=%.4f.\n",
        Nx, Ny, Nz, Nx*Ny*Nz/1e6, ν_total)
@printf("Running 500 Newtonian steps on %s...\n", typeof(backend))
flush(stdout)

t0 = time()
r = run_sphere_libb_3d(; Nx=Nx, Ny=Ny, Nz=Nz, radius=R, cx=cx, cy=cy, cz=cz,
                        u_in=u_in, ν=ν_total, inlet=:uniform,
                        max_steps=500, avg_window=100,
                        backend=backend, T=FT)
dt = time() - t0
@printf("SUCCESS after %.1fs — Cd=%.3f\n", dt, r.Cd)
