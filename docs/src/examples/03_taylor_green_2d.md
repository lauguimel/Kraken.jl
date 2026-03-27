```@meta
EditURL = "03_taylor_green_2d.jl"
```

# Taylor--Green Vortex (2D)


## Problem statement

The Taylor--Green vortex is an exact unsteady solution of the incompressible
Navier--Stokes equations on a doubly-periodic domain.  The initial velocity
field is

```math
u_x(x,y,0) =  u_0 \sin(kx)\cos(ky), \qquad
u_y(x,y,0) = -u_0 \cos(kx)\sin(ky)
```

with wavenumber ``k = 2\pi/N``.  The kinetic energy decays exponentially
[Taylor & Green (1937)](@cite taylor1937mechanism):

```math
E(t) = E_0 \exp(-2\nu k^2 t), \qquad E_0 = \tfrac{1}{2}u_0^2
```

This test validates the temporal accuracy of the collision operator and the
correct implementation of periodic boundary conditions.

## LBM setup

| Parameter | Value |
|-----------|-------|
| Lattice   | D2Q9, fully periodic |
| Resolution| ``N \times N`` |
| Collision | BGK |
| Mach number | ``\text{Ma} = u_0 / c_s = u_0 \sqrt{3}`` (kept ``\ll 1``) |

## Simulation

```julia
using Kraken
using CairoMakie

N  = 64
ν  = 0.01
u0 = 0.01

ρ, ux, uy, config, u0_out, k, max_steps = run_taylor_green_2d(;
    N=N, ν=ν, u0=u0, max_steps=2000)
```

## Results

Compute the domain-averaged kinetic energy from the final fields and compare
the time evolution with the analytical decay.  We sample the energy at a few
time checkpoints by re-running with different `max_steps`.

```julia
steps_list = 0:200:2000
E_num = Float64[]
E_ana = Float64[]
E0    = 0.5 * u0^2

for s in steps_list
    if s == 0
        push!(E_num, E0)
    else
        ρ_s, ux_s, uy_s, _ = run_taylor_green_2d(; N=N, ν=ν, u0=u0, max_steps=s)[1:4]
        KE = 0.0
        for j in 1:N, i in 1:N
            KE += 0.5 * (ux_s[i,j]^2 + uy_s[i,j]^2)
        end
        push!(E_num, KE / (N * N))
    end
    push!(E_ana, E0 * exp(-2ν * k^2 * s))
end

fig = Figure(size=(600, 420))
ax = Axis(fig[1, 1];
    xlabel = "Time step",
    ylabel = "Mean kinetic energy",
    title  = "Taylor--Green vortex decay — N = $N")
lines!(ax, collect(steps_list), E_ana; label="Analytical", linewidth=2)
scatter!(ax, collect(steps_list), E_num; label="LBM", markersize=10)
axislegend(ax; position=:rt)
fig
save("taylor_green_energy.svg", fig) #hide
```

## Vorticity field

Visualise the vorticity at the final time step using a finite-difference
approximation ``\omega_z = \partial u_y / \partial x - \partial u_x / \partial y``.

```julia
ωz = zeros(N, N)
for j in 1:N, i in 1:N
    ip = mod1(i + 1, N); im = mod1(i - 1, N)
    jp = mod1(j + 1, N); jm = mod1(j - 1, N)
    ωz[i, j] = 0.5 * (uy[ip, j] - uy[im, j]) - 0.5 * (ux[i, jp] - ux[i, jm])
end

fig2 = Figure(size=(500, 450))
ax2 = Axis(fig2[1, 1]; title="Vorticity at t = $max_steps", aspect=DataAspect())
hm = heatmap!(ax2, 1:N, 1:N, ωz; colormap=:balance)
Colorbar(fig2[1, 2], hm; label="ω_z")
fig2
save("taylor_green_vorticity.svg", fig2) #hide
```

## References

- [Taylor & Green (1937)](@cite taylor1937mechanism) — Original analytical solution
- [Chen & Doolen (1998)](@cite chen1998lattice) — LBM review
- [Kruger *et al.* (2017)](@cite kruger2017lattice) — LBM textbook

