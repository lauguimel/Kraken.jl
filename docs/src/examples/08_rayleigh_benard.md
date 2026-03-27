```@meta
EditURL = "08_rayleigh_benard.jl"
```

# Rayleigh--Benard Convection (2D)


## Problem statement

Rayleigh--Benard convection arises when a horizontal fluid layer is heated
from below.  The hot bottom plate and cold top plate create an unstable
density stratification.  When the Rayleigh number exceeds the critical value
``\text{Ra}_c \approx 1708``, the conductive state becomes unstable and
convection rolls develop.

The Rayleigh number is defined as

```math
\text{Ra} = \frac{g\,\beta\,\Delta T\, H^3}{\nu\,\alpha}
```

where ``\beta`` is the thermal expansion coefficient, ``\Delta T = T_\text{hot}
- T_\text{cold}``, ``H`` is the channel height, ``\nu`` the kinematic
viscosity, and ``\alpha = \nu / \text{Pr}`` the thermal diffusivity.

The Boussinesq approximation models buoyancy as a body force in the momentum
equation, which is implemented via the
[Guo forcing scheme](@cite guo2002boussinesq) in the LBM framework.

## LBM setup

| Parameter | Value |
|-----------|-------|
| Lattice   | D2Q9 (flow) + D2Q9 (thermal DDF) |
| Domain    | ``128 \times 32`` |
| Top/Bottom| Isothermal walls (Dirichlet), no-slip |
| Left/Right| Periodic |
| ``\text{Ra}`` | 5000 (supercritical) |
| ``\text{Pr}`` | 1.0 |

## Simulation

```julia
using Kraken
using CairoMakie

Ra     = 5000.0
Pr     = 1.0
T_hot  = 1.0
T_cold = 0.0

ρ, ux, uy, Temp, config, Ra_out, Pr_out, ν, α = run_rayleigh_benard_2d(;
    Nx=128, Ny=32, Ra=Ra, Pr=Pr, T_hot=T_hot, T_cold=T_cold, max_steps=30000)
```

## Results — temperature field

Supercritical convection should produce characteristic convection rolls
visible as mushroom-shaped plumes in the temperature field.

```julia
Nx, Ny = size(Temp)

fig = Figure(size=(800, 350))
ax  = Axis(fig[1, 1]; title="Temperature — Ra = $Ra, Pr = $Pr",
           xlabel="x", ylabel="y", aspect=DataAspect())
hm  = heatmap!(ax, 1:Nx, 1:Ny, Temp;
               colormap=:thermal, colorrange=(T_cold, T_hot))
Colorbar(fig[1, 2], hm; label="T")
fig
save("rayleigh_benard_temperature.svg", fig) #hide
```

## Velocity field

The velocity magnitude shows the convection roll structure.

```julia
umag = @. sqrt(ux^2 + uy^2)

fig2 = Figure(size=(800, 350))
ax2  = Axis(fig2[1, 1]; title="Velocity magnitude — Ra = $Ra",
            xlabel="x", ylabel="y", aspect=DataAspect())
hm2  = heatmap!(ax2, 1:Nx, 1:Ny, umag; colormap=:viridis)
Colorbar(fig2[1, 2], hm2; label="|u|")
fig2
save("rayleigh_benard_velocity.svg", fig2) #hide
```

## Temperature profile

Extract the horizontally-averaged temperature profile and compare with the
conductive (linear) solution.  Convective transport creates a more uniform
temperature in the bulk with thin boundary layers near the walls.

```julia
T_avg = [sum(Temp[:, j]) / Nx for j in 1:Ny]
y_norm = [(j - 0.5) / (Ny - 1) for j in 1:Ny]
T_lin  = [T_hot - (T_hot - T_cold) * y for y in y_norm]

fig3 = Figure(size=(500, 400))
ax3  = Axis(fig3[1, 1]; xlabel="<T>_x", ylabel="y / H",
            title="Mean temperature profile")
lines!(ax3, T_lin, y_norm; label="Conductive", linestyle=:dash, color=:gray)
lines!(ax3, T_avg, y_norm; label="Convective (Ra=$Ra)", linewidth=2)
axislegend(ax3; position=:rt)
fig3
save("rayleigh_benard_profile.svg", fig3) #hide
```

## References

- [Guo *et al.* (2002)](@cite guo2002boussinesq) — Boussinesq forcing in LBM
- [He *et al.* (1998)](@cite he1998novel) — Thermal DDF model
- [Kruger *et al.* (2017)](@cite kruger2017lattice) — LBM textbook

