```@meta
EditURL = "07_heat_conduction.jl"
```

# 1D Heat Conduction


## Problem statement

Pure heat conduction between two horizontal plates at different temperatures
is the simplest validation case for a thermal LBM solver.  With no fluid
motion, the steady-state temperature profile is linear:

```math
T(y) = T_\text{hot} - \frac{T_\text{hot} - T_\text{cold}}{H}\, y
```

where the hot plate is at the bottom (``y = 0``) and the cold plate at the
top (``y = H``).  This test verifies the double distribution function (DDF)
thermal model before enabling buoyancy coupling
[He *et al.* (1998)](@cite he1998novel).

## LBM setup

| Parameter | Value |
|-----------|-------|
| Lattice   | D2Q9 (flow + thermal DDF) |
| Domain    | ``128 \times 32`` (periodic in *x*) |
| Bottom    | ``T = T_\text{hot} = 1`` |
| Top       | ``T = T_\text{cold} = 0`` |
| ``\text{Ra}`` | 100 (sub-critical, ``\text{Ra}_c \approx 1708``) |
| ``\text{Pr}`` | 1.0 |

At sub-critical Rayleigh numbers (``\text{Ra} < 1708``), buoyancy is too weak
to trigger convective instability and the solution remains purely conductive.

## Simulation

```julia
using Kraken
using CairoMakie

Ra    = 100.0
Pr    = 1.0
T_hot = 1.0
T_cold = 0.0

ρ, ux, uy, Temp, config, Ra_out, Pr_out, ν, α = run_rayleigh_benard_2d(;
    Nx=128, Ny=32, Ra=Ra, Pr=Pr, T_hot=T_hot, T_cold=T_cold, max_steps=20000)
```

## Results

Compare the temperature profile along a vertical line with the analytical
linear solution.

```julia
Ny = size(Temp, 2)
H  = Ny - 1
j_fluid = 2:Ny-1
y_phys  = [(j - 1.5) / H for j in j_fluid]   # normalised [0, 1]
T_ana   = [T_hot - (T_hot - T_cold) * y for y in y_phys]
T_num   = [Temp[64, j] for j in j_fluid]      # mid-column

fig = Figure(size=(600, 420))
ax  = Axis(fig[1, 1];
    xlabel = "Temperature",
    ylabel = "y / H",
    title  = "Heat conduction — Ra = $Ra (sub-critical)")
lines!(ax, T_ana, y_phys; label="Analytical (linear)", linewidth=2)
scatter!(ax, T_num, y_phys; label="LBM", markersize=8)
axislegend(ax; position=:rt)
fig
save("heat_conduction_profile.svg", fig) #hide
```

## Error analysis

Compute the relative ``L_2`` error to confirm that the thermal solver
reproduces the conductive solution.

```julia
L2_error = sqrt(sum((T_num .- T_ana).^2) / sum(T_ana.^2))
@info "Heat conduction" L2_error
```

## Temperature contour

The temperature field should show horizontal isotherms (no convection).

```julia
fig2 = Figure(size=(700, 300))
ax2  = Axis(fig2[1, 1]; title="Temperature field — Ra = $Ra",
            xlabel="x", ylabel="y", aspect=DataAspect())
hm   = heatmap!(ax2, 1:size(Temp,1), 1:Ny, Temp;
                 colormap=:thermal, colorrange=(T_cold, T_hot))
Colorbar(fig2[1, 2], hm; label="T")
fig2
save("heat_conduction_contour.svg", fig2) #hide
```

## References

- [He *et al.* (1998)](@cite he1998novel) — Thermal DDF lattice Boltzmann
- [Guo *et al.* (2002)](@cite guo2002boussinesq) — Boussinesq LBM
- [Kruger *et al.* (2017)](@cite kruger2017lattice) — LBM textbook

