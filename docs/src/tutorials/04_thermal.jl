# # Tutorial 4 — Thermal Flows
#
# So far every simulation was **isothermal**.  Now we enable the thermal
# module and simulate **Rayleigh--Benard convection** — a fluid layer
# heated from below that develops convection rolls.
#
# **You will learn:** `Module thermal`, Prandtl/Rayleigh numbers,
# temperature BCs, and the Double Distribution Function (DDF) approach.
#
# ---
#
# ## The configuration file
#
# The shipped `rayleigh_benard.krk` uses a `Preset` shortcut; here we
# show every directive explicitly.
#
# ```
# Simulation rayleigh_benard D2Q9
# Domain  L = 2.0 x 1.0  N = 128 x 64
#
# Module thermal
#
# Physics nu = 0.02  Pr = 0.71  Ra = 1e4
#
# Boundary x periodic
# Boundary south wall  T = 1.0
# Boundary north wall  T = 0.0
#
# Run 30000 steps
# Output vtk every 5000 [rho, ux, uy, T]
# ```
#
# ### `Module thermal`
#
# This single line activates the thermal DDF solver.  Without it, Kraken
# runs a purely isothermal simulation and ignores any `Pr`, `Ra`, or
# temperature boundary conditions.
#
# ### `Physics nu = 0.02  Pr = 0.71  Ra = 1e4`
#
# - **`nu`** — kinematic viscosity (lattice units).
# - **`Pr`** — Prandtl number ``\nu / \alpha`` (0.71 = air).
# - **`Ra`** — Rayleigh number; above ``Ra_c \approx 1708`` the
#   conductive state is unstable and convection rolls appear.
#
# Kraken derives ``\alpha = \nu / Pr`` and the Boussinesq coupling
# ``\beta g`` automatically.
#
# ### Temperature BCs
#
# `T = 1.0` on the south wall (hot) and `T = 0.0` on the north wall
# (cold).  Left/right are periodic — the convection pattern picks its
# own wavelength.
#
# ---
#
# ## What is DDF?
#
# The **Double Distribution Function** method uses two sets of D2Q9
# populations — one for flow, one for temperature — each with its own
# relaxation rate.  The two are coupled through the **Boussinesq force**:
# local temperature creates a buoyancy term in the flow collision step.
#
# ---
#
# ## Running the simulation

using Kraken

KRK = joinpath(@__DIR__, "..", "..", "..", "examples")
result = run_simulation(joinpath(KRK, "rayleigh_benard.krk"))

ρ, ux, uy, Temp = result.ρ, result.ux, result.uy, result.Temp

# The thermal module adds a `Temp` field to the result.
#
# ---
#
# ## Temperature field

Nx, Ny = size(Temp)

using CairoMakie

fig = Figure(size = (800, 400))
ax  = Axis(fig[1, 1];
    title  = "Temperature — Rayleigh-Bénard Ra = 10⁴",
    xlabel = "x", ylabel = "y", aspect = DataAspect())
hm = heatmap!(ax, 1:Nx, 1:Ny, Temp; colormap = :coolwarm)
Colorbar(fig[1, 2], hm; label = "T")
save(joinpath(@__DIR__, "04_rb_temperature.svg"), fig)
fig

# ![Temperature field with hot plumes rising and cold plumes
# sinking.](04_rb_temperature.svg)
#
# ---
#
# ## Velocity field

umag = sqrt.(ux .^ 2 .+ uy .^ 2)

fig2 = Figure(size = (800, 400))
ax2  = Axis(fig2[1, 1];
    title  = "Velocity magnitude — Rayleigh-Bénard Ra = 10⁴",
    xlabel = "x", ylabel = "y", aspect = DataAspect())
hm2 = heatmap!(ax2, 1:Nx, 1:Ny, umag; colormap = :viridis)
Colorbar(fig2[1, 2], hm2; label = "|u|")
save(joinpath(@__DIR__, "04_rb_velocity.svg"), fig2)
fig2

# ![Velocity magnitude showing counter-rotating convection
# rolls.](04_rb_velocity.svg)
#
# ---
#
# ## Going further
#
# - **Higher Ra:** ``10^5`` or ``10^6`` for time-dependent convection.
# - **Natural convection:** differentially-heated cavity uses the same DDF.
# - **Heat conduction:** set ``Ra \ll Ra_c`` — see
#   [heat conduction](../examples/07_heat_conduction.md).
