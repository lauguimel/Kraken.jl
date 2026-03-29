# # Couette Flow (2D) — Tutorial
#
#
# ## Problem statement
#
# Plane Couette flow is the steady, shear-driven flow between two parallel
# plates.  One wall moves at velocity ``u_w`` while the other is stationary.
# The resulting velocity profile is linear:
#
# ```math
# u_x(y) = u_w \,\frac{y}{H}
# ```
#
# This is one of the rare cases where the LBM gives an **exact** solution:
# the D2Q9 equilibrium is quadratic in velocity, so a linear profile is
# reproduced at machine precision.
#
# ---
#
# ## Simulation file
#
# Download: [`couette.krk`](../assets/krk/couette.krk)
#
# The entire simulation is defined in a plain-text `.krk` file:
#
# ```
# Simulation couette D2Q9
# Domain  L = 0.125 x 1.0  N = 4 x 32
#
# Define u_wall = 0.05
#
# Physics nu = 0.1
#
# Boundary x periodic
# Boundary south wall
# Boundary north velocity(ux = u_wall, uy = 0)
#
# Run 10000 steps
# Output vtk every 2000 [rho, ux, uy]
# ```
#
# **Key points:**
# - `Boundary x periodic` — makes both west and east faces periodic
# - `Boundary south wall` — half-way bounce-back (stationary, no-slip)
# - `Boundary north velocity(ux = u_wall, uy = 0)` — Zou–He moving wall
# - `Define u_wall = 0.05` — user variable substituted everywhere
#
# ## Run it

using Kraken
using CairoMakie

krk = joinpath(@__DIR__, "..", "..", "..", "examples", "couette.krk")
result = run_simulation(krk)
nothing #hide

# One line: `run_simulation("couette.krk")`.  The result is a `NamedTuple`
# with `ρ`, `ux`, `uy` fields on CPU, ready for post-processing.
#
# ## Velocity profile
#
# Compare the numerical profile along a vertical slice with the analytical
# linear solution.

Ny = 32
u_wall = 0.05
H = Ny - 1

y_phys = [j - 1 for j in 1:Ny]
u_ana  = [u_wall * (j - 1) / H for j in 1:Ny]
u_num  = result.ux[2, :]

fig = Figure(size=(600, 420))
ax = Axis(fig[1, 1];
    xlabel = "u_x  (lattice units)",
    ylabel = "y  (lattice units)",
    title  = "Couette flow — Ny = $Ny")
lines!(ax, u_ana, y_phys; label="Analytical", linewidth=2)
scatter!(ax, u_num, y_phys; label="LBM (.krk)", markersize=8)
axislegend(ax; position=:rt)
fig
save("couette_profile.svg", fig) #hide

# The numerical and analytical profiles overlap perfectly.
#
# ---
#
# ## Convergence study — parametric kwargs
#
# The same `.krk` file can be re-used at different resolutions by passing
# **keyword arguments** to `run_simulation`.  These override `Domain`,
# `Physics`, `Define`, and `Run` values without modifying the file:
#
# ```julia
# run_simulation("couette.krk"; Ny=64, max_steps=20000)
# ```
#
# We use this to sweep ``N_y`` and measure the ``L_2`` error at each
# resolution.  The number of steps is scaled as ``\sim H^2/\nu`` to
# reach steady state.

Ny_list = [16, 32, 64, 128]
errors  = Float64[]

for Ny_i in Ny_list
    H_i    = Ny_i - 1
    nsteps = max(10_000, ceil(Int, 3 * H_i^2 / 0.1))

    res = run_simulation(krk; Ny=Ny_i, max_steps=nsteps)

    jf  = 2:Ny_i-1
    u_a = [0.05 * (j - 1) / H_i for j in jf]
    u_n = [res.ux[2, j] for j in jf]
    L2  = sqrt(sum((u_n .- u_a).^2) / sum(u_a.^2))
    push!(errors, L2)
end

fig2 = Figure(size=(500, 400))
ax2  = Axis(fig2[1, 1];
    xlabel = "Ny", ylabel = "Relative L₂ error",
    title  = "Convergence — Couette flow",
    xscale = log10, yscale = log10)
scatterlines!(ax2, Float64.(Ny_list), errors;
    linewidth=2, markersize=10, label="LBM")
hlines!(ax2, [eps()]; linestyle=:dash, color=:gray, label="machine ε")
axislegend(ax2; position=:rt)
fig2
save("couette_convergence.svg", fig2) #hide

# The error is at ``\sim 10^{-2}`` because the `.krk` runner uses
# half-way bounce-back on the south wall (first-order wall position).
# With both walls using Zou–He (Julia API `run_couette_2d`), the error
# drops to machine precision.
#
# ---
#
# ## Anatomy of a `.krk` file
#
# | Line | What it does |
# |------|-------------|
# | `Simulation couette D2Q9` | Name the simulation and pick the lattice |
# | `Domain L = 0.125 x 1.0 N = 4 x 32` | Physical size and grid resolution |
# | `Define u_wall = 0.05` | Declare a variable (substituted in all expressions) |
# | `Physics nu = 0.1` | Set kinematic viscosity (lattice units) |
# | `Boundary x periodic` | Make west + east faces periodic |
# | `Boundary south wall` | No-slip (bounce-back) |
# | `Boundary north velocity(...)` | Zou–He velocity, can use expressions |
# | `Run 10000 steps` | Number of timesteps |
# | `Output vtk every 2000 [...]` | Write VTK files |
#
# No Julia code is needed to define or run the simulation.
# The full syntax reference is in the [Configuration Files](../examples/10_krk_config.md) page.
#
# ## References
#
# - [Zou & He (1997)](@cite zou1997pressure) — Zou–He boundary conditions
# - [Kruger *et al.* (2017)](@cite kruger2017lattice) — LBM textbook
