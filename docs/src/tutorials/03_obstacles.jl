# # Tutorial 3 — Flow Around Obstacles
#
# In the previous tutorials we simulated empty channels.  Now we place a
# **circular cylinder** inside the domain and compute the flow at Re = 20.
#
# **You will learn:** geometry predicates, spatial inlet profiles, pressure
# outlets, and drag/lift monitoring.
#
# ---
#
# ## The configuration file
#
# Here is `examples/cylinder.krk` in full — new directives are explained
# below.
#
# ```
# Simulation cylinder D2Q9
# Domain  L = 10.0 x 2.5  N = 200 x 50
#
# Define U  = 0.05
# Define H  = 2.5
# Define cx = 2.5
# Define cy = 1.25
# Define R  = 0.5
#
# Physics nu = 0.05
#
# Obstacle cylinder { (x - cx)^2 + (y - cy)^2 <= R^2 }
#
# Boundary west  velocity(ux = 4*U*y*(H - y)/H^2, uy = 0)
# Boundary east  pressure(rho = 1.0)
# Boundary south wall
# Boundary north wall
#
# Run 20000 steps
# Output vtk every 1000 [rho, ux, uy]
# Diagnostics every 100 [step, drag, lift]
# ```
#
# ### `Define` — user variables
#
# `Define U = 0.05` creates a symbolic constant reusable everywhere in
# the file (boundary expressions, obstacle predicates).  This avoids
# magic numbers and makes parametric studies easy.
#
# ### `Obstacle` — geometry predicates
#
# `Obstacle cylinder { ... }` flags every lattice node satisfying the
# predicate as **solid**.  Half-way bounce-back is applied there.  Any
# boolean expression of `x` and `y` works — circles, rectangles, or
# arbitrary shapes.
#
# ### `Boundary west velocity(...)` — spatial expressions
#
# The inlet imposes a parabolic profile ``u_x(y) = 4 U y (H-y) / H^2``
# via Zou–He.  Because the expression depends on `y`, Kraken evaluates it
# at every node along the west face.
#
# ### `Boundary east pressure(rho = 1.0)` — pressure outlet
#
# Fixes the density (hence pressure ``p = \rho c_s^2``) and lets velocity
# adjust freely — the standard open-boundary treatment.
#
# ### `Diagnostics` — monitoring
#
# Prints drag and lift every 100 steps.  The drag is computed via the
# **Momentum Exchange Algorithm** (MEA) — summing momentum transfers at
# all fluid–solid links.
#
# ---
#
# ## Running the simulation

using Kraken

KRK = joinpath(@__DIR__, "..", "..", "..", "examples")
result = run_simulation(joinpath(KRK, "cylinder.krk"))

ρ  = result.ρ
ux = result.ux
uy = result.uy

# `run_simulation` parses the `.krk` file, builds the lattice, and runs
# the time loop.  The returned fields are plain 2D arrays on the CPU.
#
# ---
#
# ## Velocity field

Nx, Ny = size(ux)
umag = sqrt.(ux .^ 2 .+ uy .^ 2)

using CairoMakie

fig = Figure(size = (900, 300))
ax  = Axis(fig[1, 1];
    title  = "Velocity magnitude |u| — cylinder Re = 20",
    xlabel = "x", ylabel = "y", aspect = DataAspect())
hm = heatmap!(ax, 1:Nx, 1:Ny, umag; colormap = :viridis)
Colorbar(fig[1, 2], hm; label = "|u|")
save(joinpath(@__DIR__, "03_cylinder_umag.svg"), fig)
fig

# The wake is clearly visible downstream of the cylinder.
#
# ---
#
# ## Key concepts
#
# | Concept | Directive | Purpose |
# |---------|-----------|---------|
# | User variables | `Define U = 0.05` | Avoid magic numbers, enable sweeps |
# | Geometry predicate | `Obstacle { ... }` | Flag solid nodes via boolean expr |
# | Spatial BC | `velocity(ux = f(y))` | Impose space-varying inlet profile |
# | Pressure outlet | `pressure(rho = 1)` | Open boundary for external flows |
# | Force monitoring | `Diagnostics [drag, lift]` | MEA-based force computation |
#
# **Next tutorial:** [Tutorial 4 — Thermal flows](04_thermal.md) adds
# temperature and buoyancy to the solver.
