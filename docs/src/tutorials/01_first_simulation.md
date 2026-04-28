```@meta
EditURL = "01_first_simulation.jl"
```

# Tutorial 1 — Your First LBM Simulation

Kraken.jl is a GPU-portable Lattice Boltzmann Method (LBM) framework
written in Julia. It provides a `.krk` domain-specific language so you
can set up and run fluid-flow simulations without writing any Julia code.

In this tutorial you will simulate the **lid-driven cavity** — the
canonical first test for any CFD code — and learn every directive in
the `.krk` file format along the way.

---

## 1. The `.krk` configuration file

Here is the complete file `cavity.krk`:

```
Simulation cavity D2Q9
Domain  L = 1.0 x 1.0  N = 128 x 128
Physics nu = 0.128

Boundary north velocity(ux = 0.1, uy = 0)
Boundary south wall
Boundary east  wall
Boundary west  wall

Run 60000 steps
Output vtk every 10000 [rho, ux, uy]
```

Let us go through it **line by line**.

### `Simulation cavity D2Q9`

Declares a simulation named `cavity` using the **D2Q9** lattice (2
dimensions, 9 discrete velocities). D2Q9 is the standard lattice for
2D incompressible flows.

### `Domain L = 1.0 x 1.0  N = 128 x 128`

The physical domain is a 1 × 1 square, discretised on a 128 × 128
lattice. The lattice spacing is ``\Delta x = L_x / N_x``. In LBM the
lattice spacing and timestep are both unity in lattice units, so all
physical parameters are expressed in that system.

### `Physics nu = 0.128`

Sets the kinematic viscosity ``\nu = 0.128`` (lattice units). The BGK
relaxation rate is ``\omega = 1/(3\nu + 0.5) \approx 1.19``. For
stability, ``\omega`` must stay in ``(0, 2)``.
With ``u_\text{lid} = 0.1`` and ``N = 128`` this gives
``\text{Re} = u_\text{lid} \cdot N / \nu = 100``.

### `Boundary north velocity(ux = 0.1, uy = 0)`

The **north** (top) wall moves at horizontal velocity 0.1. This is
imposed via a **Zou--He velocity** boundary condition, which solves for
the unknown populations analytically.

### `Boundary south/east/west wall`

The three remaining sides use **half-way bounce-back** — the simplest
no-slip condition. Populations hitting a wall are reflected back,
placing the effective wall surface half a cell from the boundary node.

### `Run 60000 steps`

The solver performs 60 000 time-steps (stream → boundary → collide →
macroscopic). At Re = 100 this is enough for steady state.

### `Output vtk every 10000 [rho, ux, uy]`

Write VTK files every 10 000 steps for post-processing in ParaView.

---

## 2. Running the simulation

A single Julia call is all you need:

```julia
using Kraken

result = run_simulation(
    joinpath(@__DIR__, "..", "..", "..", "examples", "cavity.krk"))
```

`result` is a `NamedTuple` with fields `ρ`, `ux`, `uy` (CPU arrays)
and `setup` (the parsed configuration).

---

## 3. Inspecting the results

```julia
typeof(result.ρ)   # Matrix{Float64}, size (128, 128)
extrema(result.ux) # horizontal velocity range
extrema(result.uy) # vertical velocity range
```

---

## 4. Visualisation — velocity magnitude

We plot the magnitude ``|\mathbf{u}|`` normalised by the lid velocity.

```julia
using CairoMakie

N     = 128
u_lid = 0.1
umag  = sqrt.(result.ux .^ 2 .+ result.uy .^ 2) ./ u_lid

fig = Figure(size = (520, 450))
ax  = Axis(fig[1, 1]; xlabel = "x", ylabel = "y",
           title  = "Velocity magnitude |u|/u_lid — Re = 100",
           aspect = DataAspect())
hm = heatmap!(ax, 1:N, 1:N, umag; colormap = :viridis)
Colorbar(fig[1, 2], hm; label = "|u| / u_lid")
save(joinpath(@__DIR__, "01_cavity_umag.svg"), fig)
fig
```

The primary recirculation vortex is clearly visible: fast flow near the
lid and a quiet core in the centre.

---

## 5. Parametric override

You can override any `.krk` parameter from Julia without editing the
file. Here we halve the resolution and lower the viscosity:

```julia
result_coarse = run_simulation(
    joinpath(@__DIR__, "..", "..", "..", "examples", "cavity.krk");
    Nx = 64, Ny = 64, nu = 0.064, max_steps = 30000)
```

This is handy for quick convergence tests or parameter sweeps.

---

## 6. What we learned

| Directive | Meaning |
|-----------|---------|
| `Simulation name lattice` | Declare case name and lattice type |
| `Domain L = ... N = ...` | Physical size and grid resolution |
| `Physics nu = ...` | Kinematic viscosity (lattice units) |
| `Boundary face type(...)` | Set BCs per face |
| `Run N steps` | Number of iterations |
| `Output vtk every N [fields]` | VTK snapshots for ParaView |

Everything goes through the `.krk` file — the Julia API
(`run_simulation`) just reads it and runs it.

**Next:** [Tutorial 2 — Body forces (Poiseuille flow)](02_body_forces.md)

