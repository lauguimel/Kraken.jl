# Build a KRK, step by step

We construct a complete `.krk` file for Poiseuille flow from nothing —
one directive at a time. By the end you'll recognize every block of a
Kraken config file and understand why each one is there.

**Goal:** reproduce the canonical Poiseuille test case:

- 2D channel, periodic in x, no-slip walls top/bottom
- Incompressible flow driven by a constant body force
- Analytical target: parabolic profile
  $u_x(y) = \frac{F_x}{2\nu}\, y (L_y - y)$

## Step 1 — the `Simulation` header

Every `.krk` file begins by naming the run and picking a lattice:

```krk
Simulation poiseuille D2Q9
```

- `poiseuille` is an identifier used in output filenames (`output/poiseuille_*.vtr`)
  and log tags. Rename it freely.
- `D2Q9` is the 2D velocity set. Use `D3Q19` for 3D.

No other lattice families are accepted in v0.1.0; see
[Theory → D2Q9](../../theory/02_d2q9_lattice.md) for the standard.

## Step 2 — the `Domain` block

Give the solver a physical extent and a grid:

```krk
Domain L = 0.125 x 1.0   N = 4 x 32
```

- `L = Lx x Ly` — physical size (lattice units). The literal `x` is
  the separator.
- `N = Nx x Ny` — grid resolution.

A narrow channel (4 cells wide, 32 tall) is enough because we only care
about the cross-stream profile — periodicity in x makes the flow
effectively 1D.

## Step 3 — the `Physics` block

Set kinematic viscosity and body force:

```krk
Physics nu = 0.1  Fx = 1e-5
```

- `nu` — kinematic viscosity. Converts to LBM relaxation time
  $\tau = 3\nu + 0.5$ at run time.
- `Fx` — constant body force along x, uniform across the domain. Acts
  like a pressure gradient driving the flow.

`kraken info` will warn if `τ < 0.55` (unstable) or if the resulting
Mach number exceeds 0.1 (incompressibility breakdown).

## Step 4 — the `Boundary` lines

Boundary conditions follow the face names of the domain:

```krk
Boundary x     periodic
Boundary south wall
Boundary north wall
```

- `Boundary x periodic` — periodicity in the x direction (pairs east ↔
  west). Matches the body-force-driven assumption.
- `Boundary south wall` / `Boundary north wall` — no-slip bounce-back on
  the top and bottom walls.

Face aliases (`south`, `north`, `east`, `west`, …) are listed in the
[BC types reference](../../krk/bc_types.md).

## Step 5 — the `Run` directive

Pick how long to integrate:

```krk
Run 10000 steps
```

Poiseuille reaches steady state in $O(L_y^2 / \nu)$ lattice steps. With
$L_y = 32$ and $\nu = 0.1$ that's ~10 000 steps — hence the choice.

## Step 6 — the `Output` directive

Declare what to dump and how often:

```krk
Output vtk every 2000 [rho, ux, uy]
```

- `vtk` — structured-grid `.vtr` files, loadable in ParaView.
- `every 2000` — snapshot cadence. Five files total for this run.
- `[rho, ux, uy]` — fields to write. Add `T` when the thermal module is
  active.

See [Output reference](../../krk/directives.md) for PVD collection and
custom field names.

## The complete file

Put it all together:

```krk
# examples/poiseuille.krk
Simulation poiseuille D2Q9
Domain  L = 0.125 x 1.0  N = 4 x 32
Physics nu = 0.1  Fx = 1e-5

Boundary x periodic
Boundary south wall
Boundary north wall

Run 10000 steps
Output vtk every 2000 [rho, ux, uy]
```

That's 8 non-blank lines for a fully-specified, reproducible channel
flow simulation.

## Run it

```bash
kraken run examples/poiseuille.krk
```

Open `output/poiseuille.pvd` in ParaView. A cross-cut through the channel
at `x = Lx/2` should match the parabolic target

$$
u_x(y) = \frac{F_x}{2\nu}\, y\,(L_y - y)
\quad\Rightarrow\quad u_{\max} = \frac{F_x L_y^2}{8\nu} = 1.25 \times 10^{-3}.
$$

If your run disagrees, inspect the setup first:

```bash
kraken info examples/poiseuille.krk
```

## What's next?

Ready to go beyond the canonical shape? The [Cookbook](03_cookbook.md)
collects copy-pasteable fragments for the most common extensions:

- Add an obstacle (analytic shape or STL)
- Set a parabolic inlet profile
- Turn on thermal coupling with `Module thermal`
- Set grid refinement patches
- Start from a preset and override only what you need
