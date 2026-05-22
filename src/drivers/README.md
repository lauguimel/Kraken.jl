# drivers/

High-level entry points — the `run_*` functions users actually call. Each
driver assembles a `BCSpec`, picks a kernel chain (collide / stream / BC),
wires output hooks, and runs the timestep loop.

## Key entry points

| File | Symbol | Purpose |
|---|---|---|
| `basic.jl` | `run_cavity_2d`, `run_cavity_3d`, `run_poiseuille_2d`, `run_couette_2d`, `run_taylor_green_2d` | Canonical benchmarks (single-phase Newtonian) |
| `cylinder_libb.jl` | `run_cylinder_libb_2d` | Cylinder with LI-BB (Lallemand-Luo interpolated bounce-back) — Schaefer-Turek validated |
| `thermal.jl` | `run_rayleigh_benard_2d`, thermal cavity | DDF-coupled thermal flows (Boussinesq) |
| `axisymmetric.jl` | `run_hagen_poiseuille_2d` | Axisymmetric pipe flow (Li 2010 scheme) |
| `multiphase.jl` | VOF / phasefield drivers | Two-phase flows (outside v0.1.0 scope) |
| `rheology.jl` | GNF drivers | Non-Newtonian (Power-law, Carreau, Bingham) |
| `viscoelastic.jl` | Oldroyd-B / FENE-P drivers | Polymeric flows (outside v0.1.0 scope) |

## Critical invariants

- Each `run_*` function returns a `SimulationResult` carrying the final
  population array, macroscopic fields, and any diagnostic time series
  requested.
- Drivers do NOT mutate global state — they accept a `backend` kwarg
  (default platform-dependent) and pass it through to all allocations.
- **`run_simulation(setup)` (in `src/simulation_runner.jl`) plants
  `setup.mesh` but the struct exposes `setup.domain` at `2d27bf68`** —
  this is a pre-existing source bug. Use the per-flow `run_*` entry
  points instead. See engineer memory of the voie D probe campaign.

## Cross-module dependencies

Reads from: every other physics module — `kernels`, `lattice`,
`refinement`, `multiblock`, `curvilinear`, `rheology` — depending on
which features the flow needs.
Reads from: `io` (when called via a `.krk` file path rather than the Julia
API directly).
Provides to: end-users; also called by `benchmarks/` scripts and the
test suite.

## Status / scope notes

- `viscoelastic.jl` and `multiphase.jl` ship outside v0.1.0 strict scope
  (announced in `release/v0.1.0` README).
- Drivers are deliberately the only module that knows about timestep
  loops and output cadence — pushing this concern up keeps `kernels/`
  composable.
