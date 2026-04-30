# Capabilities matrix

This page is the source of truth for the public documentation of this branch.
It separates what the current code can run from what is only parsed, planned,
or present in another development branch.

For the planned integration path and the version where each future feature can
be checked as available, see the [integration roadmap](integration_roadmap.md).

Status legend:

- **supported**: present in `src/`, reachable from the public Julia API or
  `.krk` runner, and covered by examples or tests in this branch.
- **parser-only**: the `.krk` parser accepts the syntax, but the v0.1.0 runner
  rejects or ignores it.
- **not in this branch**: do not document as usable here.

## Supported workflow

The recommended user path is:

1. Write or copy a `.krk` file.
2. Run it with `run_simulation("case.krk")`.
3. Inspect VTK/PNG/GIF outputs.
4. Move to the Julia API only when scripting, callbacks, or custom
   post-processing are needed.

## Core LBM

| Capability | 2D | 3D | GPU path | `.krk` runner | Reference |
|---|:-:|:-:|:-:|:-:|---|
| D2Q9 lattice | supported | - | supported | supported | [Lattice](api/lattice.md) |
| D3Q19 lattice | - | supported | supported | cavity-style runs | [From 2D to 3D](theory/06_from_2d_to_3d.md) |
| BGK collision | supported | supported | supported | supported | [BGK](theory/03_bgk_collision.md), [Collision API](api/collision.md) |
| Guo body forcing | supported | supported | supported | supported for constant `Fx`, `Fy`, `Fz` | [Body forces](theory/07_body_forces.md) |
| MRT collision | not in this branch | not in this branch | - | - | planned/development |
| Axisymmetric LBM | not public in this branch | - | - | parser rejects at runner | planned/development |
| Grid refinement | parser-only | parser-only | - | rejected by runner | planned/development |

## Boundary conditions

| Boundary | 2D | 3D | Spatial expression | Time expression | Notes |
|---|:-:|:-:|:-:|:-:|---|
| Wall / bounce-back | supported | supported | - | - | no-slip walls |
| Velocity / Zou-He | supported | supported | west/south/north in 2D | limited | east velocity is not a general public path |
| Pressure outlet | east in 2D | east/top in 3D | limited | limited | use conservatively |
| Periodic | supported | supported | - | - | axis shorthand accepted |
| Fixed temperature | supported | supported | scalar only | scalar only | thermal module |
| Symmetry/outflow | parser-recognized | parser-recognized | - | - | not a validated public BC |

See [Boundary conditions](theory/05_boundary_conditions.md),
[Spatial BCs](theory/19_spatial_bcs.md), and [BC types](krk/bc_types.md).

## Thermal

| Capability | Status | Entry point |
|---|---|---|
| Passive thermal DDF | supported | `collide_thermal_2d!`, `collide_thermal_3d!` |
| Fixed-temperature walls | supported | `.krk` wall `T = ...`, thermal boundary kernels |
| Boussinesq natural convection 2D | supported | `run_natural_convection_2d`, `run_rayleigh_benard_2d` |
| Natural convection 3D | public Julia API, limited validation | `run_natural_convection_3d` |
| Temperature-dependent viscosity (`Rc`) | Julia API only | `run_natural_convection_2d(; Rc=...)` |
| Thermal grid refinement | not public in this branch | development branch only |

Locally rerun checks on 2026-04-30:

- Poiseuille convergence: order 2.00 over `Ny = 16, 32, 64, 128`.
- Taylor-Green convergence: order 1.99/2.00/2.00 over `N = 16, 32, 64, 128`.
- Thermal conduction: order 1.00 with the current fixed-temperature wall
  treatment, matching the documented half-cell boundary error.
- Natural convection at `Ra = 1e3`, `N = 64`: `Nu = 1.1423` versus
  De Vahl Davis `1.118`, relative error `2.17%`.

## `.krk` DSL

| Feature | Status | Reference |
|---|---|---|
| `Simulation`, `Domain`, `Physics`, `Run` | supported | [Directives](krk/directives.md) |
| `Define` and kwarg overrides | supported | [Directives](krk/directives.md) |
| `Boundary` | supported with the limits above | [BC types](krk/bc_types.md) |
| `Obstacle` / `Fluid` with expressions | supported | [Directives](krk/directives.md) |
| STL syntax | parser-only in this branch | [Directives](krk/directives.md) |
| `Initial` and `Velocity` expressions | supported for the generic 2D runner | [Expressions](krk/expressions.md) |
| `Module thermal` | supported | [Modules](krk/modules.md) |
| `Module axisymmetric` | parser-visible but runner rejects | [Modules](krk/modules.md) |
| `Refine` | parser-only; runner rejects | [Directives](krk/directives.md) |
| `Rheology` | parser-only; runner rejects | planned/development |
| `Sweep` | parser-supported | [Directives](krk/directives.md) |

## Output and diagnostics

| Output | Status | Syntax/API |
|---|---|---|
| VTK/PVD | supported | `Output vtk every 1000 [rho, ux, uy]` |
| PNG snapshots | supported when CairoMakie extension is loaded | `Output png every 500 [ux]` |
| GIF animations | supported when CairoMakie extension is loaded | `Output gif every 100 [ux] fps=15` |
| Diagnostics logging | supported | `Diagnostics every 100 [step, KE, uMax]` |
| ParaView helper | supported | `open_paraview("output/"; name="case")` |

## Hardware and benchmarks

The public benchmark story for this branch should use CPU baseline plus H100
only after the corresponding CSV files are present in `benchmarks/results/`.

Current traceable artifacts:

- Poiseuille convergence CSVs exist for `apple_m2` and `aqua_h100`; both match
  the local rerun exactly.
- CPU throughput CSVs exist for a legacy `apple_m2` label.
- Metal/M3 Max throughput exists as a local artifact, but should not be used
  as the headline comparison.
- H100 throughput numbers previously shown in the docs are not backed by a
  matching CSV in this branch and are therefore not published here.

See [Accuracy](benchmarks/accuracy.md), [Performance](benchmarks/performance.md),
and [Hardware](benchmarks/hardware.md).

## Not public in this branch

Do not present the following as usable v0.1.0 features in this documentation:

- MRT collision.
- Axisymmetric Hagen-Poiseuille.
- Grid refinement / FH patches.
- Multiphase VOF / PLIC.
- Phase-field.
- Shan-Chen.
- Species transport.
- Non-Newtonian rheology.
- Viscoelasticity.
- SLBM/body-fitted curvilinear work from `slbm-paper`.

Those items may exist in another branch or design document, but they need
separate validation and their own documentation pass before being advertised.
