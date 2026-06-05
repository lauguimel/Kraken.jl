```@raw html
---
layout: home

hero:
  name: "Kraken.jl"
  text: "GPU-native multiphysics LBM in Julia"
  tagline: "Write a reproducible .krk case or a direct Julia driver, then run it on CUDA, Metal, or CPU backends."
  image:
    src: /logo.png
    alt: Kraken
  actions:
    - theme: brand
      text: Get started
      link: /getting_started
    - theme: alt
      text: KRK reference
      link: /users/krk-reference
    - theme: alt
      text: Validation matrix
      link: /users/benchmarks/validation-matrix
    - theme: alt
      text: Cite
      link: /#citing-kraken

features:
  - title: Reproducible cases
    details: Canonical flows can be launched from declarative .krk files or direct Julia drivers.
    link: /examples/04_cavity_2d
  - title: Validated multiphysics
    details: Cavity, thermal, sphere-drag, viscoelastic, and refinement checks are tracked in the v0.2 validation matrix.
    link: /users/benchmarks/validation-matrix
  - title: GPU certification
    details: Backend checks document CPU, CUDA-class, and Apple Silicon execution paths.
    link: /users/benchmarks/gpu-certification
  - title: Grid refinement
    details: Nested patches and route-native validation are documented for v0.2 refinement workflows.
    link: /examples/20_grid_refinement_cavity
---
```

```@raw html
<div align="center">
  <img src="./assets/showcases/vonkarman_re200.gif" alt="Von Kármán vortex street past a cylinder, simulated with Kraken.jl" width="100%"/>
  <br/><em>Von Kármán vortex street (Re = 200) — D2Q9 LBM on GPU.</em>
</div>
```

## Quick start

```julia
using Kraken

# Lid-driven cavity at Re = 100 on a 128×128 grid
N = 128
ν = 0.1 * N / 100  # ν = u_lid · N / Re
config = LBMConfig(D2Q9(); Nx=N, Ny=N, ν=ν, u_lid=0.1, max_steps=30000)
result = run_cavity_2d(config)
```

The same simulation can be written declaratively in a `.krk` file and launched
from a shell:

```
# cavity.krk — lid-driven cavity at Re = 100
Simulation cavity D2Q9
Domain     L = 1.0 x 1.0   N = 128 x 128
Physics    nu = 0.128
Boundary   north velocity(ux = 0.1, uy = 0)
Boundary   south wall
Boundary   east  wall
Boundary   west  wall
Run        30000 steps
```

```bash
krk cavity.krk
```

## Where to go next

- **[Installation](installation.md)** — set up Kraken.jl and its GPU backends.
- **[Getting started](getting_started.md)** — from zero to a running simulation.
- **[Concepts](concepts_index.md)** — the ideas behind the solver.
- **[Capabilities](capabilities.md)** — what the v0.2 release ships.
- **[Theory](theory/01_lbm_fundamentals.md)** — progressive chapters, from kinetic theory to lattice Boltzmann.
- **[Examples](examples/04_cavity_2d.md)** — validated runs with plots and convergence studies.
- **[Validation matrix](users/benchmarks/validation-matrix.md)** — every benchmark and its reference.
- **[Performance benchmarks](benchmarks/performance.md)** — MLUPS across CPU and GPU backends.
- **[`.krk` reference](users/krk-reference.md)** — the declarative configuration language.
- **[API reference](api/config.md)** — every public function, documented.

## Physics capabilities

Kraken is organised as a set of physics modules that share one kinetic core,
so the same BGK/MRT solver, body forcing, and boundary machinery extend across
every regime.

| Module | What it enables | Representative drivers |
|:-------|:----------------|:-----------------------|
| **Newtonian** | Incompressible flow in 2D, 3D, and axisymmetric geometries (BGK & MRT collision, Guo forcing) | `run_cavity_2d`, `run_cavity_3d`, `run_poiseuille_2d`, `run_cylinder_2d`, `run_hagen_poiseuille_2d` |
| **Thermal** | Boussinesq natural convection via a coupled double-distribution temperature field | `run_rayleigh_benard_2d` |
| **Viscoelastic** | Oldroyd-B polymer stress coupled to the flow ([validated vs RheoTool](users/benchmarks/viscoelastic-cylinder.md)) | `run_viscoelastic_logfv_channel_2d`, `run_viscoelastic_cylinder_2d` |
| **Grid refinement** | Nested patch refinement with Filippova–Hänel rescaling and route-native validation | `run_cavity_2d` with `Refine { … }` patches |
| **Geometry / immersed boundaries** | STL-driven masks, voxelisation, and linearly-interpolated bounce-back for curved walls | `run_sphere_libb_3d`, STL loader + cut-link drag |
| **GPU backends** | One physics layer, multiple runtimes selected at launch — CUDA, Apple Silicon (Metal), and multi-threaded CPU | every driver, via the backend argument |

See the [capabilities page](capabilities.md) for the full module breakdown and
the [validation matrix](users/benchmarks/validation-matrix.md) for the
literature references each module is checked against.

## Showcase gallery

```@raw html
<div align="center">
  <img src="./assets/showcases/cavity_re1000.gif" alt="Lid-driven cavity at Re = 1000" width="32%"/>
  <img src="./assets/showcases/rayleigh_benard_ra1e5.gif" alt="Rayleigh–Bénard convection at Ra = 1e5" width="32%"/>
  <img src="./assets/showcases/taylor_green_decay.gif" alt="Taylor–Green vortex decay" width="32%"/>
  <br/>
  <em>Left: lid-driven cavity at Re = 1000 (primary vortex + corner eddies).
  Centre: Rayleigh–Bénard convection cells at Ra = 1e5.
  Right: Taylor–Green vortex decay.</em>
</div>
```

Every animation on this page is produced by the validated drivers in the table
above; reproduce them from the [examples](examples/04_cavity_2d.md) and the
[validation matrix](users/benchmarks/validation-matrix.md).

## Citing Kraken

If you use Kraken.jl in your research, please cite it. The repository ships a
[`CITATION.cff`](https://github.com/lauguimel/Kraken.jl/blob/main/CITATION.cff),
so GitHub's **"Cite this repository"** button (top-right of the
[repository](https://github.com/lauguimel/Kraken.jl)) gives a ready-made entry.

```bibtex
@software{kraken_jl,
  author  = {Maitrejean, Guillaume and Sauret, Emilie},
  title   = {{Kraken.jl}},
  year    = {2026},
  version = {0.2.0},
  url     = {https://github.com/lauguimel/Kraken.jl}
}
```

A citable **DOI** (Zenodo) will be added with the archived `v0.2.0` release —
update the `doi` field in `CITATION.cff` and the `doi = {...}` line above once it
is minted.
