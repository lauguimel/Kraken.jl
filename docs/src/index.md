```@raw html
---
layout: home

hero:
  name: "Kraken.jl"
  text: ".krk-first Lattice Boltzmann simulations in Julia"
  tagline: "Run reproducible LBM cases from small text files, then drop into Julia when you need direct control."
  image:
    src: /assets/showcases/vonkarman_re200.gif
    alt: Vortex street simulation
  actions:
    - theme: brand
      text: Run a .krk file
      link: /getting_started
    - theme: alt
      text: .krk reference
      link: /krk/overview
    - theme: alt
      text: Julia API
      link: /api/public_api

features:
  - icon: <img width="64" src="/assets/showcases/cavity_re1000.gif"/>
    title: Reproducible cases
    details: Each public example is backed by a checked-in .krk file.
    link: /examples/04_cavity_2d
  - icon: <img width="64" src="/assets/showcases/taylor_green_decay.gif"/>
    title: Verified convergence
    details: Poiseuille and Taylor-Green convergence are rerun locally and CSV-backed.
    link: /benchmarks/accuracy
  - icon: <img width="64" src="/assets/showcases/rayleigh_benard_ra1e5.gif"/>
    title: Thermal DDF
    details: Heat conduction and Boussinesq natural convection are in the v0.1.0 scope.
    link: /examples/08_rayleigh_benard
  - icon: <img width="64" src="/assets/showcases/vonkarman_re200.gif"/>
    title: Agent context
    details: A compact llms.txt is shipped for LLM-assisted work.
    link: /llms
---
```

## Start from a `.krk` file

Kraken.jl's public workflow starts with a small declarative case file:

```text
Simulation cavity D2Q9
Domain     L = 1.0 x 1.0   N = 128 x 128
Setup      reynolds = 100 L_ref = 128 U_ref = 0.1
Boundary   north velocity(ux = 0.1, uy = 0)
Boundary   south wall
Boundary   east  wall
Boundary   west  wall
Run        10000 steps
Output     vtk every 1000 [rho, ux, uy]
```

Run it from Julia:

```julia
using Kraken

result = run_simulation("examples/cavity.krk"; max_steps=1000)
```

Or parse first when you want to inspect or override the setup:

```julia
setup = load_kraken("examples/cavity.krk"; Nx=256, Ny=256)
result = run_simulation(setup)
```

## v0.1.0 scope

The documentation for this branch is intentionally conservative. It covers
features that are present in `src/`, exercised by examples or tests, and
described by the `.krk` parser/runner in this branch.

| Area | Public status |
|---|---|
| D2Q9 / D3Q19 lattices | Supported |
| BGK collision | Supported |
| Guo body forcing | Supported |
| Wall, velocity, pressure, periodic BCs | Supported, with documented limits |
| Expression-based `.krk` geometry | Supported |
| Thermal DDF and Boussinesq coupling | Supported |
| VTK, PNG and GIF outputs | Supported |
| MRT, axisymmetric, grid refinement, VOF, rheology, viscoelasticity | Not public in this branch |

See the [capabilities matrix](capabilities.md) for the exact status, the
[integration roadmap](integration_roadmap.md) for planned features, and the
[LLM context page](llms.md) for the compact machine-readable summary.

## References

```@bibliography
```
