```@raw html
---
layout: home

hero:
  name: "Kraken.jl"
  text: "GPU-native Lattice Boltzmann in Julia"
  tagline: "Composable, multi-backend CFD — single-phase, thermal, axisymmetric, grid-refined."
  image:
    src: /assets/showcases/vonkarman_re200.gif
    alt: Von Kármán vortex street at Re=200
  actions:
    - theme: brand
      text: Get Started
      link: /getting_started
    - theme: alt
      text: Examples
      link: /examples/04_cavity_2d
    - theme: alt
      text: View on GitHub
      link: https://github.com/lauguimel/Kraken.jl

features:
  - icon: <img width="64" src="/assets/showcases/rayleigh_benard_ra1e5.gif"/>
    title: Thermal convection
    details: Rayleigh-Bénard natural convection with Boussinesq coupling, validated Ra=1e3–1e8.
    link: /examples/08_rayleigh_benard
  - icon: <img width="64" src="/assets/showcases/taylor_green_decay.gif"/>
    title: Taylor-Green decay
    details: Canonical vortex decay — spectral accuracy on structured grids.
    link: /examples/03_taylor_green_2d
  - icon: <img width="64" src="/assets/showcases/cavity_re1000.gif"/>
    title: Lid-driven cavity
    details: Reference benchmark at Re=100–10000, 2D and 3D drivers.
    link: /examples/04_cavity_2d
  - icon: <img width="64" src="/assets/showcases/vonkarman_re200.gif"/>
    title: Flow past obstacles
    details: Cylinder drag, STL bodies, moving boundaries via momentum exchange.
    link: /examples/06_cylinder_2d
---
```

## Why Kraken.jl?

Kraken.jl provides a composable, high-performance LBM solver for incompressible
flows with automatic GPU acceleration via
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl).

- **Multi-backend GPU** — write once, run on CUDA, Metal (Apple Silicon), AMD ROCm, and CPU
- **D2Q9 & D3Q19 lattices** — standard lattice Boltzmann velocity sets
- **BGK & MRT collision** — single- and multiple-relaxation-time with Guo forcing
- **Boundary conditions** — bounce-back, Zou-He velocity/pressure, periodic, spatial/STL
- **Thermal LBM** — double distribution function with Boussinesq coupling
- **Axisymmetric LBM** — cylindrical coordinates via Li et al. (2010) scheme
- **Grid refinement** — patch-based nested refinement with Filippova-Hanel rescaling
- **Momentum exchange** — drag/lift computation for immersed bodies
- **VTK output** — `.vti` / `.pvd` for ParaView visualization
- **.krk DSL** — declarative, Gerris-like simulation config

## Quick Start

```julia
using Kraken

# Lid-driven cavity at Re = 100 on a 128×128 grid
N = 128
ν = 0.1 * N / 100  # ν = u_lid · N / Re
config = LBMConfig(D2Q9(); Nx=N, Ny=N, ν=ν, u_lid=0.1, max_steps=30000)
result = run_cavity_2d(config)
```

See [Installation](installation.md) and [Getting Started](getting_started.md) for the full setup.

## Physics Capabilities

| Capability | Lattice | Driver |
|:-----------|:--------|:-------|
| Lid-driven cavity | D2Q9, D3Q19 | `run_cavity_2d`, `run_cavity_3d` |
| Channel flow (Poiseuille) | D2Q9 | `run_poiseuille_2d` |
| Couette flow | D2Q9 | `run_couette_2d` |
| Taylor-Green vortex | D2Q9 | `run_taylor_green_2d` |
| Cylinder drag | D2Q9 | `run_cylinder_2d` |
| Thermal convection | D2Q9 | `run_rayleigh_benard_2d` |
| Axisymmetric pipe flow | D2Q9 | `run_hagen_poiseuille_2d` |

## References

```@bibliography
```
