# Kraken.jl

[![Build Status](https://github.com/lauguimel/Kraken.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/lauguimel/Kraken.jl/actions/workflows/CI.yml)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://lauguimel.github.io/Kraken.jl/dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A GPU-portable Lattice Boltzmann (LBM) framework written in Julia, targeting
single-phase incompressible and thermal flows on CPU, CUDA, and Apple Metal
backends. Kraken.jl provides a compact kernel core (D2Q9, D3Q19), a
declarative `.krk` configuration DSL, and spatial boundary conditions — all
behind a small, hackable API.

## Scope (v0.1.0)

- Single-phase Newtonian LBM in 2D and 3D (BGK collision)
- Thermal flows via double-distribution function (DDF) coupling
- Boundary conditions: Zou-He velocity/pressure, bounce-back, periodic, outflow
- Spatial and time-dependent boundary expressions
- `.krk` configuration DSL for parametric runs
- GPU-portable kernels via `KernelAbstractions.jl` (CPU / CUDA / Metal)
- VTK output for ParaView post-processing

## Installation

Kraken.jl is not yet registered. Install from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/lauguimel/Kraken.jl")
```

For GPU execution, also add the backend of your choice (`CUDA.jl` or
`Metal.jl`).

## Quickstart

### From a `.krk` configuration file

```julia
using Kraken

result = run_simulation("examples/cavity.krk")

# Inspect the final velocity field
ux, uy = result.ux, result.uy
```

### From the Julia API

```julia
using Kraken

config = LBMConfig(D2Q9(); Nx=128, Ny=128, ν=0.1, u_lid=0.1, max_steps=20000)
result = run_cavity_2d(config)
```

### Parametric override via `.krk`

```julia
result = run_simulation("examples/cavity.krk"; Nx=256, Ny=256, nu=0.05)
```

## Examples

| Example | Physics | .krk |
|---------|---------|------|
| Poiseuille flow | Body-force driven channel | `poiseuille.krk` |
| Couette flow | Shear-driven channel | `couette.krk` |
| Taylor-Green vortex | Decaying vortex (periodic) | `taylor_green.krk` |
| Lid-driven cavity 2D | Recirculating flow | `cavity.krk` |
| Lid-driven cavity 3D | 3D extension | `cavity_3d.krk` |
| Cylinder flow | Obstacle via predicate | `cylinder.krk` |
| Heat conduction | 1D thermal diffusion | `heat_conduction.krk` |
| Rayleigh-Benard | Buoyancy-driven convection | `rayleigh_benard.krk` |

See `docs/` and `examples/` for full documentation with validation against
analytical solutions and reference data (Ghia et al. 1982, De Vahl Davis 1983).

## Documentation

Full documentation (theory, examples, API reference) is built with
Documenter.jl:

```bash
julia --project=docs docs/make.jl
```

## Citation

If you use Kraken.jl in academic work, please cite:

```bibtex
@software{kraken_jl,
  author  = {Maitrejean, Guillaume},
  title   = {Kraken.jl: a GPU-portable Lattice Boltzmann framework in Julia},
  year    = {2026},
  url     = {https://github.com/lauguimel/Kraken.jl},
  version = {0.1.0},
}
```

## License

Kraken.jl is released under the MIT License. See [LICENSE](LICENSE) for
details.
