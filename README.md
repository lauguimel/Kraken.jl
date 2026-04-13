# Kraken.jl

[![Build Status](https://github.com/lauguimel/Kraken.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/lauguimel/Kraken.jl/actions/workflows/CI.yml)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://lauguimel.github.io/Kraken.jl/dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A GPU-portable Lattice Boltzmann (LBM) framework written in Julia, targeting
single-phase incompressible and thermal flows on CPU, CUDA, and Apple Metal
backends. Kraken.jl provides a compact kernel core (D2Q9, D3Q19), a
declarative `.krk` configuration DSL, patch-based grid refinement, and
spatial boundary conditions — all behind a small, hackable API.

## Scope (v0.1.0)

- Single-phase LBM in 2D and 3D (BGK and MRT collisions)
- Thermal flows via double-distribution (DDF) coupling
- Patch-based nested grid refinement (Filippova–Hänel rescaling)
- Spatial boundary conditions (Zou–He, bounce-back, periodic, outflow)
- `.krk` configuration DSL for parametric runs
- GPU-portable kernels via `KernelAbstractions.jl` (CPU / CUDA / Metal)

Multiphase flows, non-Newtonian rheology, and viscoelastic models are
present in the source tree but are not part of the v0.1.0 public API.

For a complete, up-to-date feature matrix — with status (✓/~/✗), links to
theory pages, examples, and API — see the
[**Capabilities matrix**](docs/src/capabilities.md).

## Installation

Kraken.jl is not yet registered. Install from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/lauguimel/Kraken.jl")
```

For GPU execution, also add the backend of your choice (`CUDA.jl` or
`Metal.jl`).

## Quickstart

Run a 2D lid-driven cavity from a `.krk` configuration file:

```julia
using Kraken

# Load a parametric configuration
cfg = Kraken.load_krk("examples/configs/cavity_2d.krk")

# Run the simulation (CPU by default; set backend=:cuda or :metal for GPU)
sol = Kraken.run(cfg; backend = :cpu)

# Inspect the final velocity field
ux, uy = sol.u[:, :, 1], sol.u[:, :, 2]
```

Or build a case directly from Julia:

```julia
using Kraken

grid   = Kraken.Grid2D(Nx = 256, Ny = 256)
params = Kraken.LBMParams(nu = 1e-3, u_lid = 0.1)
bcs    = Kraken.cavity_bcs(grid)

sol = Kraken.simulate(grid, params, bcs; nsteps = 10_000, backend = :cpu)
```

See `docs/` and the `examples/` directory for more cases (Poiseuille,
Couette, Taylor–Green, cylinder flow, Rayleigh–Bénard, Hagen–Poiseuille,
3D cavity, and grid-refinement demos).

## Features

- D2Q9 and D3Q19 lattices
- BGK and MRT collision operators
- Thermal coupling via double-distribution functions
- Patch-based nested grid refinement with conservative rescaling
- Declarative `.krk` DSL for reproducible parametric studies
- GPU-portable kernels: single source, runs on CPU / CUDA / Metal
- VTK output for ParaView post-processing
- ~2000 unit tests covering kernels, BCs, refinement, and end-to-end drivers

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
