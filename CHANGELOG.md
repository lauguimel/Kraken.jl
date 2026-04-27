# Changelog

All notable changes to Kraken.jl will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] — 2026-04-27

First public release. Scope: single-phase Newtonian + thermal (DDF) flows.

### Features
- **Core LBM solver**: D2Q9 and D3Q19 lattices, BGK collision,
  Guo discrete forcing, streaming (periodic, wall)
- **Boundary conditions**: Zou-He velocity/pressure (2D+3D), bounce-back,
  spatially varying and time-dependent BCs via KrakenExpr expressions
- **Thermal LBM**: double distribution function with Boussinesq coupling,
  Rayleigh-Benard and natural convection drivers (2D+3D)
- **.krk configuration DSL**: declarative simulation setup, presets
  (cavity_2d, poiseuille_2d, couette_2d, taylor_green_2d, rayleigh_benard_2d),
  parametric overrides, sanity checks, spell-correction
- **GPU backends**: CPU, CUDA, Metal via KernelAbstractions.jl
- **I/O**: VTK output (.vti/.pvd), diagnostics logger
- **Post-processing**: extract_line, probe, field_error, domain_stats
- **VS Code extension**: `.krk` syntax highlighting, IntelliSense, validation
- **CLI wrapper**: `bin/kraken run/info` for command-line usage
- **Documentation**: theory pages, 9 validated examples, API reference

### Validated benchmarks
- Poiseuille flow (2nd order convergence)
- Couette flow (machine precision)
- Taylor-Green vortex decay
- Lid-driven cavity 2D (Ghia et al. 1982)
- Lid-driven cavity 3D
- Cylinder flow (drag validation)
- Heat conduction (1D profile)
- Rayleigh-Benard convection (De Vahl Davis 1983)
