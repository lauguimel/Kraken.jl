# Changelog

All notable changes to Kraken.jl will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.1.1] — 2026-05-14

Patch release.

### Fixed
- **Guo body force convention**: `collide_guo_2d!` is now explicitly declared
  as Convention I (integrated), and the post-collision readout uses
  `compute_macroscopic_2d!` instead of `compute_macroscopic_forced_2d!`. The
  previous pairing accumulated a `+gx/2`-per-step offset on the periodic-box
  mean velocity (5e-6 over 500 steps for `gx = 1e-5`). Production callsites
  updated in `src/simulation_runner.jl` and `src/drivers/basic.jl`. A
  regression test (`test/test_guo_convention_pairs.jl`) covers both the
  fixed production pair and the historically broken pair as a sentinel.

### Internal
- Docstring on `collide_guo_2d!` documenting its Convention I status and the
  canonical pair member `compute_macroscopic_2d!`.

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
