# Changelog

All notable changes to Kraken.jl will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] — 2026-04-10

### Added
- **Core LBM solver**: D2Q9 and D3Q19 lattices, BGK and MRT collision,
  Guo forcing, streaming (periodic, wall, axisymmetric)
- **Boundary conditions**: Zou-He velocity/pressure (2D+3D), bounce-back,
  spatially varying BCs via KrakenExpr expressions
- **Thermal LBM**: double distribution function with Boussinesq coupling,
  Rayleigh-Benard and natural convection drivers
- **Axisymmetric LBM**: Li et al. (2010) collision kernel,
  Hagen-Poiseuille pipe flow driver
- **Grid refinement**: patch-based with Filippova-Hanel rescaling,
  temporal interpolation, bilinear prolongation/restriction
- **.krk configuration DSL**: declarative simulation setup, presets
  (cavity_2d, poiseuille_2d, couette_2d, taylor_green_2d, rayleigh_benard_2d),
  Reynolds/Rayleigh helpers, sanity checks, spell-correction, parameter sweeps
- **KrakenView**: interactive CairoMakie-based viewer with heatmap, profile,
  convergence, and streamline figure types
- **GPU backends**: CPU, CUDA (H100/A100 tested), Metal (Apple Silicon)
  via KernelAbstractions.jl — single-source GPU portability
- **I/O**: VTK output (.vti/.pvd), STL import + voxelizer, diagnostics logger
- **Post-processing**: extract_line, probe, field_error, domain_stats
- **Documentation**: 13 theory pages, 11 validated examples-tutorials,
  dual API reference (Julia + .krk DSL), getting-started guide, cookbook
- **Benchmarks**: convergence studies (Poiseuille order 2, Taylor-Green order 2,
  cavity vs Ghia 1982), MLUPS performance (7675 MLUPS on H100)

### Known limitations
- Multiphase, rheology, viscoelastic, and species transport are implemented
  but not included in the v0.1.0 scope (available on dev branch)
- KrakenView is 2D only; 3D viewer planned for v0.2.0
- .krk runner does not dispatch on non-thermal refined cases
- Grid refinement cavity benchmark requires the Julia API directly
