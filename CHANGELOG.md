# Changelog

All notable changes to Kraken.jl will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] — 2026-04-14

### Added (since audit, 2026-04-13/14)
- **Unified .krk dispatch for refinement** (2D + 3D, isothermal + thermal)
  via `_run_refined` / `_run_refined_3d` — no dedicated driver needed.
- **Fine-grid sanity checks**: `τ_T_fine` (thermal refinement) and `N/Re`
  on refined patches, with `[2D]/[3D]` tag in the parameter summary.
- **Thermal BL resolution check**: warns when `N_eff < 3·Ra^(1/4)`,
  accounting for refinement ratios near thermal walls.
- **Capabilities matrix page** (`docs/src/capabilities.md`) listing every
  feature with status, links to theory/examples/API.
- **CLI wrapper** (`bin/krk`), VSCode `.krk` syntax highlighting, ASCII
  kwargs aliases (nu/rho/sigma/tau/…).

### Fixed (since audit)
- **Metal GPU refinement crash**: `trunc(Int,...)` replaced by
  `unsafe_trunc(Int,...)` in all 2D refinement, thermal-refinement, and
  dual-grid kernels (previously allocated on GPU → InvalidIRError).
- **3D FH kernels**: `stencil_clamped` guard removed — it forced α=0 at
  domain boundaries, which inflated Nu ~70% for 3D natconv refined.
  Root cause (prev buffer size) was already fixed in 534bb62.
- **test/Project.toml**: declares `KernelAbstractions` (was missing,
  causing `Pkg.test()` to error on Poiseuille 3D / thermal / species).
- **CI test suite**: `test_rheology.jl` and `test_viscoelastic.jl` added
  to `runtests.jl` (were present but not wired in).

## [0.1.0-dev] — 2026-04-10

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
- .krk runner does not dispatch on non-thermal refined cases
- Grid refinement cavity benchmark requires the Julia API directly
