# Changelog

All notable changes to Kraken.jl will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.3.1] — 2026-07-24

### Added
- HPC cluster guide (`users/hpc`): user-space juliaup install, CUDA runtime
  pinning gotcha, complete PBS Pro example (Aqua, QUT), SLURM variant,
  Apptainer note, batch best practices.

### Fixed
- `OldroydB` name clash between the legacy driver spec and the rheology model
  (renamed `OldroydBSpec`) — broke precompilation on Julia 1.11.
- Known test failures from the v0.3.0 cut (#16): rtol-based conservative-tree
  assertions, streaming thresholds with margin, STL fixture path.
- Requires Julia >= 1.11 (extension mechanism); stdlib compat widened.

## [0.3.0] — 2026-07-22

### Added
- **Platform contract layer** (`src/platform/`): `AbstractProblem/Method/Solution/
  Observable/Closure`, `Capability` enum, and the verbs `solve`/`sample`/`observe`/
  `predict` — LBM wrapped bit-for-bit; `IncNS` runs under the same contract.
- **Residual/adjoint seam**: `residual` + `adjoint_vjp` over four parameter natures
  (geometry, scalar ν, ν(x) field, thermal), delegating to the validated AD paths.
- **Calibration stack**: `ParameterSpace`, `loss`, `fit` (projected Barzilai–Borwein
  + Armijo, zero new deps) and `fit(method=:lbfgs)` with Tikhonov regularisation via
  the `KrakenOptimExt` weak dependency. Twin experiments: scalar ν recovered to 4% in
  5 iterations; sine ν(y) field to 7.5% rel-L2.
- **Steady incompressible Navier–Stokes co-solver** (FVFD): SIMPLE/SIMPLEC with
  2nd-order scalar and momentum convection (implicit upwind + deferred correction),
  matrix-free multigrid Poisson, steady scalar transport with dedicated BCs.
- **GPU ablation ladder** (lid-driven cavity): A100 32.9×, H100 44.1× (C3, F64),
  RTX A6000 36.3× (C4 mixed precision — the winning rung on consumer silicon);
  1024² converges in 100.4k iterations, Ghia error 2.05%.
- **LinearSolve.jl front-end + cuDSS direct solve** behind `[weakdeps]`
  (`solve_poisson_direct`): assembled sparse alternative to the MG path on the same
  discretization; CPU MMS order 2.00, GPU-validated (parity ≤1.6e-12).

### Changed
- Documentation reframed method-agnostic (architecture page, maturity table) — the
  DocumenterVitepress site from the v0.2 line is the canonical doc toolchain.

### Fixed
- CUDSS extension trigger: CUDA removed from `[weakdeps]` (it is a strong dep;
  extensions use parent strong deps since Julia 1.11). Known Julia 1.12 caveat
  documented: sibling declared weakdeps must be co-installed for the CUDSS
  extension to precompile.

## v0.2.1

Documentation patch — no functional or source changes.

- Landing page: reworked navigation and the "What Kraken can do" capability grid.
- Examples: per-example `.krk` download dropdowns; boundary-condition schematics
  redrawn with a shared toolkit.
- DocumenterVitepress theme polish (custom CSS); `.krk` syntax highlighting;
  velocity-field lead plots; axisymmetric reference; theory-page cleanups.

## v0.2.0

Multiphysics release.

- **Units module** (LU ↔ physical): explicit conversion between lattice
  units and physical SI quantities for setup and post-processing.
- **Geometry / STL immersed boundary**: arbitrary solid geometries via STL
  import with cut-link (interpolated bounce-back) boundary treatment.
- **Viscoelastic Oldroyd-B cylinder**: validated to within <1% of RheoTool
  reference drag.
- **Thermal natural convection**: validated against the de Vahl Davis
  differentially heated cavity benchmark.
- **GPU certification**: reference benchmarks certified on GPU backends.

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
