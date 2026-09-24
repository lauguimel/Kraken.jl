# Changelog

All notable changes to Kraken.jl will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Changed
- Electroconvection `.krk` cases now apply paired `Boundary west wall` / `Boundary east wall` as stationary no-slip sidewalls instead of ignoring them; omitted sides retain free slip. Explicit non-lateral `Boundary` declarations are rejected because electrode/plate conditions are built into this driver.

### Fixed
- **Corrected the 0.5.0 notes** (#52). The four log-conformation closures run on the
  FVFD path only; the LBM-CDE drivers (sphere, Couette, Poiseuille) accept Oldroyd-B
  only, in direct conformation form. Couette and LBM-CDE Poiseuille are not reachable
  from `.krk`, the planar extension is reachable in imposed-velocity mode only, and the
  Oldroyd-B sphere driver is not validated against a viscoelastic reference. Several
  figures were misstated: the FVFD near-wall and velocity errors per resolution, the
  planar-extension `C_yy` error and the cause of its 0.39 % `C_xx` gap, the number of
  `@test_broken` (13, not 22) and of files over 700 lines (13, not 8), and `L² → ∞`
  "byte-for-byte". The LBM-CDE Poiseuille error is now listed as a known issue of this
  release rather than as a changed result. `docs/src/capabilities.md` §9 now lists the
  3D viscoelastic models instead of reporting rheology as 2D-only, and
  `docs/src/users/benchmarks/ve3d-poiseuille-convergence.md` states the velocity error
  per resolution.
- **3D FVFD advection is second order next to a periodic z face** (#54). The
  MUSCL-Superbee operator fell back to first-order upwind within two cells of the z
  faces even when z is periodic, i.e. everywhere in a thin quasi-2D box. The 0.5.0
  planar-extension canary moves from 0.39 % to 0.0023 % on `C_xx` (the 0.5.0 notes
  attributed the 0.39 % to slow relaxation; it was this fallback), and the FENE-P
  canary now matches its transcendental fixed point to 1e-5. The coupled
  planar-extension test moves from 0.06 % to 0.83 % on `C_xx` against the nominal
  strain rate (gate 1 %): the coupled flow runs 1.4 % above the nominal rate, which
  the old fallback happened to offset; against the measured rate it is -0.59 %.
- `compute_polymeric_force_3d!` takes `wall_x` / `wall_z`, giving x and z walls the
  second-order one-sided difference the y walls use (a clamped stencil returned half
  the derivative there). Faces that are neither periodic nor walls keep the clamp.
- `logfv_max_grad_norm_3d` reduces on the arrays' device instead of copying nine
  fields to the host every step.

## [0.5.0] — 2026-09-24

### Added
- **3D viscoelastic flow, four constitutive models** (`src/kernels/logconformation_fv_3d.jl`,
  `src/kernels/logconformation_lbm_3d.jl`, `src/fvfd/operators_3d*.jl`,
  `src/drivers/viscoelastic_*_3d.jl`, `src/rheology/linalg_3d.jl`): log-conformation
  constitutive step in 3D with Oldroyd-B, FENE-P, Giesekus and PTT (linear and
  exponential), on the FVFD finite-volume grid (`run_viscoelastic_fvfd_poiseuille_3d`,
  `run_viscoelastic_fvfd_extensional_3d`). The LBM-CDE drivers (Oldroyd-B sphere, planar
  Couette, planar Poiseuille) evolve the conformation directly and accept Oldroyd-B only.
  Reachable from `.krk`: the sphere, FVFD Poiseuille and FVFD planar extension (imposed
  velocity only; the coupled mode is Julia-only); Couette and LBM-CDE Poiseuille are
  Julia-only. The sphere driver is not validated against a viscoelastic reference: its
  tests check the Newtonian limit against Kraken's Newtonian sphere driver and a NaN-free
  run at `Wi = 0.01`. Each closure matches its closed-form steady simple-shear fixed point
  (Giesekus, PTT residual ≤ 1e-6; FENE-P and the Oldroyd-B limit ≤ 1e-3); setting `α = 0`
  or `ε = 0` recovers the Oldroyd-B trajectory byte for byte, and FENE-P approaches it as
  `L²` grows (1.3e-8 on `C_xx` at `L² = 1e8` in planar extension).
  *Corrected 2026-09-24 (#52). The entry as tagged said the four closures ran "on either
  the LBM lattice or the FVFD finite-volume grid", that all five drivers were "reachable
  from `.krk`", and that `L² → ∞` recovered Oldroyd-B "byte-for-byte"; none was true.*
- **FVFD transport cures the near-wall conformation error of the LBM-CDE path.** On
  identical `N_y = 32` viscoelastic Poiseuille, the near-wall `C_xy` error is 3.9e-15
  against an anti-tautological reference built from the measured shear (at most 1.9e-7
  absolute, 3.9e-7 relative, over `N_y = 32–128`), where the diffusive LBM-CDE path
  returns 25.9 % error and a peak velocity ratio of ≈ 1.13. The FVFD peak velocity is
  within 0.086 % of the analytic parabola at `N_y = 32` and 0.0095 % at `N_y = 128` (H100,
  CUDA Float64). Both paths ship in this release; see Known issues for LBM-CDE.
  *Corrected 2026-09-24 (#52). The entry as tagged said "machine-exact (≤ 1.9e-7, i.e.
  ≤ 2e-5 %)" and "to 0.01 %" at `N_y = 32`: 1.9e-7 is the absolute `N_y = 128` value, and
  the velocity is within 0.01 % only at `N_y = 128`.*
- **Resumable simulation state and HDF5 checkpoints** (#26) (`src/platform/state.jl`,
  `src/io/checkpoint_hdf5.jl`, `src/drivers/ehd_ec_state.jl`): the `init_state` /
  `advance!` / `solution` contract, checkpoints written as HDF5, and the electroconvection
  driver running on the contract bit-identically to the frozen legacy driver. An
  interrupted run resumes instead of restarting. New direct dependency: `HDF5`.
  Documented in `docs/src/users/simulation-state-checkpoints.md` and `docs/src/api/platform.md`.
- **RheoTool cross-validation** of FVFD-3D Poiseuille, Oldroyd-B planar extension and
  FENE-P planar extension. Runs performed in a local `openfoam9-rheotool:v1.2` container;
  OpenFOAM case templates under `bench/rheotool/`, raw output and comparison tables under
  `benchmarks/results/rheotool_compare/`. On Poiseuille, Kraken's relative L2 error against
  the analytic profile is 1.2e-3 on velocity and 4.1e-9 on `N1`, where RheoTool gives 1.3e-3
  and 4.6e-3. On Oldroyd-B planar extension RheoTool reaches the analytic fixed point to
  machine precision and Kraken's 1000-step canary sits 0.39 % below on `C_xx` and 0.0015 %
  off on `C_yy`; these gaps come from a first-order advection fallback in the periodic z
  direction (#54), not from a slow relaxation as the tagged entry said. The 10.9 % FENE-P
  `C_xx` gap is a closure-variant difference (Peterlin argument `tr C` vs `tr A`), not a
  defect. These are benchmarks, not automated gates: no test reads the reference data.
- **Steady-state time estimate and viscoelastic parameter-stability check**
  (`src/units/steady_state.jl`).
- **Finite-difference cross-check of the viscoelastic polymer-drag adjoint** (#41)
  (`test/ad/test_ad_ve_fd_check.jl`): four design directions against central finite
  differences of the reconverged primal, with a three-point Richardson error estimate. The
  19 pre-existing adjoint assertions all pass under a known miscompile; this one does not.
- **References page** in the documentation (#46). `DocumenterCitations` was loaded and
  `docs/refs.bib` held 39 entries, but no page carried a `@bibliography` block, so all 37
  cited keys rendered as unlinked text.

### Changed
- The test suite is a two-tier merge gate (#24, #27, #30, #31): a fast pull-request tier, a
  full tier on the integration branch and nightly, superseded runs cancelled, and any
  single test file runnable on its own locally or on CI.
- The four Enzyme-driven test files run with production bounds checking
  (`--check-bounds=auto`) (#39, #41, #43). Under the `Pkg.test` default Enzyme's reverse
  mode miscompiles them; reported upstream as EnzymeAD/Enzyme.jl#3614.
- Known failures are carried as `@test_broken` rather than kept out of the suite (#29);
  the full CI run of this release carries 13 (12 in the LBM tier, 1 in the IncNS tier;
  the tagged entry said 22).
- The contribution model replaces the `src/` edit ban with directory ownership and review
  (#25); `AGENTS.md` and `.github/CODEOWNERS` carry the rules.
- `CITATION.cff`: G. Maitrejean's affiliation is now Grenoble INP, Université Grenoble Alpes.

### Fixed
- Type inference no longer expands the Enzyme shape-sensitivity thunks when
  `run_simulation` is called without automatic differentiation (#39, #40), which crashed
  the full test tier on Julia 1.12 / x86_64 Linux.
- Two IncNS tests asserted round-off residues against macOS literals; they now assert
  bounds (#36).
- `docs/node_modules` is no longer tracked (#45). The committed copy held the macOS
  `rollup` binary, so the documentation build failed on Linux with
  `Cannot find module @rollup/rollup-linux-x64-gnu`.
- The citation block on the documentation home page declared version 0.2.0 with no DOI and
  announced a DOI that had already been minted (#48).

### Known issues
- Multiblock ghost exchange reads the wrong source block at `n_ghost = 2`, and a topology
  guard is missing (#28).
- `zou_he_pressure_3d_kernel!` still omits the wall-parallel diagonal populations that the
  velocity BC includes (#20).
- DDF potential stopping can accept an unconverged electric-field moment (#23).
- Enzyme reverse mode miscompiles a loop branching on a `Const` Bool mask at neighbour
  indices under `--check-bounds=yes`: segfault on x86_64 Linux, wrong gradient on aarch64
  macOS (#41, EnzymeAD/Enzyme.jl#3614). Production mode (`--check-bounds=auto`) is exact
  and is what CI and the documented workflow use.
- 13 files exceed the 700-LOC budget (#15; the tagged entry said 8).
- The LBM-CDE 3D Poiseuille driver (`run_conformation_poiseuille_libb_3d`) over-smooths the
  conformation near walls: 25.9 % near-wall `C_xy` error and a peak velocity ratio ≈ 1.13
  at `N_y = 32` (4 `@test_broken`). Use `run_viscoelastic_fvfd_poiseuille_3d`.

## [0.4.0] — 2026-09-14

### Added
- **Electrohydrodynamics / electroconvection** (`src/kernels/ehd_*`, `src/drivers/`):
  charge-transport and electric-potential LBM sub-solvers, Coulomb body force,
  free-slip sidewalls, MRT collision, GPU-ready kernels, a `.krk` user surface
  (`benchmarks/krk/ehd/`) and a critical-threshold sweep script
  (`benchmarks/ehd/tc_sweep.jl`). Hydrostatic base state matches the analytical
  charge-density and field profiles below 1%; onset threshold `T_c ~ 166.5`
  against 163.5 from Luo, Wu, Yi & Tan, Phys. Rev. E 93, 023309 (2016), a +1.8%
  deviation — mesh-convergence study for `T_c` still pending, so read it as
  consistent-with rather than converged.
- **West pressure boundary in 2D** (`apply_zou_he_pressure_west_2d!`), completing
  the east/west pair. Validated against analytic Poiseuille: max relative error
  2.03e-3 at Ny=16 and 5.15e-4 at Ny=32 (2nd order).
- **Thermal conduction driver** accepts `nu`, `alpha` and an `orientation`
  (`:vertical` default, `:horizontal` for west/east).
- Regression tests exercising the public `.krk` runner rather than the drivers
  directly: `test/analytical/H2-004-route.jl`, `test/analytical/TH-002-route.jl`.
- `AGENTS.md` and `.github/CODEOWNERS`: contributor working rules, ownership per
  directory, bug-report-as-failing-test convention.

### Fixed
- **MRT Guo forcing injected only half the requested body force.** Any MRT run
  with a body force before this release was forced at half the intended value.
- `_D2Q9_CX` was defined twice in the module with different types, which made the
  package fail to load on Julia 1.11 — the declared minimum (#17).
- The generic `.krk` runner silently ignored west pressure boundaries; unsupported
  faces now raise an explicit error instead of a no-op (#18).
- The conduction `.krk` fallback dropped `nu`, `alpha` and the thermal face
  settings, silently solving a different problem. `examples/heat_conduction.krk`
  is itself a west/east case and had been running south/north defaults at nu=0.05
  instead of its documented nu=0.1, alpha=0.01 (#19).
- 3D cavity test ran at omega = 1.8868, above the measured BGK stability ceiling
  of ~1.81, and produced NaN at step 194. Reconfigured at the same Re = 32 with
  tightened assertions.

### Changed
- **Breaking:** driver spec `OldroydB` renamed `OldroydBSpec` (name clash with the
  rheology model of the same name). Scripts naming the old type must be updated.
- `main` reconciled with the development line; the two now carry the same content.

### Known issues
- 3 failures in multi-block ghost exchange (`test_multiblock_exchange.jl`).
- `zou_he_pressure_3d_kernel!` still omits the wall-parallel diagonal populations
  that the velocity BC now includes (#20).

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
