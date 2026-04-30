# Integration roadmap

This page tracks what is already available in the public documentation branch
and what remains to integrate. It is deliberately conservative: an item is
checked only when the code, `.krk` access path, validation, and documentation
all exist in the release branch.

Last local audit: 2026-04-30.

## How an item becomes checked

`Available since` is filled only after all release gates are met:

1. The implementation is merged into the release branch.
2. The public access path is clear: `.krk` first when possible, or an explicit
   Julia API when a feature is not a `.krk` feature.
3. Tests pass in the release branch.
4. Validation data is reproducible and stored in `benchmarks/results/` when the
   feature makes numerical or performance claims.
5. Documentation exists for the concept, the `.krk` syntax or Julia API, and at
   least one user-facing example.
6. `docs/make.jl` builds with strict links.
7. [Capabilities](capabilities.md) and the compact agent context are updated.

If convergence, stability, API design, or documentation is not ready, the item
stays `To do`.

## Checked and available

| Feature | Status | Available since | Description/tutorial | Evidence |
|---|---|---:|---|---|
| `.krk` workflow | checked | v0.1.0 | [Getting started](getting_started.md), [.krk overview](krk/overview.md), [.krk config reference](examples/10_krk_config.md) | `run_simulation("case.krk")`, `load_kraken("case.krk")`, public examples. |
| D2Q9/D3Q19 BGK | checked | v0.1.0 | [LBM fundamentals](theory/01_lbm_fundamentals.md), [D2Q9 lattice](theory/02_d2q9_lattice.md), [From 2D to 3D](theory/06_from_2d_to_3d.md) | Unit tests and documented cavity/Poiseuille/Taylor-Green examples. |
| Guo forcing | checked | v0.1.0 | [Body forces](theory/07_body_forces.md), [Body-forces tutorial](tutorials/02_body_forces.md) | Poiseuille convergence rerun on 2026-04-30. |
| Thermal DDF | checked | v0.1.0 | [Thermal DDF](theory/08_thermal_ddf.md), [Thermal tutorial](tutorials/04_thermal.md), [Heat conduction](examples/07_heat_conduction.md) | Heat conduction and natural-convection checks rerun on 2026-04-30. |
| Boussinesq natural convection 2D | checked | v0.1.0 | [Rayleigh-Benard](examples/08_rayleigh_benard.md), [Accuracy](benchmarks/accuracy.md) | `Ra = 1e3`, `N = 64`: `Nu = 1.1423`, error `2.17%`. |
| VTK/PNG/GIF outputs | checked | v0.1.0 | [.krk directives](krk/directives.md), [IO API](api/io.md) | Public examples and generated showcase assets. |
| Agent context | checked | v0.1.0 | [LLM and agent context](llms.md), `/llms.txt` | Compact scope and result policy included in the built docs. |

## To do

| Feature | Status | Description/tutorial | Integration note |
|---|---|---|---|
| H100 throughput report | To do | To do | Commit matching CSV files, hardware metadata, command lines, and docs. |
| Laptop benchmark label cleanup | To do | To do | Keep M2/M3 labels as CSV provenance only; publish CPU baseline plus H100 when backed by artifacts. |
| Public API docstring coverage | To do | [Public API inventory](api/public_api.md) | Add docstrings for exported functions and keep the inventory generated/traceable. |
| Backports from `slbm-paper` bug fixes | To do | To do | Backport only fixes that affect current public behavior; rerun tests and convergence. |
| MRT collision | To do | To do | Needs stable API, `.krk` selection, and regression tests against BGK limits. |
| Axisymmetric LBM | To do | To do | Needs axisymmetric benchmark, boundary-condition docs, and `.krk` module path. |
| Generalized Newtonian rheology | To do | To do | Needs Power-law/Carreau/Cross/Bingham/Herschel-Bulkley tests and `.krk` syntax. |
| Viscoelasticity | To do | To do | Needs Oldroyd-B/FENE-P/Saramito benchmarks, stress-output docs, and stability limits. |
| Two-phase VOF/PLIC | To do | To do | Needs mass conservation, Zalesak/reversed-vortex/capillary/static-droplet validation. |
| Phase-field two-phase | To do | To do | Needs interface-thickness study, spurious-current checks, and `.krk` module docs. |
| Shan-Chen multiphase | To do | To do | Needs static-droplet validation, stability envelope, and parameter guide. |
| Species transport | To do | To do | Needs advection-diffusion validation and coupling rules with thermal/flow fields. |
| Grid refinement | To do | To do | Needs conservation tests across interfaces, time-stepping docs, and benchmark deltas. |
| Curvilinear SLBM 2D | To do | To do | Needs mesh-quality docs, manufactured/convergence cases, and `.krk` examples. |
| Body-fitted cylinder with LI/DA bounce-back | To do | To do | Needs drag/lift validation, force-scaling audit, and Schaefer-Turek comparison. |
| STL reader and voxelizer | To do | To do | Needs geometry fixtures, parser-to-runner path, and failure-mode docs. |
| Gmsh import | To do | To do | Needs minimal public mesh format, examples, and CI fixture meshes. |
| Multiblock topology/exchange | To do | To do | Needs interface conservation tests, block-orientation docs, and `.krk` or Julia API. |
| 3D SLBM/body-fitted flow | To do | To do | Needs sphere/channel validation, GPU memory checks, and documented limitations. |
| AA/fused/persistent kernels | To do | To do | Needs same numerical results as reference kernels and hardware-specific benchmark CSVs. |
| Dual-grid kernels | To do | To do | Needs explicit numerical contract, tests, and docs explaining when to use them. |
| Enzyme/AD support | To do | To do | Needs supported AD workflow, gradient validation, and dependency policy. |
| External benchmark suite | To do | To do | Needs reproducible scripts, fixed hardware labels, and committed outputs. |

## Documentation update rule

When a row becomes checked, update these files in the same pull request:

- `docs/src/capabilities.md`
- `docs/src/integration_roadmap.md`
- `docs/src/llms.md`
- `docs/src/public/llms.txt`
- the relevant `.krk`, theory, API, example, and benchmark pages

This keeps the human documentation and the agent context aligned with the code
that users can actually run.
