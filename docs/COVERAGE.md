# Documentation Coverage Matrix

Last updated: 2026-04-30

This matrix covers the public `release/v0.1.0` documentation scope. Planned
integrations from `slbm-paper` or other development branches are tracked in
`docs/src/integration_roadmap.md`.

## Public scope and navigation

| Page | Status |
|---|---|
| `docs/src/index.md` | current |
| `docs/src/getting_started.md` | current `.krk` first path |
| `docs/src/concepts_index.md` | current |
| `docs/src/capabilities.md` | current branch source of truth |
| `docs/src/integration_roadmap.md` | current planned integration ledger |
| `docs/src/llms.md` | current agent-facing summary |
| `docs/src/public/llms.txt` | current compact agent context |

## Public theory pages

| Area | Page | Status |
|---|---|---|
| LBM fundamentals | `docs/src/theory/01_lbm_fundamentals.md` | current |
| D2Q9 lattice | `docs/src/theory/02_d2q9_lattice.md` | current |
| BGK collision | `docs/src/theory/03_bgk_collision.md` | current |
| Streaming | `docs/src/theory/04_streaming.md` | current |
| Boundary conditions | `docs/src/theory/05_boundary_conditions.md` | current |
| From 2D to 3D | `docs/src/theory/06_from_2d_to_3d.md` | current |
| Body forces | `docs/src/theory/07_body_forces.md` | current |
| Thermal DDF | `docs/src/theory/08_thermal_ddf.md` | current |
| Limitations | `docs/src/theory/10_limitations.md` | current |
| Spatial BCs | `docs/src/theory/19_spatial_bcs.md` | current |

## Public examples

| Example | Backing `.krk` | Status |
|---|---|---|
| Poiseuille 2D | `examples/poiseuille.krk` | current |
| Couette 2D | `examples/couette.krk` | current |
| Taylor-Green 2D | `examples/taylor_green.krk` | current |
| Lid-driven cavity 2D | `examples/cavity.krk` | current |
| Lid-driven cavity 3D | `examples/cavity_3d.krk` | current |
| Cylinder 2D | `examples/cylinder.krk` | current |
| Heat conduction | `examples/heat_conduction.krk` | current |
| Rayleigh-Benard | `examples/rayleigh_benard.krk` | current |
| `.krk` config walk-through | docs page only | current |

## API pages

| Page | Status |
|---|---|
| `docs/src/api/public_api.md` | current export inventory |
| `docs/src/api/lattice.md` | current |
| `docs/src/api/collision.md` | corrected to v0.1.0 scope |
| `docs/src/api/streaming.md` | corrected to v0.1.0 scope |
| `docs/src/api/boundary.md` | needs more examples |
| `docs/src/api/macroscopic.md` | current |
| `docs/src/api/drivers.md` | corrected to v0.1.0 scope |
| `docs/src/api/io.md` | needs review for unsupported refined/STL wording |
| `docs/src/api/postprocess.md` | current |
| `docs/src/api/config.md` | current |

## Benchmarks

| Check | Status |
|---|---|
| Poiseuille convergence | rerun locally 2026-04-30, CSV-backed |
| Taylor-Green convergence | rerun locally 2026-04-30 |
| Thermal conduction | rerun locally 2026-04-30; first-order with current wall treatment |
| Natural convection `Ra=1e3` | rerun locally 2026-04-30 |
| H100 throughput | not published until matching CSV is committed |
| External comparisons | draft only |

See `benchmarks/results/VALIDATION_2026-04-30.md`.

## Explicitly out of public scope

The following topics must not be marked "Done" for this branch:

- axisymmetric LBM;
- MRT collision;
- grid refinement;
- VOF/PLIC;
- phase-field;
- Shan-Chen;
- species transport;
- rheology;
- viscoelasticity;
- SLBM/body-fitted curvilinear methods.

If they are documented later, update the code path, tests, benchmark
provenance, `docs/src/integration_roadmap.md`, and public documentation links
in the same PR.
