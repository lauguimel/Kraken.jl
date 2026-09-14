# RheoTool Mesh-Divergence Audit - 2026-05-11

Scope: cell-centered log-FV cylinder path on `dev-viscoelastic`, compared with
local RheoTool Oldroyd-BLog cylinder data.

## Observed Defect

The Newtonian baseline remains close to RheoTool, but finite-Wi log-FV drifts
low as the cylinder mesh is refined.

Recent A100 run `wall_tangent_retest_21125981.aqua`:

| R | Wi | Cd | RheoTool mean | err |
|---:|---:|---:|---:|---:|
| 30 | 0.1 | 129.391531 | 130.428774 | -0.795% |
| 35 | 0.1 | 128.719242 | 130.428774 | -1.311% |
| 30 | 0.5 | 115.886379 | 119.288284 | -2.852% |
| 35 | 0.5 | 114.596099 | 119.288284 | -3.933% |

The error signature is a growing drag reduction, not a Newtonian wall/drag
normalization error.

## Eliminated Suspects

`force_boundary_fill=:nearest` is not the source. Recomputing the R30/R35 Wi=0.5
total force from the dumped `u/tau` fields with the BC-aware path and no nearest
fill changes the fluid-summed `fx_total` only from `0.002570781` to `0.002516111`
at R30 and from `0.002227400` to `0.002197417` at R35.

The global stress-divergence accounting is not the source. On the dumped fields,
`sum(div tau)` matches the outer-domain stress flux minus the cylinder traction
to about `5e-4` at R30 and `1e-4` at R35.

The current auto-subcycle estimate is too low if judged from actual near-wall
gradients (`lambda*max||grad u|| ~= 6.9`, which would imply about 100 memory
substeps with the `0.07` criterion), but this is not the Cd drift driver. A local
R8 Wi=0.5 canary with the current code gave:

| polymer_substeps | Cd |
|---:|---:|
| 8 | 115.194279 |
| 32 | 115.195262 |
| 100 | 115.195485 |

So the observed R30/R35 mesh drift is not fixed by source subcycling alone.

## Root Cause

The coupled cylinder uses two different embedded geometries:

- solvent LBM and drag use the sub-cell circular cut-link geometry `q_wall`;
- log-FV polymer operators use only the cell-center `is_solid` mask.

The FV face/advection/gradient/divergence/BSD path is therefore a staircase
polymer domain coupled to a cut-link solvent domain. This violates the branch
contract requirement that the same explicit geometry spec feed advection,
velocity gradients, stress divergence, stabilization, drag, and diagnostics.

For the RheoTool cylinder geometry, most cut information is not represented by
axis-aligned FV faces:

| R | cut cells | axis-cut cells | diagonal-only cut cells | diagonal link fraction |
|---:|---:|---:|---:|---:|
| 8 | 66 | 46 | 20 | 0.587 |
| 20 | 162 | 114 | 48 | 0.586 |
| 30 | 242 | 170 | 72 | 0.585 |
| 35 | 282 | 198 | 84 | 0.585 |

Those diagonal/cut-cell regions are exactly the high-gradient/high-stress
near-wall band. The current FV kernels cannot impose the RheoTool-like
linear-extrapolated polymer wall behavior on that cut geometry because `q_wall`
is not part of their lowered geometry input.

## Implemented First Fix

Introduce a lowered log-FV geometry spec for embedded boundaries, not another
case-specific cylinder branch:

- keep `is_solid`, but add precomputed cut-link data derived from `q_wall`;
- feed that lowered geometry to the velocity-gradient path first, because this
  is where `lambda * grad(u)` enters the log-conformation source;
- keep the lowered representation modular (`LogFVEmbeddedBoundary2D`) so face
  velocity/advection, `div(tau)`, BSD, polymer wall extrapolation, drag, and
  field dumps can be migrated without case-specific cylinder branches.

The first production correction uses diagonal, non-halfway cut-links by default.
Axis-aligned and half-way links remain opt-in so existing staircase/BFS patch
tests do not silently change semantics.

Validation before re-running RheoTool:

- local CPU patch ladder: `14082/14082` pass;
- local Metal smoke: `2138/2138` pass;
- local benchmark smoke writes `rho/u/psi/C/tau/fx/fy` JLS and VTK fields;
- Aqua CPU patch ladder after sync: `14082/14082` pass.

Next benchmark:

```bash
qsub bench/viscoelastic_logfv/run_rheotool_embedded_grad_a100.pbs
```

## Embedded-Gradient Diagnostic Freeze

The first embedded-gradient macro sweep improved the R20 finite-Wi cases but
made R30/Wi=0.1 non-finite. This is now frozen as a partial-geometry bug, not a
constitutive relaxation bug.

Key runs:

| run | change | result |
|---|---|---|
| `21146973.aqua` | embedded gradient on, diagnostic stride 10 | first non-finite at step 4030, `rho(439,22)` |
| `21146813.aqua` | embedded gradient on, `polymer_substeps=64` | same failure at step 5000 with stride 1000 |
| `21147090.aqua` | embedded gradient off, 5000 steps | finite through step 5000 |
| local Metal `tmp/local_metal_r30_wi01_4010_embedded` | embedded gradient on, step 4010 | still finite but `C_trace≈43.2`, `|f_poly|≈1.3e-3` at a near-wall cut cell |

The last finite Metal snapshot localizes the source before blow-up to a
near-tangent cut cell around `(439,33)` with distance to the cylinder surface
about `0.004` lattice units. The embedded velocity-gradient source creates a
large conformation/stress spike there. Because advection, stress divergence,
BSD force correction, volume weighting, and traction diagnostics are still on
the staircase FV domain, the spike is injected into a geometry-inconsistent
momentum coupling and leaks into the wake.

Conclusion: enabling only embedded velocity gradients is an invalid production
fix. It remains an opt-in diagnostic (`KRAKEN_LOGFV_EMBEDDED_GRADIENT=1`) until
the same lowered cut-cell geometry also feeds FV advection, polymer wall
extrapolation, stress divergence, BSD/DEVSS correction, volume weighting, drag,
and field diagnostics.

## FVFD Modular Operator Slice

Status on 2026-05-11:

- `q_wall` lowering now exports reusable FVFD embedded-boundary fields:
  `wall_nx`, `wall_ny`, `wall_distance`, `wall_inv_distance`,
  `cell_fraction`, and `cut_count`;
- `FVFDFieldBC2D` now carries explicit west/east/south/north field values for
  velocity and polymer/log-conformation fields;
- cell-to-face velocity lowering now delegates through
  `fvfd_cell_velocity_to_faces_2d!`;
- scalar and symmetric-2 tensor upwind advection now delegate through
  `fvfd_advect_upwind_2d!` and `fvfd_sym2_advect_upwind_2d!`;
- log-FV advection has an explicit `dx, dy` wrapper and the production drivers
  now call it with patch spacing instead of relying on implicit unit spacing;
- open-boundary field values are validated on the host before launching FVFD
  face-velocity and scalar-advection kernels;
- the previous generic `logfv_advect_upwind_2d!` test kernel has been removed
  from the durable API surface and renamed as an explicit interior canary;
- the log-FV polymer-force path now delegates `div(tau)` to
  `fvfd_tensor_divergence_2d!`;
- the log-FV BSD force correction now delegates its Laplacian to
  `fvfd_bsd_force_2d!`;
- compatibility wrappers keep the previous `logfv_*` entry points, but the
  implementation is now driven by the same FVFD BC and geometry contracts.

Validation:

| location | command/case | result |
|---|---|---|
| local CPU | `julia --project=. test/test_fvfd_operators_2d.jl` | `886/886` pass |
| local CPU | `julia --project=. test/test_viscoelastic_logfv_patch_ladder.jl` | `14087/14087` pass |
| local Metal | `julia --project=. test/test_viscoelastic_logfv_gpu_smoke.jl` | `2138/2138` pass |
| local CPU | `julia --project=. test/test_logconformation_3d.jl` | `48/48` pass |
| local Metal | simple validation outputs | 4/4 cases pass, dashboard generated |
| Aqua CPU | `test/test_fvfd_operators_2d.jl` | `886/886` pass |
| Aqua CPU | `test/test_viscoelastic_logfv_patch_ladder.jl` | `14087/14087` pass |
| Aqua CPU fallback | simple validation outputs | 4/4 cases pass |
| Aqua A100 CUDA | `21150217.aqua` simple validation outputs | 4/4 cases pass, `Exit_status=0` |
| Aqua A100 CUDA | `21159703.aqua` `test/test_viscoelastic_logfv_gpu_smoke.jl` | `2138/2138` pass, `Exit_status=0` |

Aqua interactive CUDA was not functional in this session, so the first simple
validation output run fell back to CPU. The batch A100 run `21150217.aqua`
then executed the same gate with backend `:CUDA` and wrote the same
`rho/u/psi/C/tau/fx/fy` VTK field set. The later batch smoke run
`21159703.aqua` validated the migrated FVFD-backed log-FV wrappers on CUDA
after the face velocity and upwind advection migration, explicit interior
canary rename, addition of east/north and periodic advection canaries, explicit
non-unit-spacing log-FV advection wrapper, and open-field-BC validation
canaries.

The FVFD count includes negative-velocity east/north open inflow and periodic
scalar/symmetric-tensor upwind wrapping canaries.

Dashboards:

- local Metal:
  `tmp/logfv_simple_validation_outputs/20260511_fvfd_modular_local/dashboard.html`;
- Aqua CPU fallback, synced locally:
  `tmp/logfv_simple_validation_outputs/20260511_fvfd_modular_aqua/dashboard.html`;
- Aqua A100 CUDA, synced locally:
  `tmp/logfv_simple_validation_outputs/20260511_fvfd_modular_a100/dashboard.html`.
