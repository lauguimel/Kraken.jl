# Step-geometry isolation plan — 2026-05-01

## Decision

Run axis-aligned step benchmarks before any obstacle benchmark.

The useful order is:

0. **Bulk constitutive checks**: exact simple shear, planar elongation, and
   imposed Poiseuille before any wall/geometry coupling.
1. **4:1 sudden contraction**: first step-geometry isolation case.
2. **Backward-facing step**: second isolation case, after partial inlet/outlet
   boundary conditions exist.
3. **Cylinder obstacle**: only after the step cases are coherent.

Implementation rule: these cases must share one modular step-channel pipeline,
not one solver driver per geometry. The current implementation follows the
SLBM pattern: build a geometry/spec object once, transfer it to the backend,
then pass it to the common solver.

The bulk pre-gate is:

```bash
KRAKEN_BACKEND=metal KRAKEN_BULK_CASES=shear,elongation,poiseuille \
  KRAKEN_MODELS=direct,logconf KRAKEN_WI_LIST=0.05,0.1,0.2 \
  julia --project=. hpc/bulk_constitutive.jl
```

On AQUA:

```bash
qsub hpc/run_bulk_constitutive.pbs
```

## Why contraction first

The current 4:1 contraction driver already exists and avoids the two largest
sources of ambiguity in the cylinder audit:

- no curved obstacle wall;
- no `Cd` or polymer-surface traction quadrature.

The remaining stress is concentrated around two symmetric, axis-aligned
re-entrant corners. That makes it a good test for:

- direct conformation versus log-conformation;
- CNEBB versus no polymer wall reconstruction;
- Liu-direct versus CE-corrected Hermite source;
- conformation positivity and top/bottom symmetry.

The audit script now runs one or more `StepChannelGeometry2D` specs through
the same solver:

```bash
KRAKEN_BACKEND=metal KRAKEN_GEOMETRIES=contraction,bfs \
  julia --project=. hpc/contraction_step_isolation.jl
```

On AQUA, use:

```bash
qsub hpc/run_step_geometry_isolation.pbs
```

The PBS defaults to the priority matrix: contraction only, `H_out=8`,
`direct/logconf × none/CNEBB × Liu/CE`, and `Wi=0.05,0.1,0.2`. Repeat with
`KRAKEN_HOUT=16 qsub hpc/run_step_geometry_isolation.pbs` for the first
refinement leg.

It writes a CSV under `tmp/step_geometry_isolation_*` and reports:

- `xrec_s`, `xrec_n`, `xrec_asym`;
- `ux_sym`, `uy_asym`, `Cxx_sym`, `Cxy_asym`;
- `min_eig_C`, `max_trace_C`;
- `max_abs_Cxy`, `max_abs_N1`, `max_abs_tau_xy`;
- inlet/outlet mean velocity and density extrema.

These are pre-`Cd` diagnostics. They should be coherent before using any
force-integral or cylinder drag number as validation evidence.

## Why BFS is not first

A backward-facing step with a horizontal top wall and a lower inlet block would
be cleaner than the symmetric contraction because it has one expansion corner
instead of two symmetric contraction corners.

The first contraction audit showed that this was not only a BFS issue: the
4:1 contraction also has a partial east outlet. Whole-face `ZouHePressure`
was therefore being applied in solid blocks. The code now has
`MaskedZouHeVelocity` and `MaskedZouHePressure`, and the contraction driver
uses a masked east outlet. This is the infrastructure needed for BFS.

Current modular API:

- `contraction_step_geometry_2d(...)` builds the 4:1 contraction spec.
- `backward_facing_step_geometry_2d(...)` builds the BFS spec.
- `run_conformation_step_libb_2d(; geometry, ...)` runs either spec.
- `run_conformation_contraction_libb_2d(...)` is now only a compatibility
  wrapper that builds a contraction spec and calls the generic driver.
- `hpc/contraction_step_isolation.jl` is a geometry matrix, not a contraction
  driver; use `KRAKEN_GEOMETRIES=contraction,bfs` to run both cases.

## Acceptance criteria before cylinder

For the contraction matrix:

- direct and log-conformation should agree at low `Wi`;
- CNEBB should not break top/bottom symmetry;
- `min_eig_C` should remain positive;
- `NoPolymerWallBC` should clearly separate wall-reconstruction effects from
  bulk source effects;
- Liu-direct versus CE-corrected source should change stress amplitudes
  predictably, not invert the `Wi` trend.

Only after these pass should the cylinder audit return to force decomposition
and comparison with Liu/rheoTool.

## First bug found

The first Metal isolation run found a pre-`Cd` log-conformation bug: the
contraction driver imposed analytical inlet `C` components directly into the
evolved `Ψ=log(C)` populations. That produced `max_trace_C ≈ 5.44` in log-conf
versus `≈ 2.02` in direct conformation at the same short `Re=1`, `Wi=0.1`
case.

The contraction driver now converts the inlet profile to `log(C)` in
log-conformation mode, matching the cylinder driver. The post-fix Metal check
gives `max_trace_C ≈ 2.016` for direct conformation and `≈ 2.010` for
log-conformation on the same short case.

The second issue was post-processing only: `X_R` used an exact `sign` at one
corner-adjacent velocity sample. At low `Wi` this produced false asymmetric
`0` values in otherwise symmetric fields. `vortex_length_contraction_2d` now
uses a tolerance and skips zero/noisy corner samples.

## Current masked-contraction check

Short Metal check, `H_out=8`, `Re=1`, `β=0.59`, `τp,1=1`, CNEBB,
Liu-direct Hermite source:

| `Wi` | formulation | `min_eig_C` | `max_trace_C` | `max_abs_Cxy` | `max_abs_N1` |
|---:|---|---:|---:|---:|---:|
| 0.05 | direct | 0.915863 | 2.009511 | 0.053483 | 0.065696 |
| 0.05 | log-conf | 0.918868 | 2.008649 | 0.053415 | 0.067338 |
| 0.10 | direct | 0.884967 | 2.016205 | 0.070345 | 0.084959 |
| 0.10 | log-conf | 0.886421 | 2.009665 | 0.070040 | 0.088115 |
| 0.20 | direct | 0.857283 | 2.036857 | 0.083486 | 0.101960 |
| 0.20 | log-conf | 0.852217 | 2.010487 | 0.082509 | 0.107474 |

Interpretation: after the inlet `log(C)` fix and masked outlet fix, direct and
log-conformation are close on the axis-aligned contraction before any obstacle
or `Cd` computation. The remaining cylinder discrepancy is therefore more
likely in curved-wall/force accounting than in the bulk conformation equation
for this low-`Wi`, `τp,1=1` regime.
