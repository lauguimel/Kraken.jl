# AMR-D Temporal Macro-Flow Dashboards, 2026-05-07

## Purpose

`benchmarks/amr_d_macroflow_temporal_convergence_2d.jl` runs the AMR-D
macro-flow KRK cases with temporal convergence checkpoints and writes one
dashboard per case. The mesh is overlaid directly on the field panels so the
field, the refinement pattern, and the profile probe location are inspected in
one reproducible image.

The output contract is intentionally narrow:

- `debug_dashboard.png`: Cartesian transient reference fields, AMR-D fields,
  mesh wireframes overlaid on every field panel, profile probes, steady
  analytic references when available, and temporal convergence traces.
- `convergence.csv`: checkpoint deltas and final status.
- `values.csv`: AMR/reference scalar metrics, including raw mass correction.

There is no separate `mesh_levels.png` anymore. The dashboard is the visual
artifact to archive for documentation and debugging.

## Reproduction

```bash
julia --project=. -e 'ENV["KRK_AMR_D_TEMP_OUTDIR"]="benchmarks/results/quicklook/amr_d_temporal_convergence_20260507"; ENV["KRK_AMR_D_TEMP_MAX_STEPS"]="2560"; include("benchmarks/amr_d_macroflow_temporal_convergence_2d.jl"); main()'
```

To reuse completed case folders and only run missing cases:

```bash
julia --project=. -e 'ENV["KRK_AMR_D_TEMP_OUTDIR"]="benchmarks/results/quicklook/amr_d_temporal_convergence_20260507"; ENV["KRK_AMR_D_TEMP_MAX_STEPS"]="2560"; ENV["KRK_AMR_D_TEMP_SKIP_EXISTING"]="1"; include("benchmarks/amr_d_macroflow_temporal_convergence_2d.jl"); main()'
```

Local Metal and Aqua/H100 CUDA debug runs use the same runner:

```bash
KRK_AMR_D_TEMP_BACKEND=metal \
KRK_AMR_D_TEMP_T=float32 \
KRK_AMR_D_TEMP_CASES=poiseuille_xband_nested4_debug.krk,poiseuille_yband_nested4_debug.krk,poiseuille_wall_ybands_nested4_debug.krk,couette_yband_nested4_debug.krk \
KRK_AMR_D_TEMP_MAX_STEPS=12800 \
KRK_AMR_D_TEMP_SINGLE_STEP=1 \
KRK_AMR_D_TEMP_OUTDIR=benchmarks/results/quicklook/amr_d_metal_nested_channels_long_20260508 \
julia --project=. benchmarks/amr_d_macroflow_temporal_convergence_2d.jl
```

Use `KRK_AMR_D_TEMP_BACKEND=cuda` on Aqua/H100, or
`KRK_AMR_D_TEMP_BACKEND=auto` to select CUDA first, then Metal, then CPU.
Set `KRK_AMR_D_TEMP_REFERENCE=none` for an AMR-only GPU dashboard; the default
`auto` mode also runs the classic Cartesian channel reference when available.

The GPU path is currently wired for route-native nested channel AMR-D:
Poiseuille with periodic-x wall-y boundaries and Couette with periodic-x
moving-wall-y boundaries. It uses `Float32` by default on Metal/CUDA, applies
the same subcycle scheduler as the CPU reference, and performs a per-step global mass
correction for local validation. The mass reduction and correction stay on the
device during the time loop; only the final scalar diagnostic is copied back.
The runner has both an atomic scalar reduction and a chunked reduction. The
runner uses the chunked path for macro-flow validation because the simpler
atomic Float32 reduction is faster on some local Metal debug cases but can
leave a visible mass residual.
Nested solid probes still run through the CPU AMR-D route path until the
solid-interface GPU ledgers are validated.
The classic Cartesian channel reference uses the same backend as the AMR-D
run when `KRK_AMR_D_TEMP_BACKEND=metal|cuda|auto` resolves to a GPU backend;
otherwise it uses the dense CPU integrated-population solver. Each case writes
`runtime.csv` so the AMR-D backend, reference backend, precision, step count,
and elapsed time are explicit instead of inferred from activity monitors.
`KRK_AMR_D_TEMP_SINGLE_STEP=1` is recommended for local Metal runs because the
default temporal sweep reruns each case from zero at every checkpoint
(`max_steps`, `2*max_steps`, ...), which is useful for convergence traces but
too expensive for interactive GPU smoke tests.

## 2026-05-07 Local Results

Output root:
`benchmarks/results/quicklook/amr_d_temporal_convergence_20260507`

Completed cases:

- `amr_d_poiseuille_xband_scale1`: max step cap, 2560 steps.
- `amr_d_poiseuille_yband_scale1`: max step cap, 2560 steps.
- `amr_d_couette_scale1`: max step cap, 2560 steps.
- `amr_d_bfs_scale1`: max step cap, 2560 steps.
- `amr_d_square_scale1`: converged at 1280 steps.
- `amr_d_cylinder_scale1`: converged at 2560 steps.
- `amr_d_poiseuille_xband_nested4_debug`: max step cap, 2560 steps.
- `amr_d_poiseuille_yband_nested4_debug`: max step cap, 2560 steps.
- `amr_d_poiseuille_wall_ybands_nested4_debug`: wall-refined y-band control
  case added after the first dashboard sweep.
- `amr_d_couette_yband_nested4_debug`: max step cap, 2560 steps.
- `amr_d_cylinder_nested4_probe`: max step cap, 2560 steps.

Dashboard interpretation:

- The first row is the classic Cartesian transient reference when a reference
  is available: mesh, `ux`, and `rho`. It is run to the same physical time as
  the AMR-D checkpoint, not necessarily to the steady analytic limit.
- The second row is AMR-D: mesh, `ux`, and `rho`.
- The dashed vertical line is the local profile probe. It is placed at the
  center of the maximum-refinement region, not at a fixed fraction of the
  domain.
- The bottom profiles distinguish `row-mean ux(y)` from the local vertical
  probe. The row mean averages all fluid cells in `x`; on a local refinement
  band it intentionally includes coarse and fine regions and can show level
  transition stair steps. The vertical probe is the local validation slice.
- When convergence rows are available, the last row shows temporal `ux`, `rho`,
  and corrected mass drift.

Current diagnosis:

- A 2026-05-07 follow-up fixed a real initialization bug in the buffered
  scheduler: inactive parent restriction buffers are now built immediately
  after the active state is stored, before the first `sync_down`. Without this,
  predictor and spatial reconstruction paths saw zero-valued inactive parents
  at the first coarse/fine exchange. Surgical guards now cover four-level
  periodic-x wall-y rest states with both `coarse_to_fine_predictor_weight=0.5`
  and `coarse_to_fine_prolongation=:limited_linear`.
- The nested x-band and y-band dashboards use the same classic Cartesian
  transient reference for Poiseuille. If those rows look different, inspect the
  AMR-D row/probe/axis scaling first; the reference CSV values are identical
  for the two nested channel cases at the same final step.
- `amr_d_poiseuille_xband_nested4_debug` exposed a previous plotting/probe
  problem. The old probe was outside the refined band. With the probe moved
  into the maximum-level region, the local profile is close to the classic
  Cartesian reference at 2560 AMR-D steps. The row mean can still show steps
  because it averages across both coarse and refined regions.
- `amr_d_poiseuille_yband_nested4_debug` is not a plotting-only problem and is
  not fixed by the current 2560-step run. After the initial-restriction fix,
  the 640-step local dashboard is much healthier (`ux_max=8.63e-4` vs
  Cartesian `9.60e-4`), but the 2560-step run still plateaus low
  (`ux_max=1.269e-3` vs Cartesian `2.306e-3`) with zero corrected mass drift.
  Treat this as a negative nested/interface diagnostic, not as a validation
  gate.
- A separate surgical canary verifies that a full-domain nested Poiseuille
  tree reproduces the uniform Cartesian transient profile at the same physical
  time. This isolates the y-band failure to wall-normal refinement/interface
  placement: the centered y-band leaves the physical walls coarse and places
  refinement transitions across the dominant shear direction.
- `poiseuille_wall_ybands_nested4_debug.krk` is the reproducible control for
  this diagnosis. It refines both physical wall bands with the same ratio-16
  nesting and is now part of the temporal dashboard case list. It matches the
  Cartesian transient closely at 200 steps (`linf_profile_vs_reference =
  4.2e-6`) but overshoots by 2560 steps (`ux_max = 3.86e-3` vs Cartesian
  `2.31e-3`). This confirms that wall-normal coarse/fine interface closure is
  still not validation-grade.
- A local Filippova-Hänel scalar rescaling A/B check at 640 steps was rerun
  after the restriction fix. `alpha_c2f=2`, `alpha_f2c=0.5` still does not
  recover the missing y-band velocity and is slightly worse than `1,1`; this
  points to the wall-normal interface/closure placement rather than a simple
  missing constant alpha in the current packet reconstruction.
- A conservative coarse-to-fine temporal predictor is now wired into the
  subcycled macro-flow runners. It uses a 50% blend between the committed
  parent state and a local post-collision parent predictor for coarse-to-fine
  packets, while keeping the flat packet geometry. Surgical tests show that it
  reduces the short-time wall-normal y-band bias while preserving roundoff mass
  conservation and keeping the x-band `linf` regression below 5% on the canary.
- Limited-linear spatial prolongation is available only as an explicit
  experimental kernel option. It preserves four-level rest mass after the
  restriction fix and slightly improves short-time y-band L2 in local A/B runs,
  but it is not the production default. The production macro-flow runners
  intentionally stay on flat coarse-to-fine packets until the wall-normal
  interface closure has a stronger validation story and a cheaper packet cache.
- The limited-linear A/B path is now reproducible from KRK through
  `Define c2f_prolongation = 1`; `poiseuille_yband_nested4_limited_debug.krk`
  is the reference input. At 640 steps it improves the y-band quicklook
  (`ux_max=9.01e-4`, `linf_profile_vs_reference=2.51e-4`) relative to the flat
  y-band run (`ux_max=8.63e-4`, `linf_profile_vs_reference=3.04e-4`), while
  keeping corrected mass drift at zero. Its raw mass correction is larger and
  the current implementation resamples routes, so it remains an experimental
  diagnostic rather than the default production path.
- A route-sampling audit isolated another candidate cause: subcycled coarse
  same-level packets should eventually move one cell of their own level, while
  the current production table keeps the historical leaf-equivalent sampling.
  `create_conservative_tree_route_table_2d` now has explicit experimental
  `sampling=:level_native` and `sampling=:subcycled_hybrid` modes, and the
  subcycling ledgers have an explicit `interface_time_scaling` switch. The
  surgical canary proves that native direct routes and native time weighting
  preserve global rest mass. A follow-up patch tightened native coarse-to-fine
  face injection: direct coarse routes move by one coarse cell away from
  interfaces, while coarse-to-fine packets use a boundary-layer child stencil
  without a direct residual. The axis-only rest-state canary is now green.
  A second patch made native diagonal fine-to-coarse corner reflux use a
  final-time destination, reducing the full D2Q9 residual to the fine corner
  cells. Full D2Q9 local rest invariance is still red because diagonal coarse
  ghost packets at face/corner contacts must be paired conservatively with that
  reflux. Therefore production macro-flow runners intentionally remain on the stable
  `sampling=:leaf_equivalent` path until that diagonal closure is fixed.
- `poiseuille_analytic_profile_2d` now uses the same halfway bounce-back wall
  convention as the Cartesian Poiseuille tests: walls are located half a cell
  outside the first and last fluid cell centers. The black curve in dashboards
  should be read as the steady analytic target, not as the transient
  Cartesian state.

Important caveat: the nested cylinder is a diagnostic probe, not a closed
validation gate. It needs a relaxed KRK `mass_guard_rtol` and reports
`max_raw_mass_rel_drift` in `values.csv`; this identifies a solid-interface
mass-closure issue to audit before declaring nested obstacles validated.
