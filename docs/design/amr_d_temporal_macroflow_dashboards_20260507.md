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
  not fixed by the current 2560-step run. AMR-D reaches `ux_max=1.197e-3`
  while the Cartesian reference reaches `ux_max=2.306e-3`; the last checkpoint
  has `ux_linf_delta=3.33e-5`, so the run is already near a plateau. Treat this
  as a negative nested/interface diagnostic, not as a validation gate.
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
- A local Filippova-Hänel scalar rescaling A/B check at 640 steps
  (`alpha_c2f=2`, `alpha_f2c=0.5`) did not recover the missing y-band velocity;
  this points to the wall-normal interface/closure placement rather than a
  simple missing constant alpha in the current packet reconstruction.
- A conservative coarse-to-fine temporal predictor is now wired into the
  subcycled macro-flow runners. It uses a 50% blend between the committed
  parent state and a local post-collision parent predictor for coarse-to-fine
  packets, while keeping the flat packet geometry. Surgical tests show that it
  reduces the short-time wall-normal y-band bias while preserving roundoff mass
  conservation and keeping the x-band `linf` regression below 5% on the canary.
- Limited-linear spatial prolongation remains a separate future patch. The
  local A/B audit showed that it can improve some y-band profiles, but it must
  close split, direct residual, boundary, and recursive parent states as one
  link-level conservative packet group before it is safe to expose.
- `poiseuille_analytic_profile_2d` now uses the same halfway bounce-back wall
  convention as the Cartesian Poiseuille tests: walls are located half a cell
  outside the first and last fluid cell centers. The black curve in dashboards
  should be read as the steady analytic target, not as the transient
  Cartesian state.

Important caveat: the nested cylinder is a diagnostic probe, not a closed
validation gate. It needs a relaxed KRK `mass_guard_rtol` and reports
`max_raw_mass_rel_drift` in `values.csv`; this identifies a solid-interface
mass-closure issue to audit before declaring nested obstacles validated.
