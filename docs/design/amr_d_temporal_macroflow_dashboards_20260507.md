# AMR-D Temporal Macro-Flow Dashboards, 2026-05-07

## Purpose

`benchmarks/amr_d_macroflow_temporal_convergence_2d.jl` runs the AMR-D
macro-flow KRK cases with temporal convergence checkpoints and writes one
dashboard plus one mesh-level image per case.

The output contract is intentionally narrow:

- `debug_dashboard.png`: fields, profiles, references when available, and
  temporal convergence trace.
- `mesh_levels.png`: three mesh wireframe panels using `viridis`, `magma`, and
  `plasma`; wireframe color is tied to AMR level.
- `convergence.csv`: checkpoint deltas and final status.
- `values.csv`: AMR/reference scalar metrics, including raw mass correction.

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
- `amr_d_couette_yband_nested4_debug`: max step cap, 2560 steps.
- `amr_d_cylinder_nested4_probe`: max step cap, 2560 steps.

Important caveat: the nested cylinder is a diagnostic probe, not a closed
validation gate. It needs a relaxed KRK `mass_guard_rtol` and reports
`max_raw_mass_rel_drift` in `values.csv`; this identifies a solid-interface
mass-closure issue to audit before declaring nested obstacles validated.
