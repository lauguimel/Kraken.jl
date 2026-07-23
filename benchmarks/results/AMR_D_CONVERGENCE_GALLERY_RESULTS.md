# AMR D Convergence Gallery Result Manifest

Date: 2026-05-05

## Source Cases

- `benchmarks/krk/amr_d_convergence_2d/*.krk`

Scale-1 `.krk` files are the source templates. The runner scales dimensions,
patch ranges and obstacle geometry with `KRK_AMR_D_GALLERY_SCALES`.

## Runner

- `benchmarks/amr_d_convergence_gallery_2d.jl`
- `hpc/amr_d_convergence_gallery_2d_aqua.pbs`

The primary plotted metric is error against the dense leaf oracle:

- Couette and Poiseuille: `linf_profile_error_vs_leaf`;
- BFS and square: `ux_abs_error_vs_leaf`;
- cylinder: `Cd_rel_error`.

The profile rows also retain the analytic profile L2 error as
`secondary_error`.

## Local Smoke

Command:

```bash
KRK_AMR_D_GALLERY_SCALES=1 \
KRK_AMR_D_GALLERY_BASE_STEPS=4 \
KRK_AMR_D_GALLERY_STEP_EXPONENT=2 \
KRK_AMR_D_GALLERY_AVG_WINDOW=2 \
KRK_AMR_TAG=gallery_smoke_parse \
julia --project=. benchmarks/amr_d_convergence_gallery_2d.jl
```

Artifacts:

- `amr_d_convergence_gallery_2d_gallery_smoke_parse.csv`
- `amr_d_cylinder_nested4_probe_2d_gallery_smoke_parse.csv`
- `figures/amr_d_convergence_errors_2d_gallery_smoke_parse.png`
- `figures/amr_d_convergence_errors_2d_gallery_smoke_parse.pdf`
- `figures/amr_d_convergence_cost_2d_gallery_smoke_parse.png`
- `figures/amr_d_convergence_cost_2d_gallery_smoke_parse.pdf`
- `figures/amr_d_cylinder_nested4_probe_2d_gallery_smoke_parse.png`
- `figures/amr_d_cylinder_nested4_probe_2d_gallery_smoke_parse.pdf`

## Local Ramp

Command:

```bash
KRK_AMR_D_GALLERY_SCALES=1,2 \
KRK_AMR_D_GALLERY_BASE_STEPS=200 \
KRK_AMR_D_GALLERY_STEP_EXPONENT=2 \
KRK_AMR_D_GALLERY_AVG_WINDOW=50 \
KRK_AMR_TAG=gallery_local_20260505 \
julia --project=. benchmarks/amr_d_convergence_gallery_2d.jl
```

Artifacts:

- `amr_d_convergence_gallery_2d_gallery_local_20260505.csv`
- `amr_d_cylinder_nested4_probe_2d_gallery_local_20260505.csv`
- `figures/amr_d_convergence_errors_2d_gallery_local_20260505.png`
- `figures/amr_d_convergence_errors_2d_gallery_local_20260505.pdf`
- `figures/amr_d_convergence_cost_2d_gallery_local_20260505.png`
- `figures/amr_d_convergence_cost_2d_gallery_local_20260505.pdf`
- `figures/amr_d_cylinder_nested4_probe_2d_gallery_local_20260505.png`
- `figures/amr_d_cylinder_nested4_probe_2d_gallery_local_20260505.pdf`

## Local Scale 1-2-4 Ramp

Command:

```bash
KRK_AMR_D_GALLERY_SCALES=1,2,4 \
KRK_AMR_D_GALLERY_BASE_STEPS=80 \
KRK_AMR_D_GALLERY_STEP_EXPONENT=2 \
KRK_AMR_D_GALLERY_AVG_WINDOW=20 \
KRK_AMR_TAG=gallery_local_scale124_20260505 \
julia --project=. benchmarks/amr_d_convergence_gallery_2d.jl
```

Artifacts:

- `amr_d_convergence_gallery_2d_gallery_local_scale124_20260505.csv`
- `amr_d_cylinder_nested4_probe_2d_gallery_local_scale124_20260505.csv`
- `figures/amr_d_convergence_errors_2d_gallery_local_scale124_20260505.png`
- `figures/amr_d_convergence_errors_2d_gallery_local_scale124_20260505.pdf`
- `figures/amr_d_convergence_cost_2d_gallery_local_scale124_20260505.png`
- `figures/amr_d_convergence_cost_2d_gallery_local_scale124_20260505.pdf`
- `figures/amr_d_cylinder_nested4_probe_2d_gallery_local_scale124_20260505.png`
- `figures/amr_d_cylinder_nested4_probe_2d_gallery_local_scale124_20260505.pdf`

## Nesting Verdict

`cylinder_nested4_probe.krk` parses four nested `Refine` levels. Conservative
tree D then rejects it with:

```text
nested Refine parent blocks are not yet supported by conservative-tree AMR
```

So the current plots are valid for static one-level D. A real four-level
cylinder calculation still requires multi-level route ownership and routing.
