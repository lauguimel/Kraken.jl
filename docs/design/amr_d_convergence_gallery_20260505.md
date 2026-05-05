# AMR D Convergence Gallery

Date: 2026-05-05

## Scope

This gallery adds reproducible convergence-style plots for the current static
one-level D stream:

- Couette;
- Poiseuille with the vertical X band;
- Poiseuille with the horizontal Y band;
- BFS;
- square obstacle;
- cylinder.

The comparison is `cartesian_coarse` versus `amr_route_native`, with
`leaf_oracle` as the reference. For profiles the plotted metric is profile
error versus the dense leaf oracle; the analytic profile error is kept in the
CSV as secondary context.

## Artifacts

Manifest:

- `benchmarks/results/AMR_D_CONVERGENCE_GALLERY_RESULTS.md`

Runner:

- `benchmarks/amr_d_convergence_gallery_2d.jl`

Source `.krk` templates:

- `benchmarks/krk/amr_d_convergence_2d/*.krk`

Local ramp outputs:

- `benchmarks/results/amr_d_convergence_gallery_2d_gallery_local_20260505.csv`
- `benchmarks/results/figures/amr_d_convergence_errors_2d_gallery_local_20260505.png`
- `benchmarks/results/figures/amr_d_convergence_cost_2d_gallery_local_20260505.png`
- `benchmarks/results/figures/amr_d_cylinder_nested4_probe_2d_gallery_local_20260505.png`

Three-point local ramp outputs:

- `benchmarks/results/amr_d_convergence_gallery_2d_gallery_local_scale124_20260505.csv`
- `benchmarks/results/figures/amr_d_convergence_errors_2d_gallery_local_scale124_20260505.png`
- `benchmarks/results/figures/amr_d_convergence_cost_2d_gallery_local_scale124_20260505.png`
- `benchmarks/results/figures/amr_d_cylinder_nested4_probe_2d_gallery_local_scale124_20260505.png`

## Commands

Surgical test:

```bash
julia --project=. -e 'using Test; include("test/test_amr_d_convergence_gallery.jl")'
```

Local ramp:

```bash
KRK_AMR_D_GALLERY_SCALES=1,2 \
KRK_AMR_D_GALLERY_BASE_STEPS=200 \
KRK_AMR_D_GALLERY_STEP_EXPONENT=2 \
KRK_AMR_D_GALLERY_AVG_WINDOW=50 \
KRK_AMR_TAG=gallery_local_20260505 \
julia --project=. benchmarks/amr_d_convergence_gallery_2d.jl
```

Three-point local ramp:

```bash
KRK_AMR_D_GALLERY_SCALES=1,2,4 \
KRK_AMR_D_GALLERY_BASE_STEPS=80 \
KRK_AMR_D_GALLERY_STEP_EXPONENT=2 \
KRK_AMR_D_GALLERY_AVG_WINDOW=20 \
KRK_AMR_TAG=gallery_local_scale124_20260505 \
julia --project=. benchmarks/amr_d_convergence_gallery_2d.jl
```

Aqua ramp:

```bash
KRK_AMR_D_GALLERY_SCALES=1,2,4 \
KRK_AMR_D_GALLERY_BASE_STEPS=1200 \
KRK_AMR_D_GALLERY_STEP_EXPONENT=2 \
KRK_AMR_D_GALLERY_AVG_WINDOW=300 \
KRK_AMR_TAG=aqua_D_gallery_20260505 \
qsub hpc/amr_d_convergence_gallery_2d_aqua.pbs
```

## Nesting Probe

`benchmarks/krk/amr_d_convergence_2d/cylinder_nested4_probe.krk` defines four
nested cylinder refinement levels. The parser accepts the file, then the
conservative-tree D helper rejects it before runtime because nested `parent`
refinement is not implemented.

This is the correct current behavior. The final four-level cylinder calculation
needs a new multi-level D milestone:

- nested ownership across levels;
- 2:1 balance enforcement;
- route tables across level jumps;
- collision and streaming over all active leaves;
- restriction/prolongation between adjacent levels only;
- cylinder drag measured on the finest active leaves.
