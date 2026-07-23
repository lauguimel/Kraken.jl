# AMR D Closure Report

Date: 2026-05-05

## Scope

This closes the publication-facing D stream for static fixed-patch 2D AMR:

- D2Q9;
- one fixed ratio-2 patch;
- interface-buffered obstacle patch;
- square obstacle `u/v`;
- cylinder `u` and `Cd`;
- comparison against coarse Cartesian and dense Cartesian leaf oracle;
- accuracy and runtime-efficiency columns.

It does not close compact-patch arbitrary obstacle placement, subcycling,
dynamic adaptation, GPU AMR, or 3D obstacle drag.

## Aqua Run

PBS job: `20812821.aqua`

Command path:

```bash
KRK_AMR_D_PATCH_STRATEGY=interface_buffered \
KRK_AMR_D_SCALES=1,2,4 \
KRK_AMR_D_BASE_STEPS=2400 \
KRK_AMR_D_STEP_EXPONENT=1 \
KRK_AMR_D_AVG_WINDOW=600 \
KRK_AMR_TAG=aqua_D_pub_20260505_long \
qsub hpc/amr_d_publication_table_2d_aqua.pbs
```

Artifacts:

- `benchmarks/results/amr_d_publication_raw_2d_aqua_D_pub_20260505_long.csv`
- `benchmarks/results/amr_d_publication_summary_2d_aqua_D_pub_20260505_long.csv`
- `benchmarks/results/logs/amr_d_publication_2d_aqua_D_pub_20260505_long.log`

## Reproducibility And Figures

All retained publication-D outputs are indexed in:

- `benchmarks/results/AMR_D_PUBLICATION_RESULTS.md`

The reproducible case sources are the `.krk` files in:

- `benchmarks/krk/amr_d_publication_2d/*.krk`

They are executable through:

```bash
KRK_AMR_TAG=from_krk_20260505 \
julia --project=. benchmarks/amr_d_publication_from_krk_2d.jl
```

The `.krk` runner emits the same raw and summary schema as the aqua runner.
It also computes the AMR active-cell count directly from the parsed `Refine`
block, so the result table remains tied to the case file rather than to a
separate hardcoded patch table.

Publication figures are generated from the retained summary CSV with:

```bash
julia --project=. benchmarks/plot_amr_d_publication_2d.jl
```

Stored outputs:

- `benchmarks/results/figures/amr_d_publication_2d_summary.png`
- `benchmarks/results/figures/amr_d_publication_2d_summary.pdf`

## Accuracy Verdict

The reference is `leaf_oracle`. The coarse Cartesian baseline is intentionally
half the leaf resolution.

Cylinder `Cd` relative error:

| scale | cartesian_coarse | amr_route_native |
|---:|---:|---:|
| 1 | 6.931 | 0.0598 |
| 2 | 7.129 | 0.0563 |
| 4 | 5.249 | 0.0363 |

Cylinder mean-`u` absolute error:

| scale | cartesian_coarse | amr_route_native |
|---:|---:|---:|
| 1 | 2.036e-3 | 1.618e-4 |
| 2 | 8.242e-3 | 6.171e-4 |
| 4 | 2.837e-2 | 1.711e-3 |

Square mean-`u` absolute error:

| scale | cartesian_coarse | amr_route_native |
|---:|---:|---:|
| 1 | 2.000e-3 | 1.366e-4 |
| 2 | 7.198e-3 | 4.855e-4 |
| 4 | 2.544e-2 | 1.405e-3 |

Transverse `v` stays near roundoff for these symmetric forced obstacle cases.
Mass drift stays at roundoff scale for all AMR rows, with worst AMR relative
drift `2.08e-12`.

Accuracy closure: passed for the publication D claim. AMR is consistently much
closer to the dense Cartesian leaf oracle than the coarse Cartesian baseline.

## Efficiency Verdict

The interface-buffered AMR patch uses `87.5%` of the dense leaf cell count in
this obstacle setup.

Runtime speedup versus leaf oracle on aqua:

| flow | scale 1 | scale 2 | scale 4 |
|---|---:|---:|---:|
| square | 0.379 | 0.691 | 0.673 |
| cylinder | 0.751 | 0.679 | 0.676 |

Runtime efficiency closure: not passed as a speedup claim on CPU. The current
route-native prototype reduces active cell count but is slower wall-clock than
the dense leaf oracle because route dispatch and composite bookkeeping dominate
at these sizes.

The publishable wording is therefore:

> Static fixed-patch AMR D recovers near-leaf obstacle accuracy with fewer
> active cells than the dense leaf grid. The current CPU route-native prototype
> is not yet a runtime speedup; performance work belongs to the packed route,
> subcycling and GPU milestones.

## D Closure

D is closed as a correctness/accuracy publication feature for static
fixed-patch 2D AMR:

- Couette and Poiseuille route-native gates remain covered by surgical tests;
- square obstacle `u/v` accuracy is tabled against Cartesian baselines;
- cylinder `Cd` accuracy is tabled against Cartesian baselines;
- aqua long run records scale `{1,2,4}`;
- limits are explicit and not hidden behind abstraction.

Next work after D:

1. packed route CPU microbench to reduce route dispatch overhead;
2. subcycling to handle compact-patch obstacle placement;
3. GPU route kernel after packed route parity;
4. 3D obstacle drag only after 3D profile/subcycling gates.
