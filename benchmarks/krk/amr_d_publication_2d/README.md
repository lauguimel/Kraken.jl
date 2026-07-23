# AMR D Publication Cases

These `.krk` files are the reproducibility source for the AMR D publication
tables.

Each file describes one physical benchmark case:

- `square_scale{1,2,4}.krk`
- `cylinder_scale{1,2,4}.krk`

The runner `benchmarks/amr_d_publication_from_krk_2d.jl` reads these files and
executes three methods for each case:

- `cartesian_coarse`
- `leaf_oracle`
- `amr_route_native`

The `.krk` `Refine publication_patch` block is the static D patch. The current
publication strategy is interface-buffered: the obstacle and near wake live in
the refined band, and the coarse/fine interfaces are moved away from the body.

Reproduce the table:

```bash
KRK_AMR_TAG=from_krk_20260505 \
julia --project=. benchmarks/amr_d_publication_from_krk_2d.jl
```

Quick smoke:

```bash
KRK_AMR_D_CASES=square_scale1.krk,cylinder_scale1.krk \
KRK_AMR_D_STEPS_OVERRIDE=40 \
KRK_AMR_D_AVG_WINDOW_OVERRIDE=20 \
KRK_AMR_TAG=from_krk_smoke \
julia --project=. benchmarks/amr_d_publication_from_krk_2d.jl
```

Plot the aqua summary:

```bash
julia --project=. benchmarks/plot_amr_d_publication_2d.jl
```
