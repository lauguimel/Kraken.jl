# AMR D Convergence Gallery Cases

These `.krk` files are the source templates for the AMR D convergence gallery.
The runner scales the scale-1 dimensions, patch ranges and obstacle geometry
with `KRK_AMR_D_GALLERY_SCALES`.

Run a local smoke:

```bash
KRK_AMR_D_GALLERY_SCALES=1 \
KRK_AMR_D_GALLERY_BASE_STEPS=40 \
KRK_AMR_D_GALLERY_AVG_WINDOW=10 \
KRK_AMR_TAG=gallery_smoke \
julia --project=. benchmarks/amr_d_convergence_gallery_2d.jl
```

Run the local ramp used for doc figures:

```bash
KRK_AMR_D_GALLERY_SCALES=1,2 \
KRK_AMR_D_GALLERY_BASE_STEPS=200 \
KRK_AMR_D_GALLERY_STEP_EXPONENT=2 \
KRK_AMR_D_GALLERY_AVG_WINDOW=50 \
KRK_AMR_TAG=gallery_local \
julia --project=. benchmarks/amr_d_convergence_gallery_2d.jl
```

Run a three-point local ramp:

```bash
KRK_AMR_D_GALLERY_SCALES=1,2,4 \
KRK_AMR_D_GALLERY_BASE_STEPS=80 \
KRK_AMR_D_GALLERY_STEP_EXPONENT=2 \
KRK_AMR_D_GALLERY_AVG_WINDOW=20 \
KRK_AMR_TAG=gallery_local_scale124 \
julia --project=. benchmarks/amr_d_convergence_gallery_2d.jl
```

The nested cylinder file is a probe, not a validated production input. It
parses four nested `Refine` levels, then the conservative-tree D helper must
reject it until multi-level route ownership and routing are implemented.

Files ending in `nested4_debug.krk` are quicklook/debug inputs, not publication
convergence gates. They use stronger local forcing or wall speed so field and
profile plots are readable after a short local run. The
`cylinder_lift_nested4_probe.krk` input is an off-centre cylinder target for
future lift/CD validation; today it is expected to produce static mesh and
solid-mask plots only.

Nested subcycled channel/solid cases accept two optional numeric A/B knobs:

- `Define route_sampling = 0` selects the production nested AMR-D route
  contract. `1` enables the experimental level-native route table, and `2`
  selects the hybrid route table. Nested D defaults to `0`; keep
  `route_sampling = 1` for surgical transport experiments only until it passes
  the nested y-band Poiseuille invariance tests.
- `Define c2f_prolongation = 0` keeps the production flat coarse-to-fine
  packet geometry. `1` enables the explicit experimental limited-linear
  prolongation. `poiseuille_yband_nested4_limited_debug.krk` is the reference
  A/B input for this path.
- `Define coarse_to_fine_predictor_weight = 0.5` controls the conservative
  temporal predictor blend. Use `0` for committed parent state only and `1`
  for the local post-collision parent predictor. The default is `0.5` for the
  production leaf-equivalent route contract and `1` only when
  `route_sampling = 1` is requested explicitly.
