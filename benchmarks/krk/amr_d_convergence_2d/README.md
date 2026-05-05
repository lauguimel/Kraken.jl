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
