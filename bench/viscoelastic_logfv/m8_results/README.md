# M8 — Oldroyd-B cylinder benchmark results

Raw result CSVs from the M8 viscoelastic cylinder benchmark (Kraken vs RheoTool).
All Kraken runs: Aqua A100, Float64, 300 000 steps, diffusive scaling
(ν_total = 0.15, τ = 0.95), MUSCL–superbee, half-way bounce-back unless noted.

## Kraken (this directory)

| File | Job | Content |
|------|-----|---------|
| `matrix_halfwayBB_R{10,30,50}.csv` | 22135355 | Cd vs Wi∈{0.1,0.5,1.0} per R, default (halfwayBB) wall — the M8 gate |
| `matrix_bouzidi_fl_R{10,30,50}.csv` | 22135355 | same, Bouzidi-FL wall (NaN at R≥50 Wi≥0.5 — wall-BC NO-GO) |
| `beta_R50.csv` | 22135448 | β-sweep β∈{0.59,0.3,0.1,0.01} × Wi∈{0.5,1.0}, R=50 (β≤0.1 → NaN) |
| `staircase_R{30,50}.csv` | 22135448 | Wi_max staircase Wi∈{1.5,2,3,5,7,10}, β=0.59 (stability envelope) |

Column `Cd_kraken` is the drag coefficient; `nan_flag`=true marks a diverged case.

## RheoTool references — `../../rheotool/m8_refs/`

`beta{0.59,0.3}_wi{0.5,1.0}_Cd.txt` — rheoFoam (OF-9, Oldroyd-BLog, log-conf),
job 22135484, 2-core MPI, steady at t=20. β≤0.1 diverged (PETSc FPE) and is not
referenced. `N1_comparison.csv` holds the wake-N1 comparison (see the benchmark
page §"First normal-stress difference").

## Headline (R=50, halfwayBB, Wi=1.0)

Kraken Cd 119.24 vs RheoTool 120.38 → −0.96 % (mandate gate <1 % met across
Wi∈{0.1,0.5,1.0}). Full discussion: `docs/src/benchmarks/viscoelastic_cylinder.md`.
