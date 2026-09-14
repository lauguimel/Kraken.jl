# Viscoelastic Cd decomposition

Isolated AQUA run in `~/Kraken_visco_runs/visco_debug_20260428_093552/repo`.

Target values in `hpc/low_mach_visco_validation.jl` are Liu-style
grid-reference values for the confined cylinder (`R=20 → Cd=129.42`,
`R=30 → 130.36`, `R=40 → 130.79`). The rheoTool inertial Oldroyd-BLog
run at the same physical case gives `Cd=130.428774404`, useful as an
independent reference but not the `R=20` table target.

## Sweep result with explicit drag split

| Case | R | Cd ref | Cd_s | Cd_p link | Cd total | Error |
|---|---:|---:|---:|---:|---:|---:|
| logconf | 20 | 129.42 | 120.276 | 13.093 | 133.369 | +3.05% |
| direct | 20 | 129.42 | 121.717 | 13.413 | 135.130 | +4.41% |
| logconf | 30 | 130.36 | 120.787 | 17.444 | 138.231 | +6.04% |
| direct | 30 | 130.36 | 122.378 | 18.167 | 140.545 | +7.81% |
| logconf | 40 | 130.79 | 121.241 | 20.464 | 141.706 | +8.35% |
| direct | 40 | 130.79 | 122.928 | 21.520 | 144.448 | +10.44% |

`Cd_s` is the cut-link Mei momentum exchange before the Hermite source.
`Cd_p link` is the current `compute_polymeric_drag_2d` staircase/link
stress integral. `Cd_post` is the raw MEA after CE-corrected Hermite stress
embedding into the hydrodynamic populations. `Cd_scaled` keeps the same
CE-corrected source for bulk dynamics, but rescales only the immediate source
contribution to boundary MEA by `(1-s_plus/2)`; this is now the default
reported `Cd`. `Cd_s + Cd_p link` remains a diagnostic split.

## R20 polymer drag diagnostic

| Case | Cd ref | Cd_s | Needed Cd_p | Link Cd_p | Circle Cd_p offset 1.0 | Post-MEA |
|---|---:|---:|---:|---:|---:|---:|
| logconf | 129.42 | 120.246 | 9.174 | 13.083 | 4.248 | 144.894 |
| direct | 129.42 | 121.685 | 7.735 | 13.402 | 4.356 | 146.945 |

Interpretation:

1. Newtonian LI-BB is valid at low Mach, so the hydrodynamic cut-link
   MEA is not the primary error.
2. Homogeneous conformation tests pass for relaxation, shear,
   Poiseuille `N1`, and planar elongation, so the constitutive source is
   not the first suspect.
3. The current link-based `Cd_p` is not a convergent surface quadrature:
   for the analytic field `τ_xx=x-cx`, exact `F_x=πR²`, while
   `compute_polymeric_drag_2d` gives about `3.1×` too much.
4. A naive circle interpolation gives too little polymer drag for R20.
   Therefore the remaining blocker is the force accounting path:
   Liu/Yu computes total `Cd` by post-source MEA, while an explicit
   `Cd_s + ∮τ_p·n ds` split needs its own geometry-consistent quadrature.

Next debugging step: validate the Liu/Yu path (`Cd = Cd_post`) against
R20/R30/R40. If it remains biased, debug the Hermite source amplitude and
source/paroi ordering. In parallel, keep the explicit split as a diagnostic
and replace `compute_polymeric_drag_2d` by a geometry-consistent quadrature
before using `Cd_s + Cd_p` as a physical result.

## Unit tests before full Cd

`test/test_viscoelastic_force_accounting.jl` isolates the path
`τ_p → Hermite source → post-source MEA` without running a cylinder flow:

- Standalone Hermite source conserves mass and momentum and recovers the
  expected second-moment closure:
  `ΔΠ_αβ = -s_plus τ_αβ / (1 - s_plus/2)`.
- The standalone post-collision source is larger than the in-collision
  Liu/Yu-style source by exactly `1/(1 - s_plus/2)`.
- A single artificial cut link confirms `compute_drag_libb_mei_2d` is doing
  exactly what the source puts in the populations; this is not a host/GPU or
  f_in/f_out bookkeeping bug.
- For the analytic circular stress field `τ_xx=x-cx`, exact
  `F_x=πR²`; post-source MEA gives ratios `3.49` at `R=20` and `3.41` at
  `R=40`. This reproduces the pre-Cd force-accounting problem directly.
- The same analytic test with the in-collision/no-CE-denominator source gives
  ratios `1.31` at `R=20` and `1.28` at `R=40`. The ratio between both paths
  is exactly `1/(1-s_plus/2)`, so the excess splits into a CE-denominator
  factor and a smaller link-geometry factor.

Conclusion: the immediate blocker is not the conformation bulk update. It is
the combination of Hermite source amplitude/timing with boundary MEA.
`run_conformation_cylinder_libb_2d` now keeps
`hermite_source_mode=:ce_corrected` by default for the dynamics, and reports
`drag_mode=:source_scaled_mea` by default for the force.

The rejected alternative was `hermite_source_mode=:liu_direct` for the whole
driver: a full R20 Metal run gave `Cd≈91.74`, far below Liu `129.42`, so it
under-couples polymer stress in the bulk.

Full AQUA/A100 runs with `hermite_source_mode=:ce_corrected` show why the
reported force must use `source_scaled_mea` rather than raw `post_source_mea`:

| Case | R | Cd_post | Cd_scaled | Liu ref | Error scaled | Cd_s | Cd_split |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logconf | 20 | 144.943 | 126.725 | 129.420 | -2.08% | 120.276 | 133.369 |
| logconf | 30 | 142.719 | 128.394 | 130.360 | -1.51% | 120.787 | 138.231 |
| logconf | 40 | 140.525 | 129.235 | 130.790 | -1.19% | 121.241 | 141.706 |

Raw `Cd_post` is too high by 7–12%; explicit `Cd_s + Cd_p link` also drifts
high with resolution. `Cd_scaled` is consistently within about 1–2% of Liu
and improves with R. This unblocks the Cd validation path; the remaining
bias is likely the residual link-geometry factor exposed by the analytic
force-accounting unit test.

## Current small-difference suspect: conformation TRT magic

The code now exposes the conformation/log-conformation TRT parameter as
`conformation_magic`, with environment control through
`KRAKEN_CONFORMATION_MAGIC`. Kraken's historical default remains `0.25`.
The Liu/Yu convention documented in `bench/equations_cross_check.md` uses a
much smaller Λₚ, typically `≈1e-6` or case-optimised near walls.

This controlled test was run on AQUA in
`~/Kraken_visco_runs/visco_magic_20260428_131539/repo`:

```bash
KRAKEN_R_LIST=30 \
KRAKEN_FORMULATIONS=logconf \
KRAKEN_MAGIC_LIST=0.25,0.01,0.0001,0.000001 \
KRAKEN_DRAG_MODE=source_scaled_mea \
KRAKEN_HERMITE_SOURCE_MODE=ce_corrected \
julia --project=. hpc/visco_magic_sweep.jl
```

Result at `R=30`, log-conformation, A100/Float64:

| Λₚ | Cd_scaled | Liu ref | Error |
|---:|---:|---:|---:|
| 0.25 | 128.3937 | 130.3600 | -1.51% |
| 0.01 | 174.5943 | 130.3600 | +33.93% |
| 0.0001 | 182.1699 | 130.3600 | +39.74% |
| 0.000001 | NaN | 130.3600 | NaN |

Conclusion: blindly switching to a tiny Liu-style Λₚ does not fix the
remaining 1–2%; it destabilises this implementation with the current wall
treatment. Keep `conformation_magic=0.25` as the working value until the
wall-bounded Liu/Yu Λₚ optimisation is reproduced explicitly.

## Rejected MEA postpair hypothesis

Another plausible small bug was that `compute_drag_libb_mei_2d` might
reconstruct a Bouzidi reflection although `f_out` already contains the
overwritten reflected population. A diagnostic direct-pair mode was added as
`momentum_exchange_mode=:postpair`.

AQUA run `~/Kraken_visco_runs/visco_mea_postpair_20260428_133349/repo`,
`R=30`, A100/Float64:

| Case | MEA mode | Cd | Reference | Error |
|---|---|---:|---:|---:|
| Newtonian | postpair | 87.6764 | 132.3622 | -33.76% |
| logconf | postpair | 75.0969 | 130.3600 | -42.39% |

Conclusion: the direct post-boundary pair formula is not the correct
interpolated LI-BB force for this driver. The existing
`momentum_exchange_mode=:mei_reconstruct` remains the only validated MEA
path because it preserves the Newtonian benchmark.
