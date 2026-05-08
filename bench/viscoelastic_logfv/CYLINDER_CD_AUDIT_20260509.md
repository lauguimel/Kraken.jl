# Cylinder Cd Audit - 2026-05-09

Benchmark harness: `bench/viscoelastic_logfv/cylinder_cd_convergence.jl`

Remote execution copy: `aqua:Kraken.jl-viscoelastic-run`

Jobs:

- Smoke: `21022455.aqua` passed on A100.
- Liu direct-C, `Lambda_p=1e-6`: `21022479.aqua`.
- Liu direct-C, `Lambda_p=2.5e-7`: `21022480.aqua`.
- RheoTool R30, `Lambda_p=1e-6`: `21022863.aqua`.
- RheoTool R30, `Lambda_p=2.5e-7`: `21022864.aqua`.

Local CSV copies are under `tmp/aqua_viscoelastic_logfv/`.

## Newtonian Check

Reference: RheoTool Newtonian `Cd = 132.362236515`.

| R | Cd | err |
|---:|---:|---:|
| 20 | 131.295992 | -0.806% |
| 30 | 131.987599 | -0.283% |
| 35 | 132.176933 | -0.140% |

The Newtonian baseline is converging and does not show the large viscoelastic
defect.

## Liu Direct-C

`Cd_report == Cd_post` here because `solvent_source_mode=integrated_collision`.
`Cd_split` is not available in this force path.

| Lambda_p | R | Wi | Cd | Liu Cd | err | min det C |
|---:|---:|---:|---:|---:|---:|---:|
| 1e-6 | 20 | 0.1 | 128.234061 | 129.42 | -0.916% | 0.996 |
| 1e-6 | 20 | 0.5 | 124.066811 | 125.17 | -0.881% | 0.636 |
| 1e-6 | 20 | 1.0 | 201.488205 | 164.26 | +22.664% | -284 |
| 1e-6 | 30 | 0.1 | 129.514036 | 130.36 | -0.649% | 0.997 |
| 1e-6 | 30 | 0.5 | 125.550966 | 126.31 | -0.601% | 0.779 |
| 1e-6 | 30 | 1.0 | 176.515185 | 151.31 | +16.658% | -938 |
| 1e-6 | 35 | 0.1 | 129.837007 | 130.77 | -0.713% | 0.997 |
| 1e-6 | 35 | 0.5 | 125.767871 | 127.72 | -1.528% | 0.844 |
| 1e-6 | 35 | 1.0 | 164.388180 | 149.04 | +10.298% | -825 |
| 2.5e-7 | 20 | 0.1 | 128.131626 | 129.42 | -0.995% | 0.993 |
| 2.5e-7 | 20 | 0.5 | 124.169678 | 125.17 | -0.799% | 0.566 |
| 2.5e-7 | 20 | 1.0 | 198.991674 | 164.26 | +21.144% | -253 |
| 2.5e-7 | 30 | 0.1 | 129.466742 | 130.36 | -0.685% | 0.984 |
| 2.5e-7 | 30 | 0.5 | 125.773281 | 126.31 | -0.425% | 0.729 |
| 2.5e-7 | 30 | 1.0 | 177.918151 | 151.31 | +17.585% | -1049 |
| 2.5e-7 | 35 | 0.1 | 129.802203 | 130.77 | -0.740% | 0.980 |
| 2.5e-7 | 35 | 0.5 | 125.887730 | 127.72 | -1.435% | 0.800 |
| 2.5e-7 | 35 | 1.0 | 165.427477 | 149.04 | +10.995% | -926 |

Finding: low and medium Wi reproduce Liu at O(1%) without magic fitting. Wi=1
does not: the direct-C conformation loses SPD and Cd becomes too high.
Changing `Lambda_p` from `1e-6` to `2.5e-7` does not remove the defect.

## RheoTool R30

Direct-C comparison against local RheoTool means:

| Lambda_p | Wi | Cd | RheoTool Cd | err | min det C |
|---:|---:|---:|---:|---:|---:|
| 1e-6 | 0.05 | 130.860279 | 131.813036 | -0.723% | 0.994 |
| 1e-6 | 0.1 | 129.514036 | 130.428774 | -0.701% | 0.997 |
| 1e-6 | 0.2 | 126.672926 | 126.831070 | -0.125% | 0.990 |
| 1e-6 | 0.5 | 125.550966 | 119.288284 | +5.250% | 0.779 |
| 1e-6 | 1.0 | 176.515185 | 116.995397 | +50.874% | -938 |
| 2.5e-7 | 0.05 | 130.821240 | 131.813036 | -0.752% | 0.987 |
| 2.5e-7 | 0.1 | 129.466742 | 130.428774 | -0.738% | 0.984 |
| 2.5e-7 | 0.2 | 126.644519 | 126.831070 | -0.147% | 0.986 |
| 2.5e-7 | 0.5 | 125.773281 | 119.288284 | +5.436% | 0.729 |
| 2.5e-7 | 1.0 | 177.918151 | 116.995397 | +52.073% | -1049 |

Finding: the near-Newtonian/cylinder force path is not the 8-10% issue. It is
within 1% up to Wi=0.1 and excellent at Wi=0.2. The direct-C defect starts
between Wi=0.2 and Wi=0.5, then becomes SPD loss at Wi=1.

## Log-Conformation Population Path

The legacy log-conformation population path returns `Cd=NaN` for every
RheoTool case from Wi=0.05 to Wi=1.0, for both `Lambda_p` values. Final
`min_det_C` stays positive and close to one while `nonfinite_C=true`.

Finding: this is not the same signature as the direct-C high-Wi SPD loss.
The log-conf NaN is likely in the hydrodynamic/source/force chain, not simply
in the SPD algebra of `C`.

## Next Canaries

1. Direct-C high-Wi canary: frozen smooth cylinder or channel velocity, no
   solvent feedback, track `min_det_C` and local stress growth from Wi=0.2 to
   1.0.
2. Direct-C source canary: isolate integrated Hermite source with a prescribed
   positive-definite stress field and check whether the momentum update alone
   creates the Wi=0.5 drift.
3. Log-conf force-chain canary: one-step log-conf cylinder/channel with
   `LogFieldWallBC`, then inspect `C`, `tau_p`, Hermite source populations,
   `rho/ux/uy`, and MEA force separately. Since `C` remains SPD-like while
   `Cd` is NaN, start at the source-to-solvent coupling, not the log algebra.
4. Do not tune `Lambda_p` as a fix. The two tested values have the same failure
   pattern.
