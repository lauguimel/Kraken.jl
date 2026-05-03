# Voie D Phase P Summary

Scope: fixed ratio-2 patch refinement for D2Q9, validated on Couette, Poiseuille bands, square obstacle drag, and backward-facing-step (BFS/VFS) smoke comparison.

## Couette

| case | steps | mass drift | L2 error | Linf error |
|---|---:|---:|---:|---:|
| Couette central patch | 3000 | -5.206857e-11 | 3.945763e-04 | 6.352239e-04 |

## Poiseuille Bands

| case | steps | mass drift | L2 error | Linf error |
|---|---:|---:|---:|---:|
| vertical band x=L/2 | 5000 | -9.640644e-11 | 1.197781e-03 | 2.374683e-03 |
| horizontal band y=L/2 | 5000 | -1.003286e-10 | 1.126086e-03 | 2.374685e-03 |

## Square Obstacle

| case | steps | avg window | mass drift | Fx | Fy | ux mean |
|---|---:|---:|---:|---:|---:|---:|
| refined patch | 1200 | 300 | -5.837819e-11 | 3.184820e-03 | -2.312965e-20 | 1.613849e-03 |
| coarse Cartesian | 1200 | 300 | -8.981260e-12 | 3.228638e-03 | 2.324529e-17 | 5.045423e-04 |

Fx ratio refined/coarse: 0.986428

## Backward-Facing Step

| case | steps | mass final | ux mean | uy mean |
|---|---:|---:|---:|---:|
| voie D | 800 | 3.614053e+02 | 2.384672e-02 | -1.732332e-03 |
| Cartesian leaf | 800 | 3.611192e+02 | 2.371713e-02 | -1.714641e-03 |

BFS ux mean delta: 1.295901e-04
BFS uy mean delta: -1.769182e-05

## Acceptance Gates

| metric | gate |
|---|---:|
| Couette mass drift | `< 1e-8` |
| Couette L2 / Linf | `< 1e-3` / `< 2e-3` |
| Poiseuille band mass drift | `< 1e-8` |
| Poiseuille band L2 / Linf | `< 2e-3` / `< 3e-3` |
| Square obstacle mass drift | `< 1e-8` |
| Square obstacle drag ratio | `0.85 < Fx_refined/Fx_coarse < 1.15` |
| Square obstacle lift ratio | `< 1e-10` |
| BFS/VFS mean velocity deltas | `|dux| < 5e-4`, `|duy| < 5e-5` |
| BFS/VFS mass-final delta | `< 1.0` |

Non-claims: no dynamic AMR, no native dx-local streaming, no subcycling, no cylinder validation in Phase P.
