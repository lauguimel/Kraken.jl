# Force/source scheme matrix — R30 centered geometry

Run: `20474525.aqua`, isolated directory
`~/Kraken_visco_runs/force_matrix_centered_20260428_1655/repo`.

Configuration:
- backend: CUDA A100, Float64
- geometry mode: `centered_legacy`
- grid: `Nx=900`, `Ny=120`, `R=30`, `cx=450`, `cy=59.5`
- flow: `u_mean=0.005`, `Re_R=1`, `Re_D=2`
- visco: `β=0.59`, `Wi=0.1`, log-conformation, `Λp=0.25`
- time: `steps_per_R=4000`, `avg_divisor=10`, `drag_stride=200`

Reference values:
- Newtonian rheoTool Re=1 target: `Cd=132.362236515`
- Liu/Yu visco target at R30, Wi=0.1: `Cd=130.36`

| family | case | Cd | Cl | reference | error |
|---|---|---:|---:|---:|---:|
| Newtonian | `libb_mei` | 132.076371 | 3.97e-13 | 132.362237 | -0.216% |
| Newtonian | `libb_liu_eq63` | 132.076371 | 3.97e-13 | 132.362237 | -0.216% |
| Newtonian | `libb_simple` | 131.053103 | 5.12e-13 | 132.362237 | -0.989% |
| Newtonian | `libb_postpair` | 87.922487 | -5.39e-13 | 132.362237 | -33.574% |
| Visco | `post_ce_scaled` | 128.439103 | 1.17e-13 | 130.36 | -1.474% |
| Visco | `post_ce_scaled_liu` | 128.439103 | 1.17e-13 | 130.36 | -1.474% |
| Visco | `post_ce_raw` | 142.693038 | -1.43e-14 | 130.36 | +9.461% |
| Visco | `post_liu_raw` | 100.781020 | -1.46e-13 | 130.36 | -22.690% |
| Visco | `integrated_ce_raw` | 142.673800 | -1.57e-13 | 130.36 | +9.446% |
| Visco | `integrated_ce_liu` | 142.673800 | -1.57e-13 | 130.36 | +9.446% |
| Visco | `integrated_liu_raw` | 100.774737 | -6.21e-13 | 130.36 | -22.695% |

Conclusions:
- `centered_legacy` fixes the lift symmetry issue: `Cl` is zero to roundoff.
- It keeps the Newtonian benchmark valid: `Cd=132.076`, within `0.22%` of rheoTool.
- Liu Eq. 63 remains exactly equivalent to `mei_reconstruct`.
- Post-collision source and integrated collision source remain equivalent.
- The visco discrepancy is unchanged in nature: best current `Cd=128.439`, `-1.47%` vs Liu.
- A separate partial `exact_nodes` run (`Nx=901`, `Ny=121`, `cy=60`) also gave `Cl≈0`, but shifted Newtonian Cd to `128.613` (`-2.83%`), so it is not the right target grid for Liu/rheoTool comparison.
