# Force/source scheme matrix — R30

Run: `20474455.aqua`, isolated directory
`~/Kraken_visco_runs/force_matrix_20260428_1609/repo`.

Configuration:
- backend: CUDA A100, Float64
- grid: `Nx=900`, `Ny=120`, `R=30`
- flow: `u_mean=0.005`, `Re_R=1` in Liu/Yu convention, `Re_D=2` if diameter-based
- visco: `β=0.59`, `Wi=0.1`, log-conformation, `Λp=0.25`
- time: `steps_per_R=4000`, `avg_divisor=10`, `drag_stride=200`

Reference values:
- Newtonian rheoTool Re=1 target: `Cd=132.362236515`
- Liu/Yu visco target at R30, Wi=0.1: `Cd=130.36`

| family | case | force/source | Cd | reference | error |
|---|---|---:|---:|---:|---:|
| Newtonian | `libb_mei` | `mei_reconstruct` | 131.959965 | 132.362237 | -0.304% |
| Newtonian | `libb_liu_eq63` | `liu_eq63` | 131.959965 | 132.362237 | -0.304% |
| Newtonian | `libb_simple` | `simple_halfway` | 128.629206 | 132.362237 | -2.820% |
| Newtonian | `libb_postpair` | `postpair` | 87.676365 | 132.362237 | -33.760% |
| Visco | `post_ce_scaled` | post source, CE scaled | 128.393701 | 130.36 | -1.508% |
| Visco | `post_ce_scaled_liu` | post source, CE scaled, Eq. 63 | 128.393701 | 130.36 | -1.508% |
| Visco | `post_ce_raw` | post source, CE raw MEA | 142.719258 | 130.36 | +9.481% |
| Visco | `post_liu_raw` | post source, Liu-direct source | 100.744492 | 130.36 | -22.718% |
| Visco | `integrated_ce_raw` | integrated source, CE raw MEA | 142.700037 | 130.36 | +9.466% |
| Visco | `integrated_ce_liu` | integrated source, CE raw MEA, Eq. 63 | 142.700037 | 130.36 | +9.466% |
| Visco | `integrated_liu_raw` | integrated source, Liu-direct source | 100.738215 | 130.36 | -22.723% |

Conclusions:
- Liu/Yu Eq. 63 and Kraken `mei_reconstruct` are numerically identical for LI-BB.
- `postpair` is not a valid force formula for Kraken's interpolated LI-BB storage.
- Post-collision source and integrated collision source give the same Cd at fixed source amplitude.
- The remaining R30 visco gap is not caused by force Eq. 63 implementation or by post-vs-integrated source placement.
- The discrepancy is now isolated to source amplitude/CE accounting and/or the hydrodynamic/conformation boundary scheme difference versus Liu/Yu.
- The original run did not print `Cl`. A follow-up Newtonian R30 check showed
  `cy=2R` gives `Cl=-0.630`; with the current node-coordinate convention,
  exact channel symmetry is at `cy=(Ny-1)/2`, giving `Cl≈7.5e-4`.
