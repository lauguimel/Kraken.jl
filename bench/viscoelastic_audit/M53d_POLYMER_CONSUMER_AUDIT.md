# M53d — polymer-chain consumer audit post-bifurcation
Date: 2026-05-26

## TL;DR
Verdict: 2 (R), 0 (C), 10 (P). The only direct M53c fixture fallout is M2c; every polymer-chain failure inspected is on non-embedded or embedded-disabled paths and has no `wall_distance` / `wall_inv_distance` consumer to update.

## Per-failure classification

| # | Test | Failure value | Classification | Source kernel | Notes |
|---|------|--------------|----------------|---------------|-------|
| 1 | M2c diag1 `dudx[3,3]` | `-0.7347401208725196` vs `-sqrt(2)/2` | (R) | `src/fvfd/operators_2d.jl:127-134`, `1094-1129` | Plane-distance embedded gradient; fixture used centroid-set `ux`. |
| 2 | M2c diag2 `dudy[3,3]` | `-0.7347401208725196` vs `-sqrt(2)/2` | (R) | `src/fvfd/operators_2d.jl:127-134`, `1094-1129` | Same expected value as diag1. |
| 3 | M5d `fine.max_uy` | `0.006706453084356583` | (P) | `src/drivers/viscoelastic_logfv_2d.jl:2385-2418` | Periodic-x wall-y coupled Poiseuille; no embedded wall field. |
| 4 | M5e Couette `max_c_error` | `0.6702410166183791` | (P) | `src/drivers/viscoelastic_logfv_2d.jl:1313-1358` | Frozen analytic channel, no embedded geometry. |
| 5 | M5e Couette `max_tau_error` | `0.02010723049855137` | (P) | `src/drivers/viscoelastic_logfv_2d.jl:1351-1358` | Propagates from bad C. |
| 6 | M5e Couette `max_total_force_error` | `0.16085047124137178` | (P) | `src/drivers/viscoelastic_logfv_2d.jl:1356-1358` | BSD/force consequence of bad C/stress. |
| 7 | M5e Poiseuille `max_c_error` | `0.003811814285554438` | (P) | `src/drivers/viscoelastic_logfv_2d.jl:1313-1358` | Same non-embedded frozen path. |
| 8 | M5e Poiseuille `max_tau_error` | `0.00011435442856663324` | (P) | `src/drivers/viscoelastic_logfv_2d.jl:1351-1358` | Propagates from C. |
| 9 | M5e Poiseuille `max_total_force_error` | `0.000935110482890503` | (P) | `src/drivers/viscoelastic_logfv_2d.jl:1356-1358` | Force regression, not a fixture. |
| 10 | M7d square channel `rho` delta | `0.0008146915798177279` | (P) | `src/drivers/viscoelastic_logfv_2d.jl:419-474` | Default `embedded_gradient=false`; no wall field read. |
| 11 | M8h BFS `ux` delta | `0.00023373470870723275` | (P) | `src/drivers/viscoelastic_logfv_2d.jl:419-474` | Default `embedded_gradient=false`; no wall field read. |
| 12 | M8h BFS `rho` delta | `0.0011236871631143952` | (P) | `src/drivers/viscoelastic_logfv_2d.jl:419-474` | Same near-Newtonian mismatch. |

## Detail per-test

### M2c (R)
- Assertion + setup: `test/test_viscoelastic_logfv_patch_ladder.jl:632-640` calls `logfv_velocity_gradient_embedded_bc_aware_2d!` after setting `ux[3,3] = embedded_h.wall_distance[3,3]`, then asserts `dudx[3,3] ≈ -sqrt(2)/2` and `dudy[3,3] ≈ -sqrt(2)/2`.
- Source: `src/fvfd/operators_2d.jl:127-134` now computes the no-slip normal derivative with `wall_inv_distance_to_center`; the embedded kernel passes that field at `src/fvfd/operators_2d.jl:1094-1127` and the wrapper passes `embedded.wall_inv_distance_to_center` at `src/fvfd/operators_2d.jl:1157-1170`.
- Field/value: `dudx[3,3]` and `dudy[3,3]` are `-0.7347401208725196`.
- Proposed action: rebaseline both expected diagonal values to `-0.7347401208725196`, or better change the fixture velocity to be plane-distance-consistent and keep the analytic normal target.

### M5d (P)
- Assertion + setup: `test/test_viscoelastic_logfv_patch_ladder.jl:1505-1518` runs `run_viscoelastic_logfv_poiseuille_coupled_2d` with `polymer_substeps=:auto`, then asserts `fine.max_uy < 1e-12`.
- Field/value: `fine.max_uy = 0.006706453084356583`.
- Source: `src/drivers/viscoelastic_logfv_2d.jl:2385-2418` computes non-embedded `logfv_velocity_gradient_bc_aware_2d!`, applies halfway wall gradient correction, steps log-C from `dudx,dudy,dvdx,dvdy`, computes stress, BSD force, and Guo field forcing.
- Wall-field consumption: none. This driver allocates no embedded boundary and never reads `wall_distance`, `wall_inv_distance`, or `wall_inv_distance_to_center`.
- Classification: (P). A symmetric Poiseuille setup develops transverse velocity, so this is a coupled source-force physics regression, not a bifurcated-field consumer.

### M5e Couette (P, alarming)
- Assertion + setup: `test/test_viscoelastic_logfv_patch_ladder.jl:1535-1559` runs frozen Couette with `initial=:steady`, `max_steps=1`, `polymer_substeps=128`, and asserts `result.max_c_error < 5.0e-5`, `max_tau_error < 1.5e-6`, `max_total_force_error < 1.0e-12`.
- Field/value: `max_c_error = 0.6702410166183791`, `max_tau_error = 0.02010723049855137`, `max_total_force_error = 0.16085047124137178`.
- Source: `src/drivers/viscoelastic_logfv_2d.jl:1266-1271` initializes exactly from `_logfv_channel_reference_fields`; `src/drivers/viscoelastic_logfv_2d.jl:1313-1319` computes non-embedded channel gradients and halfway correction; `src/drivers/viscoelastic_logfv_2d.jl:1330-1344` advances log-C; `src/drivers/viscoelastic_logfv_2d.jl:1385-1423` compares against the analytic reference.
- WHY did C error jump from machine-precision to 0.67? Not through `wall_distance`: this frozen driver sets `fill!(is_solid,false)`, does not build embedded geometry, and uses analytic `y=(j-0.5)*dy` channel fields. The only plausible audited mechanism is a non-embedded channel-gradient/source-splitting regression: steady analytic C is no longer a fixed point under the computed gradient used by `logfv_step_oldroydb_log_2d!`.
- Action: revert or trace the non-embedded channel-gradient/source path before any rebaseline. No consumer switch to `wall_inv_distance_to_center` is available here.

### M5e Poiseuille (P, milder)
- Assertion + setup: same loop as Couette, with `flow=:poiseuille`; tolerances are `max_c_error < 1.5e-4`, `max_tau_error < 5.0e-6`, `max_total_force_error < 8.0e-6`.
- Field/value: `max_c_error = 0.003811814285554438`, `max_tau_error = 0.00011435442856663324`, `max_total_force_error = 0.000935110482890503`.
- Source/wall-field consumption: same non-embedded source as Couette, `src/drivers/viscoelastic_logfv_2d.jl:1313-1358`; no `wall_*` field.
- Classification: (P). Smaller than Couette but still a failed analytic steady-state preservation test, not a fixture.

### M7d (P)
- Assertion + setup: `test/test_viscoelastic_logfv_patch_ladder.jl:1680-1703` compares square-channel viscoelastic near-Newtonian flow against total-viscosity hydro and asserts `maximum(abs.(visco.rho[fluid] .- hydro.rho[fluid])) < 4e-4`.
- Field/value: `max_rho_delta = 0.0008146915798177279`.
- Source: `run_viscoelastic_logfv_square_channel_coupled_2d` builds `square_obstacle_channel_geometry_2d` at `src/drivers/viscoelastic_logfv_2d.jl:798-810` and enters `_run_viscoelastic_logfv_step_channel_coupled_2d`; default `embedded_gradient=false` at `src/drivers/viscoelastic_logfv_2d.jl:193`, so the step uses non-embedded `fvfd_velocity_gradient_2d!` at `src/drivers/viscoelastic_logfv_2d.jl:419-425`.
- Wall-field consumption: none in this failing default path.
- Classification: (P). Near-Newtonian density no longer matches the total-viscosity hydro baseline.

### M8h (P)
- Assertion + setup: `test/test_viscoelastic_logfv_patch_ladder.jl:2231-2254` compares BFS coupled near-Newtonian flow to passive hydro and asserts `ux` delta `< 2e-4` and `rho` delta `< 3e-4`.
- Field/value: `max_ux_delta = 0.00023373470870723275`; `max_rho_delta = 0.0011236871631143952`.
- Source: `run_viscoelastic_logfv_bfs_coupled_2d` builds `backward_facing_step_geometry_2d` at `src/drivers/viscoelastic_logfv_2d.jl:732-746`, then uses the same default non-embedded step path (`src/drivers/viscoelastic_logfv_2d.jl:419-474`).
- Wall-field consumption: none in this failing default path.
- Classification: (P). Near-Newtonian velocity/density drift is a coupled physics mismatch, not a wall-distance consumer issue.

## Recommended fix mission

### Re-baselines (R)
- M2c diag1/diag2 only: new observed values are both `-0.7347401208725196`.
- Prefer a fixture repair over a raw expected-value edit: because the setup uses centroid `wall_distance` to synthesize `ux` while the gradient operator now uses plane distance, set the fixture from the plane-distance contract if that field is intended test input.

### Consumer updates (C)
- None found. `src/fvfd/lowering_2d.jl:1-6` defines both centroid and plane-distance fields; `src/fvfd/lowering_2d.jl:204-209`, `501-506`, `523-528`, and `546-551` populate both; `src/fvfd/operators_2d.jl:127-134` and `1167-1170` already use `wall_inv_distance_to_center` for embedded gradient.
- Estimated LOC: 0 for consumer-field switches.

### Physics regressions (P)
- M5d, all six M5e failures, M7d, and both M8h failures should be handled as physics regressions. Recommended next mission: trace the non-embedded channel gradient/source/BSD path around `src/drivers/viscoelastic_logfv_2d.jl:1313-1358` and `2385-2418`; if M53c is the only intended change, revert until M5e Couette steady C is again a fixed point.

## Anti-pattern flags
- The M5e Couette hypothesis about embedded `wall_distance` is falsified by code: `run_viscoelastic_logfv_frozen_channel_cde_2d` does not build embedded geometry.
- The macro drivers construct embedded metadata in `_run_viscoelastic_logfv_step_channel_coupled_2d`, but the failing M7d/M8h calls keep `embedded_gradient=false`; do not blame `wall_inv_distance_to_center` without enabling that path.
- Existing worktree was dirty before this audit, including `src/` and `test/`; this report did not edit those files.
