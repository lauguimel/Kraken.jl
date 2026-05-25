# M50 — FVFD wall-derivative stencil caller audit
Date: 2026-05-26

## Stencil definitions
| Stencil | File:lines | What it computes (cell-center vs wall) |
|---|---|---|
| _fvfd_solid_bc_derivative_x_2d (:quadratic) | src/fvfd/operators_2d.jl:13-43 | Cell-center derivative at the first fluid center near solids; M49 shows it is not the halfway wall derivative. |
| _fvfd_solid_bc_derivative_x_2d (:linear) | src/fvfd/operators_2d.jl:25-34 | One-sided fluid-cell secant; acceptable as a low-order cell-centered fallback, not a halfway wall derivative. |
| _fvfd_solid_bc_derivative_y_2d (:quadratic) | src/fvfd/operators_2d.jl:45-75 | Cell-center derivative at the first fluid center near solids; M49 shows it is not the halfway wall derivative. |
| _fvfd_solid_bc_derivative_y_2d (:linear) | src/fvfd/operators_2d.jl:57-66 | One-sided fluid-cell secant; acceptable as a low-order cell-centered fallback, not a halfway wall derivative. |
| _fvfd_apply_embedded_wall_gradient_2d | src/fvfd/operators_2d.jl:127-139 | Embedded wall-normal correction using `phi[i,j] * wall_inv_distance`; this is a separate wall-position correction after the solid-aware derivative seed. |

## Direct call-sites in src/
| Call-site | File:lines | Consumer purpose | Intent: (a)/(b)/(c) | Notes |
|---:|---|---|---|---|
| 1 | src/fvfd/operators_2d.jl:737 | `∂τxx/∂x` term in `fx = div(tau)` at a fluid cell. | (a) cell-center | Correct usage: FVFD stress divergence is cell-centered. |
| 2 | src/fvfd/operators_2d.jl:740 | `∂τxy/∂y` term in `fx = div(tau)` at a fluid cell. | (a) cell-center | Correct usage. |
| 3 | src/fvfd/operators_2d.jl:744 | `∂τxy/∂x` term in `fy = div(tau)` at a fluid cell. | (a) cell-center | Correct usage. |
| 4 | src/fvfd/operators_2d.jl:747 | `∂τyy/∂y` term in `fy = div(tau)` at a fluid cell. | (a) cell-center | Correct usage. |
| 5 | src/fvfd/operators_2d.jl:1077 | `dudx` from `ux` in `fvfd_velocity_gradient_2d_kernel!`. | (b) wall | Wall-row output feeds log-Ψ source in production; M48/M49 culprit path. |
| 6 | src/fvfd/operators_2d.jl:1080 | `dudy` from `ux` in `fvfd_velocity_gradient_2d_kernel!`. | (b) wall | Wall-normal shear at halfwayBB wall row is returned at first-fluid center, not wall. |
| 7 | src/fvfd/operators_2d.jl:1083 | `dvdx` from `uy` in `fvfd_velocity_gradient_2d_kernel!`. | (b) wall | Same production gradient array. |
| 8 | src/fvfd/operators_2d.jl:1086 | `dvdy` from `uy` in `fvfd_velocity_gradient_2d_kernel!`. | (b) wall | Same production gradient array. |
| 9 | src/fvfd/operators_2d.jl:1110 | Seed `ux_gx` before embedded wall-gradient correction. | (c) ambiguous | Corrected at src/fvfd/operators_2d.jl:1122; residual tangential meaning needs a separate embedded audit. |
| 10 | src/fvfd/operators_2d.jl:1113 | Seed `ux_gy` before embedded wall-gradient correction. | (c) ambiguous | Not the M48 halfwayBB path when `embedded_gradient=false`. |
| 11 | src/fvfd/operators_2d.jl:1116 | Seed `uy_gx` before embedded wall-gradient correction. | (c) ambiguous | Corrected normal component, tangential component remains stencil-seeded. |
| 12 | src/fvfd/operators_2d.jl:1119 | Seed `uy_gy` before embedded wall-gradient correction. | (c) ambiguous | Needs runtime/analytical embedded-wall canary to classify fully. |

## Indirect consumers (via fvfd_velocity_gradient_2d! etc.)
| Wrapper | File:lines | Downstream consumer | Intent | Notes |
|---|---|---|---|---|
| `fvfd_velocity_gradient_2d!` | src/fvfd/operators_2d.jl:1137-1155 | Public/exported wrapper at src/Kraken.jl:332. | (b) for wall-row consumers | No kwarg to request wall vs cell-center semantics. |
| `fvfd_velocity_gradient_2d!(..., geometry)` | src/fvfd/operators_2d.jl:1178-1187 | Geometry wrapper; delegates to non-embedded stencil. | (b) for halfway wall rows | Same semantics as above. |
| `logfv_velocity_gradient_bc_aware_2d!` | src/kernels/logconformation_fv_2d.jl:919-926 | Thin wrapper around `fvfd_velocity_gradient_2d!`. | (b) | Used by validation/helper paths as log-FV gradient provider. |
| `run_viscoelastic_logfv_cylinder_coupled_2d` step loop | src/drivers/viscoelastic_logfv_2d.jl:418-438 | `dudx,dudy,dvdx,dvdy` feed `logfv_step_constitutive_log_2d!`. | (b) wall | Main M48/M49 mechanism when `embedded_gradient=false`, the M46/M48 halfwayBB setup. |
| Same step loop, polymer force | src/drivers/viscoelastic_logfv_2d.jl:462-465 | `logfv_polymer_force_bc_aware_2d!` -> tensor divergence. | (a) cell-center | Uses stress divergence, not velocity wall gradient; keep separate from fix. |
| Same step loop, BSD drag | src/drivers/viscoelastic_logfv_2d.jl:501-518 | Builds BSD stress/drag from the same gradient arrays. | (b) wall | Diagnostic/drag readout also inherits wall-row gradient bias. |
| `run_viscoelastic_logfv_cavity_coupled_2d` | src/drivers/cavity_driver_2d.jl:216-232 | Calls `fvfd_velocity_gradient_2d!`, then applies cavity half-cell wall correction before log source. | (b) wall, locally mitigated | The correction at src/drivers/cavity_wall_correction_2d.jl:49-83 is a precedent for wall-position handling. |
| `run_viscoelastic_logfv_cavity_coupled_2d` force | src/drivers/cavity_driver_2d.jl:248-252 | `logfv_polymer_force_bc_aware_2d!` -> tensor divergence. | (a) cell-center | Correct cell-centered force divergence usage. |
| `run_viscoelastic_logfv_frozen_channel_cde_2d` | src/drivers/viscoelastic_logfv_2d.jl:1306-1328 | Frozen channel gradient feeds Oldroyd-B log source. | (b) wall | In-tree validation/helper path, same wall-row issue. |
| `run_viscoelastic_logfv_poiseuille_coupled_2d` | src/drivers/viscoelastic_logfv_2d.jl:2372-2381 | Coupled Poiseuille gradient feeds Oldroyd-B log source. | (b) wall | Axis-aligned halfway wall path. |
| `run_viscoelastic_logfv_square_periodic_2d` | src/drivers/viscoelastic_logfv_2d.jl:2623-2634 | Periodic square-obstacle gradient feeds Oldroyd-B log source. | (b) wall | Embedded solid mask but non-embedded gradient wrapper. |
| `run_viscoelastic_logfv_bfs_passive_2d` | src/drivers/viscoelastic_logfv_2d.jl:2879-2890 | BFS passive CDE gradient feeds Oldroyd-B log source. | (b) wall | Open-x/wall-y helper path. |
| Embedded velocity-gradient wrapper | src/fvfd/operators_2d.jl:1157-1175 | Applies `_fvfd_apply_embedded_wall_gradient_2d` after the seed stencil. | (c) ambiguous | Normal derivative has wall intent; tangential seed needs separate classification. |

## Classification summary
- Direct call-sites total: 12
- Intent (a) cell-center: 4 (correct usage, no change needed)
- Intent (b) wall: 4 (BUGGY — needs proper wall stencil)
- Intent (c) ambiguous: 4 (needs runtime trace or embedded-wall canary to disambiguate)

## Fix scope recommendation
- Smallest change that fixes all (b) without disturbing (a): add an explicit wall-position derivative mode for velocity-gradient wall rows only; do not change `fvfd_tensor_divergence_2d!` default semantics.
- Files that would need to change: `src/fvfd/operators_2d.jl`, `src/kernels/logconformation_fv_2d.jl`, `src/drivers/viscoelastic_logfv_2d.jl`, likely `src/drivers/cavity_driver_2d.jl` only if unifying its existing correction path, plus focused tests/canaries.
- Estimated LOC: 80-150 LOC for mode plumbing plus tests; smaller if limited to non-embedded halfwayBB.
- Existing kwargs that could host the new mode (e.g. polymer_wall_extrap): yes for stress divergence only; better add a separate velocity-gradient kwarg because `polymer_wall_extrap` currently belongs to `fvfd_tensor_divergence_2d!`.
- Affected tests / benches that might need re-baselining: `test/test_fvfd_operators_2d.jl`, M49 wall-stencil canary, M48 halfway mesh convergence, M44/M46 cylinder follow-ups, BSD drag diagnostics using `_logfv_compute_bsd_drag_2d`.

## Anti-pattern flags
- One helper name (`polymer_wall_extrap`) is used for stress divergence but the suspected bug is velocity-gradient wall-row semantics; reusing it blindly would mix cell-center force and wall-gradient contracts.
- Public `fvfd_velocity_gradient_2d!` has no way to state whether boundary-adjacent values are cell-centered gradients or wall gradients.
- Embedded and non-embedded gradient paths have different correction semantics hidden behind similarly named wrappers.

## Recommendation to Boss
- Next mission: design + implement a new explicit halfway wall-gradient mode for velocity-gradient wall rows, rerun M49 to PASS for P2/P3, then rerun M48 to test whether the halfwayBB Wi=1 U-shape flattens.
- Caveats / unknowns: embedded-gradient tangential components need a separate canary; tensor-divergence call-sites should remain cell-centered; cavity already has a half-cell wall correction and should not be double-corrected.
