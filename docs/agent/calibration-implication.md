---
module: platform/calibration
path: src/platform/calibration.jl
owner_concern: steady-calibration-stack
status: phase-2b-2
last_verified: 2026-06-12
depends_on: [platform/residual.jl, ad/ad_api.jl, ad/ad_adjoint.jl, ad/ad_forward.jl, platform/observe.jl]
---

# platform/calibration — implication map

Implements the steady calibration stack: `ParameterSpace` / `loss` / `fit` / `CalibResult`.
Enzyme-free except for the adjoint gradient, which delegates to `_ad_pvjp_nu`,
`_ad_vjp_GtT`, `_ad_dJdf`, and `_ad_dqwall_terms` stubs.

## Call graph: `fit`

```
fit(problem, ::LBM, data, p0, pspace; observables)
  ├─ to_flat(pspace, p0) → x0
  ├─ iteration k:
  │   ├─ from_flat(pspace, xk, p0) → p_named
  │   ├─ ad_forward_solve(; problem..., nu=νk) → fwd             [EXISTING]
  │   ├─ _obs_lineprofile_ux_from_f(fwd.f_star, obs) → preds     [NEW local]
  │   ├─ loss(preds, data) → L                                   [NEW]
  │   ├─ _dJ_df_lineprofile_ux(...) or _ad_dJdf(...) → dL/df      [NEW / EXISTING]
  │   ├─ gmres_adjoint(apply_GtT, dL/df) → λ                     [EXISTING]
  │   │     └─ apply_GtT = v -> _ad_vjp_GtT(fwd.f_star, v, ...)   [EXISTING ext]
  │   ├─ _ad_pvjp_nu(fwd.f_star, λ, LBMScalarParams(..., νk)) → dL/dν
  │   └─ project!(pspace, xk - α∇L) → xk+1
  └─ return CalibResult(p_opt, loss_final, loss_trace, grad_trace, n_iter, converged, message)
```

## ν chain

ν → `(s_plus, s_minus)` → `G` → `L`. `_ad_pvjp_nu` differentiates the ν→G chain
in one Enzyme call over `ad_step_nu!`. The calibrated `fit` loop reconstructs
`LBMScalarParams(geom, ν)` after bound projection, so the bundle passed to
`_ad_pvjp_nu` matches the ν used by the forward solve.

## Geometry parity guarantee

For `pspace={:radius}`, `fit` calls `_ad_dqwall_terms` and
`ad_assemble_radius_terms`, the same chain used by `_steady_drag_sensitivity`.
For `FieldReduction(:Cd, identity)`, the direct Cd residual scales the explicit
q-wall and `D = 2R` terms, while λ carries the implicit state response.
Gate 4c pins the direct chain against `steady_shape_sensitivity` at rel < 1e-6.

## Optimizer design

Projected gradient descent + Armijo backtracking. No new Project.toml deps.
Upgrade path: add Optim.jl as a weak-dependency extension for L-BFGS in a later phase.

## Reads from

- `src/platform/residual.jl`: `LBMGeomParams`, `LBMScalarParams`
- `src/ad/ad_api.jl`: `_ad_pvjp_nu`, `_ad_vjp_GtT`, `_ad_dJdf`, `_ad_dqwall_terms`
- `src/ad/ad_adjoint.jl`: `gmres_adjoint`, `AD_LINEAR_RES_TOL`
- `src/ad/ad_forward.jl`: `ad_forward_solve`
- `src/platform/observe.jl`: `Prediction`, `LineProfile`, `FieldReduction`
- `src/ad/ad_geometry.jl`: `dq_wall_dR_cylinder`, `ad_assemble_radius_terms`

## Writes to

Nothing persistent. `fit` allocates intermediate arrays per iteration and keeps no global state.
