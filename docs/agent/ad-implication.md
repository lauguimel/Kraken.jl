---
module: ad
path: src/ad/; ext/KrakenADExt.jl
owner_concern: lbm-operator
status: implemented
last_verified: 2026-06-03
depends_on:
  - lbm
  - physics-newtonian
  - bc
  - geometry
  - io-krk
---

# ad — module implication map

The `src/ad/` path owns the steady cylinder shape-adjoint capability:
`dCd/dR` for a residual-converged D2Q9 TRT/Li-BB cylinder. Core code is
Enzyme-free; `ext/KrakenADExt.jl` supplies the reverse-mode seams when the
weak dependency `Enzyme` is loaded. The central maintenance contract is that
`src/ad/ad_step.jl` and `src/ad/ad_qoi.jl` are bit mirrors of the production
TRT/Li-BB step and MEI drag path, guarded by the AD anti-drift tests.

## Public surface

- `steady_shape_sensitivity(; Nx, Ny, radius, u_in, ν, qoi=:drag, wrt=:radius, tol=1e-12, ...)` — exported API; returns a NamedTuple with `gradient` (`dCd/dR`), `qoi_value`, `value`, `solver`, `terms`, forward residual fields, and optional `fd_check`.
- `.krk` dispatch: a parsed `Sensitivity { qoi = drag, wrt = radius }` request is routed by `run_simulation(setup)` to `steady_shape_sensitivity`.
- Core implementation helpers are de-facto public as `Kraken.<name>` but are not exported: `ad_forward_solve`, `ad_step!`, `cd_pure`, `cd_production`, `gmres_adjoint`, `ad_assemble_radius_terms`, `ad_fd_dCd_dR`.
- Extension seam methods imported by `KrakenADExt`: `_ad_dJdf`, `_ad_vjp_GtT`, `_ad_dqwall_terms`. In core they throw "Load Enzyme"; the extension replaces them with Enzyme reverse passes.

## Reads from

- `lbm` / `physics-newtonian` — D2Q9 weights, directions, opposite pairs, TRT rate convention, and the production TRT/Li-BB algebra that `ad_step!` mirrors.
- `bc` — the west velocity / east pressure Zou-He rebuild algebra mirrored inside `ad_apply_zou_he_rebuild!`.
- `geometry` — `precompute_q_wall_cylinder` for the cut-link geometry and `dq_wall_dR_cylinder` for the analytic radius derivative.
- `io-krk` — the parser-owned `Sensitivity` request and runner extraction of `Nx`, `Ny`, `radius`, `u_in`, `ν`, `ρ_out`, `tol`, `gmres_tol`, and `adjoint_tol`.
- `Project.toml` weakdep/extension metadata — `KrakenADExt = "Enzyme"` keeps Enzyme out of the unconditional `using Kraken` load path.

## Writes to

- Returns a fresh NamedTuple from `steady_shape_sensitivity`; it writes no files and mutates no global registries.
- Mutates local host arrays during the forward fixed-point solve (`f_in`, `f_out`) and during one-step AD products (`out`, cotangent arrays). These arrays are allocated inside the call or copied from inputs.
- `ext/KrakenADExt.jl` allocates cotangent work arrays for `dJ/df`, `(dG/df)^T v`, `dCd/dq_wall`, and `d(lambda^T G)/dq_wall`.
- Optional `fd_check=true` re-runs two additional radius-perturbed forwards and returns their finite-difference data; it does not cache results globally.

## Backend constraints

- CPU Float64 only. The public API does not accept a backend keyword, and `.krk` sensitivity dispatch rejects non-`Float64` execution.
- No GPU kernel is differentiated. `ad_step!` is a plain Julia, host-side mirror of the production fused TRT/Li-BB path; production GPU forward runs still use `run_cylinder_libb_2d`.
- Memory is O(1) in the number of forward steps because Enzyme tapes one step at the converged `f*`, not the transient history. GMRES still allocates host Krylov work arrays proportional to the state size and restart.
- Enzyme is optional at package load but mandatory at call time. Without `using Enzyme`, core stubs throw before any derivative work starts.
- The q-wall Enzyme paths have finite-difference fallbacks and directional finite-difference guards because Enzyme can return nonfinite or zero cut-link cotangents for this mutation-heavy path.

## Failure modes

- **Weakdep seam not loaded** — calling `steady_shape_sensitivity` after only `using Kraken` fails at `_ad_dJdf` with the explicit "Load Enzyme" error. This is intentional packaging behavior from M-AD-P5.
- **Bit-mirror drift** — any production change to `fused_trt_libb_v2_step!`, the Zou-He rebuild, `compute_drag_libb_mei_2d`, direction tables, or TRT rates must be reflected in `src/ad/ad_step.jl` / `src/ad/ad_qoi.jl`. The AD anti-drift test in `test/ad/test_ad_sensitivity.jl` is the receipt: inline `Cd` and the production fused path must remain equivalent.
- **Dropped direct diameter term** — `dCd/dR` includes `direct_D = -Cd/R` because `Cd = 2Fx/(u_ref^2 D)` and `D=2R`. Losing this term gives a plausible but wrong sign/magnitude; see `ad_assemble_radius_terms`.
- **Differentiating the Boolean mask** — `is_solid` is held constant. The derivative is valid only within a smooth cut-set interval; finite differences crossing a topology change can disagree by construction.
- **Forward not converged tightly enough** — drag can lag the population residual, so sensitivity runs use residual-converged forwards (`tol` near `1e-12` in production validation). A loose forward tolerance contaminates both the adjoint and central FD check.
- **Unsupported API symbols** — anything except `qoi=:drag`, `wrt=:radius` throws `ArgumentError`. Do not document unvalidated pairs as extension points that work today.

## Touch order

For an AD sensitivity bug, inspect in this order:

1. `src/ad/ad_api.jl` — public keyword contract, Enzyme-stub calls, result fields, supported `qoi`/`wrt`, and tolerance wiring.
2. `ext/KrakenADExt.jl` — Enzyme reverse passes for `dJ/df`, `(dG/df)^T v`, q-wall cotangents, runtime-activity fallback, and directional-FD guards.
3. `src/ad/ad_geometry.jl` — `direct_D`, q-wall contractions, finite-difference `dCd/dR`, and returned `terms`.
4. `src/ad/ad_qoi.jl` — inline MEI drag mirror, q-wall finite-difference fallbacks, and `lambda^T G` scalar used for the implicit geometry term.
5. `src/ad/ad_adjoint.jl` — Richardson-to-GMRES solve, linear residual norm, restart limits, and `rhohat` stall detection.
6. `src/ad/ad_step.jl` — the unfused TRT/Li-BB plus Zou-He bit mirror; check this against `src/kernels/li_bb_2d_v2.jl` and `src/bc/rebuild_2d.jl` before changing constants.
7. `src/ad/ad_forward.jl` — residual-converged host solve, cylinder geometry setup, inlet profile, and `Cd` evaluation at `f*`.
8. `.krk` path: `src/io/krk/directives.jl`, `src/io/krk/parser.jl`, and `src/simulation_runner.jl` — only after the direct Julia API works but `Sensitivity { ... }` dispatch fails.
