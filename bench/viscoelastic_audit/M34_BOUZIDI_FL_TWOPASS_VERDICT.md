# M34 Bouzidi-FL two-pass — implementation verdict

Date     : 2026-05-22
Branch   : dev-viscoelastic
Mission  : M34-execute (Bouzidi-FL two-pass split fix, additive only)
Status   : **GREEN — implementation complete, smoke clean, no regression.**

---

## Source-of-truth for the fix

`bench/viscoelastic_audit/M30_PHASE2B_AUDIT_VERDICT.md` §"Proposed minimal fix"
— Claude + Codex CONCORDANT-HIGH on root cause B (single-pass lag-1 on x_ff)
and on the two-pass split as the minimal correct intervention.

## Files modified (additive only)

| File | Net additions | Role |
|---|---|---|
| `src/kernels/dsl/bricks.jl`           | +140 LOC | new `ApplyBouzidiFLPostCollideTwoPass` brick (reads lag-0 `f_out` at both `x_f` and `x_ff`; reads lag-0 `ρ_out` for `rho_w`). Existing `_bouzidi_fl_post_value` helper reused unchanged. |
| `src/kernels/li_bb_2d_v2.jl`          | +47 LOC  | new `_TRT_LIBB_V2_GUO_FIELD_BOUZIDI_FL_TWOPASS_PASS2_SPEC` (single-brick spec); new dispatch `_fused_trt_libb_v2_guo_field_step!(::Val{:bouzidi_fl_twopass}, …)` that launches pass-1 (`_TRT_LIBB_V2_GUO_FIELD_SPEC`), calls `KernelAbstractions.synchronize(backend)`, then launches pass-2; kwarg whitelist extended to `(:halfwayBB, :bouzidi_fl, :bouzidi_fl_twopass)`. |
| `src/drivers/viscoelastic_logfv_2d.jl`| +3 LOC (whitelist+error message only) | extend `wall_bc` validation tuple to include `:bouzidi_fl_twopass`. Default remains `:halfwayBB`. |
| `test/test_bouzidi_fl_twopass_smoke.jl` | +153 LOC (NEW) | tiny cylinder R=8 in a 60×40 closed bounce-back box; primes ρ via no-op step then runs 100 steps for both `:bouzidi_fl_twopass` and `:halfwayBB`; asserts no NaN, density bounded, mass drift of two-pass ≤ 10× halfwayBB baseline, halfwayBB still runs cleanly, unknown wall_bc throws ArgumentError. |
| `test/runtests.jl`                    | +1 LOC | include the new smoke. |

Existing `:halfwayBB` and single-pass `:bouzidi_fl` code paths: **bit-exact unchanged** (no `-` lines in those regions).

## Architectural notes

- Pass-1 = existing `_TRT_LIBB_V2_GUO_FIELD_SPEC` (`PullHalfwayBB → SolidInert → ApplyLiBBPrePhase → Moments → CollideTRTDirectGuoField → WriteMoments`). Writes `f_out[*, *, *]` and `ρ_out[*, *]` everywhere.
- Synchronisation: `KernelAbstractions.synchronize(backend)` after pass-1 — project idiom (cf. `src/kernels/ghost_fluid_2d.jl:109`, `src/kernels/viscoelastic_3d.jl:91/146/160`, `src/kernels/logconformation_lbm_2d.jl:337/358/396/432`).
- Pass-2 = single-brick spec containing only `ApplyBouzidiFLPostCollideTwoPass`. The brick `required_args` set excludes `:f_in` — the kernel signature for pass-2 is `(f_out, ρ_out, is_solid, q_wall, uw_link_x, uw_link_y, Nx, Ny)` in canonical sort order, so no lag-1 read is even *expressible* in pass-2 emit code. (Architectural guarantee.)
- The pass-2 brick does NOT re-emit `Moments`/`WriteMoments` — moments are already correct from pass-1's WriteMoments, and the cut-link rewrites preserve mass at the algebraic level (only the qbar pop is rewritten per link, with the same magnitude as the lag-0 reference).

## Smoke test result (local CPU F64, Julia 1.12.5)

```
Test Summary:               | Pass  Total  Time
Bouzidi-FL two-pass — smoke |   10     10  2.1s
```

Key numbers (60×40 box, R=8, 100 steps, u0=(0.02, 0.01)):
- `drift_two_pass  = 8.23e-05`
- `drift_halfwayBB = 9.07e-05`  (baseline)
- `ρmin = 0.9900`, `ρmax = 1.0094` (well within ±10% bound)
- two-pass mass drift is **9% lower** than halfwayBB on the closed bounce-back box — consistent with the theoretical expectation that closing the lag-1 defect improves conservation, though the absolute gap is small at this small-N / stationary-cylinder setting (the Phase 2b audit predicts the real gain manifests at large N + transient stagnation pressure peaks, hence the cylinder NaN at step ~40k).

## Pkg.test status

Pre-existing baseline failure on `dev-viscoelastic`: `test_poiseuille.jl:5` raises `BoundsError: attempt to access 4×32×9 Array{Float64, 3} at index [1, 0, 3]` inside `src/kernels/stream_periodic_2d.jl:17` (`cpu_stream_periodic_x_wall_y_2d_kernel!`). Reproduced verbatim with M34 changes stashed (`tmp/m34_pkg_test_baseline.log`) → 58 passed, 2 errored, identical signature.

`Pkg.test()` therefore halts on `test_poiseuille.jl` before reaching the new smoke. The new smoke passes cleanly when included directly (`julia --project=. -e 'include("test/test_bouzidi_fl_twopass_smoke.jl")'`).

Verdict: **no M34-induced regression**. The Poiseuille 2D body-force `BoundsError` is unrelated to the M30/M32/M34 Bouzidi-FL pipeline (different kernel, different driver, untouched file `src/kernels/stream_periodic_2d.jl`).

## Aqua submission

**Deferred to next mission.** The brief explicitly stated Aqua submission is OUT OF SCOPE for M34. The implementation is now ready for a M35-style follow-up: Wi×R sweep on Aqua A100 F64 with `wall_bc=:bouzidi_fl_twopass`, expected to (a) eliminate the cylinder R=30 Wi=1 NaN at step ~40k and (b) push K/rT front-pole ratio from 0.59 → > 0.85 per the M30 Phase 2c prediction.

## Memory candidates

1. **`feedback_codex_planning_loop_recovery`** — When a Codex Engineer mission spins on planning without writing implementation code (e.g. M34-pre claude-subagent recovery), the Department should execute INLINE rather than respawn Codex. The boundary signal is "PLAN.md was written but no source file was edited within 15 min" → switch to claude-subagent execution. Extends `[[feedback_monitor_antipattern]]`.

2. **`project_kraken_dsl_two_pass_idiom`** — Two-pass kernel launch idiom in the DSL: pass-1 = standard SPEC writes `f_out` + `ρ_out` everywhere; `KernelAbstractions.synchronize(backend)`; pass-2 = single-brick SPEC over a SUBSET of args (drop `:f_in` from `required_args` to architecturally forbid lag-1 reads). Canonical project examples: `src/kernels/viscoelastic_3d.jl`, `src/kernels/logconformation_lbm_2d.jl`. M34 (`_fused_trt_libb_v2_guo_field_step!(::Val{:bouzidi_fl_twopass})`) is the first DSL-built case.

3. **`feedback_test_invariant_match_geometry`** — Smoke-test mass-conservation invariants MUST match the BC topology of the test setup. A free-running cylinder without inlet/outlet BC does NOT conserve mass on the fluid sub-domain (lateral edges leak via halfwayBB at domain edges). A closed bounce-back box DOES conserve mass to ~1e-4 (SolidInert is a weak source/sink, NOT exact). Asserting machine-precision conservation on an open kernel fails for non-bug physical reasons. Correct pattern: compare relative drift between two BC variants run on the same setup.

---

## Verification commands

```bash
cd /Users/guillaume/Documents/Recherche/Kraken.jl-viscoelastic
grep -q 'ApplyBouzidiFLPostCollideTwoPass' src/kernels/dsl/bricks.jl                          # PASS
grep -q '_TRT_LIBB_V2_GUO_FIELD_BOUZIDI_FL_TWOPASS_PASS2_SPEC' src/kernels/li_bb_2d_v2.jl     # PASS
grep -q ':bouzidi_fl_twopass' src/kernels/li_bb_2d_v2.jl                                      # PASS
test -f test/test_bouzidi_fl_twopass_smoke.jl                                                  # PASS
julia --project=. -e 'include("test/test_bouzidi_fl_twopass_smoke.jl")'                       # 10/10 PASS
```
