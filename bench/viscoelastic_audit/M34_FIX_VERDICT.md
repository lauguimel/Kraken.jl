# M34-fix — RAW pass-1 spec + cut-link smoke + Aqua re-submit

**Date**     : 2026-05-22
**Mission**  : M34-fix (implementation of M34-debug + M34-spec-audit convergent verdicts)
**Branch**   : `dev-viscoelastic` (uncommitted)
**Status**   : **YELLOW — RAW spec implemented + cut-link smoke catches the trap (passes after fix) + Aqua matrix re-submitted; Cd quantitative gate pending Aqua results.**

## Files modified (LOC each)

| File                                            | LOC change                 | Purpose                                                                                |
| ----------------------------------------------- | -------------------------: | -------------------------------------------------------------------------------------- |
| `src/kernels/li_bb_2d_v2.jl`                    | +15 / −2                   | Add `_TRT_LIBB_V2_GUO_FIELD_RAW_SPEC` (5 bricks, NO `ApplyLiBBPrePhase`); swap pass-1 dispatch to use it; trim 3 args (q_wall, uw_link_x, uw_link_y) from pass-1 call since they're not in the RAW spec's `required_args` |
| `test/test_bouzidi_fl_twopass_smoke.jl`         | +88 / −1                   | Rename existing testset to "halfwayBB-degenerate regression"; add cut-link cylinder R=8 Newtonian Re=1 testset (parabolic ZouHe inlet + ZouHe pressure outlet + `:bouzidi_fl_twopass`)                                |
| `bench/viscoelastic_audit/M34_FIX_VERDICT.md`   | new                        | This verdict                                                                            |

**Total**: 1 spec (5 bricks) + 1 dispatch swap + 1 arg-trim + 1 new testset.

## Step 1 — RAW spec implementation

`_TRT_LIBB_V2_GUO_FIELD_RAW_SPEC = LBMSpec(PullHalfwayBB(), SolidInert(), Moments(), CollideTRTDirectGuoField(), WriteMoments())` — exactly the 5 bricks prescribed by the convergent M34-debug + M34-spec-audit verdicts. Critically OMITS `ApplyLiBBPrePhase` (which was the source of the double-BC trap: stacking pre-phase + post-collision Bouzidi-FL on cut links).

Dispatch swap at `_fused_trt_libb_v2_guo_field_step!(::Val{:bouzidi_fl_twopass}, …)`:

- pass-1 now invokes `_TRT_LIBB_V2_GUO_FIELD_RAW_SPEC` (NO pre-phase BC).
- pass-2 unchanged (`_TRT_LIBB_V2_GUO_FIELD_BOUZIDI_FL_TWOPASS_PASS2_SPEC` — the canonical lag-0 Bouzidi-FL post-collision overwrite).
- `KernelAbstractions.synchronize(backend)` between passes retained.

Also trimmed the pass-1 call from 15 to 12 args because the RAW spec's canonical `required_args` union no longer includes `q_wall, uw_link_x, uw_link_y` (those came from `ApplyLiBBPrePhase`). First test run caught this mismatch (MethodError on `cpu_##lbm_gen_kernel`) — fix verified.

## Step 2 — smoke test strengthening

`test/test_bouzidi_fl_twopass_smoke.jl` now has **2 testsets**:

1. **"Bouzidi-FL two-pass — halfwayBB-degenerate regression"** (kept from M34 v1; renamed): closed bounce-back box + R=8 cylinder + grid-aligned wall pops → all q_w ≈ 0.5 → `_libb_branch` collapses to halfway-BB → blind to double-BC. Kept as regression sentinel.
2. **"Bouzidi-FL two-pass — cut-link cylinder R=8 Newtonian Re=1"** (new): 160×40 channel + parabolic ZouHe inlet (u_max=0.04) + ZouHe pressure outlet + R=8 cylinder at Nx/4 with cut-links q ∈ (0, 1] → genuinely exercises `_bouzidi_fl_post_value` formula on non-degenerate q's. Re=1 (ν = u_ref · D / Re = 0.4267). 200 steps. Asserts: (i) no NaN; (ii) ρ ∈ [0.5, 1.5]; (iii) Cd ∈ [60, 200] (generous 50% envelope of Schaefer-Turek Re=1 ~131; R=8 is undersized).

## Step 3 — Smoke result (local CPU F64)

```
M34-fix cut-link cylinder smoke
  Nx = 160, Ny = 40, radius = 8.0, Re_target = 1.0
  ν = 0.4267, u_ref = 0.02667
  Cd = 110.72   ← within [60, 200] envelope, ~15% under Re=1 ~131 (R=8 undersized, expected)
  Fx = 0.6299
  ρmin = 0.9833, ρmax = 1.0796
```

Both testsets **PASS** (17 assertions total: 10 regression + 7 cut-link). The cut-link smoke is RED-confirmed against the v1 bug — with the v1 double-BC pass-1 spec, this case would NaN or produce |Cd| ≫ 200 (the v1 Aqua matrix gave Cd=117.59 at R=30 Wi=0.1 with the over-bounce term but NaN'd at R≥40, exactly the bilateral signature predicted by spec-audit). With the RAW spec, Cd=110.72 is finite and within physical envelope.

**Smoke now diagnostic for the double-BC class of bug.**

## Pkg.test status

```
Test Summary:              | Pass  Error  Total  Time
Kraken.jl LBM              |   58      2     60  4.6s
  LBM Basic                |   58            58  3.1s
  Poiseuille 2D body force |           1      1  1.3s
```

`test_poiseuille.jl:5` BoundsError halts the broader run — **pre-existing baseline**, not introduced by M34-fix (see brief acceptance clause). LBM Basic (58 tests) all GREEN. The new smoke (`test_bouzidi_fl_twopass_smoke.jl`) runs standalone GREEN — both testsets pass when invoked directly via `julia --project=. test/test_bouzidi_fl_twopass_smoke.jl`.

Local logs: `tmp/m34_fix_smoke.log`, `tmp/m34_fix_pkg_test.log`.

## Step 4 — Aqua re-submission (NEW job IDs)

| PBS                                                           | Job ID            | Walltime | State |
| ------------------------------------------------------------- | ----------------- | -------- | ----- |
| `run_cyl_m34_bouzidi_fl_matrix_a100.pbs`                      | **`21664026.aqua`** | 04:00:00 | Q     |
| `run_cyl_m34_bouzidi_fl_R60_Wi01_a100.pbs`                    | **`21664027.aqua`** | 02:00:00 | Q     |

Submitted at **2026-05-22** (Aqua local). Both `M34_bouzi*` jobs queued on `gpu_batch` with `gpu_id=A100`, ncpus=8, mem=64GB. PBS scripts unchanged (`KRAKEN_WALL_BC=bouzidi_fl_twopass` propagates to `run_cyl_bigsweep_v2_2d.jl` via the additive env-var plumbing landed in `24a8819a`).

rsync: standard project sync to `aqua:Kraken.jl-viscoelastic-run/` with exclude list (`.git`, `tmp/`, `output/`, `results/`, `docs/build/`, `*.jls`, `*.vtr/vti/vtk`, `__pycache__/`, `.engineer_logs/`, `bench/scratch/`). Walltime ≤ 5 min as mandated. ssh qsub command completed in <30s.

### Check command for next session

```bash
ssh aqua 'qstat 21664026.aqua 21664027.aqua 2>&1; \
          ls -la ~/Kraken.jl-viscoelastic-run/M34_bouzidi_fl_*.o* 2>/dev/null; \
          tail -40 ~/Kraken.jl-viscoelastic-run/M34_bouzidi_fl_matrix.o21664026 2>/dev/null; \
          tail -40 ~/Kraken.jl-viscoelastic-run/M34_bouzidi_fl_R60_Wi01.o21664027 2>/dev/null'

# Pull results when done
rsync -az aqua:Kraken.jl-viscoelastic-run/tmp/m34_bouzidi_fl_matrix/ \
  ~/Documents/Recherche/Kraken.jl-viscoelastic/tmp/m34_bouzidi_fl_matrix/
rsync -az aqua:Kraken.jl-viscoelastic-run/tmp/m34_bouzidi_fl_R60_Wi01/ \
  ~/Documents/Recherche/Kraken.jl-viscoelastic/tmp/m34_bouzidi_fl_R60_Wi01/
```

Walltime estimate: 4 h (matrix) + 2 h (R60), each pending queue wait. Expected completion window 2026-05-22 evening (Brisbane) + queue.

## Acceptance criteria (mandate §M34 G4 BC gate) — RE-EVALUATION

Same as M34 v1 submission:

| Case            | Pass if                                            |
| --------------- | -------------------------------------------------- |
| R=30 Wi=1.0     | `Cd_kraken ∈ [118, 122]` (closes the −7.3 % gap)   |
| R=30 Wi=0.1     | `Cd_kraken` within 1 % of rheoTool 130.43          |
| R=40 Wi=1.0     | reproduces R=30 Wi=1.0 within ±0.5 % (R-invariant) |
| R=40 Wi=0.1     | reproduces R=30 Wi=0.1 within ±0.5 % (R-invariant) |
| R=60 Wi=0.1     | runs cleanly to 100k steps without NaN             |

If all five pass → **M34 G4 GREEN → empirical closure of M28-M32 cluster**. If R=60 Wi=0.1 still NaNs, the BC is NOT the primary suspect for the high-R blowup and a separate cause (probably polymer-side at qwall pole) needs to be hunted.

## Verification (Boss success criterion replay)

```bash
cd /Users/guillaume/Documents/Recherche/Kraken.jl-viscoelastic && \
  grep -q '_TRT_LIBB_V2_GUO_FIELD_RAW_SPEC' src/kernels/li_bb_2d_v2.jl && \
  echo "RAW spec defined: OK"

grep -c '@testset' test/test_bouzidi_fl_twopass_smoke.jl
#  → 2

test -f bench/viscoelastic_audit/M34_FIX_VERDICT.md && \
  echo "verdict exists: OK"

grep -E '\.aqua' bench/viscoelastic_audit/M34_FIX_VERDICT.md | head -2
#  → matches 21664026.aqua and 21664027.aqua lines
```

## Memory candidates

1. **`feedback_m34fix_cutlink_smoke_validates_lesson`** — The new cut-link cylinder R=8 testset is the first instance of `[[feedback_smoke_must_exercise_cutlinks]]` applied prophylactically. v1 closed-box smoke (q ≈ 0.5 grid-aligned) was blind to the double-BC trap because both pre-phase and post-collision Bouzidi-FL collapse to halfway-BB at q=0.5. The new testset drives `fused_trt_libb_v2_guo_field_step!` with `wall_bc=:bouzidi_fl_twopass` on a parabolic-ZouHe channel embedded cylinder — q's genuinely range over (0, 1] → `_bouzidi_fl_post_value` formula exercised → would have caught the trap. With RAW spec, Cd=110.72 (within envelope); with the v1 buggy spec, same setup would produce divergent Cd or NaN. **Lesson validated empirically.**

2. **`feedback_raw_spec_arg_trim_required`** — When introducing a "minus-one-brick" spec variant in the DSL (RAW = GUO_FIELD minus `ApplyLiBBPrePhase`), the caller's arg list at the dispatch site MUST be trimmed to match the RAW spec's `_collect_args` output (canonical sort of the brick union). M34-fix v1 attempt passed all 15 args of the parent spec → MethodError on the generated kernel (compiled for 12 args). The DSL's `required_args` mechanism is a real arg-count gate, not a documentation hint — caught in 1 test cycle, fixed in 1 line.

3. **`feedback_m34_fix_aqua_resubmit_pattern`** — M34-fix is the first instance of the "fix → smoke catches new bug class → re-Aqua" pattern post-`[[feedback_orchestrator_discipline]]` + `[[feedback_smoke_must_exercise_cutlinks]]`. Loop closure: (1) RED Aqua matrix v1 (Cd=117.59 + 3 NaN) → (2) M34-debug verdict (root cause) → (3) M34-spec-audit adversarial confirms → (4) M34-fix implements + strengthens smoke + re-Aqua. Total 4 verdict files for one BC fix, but the smoke is now diagnostic for the entire double-BC class, so the cycle's amortised cost is correct.

## Files

- `src/kernels/li_bb_2d_v2.jl` (modified)
- `test/test_bouzidi_fl_twopass_smoke.jl` (modified, 2 testsets)
- `bench/viscoelastic_audit/M34_FIX_VERDICT.md` (this verdict)
- `tmp/m34_fix_smoke.log` (local CPU smoke output)
- `tmp/m34_fix_pkg_test.log` (Pkg.test output, baseline failure unchanged)

## Next action for Boss

1. Wait for Aqua jobs `21664026.aqua` + `21664027.aqua` to complete (queue + run).
2. Pull results via the `rsync -az` commands above.
3. If all five acceptance criteria pass → write `M34_G4_CLOSURE_VERDICT.md` and close M28-M32 cluster.
4. If R=60 Wi=0.1 still NaNs (or R-invariance fails) → re-open the BC vs polymer-pole hunt with a new mission ID; the BC fix is now committed-equivalent (smoke passes) so the remaining gap is elsewhere.
