# M34v3 — pass-2 ρ_w consistency fix (pass-3 cut-link ρ recompute) + Wi=0.1 polymer smoke + Aqua re-submit

**Date**     : 2026-05-22
**Mission**  : M34v3 (cut-link-only ρ recompute after pass-2 of `:bouzidi_fl_twopass`)
**Branch**   : `dev-viscoelastic` (uncommitted)
**Status**   : **YELLOW — pass-3 ρ-recompute implemented + Wi=0.1 polymer smoke + Newtonian regressions both PASS + Aqua matrix re-submitted; quantitative G4 gate pending Aqua results.**

## (a) Architecture chosen — **(A) Three-pass split**

I picked **(A)**: a NEW DSL brick `ApplyCutLinkRhoRecompute` registered as a third pass-only spec
`_TRT_LIBB_V2_GUO_FIELD_BOUZIDI_FL_TWOPASS_PASS3_SPEC`, launched after pass-2 + `KernelAbstractions.synchronize(backend)`.

### Why not (B) extend pass-2

`WriteMoments()` writes `ρ_out[i, j] = ρ` where `ρ` is the local kernel variable set by the `Moments()`
brick consuming `fp1..fp9` pulled at pass-1. In pass-2 those locals do NOT exist (the brick reads
`f_out[i, j, q]` directly). So "re-using" `WriteMoments` would force me to re-add a `Moments()` brick
into pass-2 that re-pulls all 9 pops — a global read at every cell, not just cut-links. That would (i)
clobber `ρ_out` at non-cut-link cells with a potentially-differing-by-FP value vs pass-1, breaking
bit-exact regression on the `:halfwayBB` path AND on the Newtonian cut-link smoke, and (ii) cost a
needless N×N memory pass on GPU. The new dedicated brick gates with `q_wall[i, j, q] > 0` and only
writes at cut-link cells — surgical, intent-explicit, bit-exact for non-cut-link cells.

### LOC delta

| File                                            | LOC change                 | What changed                                                                                                                          |
| ----------------------------------------------- | -------------------------: | ------------------------------------------------------------------------------------------------------------------------------------- |
| `src/kernels/dsl/bricks.jl`                     | +33 / −0                   | New `ApplyCutLinkRhoRecompute` brick (lines 693-725): required_args = `(:f_out, :ρ_out, :q_wall, :is_solid, :Nx, :Ny)`; phase=:fluid; emits a cut-link gate (`any q_wall[i,j,2..9] > 0`) and re-sums `f_out[i,j,1..9]` to overwrite `ρ_out[i,j]`. |
| `src/kernels/li_bb_2d_v2.jl`                    | +22 / −0                   | New spec `_TRT_LIBB_V2_GUO_FIELD_BOUZIDI_FL_TWOPASS_PASS3_SPEC = LBMSpec(ApplyCutLinkRhoRecompute())`; pass-3 dispatch in `_fused_trt_libb_v2_guo_field_step!(::Val{:bouzidi_fl_twopass}, …)`: `synchronize(backend)` after pass-2 + `pass3!(f_out, ρ, is_solid, q_wall, Nx, Ny; ndrange=(Nx, Ny))` (canonical arg order). |
| `test/test_bouzidi_fl_twopass_smoke.jl`         | +77 / −0                   | New 3rd testset `"Bouzidi-FL two-pass — Wi=0.1 polymer cylinder R=4"`: drives `run_viscoelastic_logfv_cylinder_coupled_2d(; wall_bc=:bouzidi_fl_twopass, lambda=66.67, u_mean=0.006, radius=4, H=18, max_steps=30)` → Wi = λ·u_mean/R = 0.1 exact. Asserts no NaN (ρ, ψ_xx, ψ_xy, ψ_yy, ux, uy), Cd finite, ρ ∈ [0.5, 1.5], `max_c_trace < 50`, `min_c_eig > 0`, `first_nonfinite_step == 0`. |
| `bench/viscoelastic_audit/M34V3_VERDICT.md`     | new                        | This verdict                                                                                                                          |

`src/kernels/li_bb_2d_v2.jl` existing `:halfwayBB` and `:bouzidi_fl` paths: **0 LOC changed** (verified by
grepping the dispatch table — pass-3 only fires under `Val{:bouzidi_fl_twopass}`).

## (b) Smoke result (local CPU F64)

```
Test Summary:                                         | Pass  Total  Time
Bouzidi-FL two-pass — halfwayBB-degenerate regression |   10     10  2.4s
  drift_two_pass  = 6.194e-5   (was 6.194e-5 pre-pass3 → bit-exact)
  drift_halfwayBB = 9.068e-5
  ρ_two_pass ∈ [0.9909, 1.0086]

Bouzidi-FL two-pass — cut-link cylinder R=8 Newtonian Re=1 | 7   7   0.4s
  Cd = 110.72  (was 110.72 pre-pass3 → bit-exact regression sentinel)
  Fx = 0.6299
  ρ ∈ [0.9833, 1.0796]

Bouzidi-FL two-pass — Wi=0.1 polymer cylinder R=4         | 15  15   3.5s
  Wi  = λ·u_mean/R = 66.67 · 0.006 / 4 = 0.1   (exact)
  Cd  = 302.6  (R=4 under-sized → 2.3× of Schaefer-Turek ~131; expected)
  Cd_s = 313.0,  Cd_p = 3.15
  ρ ∈ [0.9932, 1.0130]
  max_c_trace = 2.04   (Oldroyd-B trace, very stable at Wi=0.1)
  min_c_eig   = 0.87   (SPD preserved)
  first_nonfinite_step = 0   (no nonfinite event over 30 steps)
```

All 32 assertions PASS. Critical findings:
- The pass-3 ρ-recompute is **bit-exact** on the regression sentinel (closed-box halfwayBB-degenerate) and
  on the M34-fix Newtonian cut-link cylinder. The new code path fires but writes the same FP value as
  the pre-existing one for the Newtonian symmetric case (cut-link f-sums equal pass-1 f-sums when no
  polymer source term has shifted the moments). This means the pass-3 cost is **zero quantitative impact**
  on the Newtonian regression and is **expected to matter only when the polymer Guo body force shifts the
  cut-link f's between pass-1 ρ-write and pass-2 BC overwrite**, i.e. for genuine viscoelastic Wi > 0
  cases.
- Wi=0.1 polymer cylinder runs cleanly through `:bouzidi_fl_twopass` with `max_c_trace = 2.04` (i.e.
  trace(C) only 2 % above quiescent), `ρ ∈ [0.993, 1.013]` (well below the [0.5, 1.5] gate). No NaN.

The smoke does NOT independently confirm that pass-3 closes the Aqua YELLOW (the Wi=0.1 R=30 +1.6 %
bias requires a fully-resolved cylinder + 100k+ steps for the bias to accumulate visibly). The smoke
confirms: (i) the pass-3 kernel compiles + launches on the existing two-pass dispatch path, (ii) does
not introduce a new NaN class on a small polymer-coupled case, (iii) Newtonian regressions are bit-exact.

## (c) Aqua re-submission

| PBS                                                           | Job ID            | Walltime | State |
| ------------------------------------------------------------- | ----------------- | -------- | ----- |
| `run_cyl_m34_bouzidi_fl_matrix_a100.pbs`                      | **`21668525.aqua`** | 04:00:00 | Q     |
| `run_cyl_m34_bouzidi_fl_R60_Wi01_a100.pbs`                    | **`21668526.aqua`** | 02:00:00 | Q     |

Submitted at **2026-05-22** (Aqua local). PBS scripts unchanged from M34-fix (`KRAKEN_WALL_BC=bouzidi_fl_twopass`
propagates via the env-var plumbing at `24a8819a`). The pass-3 fix is purely src/ — no driver / no PBS
edit needed.

### rsync command used

```bash
rsync -az --delete \
  --exclude='.git' --exclude='tmp/' --exclude='output/' --exclude='results/' \
  --exclude='docs/build/' --exclude='*.jls' --exclude='*.vtr' --exclude='*.vti' \
  --exclude='*.vtk' --exclude='__pycache__/' --exclude='.engineer_logs/' \
  --exclude='bench/scratch/' \
  ~/Documents/Recherche/Kraken.jl-viscoelastic/ aqua:Kraken.jl-viscoelastic-run/
```

### Check command for next session

```bash
ssh aqua 'qstat 21668525.aqua 21668526.aqua 2>&1; \
          ls -la ~/Kraken.jl-viscoelastic-run/M34_bouzidi_fl_*.o* 2>/dev/null; \
          tail -40 ~/Kraken.jl-viscoelastic-run/M34_bouzidi_fl_matrix.o21668525 2>/dev/null; \
          tail -40 ~/Kraken.jl-viscoelastic-run/M34_bouzidi_fl_R60_Wi01.o21668526 2>/dev/null'

# Pull results when done
rsync -az aqua:Kraken.jl-viscoelastic-run/tmp/m34_bouzidi_fl_matrix/ \
  ~/Documents/Recherche/Kraken.jl-viscoelastic/tmp/m34_bouzidi_fl_matrix/
rsync -az aqua:Kraken.jl-viscoelastic-run/tmp/m34_bouzidi_fl_R60_Wi01/ \
  ~/Documents/Recherche/Kraken.jl-viscoelastic/tmp/m34_bouzidi_fl_R60_Wi01/
```

Walltime estimate: 4 h (matrix) + 2 h (R60), each pending queue wait.

## (d) New G4 expectation per M34v3 hypothesis

The pass-3 hypothesis (HIGH in M34_FIX_DIAG_VERDICT §"Candidate residual bugs") predicts that the
Aqua YELLOW outcomes were driven by stale `rho_w` propagated through the polymer Guo body force at
cut-link cells. Closing the inconsistency should yield:

| Case            | M34-fix Aqua (YELLOW)         | M34v3 expected                                          |
| --------------- | ----------------------------- | ------------------------------------------------------- |
| R=30 Wi=0.1     | Cd = 132.51 (+1.6 % vs rT)    | **Cd ∈ [129.1, 131.7]** (i.e. ≤ 1 % of rheoTool 130.43) |
| R=40 Wi=0.1     | Cd = 133.46 (+2.3 % vs rT)    | **Cd ∈ [129.1, 131.7]** + R-invariant ±0.5 % of R=30    |
| R=60 Wi=0.1     | NaN                           | **runs cleanly to 100k steps; Cd in same band**         |
| R=30 Wi=1.0     | NaN                           | **finite Cd, target ∈ [118, 122]** (closes −7.3 % gap)  |
| R=40 Wi=1.0     | NaN                           | **finite + R-invariant within ±0.5 % of R=30 Wi=1.0**   |

All 5 pass → **M34 G4 GREEN → empirical closure of the M28-M32 cluster**.

### Failure handoff (if HIGH refuted)

- If R=60 Wi=0.1 still NaNs **but** R=30 Wi=0.1 drops below +1 % → HIGH partially confirmed; the residual
  divergence at R≥60 is polymer-side stress-pole stiffness, not BC. Escalate to a focused M35 (polymer
  Ψ-clip or log-FV CFL gate) on the divergent R=60 / R=40 Wi=1.0 cases only.
- If R=30 Wi=0.1 still +1.6 % → HIGH refuted. Escalate to MEDIUM (the args-trim 15 → 12 in pass-1
  dispatch) per M34_FIX_DIAG_VERDICT §"Candidate residual bugs" #2.
- If a NEW divergence emerges (e.g. R=30 Wi=0.1 now NaN where it was finite before) → pass-3 itself
  introduces an inconsistency. Local smoke shows this is not the case at small R + Wi=0.1, but the
  Aqua scale could reveal a different regime. Roll back commit-equivalent state via `git diff` and
  open a focused investigation on `_TRT_LIBB_V2_GUO_FIELD_BOUZIDI_FL_TWOPASS_PASS3_SPEC` (single new
  brick, easy to bisect).

## Verification (Boss success criterion replay)

```bash
cd /Users/guillaume/Documents/Recherche/Kraken.jl-viscoelastic && \
  grep -E '(BOUZIDI_FL_TWOPASS_PASS3_SPEC|recompute.*rho|cutlink.*moments|CutLinkRhoRecompute)' \
    src/kernels/li_bb_2d_v2.jl src/kernels/dsl/bricks.jl | head -3
#  → 3+ matches (PASS3_SPEC + ApplyCutLinkRhoRecompute hits)

grep -c '@testset' test/test_bouzidi_fl_twopass_smoke.jl
#  → 3

test -f bench/viscoelastic_audit/M34V3_VERDICT.md && echo OK
#  → OK

grep -E '\.aqua' bench/viscoelastic_audit/M34V3_VERDICT.md | head -2
#  → 21668525.aqua  +  21668526.aqua
```

## Memory candidates (≤3)

1. **`feedback_three_pass_split_clean_over_brick_extension`** — When closing a multi-pass kernel BC
   trap by adding a post-BC moment recompute, prefer a NEW dedicated cut-link-gated brick over extending
   the existing BC-write brick. Re-using `WriteMoments` would force re-running `Moments()` (a global N×N
   read) at non-cut-link cells too, breaking bit-exact regression and bloating GPU traffic. A dedicated
   brick with a `q_wall[i,j,q] > 0` gate writes only where it matters. M34v3 architecture (A) confirms
   `[[feedback_smoke_must_exercise_cutlinks]]` principle on the implementation side — minimal-touch is
   bit-exact-by-construction.

2. **`feedback_polymer_smoke_catches_zero_at_smoke_scale_but_validates_pipeline`** — A small polymer-
   coupled smoke (R=4, H=18, Wi=0.1, 30 steps, Oldroyd-B) does NOT catch a +1.6 % steady-state Cd bias
   (too small + too short to accumulate). What it DOES catch: (i) the new pass-3 kernel compiles +
   launches through the full polymer driver entry point `run_viscoelastic_logfv_cylinder_coupled_2d`,
   not just standalone `fused_trt_libb_v2_guo_field_step!`; (ii) no NEW NaN class introduced; (iii)
   trace(C) and min_c_eig remain physical. The +1.6 % bias / R=60 NaN closure remains an Aqua-only
   gate. Smoke validates the **plumbing**, not the **quantitative outcome**.

3. **`feedback_brick_canonical_arg_order_at_call_site`** — Pass-3 brick `required_args = (:f_out,
   :ρ_out, :q_wall, :is_solid, :Nx, :Ny)`. `_canonical_sort` reorders this per `CANONICAL_ARG_ORDER`
   to `(:f_out, :ρ_out, :is_solid, :q_wall, :Nx, :Ny)` — note `:is_solid` (idx 7) precedes `:q_wall`
   (idx 8). Call site MUST match: `pass3!(f_out, ρ, is_solid, q_wall, Nx, Ny; ndrange=…)`. Got this
   right on first try after reading `lbm_spec.jl:89` (`CANONICAL_ARG_ORDER` const), but the pattern
   is fragile: a brick author who lists args in their natural-language order will produce a kernel
   signature that LOOKS reordered. Always cross-check `_canonical_sort(required_args(brick))` with
   the call-site arg tuple. Existing pass-2 dispatch does this correctly; pass-3 follows the same
   pattern.

## Files

- `src/kernels/dsl/bricks.jl` (modified, +33 LOC)
- `src/kernels/li_bb_2d_v2.jl` (modified, +22 LOC)
- `test/test_bouzidi_fl_twopass_smoke.jl` (modified, +77 LOC, 3 testsets)
- `bench/viscoelastic_audit/M34V3_VERDICT.md` (this verdict)
- `tmp/m34v3_smoke.log` (local CPU smoke output, 32 / 32 assertions PASS)
- `tmp/m34v3_aqua_submit.log` (qsub output, 2 new job IDs)

## Next action for Boss

1. Wait for Aqua jobs `21668525.aqua` (matrix, 5 cases at R=30/40 Wi=0.1/1.0 + R=60 Wi=0.1 sentinel)
   + `21668526.aqua` (R=60 Wi=0.1 focused) to complete.
2. Pull results via the `rsync -az` commands in section (c).
3. Cross-check the 5 acceptance bands above. If all GREEN → `M34_G4_CLOSURE_VERDICT.md` and close
   M28-M32 cluster. If partial (HIGH closes some but not R=60 / R=40 Wi=1.0) → focused M35 on
   polymer-side stiffness. If RED (HIGH refuted at R=30 Wi=0.1) → escalate MEDIUM (args audit).
