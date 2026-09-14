# M55 status audit — Boss step-back

Written after user pushback : "ça fait 15 fixes que tu me dis que ça va
marcher". Honest separation of CONFIRMED facts vs SPECULATIVE claims
that have been propelled into the chain of fix attempts.

## A — CONFIRMED (empirical, reproducible this session)

| # | Fact | Source |
|---|------|--------|
| A1 | M44 Guo half-step fix (commit 9fd92ab0): R=30 anchor Cd 111.09 → 118.10 (78% closure of original gap vs rT 120.38) | M44_GUO_FIX_VERDICT, Aqua F64 100k |
| A2 | M51 axis-aligned wall helper (commit bccef5d7): canari M49 4.97e-14, 953/953 PASS | M51_VERDICT |
| A3 | M48 baseline (post-M51, M55 reverted) R=10/5k coupled = Cd 111.23 FINITE | This session |
| A4 | M48 baseline R=30/10k coupled = Cd 106.41 FINITE | This session |
| A5 | Old M55 Taylor: L1 canari 8.3e-4 GREEN, **L3 frozen-u FAIL (19× amplification)**, M48 R=30/180k NaN at step ~178k | M55_AUDIT_codex, A.3 partial |
| A6 | M55b bilinear: L1 6.1e-3 GREEN, L3 frozen-u PASS (1.09×), L3-recompute PASS (drift=0), L3-Guo PASS (u pinned), L5b qwall Poiseuille PASS (0.14% err) | M55b_FIX_VERDICT, M57_VERDICT |
| A7 | M55b R=10/5k coupled = Cd 110.42 FINITE | This session |
| A8 | **M55b R=30/10k coupled = NaN at step ~2145**, first NaN cell (439,32) θ=-113°, neighbor (439,33) τ=0.244, |u| in NaN-cell = 0.09 = 18× U_mean | This session, M57 forensic |

## B — INTERPRETATIONS treated as fact (should be marked SPECULATIVE)

| # | Claim | Status |
|---|-------|--------|
| B1 | "Cluster A (FVFD wall-gradient stencil) is the load-bearing residual after M44" — M54 audit conclusion | SPECULATIVE: based on circumstantial evidence (S06 + S10 + audit reasoning). Never falsified or confirmed by a stable M55 actually closing the U-shape. |
| B2 | "M55 (q_w-aware quadratic) would close the U-shape if stable" | SPECULATIVE: never empirically demonstrated. We have no stable M55 with M48 R-sweep results to compare to baseline. |
| B3 | "Stencil discontinuity at q_w_min + Guo feedback is the cause of M55b NaN" — Codex M57 localisation | SPECULATIVE: consistent with forensic evidence but no fix has actually closed the NaN. Could also be: stencil-everywhere amplification, polymer chain stiffness, MUSCL antidiffusion, F_poly fill near wall. |
| B4 | "Smoothing q_w_min transition will fix M55b coupled NaN" | SPECULATIVE+ : no test. Even if B3 is right, the fix might shift NaN, not eliminate it. |
| B5 | M44 Guo fix was the correct fix at the relevant scope | TRUSTED but NOT INDEPENDENTLY VERIFIED on this branch. The Aqua F64 100k closure is on M48 fixture pre-M48-investigation. We haven't re-run M48 R=30 to 100k on the CURRENT worktree to confirm baseline still gives 118.10. |

## C — ASSUMPTIONS never tested this session

| # | Assumption | Why it matters |
|---|------------|----------------|
| C1 | Baseline M48 R=30 stable at FULL 180k steps on current worktree | Audit memory (M48_HW_MESHCONV) was on an older worktree. There's been uncommitted drift (`src/drivers/viscoelastic_3d.jl`, `viscoelastic_spec.jl`, `kernels/li_bb_2d_v2.jl`). If baseline silently NaN's at 180k now, M55 is NOT the cause of M48 R=30 instability. |
| C2 | Baseline M48 R=30 gives Cd=117.62 plateau (per memory) | Same: never re-verified on current worktree. Could be different now. |
| C3 | M55b R=30 NaN at step 2145 is the SYMPTOM, not the CAUSE; the cause is a gradient bug at the failure cell | The forensic shows huge |u|=0.09 and τ=0.244. These could be **CONSEQUENCES** of bug accumulated over 2000 steps, not the bug itself. The bug might have been small at step 100 and only become catastrophic at step 2145. |
| C4 | M55b CPU PASS L3 means M55b Metal F32 actually fires the same code path with the same numerics in M48 | M48 uses Metal F32 with full coupling, L3 used same backend but simpler path. Provenance never end-to-end audited. |
| C5 | The "M55 better than FO" assumption (i.e. M55 is the CORRECT answer and FO is wrong) | Never empirically demonstrated. We assumed M55 is correct because the math is consistent with the M55 derivation. But the M55-vs-FO disagreement at the failure site (0.149 = 15% of local gradient) could mean M55 is WRONG, not that FO is wrong. The Newtonian-frozen field is curved (not quadratic), and M55 doing 15% different than FO is suspicious without an analytic reference. |

## D — Chain-of-fix-proposals retrospective

| Iteration | Claim | Actual outcome |
|-----------|-------|----------------|
| M55 impl (Codex) | "Bilinear u₂ too imprecise on canari, use Taylor" | Canari 8.3e-4 GREEN but **goalpost-hit**, M48 R=30 NaN. |
| M55b fix (Boss + Codex) | "Replace Taylor with bilinear + q_w_min=0.1" | L3 GREEN but **M48 R=30 NaN at step 2145**. |
| M55c smoothing (Boss proposal) | "Smooth q_w_min blend will fix it" | NOT RUN. We have ZERO evidence for this claim. |

**Three "this will fix it" iterations, two empirically falsified, one untested but proposed with same confidence.**

## E — What Boss-level move would actually be informative

In descending order of information value:

### E1: Re-run baseline M48 R=30 to full 180k Metal F32 on CURRENT worktree

Tests C1+C2. ~25 min Metal. If baseline plateau is NOT ~117.62 (per
memory), or if it NaN's anywhere before 180k, then a silent drift bug
exists INDEPENDENT of M55. This recasts the entire problem.

### E2: Forensic M48 R=30 baseline (M55 reverted) at step 2145

Tests whether the (439, 32) θ=-113° location is a NaN-only-with-M55b
artifact or a stability hot-spot that exists in baseline too. Compare
|u| and τ magnitudes at (439,32) between baseline-step-2145 and
M55b-step-2144. If baseline shows already-elevated |u| or τ there,
the failure cell is a numerical hot spot independent of M55b.

### E3: Pull out the "Guo bug" hypothesis the user mentioned

User: "on avait un bug sur Guo, on a peut-être un autre bug autre part".
The M44 Guo fix was a slbm-paper port. We TRUST it but haven't audited
its application this branch. Specifically: are there OTHER Guo-side
sites in this branch (viscoelastic_logfv vs slbm) that DIDN'T get the
half-step fix?

### E4: Reality-check that M55 is the right direction at all

The cluster A hypothesis (M54 audit) said M55 should close the U-shape.
But the only "evidence" for cluster A is circumstantial. If baseline
M48 R=30 is actually fine at 180k (E1), then the U-shape is REAL not
a bug, and M55 is chasing a non-bug while introducing real instability.

### E5: STOP and accept M48 as known limit, write paper with current state

If E1-E4 don't change the picture, accept that the M48 U-shape is an
open research question, not solvable in this session. Pivot to paper
with: M44 fix + M51 stencil + V&V suite + canaries + L3+L5b
infrastructure. M55b stays stashed for future.

## F — Concrete next action

Do **E1 first**: re-run baseline M48 R=30 to 180k on current worktree.
This is the cheapest test (~25 min, no code) that can RESHAPE the
problem (if baseline is actually broken, M55 chase is moot).

ONLY after E1 confirms baseline is stable, do we know there's a real
M55-specific bug to chase. If E1 fails, we have a different problem
to debug, possibly upstream of M48.

After E1, the choice (E2/E3/E4/E5) depends on what E1 shows.
