# M47 — Patch-test empirical verdict
Date: 2026-05-26
Mission: M47-PT-RUN
Engine: Codex

## TL;DR
H1 is REFUTED by these two local CPU F64 patch tests. Claude's q_w arc test produced Bouzidi > halfway, but below its confirmation threshold; Codex's real one-step Bouzidi-FL wall residual test showed no Bouzidi amplification relative to the halfwayBB baseline.

## PT_bouzidi_fl_polymer_qw_arc.jl (Claude design)
- Run command + wall time: `/Users/guillaume/.julia/juliaup/julia-1.12.5+0.aarch64.apple.darwin14/Julia-1.12.app/Contents/Resources/julia/bin/julia --project=. bench/viscoelastic_validation/patch_tests/PT_bouzidi_fl_polymer_qw_arc.jl`; 2.38 s. The `julia` launcher itself failed before execution because it could not create its juliaup lockfile in this sandbox.
- Final metric values: `max_trace_C_A=14.919908037350394`, `max_trace_C_B=23.41516573750231`, `delta_BA=11.96350616684127`; final baseline-subtracted residuals: halfway `0.19977001205528033`, Bouzidi arc `5.006368995102058`.
- Pass criterion: `delta_BA > 10 * maximum(abs(trace_C_A_wall - 2))` at step 1000.
- Result: FAIL
- Interpretation: The q_w-modulated arc does increase trace_C, but the effect is not large enough to satisfy the designed H1 confirmation threshold. This is a weak refutation because the arc input is a prescribed q_w velocity perturbation, not a full Bouzidi-FL population reconstruction.

## PT_M47_bouzidi_logfv_wall_residual.jl (Codex design)
- Run command + wall time: `/Users/guillaume/.julia/juliaup/julia-1.12.5+0.aarch64.apple.darwin14/Julia-1.12.app/Contents/Resources/julia/bin/julia --project=. bench/viscoelastic_validation/patch_tests/PT_M47_bouzidi_logfv_wall_residual.jl`; 3.58 s.
- Final metric values: halfway wall max trace_C `7.483984498089594`, Bouzidi wall max trace_C `7.154757320809638`; halfway wall-minus-bulk excess `-230.70717374103833`, Bouzidi wall-minus-bulk excess `-231.0364009183183`, delta `-0.32922717727996087`, ratio `-1.0014270348508951`.
- Pass criterion: Bouzidi residual minus halfway residual `> 1e-4` and Bouzidi/halfway residual ratio `> 10`.
- Result: FAIL
- Interpretation: The real CPU F64 wall-BC path did not produce a Bouzidi-FL wall-band polymer residual above the halfwayBB q=0.5 baseline. Bulk stretch dominated this tiny analytic velocity field, but the like-for-like wall residual still moved in the opposite direction from H1.

## Concordance analysis
- Do both PTs point to the same root cause? No. Neither test confirms the proposed Bouzidi-FL q_w wall-gradient amplification as the root cause.
- Quantitative agreement: Both tests are non-confirming. Claude's synthetic arc shows a moderate Bouzidi increase, while Codex's real wall-BC path shows Bouzidi wall trace_C slightly below halfway.
- Any contradictions? The only tension is that the synthetic q_w arc is directionally Bouzidi > halfway, but the real one-step LBM Bouzidi-FL path is not. That weakens the specific "Bouzidi-FL twopass overwrites feed q_w-modulated ∇u" mechanism.

## Recommendation to Boss
- Top hypothesis: refuted by these local patch tests.
- Next mission proposed: instrument the beta=1 Bouzidi cylinder run at the first time trace_C separates and log per-cell `ux, uy, dudx, dudy, dvdx, dvdy, q_wall, trace_C` maxima with code-path provenance.
- Concerns / caveats: The launcher `julia` command was sandbox-blocked, so the installed Julia binary was invoked directly. Claude's PT is a simplified straightened-arc constitutive canary, not a full Bouzidi population canary. Codex's PT is a one-step local discriminator, not a long coupled cylinder run.
