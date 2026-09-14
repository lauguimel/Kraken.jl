# Next session prompt — Kraken.jl viscoelastic M48 post-toggle-flip cleanup

## TL;DR

Session 2026-05-26 (1ff91685) burned ~6h chasing "R=50 NaN" + "+14%
plateau" mysteries. **Root cause: single uncommitted `embedded_gradient`
toggle flip** in the M48 fixture (`true` instead of historical `false`).
All M55 / M55b / M56-M58 work in `bench/viscoelastic_audit/M55_*`,
`M56_*`, `M57_*`, `M58_*` tested the WRONG code path and concluded valid
but irrelevant things about the embedded path.

**Reverting the toggle (eg=false) reproduces historical baseline to
within 0.25-2.1%.** Worktree is currently in a half-cleanup state.

See `[[m48-toggle-flip-postmortem]]` for full story. See updated
`[[code-path-provenance]]` for the new "fixture toggle audit" rule.

## Resumption check

```bash
cd ~/Documents/Recherche/Kraken.jl-viscoelastic
git status --short                  # expect: lots of stash, lots of uncommitted
git log --oneline -5                # 6e72457f9 (C3 archive) → bccef5d7a (C1 M51+M53c)
git stash list                      # expect 3-4 stashes
cat scratch/M48_eg_false_baseline/summary.csv 2>&1 || echo "(re-run if missing)"
```

Worktree state details:
- `src/fvfd/lowering_2d.jl`, `operators_2d.jl` reverted to `546808b21` (pre-C1)
- `src/Kraken.jl` reverted to `546808b21` BUT with manual `include("diagnostics/trace.jl")` re-added
- `src/fvfd/halfway_wall_gradient_correction_2d.jl` (M51 helper file) **deleted** from worktree
- `src/fvfd/FVFD.jl`, `src/drivers/cavity_*.jl`, `test/*.jl` reverted to pre-C1
- `src/drivers/viscoelastic_logfv_2d.jl` reverted to pre-C1 BUT with manual step_callback payload extension added back (4 extra fields: `f_out, q_wall, uwx, uwy`)
- 3 drift files restored from stash (`viscoelastic_3d.jl`, `viscoelastic_spec.jl` FENE-P, `li_bb_2d_v2.jl` @trace_enter)
- **M48 fixture `bench/viscoelastic_validation/discriminators/M48_halfway_meshconv.jl` STILL has `embedded_gradient=true` (the bug)** — never reverted

Stashes:
- `stash@{0}: M55b_active_pre_substeps_sweep` (M55b helper file + src/fvfd modifs)
- `stash@{1}: M55b_bilinear_v1` (early M55b state)
- `stash@{2}: cleanup before fresh session` (on dev/v0.2-architecture, unrelated)

## Verified empirical facts

| R | Cd_final (Metal F32, eg=FALSE, post-revert worktree) | Cd_final historical (audit memory) | Δ |
|---|------------------------------------------------------|-------------------------------------|----|
| 10 | 114.77 | 114.48 | +0.25% |
| 30 | 116.28 | 117.62 | -1.1% |
| 50 | 111.84 | 114.26 | -2.1% |

All FINITE, no NaN at any R, full 300k steps completed at R=50.
Sweep artifact: `scratch/M48_eg_false_baseline/`.

The ~1-2% residual drift candidates:
- F32 vs F64 offset (~0.4% per audit Aqua F64 anchor)
- Metal compiler / Julia version drift since audit
- Other untracked-file modifications we can't reconstruct

## Recommended next-session opening moves

**Step 1: Clean up the worktree, settle on a known state.**

Two options (user to choose):

**A) Un-revert C1, restore fixture toggle, commit fixture fix.**
   - `git checkout HEAD -- src/Kraken.jl src/drivers/cavity_*.jl src/drivers/viscoelastic_logfv_2d.jl src/fvfd/FVFD.jl src/fvfd/lowering_2d.jl src/fvfd/operators_2d.jl test/test_fvfd_operators_2d.jl test/test_viscoelastic_logfv_patch_ladder.jl`
   - `git checkout HEAD -- src/fvfd/halfway_wall_gradient_correction_2d.jl`
   - Edit `bench/viscoelastic_validation/discriminators/M48_halfway_meshconv.jl` to set `embedded_gradient=false` (REMOVE the `# M53e: test post-bifurcation embedded helper` comment too, or replace with `# historical default — see m48-toggle-flip-postmortem`)
   - Commit: `fix(viscoelastic): restore embedded_gradient=false in M48 fixture (was flipped during M53e test, never reverted)`
   - Drop stashes M55b_* (M55b is on wrong path, not useful)

**B) Keep the C1 revert as a real revert commit.**
   - C1 bifurcation is neutral on eg=false path (no functional effect)
   - But also no benefit. Reverting just adds churn.
   - Recommend A unless user has a reason.

**Step 2: Address the REAL M48 mandate — why U-shape on eg=false path?**

The original question (pre any of this session's chasing):

> Cd plateau (path eg=false, halfwayBB, Wi=1, β=0.59, Re=1, BSD=1, L_up=L_down=15R):
> - R=10: 114.48 (-4.9% vs rT 120.40)
> - R=30: 117.62 (-2.3% vs rT) ← best
> - R=50: 114.26 (-5.1% vs rT)
>
> **Non-monotone in R (U-shape). Why?**

This is the same U-shape originally observed. Need TWO competing effects:
- Effect A: increases Cd with R (some physical/numerical resolution gain)
- Effect B: decreases Cd with R (some R-dependent bias)

**Candidate effects** (per original mandate analysis, all to RE-TEST now
that we have a stable baseline):

- *Effect B*: polymer chain stiffness λ = R; substep cap = 64 might cap
  out at higher R. Test: substep sweep R=50 ∈ {32, 64, 128, 256, 512}.
  (Note: we already tested this on eg=TRUE path and it had no effect,
  but eg=FALSE path could behave differently.)
- *Effect B*: lattice channel length 30R in lu → more cells, more
  boundary effects.
- *Effect B*: F32 noise accumulation (more steps at higher R).
- *Effect A*: cylinder curvature resolution (cylinder discretization
  improves with R).
- *Effect A*: FVFD wall stencil error scales O(h²) → /R².

The simplest mandate-aligned first move: **substep sweep R=50** on
eg=false (now that we have stable baseline). If Cd_R=50 climbs with
cap → polymer cap is one of the competing effects.

**Step 3 (research-level, not for this session)**: investigate why
embedded_gradient=true destabilizes at R=50 — could be a real finding
about the embedded path's stability properties, useful for future v0.3
work. NOT a priority; the embedded path is not the production benchmark.

## Critical lessons (already written to memory)

- `[[m48-toggle-flip-postmortem]]`: full story
- `[[code-path-provenance]]` updated rule 5: FIXTURE TOGGLE audit via
  JSONL grep BEFORE any hypothesis

## Audit files to flag as "wrong path" (NOT delete, but mark)

The following verdict files in `bench/viscoelastic_audit/` are
technically correct but on the wrong code path (eg=true), and their
conclusions about "M55 needs fixing" / "stencil discontinuity" /
"cluster A" do NOT apply to the production benchmark (eg=false):

- `M55_AUDIT_codex.md`, `M55_AUDIT_claude.md`, `M55_DERIV_*.md`,
  `M55b_FIX_VERDICT.md`, `M55_IMPL_VERDICT.md`, `M55_STATUS_AUDIT_BOSS.md`
- `M56_VV_LADDER_VERDICT.md`
- `M57_BEEFED_LADDER_VERDICT.md`
- `M58_ANALYTIC_CHAIN_VERDICT.md`

Recommend adding a header line to each: `**NOTE 2026-05-27**: This
audit ran on embedded_gradient=true code path (toggle flip in M48
fixture, since reverted). Conclusions describe valid behavior of the
embedded path but do NOT apply to the historical eg=false production
baseline. See [[m48-toggle-flip-postmortem]].`

(Optional task for next session; not blocking.)

## Active waiters / processes

None. All background tasks completed before session-end.

## Memory entries written this session

- `project_m48_toggle_flip_postmortem.md` (NEW)
- `feedback_code_path_provenance.md` updated (rule 5 added)
- `MEMORY.md` index entry added

## Key files for next session

- `~/.claude/projects/-Users-guillaume-Documents-Recherche-Kraken-jl/memory/project_m48_toggle_flip_postmortem.md` — this session's lesson
- `~/.claude/projects/-Users-guillaume-Documents-Recherche-Kraken-jl/memory/feedback_code_path_provenance.md` — updated rule
- `scratch/M48_eg_false_baseline/summary.csv` — verified baseline reproduction
- `bench/viscoelastic_validation/discriminators/M48_halfway_meshconv.jl` — fixture STILL needs toggle fix (eg=true → eg=false)
- `scratch/M48_R10_30_50_post_revert.jl` — Boss-direct sweep script (eg=false confirmed)

## NOT to do (anti-patterns from this session)

- Do NOT continue M55/M55b/M55c iteration — wrong path, all work irrelevant
- Do NOT trust audit memory plateau values without first verifying with current worktree (~5 min smoke)
- Do NOT skip fixture toggle audit when memory says "X was stable" but reality differs

End of next session prompt.
