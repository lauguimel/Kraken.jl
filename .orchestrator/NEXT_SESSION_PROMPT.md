# Next session prompt — Kraken.jl viscoelastic R=60 Wi=1 time/BC investigation

## Resumption check

```bash
cd ~/Documents/Recherche/Kraken.jl-viscoelastic
git log --oneline -8   # session arc
ls bench/viscoelastic_audit/M44_GUO_FIX_VERDICT.md  # M44 closed cluster M28-M42
ls bench/viscoelastic_audit/M45_RESIDUAL_VERDICT.md # residual audit
ls bench/viscoelastic_audit/M46_NEWT_AND_TCONV_VERDICT.md # latest finding
ls tmp/m44_postfix_sweep/21827394.aqua/             # M44 48-case .jls
ls tmp/m46b_tconv/21862685.aqua/                    # R=60 @ 100k/200k/400k
```

Recent commits :
- `ce5fa838` docs(viscoelastic): M46 Newt sweep + M46-B time-convergence
- `83cb3efe` docs(viscoelastic): M45 post-M44 residual audit (B + C)
- `4df9d431` docs(viscoelastic): M44 post-fix sweep verdict (4 R × 4 Wi × 3 β)
- `9fd92ab0` fix(viscoelastic): port slbm-paper 5ec27044 Guo half-step double-count

---

## Context (compact)

**M44 closed cluster M28-M42** (8 days, 10+ RED missions on advection-limiter)
by porting `slbm-paper` commit `5ec27044` (Guo half-step `+F/2`
double-count in `logfv_compute_macroscopic_forced_field_2d_kernel!`).

**Validation**:
- Cylinder Wi=1 R=30 β=0.59 Aqua F64 100k: **Cd 111.09 → 118.10**
  (78% closure of 9.54% gap vs rT 120.38). **Temporally rock-solid**
  (M46-B R=30@400k = 118.099 → ±0.003 vs 100k).
- 48-case sweep (R∈{30,40,50,60} × Wi∈{0.1,0.3,0.5,1.0} × β∈{0.59,0.8,0.9})
  0/48 NaN. All R=60 cases NaN-free (pre-fix were NaN per M42 G5 v3).
- M5b test rewritten as pair test asserting N·F (not N·F + F/2),
  PASS at 1e-12.

**OPEN ISSUE — found 2026-05-25 evening (M46 + M46-B)**:

At R=60 Wi=1 β=0.59, Cd is **NOT temporally converged at 100k**.
Time-convergence probe revealed:
- R=60 @ 100k: Cd=113.234 (= M44 sweep value)
- R=60 @ 200k: Cd=112.130 (−1.10)
- R=60 @ 400k: Cd=109.424 (**−3.81, drift ACCELERATING**)

The M44 sweep "Cd-decrease with R at Wi=1" (118.10 R=30 → 113.23 R=60)
is **NOT a mesh effect** but **incomplete temporal convergence scaling
with flow-through**. At R=30, 100k = 0.56 flow-through (converged).
At R=60, 100k = 0.28 flow-through (wake still developing); 400k = 1.11
flow-through (still drifting).

**M46 Newtonian sweep (β=1.0)**: Cd INCREASES monotonically with R
(halfwayBB +0.60, Bouzidi-FL +2.80 across R=30→60). Opposite direction
to Wi=1 → eliminates β-class hypotheses (lattice-distance, TRT, domain).

**Newt Bouzidi-FL trace_C_max anomaly**: 209 (R=30) → **1.4e7 (R=60)**
despite β=1.0 (polymer dormant LBM-side, no F_poly). Polymer C-tensor
unstable under Bouzidi-FL Newt R=60. Cd unaffected but suggests a
**separate Bouzidi-FL polymer-chain bug** worth its own audit.

---

## User-hinted hypotheses (load-bearing)

1. **Bouzidi BCs may be buggy** (user's hypothesis 2026-05-25). The
   trace_C explosion under Bouzidi Newt is concrete evidence of a
   Bouzidi-side polymer-chain bug. The M44 sweep + M46-B used
   halfwayBB (NOT Bouzidi), so the R=60 drift is not directly caused
   by Bouzidi, but a similar bug class may exist for halfwayBB + polymer
   at long times.

2. **"Guo en newtonien" residual** (user 2026-05-25): K Newt halfwayBB
   R=30 = 132.076 vs rT 132.37 = −0.22%. Small residual in Newtonian
   path post-M44. Codex M44 G2/G4/G5/G6/G7 inventory items remain
   unfixed (Codex said not on cylinder path; worth re-verifying after
   M46 anomalies).

3. **Multiple confounded variables**: resolution (R) + Wi + polymer
   coupling + BC type. Disentangle systematically with one variable
   at a time, max_steps long enough at each R to ensure flow-through
   convergence.

---

## Starting mission for next session

**Mission 0 (P0, USER-FLAGGED 2026-05-25)** — Bouzidi-FL polymer-chain
bug audit:

User observation: trace_C_max under Bouzidi-FL Newt scales
catastrophically with R: **209 (R=30) → 321 (R=60) for halfwayBB,
vs 1938 → 1.4e7 for Bouzidi-FL**. Bouzidi shows **7200× growth** vs
halfwayBB's normal 1.5×. With β=1.0 (no LBM coupling) the explosion
is harmless (Cd unaffected). But with polymer (β<1), the inflated
C tensor → inflated τ_p → injected as F_poly into LBM → spurious
drag + likely NaN. **The R=60 NaN in M30 R-sweep + M32 Phase 4
at Wi=1 may be entirely Bouzidi-driven, not polymer-physics.**

This is the same bug PATTERN as M44 Guo (kernel writes wrong moment
that propagates through downstream chain). M44 fixed the Guo half-step
+F/2; this would be the analogous Bouzidi half-step or read-write
ordering bug.

Audit plan:
- Static kraken-trace (Tool 1 which + Cthulhu) on the Bouzidi-FL
  kernels at R=60 Newt: identify exactly which moment field is
  exploding (rho? momentum? gradient?)
- Compare halfwayBB vs Bouzidi-FL writes to f_post at the cylinder
  cells, identify the diff
- Possibly: an `slbm-paper` sister-branch fix exists (per
  [[feedback_port_sister_branch_fixes]] — audit `git log --all |
  grep -i bouzidi` before designing fix from scratch)

**Mission 1 (priority after M0)** — characterize R=60 Wi=1 halfwayBB
time behavior to determine if a steady state exists:

- Re-run R=60 Wi=1 β=0.59 muscl_superbee halfwayBB qwall at
  `max_steps ∈ {800k, 1.6M}` (2 cases) on Aqua F64. Aim for
  ~3.5-4.5 flow-throughs. Compare Cd progression.
- If Cd plateaus by 800k → real steady-state is below 109 Cd; the
  M44 sweep R-trend reflects under-sampling but the residual gap
  vs rT is REAL and large.
- If Cd keeps drifting linearly → numerical drift, not steady state.
  Probably mass/momentum conservation issue. Need diagnostic.
- If Cd oscillates → vortex shedding (Hopf bifurcation crossed at
  R=60 Wi=1 with L=15R). Real physics, need to time-average over
  shedding period.

PBS template: `bench/viscoelastic_logfv/run_cyl_m46b_tconv_a100.pbs`
(adapt MAX_STEPS_BASE values + output dir).

**Mission 2 (in parallel)** — Cd time-series instrumentation:

- Modify `src/drivers/viscoelastic_logfv_2d.jl` to log Cd every
  N=1000 steps to a CSV. Minimal patch, ~10 LOC.
- Re-run R=60 Wi=1 at max_steps=800k with logging enabled.
- Inspect Cd(t) trajectory: monotone drift, oscillating, or
  converging?

**Mission 3 (audit)** — Bouzidi-FL polymer-chain bug:

- Why does Bouzidi-FL Newt trace_C_max go 209 → 1.4e7 between R=30
  and R=60? With β=1.0 (ν_p=0), polymer should be quiescent.
- Possible mechanism: Bouzidi-FL writes some intermediate buffer
  that the polymer chain reads with wrong scaling at high R.
- Static audit (codex-style kraken-trace) of `_apply_bouzidi_fl_*`
  vs polymer ψ-advection input fields.

**Mission 4 (Newt residual)** — Guo fix completeness:

- K Newt halfwayBB R=30 = 132.076 vs rT 132.37 = −0.22%. 0.3 Cd
  unaccounted for. Could be:
  - rT under-converged at R=30
  - Residual G2/G4/G5/G6/G7 readout bug Codex flagged
  - Bouzidi-vs-halfwayBB BC discrepancy of ~0.56 Cd at R=30 (in M41
    data)
- Generate rT mesh refinement to confirm rT R=30 value at higher
  resolution. If rT converges to ~132.5 → Kraken 132.08 is the
  residual, ~0.4 Cd, possibly from Guo G2 fix completeness or other.

---

## Working notes for next session

- **Don't trust M44 sweep R=40/50/60 verdict** — those were 100k
  runs, possibly all under-converged at the faster mesh.
- **M44 R=30 verdict IS solid** — temporally validated by M46-B.
- **rT mesh refinement was on hold pending temporal-convergence
  resolution** — still on hold. No point matching rT to Kraken target
  that's moving.
- **Per-θ M45 decomposition at R=60** was on non-converged snapshot
  → conclusions about "Cd_solv shoulder + Cd_pres wake residual" are
  unsupported.
- **User constraint** (per `[[feedback_department_bail_out_pattern]]`):
  Boss invokes Codex directly via `run-engineer.sh` for any
  Codex-wait mission. Do NOT use Department subagents for spawn+wait.
- **User constraint** (CLAUDE.md HPC policy): confirm before any
  rsync to Aqua, qsub, or destructive command.
- **Compaction status**: boss.md was at 1896 lines pre-2026-05-24,
  compacted to ~407 + new 2026-05-24/25 sections. Check size at
  session start; compact again if past ~600 lines.

---

## Key files

- `.orchestrator/memory/boss.md` — Boss memory (this session's
  2026-05-24/25 entries are the active context)
- `.orchestrator/memory/boss_archive_M1_M34_pre_20260523.md` —
  pre-M44 history archive
- `.orchestrator/mandate.md` — project mandate
- `bench/viscoelastic_audit/M44_GUO_FIX_VERDICT.md` — root-cause fix
- `bench/viscoelastic_audit/M44_GUO_AUDIT_CODEX.md` — Codex G1-G7 audit
- `bench/viscoelastic_audit/M44_SWEEP_VERDICT.md` — 48-case sweep
- `bench/viscoelastic_audit/M45_RESIDUAL_VERDICT.md` — residual
  (now partially superseded by M46-B finding)
- `bench/viscoelastic_audit/M45_RESIDUAL_AUDIT_CODEX.md` — Codex α/β/γ
- `bench/viscoelastic_audit/M46_NEWT_AND_TCONV_VERDICT.md` — latest
  load-bearing finding
- `src/kernels/logconformation_fv_2d.jl:1047-1050` — the M44 fix
- `src/kernels/macroscopic.jl:71-75` — base 2D G1 fix
- `test/test_viscoelastic_logfv_patch_ladder.jl:1416-1455` — M5b
  pair test

---

## Active waiters / processes

None at session end (all Aqua jobs completed and rsync'd).

## Memory entries (auto-memory `~/.claude/projects/.../memory/`)

- `project_m44_guo_halfstep_fix.md` — root cause + commit
- `project_m44_m45_sweep_residual.md` — sweep + per-θ + Codex audit
- `feedback_port_sister_branch_fixes.md` — process lesson for
  pro-active sister-branch audits

Next session can extend these as needed.

End of next session prompt.
