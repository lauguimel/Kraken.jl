# Next session prompt — Kraken cylinder Cd, M29 cluster post-mortem

Copy-paste below to start a fresh session.

---

Continue work on branch `dev-viscoelastic` of Kraken.jl
(worktree `~/Documents/Recherche/Kraken.jl-viscoelastic`).

Resume via orchestrator. Open `~/.claude/skills/orchestrator/SKILL.md` first.
The Boss role continues — Departments + Engineers will absorb the detail;
Boss stays strategic.

## State at handoff (2026-05-19 evening)

### What landed (5 commits pushed this session)

```
42d2177a feat(viscoelastic): M29b MUSCL-superbee + initial src/fvfd tracking
94f4b82d feat(viscoelastic): M29 tau-field comparison locates gap to Rusanov upwind
2945b198 feat(viscoelastic): cylinder Cd M28 cluster synthesis + Liu/rheoTool cross-validation
d708da57 feat(viscoelastic): cylinder Cd Phase 0+0b verdicts + M26 bug localisation
e602726f fix(viscoelastic): world-age trap in CUDA detection causes silent CPU fallback
```

### Three-way reference Cd, β=0.59 Re=1 R=30 Wi=1.0 (production target)

| source | Cd | comment |
|---|---|---|
| Liu CNEBB | **130.36** | Liu Table 3 (columns descending Wi 1.0/0.5/0.1 — verified) |
| rheoTool | **120.40** | `bench/rheotool/cylinder_wi1.0/`, cubista HRS, converged |
| Kraken Rusanov M29b | **111.55** | committed `42d2177a`, stable to 200k steps |
| Kraken muscl_superbee M29c-v2 | **115 (transient)** | uncommitted, NaN at step 92,200 (elastic runaway) |

### M28 cluster verdict (committed `2945b198`)

8 sub-missions ratched OUT 7 hypotheses :
- BSD architecture (M28b) — not the cause
- Time integration (M28c) — 100k = converged to 1M (Δ machine ε)
- Wake truncation (M28f) — matched domain L_up=20 L_down=60 gives same Cd ±0.38
- Mesh refinement (M28e) — Cd plateaus at 111.4 for R ∈ {20,30,40}
- Liu 151.31 citation — column mis-read, true value 130.36
- M26 H1 (drag formula) and H3 (circle quadrature) — refuted by empirical

Locus pinned (M29 commit `94f4b82d`) : **first-order Rusanov upwind on
log-conformation Ψ advection smears the polymer stress peak** at the
leeward shoulder (Kraken τ_xx peak 75.3 vs rheoTool 135.5 = 44 % under).

### M29 thread current state (open)

| sub-mission | status | result |
|---|---|---|
| M29 tau-field comparison | DONE (committed) | gap locus = Rusanov on Ψ |
| M29b MUSCL-superbee interior | DONE (committed `42d2177a`) | Cd 111.55 → 116.47 at Wi=1, closes 56 % of gap |
| M29c boundary 1-sided | FAIL (working tree, uncommitted) | CD2 fallback was anti-TVD, Cd = −1571 (sign flip) |
| M29c-postmortem-math | DONE (uncommitted verdict) | confirms CD2 anti-TVD, proposes 1-line fix |
| M29c-postmortem-empirical (1D bench) | DONE (uncommitted verdict) | confirms fix on 1D scalar advection |
| M29c-v2 (1-line fix applied) | FAIL different mode (working tree) | NaN at step 92,200 on `rho` LBM density |
| M29c-v2-postmortem-locate | DONE (uncommitted verdict) | **plateau Cd=115 stable 5k-80k**, NaN at 92k = elastic stiffness runaway, NOT the boundary fix |
| M29c-v2-postmortem-diff | DONE (uncommitted verdict) | predicts S1-extended (BC helpers unguarded `phi[solid]=0`) — partially wrong vs locate |

**Critical**: the M29c-v2-postmortem-locate Department found the M29c-v2
fix actually **works** at Wi=1 R=30 for ~80,000 steps (Cd=115-116 plateau,
exactly what we wanted). The NaN at step 92,200 is a **SEPARATE,
independent failure mode**: late-stage elastic stiffness runaway at
high Wi (max_speed peaks at 63·u_mean → LBM density crashes upstream).
M29b doesn't NaN because Rusanov dissipation suppresses the buildup.

## Working tree state at handoff (uncommitted)

```
M src/fvfd/operators_2d.jl   <-- M29c-asis + M29c-v2 1-line fix (CD2 → upwind)
?? bench/viscoelastic_audit/M29C_POSTMORTEM_MATH_VERDICT.md
?? bench/viscoelastic_audit/M29C_POSTMORTEM_EMPIRICAL_VERDICT.md
?? bench/viscoelastic_audit/M29C_V2_LOCATE_VERDICT.md
?? bench/viscoelastic_audit/M29C_V2_DIFF_VERDICT.md
?? bench/scratch/m29c_postmortem*/, m29c_v2_locate_metal/
?? bench/viscoelastic_logfv/M29C_BOUNDARY_VERDICT.md
?? .engineer_brief_M29c*.md
?? .engineer_logs/M29c_*.log
?? docs/git-lessons-fvfd-session.md  <-- pedagogical doc for user
```

## Key open decision (the next Boss makes the call)

Pick one of:

**A. Commit M29c-v2 as opt-in + open M29d** (recommended path)
- Commit M29c-v2 patch as `:muscl_superbee` with `:rusanov` default unchanged
- Document the elastic-runaway caveat ("stable to ~80k steps at Wi=1, beyond requires production hardening")
- Open M29d for elastic stiffness mitigation : polymer-stress diffusion,
  tighter polymer subcycling, BSD=0.5 instead of 1.0, force clipping
- The +4 Cd improvement (M29b 116.47 → M29c-v2 115 transient at 30k-80k) is real and useful for Wi < 1 production
- Risk : M29c-v2 may be unsafe at very long integration even at Wi<1; needs more testing first

**B. Rollback M29c entirely, ship M29b only**
- `git checkout src/fvfd/operators_2d.jl` discards M29c work
- Keep M29b's clean 56 % gap closure as the production answer
- Document M29c as "attempted, NaN'd at 92k due to elastic runaway, deferred to M29d"

**C. Investigate M29d mitigations FIRST**
- Spawn Codex with one mitigation (e.g. polymer-stress diffusion with a
  small diffusion coefficient ~ 1e-4 · u·R)
- If it stabilises M29c-v2, ship the bundle ; if not, fall back to (B)

**Memory candidate** (Boss-level): 1D microbench cannot diagnose
late-stage elastic stiffness — that needs ≥10⁴ steps production
simulation. The empirical-1D postmortem said "fix works" with
high confidence ; production reveals an orthogonal failure mode.
Always run a finite-Wi production smoke at ≥100k steps BEFORE
declaring a Ψ-advection upgrade safe.

## Reference docs (next Boss reads first)

1. `.orchestrator/mandate.md` §5 (M28/M29/M29b entries are up to date;
   M29c entries are NOT yet written to mandate — the next Boss writes them)
2. `.orchestrator/memory/boss.md` 2026-05-19 entries (M28 closure +
   M29 closure + M29b PARTIAL)
3. `.orchestrator/memory/engineer.md` 2026-05-19 entries (Rusanov defect,
   MUSCL boundary fall-back load-bearing, `KRAKEN_ADVECTION_SCHEME` env)
4. `bench/viscoelastic_audit/M29C_V2_LOCATE_VERDICT.md` — the empirical
   postmortem that revealed the elastic runaway mechanism (uncommitted)
5. `bench/viscoelastic_audit/M29C_V2_DIFF_VERDICT.md` — structural diff
   M29c-v2 vs M29b (uncommitted; its prediction was partially wrong but
   the structural analysis remains correct for understanding the kernel)
6. `bench/viscoelastic_logfv/M29B_HRS_VERDICT.md` — M29b verdict (committed)
7. `bench/viscoelastic_audit/CYL_TAU_COMPARE_M29_VERDICT.md` — gap locus audit (committed)

## Memory updates pending (Boss writes after deciding A/B/C)

- `mandate.md`: M29c entry (FAIL, working tree), M29c-v2 entry (PARTIAL,
  see locate verdict for caveat), M29d entry (PLANNED).
- `boss.md`: the meta-lesson about 1D microbench limitations + the
  decision A/B/C taken.
- `department.md`: the rsync src/fvfd/ subdir trap from M29b ;
  the early-exit pattern (M28b, M29c-boundary, M28-rheotool-sweep all
  exited prematurely thinking they could wait for a callback).
- `engineer.md`: the elastic runaway signature (max_speed → 63·u_mean,
  rho loses positivity upstream south wall) ; the BC helper unguarded
  read of `phi[is_solid]=0`.

## Mandate state pointer

The Mandate is in good shape through M28/M29/M29b. The M29c thread
(c-asis, postmortems, c-v2) is documented only in scratch markdowns ;
it's the next Boss's job to consolidate.

## Don'ts for the next Boss (per orchestrator step-back rules)

- Don't spawn another postmortem on M29c without a clear new question.
  Two adversarial postmortems already produced contradictory predictions
  on this thread — adding a third risks chasing diminishing returns.
- Don't try to fix elastic runaway in M29c — that's M29d.
- Don't read the raw Department transcripts (`.jsonl`) — use the
  verdict markdowns.

End of prompt.
