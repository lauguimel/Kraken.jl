# Next session prompt — Kraken.jl viscoelastic post-M53 step-back

## Resumption check

```bash
cd ~/Documents/Recherche/Kraken.jl-viscoelastic
git status --short              # uncommitted M51+M53 infra
git log --oneline -8            # session arc
ls bench/viscoelastic_audit/M{49,50,51,52a,52b,53a,53b,53c,53d}*.md \
  bench/viscoelastic_audit/M48_POSTFIX_RESULT.md
ls bench/viscoelastic_validation/patch_tests/PT_*.jl
ls bench/viscoelastic_validation/discriminators/M48_halfway_meshconv.jl
ls src/fvfd/halfway_wall_gradient_correction_2d.jl
julia --project=. test/test_fvfd_operators_2d.jl 2>&1 | tail -3        # expect 953/953
julia --project=. test/test_viscoelastic_logfv_patch_ladder.jl 2>&1 | tail -3   # expect 18213/18213
julia --project=. bench/viscoelastic_validation/patch_tests/PT_halfway_wall_stencil.jl 2>&1 | tail -3   # M49 canari
```

Recent commits (NOTHING from this session is committed yet):
- `546808b2` docs(viscoelastic): next-session prompt with M0 Bouzidi polymer-chain bug audit
- `ce5fa838` docs(viscoelastic): M46 Newt sweep + M46-B time-convergence probe
- `83cb3efe` docs(viscoelastic): M45 post-M44 residual audit (B + C)
- `9fd92ab0` fix(viscoelastic): port slbm-paper 5ec27044 Guo half-step double-count

---

## Session arc 2026-05-26 (compact)

User pivoted away from M0 Bouzidi audit (parked) to focus on **halfwayBB
Wi=1 wrong mesh convergence** (M48). After 3 failed fix attempts, user
called STEP BACK. M48 U-shape remains unfixed but the terrain is now
clean and mapped.

**What was learned**:
- Cylinder Wi=1 halfwayBB shows U-shape mesh convergence: R=10 → 114.48,
  R=30 → 117.62 (best, gap −2.4% vs rT 120.40), R=50 → 114.26. ALL reach
  plateau within 1 flow-through Metal F32 → M46-B "R=60 drift" is
  continuation of dégradation, not under-sampling.
- Bug A (M49 axis-aligned wall stencil): `_fvfd_solid_bc_derivative_*_2d`
  returns derivative at first-fluid CENTER, not at wall. Confirmed
  by M49 canari at 0.2s. Half-cell geometric offset, factor-2-class.
- Bug B (M53a cylinder cut-cell): same stencil non-q_w-aware at
  embedded cells. Mean abs_err 0.071 on canari (max 0.141, corr 0.78
  vs q_w).
- M52a audit (surprise): `wall_bc=:halfwayBB` does NOT force q_w=0.5 —
  it dispatches `ApplyLiBBPrePhase` which IS q_w-aware (LBM-side OK).
- Bug C (cylinder U-shape mechanism): NOT fully localized. Removing
  helper applications doesn't change cylinder Cd much. Toggling
  `embedded_gradient=true` causes NaN divergence at R≥30 (first-order
  embedded helper too noisy under coupled cylinder).
- M51 over-application broke 12 tests (M5d/M5e/M7d/M8h) because polymer
  chain at cell-center consumed wall-position gradient. Cleaned up.

---

## Current state (clean baseline)

**Tests**: 18213/18213 + 953/953 GREEN. M49 + M53a canaries permanent.

**Infrastructure preserved**:
- `src/fvfd/halfway_wall_gradient_correction_2d.jl` (M51 helper,
  second-order axis-aligned wall formula `(3u₁ − u₂/3 − (8/3)u_wall)/dy`)
- `src/drivers/cavity_driver_2d.jl:221` calls helper in `:quadratic`
  mode (M51b — cavity benefits from fix)
- `FVFDEmbeddedBoundary2D` has bifurcated fields: `wall_distance` /
  `wall_inv_distance` (centroid for volume integration) +
  `wall_inv_distance_to_center` (plane for gradient helper)
- `_fvfd_apply_embedded_wall_gradient_2d` consumes the plane field

**Reverted (not the fix)**:
- M51 helper removed from shared step `_run_viscoelastic_logfv_step_channel_coupled_2d`
  (line 430) → cylinder/square/bfs back to pre-M51 wall-row behavior
- M51 helper removed from frozen_channel (1317), Poiseuille (2389),
  square_periodic (2645), bfs_passive (2906)

**Uncommitted changes** (per `git status` at end of session):
- `src/fvfd/lowering_2d.jl` — bifurcation field added
- `src/fvfd/operators_2d.jl` — helper uses new field
- `src/drivers/cavity_driver_2d.jl` — call into M51 helper
- `src/drivers/viscoelastic_logfv_2d.jl` — M51 application reverted
  (5 sites), 1 step_callback payload extension (M48 instrumentation)
- `src/Kraken.jl` — exports for new helper
- `src/fvfd/halfway_wall_gradient_correction_2d.jl` — NEW
- `test/test_fvfd_operators_2d.jl` — new bifurcation field test added
- `test/test_viscoelastic_logfv_patch_ladder.jl` — M2c fixture re-baselined
  to plane-distance input
- Many bench/, scratch/, .engineer_brief_* artifacts

---

## Starting mission for next session (user choice required)

User stepped back — the decision is theirs. Options on the table (in
descending order of "fix the U-shape" ambition):

### Option A — Build proper second-order cut-cell helper (vrai fix)

Derive a q_w-aware wall-aware quadratic formula at cut-cells (analogue
of M51's axis-aligned `(3u₁ − u₂/3 − (8/3)u_wall)/dy`, but for wall
at variable distance `q_w·dx`). Validate on M53a canary (target mean
abs_err < 1e-3 vs current first-order 0.023). Test M48 cylinder R-sweep.

- ~1 h Codex implementation + audit
- Risk: even second-order may not stabilize coupled cylinder run at R≥30
- Mathematical sketch: fit `u(s) = u_wall + a·s + b·s²` through
  `u(0) = 0`, `u(q_w·dx) = u₁`, `u((q_w+1)·dx) = u₂` (samples relative
  to wall position, not cell center). Derivative `∂u/∂n|wall = a`.

### Option B — Test M48 with Bouzidi-FL BC

Toggle `wall_bc=:bouzidi_fl_twopass` on cylinder M48. Bouzidi-FL is
explicitly q_w-aware at LBM-side. M52a noted FVFD gradient bug subsists
but the BC change alone might shift the cylinder Cd. Discriminates
"is U-shape BC-class or stencil-class".

- ~30 min Metal
- Note: M47 H1 was parked because PT empirics didn't confirm Bouzidi
  q_w-modulation mechanism for trace_C blowup. But the M46 sweep DID
  show Bouzidi Newt trace_C 209 → 1.4e7 between R=30 and R=60 → a
  real Bouzidi-side anomaly persists. Run with caution.

### Option C — Pivot to publication-ready scope (recommended if M48 not blocking paper)

Accept M48 U-shape at R≥40 as a known limit. Write up:
- M44 fix (Guo half-step) closes M28-M42 cluster with 78% closure of
  the original gap at R=30 anchor (118.10 vs rT 120.38).
- V&V suite L1 Poiseuille Wi sweep all PASS (constitutive math validated).
- Cavity refactored to use M51 second-order wall stencil.
- 2 new permanent canaries (M49 + M53a) protect the FVFD stencil from
  future regressions.

Document M48 mesh-convergence anomaly as an open research question
(could be artifact of halfwayBB on a curved wall — rheoTool uses a
different discretization). NOT a blocker for the slbm-paper / cylinder
v0.1 publication.

### Option D — Tactical commit + clear next-session

User reviews the M51 cleanup + M53b/c bifurcation infra changes,
commits them with appropriate message, then chooses A/B/C in a
fresh session.

---

## Working notes for next session

- **Per `[[feedback_small_tests_first]]`**: every fix iteration MUST
  pass the M49 + M53a canaries (<2s each) before any Aqua / Metal
  R-sweep is launched.
- **Per `[[feedback_department_bail_out_pattern]]`**: Boss-direct
  Codex via run-engineer.sh for any spawn-and-wait mission. No
  Department subagents.
- **Per CLAUDE.md HPC policy**: explicit user confirmation before
  any Aqua qsub / rsync. Local Metal F32 (per `[[feedback_gpu_local]]`)
  is the default for development.
- **Compaction status**: `boss.md` was 641 lines at session start,
  this session added a 2026-05-26 block (TODO: write that block when
  resuming — it didn't get written before STEP BACK).
- **One unwritten boss.md entry**: 2026-05-26 session (M48-M53). The
  postmortem memory `project_m51_m53_session_postmortem.md` covers
  it but boss.md timeline section needs the corresponding entry.

---

## Key files

- `.orchestrator/memory/boss.md` — Boss memory (M44-M46 era inside)
- `.orchestrator/memory/department.md`, `engineer.md` — layered patterns
- `~/.claude/projects/.../memory/project_m51_m53_session_postmortem.md` —
  THIS SESSION's full postmortem (load-bearing for next session)
- `~/.claude/projects/.../memory/project_m48_hw_meshconv.md` — M48 finding
- `~/.claude/projects/.../memory/project_m51_wall_grad_fix_partial.md` —
  M51 outcome
- `bench/viscoelastic_audit/M48_POSTFIX_RESULT.md` — M48 R-sweep
  post-M51 result (U-shape still there)
- `bench/viscoelastic_audit/M49_WALL_STENCIL_CANARY.md` — axis-aligned
  canary verdict
- `bench/viscoelastic_audit/M50_STENCIL_CALLER_AUDIT.md` — stencil
  call-site map
- `bench/viscoelastic_audit/M52a_CUTCELL_AUDIT.md` — halfwayBB IS
  q_w-aware via LI-BB, FVFD gradient is NOT (key surprise)
- `bench/viscoelastic_audit/M52b_CYL_ADJ_CANARY.md` — cylinder cut-cell
  canary (mean abs_err 0.071)
- `bench/viscoelastic_audit/M53b_EMBEDDED_HELPER_AUDIT.md` — bug
  localized: wall_distance was centroid not plane
- `bench/viscoelastic_audit/M53c_BIFURCATION_VERDICT.md` — bifurcation
  implemented
- `bench/viscoelastic_audit/M53d_POLYMER_CONSUMER_AUDIT.md` — triage
  of the 12 regressions (2 R + 10 P classification)

---

## Active waiters / processes

None. All background tasks completed.

## Memory entries written this session

- `feedback_small_tests_first.md` — user directive about micro-canaries
- `project_m48_hw_meshconv.md` — U-shape finding
- `project_m51_wall_grad_fix_partial.md` — partial fix outcome
- `project_m51_m53_session_postmortem.md` — full session postmortem

---

End of next session prompt.
