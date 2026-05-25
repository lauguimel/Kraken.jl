# Boss memory — Kraken.jl viscoelastic cavity spatial debug

Initialised 2026-05-15. Project-level facts that affect future missions
on the cavity / log-FV / viscoelastic branch.

---

## ARCHIVED — pre-2026-05-23 sessions (M1-M34)

Full narrative trace for 2026-05-15 → 2026-05-22 (M1-M34, ~1678 lines)
moved to `.orchestrator/memory/boss_archive_M1_M34_pre_20260523.md` on
2026-05-24 to keep this index lean. Compact synthesis below; consult
the archive + `mandate.md` + `bench/viscoelastic_audit/*VERDICT*.md`
for forensic detail.

### Cluster M1-M19 — Cavity 5-candidates triage (2026-05-15 → 17)

Original mandate: explain 18-24 % Kraken-vs-rheoTool cavity L2 gap at
N=64 t=8 De=1 β=0.5. Five mandated candidates (Re mismatch, wall
corner artifact, polymer upwind, Guo-vs-FD divergence, kinetic-moment
BSD) + user-added wall-BC alternative — **all six REFUTED**.

- **M1**: Re mismatch flat across u_max ∈ {5e-3, 2e-3, 1e-3}. Refuted.
- **M2**: corner artifact real but smoke-scale (5.6e-5 vs 3.5e-5).
- **M3**: polymer pipeline 4 % on frozen rheoTool U → ratched OUT.
- **M4 / M4b**: Guo-vs-FD 54 % gap is BSD operating AS DESIGNED
  (L2 falls monotonically with `bsd_fraction`). Hypothesis refuted.
- **M6-A/B**: wall-BC stencil mismatch real (12 % wall-row local) but
  does NOT propagate to global profile. Refuted on Aqua N=64 t=8.
- **M5-A/B**: kinetic-moment BSD prototype machine-eq vs FD path
  (5.85e-16) but didn't change physics — same FD limit.

**Diagnostic battery M7-M15** then localised the bug:
- **M7b (SMOKING GUN, 2026-05-16)**: Wi-INDEPENDENT 3.42 % delta on
  matched-viscosity controls (B vs C noise floor 0.014 %). The
  polymer-coupling absorption into `ν_LBM = ν_s + ζ·ν_p` is incomplete.
- **M9 grid conv**: L2 falls monotonically with N (31 → 18 → 13 → 10 %);
  asymptotic-floor extrapolation L2_∞ ≈ 7.4 % centerline u.
- **M10**: bug pinned to **wide-laplacian (2dx) vs narrow-laplacian
  (3pt) stencil mismatch** in BSD truncation.
- **M11 (RED, REVERTED)**: same-stencil fix on the monolithic driver
  produced 64 % (worse than the bug). Root: `D_corrected` captured at
  wrong pipeline step. Reframed as M17, gated on M16.
- **M16 (commit `77956ad8`)**: SPLIT cavity driver 3429 → 2934 LOC
  (wall_correction + cavity_driver modules). All files ≤700 LOC.
  M17 unblocked.
- **M17 cluster (commit `b995e304`)**: closes with reframed verdict —
  cavity bug has TWO independent defects (stencil + corner amplification);
  Option 3 same-stencil fix is wired but the residual decomposition
  (~0.4 % stencil + ~2.4 % corner + ~0.6 % BSD intrinsic) was inferred
  not measured → user pivots to Poiseuille investigation.

**Process lessons** (Boss-level, preserved):
- Cavity Kraken-vs-Kraken noise floor = 0.014 % (matched-viscosity
  Newtonian control). Anything > 0.05 % is signal at this benchmark.
- Cavity coupling bug = TWO independent defects (stencil + corner
  amplification). Single-mechanism explanations all RED.
- LLM-friendly file-size constraint: ≤500 LOC soft, ≤700 LOC hard. M11
  failed exactly because the cavity driver was 3429 LOC monolith.
- Parallel theory Departments paid off: 4 independent Codex+Claude
  derivations on M17 cluster converged on Option 3 within 1 session.

### Cluster M20-M27 — Poiseuille pivot + cylinder Cd convergence (2026-05-18)

User directive 2026-05-18: park M18 (cavity production validation) and
M19 (corner regularisation); investigate Poiseuille FIRST to understand
what BSD actually does to LBM↔FV coupling on the simplest geometry.

- **M20 (Poiseuille F_total trace, `:fd`, ζ=0.75)**: BSD operates as
  designed at operator level on Poiseuille. F_poly_wide + F_BSD_narrow
  each carry ~0.5 % rel residual vs analytic d²u/dy² — same-sign,
  ADD algebraically (do not cancel), then (1−ζ)⁻¹=4× amplification →
  3.51 % on F_total at ζ=0.75 Wi=8e-4. Collapses 380× at Wi=1
  (elastic locking). **Smoking gun: cavity 8× ratio is NOT in the
  BSD chain, lives downstream**.
- **M21 (path matrix, 7 variants)**: NO BSD reformulation beats
  `:baseline` on smooth Poiseuille. Open Q5 REFUTED at root:
  `logfv_velocity_gradient_bc_aware_2d!` is bit-identical to
  `fvfd_velocity_gradient_2d!`. Cavity bug NOT operator-side.
- **M22+M23 (cylinder Cd convergence BSD ON & OFF)**: BSD impact
  collapses with mesh refinement (Δ Cd at Wi=0.1 goes 18.7 → 13.3 →
  8.9 → 1.4 Cd points as R goes 20 → 30 → 40 → 50). BSD ON matches
  rheoTool to 1.45 % at R=30; BSD OFF gap 12 % monotone-converges UP
  to BSD ON value. User's "anti-convergence" recollection RESOLVED:
  BSD ON over-shoots at R=30-40, BSD OFF under-shoots; both converge
  to same rheoTool-consistent limit from opposite sides.
- **M25 Phase 0 (Liu-match)**: `0000_qwall` Cd at R∈{20,30,40} matches
  Liu CNEBB to −0.4 / −0.7 / −1.0 % (approximate-PASS within numerical
  noise). Required fixing 3 bugs: CUDA detection silent-fail (Julia
  1.12 world-age in `getfield(Main, :CUDA)` — commit `e602726f`
  invokelatest wrapper), β=0.5 vs Liu β=0.59 default, M22-vs-M23 kwargs.
- **M26**: embedded `1111_circle` bug closed empirically. Newtonian
  bit-exact → bug lives ENTIRELY in polymer-coupling paths.
  `fvfd_tensor_divergence_embedded_2d_kernel!` divides by `cell_fraction`
  → singular Guo on cut cells → biases `f` → inflates Cd_p via MEA.

**Process lessons** (Boss-level, preserved):
- **Matrix-sweep missions**: prefer N parallel Departments over one
  Department managing N runs. M22+M23 dual-spawn fan-out worked even
  with Codex Anthropic API drop — Boss took host execution role.
- **Codex Engineer can hand off to Boss-on-host for execution** when
  API connectivity drops mid-mission. Codex writes bench script;
  Boss runs `--full` mode locally.
- **Julia 1.12 world-age trap in CUDA detection**: NEW gotcha class.
  Any `getfield(Main, :CUDA)` after dynamic `@eval using CUDA` MUST
  use `Base.invokelatest` + surface errors with `@warn` — bare
  `catch end` masks the failure into silent CPU fallback.

### Cluster M28-M32 — Polymer-scheme attribution falsified (2026-05-19 → 22)

Liu Table 3 column mis-read was the seed: M28 misattributed
Wi-dependent residual to polymer-scheme; M29 cluster iterated on
Ψ-advection scheme upgrades; **M31 + M32 Phase 4 falsified the entire
premise via wall-stress decomposition + spatial fingerprint**.

- **M29b (commit landed)**: MUSCL-superbee closes 56 % of Wi=1 gap.
  Improvement real but plateaus.
- **M29c-v2 (ROLLED BACK)**: removing M29b's ±2-cell rusanov fallback
  near solids → NaN at step 92,200 on rho/j=1/south wall. Same
  signature as lag-1 bug in M30 P2b.
- **M30 P2b (Bouzidi-FL two-pass)**: lag-1 read on x_ff in single-pass
  kernel → unstable. Two-pass split → STABLE but still under-bouncing.
- **M31 frame audit (3rd adversarial win)**: wall-ring integrators in
  `viscoelastic_logfv` post-processing MUST use `dx = (i-1) − cx_phys`
  (`:idx` frame). The `:phys` frame was 1 LU off, biased Cd_polymer
  by +24 %. After fix: M29b actually under-predicting by ~19 %, NOT
  over by 5 %. Memory: `[[feedback_wall_ring_idx_frame]]`.
- **M32 Phase 4 trifecta (D1 + D2bis + D3)**: closes M28-M32 cluster.
  - D1: 80 % of Cd gap is `Cd_pressure` × front-pole, NOT constitutive
  - D2bis: R=60 NaN = polymer back-force divergence at front-shoulder
    (bilateral arcs θ ≈ ±45°, r-R ∈ [0,7] LU)
  - D3-finalize: kraken-trace provenance — SAME mechanism via
    WriteMoments→vel_grad→poly_force chain
  - **M28/M33 polymer-scheme hypothesis EMPIRICALLY REFUTED**
- **M34 reframed (BC fix Phase 2b)**: Bouzidi-FL unpark, one-line
  wall_bc flip + lag-1 read fix.

**Process lessons** (Boss-level, preserved):
- **Cd wall vs volume**: Cd attribution MUST start from wall
  decomposition (Cd_p + Cd_s + Cd_pressure per θ); volume L2(τ) and
  peak τ_xx are NOT monotonic in Cd. Memory:
  `[[feedback_cd_wall_vs_volume]]`.
- **Adversarial default for uncertainty** (Boss-level): any
  uncertainty (hypothesis ranking, algorithm choice, parameter
  selection) → adversarial Claude+Codex. Memory:
  `[[feedback_adversarial_default_uncertain]]`. M31's 3rd adversarial
  win confirmed: post-processing harness bug masked the real
  Cd_polymer signal direction.
- **Monitor anti-pattern**: 2× Department subagents stalled 5h+ after
  capturing Monitor data. Synthesis missions MUST use Bash timeout
  300, gate on artifact, NEVER on Monitor. Memory:
  `[[feedback_monitor_antipattern]]`.

### Cluster M34 — Bouzidi-FL two-pass BC fix (2026-05-22 → 23)

- **M34 v1 (RED, commit `c3fe5063`)**: pass-1 spec reused
  `_TRT_LIBB_V2_GUO_FIELD_SPEC` which CONTAINS `ApplyLiBBPrePhase` →
  BC fires twice/step → over-bounce. 3/4 NaN on Aqua. Memory:
  `[[feedback_double_bouzidi_two_pass_trap]]`.
- **M34-fix v2 (YELLOW, commit `dc57f373`)**: fresh RAW pass-1 spec
  (no BC) + cut-link smoke (closed-box smoke with q_wall=0.5
  collapses Bouzidi-FL to halfway-BB algebraically → cylinder R=4-8
  cut-link smoke now MANDATORY). Memory:
  `[[feedback_smoke_must_exercise_cutlinks]]`. BC over-bounce reduced
  but not eliminated.
- **M34v3 (commit `e98b9687`)**: pass-3 cut-link rho recompute → close
  rho_w consistency gap. Still YELLOW for the cylinder Cd target.

**Process lessons** (Boss-level, preserved):
- **NaN uniform vs arc diagnostic**: triage protocol — fraction ≥ 90 %
  = BC over-bounce; bilateral front-shoulder arcs θ ≈ ±45° = polymer
  back-force divergence (D2bis fingerprint). 1-min classification.
  Memory: `[[feedback_nan_uniform_vs_arc_diagnostic]]`.

---

## 2026-05-23 — V&V hierarchy established + verdict: cylinder bug is geometric

User forced a methodological step-back after 7+ failed iterations on
cylinder L4 (M28 → M29b → M29c → M30 P2b → M31 → M32 Phase 4 → M34 v1
→ M34-fix → M34v3, all RED or YELLOW). New approach: build V&V
hierarchy L0→L4 with analytic/cross-code references, find first
divergence.

### Session arc

- **M28-M34 audit (Codex+Claude adversarial, cross-engine)**: convergent
  on "moment-field inconsistency at cut-link cells" as candidate root
  cause for both Q1 (Cd_pressure deficit) and Q2 (NaN under
  bouzidi_fl_twopass). Recommended fix: extend pass-3 brick to
  recompute ρ + ux + uy at cut-link cells.
- **User counterargument** (load-bearing): Newtonian Wi=0 matches rT to
  0.22%. Same BC, same moment chain. If moment inconsistency were the
  cause, Newtonian would also drift. Therefore moment chain is fine in
  isolation; the Wi-dependent drift requires a Wi-dependent mechanism.
- **M37 inventory**: 13 Kraken viscoelastic tests + 25 benches exist
  but are scattered. rheoTool has `rheoTestFoam` (one-cell imposed-
  velocity constitutive). Basilisk has `src/test/poiseuille-oldroydb.c`
  + `.ref` regression artefacts (the discipline pattern). Kraken
  already has imposed-velocity test (`test_logfv_frozen_channel_cde.jl`)
  and inverse back-force test (`test_viscoelastic_force_accounting.jl`)
  but ad-hoc, no canonical home.
- **M38 architecture**: built `bench/viscoelastic_validation/` skeleton —
  README + REFERENCES + INVENTORY + ref/ (Basilisk Poiseuille .ref 153
  rows, Basilisk lid-oldroydb Fattal-Kupferman 52+49 pts, Waters-King
  1970 formula, Bird-Armstrong-Hassager analytic) + L1 fully implemented
  + STUBs for L0/L2/L3a/L3b/L4.
- **M39 L1 first run**: 6/10 PASS at Wi≈3e-3. 4 FAIL all wall-sampling
  artefacts in the reference (reference used CD at j=1 cell-center, but
  HWBB wall is at j=0.5). Polymer chain in bulk is healthy.
- **M40 stencil-fix + Wi-sweep**: stencil fix replaces ref γ̇ with HWBB-
  aware quadratic one-sided (`(-3u₁+4u₂-u₃)/2`). M39 dump now 10/10.
  Wi sweep {0.001, 0.01, 0.1, 0.5, 1.0} all PASS. Errors decrease with
  Wi. min_eig_C matches Oldroyd-B steady-shear analytic
  `(2+2Wi²−√(4Wi²+4Wi⁴))/2` to <0.1 %.

### Verdict (2026-05-23, load-bearing)

**Polymer unit chain CORRECT up to Wi=1.0 on planar Poiseuille
Oldroyd-B.** The constitutive + Ψ-advection (`:rusanov`) + back-force +
log-conformation + Hermite + BSD + Guo chain produces analytic-accurate
results. M28/M33 "polymer scheme is the locus" hypothesis is
**empirically refuted at the unit level**.

The cylinder R=30 Wi=1 Cd deficit must therefore be in the **curved-
coupling layer**:
- Cut-link drag computation (q_wall integration)
- BSD over curved surface
- vel_grad stencil on non-axis-aligned solid neighbours
- Polymer wall BC on curved walls (M29b limitation: MUSCL fallback
  ±2 cells around solid)
- Some coupling between BC and polymer not active at planar walls

### Codex+Claude audit refined verdict

The cross-engine audit's "moment-field inconsistency" hypothesis was
not wrong per se, but applies ONLY in presence of curvature. The
flat-channel moment chain works at Wi=1 (L1 PASS). The inconsistency at
cut-link cells does matter at curved boundary, where the geometry
forces it to interact with polymer back-force.

### Methodology lesson

Stored as `[[feedback_localize_via_vv_hierarchy]]`. Future Boss MUST
NOT iterate fixes on L4 cylinder without first checking V&V hierarchy.
If L1 (or any lower level) is broken, fixing it must come before any
L4 iteration. If L1 is OK and L4 fails, the bug is strictly at the
levels in between (geometry, curved coupling).

### Acquired infrastructure

- `bench/viscoelastic_validation/` — full V&V suite
- 3 verdicts in `bench/viscoelastic_audit/M28_M34_AUDIT_*.md`
- 1 inventory + 1 architecture + 2 results markdowns for L1
- 6 diagnostic plots for L1 Wi=3e-3 case
- 1 CSV Wi-sweep results table
- 1 Wi-sweep error curve plot

### Recommended next session

Build **L4 curved-wall isolation test**: cylinder R=8 Newtonian Re=1
(no polymer) vs Bouzidi-FL analytical reference from M30 Phase 2a.
This tests the curved-BC layer **with zero polymer**. If Kraken matches
to <1% at this level, the curved BC is OK and the bug is specifically
in the polymer-coupling-at-curvature. If Kraken fails this level, the
curved BC itself is the problem.

Alternative L2 (Couette start-up Waters-King at Wi=1) gives transient
polymer info but is redundant with L1 Wi-sweep on the constitutive
question.

### Memory candidates updated

- `[[project_kraken_vv_suite]]` (NEW) — pointer to suite + verdict
- `[[feedback_localize_via_vv_hierarchy]]` (NEW) — methodology
- `[[project_m32_phase4_verdicts]]` (existing) — superseded by V&V on
  the polymer attribution question; retained for the spatial localization
  data (front-pole pressure deficit is still a valid empirical fact)


### 2026-05-23 evening — M41 verdict: locus is polymer × curved BC coupling

**M41 L4 Newtonian curved-BC isolation** completed on Aqua (jobs 21729717 + 21729718, exit 0):

| Setup | Cd_kraken | NaN | Δ vs rT 132.37 |
|---|---|---|---|
| R=30 :halfwayBB Newt | 132.076 | false | −0.22 % (sanity ✓) |
| R=30 :bouzidi_fl_twopass Newt | **132.637** | false | **+0.20 %** (BC correcting in right direction) |
| R=40 :bouzidi_fl_twopass Newt | 133.537 | false | (no rT ref, +1.1 % vs halfwayBB) |
| R=60 :bouzidi_fl_twopass Newt | **135.436** | **false** | (no rT, +2.5 % vs halfwayBB) — **finite at R=60 Newt** |

### Empirical decomposition (combined M40 + M41)

| Subsystem | Status | Reference |
|---|---|---|
| Pure polymer chain (planar) | CORRECT to Wi=1 | M40 L1 sweep |
| Pure curved BC (Newtonian, all R) | MOSTLY CORRECT (small R-bias 0.4 → 2.5 %) | M41 |
| **Polymer × curved BC coupling** | **LOCUS** | by elimination |

Wi-dependent jump: Newt R=30 :bouzidi_fl_twopass +0.20 % → Wi=0.1 R=30 +1.60 % → Wi=1.0 R=30 NaN. The amplification is polymer-driven at curved boundary.

### Top candidate (HIGH confidence)

**M29b MUSCL boundary fallback ±2 cells around solid.** M29b
(rusanov → muscl_superbee) gave +5 Cd at Wi=1 cylinder, BUT MUSCL
falls back to 1st-order Rusanov within ±2 cells of solid. Exactly
the shoulder zone where D1 found polymer × shoulder = +3.14 deficit.
M29c-v2 attempted to remove the fallback and NaN'd at step 92,200
on rho/j=1/south wall — likely the SAME lag-bug class as M30 P2b's
Bouzidi-FL (lag-1 read on x_ff in single-pass kernel).

### Next missions

- **M41-bis instrumentation probe**: instrument the M29b ±2-cell
  fallback zone to count cells, measure max(τ_p) in zone vs bulk,
  diff(Cd with/without fallback). Confirms target before fix.
- **M42 design** (if probe GREEN): MUSCL boundary relaxation
  applied via M30 P2b two-pass architecture (split kernel, lag-0
  reads at boundary cells) to avoid the M29c-v2 step-92k NaN.

### Collateral lessons (added to verdict file)

- **Manifest.toml NOT portable Mac↔Aqua**: different package
  registry snapshots (GPUCompiler 1.13.1 local vs 1.11.1 max Aqua;
  SimpleTraits 0.9.6 local vs 0.9.5 max Aqua). Future rsync must
  `--exclude 'Manifest.toml'`, each side `Pkg.instantiate` from its
  own registry. Lost ~1h on this trap today (2 failed Aqua submits).
- **CairoMakie stays in `[weakdeps]`**, never `[deps]`: prevents
  transitive pinning to versions Aqua's registry doesn't have.
  M39 V&V Department mistakenly moved CairoMakie + added JSON3 to
  `[deps]`. Reverted via `git checkout HEAD -- Project.toml`.


### 2026-05-23 late — M42-impl + G5 v3 PARTIAL (1/4 gates PASS+, 3/4 NaN)

**M42-impl (commit pending)**: Department spawned, claude-subagent inline,
~90 min wall. Files: src/fvfd/muscl_boundary.jl (187 LOC NEW),
src/fvfd/operators_2d.jl (+13), src/drivers/viscoelastic_logfv_2d.jl
(+2), src/fvfd/FVFD.jl (+1), bench/viscoelastic_logfv/run_cyl_bigsweep_v2_2d.jl
(+2), bench/viscoelastic_logfv/run_cyl_m42_g5_a100.pbs (84 NEW),
test/test_muscl_boundary_relax.jl (178 NEW), test/runtests.jl (+1),
bench/viscoelastic_audit/M42_IMPL_VERDICT.md (NEW + updated with G5 v3).

Smoke (CPU F64, 14.4 s wall): 4/4 PASS, 42/42 assertions.

**Aqua G5 v3** (job 21736323, Wi=0 removed because driver rejects λ=0):
4 cases R={30,60} × Wi={0.1,1.0}, 6:48 walltime, Exit_status=0.

| R | Wi | Cd_kraken | NaN |
|---|---|---|---|
| 30 | 0.1 | **131.073** | false (PASS+ G5-3) |
| 30 | 1.0 | NaN | true (G5-1 FAIL) |
| 60 | 0.1 | NaN | true (G5-4 FAIL) |
| 60 | 1.0 | NaN | true |

Progression Wi=0.1 R=30 :rusanov :halfwayBB 129.39 (−0.80 %) →
:muscl_superbee_relax 131.073 (+0.49 %). Closer to rT 130.43.
But Wi=1 R=30 NaN (same envelope as M29c-v2 step 92k).

**Diagnostic candidat**: zero-slope on broken-axis + full MUSCL on
non-broken axis creates asymmetric FV update at cut-link cells. At low
Wi this is below polymer stability envelope; at high Wi/R it triggers
polymer-coupled instability.

**M42-v2 candidate fixes** (next session):
1. 1-sided minmod ψ(r) = max(0, min(r, 1)) on broken-axis (M42 design §8
   alternative — strictly less dissipative AND symmetric)
2. Narrower relaxation band (1 LU instead of 2 — narrows asymmetric zone)
3. Spatial NaN fingerprint analysis on G5 v3 .jls dumps first
   (discriminate cylinder-band vs open-wall NaN before code change)

**Aqua trap learned**: PBS-side `KRAKEN_WI_LIST="0.0,..."` triggers
driver `ArgumentError: lambda must be positive` — driver requires λ>0.
Use M41-style trick (β=1.0, λ placeholder, ν_p=0) for Newtonian via
viscoelastic driver, OR skip Wi=0 from PBS lists. Documented in M42 PBS.

**Manifest.toml NOT portable**: re-confirmed today. Past pain → future
PBS rsync MUST `--exclude 'Manifest.toml'`. Aqua's `Pkg.instantiate`
must resolve from its own registry.

### Session stats (cumulative 2026-05-22 → 2026-05-23)

15 commits over 36 hours wall-clock:
- M28-M34 cluster (8 commits, RED iteration)
- M40 V&V hierarchy + verdict polymer chain CORRECT (1 commit, STRATEGIC PIVOT)
- M41 + M41-bis curved-BC isolation + fallback locus CONFIRMED (3 commits)
- M42 design + impl + G5 v3 (3 commits — design GREEN, impl partial)

Net narrative shift: cylinder Wi=1 gap localized empirically to the
M29b ±2-cell fallback band (0.33 % of fluid, 19-65× bulk polymer
stress). M42 zero-slope relax closes Wi=0.1 R=30 toward rT but NaN's
at higher stress. Cluster M28-M42 essentially ONE iteration short of
closure.

---

## 2026-05-24 — Compaction + NaN fingerprint mission opened

boss.md compacted from 1896 → ~360 lines. M1-M34 narrative archived
to `boss_archive_M1_M34_pre_20260523.md` (1678 lines intact).
2026-05-23+ era preserved inline.

Mission of the session: **NaN fingerprint analysis on G5 v3 .jls
dumps** (cyl R=30 Wi=1, R=60 Wi=0.1, R=60 Wi=1) per triage protocol
`[[feedback_nan_uniform_vs_arc_diagnostic]]`. Discriminate
uniform-BC-overbounce vs front-shoulder-polymer-divergence before
choosing M42-v2 minmod vs narrower-band implementation.

---

## 2026-05-24 evening — M44 CLOSES M28-M42 cluster via slbm-paper Guo fix port (commit 9fd92ab0)

**Root cause finally identified**: `logfv_compute_macroscopic_forced_field_2d_kernel!`
(`src/kernels/logconformation_fv_2d.jl:1047-1048`) and
`compute_macroscopic_forced_2d_kernel!` (`src/kernels/macroscopic.jl:71-72`)
added `+F/2` to ux/uy in the post-collision readout, double-counting
the half-step that Convention-I `collide_guo_field_2d!` already
integrates. Per-cell bias `Δu = F_local/(2·ρ)` propagated into LBM
pressure → contaminated `Cd_pressure` precisely where `div(τ_p)` is
sharp (cylinder front-pole = the 80 % bucket M32 P4 D1 localized).

Fix was already on slbm-paper as commit `5ec27044` (2026-05-14,
"fix(convention): remove double-counted Guo half-step from 7 production
getters"). dev-viscoelastic had simply not inherited it.

**Validation (Aqua A100 F64 100k, job 21825614)**:
- pre-fix Cd = 111.09 (gap +9.54 %)
- post-fix Cd = **118.10** (gap +1.89 %) — **78 % closure**
- F32→F64 +0.74 Cd consistent with M32 noise caveat
- Local Metal F32 50k smoke gave 117.36 — same direction, F32 floor

**Discovery path (load-bearing for next time)**: USER prompted
"on slbm-paper on vient de regler un bug guo force, ca pourrait etre
ca?" → `git log --all | grep -i guo` surfaced `5ec27044` → audit
confirmed exact pattern match → 2-line fix + test rewrite. **8 days
of cluster M28-M42 (10+ RED missions) resolved by external user
knowledge of sister-branch state.** Adversarial Codex+Claude on M43
(advection limiter) was CONCORDANT-HIGH on the wrong framing; both
engines re-derived the same plausible-but-wrong code path because
the empirical signal pointed at advection-band stress concentrations
that were themselves a SYMPTOM of the readout bias.

**Codex audit (`bench/viscoelastic_audit/M44_GUO_AUDIT_CODEX.md`)**
inventoried 7 getters with the same pattern (G1-G7). Fixed:
- G3 viscoelastic (this commit) — primary impact
- G1 base 2D — collateral consistency

Out of scope for this commit (separate audits required):
- G2 3D `compute_macroscopic_forced_3d_kernel!`
- G4 VOF/pressure `compute_macroscopic_pressure_2d_kernel!`
- G5 phasefield `compute_macroscopic_phasefield_2d_kernel!`
- G6 thermal `macroscopic_boussinesq` (likely in-collision helper)
- G7 fused LI-BB Guo-field `WriteMoments` (overwritten by G3 downstream)

Also caught: `test/test_viscoelastic_logfv_patch_ladder.jl:1416-1435`
(M5b) **explicitly asserted the buggy `ux ≈ 0.5·fx` from rest** —
a bug-pinning test that prevented prior detection. Rewritten as a
proper N-step Convention-I pair test asserting `N·F` at 1e-12.

### Decisions taken
- Commit fix + verdict + Codex audit + PBS in single commit `9fd92ab0`
  (NOT staged: other working-tree mods unrelated to viscoelastic).
- M28-M42 cluster CLOSED. M44-VV-B (Alves 4:1) and M44-VV-C (square
  cylinder) no longer required for closure — may be revisited as V&V
  hardening later if useful.
- M44-VV-A 4-roll mill: status unchanged (driver uses
  `collide_viscoelastic_source_2d!` which does NOT exercise the fixed
  pair — its 0/48 PASS was a setup issue, not the Guo bug). Documented
  in VERDICT.

### Process lessons added
- **NEW**: `[[feedback_port_sister_branch_fixes]]` — pro-actively audit
  sister-branch git logs after 2-3 RED missions in shared-kernel areas.
- Reinforced: `[[feedback_code_path_provenance]]` — adversarial
  CONCORDANT-HIGH does NOT protect against framing errors when both
  engines start from the same (wrong) symptom localization.
- `[[feedback_department_bail_out_pattern_20260523]]` confirmed at 4/4
  even with explicit warning in brief; hard rule no-Department-for-
  Codex-wait now active.

### Reusable artifacts
- `bench/viscoelastic_audit/M44_GUO_FIX_VERDICT.md` — full adversarial
  trail + acceptance numbers
- `bench/viscoelastic_audit/M44_GUO_AUDIT_CODEX.md` — Codex's G1-G7
  inventory (reuse before any future Guo-related fix)
- `bench/viscoelastic_logfv/run_cyl_m44_guo_fix_confirm_a100.pbs` —
  Aqua A100 F64 single-case template for future Cd convergence checks

### Open follow-ups (NOT urgent, NOT blocking)
- Re-run M22+M23 mesh sweep R∈{30,40,50} Wi=0.1 post-fix: the +51 %
  anti-convergence in shoulder Cd_pressure that drove the staircase
  hypothesis should disappear; mesh convergence should be monotone.
- M32 P4 D1 re-decomposition post-fix at R=30 Wi=1: the per-θ
  Cd_pressure × front_pole bucket (was +80 % of gap = +8.34 Cd) should
  now be the dominant SHRINKAGE — quantify.
- G2 (3D forced macroscopic) and G4/G5 (VOF/phasefield) separate
  audits per Codex inventory if those subsystems are used in active
  benchmarks.

---

## 2026-05-25 — M44 sweep + M45 residual audit (commits 4df9d431, 83cb3efe)

### M44 sweep `21827394.aqua` (Aqua F64, 48 cases, 51 min)

Matrix 4 R × 4 Wi × 3 β covering the cylinder design envelope.
**0/48 NaN**. Sanity anchor R=30 Wi=1 β=0.59 reproduces M44 confirmation
to 0.002 Cd. **All R=60 cases stable** including Wi=0.1 and Wi=1 that
were NaN pre-fix → confirms Guo velocity bias was the instability
trigger, not a separate polymer mechanism. β scaling Cd_p ∝ (1−β)
confirmed. Wi minimum Cd at Wi≈0.5 (Oldroyd-B saturation + N1
contribution). Mesh convergence Wi=0.1 quasi-flat (±0.2 Cd).

### M45 residual mesh-Wi audit (B local + C Codex parallel)

At Wi=1, Cd decreases monotonically with R (118.10 → 113.23 from
R=30 → R=60). Drop entirely in Cd_s. Cd_p flat.

**B per-θ decomp** (`bench/scratch/m44_postfix_walldecomp/`,
reuses M32 P4 D1 `kraken_wall_decomp`):
- M44 fix closes **62 %** of the original M32 `Cd_pres × front_pole`
  bucket (was +8.34, now +3.21) AND continues converging with R
  (+1.81 from R=30 to R=60). Healthy.
- Cd_polymer per-region is now nearly correct at R=30 vs rT.
- Residual mesh-R drop is in **Cd_solv × shoulder (−1.08)** + **Cd_pres
  × wake (−1.61 more negative)** — Newtonian-like channels, NOT polymer.

**C Codex audit α / β / γ**
(`bench/viscoelastic_audit/M45_RESIDUAL_AUDIT_CODEX.md`):
- **α (rT under-converged)**: untestable, `bench/rheotool/` has only
  R=30 reference; larger-domain rT variant differs only 0.018 Cd.
- **β (domain effect)**: nondim D/H = 0.5 stays constant (refutes
  literal "channel grows"), BUT lattice-distance to outlet ZouHe
  pressure grows 420→840 LU + TRT s_plus scales (1.05→0.71).
- **γ (residual bug)**: NO Guo-half-step-class double-count found on
  pressure readouts. Only candidate: default
  `logfv_polymer_force_bc_aware_2d!` + `fvfd_tensor_divergence_2d!`
  are NOT q_wall cut-cell aware in default mode (sweep used
  `embedded_force=0`, `embedded_gradient=0`).

**Combined verdict**: residual is NOT a M44-class bug. Mixed β/γ
plausible. M28-M42 cluster remains CLOSED; M45 documents an open
research-grade question, NOT a blocker.

### Decisions taken
- Ship M44 + M45 documentation as-is (commits 9fd92ab0, 4df9d431,
  83cb3efe).
- Open follow-ups (NOT urgent):
  - Controlled run `embedded_force=1` + `embedded_gradient=1` at R=30
    and R=60 (~6 min Aqua) to test γ candidate.
  - L_up/L_down discriminant at fixed R=60 (~10 min Aqua) to
    separate β lattice-distance from mesh refinement.
  - Generate rheoTool R=40/50/60 reference (week of OpenFOAM work)
    to close α.

### Process lessons
- B (Boss-direct per-θ decomp local) + C (Codex Boss-direct audit)
  parallel pattern was efficient: ~30 min wall to combined verdict.
- Reusable: `bench/scratch/m44_postfix_walldecomp/run_walldecomp_postfix.jl`
  is a generic per-θ wall decomposer for any post-fix .jls. Apply
  to any cylinder benchmark .jls with the schema (rho + ux + uy +
  tauxx/xy/yy + is_solid + Nx + Ny + cylinder_x_lbm/y_lbm + radius_lbm).

---

## 2026-05-25 evening — M46 Newt sweep + M46-B time-convergence reveal R=60 NOT converged at 100k (commit ce5fa838)

### M46 K Newt sweep (job 21861929, 8 cases halfwayBB + Bouzidi-FL × R={30,40,50,60})

Via β=1.0 placeholder (zero polymer fraction; Wi=1 placeholder because
driver requires λ>0). M41 anchors all reproduced bit-for-bit.

- **halfwayBB**: Cd 132.08 → 132.68 across R=30→60 (**↑ +0.60**)
- **bouzidi_fl_twopass**: Cd 132.64 → 135.44 across R=30→60 (**↑ +2.80**)

**Direction OPPOSITE to M44 Wi=1 sweep** (which had ↓ −4.87). Eliminates
the M45 β-class hypotheses (lattice-distance, TRT scaling, domain).

**Anomaly**: Bouzidi-FL Newt trace_C_max explodes from 209 (R=30) to
**1.4e7 (R=60)** despite β=1.0 (polymer dormant LBM-side, no F_poly).
The polymer C-tensor evolution is unstable under Bouzidi-FL Newt R=60.
Cd unaffected (no F_poly), but this is a separate Bouzidi-FL
polymer-chain bug worth its own audit.

### M46-B time-convergence probe (job 21862685, R=30@400k + R=60@{100k,200k,400k} Wi=1)

| Case | Cd | Δ vs 100k |
|---|---:|---:|
| R=30 @ 400k | **118.099** | =100k ±0.003 |
| R=60 @ 100k | 113.234 | (baseline) |
| R=60 @ 200k | 112.130 | −1.10 |
| R=60 @ 400k | **109.424** | **−3.81 (drift ACCELERATING)** |

**R=30 fully temporally converged at 100k** → M44 anchor SOLID, 78%
closure validated empirically.

**R=60 NOT converged even at 400k**. Drift is accelerating, not
stabilizing. Cd_p grows with time (13.97 → 15.30) while Cd_s drops
faster (113.04 → 108.19). trace_C stable ~230, no NaN.

Flow-through analysis (u_mean=0.005, domain=30R LU):
- R=30: 100k = 0.56 flow-through, 400k = 2.22 flow-throughs ✓
- R=60: 100k = 0.28 flow-through, 400k = 1.11 flow-through ✗

**The M44 sweep "Cd decreases with R" pattern is NOT a mesh effect.**
It is **incomplete temporal convergence scaling with flow-through
requirements**. At R=60 with only 0.28 flow-through, the wake is still
developing.

### Implications for M45 verdict
- M44 fix at R=30: **VALIDATED temporally**. Cluster M28-M42 still CLOSED.
- M44 sweep at R≥40: results need re-run with longer max_steps.
- M45 γ candidate (FVFD non-q_wall-aware stencil) **unsupported** —
  the residual was measured on non-converged snapshot.

### Open items (load-bearing for next session)
1. **R=60 Wi=1 true steady-state Cd**: needs Cd time-series logging
   in driver, OR run at 1.6M-3.2M steps to verify plateau exists.
2. **User hypothesis (worth testing)**: Bouzidi BCs may be buggy.
   The trace_C explosion under Bouzidi Newt is concrete evidence of
   a Bouzidi-side polymer-chain bug. M44 sweep used halfwayBB (not
   Bouzidi), so the R=60 drift is not directly Bouzidi-driven, but
   the analogous fix may exist for halfwayBB+polymer at long times.
3. **Guo fix completeness**: K Newt halfwayBB R=30 = 132.076 vs rT
   132.37 = −0.22%. Small residual still exists in Newtonian path.
   Codex M44 G2/G4/G5/G6/G7 inventory items remain unfixed (claimed
   not on cylinder path, but worth re-verifying after this finding).
4. **Multiple confounded variables**: resolution (R) + Wi + polymer
   coupling + BC type. Disentangle systematically with one variable
   at a time, max_steps long enough at each R.

### Reusable artifacts
- `bench/viscoelastic_logfv/run_cyl_m46_newt_sweep_a100.pbs` (Newt
  sweep template, dual-BC matrix)
- `bench/viscoelastic_logfv/run_cyl_m46b_tconv_a100.pbs` (time-convergence
  probe template, single Wi/β at multiple max_steps)
- `bench/viscoelastic_audit/M46_NEWT_AND_TCONV_VERDICT.md` (full
  hypothesis matrix + observations)
- `tmp/m46_newt_sweep/21861929.aqua/` (8 Newt .jls + CSVs)
- `tmp/m46b_tconv/21862685.aqua/` (4 R=60 .jls at 100k/200k/400k +
  R=30 @ 400k)
