# Department memory — Kraken.jl viscoelastic cavity spatial debug

Patterns useful for every Department on this project. Initialised
2026-05-15.

## 2026-05-15 — Engineer-brief conventions for log-FV cavity work

- Always specify `KRAKEN_BACKEND` explicitly in the brief's validation
  env section. Default detection picks Metal on macOS (F32) which is
  NOT comparable to the Aqua A100 F64 baseline.
- The cavity comparison harness
  `bench/viscoelastic_logfv/run_cavity_oldroydb_vs_rheotool.jl` already
  consumes env vars `KRAKEN_N_LIST`, `KRAKEN_U_MAX`,
  `KRAKEN_OUTPUT_DIR`, `KRAKEN_LAMBDA_PHYS`, `KRAKEN_BSD_FRACTION`.
  Engineer briefs should set these via env, never patch the script.
- For PBS wrappers: keep `Pkg.instantiate(); Pkg.precompile()` out of
  the per-case loop. Run it once at the top of the job.

**Why**: M1 brief writing will repeatedly hit these. Skipping the env
discipline pollutes the baseline.

## 2026-05-15 — rheoTool reference loaders

`run_cavity_oldroydb_vs_rheotool.jl` defines:
- `read_rheotool_vertical_U(path)` — 4 cols (y, Ux, Uy, Uz)
- `read_rheotool_horizontal_tautheta(path)` — 13 cols (x, tau×6, theta×6)

Any new analysis script should import / replicate these instead of
re-deriving column ordering. Reference layout:
`bench/rheotool/cavity_oldroydb_log_re001_de1_b05/postProcessing/...`.

**Why**: column ordering is non-obvious; getting it wrong silently
inverts the comparison.

## 2026-05-15 — Per-case CSVs already carry rheoTool aligned to Kraken grid

`run_cavity_oldroydb_vs_rheotool.jl` writes
`profile_vertical_x0.5.csv` and `profile_horizontal_y0.75.csv` with
both Kraken AND rheoTool columns already interpolated onto the
rheoTool sample grid. Downstream analyses should read those columns,
NOT re-load the rheoTool `.xy` files.

**Why**: avoids the 4/13-col loader trap; also prevents
interpolation-method drift between scripts. Found during M1.

## 2026-05-16 — Codex sandbox cannot run julia (lockfile EPERM)

Codex CLI's `workspace-write` sandbox blocks `juliaup` / `julia` from
creating the launcher lockfile (`Operation not permitted (os error 1)`).
The Engineer therefore CANNOT execute the exit-criterion command for
any Julia mission and will stop there with a clean diff. The Department
MUST re-run the exit criterion itself on the host shell. Plan briefs
accordingly: write a self-test that prints a grep-able summary so the
Department's re-run is the single source of truth for success.

**Why**: avoids treating "Engineer stopped at validation" as failure
when the only blocker is a sandbox capability. Found during M2.

## 2026-05-16 — rheoTool cavity persisted snapshots at t≈8

The cavity reference case
`bench/rheotool/cavity_oldroydb_log_re001_de1_b05/` writes
**full cell-data** fields (not only sampleDict probes) at ~1 phys
time stride, including the time directory closest to t=8 which is
`7.999786329655222/`. Persisted gzipped fields: `U.gz`, `theta.gz`
(symmTensor = log-conformation), `tau.gz`, `p.gz`, `phi.gz`, plus
`eigVals.gz`, `eigVecs.gz`, `ddt0(U).gz`, `ddt0(theta).gz`. Mesh is
127×127 (= 16129 cells) on [0,1]² with z extruded ±0.5. Both `U` and
`theta` can be parsed by the cylinder harness's FOAM reader
(`parse_vol_vector`, `parse_vol_symmtensor`) — same regex / path
conventions.

**Why**: avoids the assumption in earlier prompts that only the 1D
sampleDict probes survive at t=8. The full field is available, so
frozen-replay across geometries does not need to extrapolate from
profile samples.

## 2026-05-16 — rheoTool `theta` IS log(C), not C

In rheoTool OldroydBLog the persisted symmTensor `theta` is already
the log-conformation `log(C) = Psi` in Kraken wording. Initialising
the Kraken polymer state from `theta` requires NO exp/log transform
— direct copy of the 6 components in symmTensor order
`(xx, xy, xz, yy, yz, zz)`. Confirmed by the cavity harness loader
docstring (~line 94 of `run_cavity_oldroydb_vs_rheotool.jl`) and
used directly in `run_rheotool_frozen_replay_cavity_2d.jl`. For
geometries where only `tau` is persisted, the cylinder harness's
`psi_from_tau` (Newtonian prefactor → C → log) remains the fallback.

**Why**: the cylinder frozen-replay re-derives Psi from `tau`. For
cavity (and any other rheoTool log-conformation case) reading
`theta` directly is shorter, exact, and skips the SPD-positivity
check.

## 2026-05-16 — Reusable polymer-pipeline frozen-replay call pattern

For a frozen-U replay of ONLY the log-FV polymer pipeline on any
axis-aligned 2D geometry without an embedded obstacle, per step:

```
logfv_cell_velocity_to_faces_bc_aware_2d!(..., logfv_bc)
logfv_advect_upwind_bc_aware_2d!(..., dummies, ux_face, uy_face,
                                 is_solid, dx, dy, logfv_bc, one(T))
fvfd_velocity_gradient_2d!(..., is_solid, dx, dy, logfv_bc)
for k in 1:n_substeps:
    logfv_step_constitutive_log_2d!(..., lambda_lu, dt_poly,
                                    LOGFV_MODEL_OLDROYDB, T(0.0))
logfv_stress_from_log_2d!(..., prefactor)
```

with `logfv_bc = logfv_wallxwally_bcspec_2d()` for closed boxes,
`logfv_periodicx_wally_bcspec_2d()` for x-periodic channels,
`logfv_openx_wally_bcspec_2d()` for inlet/outlet. DO NOT call
`_logfv_cavity_apply_wall_gradient_correction!` in a frozen replay:
rheoTool's U already encodes the lid shear at cells adjacent to the
wall, and the LBM ghost-correction would double-count.
`dt_poly = 1 / n_substeps` in LU (source kernel expects LU time).

**Why**: this is the third frozen-replay harness in the project
(cylinder, channel, now cavity). The pattern is identical modulo BC
spec; future contraction / BFS replays can copy it directly.

## 2026-05-16 — rheoTool 0/ initial-condition files are plain ASCII

The `0/` time directory of a rheoTool case (e.g.
`bench/rheotool/cavity_oldroydb_log_re001_de1_b05/0/`) stores
boundary conditions as **plain ASCII OpenFOAM dictionaries**. The
`.gz` extension appears only on persisted simulation snapshots
(later time directories), NOT on the `0/` initial conditions. No
`gunzip` step is needed to read the BC blocks for `theta`, `tau`,
`U`, etc.

**Why**: avoids spending a Department iteration on a wrong unzip
path. Found during M6-A.

## 2026-05-16 — Engineer brief git-status invariant must scope to diff

When the working tree has pre-existing dirty state (very common on
`dev-viscoelastic`), do NOT write "git status --short returns the
only allowed-zone files" as a global invariant in the Engineer brief.
Codex aborts because the pre-dirty files trip the check before it
creates its own file. Phrase it as:
"`git status --short <path-of-allowed-edit-zone>` returns only the
allowed-zone files" — scoped to the Engineer's own diff, not the
whole tree.

**Why**: caused M7 to stop on its first run; was already an issue on
earlier missions but only documented now. Affects every Department.

## 2026-05-16 — Stale `.engineer_brief.md` silently wins for the runner

`run-engineer.sh` defaults to `.engineer_brief.md` if no explicit path
is passed. If a previous mission left that file in place (because the
prior Department or Boss did not delete it), the new Engineer reads
the STALE brief instead of the current one. Symptoms: Codex appears
to work on the wrong mission or refuses to act on inputs the brief
"already mentioned". Mitigation: always pass the absolute brief path
explicitly (`bash run-engineer.sh <project> <mission> <abs_brief_path>`),
and the Boss should `rm -f .engineer_brief*` between missions as part
of the post-commit cleanup.

**Why**: found during M10's first Codex invocation — the M5-B brief
at `.engineer_brief.md` (left over from before .engineer_brief_M5B.md
was added) was picked up silently. Costs an Engineer iteration each
time.

## 2026-05-17 — runtests baseline carries pre-existing failures on dev-viscoelastic

`julia --project=. test/runtests.jl` on `dev-viscoelastic` HEAD exits
with status 1 and prints "169194 passed, 6 failed, 0 errored, 4 broken".
The 6 failures live in "Pure shear Oldroyd-B steady state, fully
periodic"; the 4 broken are LI-BB canary + P18b2c. Verified pre-M16
by stashing the M16A diff and re-running on bare HEAD. **A refactor
mission's exit criterion must therefore not be "runtests exits 0" but
"failure count and identity are PRESERVED" vs HEAD baseline.** Brief
the Engineer/Department accordingly: capture the pass/fail/broken
summary line and compare it byte-for-byte, not the exit code.

**Why**: M16 nearly got reported RED by exit-code; the Department's
own pre-flight stash baseline saved it. Save the next mission from
the same trap.

## 2026-05-17 — cmp byte-equality smoke is infeasible on dirty trees

The "stash the diff → re-run smoke on HEAD → cmp the CSVs" pattern
in `~/.claude/skills/orchestrator/department_brief_template.md`
breaks on `dev-viscoelastic` because the working tree carries ~150
M-unrelated dirty files; `git stash push -- <paths>` fails when the
new (untracked) file isn't indexed (`error: Entry … not uptodate.
Cannot merge`). Workaround attempted by M16: `mv newfile /tmp/.bak`
then `git stash push -- <edited>`. For refactor-pur missions, the
runtests baseline-preservation check + symbol-grep relocation +
diff-stat scoping are sufficient evidence; the full byte-equality
smoke can be deferred to the next production validation step (e.g.
the Aqua N=64 t=8 reproduction before M17).

**Why**: documented to spare M16b/M16c/M16d (future SPLITs of the
remaining drivers in `viscoelastic_logfv_2d.jl`) from the same dead
end. Defer cmp, lean on runtests + symbol-grep.

## 2026-05-17 — Test invocation: julia direct, NOT Pkg.test()

`test/Project.toml` on `dev-viscoelastic` is intentionally minimal
(only `Test` dependency). `Pkg.test()` activates THAT environment,
so `using Kraken` succeeds but Kraken's internal
`using KernelAbstractions` (and Metal/CUDA when applicable) fails to
resolve, resulting in a near-empty test pass (~58 tests instead of
the real 169194/6/0/4 baseline).

**Rule for all future Departments**: the test invocation MUST be:

```bash
\
julia --project=. test/runtests.jl
```

NOT `julia --project=. -e 'using Pkg; Pkg.test()'` and NOT
`julia --project=test test/runtests.jl`. Both of those use the
near-empty `test/Project.toml` and silently report wrong test counts.

The Boss is aware `test/Project.toml` is incomplete; fixing it is a
separate maintenance mission (would need the full dep list from the
main `Project.toml`, plus a decision on GPU-backend deps). Until that
mission lands, Departments MUST use the direct-script invocation.

**Why**: M17-pre v2 reported "test drift 169194 → 58" between
missions; investigation showed the v1/M16 Department used the direct
script and the v2 Department used `Pkg.test()`. The number drift was
purely the invocation difference, not a code regression. Save the
next Department from re-discovering this.

## 2026-05-18 — Post-hoc F_total decomposition pattern (M20-style)

For any coupled viscoelastic driver that returns `fx_total` (BSD-
corrected body force) + `psi*` (log-conformation) + `ux, uy` in its
result tuple, the three additive contributions to `F_total` can be
recovered post-hoc using existing kernels — NO driver patch needed:

```julia
# Step 1: rebuild τ_p from ψ (same prefactor as in driver)
prefactor = nu_p / lambda
logfv_stress_from_log_2d!(tauxx, tauxy, tauyy, psixx, psixy, psiyy, prefactor)

# Step 2: rebuild F_poly_wide from τ_p (same BC spec as driver)
fx_poly_wide = zeros(...); fy_poly_wide = zeros(...)
logfv_polymer_force_bc_aware_2d!(
    fx_poly_wide, fy_poly_wide, tauxx, tauxy, tauyy, is_solid, dx, dy, logfv_bc
)

# Step 3: extract F_BSD_narrow by algebraic identity from the driver chain
#   F_total = Fx_body + F_poly_wide − F_BSD_narrow
fx_total_no_body = fx_total .- Fx_body
fx_BSD_narrow = fx_poly_wide .- fx_total_no_body
```

Cells with `is_solid == true` should be excluded from any residual
analysis (kernels leave undefined values there). For 1D-in-y problems
(Poiseuille, channel) row-mean across x collapses the per-x noise.

Reusable for cavity, step, BFS, contraction, or any future
viscoelastic geometry that uses the same BSD-via-correction-step
chain. The pattern requires NO modification of `src/` — it is pure
instrumentation through public kernel APIs.

**Why**: avoids the next Department drafting a Codex brief that asks
for driver patching ("instrument F_total step-by-step inside the
loop") when the steady-state decomposition is sufficient and can be
extracted from the existing return tuple. M20 used this exact
pattern in 282 LOC; the cavity equivalent would be ~350 LOC with the
embedded geometry path.

## 2026-05-18 — Wrap variant sweeps in try/catch with sentinel nan_step

For any multi-variant sweep bench (M21-style: N variants in a single
script), wrap each variant call in `try / catch DomainError` and use
`nan_step = 0` as the sentinel for crash-before-first-NaN-watcher-tick
(typically caused by `logfv_log_spd_sym2_2d` raising DomainError when
C goes non-SPD inside a substep, before the per-5000-step NaN watcher
fires). Without this, a single variant's mid-substep DomainError
truncates the whole sweep and leaves the later variants unmeasured.
M21 found `:epsilon_force` at Wi=1 crashes this way (substep DomainError
at step 0); the try/catch with sentinel let the remaining variants
finish so the verdict could rank everything.

Pattern:
```julia
for variant in variants
    nan_step, result = try
        run_variant(variant)
    catch e
        e isa DomainError ? (0, nothing) : rethrow(e)
    end
    record!(variant, nan_step, result)
end
```

**Why**: M21 was almost truncated by `:epsilon_force` Wi=1; the
Department patched the try/catch wrapping post-hoc. Future matrix-sweep
bench briefs MUST include this pattern from the start, not as a
recovery after the first crash.

## 2026-05-18 — Cylinder `embedded_drag` only affects Cd_p and Cd_bsd

Driver kwarg `embedded_drag::Bool` in the viscoelastic cylinder
v2 driver toggles `drag_p` (= polymer-force drag integral, driver
line ~470) and `drag_bsd` (= BSD-correction drag integral, driver
line ~485) between LBM cut-link momentum exchange (OFF, `:qwall`
geometry) and FVFD traction integration on the embedded quadrature
(ON, `:circle` geometry). It does **NOT** affect `drag_s = compute_
drag_libb_mei_2d(...)` (`src/drivers/cylinder_libb.jl:98-163`)
which is always sourced from the LBM Mei MEA. Therefore `Cd_s` is
invariant under `embedded_drag` flip in ALL regimes; only the
composite `Cd_kraken = Cd_s + Cd_p − Cd_bsd` changes.

**Implication for Cd-related briefs**: never say "Cd_s changes
with embedded_drag" — always specify which component. The handoff
2026-05-18 (next-session prompt) said "+8.8 Cd_s ghost drag" but
the correct attribution is "+8.8 Cd_kraken with embedded ON" —
loose wording that misled the initial hypothesis. Future Department
briefs touching Cd MUST specify the component (`Cd_s`, `Cd_p`,
`Cd_bsd`, `Cd_kraken`) when discussing flag effects.

**Why**: prevents the next Department from mis-attributing a polymer-
pipeline bug to the LBM solvent side. Found during M26-impl
2026-05-18.

## 2026-05-18 — Phase 0 cylinder DoE matrix gaps

Job `21563085.aqua` Phase 0 Liu-match runs only 4 of 16 possible
embedded tuples: `0000_qwall`, `1000_qwall`, `0001_qwall`,
`1100_qwall`. Missing for full H1/H2/H3 disambiguation:
- `0010_qwall` (force-only) — isolates H2.
- `0100_qwall` (advection-only) — would isolate the M17-canary-A
  family wall-ghost effect specifically.
- `0011_qwall`, `0101_qwall`, etc. — pairwise interactions.

If Phase 0 results show `0001_qwall` ≈ `0000_qwall` (drag flag
isolated → no ghost), the +8.8 must come from `embedded_force` and/
or `embedded_advection` (which can be cross-checked against
`1100_qwall`). Phase 0 alone cannot fully disambiguate H2 from H3
without a Phase 0b that adds `0010_qwall`. Worth adding ~3 more
runs (~30 min A100) to the next sweep if Phase 0 doesn't pin the
bug to a single flag.

**Why**: Phase 0 was tuned for Liu-reference reproduction, not for
embedded-flag DoE. The DoE+Liu-match goals partially conflict;
future cylinder briefs should explicitly state which is primary.

## 2026-05-18 — Boss-level adversarial dual-spawn pattern validates

Boss-level dual-spawn (Department-A pure-analysis Claude general-
purpose, Department-B pure-impl Codex via run-engineer.sh) for the
M26 embedded_drag bug hunt produced **complementary** verdicts that
each ruled out a hypothesis the other could not. M26-analysis
identified the mechanism (cell-fraction divisor + half-cell ghost
coupling). M26-impl ratcheted the polymer-coupling localisation
(Newtonian-clean → bug must be in polymer-only paths). Together
they fully scoped M26b without needing a third spawn.

**When to use Boss-level dual**: when ONE Department alone risks
echo-chamber (e.g. math-Claude alone might over-attribute to a
known pathology like the M17-canary-A pattern; impl-Codex alone
might mis-attribute to a flag it didn't isolate from the polymer
pipeline). The dual gives independent triangulation. Cost: 2× spawn
+ Boss synthesis (~5-10 min of Boss time). Worth it for any
mission where the symptom has 3+ candidate mechanisms (per the
H1/H2/H3 structure of M26).

**Why**: this is the 5th validation of the adversarial pattern this
project (after M17-epsilon Claude+Codex, S7' multi-Department, and
the M21 BSD path-matrix). Locks in as the default for any
multi-hypothesis bug hunt going forward.

## 2026-05-19 — M28 cluster Department patterns (rheoTool sweep loader, PBS walltime trap, one-shot lifecycle)

Three durable Department-level patterns from the M28 cluster
(M28/b/c/d/e/f + rheoTool sweep + Liu-check + M26b + synthesis).

### Pattern 1 — rheoTool Cd-sweep CSV conventions

The new sweep aggregator `bench/viscoelastic_logfv/RHEOTOOL_CD_SWEEP_M28.csv`
sets a column layout that any future rheoTool-vs-Kraken cross-check
should reuse :

```
beta, Re, R, Wi, Cd_total, case_dir, n_cells, convective_terms_active,
endTime_actual, residual_U_final, residual_p_final, residual_theta_final,
steady_state_marker, notes
```

Key conventions :

- `Cd_total` = pressure + viscous + polymer (cylinder surface
  integral), NOT a Cd_s+Cd_p decomposition. rheoTool does not
  natively split.
- `steady_state_marker ∈ {flat, drifting}` is set by hand by the
  Department after reading `Cd.txt` over the last decade. "flat" means
  ≥ 9 digits stable ; "drifting" means visible motion at the 5th
  decimal (typical for Wi ≥ 1 at endTime = 10).
- `convective_terms_active = yes/no` documents `system/fvSchemes`
  status of `div(phi, U)`, `div(phi, theta)`, `div(phi, tau)`. A
  `none` on any of these falsely zeros the convective term and is the
  #1 silent-failure mode for rheoTool reference runs.
- `endTime_actual` is the converged endTime, NOT the
  `controlDict::endTime` knob (may differ if the run was extended
  after observation of slow drift). Always cross-check against
  `log.rheoFoam` last `Time = ...`.

The Wi=1.0 case is the only one that's not 12-digit-flat at
endTime=10 ; document this and any future sweep should extend
Wi=1.0 to endTime ≥ 20 if a < 0.1 Cd absolute target is needed.

### Pattern 2 — Aqua PBS walltime trap before maintenance windows

`gpu_batch_exec` queue defaults to 24 h walltime. The day before a
scheduled maintenance, jobs requesting more than the remaining
wall-clock are **held with no error** — `qstat` shows the job in Q
indefinitely, no output until manually killed. Symptoms :

- `qstat` shows `comment = Not Running: Job would cross dedicated
  time boundary`.
- The job sits in Q for hours despite empty queue.
- No way to detect this from the PBS submission itself.

**Mitigation** : explicit `qsub -l walltime=HH:MM:SS` per-job override
that's SHORTER than time-to-maintenance. The M28c session burned
30 minutes on this : jobs 21579942/943/944 had to be `qdel`'d and
resubmitted as 21579957/958/959 with explicit 15/30/60 minute
walltimes. Department briefs that wrap Aqua PBS scripts should
include a sanity check (`qstat -u USER` + look for held jobs) BEFORE
walking away from a submission. Better : check Aqua's maintenance
schedule (`/etc/motd` or admin announcements) before submitting any
job > 1 h.

### Pattern 3 — Department one-shot lifecycle (no callback for background watchers)

The M28-cluster session was orchestrated with **multiple Departments
in flight simultaneously** (M28 main, M28b, M28d, M28e, M28f, M28c,
rheoTool sweep, Liu-check, M26b, M28-synthesis, M29-tau-compare).
Some of these depend on Aqua jobs returning, which can take 1-30
min ; Department processes are one-shot — they execute their brief,
write artefacts, return a report, and TERMINATE. There is no
mechanism for a Department to "sleep until job lands, then continue".

**Anti-pattern observed** : briefs that say "wait for the job to
land then write the verdict". The Department finishes the prose
plan, exits, and the job lands later with no one to write the
verdict. Two costs : a re-spawn is required (extra context window),
and inter-Department artefacts get out of order.

**Pattern** : structure briefs as either

1. **Synchronous in-Department** : fetch the data WITHIN the brief
   (via `ssh aqua cat ...` on already-finished jobs), do the analysis,
   write verdict. M28-synthesis (this one) used this pattern.
2. **Two-stage Boss orchestration** : Department A SUBMITS the job,
   Boss waits at top-level (via Bash polling or qstat watch), Boss
   spawns Department B with the SUMMARY.csv path baked into the brief.
   M28c used this pattern.

Never write a brief that has a Department do BOTH the submission AND
the verdict — the qsub returns immediately and the Department exits
before the job lands. (This is in addition to the discipline gate of
[[feedback_orchestrator_discipline]] — same root cause, different
manifestation.)

**Why**: locks lessons that will recur for any future Aqua-backed
viscoelastic Department cluster. Skip cycles by reading this first.

## 2026-05-19 — Cd-gap missions must START with wall decomposition

Any Department brief that aims to attribute a `Cd_kraken − Cd_reference`
gap to a code component MUST include the following step BEFORE any
volume-field comparison:

1. **Extract wall ring** of fluid cells adjacent to the solid (Kraken:
   `is_solid` neighbours ; rheoTool: cells at `radius ≈ R` or
   boundary-patch values directly).
2. **Build circumferential profiles** of `(p, τ_s, τ_p)` along
   θ ∈ [0, 2π] with ~36-72 bins.
3. **Integrate** to get `Cd_pressure + Cd_solvent + Cd_polymer` per
   case (Kraken_M29b, Kraken_candidate, rheoTool_reference).
4. **Compare component-by-component** and **azimuth-bin-by-bin**.

Only after this is done may a volume L2_rel / peak comparison enter the
analysis — as a corroborating signal, not as the primary one.

**Why**: 2026-05-19 the M29c-v2 patch appeared to close 56% of the
M29b residual gap (volume L2_rel(τ_p) −38%, peak τ_xx +68% toward
rheoTool, Cd_total 110 → 116). The wall decomposition exposed this as
a cancellation of opposite-sign errors: Cd_polymer over-shot by 50%
(+6.55), Cd_pressure under-shot by 12% (+10.22), net +3 Cd. Six
missions had been spent on the wrong attribution because the M28+M29
verdicts used volume L2_rel as the Cd-gap proxy without ever
integrating on the wall.

**How to apply**: in any Cd-gap mission brief, the FIRST exit
criterion is the wall-decomposition table, not the L2_rel table.
The reusable harness lives at
`bench/scratch/m29c_wallstress/run_wallstress.jl` (produced for
this mission) and includes the rheoTool wall-patch parser, the
Kraken wall-ring extractor with `is_solid` adjacency, and the
72-bin azimuthal integrator. Copy + adapt, do not re-derive.

## 2026-05-19 — Stream idle timeout warning ; verify outputs exist

A Department mission that returns claiming success after a stream
idle timeout may have **authored** a 800-line script but **never
executed** it. Before accepting any "numbers" from a previous
Department, the next Department MUST verify:

- The output CSVs/files claimed by the brief exist on disk.
- The `.engineer_logs/<mission>_*.log` exists with a successful
  completion line.
- The verdict markdown contains actual numbers, not pending
  placeholders.

If any of the three is missing, the prior mission's "numbers" are
**not real**. Re-run the script (it's already written) ; do NOT
trust tool-use count as evidence of execution.

**Why**: 2026-05-19 M29c-tau-decompose Department wrote 862 LOC of
post-processing then idled out. A follow-up mission found no CSVs,
no log, no real numbers — the verdict file was an honest GAP marker.
A second Department executed the prepared script in 3 s and one
minimal bug fix.

**How to apply**: Department brief should include a "verify-prior"
preamble whenever a previous mission may have hit timeout. The check
is three `ls` commands.

## 2026-05-20 — Wall-ring integrator frame consistency check

Any Department brief that computes a wall integral on a rasterised
cylinder MUST verify the index↔physical convention BEFORE trusting
the resulting Cd decomposition. Specifically:

- The Kraken `precompute_q_wall_cylinder` rasterises using
  **`(i−1, j−1) ↔ physical`** (`src/kernels/li_bb_2d.jl:277`).
- A snapshot stores `cx_lbm, cy_lbm` in **physical** units.
- Therefore the **rasterised cylinder centre in lattice indices**
  is `(cx_phys + 1, cy_phys + 1)`.
- A ring integrator that sets `dx = i − cx_phys` (the `:phys` frame)
  is **1 LU off** in EVERY ring cell. For per-component Cd this
  produces a systematic +24 % bias on Cd_polymer (polymer stress is
  steepest in the wall layer) and a −2.2 % bias on Cd_total. The
  bias on Cd_pressure is small (0.03 %).
- The correct formula is `dx = (i−1) − cx_phys` (the `:idx` frame).

**Why**: 2026-05-20 the M30 Phase 0c verdict (front/rear K/rT
asymmetry 0.58 vs 0.28) was correct in DIRECTION but its absolute
per-component numbers were biased. The retrospective fix flipped
the "M29b matches Cd_polymer rheoTool to 0.05" claim into "M29b
under-predicts Cd_polymer by 15-20 %", which re-ranked H2/H3 from
DEMOTED to co-secondary.

**How to apply**: any wall-decomp brief must list a sanity check
of "compute `Cd_total_x` by ring integration and reconcile within
0.5 % of the driver's stored `Cd_kraken`". If reconciliation fails
by more than 1 %, the frame is off — debug before trusting the
per-component decomposition. The driver itself is the ground truth
(`xw = (i−1) + q_w·c_q` is the correct convention).

## 2026-05-20 — Adversarial Claude+Codex pattern: 3rd documented win

The M31 frame audit was the 3rd documented case where the Department's
first-pass Claude analysis would have shipped the wrong conclusion
without Codex's independent vote on the critical question (Q4: which
frame is physically correct). All three wins follow the same shape:

1. First pass (Claude alone) finds an "obvious" answer that aligns
   with surface evidence (here: snapshot stores `:phys` coordinates,
   so `:phys` must be correct).
2. Codex on same brief finds a deeper invariant from the source code
   (here: rasterisation uses `(i−1, j−1)` ↔ physical, so the
   physically meaningful centre in indices is `cx_phys + 1`).
3. Synthesis reveals Claude reversed; Codex's reasoning lands.

When the next Department mission involves: (a) hypothesis ranking,
(b) algebraic verification of a published formula in code,
(c) frame/convention question, or (d) anything where two engineers
might "obviously" reach different answers — invoke the adversarial
pattern by default ([[feedback_adversarial_default_uncertain]]).
The cost is one extra `bash run-engineer.sh` invocation and ~5
minutes of compute. The benefit is shipping the correct
interpretation 100 % of the time so far.
