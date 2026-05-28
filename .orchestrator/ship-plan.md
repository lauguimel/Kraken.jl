# SHIP PLAN — Kraken.jl multiphysics MVP

**Mission ID**: KRK-SHIP-001
**Drafted**: 2026-05-28 by Department (PLAN mission from Boss)
**Companion doc**: `/Users/guillaume/Documents/Recherche/Kraken.jl/.orchestrator/mandate.md`

This is the minimum-session roadmap to ship Kraken.jl with:
1. Working multiphysics scope = **thermal + viscoelastic + Cartesian** (3 axes).
2. **Tri-track docs** = `.krk`/human + Julia API + LLM module map.
3. **Per-module RheoTool benchmark** (numerical match, not just qualitative).
4. **LU/real-units module** to end the chronic lattice-units rabbit holes.

The plan is **proposal**: each mission below must be dispatched by the Boss as its own orchestrator Department brief. This document is read-only context for those briefs.

---

## §1 Current state audit

### §1.1 Branch / worktree state (from `branch-audit.sh --with-github`, 2026-05-28)

| Branch | Days | Ahead/Behind | Files diff | Worktree | Status | Notes |
|--------|------|--------------|-----------|----------|--------|-------|
| `main` | 14 | — | — | (none) | trunk | CI: failure (verify cause) |
| `release/v0.1` | 14 | 2/1 | 7 | `Kraken-main/` | stable | Recent fixups only |
| `slbm-paper` | 3 | **294/10** | **500** | `Kraken.jl/` (HEAD) | active | Paper companion; large divergence |
| `dev-viscoelastic` | 3 | **236/10** | **863** | `Kraken.jl-viscoelastic/` | active | **Richest worktree** — has thermal, VE, krk parser, rheotool benches |
| `feat/amr-port-sr` | 3 | 219/10 | 322 | `Kraken.jl-amr-port-sr/` | active | AMR state-resident port |
| `docs/module-architecture` | 6 | 279/10 | 494 | `Kraken.jl-docs-arch/` | active | **PR #6 open** — doc track parent |
| `dev/fvfd-core` | 12 | 7/0 | 12 | `Kraken.jl-fvfd-core/` | active | Recently rebased on main |
| `dev/axisymmetric-true-lbm` | 14 | 6/0 | 51 | (none) | active | No worktree — needs one if work resumes |
| `dev/v0.2-architecture` | 14 | 1/0 | 44 | (none) | active | **Mandate §8: confirm/retire** (superseded by v0.3?) |
| `dev/v0.3-campaign` | 21 | 214/10 | 315 | `Kraken.jl-v0.3-campaign/` | active | v0.3 release work |
| `lbm` | 42 | 0/10 | 0 | (none) | merged-can-delete | Mandate §8 confirm action |
| `refinement-patches-dev` | 43 | 0/31 | 0 | (none) | merged-can-delete | Likely safe to retire |
| `dev/refinement-perf-opts` | 10 | 13/10 | 14 | `Kraken.jl-refinement-perf/` | prunable | Confirm before remove |
| `dev/kraken-e-fvfd-blocks` | 11 | 10/0 | 40 | `Kraken.jl-kraken-e-blocks/` | prunable | Confirm before remove |
| `probe/amrd-golden-cylinder` | 10 | 215/10 | 320 | `Kraken.jl-amrd-golden/` | probe | Diagnostic — keep per mandate |
| `diag-amrd-symptom-pre-gate-20260512` | 18 | 249/10 | 321 | (none) | probe | Diagnostic snapshot — keep |
| `audit/modularity-performance-axisym` | 28 | 0/1 | 0 | (none) | probe | Audit-only — keep |
| `backup-amrd-before-symptom-reset-20260512` | 17 | 251/10 | 335 | (none) | archive | Recovery-only — keep |
| `deforestation` | 37 | 5/10 | 20 | `Deforestation/kraken/` | active | Cross-project consumer |
| `gh-pages` | 80 | 2/185 | 0 | (none) | auto | Auto-deploy, never edit |

**Strategic read** (cross-referenced with `mandate.md §5`):
- The **viscoelastic worktree** is the de-facto multiphysics integration point today — it has thermal kernels/drivers, VE rheology, the `.krk` parser, and the RheoTool benchmark suite, all in one place.
- `slbm-paper` and `dev-viscoelastic` have diverged heavily from `main` (236-294 commits). A near-term merge campaign is necessary BEFORE multiphysics work can ship.
- `dev/axisymmetric-true-lbm` has no worktree — work has paused. The mandate axisymmetric goal depends on resuming this.
- `dev/v0.2-architecture` and `lbm` and `refinement-patches-dev` are mandate-confirmed retirement candidates — close before shipping (cosmetic but signals readiness).

### §1.2 Multiphysics-axis completeness

Inventory based on top-of-`src/` ls per worktree (DEEP CODE READ NOT PERFORMED — flagged `_audit needed_` where assumptions made).

| Axis | Lives in (branch / dir) | Last activity | State | Evidence |
|------|------------------------|---------------|-------|----------|
| **Thermal (Boussinesq)** | `dev-viscoelastic` — `src/kernels/thermal_2d.jl`, `thermal_3d.jl`, `fused_thermal_2d.jl`, `src/drivers/thermal.jl`, `src/refinement/thermal_refinement.jl`, `examples/rayleigh_benard.krk`, `examples/heat_conduction.krk` | 2026-05-25 | **functional** (driver + kernels + refinement + .krk example all present; benchmark match _audit needed_) | `find` for `*thermal*` yields 5 source files + 2 .krk fixtures |
| **Viscoelastic (Oldroyd-B, FENE-P, log-conformation)** | `dev-viscoelastic` — `src/rheology/{models, viscosity, strain_rate, linalg, linalg_3d}.jl`, `src/drivers/viscoelastic*.jl` (5 drivers), `src/fvfd/*` (log-FV operators), `bench/viscoelastic_*` (extensive) | 2026-05-25 | **partial→functional** — M59-M62 anchored on M61 diffusive scaling; M62b planner spec written but `src/sim_planner/` **does NOT yet exist** (verified by ls) | M62b `IMPL` brief is **planned, not started**; bench suite suggests Cd/strain match in-progress |
| **Cartesian (uniform-grid 2D/3D base path)** | `dev-viscoelastic` and `slbm-paper` — `src/kernels/`, `src/lattice/{d2q9,d3q19,lattice}.jl`, `src/drivers/{basic,cavity_driver_2d,...}.jl`, `examples/{cavity,poiseuille,couette,taylor_green,cavity_3d}.krk` | 2026-05-25 | **functional** (this is the production trunk path — Newt + VE + thermal all use it) | 5 canonical Cartesian .krk examples present; no separate `src/cartesian/` because Cartesian IS the default |

**Key interpretation**: "Cartesian" in the user's spec means **the uniform-grid base path, working in 2D and 3D**, NOT a separate module. It is in best shape — used by every other axis. The shipping risk is in (a) thermal not being numerically benchmarked, and (b) viscoelastic not yet having the planner that prevents LU/non-dim errors.

### §1.3 M62b cross-reference — the LU/real-units pain point

The `.engineer_brief_M62b_IMPL.md` (in `Kraken.jl-viscoelastic/`) is the closest existing artefact to the user's "LU/real-units module" pain. It:

- Documents the **chronic pattern**: M48 (silent fixture toggle flip), M59-M61 (acoustic-scaling U-shape artifact), all rooted in hand-coded ν, λ, τ, u arithmetic in scratch files (`scratch/M48_R10_30_50_post_revert.jl`).
- Proposes `Kraken.SimulationPlanner` at `src/sim_planner/`: a single forward (`compile`) + reverse (`audit`) API that owns Re/Wi/β/R_LU → (τ, u, ν_s, ν_p, λ, max_steps) conversion, with strict/lenient modes, BSD-aware τ, TRT magic window, F32 floor, and physics-extensible registry.
- Phase 1 scope: Newtonian + VE complete; thermal/GNF/multiphase as **abstract stubs** in `physics/` (so Phase 2 = new file, zero edits to Phase 1).
- Includes 4 audit layers: `lattice_units`, `stability_cone`, `stl_audit`, `bc_consistency`.

**Status**: **PLANNED but NOT STARTED**. `src/sim_planner/` does not exist in the viscoelastic worktree (verified by ls). The design is converged (round 2 adversarial Claude+Codex settled). Phase 1 implementation is a single Codex mission, scoped at ≤6h.

**Implication for the ship plan**: M62b IS the LU/real-units module the user is asking for. Renaming / re-scoping it for the multiphysics ship is the most efficient path — see §5 for the merged design seed.

---

## §2 Target shipping scope (the "done" definition)

### §2.1 Per-axis acceptance criteria

#### Thermal (Boussinesq)
- **Acceptance**: `examples/rayleigh_benard.krk` runs end-to-end on CUDA backend and reproduces a published benchmark (Ra = 1e4 and Ra = 1e5 natural-convection-in-square-cavity, e.g. De Vahl Davis 1983 or Wan et al. 2001) within **1%** on Nu (Nusselt) and within **2%** on max(|u|).
- **Validation ladder**: analytical 1D conduction patch → coarse 32² cavity Ra=1e3 → benchmark 128² Ra=1e4 → Ra=1e5.
- **RheoTool comparison**: run the SAME case in OpenFOAM's `buoyantBoussinesqPimpleFoam` (RheoTool extends OpenFOAM 9; the Newtonian buoyant solver is in OF base), match Nu within 2%.

#### Viscoelastic (Oldroyd-B baseline; FENE-P stretch goal)
- **Acceptance**: M61 anchor reproduced — `cylinder_wi1.0_R10_shrunk15R`, `R30`, `R50` cases give Cd within **5%** across R_LU values (proves the diffusive-scaling fix holds) AND match the corresponding RheoTool case (`bench/rheotool/cylinder_oldroydb_log_re1_wi01/` and analogues) within **1%** on Cd at steady state.
- **Validation ladder**: 1D Poiseuille (analytic, λ-relaxation only) → 2D Couette polymer-extra-stress patch → cylinder Wi=0.5 (low) → Wi=1.0 (M61 anchor) → contraction 4:1 (Phan-Thien benchmark).
- **RheoTool comparison**: at least 3 cases — cylinder Re=1 Wi=0.1, cylinder Re=1 Wi=1.0, cavity Re=0.01 De=1 β=0.5. Pair with `sim-rheotool` skill on Engineer side.

#### Cartesian uniform-grid (2D + 3D)
- **Acceptance**: `examples/cavity.krk` (2D) + `examples/cavity_3d.krk` (3D) reproduce Ghia & Ghia & Shin 1982 (2D) and Yang et al. 1999 (3D) lid-driven cavity at Re=100, 400, 1000 within **1%** on centreline u, v profiles.
- **Validation ladder**: Taylor-Green decay rate analytic match → Couette analytic → cavity Re=100 coarse → cavity Re=1000 production.
- **RheoTool comparison**: trivial since cartesian uniform is the default OpenFOAM mesh — match `icoFoam` cavity Re=100/400/1000.

### §2.2 Tri-track docs criteria

Three deliverable surfaces — every public module must appear in all three:

**Track A — Human docs (GitHub Pages, narrative)**
- `docs/src/index.md` — landing page with "run your first .krk in 5 minutes" tutorial.
- `docs/src/users/krk-reference.md` — `.krk` DSL reference (every keyword, every block type, every physics).
- `docs/src/users/examples/` — one walkthrough per axis: `thermal-rayleigh-benard.md`, `viscoelastic-cylinder.md`, `cartesian-cavity.md`.
- `docs/src/users/benchmarks/` — RheoTool comparison report per axis with side-by-side plots.
- **Build verification**: `julia --project=docs/ docs/make.jl` produces a clean site, no broken links.

**Track B — Julia API docs (Documenter.jl docstrings)**
- Every exported symbol in `src/Kraken.jl` has a Documenter-style docstring with at least one runnable `# Example` block.
- `docs/src/api/<module>.md` per module (auto-generated from `@autodocs`).
- **Build verification**: `julia --project=docs/ -e 'using Documenter; doctest(Kraken)'` passes with zero failures.

**Track C — LLM module-implication map**
- `docs/agent/<module>-implication.md` per module — a structured markdown file with these mandatory sections:
  - `## Public surface` — exported symbols + 1-line semantics each.
  - `## Reads from` — which other modules' state it consumes (read-only deps).
  - `## Writes to` — what it mutates / produces.
  - `## Backend constraints` — GPU-safe? allocates? dispatches?
  - `## Failure modes` — known antipatterns / past bugs / footguns.
  - `## Touch order` — for a new bug in this module, which files to inspect in what order.
- **Format**: YAML frontmatter + markdown body. Spec doc at `docs/spec/llm-implication-map-v1.md`.
- **Build verification**: a CI script lints every `docs/agent/*-implication.md` against the schema.

**Tri-track per-module checklist** (a module is "shipped" only when all three exist and validate):

| Module | Track A | Track B | Track C |
|--------|---------|---------|---------|
| `io/krk` (`.krk` parser) | `docs/src/users/krk-reference.md` | `docs/src/api/io-krk.md` | `docs/agent/krk-implication.md` |
| `geometry/` | `docs/src/users/geometry.md` | `docs/src/api/geometry.md` | `docs/agent/geometry-implication.md` |
| `lbm/` (stream+collide) | (referenced in benchmarks) | `docs/src/api/lbm.md` | `docs/agent/lbm-implication.md` |
| `physics/newtonian` | (referenced in cartesian example) | `docs/src/api/physics-newtonian.md` | `docs/agent/physics-newtonian-implication.md` |
| `physics/viscoelastic` | `docs/src/users/examples/viscoelastic-cylinder.md` | `docs/src/api/physics-viscoelastic.md` | `docs/agent/physics-viscoelastic-implication.md` |
| `physics/thermal` | `docs/src/users/examples/thermal-rayleigh-benard.md` | `docs/src/api/physics-thermal.md` | `docs/agent/physics-thermal-implication.md` |
| `units/` (the LU↔real module) | `docs/src/users/units-and-non-dim.md` | `docs/src/api/units.md` | `docs/agent/units-implication.md` |
| `bc/` | `docs/src/users/boundary-conditions.md` | `docs/src/api/bc.md` | `docs/agent/bc-implication.md` |
| `backend/` | (one-line note) | `docs/src/api/backend.md` | `docs/agent/backend-implication.md` |

### §2.3 LU/real-units module acceptance
- A single import `using Kraken.Units` exposes `compile(physics=:viscoelastic, Re=1, Wi=1, beta=0.59, R_LU=30, ...)` → returns a typed `SimulationPlan` with all LU + non-dim + real-units fields populated.
- Round-trip identity: `audit(driver_kwargs(compile(args)))` reproduces the input `SimulationPlan` to 1e-12.
- Catches M59-B regression: `audit(u_mean=0.005, R_LU=50, ...)` warns about τ=1.25 falling outside TRT magic window.
- Zero changes to existing drivers required at module-introduction time — drivers consume `driver_kwargs(plan)` opt-in.

---

## §3 Mission graph (minimum-session ship)

Missions are designed to be **independently dispatchable** with the orchestrator pattern. Dependencies (`deps:`) gate order.

### Phase A — Foundation (parallelizable on §3.1 + §3.2)

#### M1 — Modular architecture audit
- **Type**: Explore / Plan
- **Objective**: Fill in `mandate.md §6` rows (the `_audit needed_` cells). For each `src/**/*.jl` file in `dev-viscoelastic` (richest worktree), classify primary concern (geometry / mesh / BC / physics / LBM / backend / IO / driver) and flag mixed-concern files.
- **Allowed edit zones**: `mandate.md §6` (Boss-only — Department returns a patch proposal); `bench/scratch/m1_module_audit.md` (new) for the full classification table.
- **Exit criterion**: `grep -c "_audit needed_" .orchestrator/mandate.md` drops to 0; `bench/scratch/m1_module_audit.md` lists every src/ file with a single primary concern label.
- **Runner**: `claude-subagent` with `kraken-codebase-map` loaded (read-only).
- **Deps**: none.
- **Effort**: 1 session.

#### M2 — Branch cleanup & merge plan
- **Type**: Plan
- **Objective**: For each `mandate.md §8` open question on a branch (`lbm`, `dev/v0.2-architecture`, `refinement-patches-dev`, the 3 PRUNABLE worktrees), produce a per-branch action (retire / keep / merge-first) with the destructive command and rationale. Produce the merge order for `slbm-paper` ↔ `dev-viscoelastic` ↔ `main` taking into account the multiphysics ship requirement (viscoelastic must NOT be lost in any merge).
- **Allowed edit zones**: `bench/scratch/m2_branch_cleanup_plan.md` (new); `mandate.md §4` (ADR additions, Boss-applied).
- **Exit criterion**: a numbered action list, each with the exact `git` command, and a Boss-readable rationale per action. The user confirms each destructive action separately.
- **Runner**: `claude-subagent` with `kraken-architect` + `git-audit` loaded.
- **Deps**: none.
- **Effort**: 1 session.

### Phase B — LU/real-units module (the rabbit-hole-killer)

#### M3 — LU/real-units module spec (`src/units/`)
- **Type**: Plan
- **Objective**: Take the M62b design and adapt scope to mandate §1 (multiphysics, not VE-only). Produce a spec doc `docs/spec/units-v1.md` covering: API surface, type hierarchy (`AbstractPhysicsSpec`, `LBMUnits{T}`, `SimulationPlan{T}`), physics registry pattern, forward/reverse semantics, strict-vs-lenient mode, error semantics, and the extension contract (how thermal adds itself in Phase 2 with zero Phase 1 edits).
- **Allowed edit zones**: `docs/spec/units-v1.md` (new); `mandate.md §4` (ADR entry: "units module spec frozen 2026-XX-XX"); `mandate.md §6` (new row: `units → src/units/`).
- **Exit criterion**: spec doc contains all 6 sections (API / types / registry / semantics / errors / extension); ADR entry committed.
- **Runner**: `claude-subagent` (reasoning; no Julia execution).
- **Deps**: M1 (so the audit informs where `units/` sits in the tree).
- **Effort**: 1 session. **Note**: the M62b brief already covers ~80% of this — most work is generalisation + renaming `sim_planner` → `units` (or keeping `sim_planner` if Boss prefers the M62b name).

#### M4 — LU/real-units module implementation (Phase 1: Newt + VE)
- **Type**: Implement
- **Objective**: Implement the spec from M3 in `src/units/` (or `src/sim_planner/` per M62b). Phase 1 = Newtonian + Viscoelastic concrete; thermal + GNF + multiphase as abstract stubs (raise `NotImplementedError`).
- **Allowed edit zones**: `src/units/**` (new), `src/Kraken.jl` (add `include` + `export`), `test/test_units.jl` + `test_units_audit.jl` + `test_units_stability.jl` + `test_units_krk.jl` (new), `test/runtests.jl` (add entries), `bench/scratch/m4_impl_notes.md` (audit log).
- **Exit criterion**: the M62b IMPL exit shell block (the 4 `julia --project=. test/test_units_*.jl` runs + the M61 R∈{10,30,50} numerical reproduction + the `.krk` round-trip + the M59-B audit U-shape detection). Concretely: `julia --project=. -e 'using Pkg; Pkg.test(; test_args=["units"])'` green.
- **Runner**: `codex` with `kraken-branch-governor` + `kraken-codebase-map` loaded.
- **Deps**: M3.
- **Effort**: 1 session (≤6h Codex per M62b time budget).

#### M5 — LU/real-units Phase 2: thermal physics module
- **Type**: Implement
- **Objective**: Add `src/units/physics/thermal.jl` implementing `ThermalBoussinesqSpec` with Re/Pr/Ra fields, `_build_spec`, `_compile_with_spec`, stability predicates, and BC compatibility entries. Verify zero edits to Phase 1 files.
- **Allowed edit zones**: `src/units/physics/thermal.jl` (replace stub), `test/test_units_thermal.jl` (new). NOT Phase 1 files.
- **Exit criterion**: thermal `compile(physics=:thermal_boussinesq, Re=1e3, Pr=0.71, Ra=1e5, R_LU=128, ...)` returns a valid plan; bit-identical viscoelastic plan vs M4 (proves no Phase 1 mutation); thermal driver consumes `driver_kwargs(plan)` and the Rayleigh-Benard `.krk` example uses it.
- **Runner**: `codex` with `kraken-branch-governor` loaded.
- **Deps**: M4.
- **Effort**: 1 session.

### Phase C — RheoTool benchmark matches (one per axis)

#### M6 — RheoTool benchmark: Cartesian (icoFoam cavity)
- **Type**: Validate
- **Objective**: For `examples/cavity.krk` at Re=100, 400, 1000 (2D) and `examples/cavity_3d.krk` at Re=100 (3D), run the equivalent OpenFOAM `icoFoam` case, extract centreline u/v profiles, compare to Kraken output. Produce `docs/src/users/benchmarks/cartesian-cavity.md` with side-by-side plots and a table of L2 error per Re.
- **Allowed edit zones**: `bench/cartesian_rheotool/cavity_re100/`, `cavity_re400/`, `cavity_re1000/`, `cavity_3d_re100/` (new OF case dirs); `docs/src/users/benchmarks/cartesian-cavity.md` (new).
- **Exit criterion**: max L2 error on centreline u, v < 1% vs Ghia 1982 AND vs `icoFoam`; benchmark page renders in the Pages site.
- **Runner**: `claude-subagent` with `sim-openfoam` + `sim-rheotool` skills loaded; `pbs` for HPC dispatch if local Docker too slow.
- **Deps**: M4 (so cases use planner-validated LU).
- **Effort**: 1-2 sessions.

#### M7 — RheoTool benchmark: Thermal (buoyantBoussinesqPimpleFoam)
- **Type**: Validate
- **Objective**: For `examples/rayleigh_benard.krk` at Ra ∈ {1e3, 1e4, 1e5}, run the equivalent OpenFOAM `buoyantBoussinesqPimpleFoam` case, extract Nu and max|u|, compare to Kraken. Produce `docs/src/users/benchmarks/thermal-rayleigh-benard.md`.
- **Allowed edit zones**: `bench/thermal_rheotool/rayleigh_benard_ra1e3/`, `_ra1e4/`, `_ra1e5/` (new); `docs/src/users/benchmarks/thermal-rayleigh-benard.md` (new).
- **Exit criterion**: Nu within 1%, max|u| within 2% vs Wan et al. 2001 AND vs OF reference at all 3 Ra; benchmark page in Pages.
- **Runner**: `claude-subagent` with `sim-openfoam` + `sim-rheotool` + `pbs` skills.
- **Deps**: M5 (thermal physics in planner).
- **Effort**: 1-2 sessions.

#### M8 — RheoTool benchmark: Viscoelastic (rheoFoam cylinder)
- **Type**: Validate
- **Objective**: For `bench/rheotool/cylinder_oldroydb_log_re1_wi01/` and analogous cases at Wi=0.5, Wi=1.0, run RheoTool (already-staged tutorial cases) AND the corresponding Kraken `.krk` cylinder case. Compare Cd (steady-state) and N1 max in wake. Produce `docs/src/users/benchmarks/viscoelastic-cylinder.md`.
- **Allowed edit zones**: `bench/viscoelastic_rheotool_comparison/` (new — only the comparison scripts/plots; the rheotool case dirs already exist); `docs/src/users/benchmarks/viscoelastic-cylinder.md` (new).
- **Exit criterion**: Cd match within 1% at Wi ∈ {0.1, 0.5, 1.0}; N1 max in wake within 5%; benchmark page in Pages.
- **Runner**: `claude-subagent` with `sim-rheotool` + `pbs` skills (HPC for the longer VE runs).
- **Deps**: M4 (VE in planner); M1 (to know where existing rheotool case dirs are referenced from).
- **Effort**: 2 sessions (VE is the most numerically delicate; multiple iterations likely).

### Phase D — Tri-track docs

#### M9 — LLM-implication-map spec
- **Type**: Plan / Produce
- **Objective**: Write `docs/spec/llm-implication-map-v1.md` per §2.2 Track C requirements. Includes a JSON-Schema for frontmatter, the 6 mandatory section headers, an example doc (`docs/agent/units-implication.md` as the reference), and a lint script `scripts/lint-implication-map.sh`.
- **Allowed edit zones**: `docs/spec/llm-implication-map-v1.md` (new); `docs/agent/units-implication.md` (new, as reference example); `scripts/lint-implication-map.sh` (new); `mandate.md §4` ADR entry.
- **Exit criterion**: `bash scripts/lint-implication-map.sh docs/agent/units-implication.md` passes; spec doc reviewed by user.
- **Runner**: `claude-subagent`.
- **Deps**: M3, M4 (so the reference doc has real content to describe).
- **Effort**: 1 session.

#### M10 — Tri-track docs build pipeline
- **Type**: Implement / Produce
- **Objective**: Wire `docs/make.jl` (Documenter.jl) to build Track A + Track B + Track C in one pass, with the linter from M9 gating Track C. Add a CI job (`.github/workflows/docs.yml`) that fails on any broken link or schema violation. Land it on `docs/module-architecture` branch (which has the active PR #6).
- **Allowed edit zones**: `docs/make.jl`, `docs/Project.toml`, `.github/workflows/docs.yml`, `docs/src/` (skeleton pages only — content comes in M11-M13).
- **Exit criterion**: `julia --project=docs/ docs/make.jl` produces `docs/build/` with no errors; CI green on `docs/module-architecture`.
- **Runner**: `codex` with `kraken-branch-governor` loaded.
- **Deps**: M9.
- **Effort**: 1 session.

#### M11 — Track A content: `.krk` reference + 3 axis examples
- **Type**: Produce
- **Objective**: Write `docs/src/users/krk-reference.md` (every `.krk` keyword/block with example) and the 3 axis tutorials (`thermal-rayleigh-benard.md`, `viscoelastic-cylinder.md`, `cartesian-cavity.md`) referencing the M6/M7/M8 benchmarks.
- **Allowed edit zones**: `docs/src/users/**` only.
- **Exit criterion**: pages build cleanly; user reads & signs off (no semantic gaps).
- **Runner**: `claude-subagent` (prose-heavy).
- **Deps**: M6, M7, M8, M10.
- **Effort**: 1-2 sessions.

#### M12 — Track B content: Documenter API docs
- **Type**: Implement / Produce
- **Objective**: Add Documenter docstrings to every exported symbol; create `docs/src/api/<module>.md` per module. Land via `kraken-branch-governor` discipline (one file at a time, ≤500 LOC cap respected).
- **Allowed edit zones**: docstring blocks in `src/**/*.jl` (touching only the `"""..."""` blocks above public defs, no behavioural changes); `docs/src/api/*.md` (new).
- **Exit criterion**: `julia --project=docs/ -e 'using Documenter; doctest(Kraken)'` passes; no `missing docstring` warnings on exported symbols.
- **Runner**: `codex` with `kraken-branch-governor` + `kraken-codebase-map` loaded.
- **Deps**: M1 (so we know what's exported and per which module), M10.
- **Effort**: 2-3 sessions (~50+ exported symbols expected).

#### M13 — Track C content: per-module LLM implication maps
- **Type**: Produce
- **Objective**: Write `docs/agent/<module>-implication.md` for every module in §2.2's tri-track checklist. Use the schema from M9; lint with the M9 script.
- **Allowed edit zones**: `docs/agent/**` only.
- **Exit criterion**: every checklist module has a valid implication doc; lint script green; CI job from M10 passes.
- **Runner**: `claude-subagent` with `kraken-codebase-map` loaded (this is the audit-derived doc track).
- **Deps**: M1, M9, M10, M12 (M12's docstrings are an input).
- **Effort**: 2 sessions.

### Phase E — Final integration

#### M14 — Merge campaign to `main`
- **Type**: Implement
- **Objective**: Execute the merge plan from M2. Land `dev-viscoelastic` (with planner, thermal, VE, units) and `docs/module-architecture` (with tri-track docs) on `main`. Run the validation ladder per axis post-merge.
- **Allowed edit zones**: conflict files only; no semantic changes during merge per mandate §7.
- **Exit criterion**: `git checkout main && julia --project=. -e 'using Pkg; Pkg.test()'` green; all 3 RheoTool benchmark pages render from main.
- **Runner**: `codex` with `kraken-branch-governor` loaded (per mandate §7.6).
- **Deps**: M2, M4, M5, M6, M7, M8, M10, M11, M12, M13.
- **Effort**: 1-2 sessions.

#### M15 — v0.2 / v0.3 release cut
- **Type**: Plan / Implement
- **Objective**: Tag `main` as `v0.2.0` (multiphysics MVP). Update `release/v0.1` ← `main` per mandate §5 backport rules for any critical bug fixes only. Publish GitHub Pages.
- **Allowed edit zones**: `Project.toml` (version bump), `CHANGELOG.md` (new), `release/v0.1` cherry-picks (Boss-confirmed per commit).
- **Exit criterion**: `v0.2.0` tag pushed; Pages site live; CHANGELOG covers M3-M14; mandate §5 branch map updated.
- **Runner**: `claude-subagent` for plan + Boss execution for the actual tag push (per mandate "never push without user confirmation").
- **Deps**: M14.
- **Effort**: 1 session.

### §3.X Total session estimate

| Path | Sessions |
|------|----------|
| **Realistic minimum** (parallel-friendly missions in parallel; M1+M2 same session; M6+M7+M8 partial overlap) | **9-10 sessions** |
| **Comfortable** (one mission per session, room for iteration on the VE benchmark) | **14-16 sessions** |

The benchmark missions (M6/M7/M8) are the largest single source of variance — VE numerical match at Wi=1 may need multiple Codex iterations. The docs missions (M11-M13) are heavy in volume but linearly parallelizable.

---

## §4 Mandate gap analysis

### §4.1 Non-negotiables vs delivery

| Mandate §3 non-negotiable | Delivered by | Gap? |
|---------------------------|--------------|------|
| **User-friendly LBM (non-Julia user can run a case)** | M11 (Track A docs) + existing `.krk` parser | None — already functional, just needs the docs |
| **`.krk`-driven canonical interface** | M3/M4 wire planner into `.krk`; M11 reference docs | None — but a `.krk` v1 spec is missing (mandate §8) — see §4.2 |
| **2D / 3D / axisymmetric** | 2D + 3D: M6/M7/M8 cover. **Axisymmetric: NOT in this ship plan** | **GAP** — `dev/axisymmetric-true-lbm` has no worktree, no recent activity. Either add an M16 (axisym revival) or scope-defer to a v0.3 ship plan. Recommend scope-defer with explicit ADR. |
| **GPU/CPU agnostic via KernelAbstractions** | Existing — implicit prerequisite; M12 should document the contract per module | Partial — no module dedicated to "backend dispatch" exists today (verified: no `src/backend/`). Backend choice lives inside drivers. **GAP** — needs an M17 audit / refactor mission (NOT in critical ship path; can defer). |
| **Multiphysique (Newt → VE → thermal → multiphase → AMR → STL)** | M4 (Newt+VE), M5 (thermal). **Multiphase, AMR, STL: NOT in this ship plan** | **Intentional scope cut**. Mandate §1 says "Newt → VE → multiphase → AMR → STL" — the user said "thermal + viscoelastic + Cartesian" for THIS ship. Confirm scope cut with Boss/user. |
| **Modular & pluggable (new physics = new module)** | M3/M4 enforce via PHYSICS_REGISTRY + extension contract; M5 demonstrates Phase 2 zero-edit extension | None — the extension contract is tested in M5 by construction |
| **Dual-track docs (human + LLM)** | M9/M10/M11/M12/M13 | None — fully covered |

### §4.2 ADRs the Boss should add (proposals for `mandate.md §4`)

1. **2026-05-28 — Adopt orchestrator three-layer pattern + this ship plan as KRK-SHIP-001 mandate-of-record.** Rationale: cf. mandate §1, this ship plan.
2. **2026-05-28 (proposed) — Multiphysics MVP scope = thermal + viscoelastic + Cartesian only; multiphase/AMR/STL/axisymmetric explicitly deferred to v0.3+.** Rationale: ship discipline.
3. **2026-05-28 (proposed) — `src/units/` (or `src/sim_planner/`, name TBD) is the single owner of LU↔non-dim↔real-units conversion. No driver may compute τ, ν, λ, u from Re/Wi/β by hand from this date forward.** Rationale: M48/M59-M61 rabbit-hole pattern.
4. **2026-05-28 (proposed) — LLM-implication-map format frozen by M9 spec; every public module MUST ship one before merge to main.** Rationale: mandate §3 doc-as-code non-negotiable.
5. **After M3 — `.krk` v1 spec frozen.** Mandate §8 open question gets resolved here.

### §4.3 Open questions blocking dispatch

These must be resolved BEFORE specific missions can start:

- **Before M2**: confirm `lbm` and `dev/v0.2-architecture` retirement (mandate §8). Otherwise M2 can't write the action list.
- **Before M3**: Boss decides — **`src/units/` or `src/sim_planner/`?** The M62b brief uses `sim_planner`; the mandate §6 calls it `units`. Pick one name and stick to it. _Department recommends `src/units/`_ — it matches mandate vocabulary and is more discoverable for new users (a "planner" sounds like a scheduler, "units" is unambiguous).
- **Before M5**: confirm thermal scope = **Boussinesq only** (no full energy equation, no compressibility). M62b stubs match this.
- **Before M6/M7/M8**: confirm local Docker vs HPC for benchmark runs. Affects mission duration estimates.
- **Before M14**: re-confirm merge target = `main` (not a new `dev/v0.2-multiphysics` integration branch). _Department recommends a new branch_ to avoid disrupting `release/v0.1` users until benchmarks are signed off.

Non-blocking but useful:
- **Kraken-AMR.jl relation** (mandate §8) — does not block this ship since AMR is out of scope, but the Boss should resolve before the v0.3 ship plan.

---

## §5 LU / real-units module — design seed

### §5.1 The rabbit-hole pattern (cited from M62b)

The M62b IMPL brief documents three concrete instances of the same anti-pattern:

- **M48** (fixture toggle flip): a silent change in a hand-coded `viscosity_polymer = (1-β) * ν` line caused a fixture to use different numerics than its sibling. Recovered via `scratch/M48_R10_30_50_post_revert.jl`.
- **M59 / M60** (acoustic-scaling U-shape): at fixed Re, sweeping R_LU under acoustic scaling (fixed u, ν = u·R/Re) produces a non-monotonic Cd curve because τ wanders out of the TRT magic window at large R. Three failed Codex sessions before the pattern was recognised.
- **M61** (diffusive-scaling fix): switching to diffusive scaling (fix τ, ν = (τ-0.5)/3, u = Re·ν/R) flattened the Cd curve — the correct fix, but only after the pattern was named.

**Root cause across all three**: τ, ν, u, λ, max_steps arithmetic was hand-coded per fixture, with no single owner, no validation, no audit trail. The same Reynolds number could mean three different LU configurations depending on which scratch file you read.

**Generic LBM pitfall** (independent of M62b): the LU↔non-dim mapping is non-injective without a scaling choice (acoustic vs diffusive vs BSD-aware). Users picking different conventions in different scripts produce numerically incomparable results.

### §5.2 Proposed scope

**The module owns**:
- All Re/Wi/Pr/Ra/Ma/De/Bi etc. ↔ LU (τ, ν_s, ν_p, u, λ, R_LU, dx_LU, dt_LU) conversion.
- Validation of LU inputs: τ ∈ [0.55, 1.5] TRT window, Ma ≤ 0.05, F32 floor τ ≥ 0.6, BSD-aware τ correction (fν = β + bsd·(1-β)).
- Per-physics extension via `AbstractPhysicsSpec` subtypes + `PHYSICS_REGISTRY`.
- Per-BC × physics stability predicates via `STABILITY_REGISTRY`.
- STL audit (q_wall distribution, curvature/R_LU, skewness).
- BC consistency (inlet/outlet/wall/periodic graph).
- Forward (`compile`) and reverse (`audit`) entry points.
- `.krk` binding (both mega-block and cross-reference syntaxes per M62b).
- Markdown + JSONL reports of every plan.

**Explicitly OUT of scope**:
- Time-stepping (drivers own this).
- Backend dispatch (`KernelAbstractions` machinery).
- Geometry primitives (lives in `src/geometry/`; the module CONSUMES `GeometryInfo`).
- File-format readers (lives in `src/io/`).
- Mesh refinement logic (lives in `src/mesh/`; module records refinement levels as `Vector{LBMUnits}`, doesn't generate them).
- ML/learned closures.

### §5.3 API sketch (high level)

```julia
# Public types
abstract type AbstractPhysicsSpec end
struct NewtonianSpec{T} <: AbstractPhysicsSpec   end  # Re
struct ViscoelasticSpec{T} <: AbstractPhysicsSpec end  # Re, Wi, beta, bsd_fraction, model, L_max
struct ThermalBoussinesqSpec{T} <: AbstractPhysicsSpec end  # Re, Pr, Ra (or Gr)
# stubs: PowerLawSpec, MultiphaseSpec, MHDSpec

struct LBMUnits{T}
    tau_hydro::T; nu_total_LU::T; u_LU::T
    R_LU::Int; Ma::T
    scaling::Symbol  # :acoustic | :diffusive | :auto-resolved
    # VE-specific (NaN/sentinel if not VE)
    nu_s_LU::T; nu_p_LU::T; lambda_LU::T
    # thermal-specific (sentinel if not thermal)
    alpha_LU::T; beta_thermal_LU::T
    # real-units conversion factors
    dx_real::T; dt_real::T; rho_real::T  # if user supplied
end

struct SimulationPlan{T}
    physics_spec::AbstractPhysicsSpec
    units::LBMUnits{T}
    bc::BCConfig
    geometry::GeometryInfo
    discretization::DiscretizationConfig
    refinement::Union{Nothing, Vector{LBMUnits{T}}}
    warnings::Vector{Issue}
    notes::Vector{String}
    audit_source::Symbol  # :compile | :audit
end

# Public functions
compile(; physics::Symbol, geometry, bc, refinement=nothing,
          backend=CPU(), T=Float64, strict=true, kwargs...) -> SimulationPlan{T}

audit(driver_kwargs::NamedTuple;
      physics::Symbol, geometry, bc,
      backend=CPU(), T=Float64, strict=false, kwargs...) -> SimulationPlan{T}

driver_kwargs(plan::SimulationPlan) -> NamedTuple
report(plan; io=stdout, format=:markdown) -> Union{Nothing, String}

# Extension API (for future physics)
register_physics!(sym::Symbol, T::Type)
register_stability!(bc::Type, phys::Type, predicate::Function)
register_bc_combo!(key::NTuple{4,Symbol}, status::Symbol)
```

**Error semantics**:
- `strict=true` (default for `compile`): any `fatal` Issue raises `ArgumentError` with the full failed plan attached for inspection.
- `strict=false` (default for `audit`): Issues collected into `plan.warnings`, plan still returned.
- `warn`-level Issues never raise; logged via `Logging.@warn`.
- Unknown physics symbol → immediate `error` with list of registered keys.
- Inconsistent BC combo → `error` (in `compile`) or `warn` (in `audit`).

### §5.4 Location in the architecture map

Per `mandate.md §6`:

```
src/units/                          ← THIS MODULE
├── Units.jl                        # module facade + types
├── physics_registry.jl
├── lattice_units.jl                # generic Re/R/τ/u/Ma math
├── stability_cone.jl
├── stl_audit.jl
├── bc_consistency.jl
├── report.jl
├── krk_binding.jl
├── audit_trail.jl
└── physics/
    ├── newtonian.jl                # complete
    ├── viscoelastic.jl             # complete (M4)
    ├── thermal.jl                  # complete (M5)
    ├── non_newt.jl                 # stub
    ├── multiphase.jl               # stub
    └── electromagn.jl              # stub
```

**Sibling modules it talks to** (read-only consumes):
- `src/geometry/` → `GeometryInfo`
- `src/io/krk/` → expression tree for `.krk` parsing
- `src/lattice/` → lattice topology constants

**Sibling modules that consume it**:
- `src/drivers/*.jl` → consume `driver_kwargs(plan)` opt-in.
- `src/io/krk/` → `compile()` is called when a `.krk` `Run` directive is dispatched.

### §5.5 Risks

| Risk | Mitigation |
|------|-----------|
| **Perf hit inside hot kernels** if conversion is called per timestep | Module is **compile-time only**. All `SimulationPlan` fields are `const` after construction; `driver_kwargs(plan)` returns a `NamedTuple` once before the time loop. Document this in `docs/agent/units-implication.md`. |
| **Type instability** from `physics_spec::AbstractPhysicsSpec` field | Concrete spec types are used inside drivers via `@nospecialize` boundary or via `function_barrier(plan)` that re-dispatches on `typeof(plan.physics_spec)`. Per M62b round-2 design. |
| **Float32 silent precision loss** for thermal/MP eventually | F32 floor τ ≥ 0.6 enforced in `compile`; thermal Bi number floor TBD in M5. |
| **Backward compatibility** with existing hand-coded drivers | Phase 1 lands the module WITHOUT changing existing drivers. Drivers opt in via `driver_kwargs(plan)`. Old `.krk` fixtures keep working. M14 merge campaign adds opt-in calls in the 3 axis examples. |
| **Naming collision** if a downstream package already defines `Units.compile` | Module is namespaced: `Kraken.Units.compile`. The bare `compile` is NOT exported. Users write `using Kraken.Units: compile`. |
| **`.krk` parser drift** if both mega-block and cross-ref syntax supported | Tested in M4 by parity test (case 15 in M62b validation list). |

---

_End of ship plan KRK-SHIP-001._

---

## §6 Locked decisions (Boss overlay, 2026-05-28)

The Department drafted §1–§5. The Boss subsequently locked these four decisions with the user; they OVERRIDE any conflicting wording above and feed into the §3 mission graph for dispatch.

| # | Decision | Detail |
|---|----------|--------|
| 1 | **LU/real-units module → `src/units/`** | NOT `src/sim_planner/` (M62b legacy name). M4 brief inherits M62b content with this rename. |
| 2 | **Thermal scope = Boussinesq only for ship-1** | Full thermal (variable rho/mu(T)/k(T) + dissipation) → ship-2. Department's original Boussinesq recommendation stands. M5 exit criterion = Rayleigh-Bénard Ra=1e5 match against `rheoHeatFoam/buoyantCavity` Newtonian baseline. |
| 3 | **Merge target = `dev/v0.2-multiphysics`** (new branch from `main`) | NOT `main` direct. Intermediate release-candidate; main merge happens only after `dev/v0.2-multiphysics` is fully green. Adjust M14 brief accordingly. |
| 4 | **Branch retirements approved** | `lbm`, `refinement-patches-dev`, `dev/v0.2-architecture` to be removed. M2 dispatchable. Per-branch confirmation still required at execution. |

### Implications on the mission graph (§3)

- **M2 unblocked**: retirements list locked. Safe to dispatch.
- **M4 unblocked**: name `src/units/` confirmed; reuse M62b IMPL brief verbatim with `s/sim_planner/units/g`.
- **M5 scope frozen**: Boussinesq + Rayleigh-Bénard match. Full thermal lifted out of ship-1.
- **M14 retarget**: merge into `dev/v0.2-multiphysics`, not `main`.
- **New mission proposed: KRK-SHIP-002** (separate ship plan, written later): full thermal + dissipation + WLF/Arrhenius + match against `rheoHeatFoam/channel/PTTLog` (Br=-25). Out of scope here.

### Scope cuts (deferred to later ships, ADR'd in mandate §4)

- Axisymmetric coverage
- Backend module factoring (`src/backend/{cpu,metal,cuda}`)
- Multiphase (rheoInterFoam parity)
- AMR
- STL geometry

These appear in `mandate.md §6` as `_audit needed_` and stay that way until a ship plan picks them up.
