# Boss memory — Kraken.jl

## 2026-05-28 — Bootstrap of orchestrator pattern on Kraken.jl

`.orchestrator/` created with mandate, ship-plan, and memory layers. First strategic mission (KRK-SHIP-001) drafted by Department.

## 2026-05-28 — KRK-SHIP-001 — ship plan adopted (Boussinesq variant)

The ship plan in `.orchestrator/ship-plan.md` lists 15 missions M1–M15 covering: branch cleanup, `src/units/` (M62b port), viscoelastic finalisation, thermal Boussinesq, RheoTool benchmarks per axis, tri-track docs, integration into `dev/v0.2-multiphysics`.

Estimated minimum: **9–10 sessions**. Comfortable: 14–16.

**Implication for future missions**: every Department brief on this project must point at `.orchestrator/ship-plan.md` as the mission catalogue, and at `mandate.md §4 ADRs` for scope decisions already taken. Do not re-debate frozen scope (thermal split, units name, merge target).

## 2026-05-28 — `dev-viscoelastic` is the de-facto multiphysics worktree

Department audit revealed that `dev-viscoelastic` (236 commits ahead, 863 files diff) carries the thermal + viscoelastic integration — NOT `slbm-paper` as the initial mandate §5 implied. Mandate §5 will be updated as this branch evolves.

**Implication**: any multiphysics-touching mission targets `dev-viscoelastic` as source, merges into `dev/v0.2-multiphysics` as target. `slbm-paper` is paper-only after this point.

## 2026-05-28 — Thermal scope split between ship-1 (Boussinesq) and ship-2 (full)

User initially asked for full thermal in ship-1, Department warned scope creep, Boss escalated, user accepted split. Ship-2 plan (KRK-SHIP-002) to be drafted only after ship-1 reaches MVP — do not pre-plan.

**Implication**: when ship-1 closes and the user asks "what's next on thermal?", load this entry — the answer is a new ship plan for full thermal + dissipation, matching `rheoHeatFoam/channel/PTTLog` (Br=-25, WLF/Arrhenius).

## 2026-05-28 — Scope cuts ADR'd for ship-1

Axisymmetric, backend module factoring, multiphase, AMR, STL — all deferred. Do not let any mission scope them in. If a brief touches any of these, refuse and re-scope.

## 2026-05-29 — M2 executed; v0.2-architecture retirement REVERSED

Retired `lbm` (local + `origin/lbm`) and `refinement-patches-dev` (both 0 unique commits). **Did NOT retire `dev/v0.2-architecture`** despite the 2026-05-28 ADR: inspection showed commit `74c39c633` carries `src/runtime_specs.jl` (502 LOC, "v0.2 runtime architecture skeleton: specs → lowering → plans" with typed lattice/BC/rheology/module/output codes) that exists on NO active branch. v0.3-campaign reused `axisymmetric.jl` but not `runtime_specs.jl` — so "superseded" was inaccurate.

**Implication for M3 (units spec)**: `runtime_specs.jl` on `dev/v0.2-architecture` is a prior art design of the specs→lowering→plans pattern the `src/units/` SimulationPlanner targets. The M3 Department brief MUST include it as required reading (`git show dev/v0.2-architecture:src/runtime_specs.jl`) — do not re-derive what was already prototyped. Final disposition of the branch decided after M3.

**Lesson**: ADR retirement premises ("superseded by X") must be verified by checking the unique commit content survives in X, not assumed. `git merge-base --is-ancestor` + per-file `git cat-file -e` on active branches before any `branch -d/-D`.

## 2026-05-29 — `.orchestrator/` is LOCAL-ONLY — never push to a public branch

`github.com:lauguimel/Kraken.jl` is a **PUBLIC** repo. The `.orchestrator/` scaffold (mandate, ship-plan, memory, resume-prompts) contains 50+ explicit Claude/Codex/Anthropic/Department/Engineer mentions → violates the confidentiality rule (no AI/LLM mentions in public-facing content).

Task B (resume-prompts §B) cherry-picked `ec90389e7` onto **local** `main` (commit `0a5bcdb54`). Cross-worktree access works via the shared local object store — **no push required** and none was done. `origin/main` is intentionally stale (164 commits behind local main); publication goes through scrubbed `release/*` branches (worktree `Kraken-main` on `release/v0.1`), never `main`.

**Guardrails**:
- NEVER `git push origin main` without re-checking this — it would fast-forward all 164 private commits incl. the AI-workflow scaffold onto a public branch.
- NEVER merge `main`/`slbm-paper`/`dev-*` into a public `release/*` branch without first stripping `.orchestrator/`. That merge is the one place the scaffold could leak.

## 2026-05-29 — M1 (Modular architecture audit) DONE — GREEN

Department classified all 99 `src/**/*.jl` files in `dev-viscoelastic`. Mandate §6 `_audit needed_` cells all filled (grep==0). Full table in `bench/scratch/m1_module_audit.md` (untracked scratch — `bench/` is NOT tracked on `slbm-paper`).

Headline findings for downstream missions:
- **28/99 files MIXED**, dominated (24) by **physics↔lbm-operator fusion** (constitutive math baked into collide kernels). Clean factoring seam already exists: `viscoelastic_spec.jl::AbstractPolymerModel`.
- **11 files over the 700-LOC HARD ceiling**. Two trunk SPLIT candidates blocking modularity: `io/kraken_parser.jl` (2010 LOC), `simulation_runner.jl` (1843 LOC) — the latter also holds inflow/outflow BC inline.
- **No `src/{bc,geometry,mesh,physics,backend,units,lbm}/` dirs exist yet** — every §6 target module is a net-new directory. M3/M4 (`src/units/`) is the first to create one.

**Implication**: M12/M13 (per-module docs) and any future refactor must treat `AbstractPolymerModel` as the constitutive seam and budget SPLIT missions for `kraken_parser.jl` + `simulation_runner.jl` BEFORE relocating BC/IO into target dirs.

## 2026-05-29 — M3 (units module spec) DONE — GREEN

Spec frozen at `docs/spec/units-v1.md` (588 lines) on new branch **`dev/units-module`** (worktree `/Users/guillaume/Documents/Recherche/Kraken.jl-units`, branched from local `main`). User chose a fresh isolated branch over dev-viscoelastic (clean module, single .orchestrator) — see [[where-units-lives]] decision below.

Spec covers §0–§10: API surface (`compile`/`audit`/`driver_kwargs`/`report` + `register_physics!`/`register_stability!`/`register_bc_combo!`), type hierarchy (`AbstractPhysicsSpec`, `LBMUnits{T}`, `SimulationPlan{T}`), 3-registry pattern, forward/reverse semantics, strict-vs-lenient errors, **10-file decomposition** (each ≤500/700), thermal Phase-2 zero-edit extension contract (with bit-identical proof obligation), validation gate, prior-art reconciliation, risks.

**Generalisations beyond M62b for M4 to honour**:
- Naming: units uses `SimulationPlan` / `ViscoelasticSpec` (NOT `runtime_specs.jl`'s `SimulationSpec` / `PhysicsSpec`); UInt8 lattice/BC codes via a future thin adapter, not in Phase 1.
- `src/geometry/` + `GeometryInfo` DON'T EXIST yet → spec uses a NamedTuple fallback + local `GeometryDescriptor`; a single real constructor is added when geometry lands (do not block M4 on geometry).
- Thermal/GNF/multiphase/MHD structs are pre-declared + pre-registered as stubs so M5 edits ZERO Phase-1 files.

**M4 dispatch note**: runner `codex` (kraken-branch-governor + kraken-codebase-map), deps M3 done. Brief must point at `docs/spec/units-v1.md` AND `dev/v0.2-architecture:src/runtime_specs.jl` (prior art) as required reading. Work happens on `dev/units-module` worktree.

## 2026-05-29 — M4 (units Phase 1 impl) DONE — GREEN

`src/units/` implemented on `dev/units-module` (commit `36e59adc6`), 15 source files (1149 LOC, all under 500/700), 4 test suites, .krk fixture, runtests `test_args=["units"]` dispatch, branch_contract. Codex pilot (kraken-branch-governor + kraken-resource-integrator). **165/165 tests green — re-run independently by Boss, not trusted from Codex.** Closes Phase B.

Verified non-tautological: M61 diffusive parity (τ=0.95, ν=0.15 at R∈{10,30,50}), round-trip 1e-12, strict/lenient raises, M59-B U-shape + M48 toggle audit codes fire, .krk dual-syntax parity.

**For M5 (thermal Phase-2)**: the zero-edit contract (spec §7) is the gate. M5 edits ONLY `src/units/physics/thermal.jl` (un-stub) + new `test/test_units_thermal.jl`. The §8.t proof = snapshot a Phase-1 VE plan, load thermal-concrete, recompute → assert bit-identical `units` + identical Issue.code set. `ThermalBoussinesqSpec` is already pre-declared in `Units.jl` + pre-registered in `PHYSICS_REGISTRY` (Codex shipped the seam). Codex resolved one M4 ambiguity: used §3.6 max-step formula over the PBS fixed-step note — re-confirm this doesn't bite thermal stability predicates.

**Codex pilot mechanics that worked**: brief at `<worktree>/.codex_brief.md`, `pilot.sh <wt> M4 --add-dir <sibling>` for cross-worktree reference reads (M48/M61 values on dev-viscoelastic), background launch, ~tens of min for a 1149-LOC + 165-test mission. The spec being frozen first (M3) made M4 a clean single-pass GREEN — spec-then-impl two-mission split paid off.

## 2026-05-29 — .orchestrator divergence DEBT (3 copies)

Three `.orchestrator/` copies now exist: **slbm-paper = CANONICAL** (freshest: M1 §6 patch, M2/M3 ADRs, all boss notes), `main`/`dev/units-module` = bootstrap version (`0a5bcdb54`, stale), `dev-viscoelastic` = yet another older copy. Departments MUST read the mandate slice from the slbm-paper worktree (`/Users/guillaume/Documents/Recherche/Kraken.jl/.orchestrator/`), never from a sibling worktree's copy. Consolidation (single source) is a deferred mission — for now the Boss writes only the slbm-paper copy. Deliverables (specs/code) land on their feature branch; mandate/ADR/memory stay canonical on slbm-paper.

## 2026-05-29 — Harness constraint: subagents cannot spawn sub-agents

In this Claude Code harness a Department subagent has **no Agent/Task tool** — it cannot spawn an Engineer (Layer 2). For read-only / structural missions the Department executes the work itself within the allowed zone (acceptable). For code-heavy missions needing Codex, the Boss must run `run-engineer.sh` directly rather than routing through a Department spawn. Also: a long Department `Agent` call dropped on a socket error after ~7 min mid-flight; its on-disk prep survived and a fresh re-spawn (a "resume" brief noting prep already done) completed cheaply. Keep Department missions short or checkpoint to disk.
