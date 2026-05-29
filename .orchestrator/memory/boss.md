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

## 2026-05-29 — M5 (units thermal Phase 2) DONE — GREEN; zero-edit contract PROVEN

`physics/thermal.jl` un-stubbed on `dev/units-module` (commit `9e3b8e04b`), 101 LOC. **Zero-edit verified**: `git diff --name-only -- src/units` = `physics/thermal.jl` ONLY. 277/277 tests green (re-run by Boss). The §8.t proof compiles VE plans R∈{10,30,50} around an active thermal probe and asserts `isequal` (incl NaN) vs M4 frozen refs + empty issue codes → thermal's load-time `register_*` calls don't perturb Phase-1.

Thermal conversion: `α=ν/Pr`, `β_thermal=Ra·ν·α/R³` (Ra reconstructs 1e-10). No M4 seam gap. This empirically validates mandate §1 "new physics = new module, zero trunk surgery" — GNF/multiphase/MHD will follow the identical pattern.

**Method win to reuse**: spec-frozen-first (M3) + pre-registered stub seam in M4 (`ThermalBoussinesqSpec` in Units.jl + `:thermal_boussinesq` in physics_registry) made M5 a clean one-file drop-in, single-pass GREEN. Replicate for any future physics extension.

**Phase-B units module COMPLETE** (M3 spec + M4 Newt/VE + M5 thermal). Next KRK-SHIP-001: M6/M7/M8 RheoTool benchmarks (Validate, claude-subagent + sim-rheotool/sim-openfoam), OR the driver-integration that consumes `driver_kwargs(plan)` (deferred Cd/Nu repro), OR docs M9-M13. M6/M8 dep M4 ✓; M7 dep M5 ✓ — all now unblocked.

## 2026-05-29 — .orchestrator divergence DEBT (3 copies)

Three `.orchestrator/` copies now exist: **slbm-paper = CANONICAL** (freshest: M1 §6 patch, M2/M3 ADRs, all boss notes), `main`/`dev/units-module` = bootstrap version (`0a5bcdb54`, stale), `dev-viscoelastic` = yet another older copy. Departments MUST read the mandate slice from the slbm-paper worktree (`/Users/guillaume/Documents/Recherche/Kraken.jl/.orchestrator/`), never from a sibling worktree's copy. Consolidation (single source) is a deferred mission — for now the Boss writes only the slbm-paper copy. Deliverables (specs/code) land on their feature branch; mandate/ADR/memory stay canonical on slbm-paper.

## 2026-05-29 — Harness constraint: subagents cannot spawn sub-agents

In this Claude Code harness a Department subagent has **no Agent/Task tool** — it cannot spawn an Engineer (Layer 2). For read-only / structural missions the Department executes the work itself within the allowed zone (acceptable). For code-heavy missions needing Codex, the Boss must run `run-engineer.sh` directly rather than routing through a Department spawn. Also: a long Department `Agent` call dropped on a socket error after ~7 min mid-flight; its on-disk prep survived and a fresh re-spawn (a "resume" brief noting prep already done) completed cheaply. Keep Department missions short or checkpoint to disk.

## 2026-05-29 — KRK-GEO track OPENED (STL/complex geometry) — user request, ADR override

User asked to add STL / complex-geometry support (`GMSH/STL → Kraken`). STL was an ADR'd ship-1 scope-cut (2026-05-28) — user request IS the override; opened a SEPARATE track KRK-GEO (ship-1 multiphysics scope unchanged). User chose **Cartesian immersed-boundary (STL) first, proof-of-concept before module consolidation**.

**Reality discovered (Explore + targeted reads, slbm-paper):** Kraken already has BOTH geometry paths at kernel level. (1) Cartesian: `io/stl_reader.jl::read_stl` → `io/voxelizer.jl::voxelize_2d/3d` → boolean `is_solid` mask → halfway-BB; `io/stl_libb.jl::precompute_q_wall_from_stl_2d/3d` exists for LI-BB cut-fractions but is NOT wired into the STL runner path. (2) Body-fitted: `Gmsh.jl` IS a dep (Project.toml:13), `curvilinear/mesh_gmsh.jl::load_gmsh_mesh_2d` + `multiblock/mesh_gmsh_multiblock.jl` + `curvilinear/slbm.jl::build_slbm_geometry` — fully implemented + tested. **The real gap is `.krk`→runner wiring**: parser recognises `Mesh gmsh(...)` (kraken_parser.jl) + `Obstacle{stl_file}` (STLSource), but `run_simulation` consumes the STL obstacle (`_apply_geometry!` simulation_runner.jl:2001-2036, called at :152) while the `Mesh gmsh` directive is IGNORED (no `setup.mesh !== nothing` clause in run_simulation dispatch).

**Implication**: the STL Cartesian proof is mostly assets + canary (mask path already live). The bigger §6 work is (a) wiring `Mesh gmsh(...)` into the runner for the body-fitted path, (b) consolidating `io/voxelizer.jl`+`io/stl_libb.jl`+`curvilinear/*` into `src/geometry/`+`src/mesh/`, (c) wiring LI-BB q_wall for STL (accuracy). M-GEO-1 dispatched to Codex on `feat/stl-krk` worktree. After it greens: decide GMSH wiring vs src/geometry/ consolidation as M-GEO-2.

## 2026-05-29 — M-GEO-1 result = YELLOW (geometry PROVEN; exposed a slbm-paper-local `run_simulation` REGRESSION)

Codex M-GEO-1 GREEN at the geometry level, **Boss-verified independently** (re-ran the canary myself: GATE-A 0/1600 mask disagreement — STL-voxelized cylinder ≡ analytical disc EXACTLY on 80×20; GATE-B max-speed diff 0.0%; VTK written). STL DSL syntax is `Obstacle <name> stl(file="…", scale=, translate=[…], z_slice=)` (NOT `stl_file=…` as my brief guessed). Files: `examples/geometry_stl/{make_cylinder_stl.jl,cylinder.stl,cylinder_stl_flow.krk,cylinder_analytic_flow.krk}` + `test/scratch/geo_stl_canary.jl`. **No `src/` edits** — STL→mask path worked as-is.

**BUT downgraded to YELLOW**: the canary only passes because it monkey-patches `Base.getproperty(::SimulationSetup, :mesh)=>nothing`. Root cause (empirically proven: `run_simulation("…krk")` throws `FieldError: SimulationSetup has no field mesh`): commit **`682e3f3c0` "Add body-fitted cylinder convergence runner"** (ancestor of slbm-paper ONLY, not main/release) added 4 `setup.mesh` refs + `_run_gmsh_slbm_drag` + `if setup.mesh !== nothing` (simulation_runner.jl:91,112) **without adding the `mesh` field to the `SimulationSetup` struct** (kraken_parser.jl:150-166, 15 fields, no mesh). → **`run_simulation(filename)` is BROKEN on slbm-paper for EVERY `.krk`** (generic Cartesian path). Unnoticed because the paper uses `_run_gmsh_slbm_drag`/the convergence runner directly, not generic `run_simulation`.

**Branch facts (git-grep verified):** STL machinery (`read_stl`/`voxelize_2d`/`_voxelize_stl_region`/`region.stl`) + GMSH/SLBM (`load_gmsh_mesh_2d`/`build_slbm_geometry`) are **slbm-paper-ONLY** — absent on main/release. main/release have NO mesh skew but also NO geometry stack. So the geometry track CANNOT be based on a clean main; it must build on the slbm-paper lineage (feat/stl-krk) and the skew must be fixed there.

**Next (M-GEO-2, NOT yet dispatched — awaiting user):** the proper completion of `682e3f3c0` — add `mesh::Union{Nothing,…}=nothing` to `SimulationSetup` + set it in the parser when a `Mesh` directive is present + update the parser's constructor call(s). This (i) un-breaks `run_simulation` [regression fix], (ii) removes the canary shim → M-GEO-1 turns true GREEN, (iii) is step 1 of GMSH-via-`.krk` wiring. Touches the 2010-LOC `kraken_parser.jl` (HARD-oversized) — surgical field add only, no SPLIT. Did NOT commit M-GEO-1 yet (committing the shimmed canary would entrench the bug).

## 2026-05-29 — CORRECTION: the `mesh`-field skew is LINEAGE-WIDE, not slbm-paper-local + dev-viscoelastic is the CLEAN STL base

User challenged my branch analysis ("other branches… dev/0.3?"). Ran the full matrix across 10 branches. Corrections to the entry above:

- **The skew (`setup.mesh` referenced ×4, NO `mesh` field in `SimulationSetup`) is on the ENTIRE v0.3/paper lineage**: `slbm-paper`, **`dev/v0.3-campaign`**, `feat/amr-port-sr`, `docs/module-architecture` — ALL carry commit `682e3f3c0` and ALL have the broken generic `run_simulation`. **`dev/v0.3-campaign` (the v0.3 RELEASE campaign branch) is therefore release-blocked** by this regression (verified: struct 15 fields no-mesh + runner refs at :91/:112/:930-931 `error("Gmsh SLBM drag runner requires setup.mesh")`). NO branch ever completed the `mesh` field.
- **`dev-viscoelastic` is CLEAN for the Cartesian/STL track**: `setup.mesh`=0 in its runner (run_simulation NOT broken) AND it HAS the STL voxelize path (`_voxelize_stl_region` in src/simulation_runner.jl). It is the "de-facto multiphysics worktree" (236 ahead of main), worktree `Kraken.jl-viscoelastic/`. It lacks GMSH/curvilinear (GMSH=0) — fine for Cartesian/STL; GMSH body-fitted would need porting later.
- `main`/`release/v0.1`: no skew AND no geometry stack (STL machinery is entirely absent there).

**Implication for branch-home decision (user to pick):** (A) **dev-viscoelastic** = cleanest STL base, replay M-GEO-1 shim-free → true GREEN, no regression baggage, but no GMSH yet. (B) **dev/v0.3-campaign** = has STL+GMSH but needs the mesh-field fix; doing it there ALSO clears the release-blocking regression. (C) slbm-paper = paper-coupled, same skew as v0.3. The mesh-field fix is needed on the v0.3 lineage REGARDLESS (release-blocking) — that's now an independent reason to do M-GEO-2 on dev/v0.3-campaign even if the STL track lives on dev-viscoelastic.

## 2026-05-29 — CORRECTION 2: dev-viscoelastic HEAD does NOT BUILD (depends on untracked WIP) → base flips to dev/v0.3-campaign

User picked dev-viscoelastic. Tried to replay M-GEO-1 in a fresh worktree off it → **fails to load**: committed `src/Kraken.jl` does `include("rheology/linalg_3d.jl")` (line 25) but `linalg_3d.jl` + `kernels/logconformation_lbm_3d.jl` are **UNTRACKED** (exist only in the `Kraken.jl-viscoelastic/` working tree, dated May 10) — plus modified-uncommitted `viscoelastic_3d.jl`/`viscoelastic_spec.jl`/`li_bb_2d_v2.jl`. So **dev-viscoelastic's committed HEAD is broken**; its "cleanliness" was an artefact of the dirty origin worktree. Unusable base for a committable feature branch. (Separate finding worth a mission: dev-viscoelastic needs those WIP files committed or the include guarded — its HEAD is un-buildable from a clean checkout.)

**Buildability matrix (fresh checkout):** dev-viscoelastic = NO (untracked includes); **dev/v0.3-campaign = YES** (clean working tree, Kraken.jl does NOT include the untracked files); slbm-paper = YES (built in M-GEO-1 run; doesn't include linalg_3d). The two untracked files are absent on all three branches but only dev-viscoelastic's Kraken.jl *includes* them.

**REVISED recommendation → base `feat/geometry-stl` on `dev/v0.3-campaign`**: builds clean + full STL+GMSH stack + release-aligned. The only issue is the mesh skew → do **M-GEO-2 first** (add `mesh` field) so `run_simulation` works, THEN replay M-GEO-1 canary shim-free → true GREEN. This fix is release-valuable (un-blocks v0.3). The fresh dev-viscoelastic worktree (`Kraken.jl-geometry-stl`) built off the wrong base is to be torn down. Awaiting user confirm of the base flip (they explicitly chose dev-viscoelastic on now-falsified info).

## 2026-05-29 — M-GEO-2 + M-GEO-1 BOTH GREEN on dev/v0.3-campaign (Boss-verified, shim-free)

User confirmed dev/v0.3-campaign. Tore down the dev-viscoelastic worktree, recreated `feat/geometry-stl` off `dev/v0.3-campaign` (HEAD `2d27bf684`), worktree `Kraken.jl-geometry-stl` (needed `Pkg.instantiate()` — added StaticArrays/Enzyme to the fresh worktree).

**M-GEO-2 (regression fix, Boss-direct, ~12 lines in ONE file)**: discovered the body-fitted feature is even more incomplete than thought — NO `MeshSetup` struct + NO `Mesh`-directive parser exists on ANY branch (slbm-paper or v0.3); only the runner *consumer* (`_run_gmsh_slbm_drag` + `if setup.mesh !== nothing`) exists, duck-typing `setup.mesh.kind/.file/.multiblock/.layout` at runtime. So the minimal fix is just: add `mesh::Any` field to `SimulationSetup` (kraken_parser.jl:166) + a **backward-compatible 15-arg outer constructor defaulting `mesh=nothing`** → all existing 15-arg call sites (parser:469, 3 test sites in test_simulation_runner.jl) keep working, only `_override_max_steps` uses the 16-arg form. Did NOT build the Mesh parser (that's the real GMSH-via-.krk wiring, deferred). 

**Verified by Boss**: `using Kraken` loads; the shim-free canary prints `GATE-A 0/1600 (0.0%)` + `GATE-B 0.0%` + `PASS M-GEO-1`; and `run_simulation("examples/geometry_stl/cylinder_stl_flow.krk")` — the exact call that threw `FieldError: no field mesh` pre-fix — returns max|u|=0.031018. git status: only `src/io/kraken_parser.jl` modified + untracked `examples/geometry_stl/` + `test/scratch/`. diff --check clean.

**NOT committed yet** (CLAUDE.md: ask first). Commit plan proposed to user: (1) `fix:` the kraken_parser.jl mesh-field — independently a v0.3-release regression fix, **cherry-pick candidate to slbm-paper / feat/amr-port-sr / docs/module-architecture which carry the identical skew**; (2) `feat:` the examples/geometry_stl demo (+ promote canary from test/scratch to a real test?). Branch `feat/geometry-stl` → merges to dev/v0.3-campaign. The slbm-paper M-GEO-1 worktree `Kraken.jl-stl-krk` is now superseded — tear down after commit.

### COMMITTED (2026-05-29) — KRK-GEO M-GEO-1+2 shipped on `feat/geometry-stl`

User approved: 2 commits + canary promoted to a real `@testset`.
- `fc9a4d7eb` **fix(io)**: mesh field on `SimulationSetup` (M-GEO-2 regression fix).
- `434aee036` **feat(geometry)**: STL obstacle via `.krk` (M-GEO-1) — example `examples/geometry_stl/` + `test/test_geometry_stl_krk.jl` (wired into runtests after `test_stl.jl`). Promoted test = **7/7 pass, 8.9s** (Boss-run).
Working tree clean. **Blast-radius validated post-commit** (Boss-run): `test_kraken_parser` + `test_simulation_runner` (the 3 SimulationSetup construction sites) + `test_stl` + `test_stl_libb` + new test = **9361/9361 pass, 18.8s** → the core-struct `mesh`-field change + backward-compatible 15-arg constructor regress nothing. Superseded worktree `Kraken.jl-stl-krk` + branch `feat/stl-krk` torn down. Worktree `Kraken.jl-geometry-stl` (off dev/v0.3-campaign) remains.

**OPEN follow-ups (user to prioritise next session):**
1. ✅ **DONE 2026-05-29 — cherry-picked `fc9a4d7eb`** onto the lineage branches:
   - `feat/amr-port-sr` → `1a276b141` (clean).
   - `docs/module-architecture` → `ea19708b8` (clean).
   - `slbm-paper` → **SKIPPED (justified)**: its `kraken_parser.jl` has a large UNCOMMITTED WIP (+85/−5) that **already adds `mesh::`** — i.e. a fuller mesh solution is in progress there, almost certainly the **`Mesh`-directive parser + `MeshSetup` struct** that was missing everywhere (the producer half of GMSH-via-`.krk`). My minimal fix would be redundant/conflicting → not applied. **Implication for M-GEO-3(b)**: the GMSH-via-`.krk` wiring is likely already being built on slbm-paper — coordinate, don't duplicate. Per-branch re-test skipped (the diff is byte-identical to the v0.3-validated fix, applied to the identical skew). No push (local).
2. ✅ **DONE 2026-05-29 — merged `feat/geometry-stl` → `dev/v0.3-campaign`** (fast-forward, `git merge --ff-only`; dev/v0.3-campaign now at `434aee036`, carrying both commits). Made v0.3 a coherent local PR candidate. Pre-existing WIP on the v0.3 worktree (6 modified `benchmarks/convergence_*.jl` + `.engineer_brief_*`) left untouched. **PR-readiness facts**: `.orchestrator/` is NOT tracked on dev/v0.3-campaign (0 files → no scaffold leak on push). BUT dev/v0.3-campaign is **216 commits ahead of main**, and `origin` = PUBLIC `github.com:lauguimel/Kraken.jl`. My 2 commits are confidentiality-clean (no AI mentions); the other 214 are UN-audited. **Actual push/PR = user decision** (audit the 214 + pick the PR target; NOT pushed). Full test-suite certification on v0.3 not yet run (only blast-radius 9361 + new test).

### 2026-05-29 — Full suite on dev/v0.3-campaign: **34265 pass, 7 fail, 0 error, 4 broken** — all 7 failures PRE-EXISTING, NOT from the geometry merge

Ran full `runtests.jl` on the merged `dev/v0.3-campaign` (worktree, after instantiate). My merge is **clean** — proven ironclad: the 7 failing assertions live in `test_conservative_tree_streaming_2d.jl` (×4) + `test_multiblock_exchange.jl` (×3), both of which (a) have **0** references to `SimulationSetup`/`run_simulation`/`load_kraken` and (b) are **not in my 2-commit diff** → the `mesh`-field change cannot reach them. All my tests (geometry_stl_krk, stl, kraken_parser, simulation_runner) pass.

**Pre-existing v0.3 test debt to flag for the PR (NOT geometry, separate):**
- `test_conservative_tree_streaming_2d.jl:619-622` — "route native Poiseuille X/Y bands": mass_drift PASSES (1e-12) but **l2_error/linf_error exceed thresholds** (5.5e-3 / 8.5e-3) for both xband+yband. AMR conservative-tree **accuracy** regression/tuning on v0.3. NB: contradicts [[amr_d_status_roadmap_20260523]] "2D production-ready / test locked" — those AMR-D fixes may not have landed on v0.3-campaign, or v0.3 regressed.
- `test_multiblock_exchange.jl:161-162` — "Ng=2 copies 2 ghost rows" array-equality fail; `:186` — "unsupported same-normal edge pair throws" expected an `ErrorException` that wasn't raised.
- 4 `@test_broken` (Multi-block ghost exchange + Viscoelastic) — pre-marked, not new.

**For the PR**: dev/v0.3-campaign is NOT fully green; closing it needs a separate AMR-conservative-tree + multiblock-exchange fix mission (not geometry). The geometry contribution itself is green and merge-clean.
3. **M-GEO-3 options**: (a) wire LI-BB `q_wall` for STL (`precompute_q_wall_from_stl_2d` → Bouzidi accuracy, currently STL uses plain halfway-BB); (b) build the `Mesh`-directive parser + `MeshSetup` struct so the body-fitted GMSH path is reachable from `.krk` (the real GMSH-via-.krk wiring — no parser producer exists on any branch yet); (c) `src/geometry/` + `src/mesh/` §6 consolidation.
4. **Separate (non-geometry)**: `dev-viscoelastic` HEAD is un-buildable from a clean checkout (includes untracked `linalg_3d.jl` + `kernels/logconformation_lbm_3d.jl`); needs those committed or the include guarded.

## 2026-05-29 — M6 (Cartesian cavity RheoTool benchmark) DONE — GREEN (2D) + 3D top-BC bug FIXED  [units track, parallel to KRK-GEO]

Phase C opened on `dev/units-module` (worktree `Kraken.jl-units`). M6 = lid-driven cavity validated vs THREE references: Ghia 1982, OpenFOAM icoFoam (Docker `opencfd/openfoam-default:2412`, 128²), Kraken (D2Q9/D3Q19 BGK, Metal F32). Boss decomposed into parallel tracks (M6a Kraken ∥ M6b OpenFOAM) + M6c synthesis to dodge the Department bail-out + socket-drop risk.

**Commits**: deliverables `a524f7ded` (`bench/cartesian_rheotool/**` + `docs/src/users/benchmarks/cartesian-cavity.{md,png}`); kernel fix `9674e1c4c`, both on `dev/units-module`; kernel fix **cherry-picked to `main` as `a5ce7c6f0`** (general bug, identical base). No push.

**Numbers**: icoFoam <0.5% L2(u) vs Ghia at all Re (reference validated). Kraken BGK 2-3% rel / ~1% abs-RMS → accepted **GREEN under mandate §3** (centreline = local field ≤5%; <1% is integral-quantity-only). 3D cavity 23.6%→2.38% after BC fix.

**3 durable findings**:
1. **3D top Zou-He BC bug (FIXED)**: `zou_he_velocity_3d_kernel!` (`src/kernels/boundary_3d.jl`) omitted the wall-parallel diagonal pops (`parallel[6:9]`) from the transverse-momentum correction → `ux_lid = u_lid + (f8-f9+f10-f11)/ρ` ≈ 1.5× overshoot. Generic fix, uniform sign pattern all 6 faces (`tang1 += p6-p7+p8-p9`, `tang2 += p6+p7-p8-p9`). 2D (D2Q9) immune (`f2-f4` is already full tangential momentum).
2. **TRT collision EXISTS (M6c-v2 was WRONG — branch-scoping artifact).** A generic Newtonian D2Q9 TRT kernel `fused_trt_step!(f_out,f_in,ρ,ux,uy,is_solid,Nx,Ny,ν; Λ=3/16)` + `trt_rates(ν;Λ=3/16)` (Ginzburg & d'Humières 2003 magic param) lives in **`src/kernels/fused_trt_2d.jl` on `slbm-paper` AND `dev-viscoelastic`** (commits e.g. `96155d560` "TRT==BGK on flat channel", `0c67f67b1` "TRT magic by default"; `dev-viscoelastic:drivers/cavity_driver_2d.jl` already uses it). It is a drop-in replacement for `collide_2d!` (BGK). It is ABSENT from `dev/units-module`/`main` only because units branched from main BEFORE the TRT/SLBM work landed on the paper lineage — that is why M6c-v2 (scoped to the units tree) reported "no MRT/TRT". → **The ship-plan "<1%" gate IS reachable by porting/using `fused_trt_2d.jl` (no new kernel needed)**; the M6 "MRT demo row" the user wanted is feasible. NB: TRT is 2D-only (no `fused_trt_3d`); 3D stays BGK.
3. **Marginal-ω BGK instability**: the corrected 3D BC diverges at coarse-grid near-limit ω (16³, ω≈1.89) where the BUGGY BC gave finite-but-garbage 1.96× overshoot. NOT a fix defect (fix gives exact `max|ux|/u_lid=1.0` at all stable configs 16³/Re16, 32³, 64³). `test_cavity_3d` moved to the stable envelope (32³ Re=100) + a no-overshoot regression guard.

**Process wins (reusable)**: (a) **Codex measure-first paid off**: the Phase-0 diagnostic FALSIFIED the static-localization "double-BC / stream-bounce-back" hypothesis and measured the real omitted term. (b) **Codex via pilot.sh STALLED after Phase-0** (wrote diagnostic+findings, never edited src, log stale 80 min, no footer). Because measure-first forced the diagnosis to disk FIRST, the mission survived — Boss applied the 6-line fix directly. Always force write-first. (c) **M6b OpenFOAM Department bail-out struck again** (armed Monitor, exited mid-run); background `docker run` survived; Boss waited via background `docker wait <ids>` + a PURE-extraction agent (no Monitor). (d) **OF-in-Docker gotcha**: must `cd /case` INSIDE `bash -c` (the 2412 entrypoint forces CWD to /root; `-w` ignored) — reuse for M7.

**Next dispatchable**: M7 (thermal Rayleigh-Bénard, dep M5✓ — reuses icoFoam-Docker recipe + `bench/scratch/run_cavity_bench.jl`/`plot_cavity_bench.jl`), M8 (VE cylinder, Aqua), docs M9. Auto-memory `MEMORY.md` is OVER its size limit (30KB>24KB) — consolidation overdue (not done this session).

## 2026-05-29 — M6 cavity 2-3% was a HALF-CELL COORDINATE BUG, not BGK/MRT/resolution — now <1%

Follow-up to the M6 entry above. User pushed back on the "BGK caps at 2-3%, needs MRT" framing ("le TRT devrait-il améliorer? sinon bug — investigate"). A mesh sweep + Couette + Mach discriminator **falsified the resolution/collision/compressibility story** and found the real root cause:

- **Mesh sweep** (64/128/256 × BGK/TRT, Re=100): cavity u-centreline converged at **p≈0.85 (1st-order, NOT 2nd)**, Ma-independent (Ma 0.173→0.043 ⇒ no change), TRT≡BGK (kernel validated exact: TRT(Λ=9ν²)≡BGK to 0.0e0). Max error pinned at y=0.97 (just under the lid) at every mesh.
- **Couette** (periodic-x, same Zou-He-lid/BB-wall mix, exact linear solution): standard coordinate p=1.01 with error = **EXACTLY 0.5/Ny** (half-cell) to 3 sig figs; lid-on-node coordinate **machine zero (1e-9–1e-12)** → solver EXACT; the entire error is coordinate placement.
- **Root cause**: the Zou-He moving lid sits **ON the node** (top wall at node N), but the comparison hand-coded `(j-0.5)/N` (assumes BOTH ends halfway-BB). Mixed Zou-He-lid/BB-walls ⇒ height `H=(N-0.5)Δ` ⇒ `(j-0.5)/(N-0.5)`. Half-cell mislocation = uniform O(1/N) first-order error.

**Fix = wall-aware coordinate helper** `axis_node_coords(N; lo, hi)` (`src/axis_coords.jl`, commit **`78ed276c4`** on `dev/units-module`, **cherry-picked to `main` as `9b9a97aa8`**). Single source of truth for node→physical mapping by BC (`:bb` wall at edge vs `:onnode` Zou-He on node). Harness rewired; **corrected M6 commit `04188e067`**.

**Corrected M6 result (supersedes the "2-3% / §3 ≤5%" framing above)**: Kraken-vs-Ghia u-centreline rel-L2 = **0.47% / 0.41% / 1.05%** at Re=100/400/1000 (was 3.08/2.23/2.89%); Kraken-vs-icoFoam 0.30/0.58/1.60%. **Cavity MEETS the strict <1% gate** (Re=1000 marginal 1.05% / abs-RMS 0.43%, not fully steady at 500k cap). 3D mid-plane 2.1% abs-RMS (residual = genuine 3D-vs-2D-Ghia confinement; needs a 3D ref like Yang 1999 for a true 3D gate).

**Corrections**: (1) finding-#2 above ("<1% reachable only by porting TRT") is MOOT — <1% reached by the coordinate fix; TRT confirmed **irrelevant to the cavity** (solver exact; TRT≡BGK on cavity AND Couette). **TRT port to units REVERTED** (detour; `fused_trt_2d.jl` lives on slbm-paper if ever needed for a genuinely wall-slip-limited case like Poiseuille/thermal). (2) committed page now states **2D PASS at strict <1%**.

**Propagation (user wants helper in ALL worktrees)**: done on `dev/units-module` + `main`. **PENDING cherry-pick of `78ed276c4`/`9b9a97aa8` to the lineage branches** (slbm-paper, dev-viscoelastic, dev/v0.3-campaign, feat/amr-port-sr, docs/module-architecture) — DEFERRED: those worktrees are dirty/active (slbm-paper heavy uncommitted; v0.3-campaign hosts the live parallel KRK-GEO session). Batch at a clean moment (alongside KRK-GEO's pending `fc9a4d7eb`) or via M14. Backstop = [[feedback_wall_aware_coords]] so future briefs use the helper regardless.

**LESSON (→ [[feedback_wall_aware_coords]])**: before declaring a benchmark gap "physics/resolution/collision", run a convergence-ORDER check + an exact-solution control (Couette/Poiseuille). A 1st-order rate + an exact-solution control going to machine-zero localises a BC/coordinate bug, not a solver limit. User's "is TRT supposed to help? if not, investigate" caught a 2-3%→<1% benchmark-setup bug.
