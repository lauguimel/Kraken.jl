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

## 2026-05-31 — M-GEO-7 local twin GREEN: STL drag is CORRECT (the "78% bug" was a registration artifact)

**MAJOR CORRECTION to a finding I almost shipped.** While building the M-GEO-7 3D sphere drag twin (STL `.krk` `run_obstacle_libb_3d` vs the validated analytic `run_sphere_libb_3d`), the first twin showed **Cd_krk=6.05 vs Cd_ref=3.52 (+78%)**. I diagnosed it as a `precompute_q_wall_from_stl_*` bug (q decorrelated from analytic, mean≈complement), reported it as a real 2D+3D bug, and the user authorized characterizing the 2D blast radius. **It was NOT a bug.**

**Root cause (proved by an offset sweep):** the STL voxelizer is **cell-centred** (`(i-0.5)·dx`, stl_cut_fraction.jl:102) while the analytic q_wall is **node-centred** (`xf=i-1`, li_bb_2d.jl:275 / precompute_q_wall_sphere_3d). A half-cell frame difference. Placing the STL and analytic obstacles at the SAME numeric centre put them on lattice positions **0.5 cell apart** → different cut-link sets → different drag. The 2D offset sweep (STL centre δ∈{-1,-0.5,0,0.5,1}) showed **corr(q_stl,q_analytic) jumps to 1.0 at δ=+0.5** (mean diff 2e-4). The STL q_wall is EXACT once registered. 3D confirmation: STL at (30,30,30) vs scaffold at **cx=29.5** → **Cd 6.051 vs 6.075 = 0.39%** twin.

**Two durable lessons:**
1. **LI-BB drag at coarse R is strongly registration-sensitive** — the SAME analytic sphere gives Cd=3.52 on-node (cx=30) vs Cd=6.08 half-off-node (cx=29.5), ×1.7. Not STL-specific; a real coarse-resolution artifact (R=8). Absolute drag needs high-res (Aqua) to converge toward Clift 2.6; the *twin* (krk≡scaffold at matched registration) is the right local gate.
2. **PROVE the mechanism before declaring a bug.** I had a measurement (78% + decorrelated q) that *looked* like a bug; the "prove it" gate (an offset sweep, ~2 min CPU) refuted it and saved a fix mission on correct code. The decorrelation was real but its CAUSE was my mis-registered comparison, not broken code. → near-zero per-link correlation + "half the cut-links in different slots" is the FINGERPRINT of a sub-cell registration offset, not a convention/indexing bug.

**M-GEO-7 status:** local twin GREEN on Metal F32 (0.39%, both Cd in Clift band [1,8]). Deliverables (uncommitted, gate running): `obstacle_3d.jl` +Cd (frontal-area silhouette convention), `examples/geometry_stl/{sphere_drag.stl, sphere_stl_3d_drag.krk}`, `test/test_sphere_stl_drag_krk.jl` (GPU-gated: CPU=parse+mask smoke, GPU=registered twin), runtests wiring. **Aqua F64 full-res publishable Cd = the follow-on** (user chose "twin local now + Aqua F64 after"); convention gotcha to consider documenting: STL obstacle position lands +0.5 cell vs an analytic obstacle at the same coordinate.

## 2026-05-30 — 3D pull-brick CPU segfault FIXED (closes the M-GEO-6 durable bug)

User picked "fix the segfault" as the next mission. Committed `770375a2f` on `feat/units-on-v03`, **FF-merged to `dev/v0.3-campaign` (HEAD `770375a2f`)**. Full suite **34594/7/0 — IDENTICAL** to the M-GEO-6 baseline (bit-identical fix).

**Root cause (precise):** the `PullHalfwayBB_3D` brick (kernels/dsl/bricks_3d.jl) already had `ifelse(cond, f_in[i-1,...], fallback)` guards — BUT Julia's `ifelse` is a FUNCTION that **evaluates BOTH arms**, and the neighbour index was **unclamped** (`f_in[i-1,...]`). At i=1 the discarded arm still reads `f_in[0,...]` → OOB under `@inbounds` → segfault on CPU. The **2D** `PullHalfwayBB` brick (bricks.jl:30-37) never segfaults because it **clamps** the index (`f_in[max(i-1,1),...]`). The 3D brick just lacked the clamp on all 18 neighbour reads.

**Fix:** add `min`/`max` clamps to the 18 neighbour indices in the 3D brick (mirror the 2D brick). **Provably bit-identical**: the clamp only changes the address of the DISCARDED ifelse arm, never the selected result — verified 55/55 on the 3D LI-BB cluster (incl. "interior bit-exact") + full suite unchanged. Removed the now-unneeded `_CPUHaloArray4` from `drivers/obstacle_3d.jl` (plain arrays like the sphere driver, −24 LOC).

**GPU-efficiency note (user flagged mid-mission "copy_to???"):** all `copyto!` are one-time host→device SETUP transfers (q_wall is host-computed by design), NOT per-step; the hot loop is pure device kernels + pointer swap. While simplifying I'd accidentally made drag compute EVERY step — restored the windowed average (last 20%, mirrors `run_sphere_libb_3d`). Lesson: when simplifying a driver, re-check the hot loop for accidental per-step kernel launches.

**Method win:** Boss-direct fix (not delegated) — the diagnosis (segfault repro + 2D-vs-3D brick comparison) localized a mechanical, bit-identical clamp; validated by repro + bit-identity cluster + full suite before committing. `ifelse` eager-both-arms is the gotcha to remember: guarded array reads in a hot kernel must ALSO clamp the index, not just gate with the condition.

## 2026-05-30 — M-GEO-6 DONE (generic 3D STL-obstacle FLOW runner); drag → M-GEO-7

KRK-GEO chain extended again. Committed `6c4d2774e` on `feat/units-on-v03`, **FF-merged into `dev/v0.3-campaign` (HEAD `6c4d2774e`)**. Full suite (autonomous gate, user said "fais tout en autonomie") **34594 pass (= 34585 + 9) / 7 pre-existing fails (identical) / 0 errored / 4 broken**. Now a D3Q19 `.krk` with an STL `Obstacle wall=libb` + inflow/outflow actually flows via `run_simulation` (was: `_run_d3q19` threw `"only cavity_3d is supported"`).

**Plan agent flipped the size estimate**: the mandate predicted "200-400 LOC build, the 3D analog of M-GEO-1+3a." The Plan found the **entire 3D LI-BB flow stack already exists** and runs end-to-end in `run_sphere_libb_3d` (kernels/li_bb_3d_v2.jl:201) — step kernel `fused_trt_libb_v2_step_3d!` + 6-face `BCSpec3D`/`apply_bc_rebuild_3d!` (boundary_rebuild.jl) + cut-link drag `compute_drag_libb_3d` + equilibrium init. It only built `q_wall` from ANALYTIC sphere geometry + hard-coded west-vel/east-pressure. → mission became ~282 LOC REUSE-wiring. **Lesson: always Plan-recon the reuse surface before sizing a "new runner" mission — the 3D infra was 90% there.**

**Delivered (Codex single-pass GREEN via run-engineer.sh):** new `src/drivers/obstacle_3d.jl` (206 LOC) = `run_obstacle_libb_3d` (lifts the sphere-driver loop; STL q_wall + `.krk` BC instead of analytic+hardcoded) + `_build_libb_bc_rebuild_spec_3d`. `_precompute_stl_libb_q_wall_3d` (geometry/libb_precompute.jl, wraps existing `precompute_q_wall_from_stl_3d`). Structural dispatch `_has_stl_libb_obstacle` in `_run_d3q19` (+5/−1 — the ONLY runner edit; one-concern preserved). Validation = **canary-only** (user-chosen): units flow twin + smoke; quantitative drag → M-GEO-7 (GPU/Aqua + confined-sphere reference).

**Validated (Boss independent):** flow twin `_3d_mm.krk` ≡ `_3d_lu.krk` **L∞=0.0 EXACT** (units→LU correct at FLOW level in 3D — extends M-GEO-5's mask twin); non-trivial (max|ux|=0.0062 > inlet 0.005, accelerates around sphere); stable (ρ∈[0.995,1.006], no NaN). Cluster re-run 78/78 incl. `test_sphere_libb` (template NOT regressed) + `test_cavity_3d` (dispatch OK).

**Review TRAP I almost mis-called + the durable bug it exposed:** Codex added a 36-LOC `_CPUHaloArray4` NOT in the brief — I first flagged it as over-engineering (the sphere template uses plain arrays). **Tested before judging** (the right move): `run_sphere_libb_3d` at 80×20×20 on CPU **SEGFAULTS** (`fused_trt_libb_v2_step_3d!` does `@inbounds` OOB edge reads, `lbm_builder.jl:75`). So the halo (zero-padded edge cells) is a **correct, necessary CPU-safety fix**; GPU keeps plain alloc (the validated path). The sphere CPU test only passes at 16×8×8 because OOB lands in valid heap (latent bug masked by small size). **Durable: the 3D pull brick needs proper edge guards so CPU needs no halo — future fix, flagged in mandate M-GEO-6 + ADR. NEVER judge "unrequested complexity" as over-engineering without reproducing the failure it prevents.**

## 2026-05-30 — M-GEO-5 DONE (mask-level 3D units bridge + descriptor fields); 3D FLOW runner → M-GEO-6

KRK-GEO chain extended. Committed `9acf4eb4d` on `feat/units-on-v03`, **FF-merged into `dev/v0.3-campaign` (HEAD `9acf4eb4d`)**. Full suite (user-run gate) **34585 pass / 7 pre-existing fails (identical file:line) / 0 errored / 4 broken** — `34585 = 34570 + 15` new (the 3D test); zero new failures. **0 edits to `simulation_runner.jl`** (M-GEO-4 win preserved).

**KEY: Boss recon falsified the mission premise BEFORE dispatching.** The brief assumed "3D path exists, mostly needs an example + the 3D scale." It does NOT: `run_simulation` D3Q19 → `_run_d3q19` dispatches ONLY `cavity_3d` (throws `"only cavity_3d is supported in v0.1.0"` otherwise); the generic 2D STL+LI-BB flow loop (simulation_runner.jl ~:135-239) has NO 3D twin; `fused_trt_libb_v2_step_3d!` (exists, tested) is NEVER called from the runner. The units→LU `dx_real` scale IS dim-agnostic, but there is nowhere to plug a 3D flow. → escalated via AskUserQuestion (step-back/mandate-gap trigger), user chose **Option A**: mask-level 3D twin now, **3D STL flow runner deferred to new mission M-GEO-6** (the 3D analog of M-GEO-1+M-GEO-3a — a real ~200-400 LOC runner, NOT "wiring"; runner already HARD-oversized).

**Plan agent (read-only) surfaced 3 decisive findings** that shaped the impl: (1) scale path genuinely dim-agnostic (0 change); (2) `_apply_geometry_3d!` had a DEAD `region.stl !== nothing && continue` — it never voxelized STL (fixed: new `_voxelize_stl_region_3d` + STL branch); (3) `audit_stl` early-returns on `q_wall_dist===nothing`, so `kappa_max` is invisible in the audit unless q_wall_dist is also populated, AND kappa_max's convention is **inverse-LU** (audit threshold `R_LU·kappa_max>0.5`).

**Boss design decisions (locked before Codex):** kappa_max = bbox-half-span proxy `1/R_LU` (=0.25 sphere, exact for sphere/cylinder); L_up/L_down derived from obstacle centroid in units of R via **NaN-sentinel** in `UnitsSetup` (derive-if-omitted, else respect user knob) + thread the resolved L's into `Units.compile` (so `max_steps` reflects them); q_wall_dist = `halfway_wall_distances(mask)` (0.5-per-fluid-solid-link, dim-agnostic) — all-0.5 un-gates the audit and fires ONLY `:curvature_underresolved`, no spurious cliff/skewness. New geometry helpers (`stl_kappa_max`/`obstacle_extents_in_R`/`halfway_wall_distances`) live in `src/geometry/descriptor.jl` (geometry concern, not parser). Asset = NEW sphere STL (user chose sphere over cylinder-prism: genuine 3-axis 3D voxelization).

**Process wins / traps:** (a) Boss recon (5 grep/read rounds) caught the false premise pre-dispatch — no wasted Codex run on a nonexistent path. (b) Codex single-pass GREEN (335 LOC, 9 files) via run-engineer.sh; Boss independently re-ran the key cluster (63/63: 2D regression + 3D twin + cavity_3d + slbm_libb_3d) before trusting the report. (c) **CLEANUP TRAP**: `rm -rf tmp` on the geometry-stl worktree deleted TRACKED committed diagnostic scripts (tmp/ is NOT ephemeral-scratch on this lineage — it's versioned) + the full-suite run had modified tracked `output/*.vtr`; restored both via `git restore tmp output`. **Lesson: on Kraken v0.3 worktrees, `tmp/` and `output/` are TRACKED — never blanket-`rm`; remove only the specific ephemerals you created (`.engineer_brief.md`, `.engineer_logs/`).**

## 2026-05-30 — units↔geometry: GRAFT (not merge) units onto v0.3; full suite green

User wants geometry wired to units (STL is physical → needs dim↔LU; `src/units/` IS the dim↔LU owner, already geometry/STL-aware: own `GeometryDescriptor` Units.jl:50 + `audit_stl`). Prerequisite: co-locate the two modules — they were on SEPARATE branches (units on `dev/units-module`, geometry on `dev/v0.3-campaign`).

**KEY FINDING — naive `git merge dev/units-module → dev/v0.3-campaign` is a TRAP**: `git merge-tree` preview showed DOZENS of bogus `modify/delete` conflicts (curvilinear/, kernels/dsl/, geometry/, drag_gpu…) + Project.toml removing Atomix/FFTW/ForwardDiff. Cause: units branched off ~main; v0.3 is 302 commits ahead of main, so ALL v0.3 work "is missing" on the units side and the 3-way merge reads it as deletions. **Correct operation = GRAFT the self-contained `src/units/` submodule**, not merge the divergent branch.

**Graft executed** (Boss-direct, branch `feat/units-on-v03` off v0.3): `git checkout dev/units-module -- src/units/ test/test_units*.jl benchmarks/krk/units/` (16 src + 5 tests + 1 .krk fixture) + `include("units/Units.jl")` & `export Units` in `src/Kraken.jl` (after postprocess, before exports) + nested `@testset "Kraken.Units"` INSIDE the main runtests testset. **Gotcha**: first placed the units testset AFTER the main `@testset` — the main testset THROWS at its `end` (7 pre-existing fails) so the trailing units block never ran (pass count unchanged, units absent). Fix = nest units INSIDE the main testset. units krk_binding.jl has its OWN parser (`parse_units_krk`) — no Kraken-internal deps → placement-free, self-contained. No new Project.toml deps (units only removed deps it didn't need; v0.3 has all).

**Validated**: `Kraken.Units` loads + `compile` OK; standalone units 277/277; full suite **34560 pass (= 34283 + 277) / 7 pre-existing fails / 0 errored** → graft clean, zero regression. NOT committed yet (awaiting user).

**NEXT = the actual WIRING (the feature the user wants), separate mission**: (a) geometry's `GeometryDescriptor` (src/geometry/descriptor.jl: type/blockage/q_wall_dist/stl_hash/is_solid) → units' `compile(geometry=…)` duck-typed contract (align/add kappa_max/L_up/L_down); (b) STL physical→LU: a `.krk` way to give STL + domain in physical units + a reference length + resolution → units computes `dx_real = L_phys/R_LU` → geometry's `_voxelize_stl_region` applies `scale = 1/dx_real`. Closes the user's "STL dim vs adim" point. The runner would call units `compile` → feed geometry the scale.

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

## 2026-05-30 — M7a (thermal natural-convection vs de Vahl Davis 1983) — solver VALIDATED  [KRK-SHIP-001 Phase C, ∥ to M-GEO-3a which runs in a SEPARATE session]

User chose "both cases" for M7: differentially-heated cavity (rigorous gate) + bottom-heated Rayleigh-Bénard (qualitative). M7 runs on `dev/units-module` / worktree `Kraken.jl-units` — gate-1 provenance check CONFIRMED the thermal solver (`src/kernels/{thermal_2d,thermal_3d,fused_thermal_2d}.jl` + `src/drivers/thermal.jl::run_natural_convection_2d`/`run_rayleigh_benard_2d` + `examples/rayleigh_benard.krk`) IS present on units (my "units branched before thermal" assumption was WRONG — it has the full thermal stack). No port needed; same track as M6.

**SOURCE CONTRADICTION resolved by user**: `examples/rayleigh_benard.krk` runs `run_rayleigh_benard_2d` (bottom-heated, periodic) but its comment + the ship-plan's "Wan et al. 2001" ref are the DIFFERENTIALLY-HEATED CAVITY (`run_natural_convection_2d`, which already computes Nu at the hot wall, lines 160-167). User picked BOTH.

**RESULT — solver physically correct, matches de Vahl Davis 1983 within the 1% Nu / 2% velocity gate at adequate resolution:**
- `run_natural_convection_2d(N, Ra, Pr=0.71)` ν=0.05 fixed, β_g=Ra·ν·α/(ΔT·H³) — CONSISTENT with M5's `β=Ra·ν·α/R³` (M5↔driver agree on physics; full `driver_kwargs` integration still deferred).
- Monotone Ra^(1/4)-BL convergence. Ra=1e5 (binding) Nu err: N=128 +4.71% → 192 +3.09 → 256 +2.19 → 320 +1.31 → **384 +0.79% PASS** (u* +0.23%, v* −0.37%). Ra=1e4 passes from N=320 (+0.93%). N=384 needs ~2.16M steps (N²-scaled default 1.44M under-converges → 1.05%). ~3 min/run Metal.
- **Metal F32 is TRUSTWORTHY for natconv Nu**: F32-vs-F64 Nu delta 0.07-0.09% at N=128 (<<0.3%); F32 N=192 Ra=1e5 ≡ F64 to 0.15%. → high-res ladder done on Metal F32 per user request.
- **F32 FLOOR at low Ra (durable finding → memory candidate)**: Ra=1e3 buoyancy force β_g∝1/N³ ~1e-9 LU sinks below F32 epsilon at N≥320 → flow COLLAPSES (N=384: u* −88%, Nu −10.7%, steady-verified; CSV shows it explicitly). **Ra=1e3 must use CPU F64** (cheap). CLOSURE DONE (already on disk when Boss tried to dispatch it — user said "check", correctly rejected a redundant run; the `m7a_closure_ra1e3_f64.jl` + CSV row pre-existed): **Ra=1e3 N=192 F64 = Nu +0.79%, u* +0.57%, v* +0.59% → PASSES**.

**LEG-1 (published reference de Vahl Davis 1983) = GREEN. All 3 Ra pass the 1%/2% gate** with the per-Ra recipe: Ra=1e3 F64 N=192 (+0.79%), Ra=1e4 Metal-F32 N=320 (+0.93%), Ra=1e5 Metal-F32 N=384 (+0.79%). CSV `bench/thermal_rheotool/kraken_natconv_results.csv` has 21 rows (cpu_f64 + metal_f32, `backend` column) — Boss-verified by reading the CSV directly.

## 2026-05-30 — M7 legs 2+3 (user asked "jambe 2 + .krk")

**LEG-3 (`.krk` repro) = GREEN, Boss-verified.** Added a `natural_convection_2d` Preset to `kraken_parser.jl::_expand_preset` (14 lines, mirrors `rayleigh_benard_2d`) + `examples/natural_convection.krk`. The runner already dispatches by NAME (`simulation_runner.jl:594` `occursin("natural_convection",name)→run_natural_convection_2d`), so only the preset + .krk were needed. Boss re-ran it: `run_simulation("examples/natural_convection.krk")` → Nu=2.2997 at Ra=1e4 N=128 (≡ the M7a N=128 datapoint). Differentially-heated-cavity .krk = `Boundary west/east wall T=1.0/0.0` + plain north/south walls. NOT committed (untracked: `examples/natural_convection.krk`; modified: `src/io/kraken_parser.jl` +14/-2). **Convention (memory candidate)**: thermal `.krk` dispatch is by `Simulation` name substring — preset case name must contain the driver tag.

**LEG-2 (OpenFOAM cross-check) = corroboration ESTABLISHED, but with an unresolved Ra=1e4 OF-case anomaly (NOT a Kraken issue).** A first OF Department set up + solved 3 `buoyantBoussinesqSimpleFoam` differentially-heated cavities (image `microfluidica/openfoam:latest`, OF v2512; Ra mapping via gravity g=Ra·ν·α; hot-left/cold-right ΔT=1K) THEN **BAILED before extraction** (3rd OF bail-out — left 3 containers running; Boss took over: `docker stop` the over-running Ra=1e5 at Time=8711 since it was steady (residuals ~3e-5, endTime=12000 was overkill), waited, then a 2nd extraction Department (with all Boss findings baked) succeeded WITHOUT bail). Converged Nu (snGrad, de Vahl Davis normalization):
- Ra=1e3 (128²): OF **1.111** vs dVD 1.117 / Kraken 1.126 → **−0.56% ✓**
- Ra=1e4 (128²): OF **1.986** vs dVD 2.238 → **−11.3%** (ANOMALOUS)
- Ra=1e5 (192²): OF **4.482** vs dVD 4.509 → **−0.60% ✓**

**Boss judgment: OF corroborates dVD≈Kraken where the case is well-posed (Ra=1e3, 1e5 both <0.6%). The Ra=1e4 −11% is NOT "uniform-mesh under-resolution" (the 2nd Department's claim) — that's physically inconsistent**: Ra=1e3 (same 128²) is fine and Ra=1e5 (192², THINNER BL) is fine, so a thicker-BL Ra=1e4 at 128² should be BETTER resolved, not −11%. Decisive tell: at Ra=1e4 OF gives Nu −11% BUT velocities **+20%** (u*=19.5 vs dVD 16.2) — low-Nu+high-velocity is internally contradictory for a converged buoyant cavity → the Ra=1e4 OF case is **under-converged (endTime=5000 too short) or a per-case setup defect**, isolated to that case. Kraken at Ra=1e4 is independently GREEN (+0.93% vs dVD, leg-1) so this is purely an OF-reference artifact. **OPEN**: re-run ONLY the Ra=1e4 OF case (192² + longer endTime) to close the table — deferred, user to decide (corroboration already established at 2/3 Ra).

**Corrected OF v2512 coded-FO recipe (durable → memory candidate, supersedes my wrong "root required")**: run coded functionObjects NON-root (`-u 1000` + writable `HOME=/tmp/ofhome`); v2512 `dynamicCode::checkSecurity` BLOCKS compilation under root even with FOAM_ALLOW. Need `-e FOAM_ALLOW_SYSTEM_OPERATIONS=1`; the `-allow-system-operations` FLAG is invalid in v2512. Use `bash -c` (NOT `bash -lc` — login shell breaks the OF lib path). Coded FO must be in the case `system/controlDict` `functions{}` (NOT `postProcess -dict <file>` — that finds zero functions, runs nothing). **ALSO: the coded FO in controlDict makes the SOLVER itself abort under root (checkSecurity at "Starting time loop") — solve with the coded FO REMOVED (gradT built-in only), then re-add nuExtract for the post-hoc non-root extraction.** snGrad≈volGrad Nu at full convergence (the earlier "volGrad 1.54" was a stale t=3000 grad(T) field). Deliverables untracked: `bench/thermal_rheotool/of_cavity_ra1e{3,4,5}/` + `of_natconv_results.csv` + `of_natconv_README.md`.

## 2026-05-30 — M7 "A" (re-run Ra=1e4 OF at 192²): the −11% is a CONVERGENCE pathology, NOT mesh — chase abandoned, corroboration stands at 2/3

User picked "A puis page": re-run ONLY the Ra=1e4 OF case at 192² to close the table. **Result FALSIFIED the mesh hypothesis** (mine AND the extraction Department's): at 192² Ra=1e4 Nu=**1.551** (−31%), WORSE than 128²'s 1.986 (−11%). Refinement moving AWAY from dVD=2.238 is unphysical for a converged solution. Decisive proof = **Nu still drifting between writes**: Nu@5000=1.621 → Nu@6000=1.551 (same 192² mesh), u*=10.6→13.4, v*=14.0→16.0; residuals plateau ~8e-5 above the 1e-5 target. → **the OF buoyant cavity at Ra=1e4 does NOT reach steady state in 6000 SIMPLE iters** (more DOF at 192² → converges even slower than 128²/5000). True steady (Nu→2.24) would need ~15-20k iters or a transient-to-steady run (~1-2h more).

**Boss decision: STOP the OF chase** (disproportionate for a corroboration leg; I already spent ~20 turns Boss-direct in the OF Docker weeds = the Boss-in-the-debugger anti-pattern). **Leg-2 corroboration STANDS at 2/3**: OF≈dVD≈Kraken at Ra=1e3 (−0.56%, converged) and Ra=1e5 (−0.60%, ran ~8000 iters/~1h). Ra=1e4 OF is documented honestly as **under-converged** (1.55–1.99 drifting). **Kraken's Ra=1e4 is independently GREEN vs dVD (+0.93%, leg-1)** — the OF gap is purely an OF-reference convergence artifact, zero bearing on Kraken. New case `bench/thermal_rheotool/of_cavity_ra1e4_n192/` (untracked).

**Anti-pattern lesson (→ reinforces [[feedback_monitor_antipattern]] family)**: the Boss should NOT hand-debug OpenFOAM Docker invocations turn-by-turn. After the 1st OF Department bailed, recovery should have been ONE re-dispatched extraction Department (which DID work) + accept its verdict — not ~20 Boss-direct turns chasing env-sourcing / coded-FO / convergence. Delegate verbose tool-iteration; keep Boss context for judgment.

**Process wins this session**: ALL anti-bail Departments (M7a CPU, M7a-Metal, M7a-closure, M7b-extract, M7c page, M7-3D) completed WITHOUT bail-out — the front-loaded "Bash foreground, WAIT for return, Read deliverable before report, NO Monitor" contract works every time. The ONLY bail was the 1st OF SOLVE Department (long docker run → Boss recovered via background `docker wait`). Lesson: anti-bail framing is reliable for fast/extraction missions; for LONG solver runs, Boss-direct `run_in_background` (harness re-invokes) beats delegating the wait.

## 2026-05-30 — M7-3D (thermal natural convection, CUBIC cavity) — solver VALIDATED, resolution-limited at high Ra. + sim-openfoam skill updated.

User: "valider le 3D" + "informer le skill /of-rheotool". 

**sim-openfoam skill EXTENDED** (`~/.claude/skills/sim-openfoam/SKILL.md` Common gotchas): new subsection "ESI v2512 full image + coded functionObjects" — captures the ~20-turn OF debug (image choice microfluidica-vs-opencfd, `bash -c` not `-lc`, the 4 coded-FO traps incl. non-root compile + solver-aborts-under-root, snGrad≈volGrad-at-convergence, SIMPLE-plateau-above-residualControl). `/of-rheotool` command delegates to sim-openfoam so it's informed.

**3D result (`run_natural_convection_3d`, src/drivers/thermal.jl:381 — cube hot-west/cold-east/adiabatic, D3Q19 Boussinesq-in-y, COMPUTES Nu at hot wall; gate-1 provenance confirmed)**: Metal F32, N=96, vs **Tric et al. 2000** (spectral, Pr=0.71: Nū=1.070/2.054/4.337) + Fusegi 1991:
- Ra=1e3: Nu=1.0855 (**+1.45%** ✓), Ra=1e4: 2.1233 (+3.36%), Ra=1e5: 4.6098 (+6.29%).
- **Monotone convergence toward Tric** (Ra=1e5 binding: N=48 +13.8% → 64 +9.9% → 96 +6.3%, ~÷2 per refinement) → residual = discretization not a bug. Steady-verified. **F32≡F64 to 0.04% in 3D** (buoyancy-force underflow does NOT bite — N stays small so β_g∝Ra/N³ ~1e-5..1e-6 ≫ F32 eps, UNLIKE 2D where N=384 collapsed Ra=1e3). Kraken slightly over-predicts (under-resolved hot-wall thermal BL on uniform mesh).
- **<2% at Ra=1e5 needs N≥128** (~30-40 min local Metal) or N≥192 on Aqua (H100 CUDA F64). NOT submitted. CSV `bench/thermal_rheotool/kraken_natconv_3d_results.csv` + `bench/scratch/m7_3d_natconv_run.jl` (untracked).

**Verdict: 3D thermal solver is VALIDATED** (correct + converging to the spectral reference; Ra=1e3 GREEN). Resolution story at high Ra, not a solver defect. User chose (A) document+preset+commit.

**DONE + COMMITTED (2026-05-30)**: `85bc33c3e` feat (3D `.krk` dispatch = a `setup.lattice === :D3Q19` branch INSIDE `_run_thermal` → `run_natural_convection_3d`; the top-level runner sends `:thermal` to `_run_thermal` BEFORE the D3Q19 check, so the 3D branch must live there + `natural_convection_3d` preset N=48³ + `examples/natural_convection_3d.krk`) + `ded05179d` docs (3D page section + 3D CSV). Boss-verified: 3D `.krk` → `uz` present (routes 3D), 2D `.krk` → `uz` absent (no regression). **M7 (2D+3D) FULLY COMPLETE — 4 commits this session on `dev/units-module`, no push: `123130590`+`783f2ebba` (2D) + `85bc33c3e`+`ded05179d` (3D).**

## 2026-05-31 — SPLIT-RUNNER DONE + COMMITTED (`22b62b8e0`) — extracted BC + IO from simulation_runner.jl; + found a PRE-EXISTING Cavity-3D failure on units

User asked "pourquoi simulation_runner fait des milliers de lignes? modulaire?" → I clarified: on `dev/units-module` the runner is **616 LOC** (the "thousands" = `dev/v0.3-campaign`'s 2711-LOC runner which ALSO carries the STL/LI-BB/curvilinear/multiblock/GMSH stack — that branch is hot with the parallel M-GEO session, so a SPLIT there = merge-hell). User said "delegate" → I dispatched a SPLIT on units (safe, quiescent).

**SPLIT result (`22b62b8e0` refactor)**: moved `BoundaryHandler` + BC appliers → NEW `src/bc/boundary_conditions.jl` (176 LOC, establishes the §6 `src/bc/` dir); VTK/PNG/GIF helpers → `src/io/runner_output.jl` (112 LOC). Runner **616→332 LOC** (under 500 soft cap). Behavior-preserving (verbatim moves). **`_apply_geometry!` deliberately NOT moved** (src/geometry/ doesn't exist on units — it's a v0.3 thing; deferred). Includes MUST be before `simulation_runner.jl` in Kraken.jl (BoundaryHandler used in runner signatures → include-time resolution).

**Department socket-DROPPED at ~18 min** (had created the 2 files + trimmed runner BC, but NOT wired the Kraken.jl includes NOR removed the 7 duplicated IO functions left in the runner → "Method overwriting" precompile error). Boss finished: added the 2 includes + deleted the IO dups. Verified: precompile clean, `test_simulation_runner` 24/24, 2D+3D `.krk` smoke unchanged.

**DISCOVERY (pre-existing, NOT the split, NOT M7)**: the units FULL suite is **378 pass / 3 fail / 0 error** — the 3 fails are ALL `test_cavity.jl:42 "Cavity 3D basic convergence"` → **NaN** (`run_cavity_3d` at N=16, ν=0.01, ω≈1.89, the marginal-ω BGK config). PROVEN pre-existing by `git stash -u` + re-run on clean M7 HEAD `ded05179d` → identical 3 NaN fails. **Contradicts the M6 boss-note claim that `test_cavity_3d` was "moved to the stable envelope (32³ Re=100)"** — `test/test_cavity.jl` is STILL at the divergent N=16/ω≈1.89 config (the M6 move must have been a separate bench script, not this test). **FIXED + COMMITTED (`e1b2bec6f`, user said "fix")**: moved the test to a stable BGK envelope — N=16, ν=0.1 (ω=1.25, Re=16); lid velocity now imposed EXACTLY (mean ux/u_lid = 1.0, matching the M6 "exact at 16³/Re16" claim). **Units FULL suite now 381/381 green** (Boss-verified `runtests.jl` exit 0). dev/units-module is fully GREEN end-of-session: 6 commits total (`123130590`+`783f2ebba`+`85bc33c3e`+`ded05179d` M7 2D+3D, `22b62b8e0` SPLIT, `e1b2bec6f` cavity fix), no push.

**PROCESS HAZARD (→ memory candidate, multi-worktree)**: the Bash cwd does NOT reliably persist as the units worktree — one `git add/commit` (without `cd`/`-C`) silently ran in the DEFAULT worktree (`Kraken.jl` = slbm-paper) and found "no changes" (harmless this time, but it could have staged the wrong tree). **For any git op across the 14 Kraken worktrees, ALWAYS use `git -C <abs-worktree>` (or explicit `cd` each call)** — never rely on persisted cwd. Verified all M7/SPLIT commits DID land on units (HEAD `e1b2bec6f`, has `src/bc/`).

## 2026-05-31 — kraken_parser.jl SPLIT DONE (stages 1+2, `d84424e49`+`3173ae4cf`) — the #1 oversized file 2036→464 LOC

User: "nouvelle session M8 + 2" → M8 (VE benchmark) goes to a NEW session they opened; THIS session took option (2), the parser SPLIT. Staged (the 2036 LOC was too big for one Department pass given the runner-SPLIT socket-drop):
- **Stage 1 `d84424e49`**: sanity-checks → `src/io/krk/sanity.jl` (430) + LBM-params recommender → `src/io/krk/lbm_params.jl` (294). Parser 2036→1314.
- **Stage 2 `3173ae4cf`**: 28 directive-parsers + helpers → `src/io/krk/directives.jl` (712) + sweep/presets → `src/io/krk/presets.jl` (124). Parser 1314→**464 (under 500 soft cap)**.

`kraken_parser.jl` now = 11 setup structs + tokenizer + core load/parse orchestration only. **Both Departments GREEN, no bail** (the staged size + anti-bail held). Boss-verified each: 0 duplicate defs left behind (the runner-SPLIT dup trap), `using Kraken` clean (no Method overwriting), full units suite **381/381**. Include order (call-time resolution lets the core call the moved fns): `kraken_parser → directives → presets → lbm_params → sanity → Units` (lbm_params before sanity: shared `_probe_U_ref`; lbm_params before nothing-needs-it but kept after directives whose `_apply_setup_helpers!` calls `_probe_U_ref` at runtime).

**§6 milestone: `src/io/krk/` established** (mandate target for the .krk parser). Both M1-audit top SPLIT candidates now done this session: runner (616→332, src/bc/) + parser (2036→464, src/io/krk/). **OPEN micro-items (non-blocking)**: directives.jl is 712 (12 over the 700 HARD cap — verbatim docstrings; shed via pulling `build_rheology_model` into a physics file at the constitutive-extraction mission). **Stage 3 DONE (`016f326f2`, user said "oui enchaine")**: structs → `src/io/krk/types.jl` (150) + `git mv` core → `src/io/krk/parser.jl` (316). The .krk parser now FULLY lives under `src/io/krk/` = 6 files (types 150, parser 316, directives 712, presets 124, lbm_params 294, sanity 430). Include order `types FIRST → parser → directives → presets → lbm_params → sanity` (struct sigs resolve at include time; everything else call-time). **Test-coupling caught + fixed**: `test/test_kraken_parser.jl` has a standalone-include fallback guarded by `if !isdefined(:KrakenExpr)` (skipped under runtests since `using Kraken` defines it — that's why stages 1-2 didn't break it); Stage 3 updated its paths to the full krk set → standalone now 149/149 too. Module suite 381/381. **kraken_parser.jl SPLIT COMPLETE: 2036 LOC → 6 files in src/io/krk/, all under cap except directives.jl (712, 12 over hard — micro-shed `build_rheology_model` later).** 3 commits `d84424e49`+`3173ae4cf`+`016f326f2`. Session modularity total: runner + parser, the M1-audit top-2 oversized files, both fully decomposed; src/bc/ + src/io/krk/ established.
- **RB qualitative PASS**: `run_rayleigh_benard_2d(Nx=128,Ny=64,Ra=1e5)` max|u|=0.157, vol-avg Nu≈5.32, no NaN, vigorous rolls (Ra≫Ra_c≈1708).

**Deliverables (untracked, units worktree)**: `bench/thermal_rheotool/kraken_natconv_results.csv` (5 cpu_f64 + 15 metal_f32 rows, `backend` column) + `bench/scratch/{m7a_natconv_run.jl,m7a_metal_run.jl}`. NOT committed (validation artefacts; commit when M7 page lands).

**M7 remaining (mandate §3 three-leg gate)**: leg1 published ref (de Vahl Davis) ≈ DONE (pending cheap Ra=1e3 F64 closure); **leg2 = OpenFOAM `buoyantBoussinesqPimpleFoam`/`rheoHeatFoam buoyantCavity` cross-check (mandate §6 named target) — NOT done, heavy Docker (apply M6 lessons: background docker wait, NO Monitor-bail, `cd /case` inside `bash -c`)**; leg3 = cavity `.krk` repro (only `rayleigh_benard.krk` exists; a differentially-heated-cavity `.krk` likely needs a Preset/parser touch — separate scope from the trivial F64 closure). Then M7c benchmark page `docs/src/users/benchmarks/thermal-rayleigh-benard.md`.

**Process**: both M7a Departments (CPU + Metal) completed WITHOUT bail-out — the front-loaded anti-bail ("Bash+timeout foreground, WAIT, Read CSV before report, NO Monitor") WORKED. Replicate verbatim. SendMessage tool is NOT available in this harness → cannot continue a prior agent; re-spawn fresh pointing at the on-disk script.
