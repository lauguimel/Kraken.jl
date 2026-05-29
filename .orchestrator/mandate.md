# MANDATE — Kraken.jl

This is the Boss's source of truth for the Kraken.jl project. Update it as
missions complete, decisions are taken, branches converge, or scope shifts.
Departments read only the relevant mission slice; Engineers never read this
file directly. **Only the Boss writes here** (orchestrator single-writer rule).

Bootstrap date: 2026-05-28. Initial population by Boss from user's verbal mandate.

Paired skill: [`kraken-architect`](~/.claude/skills/kraken-architect/SKILL.md) (machine-global worktree map + discipline).

---

## 1. High-level objective

**Build a user-friendly LBM (Lattice Boltzmann Method) accessible without writing Julia directly.**

Concretely:

- The user writes `.krk` files (a DSL); the code does the rest.
- A single codebase supports **2D, 3D, and axisymmetric** simulations.
- A single codebase runs on **CPU and GPU**, backend-agnostic (CPU / Metal / CUDA / etc.) via `KernelAbstractions.jl`.
- **Multiphysique**: Newtonian → viscoelastic → multiphase → AMR → STL geometry → … the code is **modular and pluggable**; new physics is a new module, not a fork of the trunk.
- **Documentation is dual-track and first-class**:
  - **Human docs** — hosted on GitHub Pages, narrative, examples-driven.
  - **LLM docs** — module-level implication map, machine-readable, kept in sync with the code (so future LLM sessions can navigate without re-deriving).

Done at the project level when:

- A non-Julia user (rheology lab, undergrad, industry user) can run a viscoelastic 3D AMR case from a `.krk` file without touching Julia source.
- A new physics module (e.g. electroviscous) drops in as a separate file/dir without surgery on the trunk.
- Performance stays within ~2× of hand-tuned reference implementations on the same backend.

## 2. Out of scope

- Replacing OpenFOAM/RheoTool for general FV CFD. Kraken is LBM-first.
- Pre-CAD geometry modelling — Kraken consumes STL / SDF / analytical shapes; does not author meshes from CAD.
- Non-isothermal coupling beyond what the LBM extension supports natively (no full DG/FEM thermal solver).
- Generic ML/AI training — out of scope, even if specific learned closures may be loaded as modules.
- Heavyweight GUI — `.krk` + headless postproc (VTK / ParaView) is the interface. Web UI is a nice-to-have but not a goal.

## 3. Constraints (non-negotiable)

### Architecture

- **Modularity is mandatory**. Boundary conditions, geometry, constitutive models, stencils, IO, `.krk` parsing, GPU/HPC execution, and 2D/3D ports MUST go through explicit specs/interfaces with reusable helpers and narrow drivers. Hard-coded shortcuts are allowed only in named canaries/audits and must be factored out before promotion.
- **Backend-agnostic at the kernel layer**. No backend-specific code in the physics modules. Backend dispatch lives in `src/backend/`.
- **`.krk` is the canonical interface**. Anything a user can do with code must also be doable with `.krk`. Internal helpers may be Julia-only.
- **Doc-as-code**. Every public module has both a human doc page and an LLM-friendly module implication map.

### Performance

- **GPU compatibility is preserved at all times**. No dynamic local allocation, host callback, or dispatch-heavy logic inside kernels. Fixed compact stencils or precomputed coefficients for cut/near-wall paths.
- **No long CPU runs are valid for development** — local coarse/Metal canaries; HPC A100/H100 for longer validation.
- Stay within ~2× of hand-tuned reference implementations on the same backend (rule of thumb; concrete benchmarks per module).

### Engineering hygiene (LLM-tractable codebase)

- **File-size budget**: ≤500 LOC soft / ≤700 LOC hard. Above the hard ceiling, briefs must scope as side-effect-only patches OR open a SPLIT mission first.
- **One file = one concern**: geometry / BC / solver / stencil / physics / constitutive / driver — never mixed.
- **Symbol-anchored references in briefs**: cite the symbol name AND the line range; line numbers rot the instant any sibling code is added.

### Process

- **Validation ladder, not skip-to-benchmark**: analytical unit → semi-analytical patch → coarse macro → production benchmark. Cd improvements do NOT prove correctness unless the lower-level canary they close is identified.
- **Branch contract** per active branch in `docs/agent/branch_contract.md`. Codex `kraken-branch-governor` reads it.
- **Commit hygiene**: commit only after a meaningful green canary or a clear diagnostic freeze. Stage only intentional files. Never push without explicit user confirmation.

### Validation per public module (NON-NEGOTIABLE)

Every public module (geometry, mesh, BC family, constitutive model, LBM operator, IO subsystem, `.krk` parser, units module, …) MUST ship the following before its PR is considered complete. A mission whose deliverable is "module X" is **not** done until all three exist:

1. **Analytical reference** — where one exists (Poiseuille profile, Couette steady-state, Taylor-Green decay rate, oscillating shear amplitude, Rayleigh-Bénard Nu(Ra), Oldroyd-B startup curve, etc.). The test reproduces the analytical answer within a documented tolerance (typically 1% on the relevant norm). Lives under `test/analytical/<module>_<case>.jl`.

2. **RheoTool numerical benchmark** — exact numerical comparison against a RheoTool case from `~/Documents/Recherche/Codes CFD/rheotool/rheoTool/of90/tutorials/<solver>/<case>/`. The brief picks the matching tutorial (e.g. Oldroyd-B cylinder → `rheoFoam/Cylinder/Oldroyd-BLog`; viscoelastic cavity → `rheoFoam/Cavity/Oldroyd-BLog`; non-isothermal channel → `rheoHeatFoam/channel/PTTLog`). Comparison artefact lives in `benchmarks/results/rheotool_compare/<module>/<case>/` and contains:
   - CSV of both solvers' outputs at matching probe points
   - field-by-field error norms (L1, L2, Linf)
   - a comparison plot in `docs/src/users/benchmarks/<module>.png`
   - **Tolerance**: ≤1% on integrated quantities (Cd, friction, flow rate, N1) and ≤5% on local field maxima, unless the brief justifies a wider band. Qualitative agreement ≠ done.

   If RheoTool does NOT cover the case (e.g. AMR-specific, GPU multi-block), the brief must (a) name an alternative reference solver (Palabos / OpenLB / Basilisk / published paper data) AND (b) justify why RheoTool is silent on the case. Approval from the Boss required before substitution.

3. **`.krk` reproduction case** — at least one `.krk` file in `benchmarks/krk/<module>/` that triggers the analytical AND RheoTool comparison via the standard runner. The `.krk` file is part of the user-facing API surface: if a user can't reproduce the benchmark from a `.krk`, the module is not user-friendly per the mandate §1 objective.

The three artefacts together are the minimum acceptance gate. Skipping any of them = mission RED. No exceptions for "small" modules.

## 4. Architecture decisions (ADRs)

Append new entries; never rewrite history.

| Date       | Decision                                                            | Rationale                                                    |
|------------|---------------------------------------------------------------------|--------------------------------------------------------------|
| 2026-05-28 | Adopt orchestrator pattern with mandate + memory layers in `.orchestrator/` | Multiple worktrees + 25+ branches; need single source of truth |
| 2026-05-28 | Kraken-AMR.jl status — open question (relation to dev/feat/amr-port-sr to clarify) | See §8 Open questions                                       |
| 2026-05-28 | **Ship plan KRK-SHIP-001 adopted** — see `.orchestrator/ship-plan.md` (15 missions, 9-10 sessions min) | Department-drafted, Boss-approved |
| 2026-05-28 | **LU/real-units module → `src/units/`** (NOT `src/sim_planner/` from M62b legacy) | Aligns with §6 modular target vocabulary; M62b brief to be ported as M4 with this rename |
| 2026-05-28 | **Thermal phasing: Boussinesq in ship-1, full thermal (mu(T), k(T), dissipation) in ship-2** | Department warned full thermal in single ship was scope-creep; split into 2 releases keeps each ship minimum-sessions |
| 2026-05-28 | **Merge target for ship-1 = `dev/v0.2-multiphysics`** (new branch from `main`), not `main` directly | Intermediate release-candidate branch; stabilise then merge to main once green |
| 2026-05-28 | **Scope-cut explicit for ship-1**: axisymmetric, backend module factoring, multiphase, AMR, STL → all deferred | Need to fit 9-10 sessions; remaining axes get their own ship plans |
| 2026-05-28 | **Branch retirements approved**: `lbm`, `refinement-patches-dev`, `dev/v0.2-architecture` | All merged or superseded (lbm/refinement → into main; v0.2 → by v0.3-campaign). Execution pending user per-branch confirmation. |
| 2026-05-28 | **Validation-per-module non-negotiable codified in §3**: every public module needs analytical + RheoTool numerical benchmark + `.krk` reproduction | User flagged that earlier wording ("validation ladder") was too generic and didn't anchor RheoTool numerical match as mandatory. Now explicit: 1% on integrated quantities, 5% on local maxima, no qualitative pass. |
| 2026-05-29 | **M2 executed**: `lbm` (local+remote `origin/lbm`) and `refinement-patches-dev` retired. **`dev/v0.2-architecture` retirement REVERSED** — kept deferred. | The 2026-05-28 "superseded by v0.3-campaign" premise was inaccurate: commit `74c39c633` carries `src/runtime_specs.jl` (502 LOC specs→lowering→plans skeleton, conceptual ancestor of the M3/M4 `src/units/` SimulationPlanner) plus 10 v0.2 milestone bilans, none reachable from any active branch. v0.3-campaign reused `axisymmetric.jl` but NOT `runtime_specs.jl`. Re-evaluate retirement after M3 design lands. |
| _(append)_ |                                                                     |                                                              |

## 5. Branch map (living)

State as of 2026-05-28. Update on every merge / branch creation / branch retirement.

### Stable / reference

| Branch | Last activity | Role | Status | Worktree |
|--------|---------------|------|--------|----------|
| `release/v0.1` | 2026-05-14 | Stable release branch — public users land here | Stable | `Kraken-main/` |
| `main` | 2026-05-14 | Trunk — release candidate integration | Stable | (none currently) |
| `gh-pages` | 2026-03-09 | Docs deploy (auto) | Auto | (none — never edit) |

### Active development

| Branch | Last activity | Role | Status | Worktree |
|--------|---------------|------|--------|----------|
| `slbm-paper` | 2026-05-25 | Current HEAD — simplified-LBM paper companion code | Active (paper) | `Kraken.jl/` |
| `feat/amr-port-sr` | 2026-05-25 | AMR state-resident port (relates to Kraken-AMR.jl?) | Active | `Kraken.jl-amr-port-sr/` |
| `dev-viscoelastic` | 2026-05-25 | Viscoelastic extension | Active | `Kraken.jl-viscoelastic/` |
| `docs/module-architecture` | 2026-05-22 | Module-arch documentation work | Active | `Kraken.jl-docs-arch/` |
| `dev/fvfd-core` | 2026-05-16 | FVFD core development | Active | `Kraken.jl-fvfd-core/` |
| `dev/v0.3-campaign` | 2026-05-07 | v0.3 release campaign | Active | `Kraken.jl-v0.3-campaign/` |
| `dev/axisymmetric-true-lbm` | 2026-05-14 | True axisymmetric LBM (mandate target) | Active | (none currently) |
| `deforestation` | 2026-04-21 | Cross-project use — Deforestation project consumes Kraken | Active (downstream) | `Deforestation/kraken/` |

### Probes / experimental / prunable

| Branch | Last activity | Role | Status | Worktree |
|--------|---------------|------|--------|----------|
| `dev/refinement-perf-opts` | 2026-05-18 | Refinement perf opts experiment | Prunable | `Kraken.jl-refinement-perf/` (prunable) |
| `dev/kraken-e-fvfd-blocks` | 2026-05-17 | Kraken-E FVFD blocks experiment | Prunable | `Kraken.jl-kraken-e-blocks/` (prunable) |
| `probe/amrd-golden-cylinder` | 2026-05-18 | AMR-D golden probe | Prunable | `Kraken.jl-amrd-golden/` (prunable) |
| `audit/modularity-performance-axisym` | 2026-04-30 | One-off audit, kept for reference | Audit | (none) |
| `dev/v0.2-architecture` | 2026-05-14 | Carries unique `src/runtime_specs.jl` (specs→lowering→plans skeleton, ancestor of M3/M4 `src/units/`) + v0.2 milestone bilans | **KEEP (deferred)** — retirement reversed 2026-05-29, see ADR §4 | (none) |

### Backup / diagnostic snapshots

| Branch | Role | Status |
|--------|------|--------|
| `backup-amrd-before-symptom-reset-20260512` | AMR-D backup before symptom reset | Recovery-only |
| `diag-amrd-symptom-pre-gate-20260512` | AMR-D diag-only snapshot | Recovery-only |

### Backport / merge candidates (review periodically)

- `slbm-paper` → `main`: paper-specific changes that have generic value (e.g. `.krk` helpers) should be cherry-picked into main before paper merge.
- `dev-viscoelastic` → `main`: when viscoelastic is stable enough (define "stable enough" via canary in branch_contract).
- `dev/fvfd-core` → `main`: when FVFD core stabilises.
- `feat/amr-port-sr` ↔ `Kraken-AMR.jl`: clarify direction first (see ADR open question).
- `release/v0.1` ← `main`: cherry-pick critical bug fixes only.

## 6. Modular architecture target

Target end-state. Use `mandate §6` to track migration progress.

| Module | Should live in | Lives today | Status | Analytical reference | RheoTool benchmark target |
|--------|----------------|-------------|--------|----------------------|---------------------------|
| `.krk` parser | `src/io/krk/` | `io/kraken_parser.jl` + `io/expression.jl` (NOT yet in target dir) | **FACTORED** — single concern, but HARD-oversized (2010 LOC) + mislocated → SPLIT+relocate candidate (M1 2026-05-29) | round-trip parse/serialise unit tests | n/a (DSL layer, no fluid) |
| LU / real-units conversion | `src/units/` | not yet created (M62b designed as `src/sim_planner/`) | **PLANNED** — dispatch as M4 | dimensional analysis closure tests | n/a (utility module — verified via downstream module benchmarks) |
| Geometry primitives (analytical, SDF, STL) | `src/geometry/` | `io/voxelizer.jl`, `io/stl_libb.jl`, `drivers/step_geometry_2d.jl` | **MIGRATING** — scattered across io/ + drivers/; no `src/geometry/` dir yet (M1 2026-05-29) | SDF distance closure, normals, intersection cases | n/a (geometry layer) — verified via BC benchmarks |
| Mesh (uniform / multi-block / AMR) | `src/mesh/` | `curvilinear/mesh.jl` + `curvilinear/generators.jl`, `refinement/*` | **MIXED** — multi-block mesh in curvilinear/, AMR mesh in refinement/; no `src/mesh/` dir yet (M1 2026-05-29) | grid count, connectivity invariants | n/a (verified via flow benchmarks) |
| BC — wall / periodic | `src/bc/{wall,periodic}` | `kernels/boundary_*`, `kernels/li_bb_*`, `fvfd/*boundary*` | **MIXED** — BC fused into collide kernels; no `src/bc/` dir yet (M1 2026-05-29) | Poiseuille profile (analytical) | `rheoFoam/Channel/Oldroyd-BLog` with `Wi=0, β=1` (Newtonian sanity) |
| BC — inflow / outflow | `src/bc/{inflow,outflow}` | inline in `simulation_runner.jl` (BoundaryHandler) | **MIXED** — inflow/outflow logic embedded in the 1843-LOC runner; no `src/bc/` dir yet (M1 2026-05-29) | flow rate conservation | `rheoFoam/Contraction41/Oldroyd-BLog` (inflow → outflow + recirculation) |
| Constitutive — Newtonian | `src/physics/newtonian` | functional (trunk) | needs RheoTool match | Poiseuille, Couette, Taylor-Green decay | `rheoFoam/Cavity/Oldroyd-BLog` with `Wi=0, β=1`; `rheoHeatFoam/buoyantCavity` (thermal coupling) |
| Constitutive — Oldroyd-B | `src/physics/oldroyd` | partial (dev-viscoelastic) | in flight | analytical Oldroyd-B startup curves | `rheoFoam/Cylinder/Oldroyd-BLog` (Cd=117.357 ref) AND `rheoFoam/Cavity/Oldroyd-BLog` |
| Constitutive — FENE-P / FENE-CR | `src/physics/{fenep,fenecr}` | _not started_ | _planned ship-2_ | analytical FENE shear closure | `rheoTestFoam/FENE-CR` (extensional) + `rheoFoam/OtherTests/Channel2D_VE` |
| Constitutive — Giesekus / PTT | `src/physics/{giesekus,ptt}` | _not started_ | _planned ship-2+_ | N1 ratio, shear-thinning slope | `rheoFoam/CrossSlot/PTTLog`, `rheoFoam/OtherTests/Channel2D_VE` |
| Constitutive — Thermal (Boussinesq) | `src/physics/thermal` | functional (dev-viscoelastic) | needs RheoTool match | Rayleigh-Bénard Nu(Ra=1e5), natural convection Nu | `rheoHeatFoam/buoyantCavity/Newtonian` |
| LBM operators (stream / collide / forcing) | `src/lbm/{stream,collide,forcing}` | functional (trunk) | needs match through above | mass / momentum conservation across one step | covered transitively via constitutive benchmarks |
| Backend dispatch | `src/backend/{cpu,metal,cuda}` | scattered today — _to factor_ | _deferred (ADR scope-cut)_ | per-backend bit-reproducibility tests | run any constitutive benchmark on each backend; field deltas ≤1e-12 |
| IO / output (VTK, JLD2, ParaView) | `src/io/{vtk,jld2,paraview}` | `io/vtk_writer.jl`, `io/postprocess.jl`, `drivers/viscoelastic_diagnostics.jl` | **MIGRATING** — VTK in io/, but diagnostics output leaks into drivers/ (M1 2026-05-29) | round-trip read/write tests | n/a (verified via downstream postproc consistency) |
| Doc generation — human + LLM-implication map | `docs/{src,agent}` | partial (`docs/agent/branch_contract.md` exists) | partial | n/a (doc layer) | n/a |

**Status legend**: `MIXED` (entangled with another concern), `FACTORED` (clean), `MIGRATING` (needs relocation/split to reach target dir). All target-module cells inventoried by M1 (2026-05-29); none remain un-inventoried.

M1 audit complete (2026-05-29): full classification of 99 `src/**/*.jl` files (dev-viscoelastic worktree) in `bench/scratch/m1_module_audit.md`. Headline: 28/99 files MIXED, dominated by physics↔lbm-operator fusion (constitutive math baked into collide kernels; `viscoelastic_spec.jl::AbstractPolymerModel` is the clean factoring seam). 11 files over the 700-LOC HARD ceiling; trunk SPLIT candidates: `kraken_parser.jl` (2010 LOC), `simulation_runner.jl` (1843 LOC). No `src/{bc,geometry,mesh,physics,backend,units,lbm}/` dirs exist yet — every §6 target is a net-new directory.

## 7. Merge procedure (pre-flight + execution)

For any merge between dev branches or into main:

### Pre-flight (all gates must pass)

1. **Worktree hygiene**: `git -C ~/Documents/Recherche/Kraken.jl worktree list --porcelain` — no uncommitted changes in source or target worktrees.
2. **Canary state**: source AND target branch each have a green canary (per `docs/agent/branch_contract.md` of each branch). Document the canary command in the brief.
3. **`.krk` compatibility audit**: list every `.krk` fixture / parser change in the source branch since divergence. If DSL semantics change, write an ADR entry §4 BEFORE merging.
4. **Module entanglement audit**: identify mixed-concern files in the divergence region. Flag for SPLIT mission if a merge would entrench mixing.
5. **Backport direction**: explicit. Default is `release/v0.1` ← `main` ← `dev/*`. Cross-dev cherry-picks need ADR justification.

### Execution

6. **Orchestrator dispatch**: the merge itself is a Department brief — not direct Boss execution. Brief lists:
   - Allowed edit zones: the conflict files only.
   - Forbidden actions: no semantic changes beyond conflict resolution.
   - Exit criterion: target-branch canary command + source-branch canary command (both must pass post-merge).
   - Runner: Codex with `kraken-branch-governor` loaded.
7. **Post-merge**: re-run the full validation ladder (analytical → patch → macro → benchmark) on the merged branch BEFORE declaring done.
8. **ADR**: append a §4 entry with the merge date, source → target, and notable resolved conflicts.

## 8. Open questions

- [ ] **Kraken-AMR.jl relation**: is `Kraken-AMR.jl` a separate fork that will eventually merge into Kraken.jl, an upstream import target, or a permanently separate project? Decide before any AMR port work proceeds.
- [x] ~~**`lbm` branch (2026-04-16)**: role? retire?~~ — RETIRED 2026-05-29 (local + `origin/lbm`)
- [ ] **`dev/v0.2-architecture`**: retirement REVERSED 2026-05-29 — carries unique `src/runtime_specs.jl` (M3/M4 `src/units/` ancestor). Decide final disposition after M3 design lands (port the skeleton into `src/units/` then retire, or keep as reference).
- [ ] **PRUNABLE worktrees**: `Kraken.jl-amrd-golden`, `Kraken.jl-kraken-e-blocks`, `Kraken.jl-refinement-perf` flagged by git. Confirm each: keep + unflag, or remove?
- [ ] **`.krk` DSL spec**: is there a versioned spec document? If not, write one and place at `docs/spec/krk-v1.md`.
- [ ] **LLM-doc format**: define the canonical format for the per-module implication map. JSON-Schema-described markdown? YAML frontmatter?
- [ ] **Backend coverage**: which backends are tested in CI today? Document, add missing.

## 9. Mission graph (active + planned)

Add missions here as they are picked up. Use orchestrator's mission template format.

### Mxx — _next mission goes here_

- **Status**: planned
- **Goal**:
- **Allowed edit zones**:
- **Exit criterion**:
- **Notes**:

## 10. Pointers

- Public docs site: GitHub Pages (link once stable URL is decided)
- Code: `~/Documents/Recherche/Kraken.jl/` (this repo)
- Cross-project consumer: `~/Documents/Recherche/Deforestation/kraken/` (worktree on `deforestation` branch)
- AMR-related separate repo: `~/Documents/Recherche/Kraken-AMR.jl/` (status to clarify — §8)
- Branch contracts: `docs/agent/branch_contract.md` per active branch
- HPC reference: `~/Documents/Clouds/UGA/Recherche/HPC/aqua/` (Aqua specs, gotchas)
- Codex skills for Engineers: `kraken-branch-governor`, `kraken-fvfd-operator-library`, `kraken-resource-integrator`, `kraken-port-fidelity`, `kraken-port-rewrite`, `kraken-amr-canary`
