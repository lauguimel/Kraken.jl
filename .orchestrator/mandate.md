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
| 2026-05-29 | **Units module spec frozen (KRK-UNITS-001, `docs/spec/units-v1.md` on `dev/units-module`)** — generalises M62b VE-only design to multiphysics: Newtonian + Viscoelastic concrete, Thermal-Boussinesq + GNF/multiphase/MHD as pre-registered stubs. Unblocks M4 (Phase-1 impl). | M3 done. Ends the M48/M59–M61 hand-coded-LU rabbit-hole; enforces §1 guardrail "no driver computes τ/ν/λ/u from Re/Wi/β by hand". Reconciles naming with `runtime_specs.jl` prior art (§9 of spec). |
| 2026-05-29 | **Units Phase 1 implemented (M4)** — `src/units/` on `dev/units-module`: Newtonian + Viscoelastic concrete, thermal/GNF/multiphase/MHD raising stubs. 165 tests green (Codex, verified independently). | Closes Phase B of KRK-SHIP-001. Module is compile-time-only LU↔non-dim owner; drivers opt in via `driver_kwargs(plan)`. Built on trunk-based branch without the VE solver — VE conversion is self-contained arithmetic; driver-integration Cd reproduction deferred to post-merge. |
| 2026-05-29 | **Units Thermal-Boussinesq Phase 2 (M5) + zero-edit contract PROVEN** — `physics/thermal.jl` un-stubbed; ONLY that file edited under `src/units/`; §8.t test asserts a VE plan is bit-identical with thermal concrete. 277 tests green. | Validates the mandate §1 "new physics = new module, zero trunk surgery" guarantee empirically. GNF/multiphase/MHD follow the identical seam (still stubs). Spec-then-impl (M3→M4) + pre-registered stub seam made M5 a clean one-file drop-in. |
| 2026-05-29 | **KRK-GEO base = `dev/v0.3-campaign`; M-GEO-2 completes commit `682e3f3c0` (mesh-field regression fix)** — `feat/geometry-stl` branched off `dev/v0.3-campaign`. Discovered `run_simulation` was broken for ALL `.krk` on the v0.3/paper lineage (slbm-paper, dev/v0.3-campaign, feat/amr-port-sr, docs/module-architecture): runner refs `setup.mesh` but `SimulationSetup` never got the field. dev-viscoelastic rejected as base (committed HEAD un-buildable — `include`s untracked `linalg_3d.jl`). | Base chosen for: builds clean + full STL+GMSH + release-aligned. The mesh-field fix is a **v0.3-release regression fix**, cherry-pick candidate to the other 3 lineage branches. Cartesian/STL chosen over GMSH/SLBM as the first end-to-end path (arbitrary geometry, GPU-friendly, no meshing). |
| 2026-05-29 | **STL / complex-geometry track OPENED — overrides the ship-1 "STL deferred" scope-cut** for a SEPARATE parallel track (KRK-GEO). User explicitly requested `GMSH/STL → Kraken`. Path chosen: **Cartesian immersed-boundary first** (STL→voxelize→is_solid mask→BB), GMSH/SLBM body-fitted second. First mission M-GEO-1 = end-to-end `.krk`+STL proof, validated against the analytical `Obstacle cylinder {}`. | User request 2026-05-29 is the authorisation department.md requires for any STL brief. Ship-1 multiphysics scope is UNCHANGED — this is a new track, not a ship-1 widening. Gate-1 cleared empirically: STL→mask path already wired (`_apply_geometry!` simulation_runner.jl:2009-2018); LI-BB q_wall for STL is NOT wired (follow-up, not in proof scope). |
| 2026-05-29 | **M6 (Cartesian cavity benchmark) DONE — GREEN 2D + 3D top-BC bug FIXED.** Lid-driven cavity validated vs Ghia 1982 + icoFoam (Docker `opencfd/openfoam-default:2412`, 128²) + Kraken BGK. icoFoam <0.5% L2(u) vs Ghia; Kraken 2-3% rel / ~1% abs-RMS (mandate §3 local-field ≤5%). 3D Zou-He top-BC omitted wall-parallel diagonal momentum (`parallel[6:9]`) → 1.5× lid overshoot; FIXED (commit `9674e1c4c` on `dev/units-module`, cherry-picked to `main` as `a5ce7c6f0`); deliverables `a524f7ded`. | Phase C opened (KRK-SHIP-001). Note: M6c-v2's "no MRT/TRT" was a branch-scoping error — a generic Newtonian TRT collision (`fused_trt_step!`, Λ=3/16) EXISTS in `src/kernels/fused_trt_2d.jl` on `slbm-paper`/`dev-viscoelastic` (absent from units/main); the "<1 %" gate is reachable by porting it (no new kernel; 2D-only). Real open item: (b) BGK marginal-ω instability at coarse grid (16³, ω≈1.89). icoFoam-Docker recipe reusable for M7. |
| 2026-05-29 | **M6 follow-up — cavity 2-3% was a half-cell COORDINATE bug, now <1%; wall-aware `axis_node_coords` adopted.** The Zou-He lid is on-node but the comparison hand-coded `(j-0.5)/N` (assumes halfway-BB) → half-cell first-order error. Mesh sweep (p≈0.85, NOT 2nd-order) + Couette control (machine-zero under corrected coord) + Mach test (Ma-independent) localised it; solver is exact, TRT≡BGK (irrelevant, port reverted). Fix: `src/axis_coords.jl::axis_node_coords(N;lo,hi)` (`78ed276c4`→main `9b9a97aa8`); corrected M6 `04188e067`. Kraken-vs-Ghia rel-L2 0.47/0.41/1.05%. | **Supersedes the "<1% needs MRT" note on the M6 row above.** New convention (mandatory): all profile/reference comparisons + coordinate output use `axis_node_coords`, never hand-coded `(j-0.5)/N`. Helper PENDING cherry-pick to lineage branches (deferred — dirty/active worktrees; batch or M14). |
| 2026-05-30 | **M-GEO-5 scoped to MASK-LEVEL 3D + descriptor fields (Option A); 3D STL FLOW runner carved out as M-GEO-6.** Boss recon (pre-dispatch) falsified the brief premise "3D path exists, just needs an example + scale": `run_simulation` D3Q19 dispatches ONLY `cavity_3d` (`_run_d3q19` throws otherwise) and `fused_trt_libb_v2_step_3d!` (exists+tested) is NEVER wired into the runner. The units→LU scale IS dim-agnostic but had nowhere to plug a 3D flow. User chose mask-level twin now, flow runner deferred. DONE `9acf4eb4d`, FF-merged to v0.3; full suite 34585 pass / 7 pre-existing / 0 errored. | Step-back trigger (mandate-gap) caught in Boss recon BEFORE any Codex dispatch. Mask twin proves the 3D scale; flow twin waits on M-GEO-6 (a real ~200-400 LOC runner path, not "wiring"). Descriptor fields: kappa_max=inverse-LU bbox proxy (conv. forced by audit `R_LU·kappa_max>0.5`), L_up/L_down from centroid (NaN-sentinel derive-else-respect), q_wall_dist=0.5-per-link to un-gate `audit_stl` early-return. 0 edits to `simulation_runner.jl`. |
| 2026-05-30 | **M7 (Thermal natural-convection RheoTool benchmark) DONE — GREEN.** Differentially-heated cavity vs de Vahl Davis 1983: Kraken Nu err **+0.79/+0.93/+0.79%** at Ra=1e3/1e4/1e5 (passes <1% Nu / <2% velocity gate). Per-Ra precision recipe: CPU-F64 N=192 at Ra=1e3 (Metal-F32 buoyancy force β_g∝1/N³~1e-9 underflows F32 at N≥320 → cell collapse), Metal-F32 N=320/384 at Ra=1e4/1e5 (Ra^¼-thinning BL needs N≈384 for 1% gate). OF `buoyantBoussinesqSimpleFoam` (ESI v2512) corroborates at Ra=1e3 (−0.56%)/Ra=1e5 (−0.60%); Ra=1e4 OF under-converged (SIMPLE stall, documented honestly — NOT a Kraken issue). `.krk`=`natural_convection_2d` preset. Committed `123130590`+`783f2ebba` on `dev/units-module`. | Closes a Phase-C RheoTool benchmark (KRK-SHIP-001). User chose "both cases" (cavity rigorous + RB qualitative). ~20 Boss-direct turns lost in the OF Docker weeds (coded-FO non-root recipe, SIMPLE convergence) = Boss-in-debugger anti-pattern — should have re-dispatched ONE extraction Dept and accepted its verdict. de Vahl Davis (published) is the authoritative anchor; OF is corroboration. |
| 2026-05-30 | **M-GEO-6 DONE — generic 3D STL-obstacle FLOW runner; drag deferred to M-GEO-7.** The Plan agent found the 3D LI-BB flow stack ALREADY exists (`run_sphere_libb_3d`) → mission was ~282 LOC REUSE-wiring, not a 200-400 LOC build. New `src/drivers/obstacle_3d.jl` + STL q_wall wrapper + structural `_run_d3q19` dispatch (+5/−1 runner). Validated: 3D STL flow units twin L∞=0.0 EXACT + non-trivial + stable. `6c4d2774e`, FF to v0.3; full suite 34594 pass / 7 pre-existing / 0 errored. | Canary-only (user-chosen): quantitative confined-sphere drag → M-GEO-7 (GPU/Aqua + reference selection). DURABLE finding: the 3D pull brick `fused_trt_libb_v2_step_3d!` SEGFAULTS on CPU with plain arrays (`@inbounds` OOB edge reads, lbm_builder.jl:75) — driver wraps CPU f in a zero-padded halo, GPU keeps plain alloc. The sphere CPU test only passes at 16×8×8 by luck. Future fix: edge-guard the kernel so CPU needs no halo. |
| 2026-05-31 | **M8 (Viscoelastic Oldroyd-B cylinder RheoTool benchmark) DONE — GREEN; committed `669bb4320` on `dev-viscoelastic` (local, 22 files).** Cd matches rheoFoam **<1%** at Wi≤1, R=50 (−0.40/−0.86/−0.96% for Wi=0.1/0.5/1.0); diffusive scaling τ=0.95, halfwayBB wall. β-sweep cross-validated (≤1% at β=0.59 & 0.30; β≤0.1 wall SHARED — both codes diverge: Kraken NaN / rheoTool PETSc FPE). Wi_max stability ≥Wi=10 (≥ LBM-VE SOTA). `.krk` repro via new `:viscoelastic` module dispatch. N1: adversarial Claude+Codex CONCORDANT = real wake-stress difference (sign-varying ratio 0.70–1.44), NOT a unit artifact — documented, not a ≤5% pass. LI-BB (`bouzidi_fl`) wall REJECTED (NaN at R≥50 Wi≥0.5, F64). | Closes Phase-C VE leg of KRK-SHIP-001. Adversarial+fine-mesh-F64 decisive (cheap discriminators mis-pointed). units gap found (no stress/N1 output conversion) → `.orchestrator/units-improvements.md` + §8. NOT pushed. |
| 2026-05-31 | **M9–M14 (KRK-SHIP-001 Phase D docs + Phase E integration) DONE — `dev/v0.2-multiphysics` RC branch cut on `dev/v0.3-campaign`.** Tri-track docs landed on `dev/v0.3-campaign` (the integration base: units+geometry+KRK-GEO already there): M9 implication-map spec+linter `e35288cff`; M10 build pipeline+CI+Track-C lint gate `88ac52b07`; M6/M7 benchmark pages ported `9ba59741d`; M11 Track-A krk-reference + 4 tutorials `a2025b980`; M12 Documenter docstrings **627/629** exports + 9 API pages `435a64e99` (diff proven docstring-only: 2916 add / 0 removed); M13 **8 implication-maps all lint-pass** `3d49db834`. **Gates GREEN:** `docs/make.jl` exit 0 (9-map lint gate + tri-track build); `runtests.jl` (direct, main env) **34598 pass / 7 pre-existing fails / 0 new / 0 errored**. RC branch `dev/v0.2-multiphysics` @ `3d49db834` (NOT pushed). | Closes KRK-SHIP-001 docs+integration. **§6-locked "RC from main" CORRECTED** by the merge-plan refresh: `main` is NOT an ancestor of v0.3 (would discard all v0.3 src), so RC = FF-copy of `dev/v0.3-campaign`. **VE log-FV solver NOT on v0.3** (M14-PROV-VE Codex: ~9000 LOC absent, no B1 dep for 2D) → assembled + **runtime-load-validated (precompiles clean)** on ISOLATED `feat/ve-logfv-on-v03` `f290ca17e`; **fold into RC is USER-REVIEW-GATED** (9000 LOC, numerically delicate). Test exit-1 was the 7 known fails only; the earlier `Pkg.test()` KA-not-found was a sandbox-target artifact (run `runtests.jl` directly). Open: thermal `natural_convection_2d` preset + `examples/natural_convection.krk` absent on v0.3 (M7 page snippet unrunnable, M11#1); 2 minor Documenter `@ref` warnings (krk-reference `[Presets]`); 2/629 undocstringed (`apply_density_correction_2d!`, `apply_zou_he_pressure_inlet_west_2d!`). |
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
| LU / real-units conversion | `src/units/` | `src/units/` (15 files, `dev/units-module`) — Newt+VE+Thermal-Boussinesq concrete; GNF/MP/MHD raising stubs | **IMPLEMENTED** Phase 1 (M4) + Thermal Phase 2 (M5), 2026-05-29; 277 tests green. Zero-edit contract §7 PROVEN. Driver-integration Cd/Nu repro deferred to post-merge | dimensional analysis closure tests (277 pass: parity 1e-12, round-trip, audit, krk, thermal α=ν/Pr + Ra recon 1e-10, zero-edit proof) | n/a (utility module — verified via downstream module benchmarks) |
| Geometry primitives (analytical, SDF, STL) | `src/geometry/` | **`src/geometry/`** (7 files: stl_reader, voxelizer, stl_cut_fraction, mask_apply, libb_precompute, descriptor + Geometry.jl) | **FACTORED** (M-GEO-3c 2026-05-30, `313611ca4`) — STL IO + voxelize + cut-fraction + mask realization + LI-BB precompute relocated here; `GeometryDescriptor` public type added. DSL parsing stays in parser; LI-BB BC spec stays in runner. `drivers/step_geometry_2d.jl` may remain (verify). | SDF distance closure, normals, intersection cases | n/a (geometry layer) — verified via BC benchmarks |
| Mesh (uniform / multi-block / AMR) | `src/mesh/` | `curvilinear/mesh.jl` + `curvilinear/generators.jl`, `refinement/*` | **MIXED** — multi-block mesh in curvilinear/, AMR mesh in refinement/; no `src/mesh/` dir yet (M1 2026-05-29) | grid count, connectivity invariants | n/a (verified via flow benchmarks) |
| BC — wall / periodic | `src/bc/{wall,periodic}` | `kernels/boundary_*`, `kernels/li_bb_*`, `fvfd/*boundary*` | **MIXED** — BC fused into collide kernels; no `src/bc/` dir yet (M1 2026-05-29) | Poiseuille profile (analytical) | `rheoFoam/Channel/Oldroyd-BLog` with `Wi=0, β=1` (Newtonian sanity) |
| BC — inflow / outflow | `src/bc/{inflow,outflow}` | inline in `simulation_runner.jl` (BoundaryHandler) | **MIXED** — inflow/outflow logic embedded in the 1843-LOC runner; no `src/bc/` dir yet (M1 2026-05-29) | flow rate conservation | `rheoFoam/Contraction41/Oldroyd-BLog` (inflow → outflow + recirculation) |
| Constitutive — Newtonian | `src/physics/newtonian` | functional (trunk) | **2D VALIDATED <1% (M6, 2026-05-29)** — cavity Re=100/400/1000 vs Ghia 1982 + icoFoam: rel-L2(u) **0.47/0.41/1.05%** (the earlier 2-3% was a half-cell coordinate bug, fixed via `axis_node_coords` `78ed276c4`→main `9b9a97aa8`; Couette control = machine-zero ⇒ solver exact, TRT irrelevant). 3D top Zou-He BC fixed (`a5ce7c6f0`). Page `docs/src/users/benchmarks/cartesian-cavity.md` | Poiseuille, Couette, Taylor-Green; **Ghia 1982 cavity (done, <1%)** | `icoFoam/cavity` Re=100/400/1000 (done — OF Newtonian equivalent); `rheoHeatFoam/buoyantCavity` (thermal → M7) |
| Constitutive — Oldroyd-B | `src/physics/oldroyd` | partial (dev-viscoelastic) | in flight | analytical Oldroyd-B startup curves | `rheoFoam/Cylinder/Oldroyd-BLog` (Cd=117.357 ref) AND `rheoFoam/Cavity/Oldroyd-BLog` |
| Constitutive — FENE-P / FENE-CR | `src/physics/{fenep,fenecr}` | _not started_ | _planned ship-2_ | analytical FENE shear closure | `rheoTestFoam/FENE-CR` (extensional) + `rheoFoam/OtherTests/Channel2D_VE` |
| Constitutive — Giesekus / PTT | `src/physics/{giesekus,ptt}` | _not started_ | _planned ship-2+_ | N1 ratio, shear-thinning slope | `rheoFoam/CrossSlot/PTTLog`, `rheoFoam/OtherTests/Channel2D_VE` |
| Constitutive — Thermal (Boussinesq) | `src/physics/thermal` | functional (trunk + `dev/units-module`) | **2D VALIDATED <1% (M7, 2026-05-30)** — de Vahl Davis 1983 cavity: Nu err **+0.79/+0.93/+0.79%** at Ra=1e3/1e4/1e5; OF `buoyantBoussinesqSimpleFoam` corroborates Ra=1e3/1e5 (<0.6%), Ra=1e4 OF under-converged (documented). `.krk`=`natural_convection_2d`. Page `docs/src/users/benchmarks/thermal-natural-convection.md`. Commits `123130590`+`783f2ebba` on `dev/units-module` | de Vahl Davis 1983 cavity Nu(Ra) (done, <1%); RB qualitative | `buoyantBoussinesqSimpleFoam` cavity (done — corroborates Ra=1e3/1e5) |
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

- [ ] **`src/units/` improvements backlog** (discovered in M8, 2026-05-31): the module does the INPUT direction (nondim→LU) well but has NO output/results→physical conversion (no stress factor → blocked the M8 N1 comparison), is ABSENT on `dev-viscoelastic` (so the VE driver can't consume it), and its `ViscoelasticSpec` path is never exercised by a real driver. Full list + fixes in [`.orchestrator/units-improvements.md`](units-improvements.md). Candidate units-Phase-3 mission post-KRK-SHIP-001.
- [ ] **Kraken-AMR.jl relation**: is `Kraken-AMR.jl` a separate fork that will eventually merge into Kraken.jl, an upstream import target, or a permanently separate project? Decide before any AMR port work proceeds.
- [x] ~~**`lbm` branch (2026-04-16)**: role? retire?~~ — RETIRED 2026-05-29 (local + `origin/lbm`)
- [ ] **`dev/v0.2-architecture`**: retirement REVERSED 2026-05-29 — carries unique `src/runtime_specs.jl` (M3/M4 `src/units/` ancestor). Decide final disposition after M3 design lands (port the skeleton into `src/units/` then retire, or keep as reference).
- [ ] **PRUNABLE worktrees**: `Kraken.jl-amrd-golden`, `Kraken.jl-kraken-e-blocks`, `Kraken.jl-refinement-perf` flagged by git. Confirm each: keep + unflag, or remove?
- [ ] **`.krk` DSL spec**: is there a versioned spec document? If not, write one and place at `docs/spec/krk-v1.md`.
- [ ] **LLM-doc format**: define the canonical format for the per-module implication map. JSON-Schema-described markdown? YAML frontmatter?
- [ ] **Backend coverage**: which backends are tested in CI today? Document, add missing.

## 9. Mission graph (active + planned)

Add missions here as they are picked up. Use orchestrator's mission template format.

### M-GEO-1 — STL obstacle end-to-end proof (Cartesian immersed-boundary, `.krk` → runner)

- **Status**: **GREEN (Boss-verified 2026-05-29, shim-free)** on branch `feat/geometry-stl` (off `dev/v0.3-campaign`, worktree `Kraken.jl-geometry-stl`). After M-GEO-2 fixed the mesh skew: canary `GATE-A 0/1600 (0.0%)` + `GATE-B 0.0%` + `PASS`, and `run_simulation("…stl_flow.krk")` works directly. **COMMITTED** as `434aee036` feat(geometry) (M-GEO-1) + `fc9a4d7eb` fix(io) (M-GEO-2) on `feat/geometry-stl`. Canary promoted to `test/test_geometry_stl_krk.jl` (wired into runtests) — 7/7 pass. Superseded slbm-paper worktree torn down. Open: cherry-pick `fc9a4d7eb` to the 3 other lineage branches; merge `feat/geometry-stl` → `dev/v0.3-campaign`; M-GEO-3 (LI-BB q_wall for STL / Mesh-parser for GMSH-via-.krk / `src/geometry` consolidation).
- **Track**: KRK-GEO (new, ADR §4 2026-05-29). Separate from KRK-SHIP-001.
- **Goal**: A `.krk` file declaring `Obstacle <name> { stl_file = "...", scale=, translate= }` runs via `run_simulation(filename)` on a coarse grid and produces a correct VTK; the resulting flow MATCHES the analytical `Obstacle <name> { (x-cx)^2+(y-cy)^2 <= R^2 }` case (same domain/Re/grid) within tolerance. Proves the STL→.krk seam works end-to-end before any `src/geometry/` consolidation.
- **Allowed edit zones**: new STL asset + generator under `examples/geometry_stl/` (or `benchmarks/krk/geometry_stl/`), new `.krk` there, new canary test `test/scratch/` (or `test/analytical/geometry_stl_cylinder.jl`), and SMALL fixes to `src/simulation_runner.jl::_apply_geometry!`/`_voxelize_stl_region` + `src/io/voxelizer.jl` ONLY if the canary exposes a coordinate-convention/z_slice mismatch.
- **Forbidden**: no commits; no multi-block/curvilinear/GMSH changes; no GPU kernel edits; no relocation into `src/geometry/` yet (that's the §6 module mission, deferred); no LI-BB q_wall-for-STL (follow-up). File-size budget respected.
- **Exit criterion**: canary script run via project Julia, CPU Float64, coarse grid — asserts (a) STL is_solid mask ≡ analytical mask on ≥99% of cells, (b) steady drag Cd (or centerline velocity profile) STL-vs-analytical within ≤2%. Plus `git diff --check`.
- **Notes**: STL→mask path already wired (gate-1 cleared). Likely deliverable is mostly assets + canary, with at most small coord-convention fixes. Validation-per-module gate §3 (analytical + RheoTool + .krk) applies to the FULL geometry module (later), NOT to this proof — proof's analytical ref IS the analytical-cylinder cross-check.

### M-GEO-2 — Complete commit `682e3f3c0`: add `mesh` field to `SimulationSetup` (un-break `run_simulation` on slbm-paper)

- **Status**: **DONE — GREEN (Boss-verified 2026-05-29)** on `feat/geometry-stl` (off `dev/v0.3-campaign`). Minimal fix: added `mesh::Any` field to `SimulationSetup` + a backward-compatible 15-arg outer constructor defaulting `mesh=nothing` (kraken_parser.jl:166). All 15-arg call sites unchanged; `_override_max_steps` uses the 16-arg form. `run_simulation` un-broken; canary shim-free GREEN. Did NOT build the `Mesh`-directive parser (no `MeshSetup` exists on any branch — that is the real GMSH-via-`.krk` wiring, deferred to M-GEO-3+).
- **Track**: KRK-GEO.
- **Goal**: Add `mesh` (default `nothing`) to `SimulationSetup` so the runner's pre-existing `setup.mesh` references (commit `682e3f3c0`) stop throwing `FieldError`. Result: `run_simulation("any.krk")` works again; M-GEO-1 canary loses its `getproperty` shim and turns true GREEN.
- **Allowed edit zones**: `src/io/kraken_parser.jl` (struct + constructor + Mesh-directive wiring ONLY — surgical, no SPLIT of the 2010-LOC file), `test/scratch/geo_stl_canary.jl` (remove the shim).
- **Exit criterion**: `run_simulation("examples/geometry_stl/cylinder_stl_flow.krk")` runs WITHOUT the shim; the canary passes shim-free; a plain analytical `.krk` also runs (regression check).
- **Notes**: This is a regression-fix that is ALSO step 1 of GMSH-via-`.krk`. Pre-existing: introduced by `682e3f3c0` (body-fitted runner) on slbm-paper only; main/release unaffected (and lack the geometry stack entirely). After GREEN: M-GEO-3 = wire LI-BB q_wall for STL (accuracy) OR `src/geometry/`+`src/mesh/` §6 consolidation.

### M-GEO-3a — LI-BB (Bouzidi) wall for STL obstacles via `.krk`

- **Status**: **DONE — GREEN (Boss-verified 2026-05-30)**, committed `2a5252953` on `feat/geometry-stl`. Canary A=2.35% (vs validated `run_cylinder_libb_2d`) + B=11.3% (vs halfway → LI-BB active); tests 7/7 + 5/5 + driver-comparison 2/2 promoted into `test_geometry_stl_krk.jl`; blast-radius 45/45 (default-loop reorganization safe). Codex (kraken-branch-governor). Syntax shipped: `Obstacle … stl(…, wall = libb)`. **Merged FF into `dev/v0.3-campaign` (HEAD `2a5252953`)** — v0.3 now carries all 3 KRK-GEO commits (mesh-fix + STL + LI-BB). NB: +180 net LOC to `simulation_runner.jl` (already HARD-oversized) → §6 SPLIT debt ↑; LI-BB helpers (`_precompute_stl_libb_q_wall_2d`, `_build_libb_bc_rebuild_spec_2d`, …) are relocation candidates for `src/geometry/`+`src/bc/`.
- **Track**: KRK-GEO.
- **Goal**: STL obstacle opts into LI-BB sub-cell bounce-back (vs default halfway-BB) via a `.krk` selector (`wall=libb`, reusing the dead `GeometryRegion.bc_type`). Integrate `precompute_q_wall_from_stl_2d` + `fused_trt_libb_v2_step!` into the generic `run_simulation` 2D loop. Default path UNCHANGED.
- **Seam (Explore-mapped)**: LI-BB only reachable via specialized drivers today; generic runner uses `collide_2d!` (halfway-BB). When STL+`:libb`: compute `q_wall[Nx,Ny,9]`, zero `uw`, replace stream+collide with the FUSED LI-BB step (mirror `run_cylinder_libb_2d` loop L233-245). Risk: fused-step vs separate-stream/collide loop mismatch.
- **Allowed edit zones**: `src/simulation_runner.jl` (conditional, additive), `src/io/kraken_parser.jl` (read `bc_type`), `examples/geometry_stl/` (new libb `.krk`), `test/test_geometry_stl_krk.jl` (+ LI-BB testset), `test/scratch/`.
- **Exit criterion**: canary — STL-cyl `wall=libb` matches `run_cylinder_libb_2d` within ≤3% (Cd or field L2); differs measurably from halfway-BB; existing M-GEO-1 test still passes.
- **Notes**: opt-in chosen over automatic (don't silently change physics / break M-GEO-1). q_wall already validated (test_stl_libb 9146). 2D only.

### M-GEO-3c — Consolidate geometry into `src/geometry/` (§6 modular target)

- **Status**: **DONE — GREEN (Boss-verified 2026-05-30)**, committed `313611ca4`, **FF-merged into `dev/v0.3-campaign`** (HEAD now `313611ca4`). Behavior-preservation PROVEN: full suite **34283 pass / 7 pre-existing fails (identical file:line) / 0 errored** — zero new failures. io trio relocated as git renames (R100 = byte-identical). Runner 2907→2711 LOC. `GeometryDescriptor` defined + tested. v0.3 now carries all 4 KRK-GEO commits (fc9a4d7eb + 434aee036 + 2a5252953 + 313611ca4).
- **Track**: KRK-GEO. Behavior-preserving PURE relocation (flat includes into `Kraken`, no submodule) → tests stay valid via the `Kraken.` namespace.
- **Goal**: New `src/geometry/` dir (6 files: stl_reader, voxelizer, stl_cut_fraction, mask_apply, libb_precompute, descriptor + Geometry.jl aggregator). MOVE: io/stl_* + io/voxelizer + `_apply_geometry!`/3d/patch + libb precompute OUT of the oversized runner (~−225 LOC). STAY: LI-BB BC spec (runner), parser DSL (GeometryRegion/STLSource). NEW `GeometryDescriptor` type (additive; units wiring is cross-branch, deferred). `src/mesh/` = separate later mission.
- **Riskiest step**: mask_apply include order — `_apply_patch_geometry!` needs `RefinementPatch` → geometry includes AFTER parser+refinement, BEFORE runner.
- **Exit criterion**: Codex validates geometry+core test subset green; **Boss runs FULL `Pkg.test` (baseline 34265 pass / 7 pre-existing fails unchanged) as the behavior-preservation gate** before commit.
- **Notes**: pays §6 geometry row (MIGRATING→FACTORED) + reduces simulation_runner.jl oversized debt. Source of truth: the architect plan (this session).

### M-GEO-UNITS-0 — DONE: graft `src/units/` onto `dev/v0.3-campaign`

- **Status**: **DONE — GREEN (Boss-verified 2026-05-30)**, committed `22672591d`, FF-merged to `dev/v0.3-campaign`. units module grafted (NOT branch-merged — naive merge was a trap, see boss.md). Full suite 34560 pass / 7 pre-existing fails / 0 errored. v0.3 now carries `src/geometry/` AND `src/units/`. This is the PREREQUISITE for the wiring (M-GEO-4).

### M-GEO-4 — Wire geometry ↔ units (STL physical-units ↔ LU)

- **Status**: **DONE — GREEN (Boss-verified 2026-05-30)**, committed `ac4785c9c`, FF-merged to `dev/v0.3-campaign` (HEAD `ac4785c9c`). Plan by architect (parse-time lowering, runner UNTOUCHED). `.krk` `Units { length, L_ref, R_LU, Re, scaling }` block → parser computes `dx_real = L_ref/R_LU`, calls `Units.compile`, writes back STL `scale`/`nu_LU`/`u_LU` → run_simulation sees raw-LU. Validated 10/10: physical-mm STL cylinder ≡ hand-scaled raw-LU twin (mask + flow identical). Full suite **34570 pass / 7 pre-existing / 0 errored**. Convention: `L_ref` & `R_LU` both = the characteristic radius (no factor-2). `simulation_runner.jl` = 0 edits (the architectural win). KRK-GEO chain COMPLETE on v0.3.
- **Goal**: (a) **descriptor pass-through** — `src/geometry/descriptor.jl::GeometryDescriptor` → units' `compile(geometry=…)` duck-typed contract (align/add `kappa_max`/`L_up`/`L_down`). (b) **STL physical→LU** — a `.krk` way to declare an STL + domain in physical units + a reference length + resolution `R_LU` → units computes `dx_real = L_phys/R_LU` → geometry's `_voxelize_stl_region` applies `scale = 1/dx_real`. Closes the user's "STL dim vs adim" need. The runner calls units `compile` → feeds geometry the scale.
- **Notes**: units is already STL-aware (`audit_stl`, own `GeometryDescriptor`). The core conversion is `nondim_to_lu(spec, kw, geom)` anchored on `R_LU` + `dx_real` (lattice_units.jl). Branch: continue on the v0.3 lineage. Validate: an STL given in physical mm reproduces the same LU sim as the equivalent hand-scaled `.krk`.

### M-GEO-5 — 3D physical-units STL (mask-level) + populate units GeometryDescriptor fields

- **Status**: **DONE — GREEN (Boss-verified 2026-05-30, full suite re-run by user)**, committed `9acf4eb4d`, **FF-merged into `dev/v0.3-campaign`** (HEAD `9acf4eb4d`). Scope = **Option A** (user-chosen): 3D units→LU bridge at the **MASK level** + descriptor fields. The 3D STL **flow runner is OUT OF SCOPE → new mission M-GEO-6** (see ADR 2026-05-30: the 3D flow path does not exist; `_run_d3q19` only dispatches `cavity_3d`, `fused_trt_libb_v2_step_3d!` is never wired into the runner). Full suite **34585 pass (= 34570 + 15 new) / 7 pre-existing fails (identical: conservative_tree_streaming:619-622 ×4 + multiblock_exchange:161/162/186 ×3) / 0 errored / 4 pre-existing broken**. **0 edits to `simulation_runner.jl`** (M-GEO-4 architectural win preserved).
- **Track**: KRK-GEO. Two coupled goals delivered:
  1. **3D mask bridge**: `_apply_units_bridge!`/`_units_geometry_namedtuple`/`_validate_units_domain!` are now `lattice`-aware; D3Q19 builds a `falses(Nx,Ny,Nz)` mask via `_apply_geometry_3d!` (which now voxelizes STL via new `_voxelize_stl_region_3d`, was a dead `continue`). Scale path (`dx_real`/`_units_scaled_stl_regions`) confirmed genuinely dim-agnostic — 0 change. Validated: physical-mm 3D sphere STL ≡ hand-scaled raw-LU twin at the MASK level (280 solid voxels identical). Validation is mask-only (NO `run_simulation` on 3D STL flow — it throws).
  2. **Descriptor fields** (geometry-side helpers in `src/geometry/descriptor.jl`, consumed by units `compile`): `stl_kappa_max` (bbox proxy, **inverse-LU** `1/R_LU`=0.25 for the sphere — convention forced by audit threshold `R_LU·kappa_max>0.5`), `obstacle_extents_in_R` (L_up=5/L_down=15 from obstacle centroid, in units of R; NaN-sentinel = derive-if-user-omits, else respect knob), `halfway_wall_distances` (0.5-per-link q_wall_dist from mask — REQUIRED to un-gate `audit_stl` which early-returns on `q_wall_dist===nothing`; all-0.5 fires only `:curvature_underresolved`, no spurious cliff/skewness). Audit now fires `:curvature_underresolved` + `max_steps` reflects derived L_up+L_down (non-trivial).
- **Asset**: new `examples/geometry_stl/{make_sphere_stl.jl, sphere.stl}` (physical radius 0.2 mm, sphere chosen over cylinder-prism for genuine 3-axis 3D voxelization) + `sphere_stl_3d_{mm,lu}.krk` twin pair + `test/test_geometry_units_3d_krk.jl` (mask-equality + descriptor-non-triviality, 15 tests).
- **Method**: Plan-then-Implement (Plan agent → Boss design + 2 user scope decisions → Codex via run-engineer.sh on `feat/units-on-v03`). Parse-time wiring, 0 runner edits.

### M-GEO-6 — 3D STL-obstacle FLOW runner (the deferred half of M-GEO-5 goal-1)

- **Status**: **DONE — GREEN (Boss-verified 2026-05-30, full suite autonomous gate)**, committed `6c4d2774e` on `feat/units-on-v03`, **FF-merged into `dev/v0.3-campaign`** (HEAD `6c4d2774e`). Full suite **34594 pass (= 34585 + 9 new) / 7 pre-existing fails (identical) / 0 errored / 4 broken**. **Validation = CANARY-ONLY** (user-chosen): 3D STL FLOW units twin + finite/stable smoke. **Quantitative sphere drag deferred → M-GEO-7** (needs GPU/Aqua + a confined-sphere reference).
- **Track**: KRK-GEO. Turned out **REUSE-heavy wiring (~282 LOC), NOT a 200-400 LOC build** — the Plan agent found the full 3D LI-BB flow stack (step kernel + 6-face `BCSpec3D`/`apply_bc_rebuild_3d!` + cut-link drag + equilibrium init) ALREADY exists and runs end-to-end in `run_sphere_libb_3d` (kernels/li_bb_3d_v2.jl:201); it only built `q_wall` from analytic sphere geometry + hard-coded west/east BC.
- **Delivered**: new driver `src/drivers/obstacle_3d.jl` (206 LOC) = `run_obstacle_libb_3d` (lifts the sphere-driver loop, feeds it STL q_wall + `.krk`-derived BC) + `_build_libb_bc_rebuild_spec_3d` (`.krk` faces → `BCSpec3D`, velocity-on-west/pressure-on-east only, transverse walls via the kernel's internal halfway-BB `apply_transverse=false`). `_precompute_stl_libb_q_wall_3d` (geometry/libb_precompute.jl, wraps the existing `precompute_q_wall_from_stl_3d`). Dispatch: structural `_has_stl_libb_obstacle` branch in `_run_d3q19` (+5/−1 — the ONLY runner edit). Test: `test_geometry_stl_flow_3d_krk.jl` (units flow twin + smoke).
- **Validated**: sphere `_3d_mm.krk` flow ≡ `_3d_lu.krk` flow, **L∞=0.0 EXACT** (units→LU correct at the FLOW level in 3D, extends M-GEO-5's mask twin); non-trivial flow (max|ux|=0.0062 > inlet 0.005); stable (ρ∈[0.995,1.006], no NaN).
- **Key finding (durable)**: the 3D pull brick `fused_trt_libb_v2_step_3d!` **SEGFAULTS on CPU with plain arrays** at non-tiny grids (`@inbounds` OOB edge reads; `lbm_builder.jl:75`) — `run_sphere_libb_3d`'s CPU test only "passes" at 16×8×8 by luck (OOB lands in valid heap). **FIXED 2026-05-30 (`770375a2f`)**: root cause was the `PullHalfwayBB_3D` brick's `ifelse(cond, f_in[i-1,...], fb)` — `ifelse` evaluates BOTH arms, and the neighbour index was UNCLAMPED, so the discarded arm read `f_in[0,...]` OOB. Fix = clamp the 18 neighbour indices with `min`/`max` (mirror the 2D brick); bit-identical (clamp only changes the discarded arm's address). The `_CPUHaloArray4` workaround was removed (driver back to plain arrays). The 3D LI-BB path now runs on CPU without a halo.

### M-GEO-7 — 3D sphere quantitative drag — DONE (both parts)

- **Status**: **DONE — GREEN (2026-05-31)**. Part (a) local twin `e06715c78` + part (b) Aqua convergence `443e11773`, **FF-merged to `dev/v0.3-campaign`** (HEAD `443e11773`). Closes the §3 physics-reference leg of the 3D STL obstacle module.
- **Part (a) — local self-consistency twin**: STL `.krk` sphere drag reproduces the validated analytic scaffold (`run_sphere_libb_3d`, vs Clift 2.6) to **0.39%** at matched lattice registration. `run_obstacle_libb_3d` returns `Cd` (frontal-area silhouette). `test_sphere_stl_drag_krk.jl` GPU-gated. Full suite 34598 pass / 7 pre-existing / 0 errored.
- **Part (b) — Aqua CUDA F64 free-stream convergence**: blockage sweep at R=16 (D=32), Re=20, 20%→6% blockage; `C_d` monotone 5.44→3.38; quadratic LSQ extrapolation `D/W→0` = **2.84 (R²=0.9998), +8.9% vs Clift 2.61**. Residual = finite resolution (R8→R16 moved Cd −2.5% at fixed blockage). Benchmark page `docs/src/users/benchmarks/sphere-drag-3d.{md,png}` + runnable bench scripts + CSVs. **Future work (optional M-GEO-8): a resolution extrapolation (R=16,32) to close the ~9% to a few %.**
- **Aqua gotchas (boss.md):** `gpu_id=A100` lands on the cluster's MIXED 40/80GB A100 nodes → the 61M-cell F64 case OOM'd at 40GB; use `gpu_mem=60gb` in the select to target any 80GB GPU (H100 or A100-80GB). PBS `tee` masks the Julia exit code → add `set -o pipefail` + `exit ${PIPESTATUS[0]}`.
- **Track**: KRK-GEO. The mandate §3 physics-reference leg for the 3D STL obstacle module.
- **CRITICAL finding (2026-05-31, see boss.md):** a first twin showed Cd_krk +78% vs scaffold — I mis-diagnosed it as a `precompute_q_wall_from_stl_*` bug. **It was NOT a bug.** Root cause: STL voxelizer is **cell-centred** `(i-0.5)dx` vs analytic **node-centred** `(i-1)` → a half-cell registration offset. An offset sweep proved `corr(q_stl,q_analytic)=1.0` at δ=+0.5; the STL q_wall is EXACT once registered. Two durable lessons (boss.md): (1) LI-BB drag at coarse R is strongly registration-sensitive (same sphere: Cd 3.5 on-node vs 6.1 half-off-node); (2) prove the mechanism before declaring a bug — near-zero per-link q correlation + half the cut-links in different slots is the FINGERPRINT of a registration offset, not a code bug.

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
