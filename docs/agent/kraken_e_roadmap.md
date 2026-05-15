# Kraken-E Implementation Roadmap

Date created: 2026-05-15

Owner branches:

- `dev/fvfd-core` (FVFD operator library, neutral owner)
- `dev/kraken-e-fvfd-blocks` (LBM bulk + block runtime + FVFD interface)

Source plan: [kraken_e_fvfd_interface_plan_2026-05-15.md](kraken_e_fvfd_interface_plan_2026-05-15.md)

Status: orchestration document. Each session below has one objective, one
derivation doc, one atomic commit, and explicit exit criteria. No session
ends without its exit criterion green.

## Branch topology

```text
main
 └── dev/fvfd-core              ← FVFD-only, neutral. Created S0. Frozen API
      │                          consumers depend on after S1 is green.
      │
      ├── dev/kraken-e-fvfd-blocks   ← Branches from fvfd-core at start of S2.
      │                                Adds block runtime, LBM bulk, FVFD/LBM
      │                                interface, derivations D1..D10.
      │
      └── (dev-viscoelastic rebases on fvfd-core post-S1 to consume the
           neutral operator core instead of carrying its own untracked copy.)
```

## Session ladder

Sessions are atomic. A session is either fully green (commit + roadmap update)
or aborted with a `blocked` note + a smaller-scope follow-up session inserted
before continuing.

| ID | Topic | Branch | Exit criterion | Status |
|----|-------|--------|----------------|--------|
| S0 | Setup: roadmap + FVFD resource bilan + branches | dev/fvfd-core created from main | Both docs committed at `31e700df`; worktree `Kraken.jl-fvfd-core` live | done |
| S1 | Extract FVFD core (src/fvfd, test, design doc) onto dev/fvfd-core | dev/fvfd-core | 923/923 green on CPU (15.9s); commit `dba83bef`; 8 isdefined gates skip viscoelastic-only fixtures | done |
| S2 | D1 ownership + D2 same-level LBM block update | dev/kraken-e-fvfd-blocks (branched from fvfd-core@dbfa09c5) | Derivation doc D1+D2 committed; uniform-block solver passes Poiseuille (L2=0.05%), Couette (L2=4e-6), Taylor-Green (slope err=0.08%, mass drift 1e-14) canaries; 396/396 green | done |
| S3 | D3 FVFD operators on block + D4 coarse/fine Cartesian face geometry | dev/kraken-e-fvfd-blocks | Derivation doc D3+D4 committed; view-based FVFD-on-LeafBlock2D adapter; CFFaceRecord2D geometry; constants & affine exact at ≤1.5e-15 (D3 grad, D3 div, D4 quadrature, D4 prolongation); corner ownership unique; 421/421 green | done |
| S4 | D5 conservative interface fluxes on 2-block patch | dev/kraken-e-fvfd-blocks | Derivation doc D5 committed; ScalarFluxField2D + TwoBlockPatch2D + explicit Euler step; telescoping err = 0.0 exactly (≪ 1e-14); 100-step mass drift = 1.4e-15 relative; 424/424 green | done |
| S5 | D7 LBM-to-moment extraction + D8 FVFD-to-LBM reconstruction | dev/kraken-e-fvfd-blocks | Inverse on chosen subspace; identity on uniform equilibrium; rho/j preserved exactly | pending |
| S6 | D9 stress consistency + Poiseuille canary with c/f interface | dev/kraken-e-fvfd-blocks | Continuous stress across c/f interface; no density/velocity jump beyond discretization order | pending |
| S7 | D10 wall + interface + Guo force (AMR-D marche 5 redo) | dev/kraken-e-fvfd-blocks | Forced Poiseuille with c/f near wall: roundoff mass conservation, no unexplained wall/interface density jump | pending |

Out of scope for sessions S0..S7:

- subcycling (D6) — deferred to S8 only after S7 green;
- 3D port (validation ladder 10..11) — deferred to v0.3 phase;
- viscoelastic coupling (D11) — deferred to post-Kraken-E phase 1;
- epoch adaptation / dynamic AMR (D12) — deferred to phase 2;
- cut cells / embedded geometry on c/f interface — deferred to phase 2;
- macro-flow benchmark tuning — forbidden until S7 green.

## Per-session contract

Every session must produce:

1. One derivation document under `docs/agent/kraken_e_S<n>_<topic>.md` with:

   - explicit invariants (rho/j conservation, SPD, telescoping, exactness);
   - the proof sketch or algebraic identity;
   - the failure modes the test must catch;
   - a one-line exit criterion the next session can verify.

2. One test under `test/kraken_e/test_S<n>_<topic>.jl` (or extension of
   existing FVFD tests) that exercises the invariants. The test must fail
   if the invariant is broken, not merely warn.

3. One atomic git commit:

   ```text
   <type>(kraken-e): <subject>

   Closes: S<n>
   Exit: <one line>
   ```

4. One roadmap line update in this file: status → `done`, with a one-line
   note pointing at the derivation doc and the test path.

## Stop criteria (mirrors plan §14)

A session immediately stops and the architecture is re-evaluated if any of
the following occurs:

- D2Q9 moment reconstruction cannot preserve `rho,j` exactly while imposing
  the chosen interface stress (S5);
- coarse/fine FVFD fluxes do not telescope to roundoff on a two-level patch
  (S4);
- wall plus interface plus Guo force creates a persistent density jump not
  explained by the discretization order (S7);
- spectral interface analysis shows an unstable interface-localized mode
  (post-S5 spectral check);
- positivity requires opaque clipping for ordinary target Mach numbers (S5);
- FVFD extraction cannot be isolated from viscoelastic branch experiments
  (S1).

If a stop criterion fires, insert a smaller-scope session before continuing.
Do not tune around a stop criterion.

## Coordination with dev-viscoelastic

After S1 is green:

1. `dev-viscoelastic` must stop carrying untracked FVFD files under
   `src/fvfd/`. The branch consumes the FVFD core from `dev/fvfd-core`
   via rebase or merge.
2. Future FVFD operator changes happen on `dev/fvfd-core` (owner branch)
   and propagate to `dev-viscoelastic` and `dev/kraken-e-fvfd-blocks` via
   cherry-pick or rebase.
3. `dev-viscoelastic` retains its viscoelastic-specific consumer code
   (e.g. `src/kernels/logconformation_fv_2d.jl`), which depends on FVFD
   core but is not part of the core.

Recorded in `fvfd_resource_bilan_2026-05-15.md`.

## Cadence

- S0..S1: longer sessions (3-4h equivalent), setup + extraction is one-shot.
- S2..S7: shorter focused sessions (1-2h equivalent), one derivation each.

Each session resumes cleanly from this roadmap. The next active session is
always the first row with status `in-progress` or the first `pending` row
after `done`.

## Current state (live)

- 2026-05-15: S0 done at `31e700df` + `00c6706e` (bootstrap + roadmap update).
- 2026-05-15: S1 done at `dba83bef` on dev/fvfd-core.
  FVFD operator library extracted from dev-viscoelastic (10 files, +3222 lines).
  test/test_fvfd_operators_2d.jl: 923/923 green on CPU in 15.9s.
  8 isdefined gates skip viscoelastic-only fixtures (logfv_* helpers and
  precompute_q_wall_cylinder, all of which live on dev-viscoelastic only).
  Pattern used: 3-layer (orchestrator → subagent → codex via pilot.sh).
  Subagent identified the cylinder fixture gap, orchestrator fixed with a
  1-line isdefined gate, then committed atomically.
- Plan reference (source of truth on slbm-paper):
  `docs/agent/kraken_e_fvfd_interface_plan_2026-05-15.md` in the
  `/Users/guillaume/Documents/Recherche/Kraken.jl` worktree.
- 2026-05-15: S2 done on dev/kraken-e-fvfd-blocks (branched from
  dev/fvfd-core@dbfa09c5). Architectural decisions taken (no reuse of
  src/multiblock/ — SLBM gmsh substrate wrong for AMR-tree; ghost cells
  with halo-exchange phase, ng=1 D2Q9; D1 ownership taxonomy paper-only
  for c/f, reflux, epoch buffers; D2 pipeline = apply_bcs! → exchange_halo!
  → collide! → stream! with half-way BB walls + Guo force).
  Derivation doc: `docs/agent/kraken_e_S2_D1_D2_leaf_block_2026-05-15.md`.
  Branch contract: `docs/agent/branch_contract_fvfd_blocks.md`.
  Module: `src/kraken_e/` (5 files: KrakenE.jl entry, leaf_block.jl,
  pipeline.jl, bcs.jl, canaries.jl).
  Tests: `test/kraken_e/test_S2_uniform_block.jl`. 396/396 green.
  Canaries on 32×32 at τ=0.8: Poiseuille L2=0.05%, Couette L2=4e-6,
  TG decay-slope err=0.08%, mass drift 1.1e-14, equilibrium-fixed 1.1e-16
  (all ≪ 1% target).
- Next session: S3. D3 FVFD operators on block + D4 coarse/fine Cartesian
  face geometry. Stays on dev/kraken-e-fvfd-blocks. First c/f geometry
  work — no fluxes yet (those at S4).
- 2026-05-15: S3 done on dev/kraken-e-fvfd-blocks. Derivation doc:
  `docs/agent/kraken_e_S3_D3_D4_block_fvfd_and_cf_faces_2026-05-15.md`.
  D3: thin view-based adapter (`src/kraken_e/fvfd_block_adapters.jl`)
  drives the existing FVFD ops on `LeafBlock2D` interior arrays with no
  reimplementation, no allocation, isotropic dx=dy=block.dx. D4: typed
  `CFFaceRecord2D{T}` static-blittable record (`src/kraken_e/cf_face_2d.jl`)
  with 2 fine faces per coarse face, trapezoidal half-half quadrature,
  3-point central tangential prolongation `(3/4, 1/4)`, side-based
  corner ownership rule. `LeafBlock2D.cf_face_records` retyped from
  empty struct to `Vector{CFFaceRecord2D{T}}`; back-compat alias
  `CFFaceRecord = CFFaceRecord2D{Float64}` retained. Tests:
  `test/kraken_e/test_S3_cf_faces_2d.jl` (206 lines, 25 assertions).
  Canary metrics: D3 const grad = 0.0; D3 affine grad err ≤ 1.5e-15;
  D3 const tensor div = 0.0; D4 weights sum = 1 exactly; D4 affine
  quadrature err = 2.8e-17; D4 prolongation const/affine err = 0.0;
  corner ownership unique (boolean). 421/421 green in 3m54s. S2
  regression untouched (Poiseuille 0.05%, Couette 4e-6, TG 0.08%).
- Next session: S4. D5 conservative interface fluxes on a two-block
  patch. Stays on dev/kraken-e-fvfd-blocks. First time c/f fluxes are
  computed — must satisfy F_coarse = sum(F_fine_k) telescoping to
  roundoff.
- 2026-05-15: S4 done on dev/kraken-e-fvfd-blocks. Derivation doc:
  `docs/agent/kraken_e_S4_D5_interface_fluxes_2026-05-15.md`. D5: scalar
  FV flux field (`ScalarFluxField2D`) and two-block patch
  (`TwoBlockPatch2D`) drive a closed-x + periodic-y explicit-Euler step
  on a coarse (8×8, dx=1) + fine (16×16, dx=0.5) patch sharing one c/f
  interface (8 coarse faces). The coarse face flux is **reconstructed
  by quadrature** from the two fine subface fluxes via
  `record.fine_to_coarse_weights = (1/2, 1/2)` — never computed
  independently from coarse-side data. This single design choice makes
  the telescoping `F_coarse · A_coarse = Σ F_fine_k · A_fine_k`
  identically zero in floating-point. New files:
  `src/kraken_e/interface_flux_2d.jl` (86 lines),
  `src/kraken_e/two_block_patch_2d.jl` (121 lines),
  `test/kraken_e/test_S4_telescoping_2d.jl` (62 lines). Canary metrics
  on Gaussian IC over 100 explicit-Euler steps at CFL=0.5 with
  prescribed v=(0.1, 0): max telescoping err = 0.0 exactly (≪ 1e-14
  threshold); max conservation drift = 1.4e-15 relative (≪ 1e-12
  threshold); synthetic-face ulp identity = 0.0. 424/424 green in
  3m55s. S2 + S3 regression bit-identical. Plan §14 stop criterion
  on D5 telescoping passed.
- Next session: S5. D7 LBM-to-moment extraction + D8 FVFD-to-LBM
  reconstruction. First time LBM populations meet the FVFD interface.
  Must preserve `rho, j` exactly while imposing the chosen interface
  stress.
