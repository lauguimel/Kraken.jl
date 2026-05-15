# Branch Contract — dev/kraken-e-fvfd-blocks

Date: 2026-05-15
Branch: `dev/kraken-e-fvfd-blocks` (branched from `dev/fvfd-core` at `dbfa09c5`)
Worktree: `/Users/guillaume/Documents/Recherche/Kraken.jl-kraken-e-blocks`
Plan: [`kraken_e_fvfd_interface_plan_2026-05-15.md`](kraken_e_fvfd_interface_plan_2026-05-15.md) (source of truth lives on `slbm-paper`)
Roadmap: [`kraken_e_roadmap.md`](kraken_e_roadmap.md)
Sister contract: [`branch_contract.md`](branch_contract.md) (the `dev/fvfd-core` contract, modelled here)
Current derivation (active session): [`kraken_e_S2_D1_D2_leaf_block_2026-05-15.md`](kraken_e_S2_D1_D2_leaf_block_2026-05-15.md)

## Branch objective

Own the Kraken-E block runtime: AMR-tree leaf blocks, LBM bulk solvers
inside regular blocks, and (in later sessions) the FVFD/LBM interface
that couples coarse/fine neighbours via FVFD fluxes and moment
reconstruction. This branch consumes the neutral FVFD operator core
from `dev/fvfd-core` read-only; it does **not** modify FVFD operators.

Scope per session:

- **S2 (active)**: D1 ownership + D2 same-level uniform-block update.
- **S3..S4**: D3 FVFD operators on block, D4 c/f Cartesian face geometry,
  D5 conservative interface fluxes on a two-block patch.
- **S5..S6**: D7 moment extraction, D8 FVFD-to-LBM reconstruction, D9
  stress consistency, Poiseuille canary with c/f interface.
- **S7**: D10 wall + interface + Guo force (AMR-D marche 5 redo).

Out of scope on this branch entirely: viscoelastic coupling (lives on
`dev-viscoelastic`), FVFD operator modifications (live on `dev/fvfd-core`),
slbm-paper publication state (lives on `slbm-paper`).

## Production architecture

```text
src/kraken_e/                        # this branch's owner directory
  KrakenE.jl                         # module entry / include hub (optional;
                                     # otherwise plain include from src/Kraken.jl)
  leaf_block.jl                      # LeafBlock2D type, allocator, accessors
  pipeline.jl                        # apply_bcs!, exchange_halo!,
                                     # compute_macro, collide!, stream!
  bcs.jl                             # half-way BB, moving wall, periodic-wrap
  canaries.jl                        # init helpers + analytical solutions for
                                     # Poiseuille / Couette / Taylor-Green

  (later sessions add:)
  cf_faces.jl                        # S3 D4
  interface_flux.jl                  # S4 D5
  moment_projection.jl               # S5 D7
  population_ghost_fill.jl           # S5 D8
  ...
```

Public names use the prefix `KrakenE*` for types and
`kraken_e_*` for free functions, distinct from the existing
top-level Kraken kernels. Rationale: the existing `stream_2d!`,
`collide_2d!` etc. operate on flat single-block arrays, while
`kraken_e_stream_2d!` operates on a `LeafBlock2D` and respects
ghost layers and cell-kind flags. Two parallel APIs avoid silent
behavioural drift when the FVFD/LBM interface is layered in.

## Allowed edit zones (this branch)

```text
src/kraken_e/                                          # block runtime + LBM bulk + interface
test/kraken_e/test_S<n>_*.jl                           # per-session canaries
docs/agent/                                             # roadmap, derivations, this contract
src/Kraken.jl                                           # include + export adjustments ONLY
test/runtests.jl                                        # include line for new test ONLY
```

Everything else is **read-only** on this branch. In particular:

- `src/fvfd/`, `src/kernels/`, `src/drivers/`, `src/lattice/`,
  `src/io/`, `src/multiblock/`, `src/refinement/`, `src/postprocess.jl`,
  `src/simulation_runner.jl` — read-only.
- `Project.toml`, `Manifest.toml` — read-only (deps already cover
  StaticArrays + KernelAbstractions).
- `benchmarks/`, `examples/`, `hpc/` — read-only.
- `docs/design/`, `docs/src/` — read-only on this branch.
- `output/`, `bench_*` files at repo root — read-only.

## Forbidden shortcuts

- **No reuse of `src/multiblock/`.** That code is gmsh-derived and
  shaped for the SLBM curvilinear path. AMR-tree leaves get their
  own type (`LeafBlock2D`) in `src/kraken_e/`. No `using` or `import`
  of `BlockState2D`, `MultiBlock2D`, `allocate_block_state_2d`, or
  any other `src/multiblock/` symbol from inside `src/kraken_e/`.
- **No reuse of `src/refinement/`.** That code is the patch-based
  nested refinement from v0.1; Kraken-E supersedes it. The two will
  coexist on the branch but `src/kraken_e/` must not depend on it.
- **No AMR before S3.** S2 ships a single-leaf canary. Multi-leaf,
  same-level peer exchange, and coarse/fine ghosts land in S3+.
- **No FVFD operator changes here.** Operator fixes/extensions go on
  `dev/fvfd-core` and are propagated by rebase. The block runtime
  consumes FVFD via the public surface declared in
  `docs/design/fvfd_operator_library.md`.
- **No SLBM, curvilinear, or O-grid features.** Kraken-E is Cartesian
  by design. The SLBM track ships on `slbm-paper` separately.
- **No macro-flow tuning.** Validation is against analytical canaries
  only. Cd, MLUPS, drag, vortex-shedding St — forbidden until S7
  green per the plan §14.
- **No tightening of canary thresholds to make the test pass.** The
  D2 plan calls for L2 ≤ 1% on 32×32 at τ=0.8. If the implementation
  cannot meet that, escalate BLOCKED with a tightened-brief proposal;
  do not relax the test.
- **No new dependencies.** `Project.toml` is read-only. StaticArrays
  and KernelAbstractions are already in `[deps]`.
- **No GPU work in S2.** CPU only. Metal/CUDA validation is deferred
  to a separate session once the CPU canary ladder is green and the
  block runtime stabilizes.

## Validation ladder

CPU first, in this order, for any kraken_e change in S2:

1. `julia --project=. -e 'using Kraken'` loads without error.
2. `git diff --check` clean (no whitespace errors in the diff).
3. `julia --project=. -e 'using Pkg; Pkg.test(test_args=["kraken_e_S2"])'`
   exits 0 with all four S2 canaries green (equilibrium-fixed,
   Poiseuille, Couette, Taylor-Green).
4. Existing FVFD operator tests still green (no regression in
   `dev/fvfd-core` consumed surface): `julia --project=. -e
   'using Pkg; Pkg.test(test_args=["fvfd"])'`.
5. (Optional but recommended in S2) `using Pkg; Pkg.test()` full
   suite to confirm no collateral damage to existing Kraken tests.

S2 closes when (1)..(3) hold. (4) is a soft check.

## Commit and staging rules

- Orchestrator (top-level Claude / Boss) commits, not Codex and not
  the Department subagent. The branch governor protocol applies.
- One atomic commit per session (S2 = one commit). Use conventional
  commit format:
  ```text
  <type>(kraken-e): <subject>

  Session: S<n>
  Pilot: codex (skill: kraken-codex-pilot)
  Exit: <one-line exit criterion satisfied>
  ```
- Stage only files inside the allowed edit zones above.
- Do not stage `Project.toml`, `Manifest.toml`, benchmark logs,
  unrelated dirty files, or anything under `output/`.
- Do not stage `.engineer_brief.md` or `.engineer_logs/`. These
  are ephemeral pilot artifacts.

## Documentation expectations

- This branch owns the Kraken-E block runtime and the FVFD/LBM
  interface (latter from S4 onward). Operator changes do **not**
  happen here.
- Every session updates one row of `docs/agent/kraken_e_roadmap.md`
  (status → `done`) with a one-line note pointing at the derivation
  doc and the test path. The row update is the last edit before the
  commit.
- Derivation docs go under `docs/agent/kraken_e_S<n>_<topic>.md` and
  follow the structure used by S2: purpose/scope, derivation, test
  plan, exit criterion (a shell command), out-of-scope list, failure
  modes the test must catch.

## Stop criteria (inherited from plan §14)

A session immediately stops and the architecture is re-evaluated if any
of the following fires. Do **not** tune around any of these.

- D2Q9 moment reconstruction cannot preserve `rho, j` exactly while
  imposing the chosen interface stress (S5);
- coarse/fine FVFD fluxes do not telescope to roundoff on a two-level
  patch (S4);
- wall + interface + Guo force creates a persistent density jump not
  explained by the discretization order (S7);
- spectral interface analysis shows an unstable interface-localized mode
  (post-S5 spectral check);
- positivity requires opaque clipping for ordinary target Mach numbers
  (S5);
- FVFD extraction cannot be isolated from viscoelastic branch experiments
  (S1; should be green by now — flag immediately if found regressed).

Additional S2-specific stops:

- `ng = 1` D2Q9 streaming kernel cannot be made uniform without
  in-kernel BC branching — escalate; would imply `ng = 2` or a
  different BC convention.
- D1 ownership has a field with no determinate owner under the
  AMR-tree taxonomy — escalate; the Boss refines D1.

## Coordination with sister branches

- `dev/fvfd-core` is the FVFD core owner. Operator changes there
  propagate here by rebase. `dev/kraken-e-fvfd-blocks` does not
  reimplement FVFD operators.
- `dev-viscoelastic` is a parallel consumer of `dev/fvfd-core`; the
  two consumer branches do not exchange code directly. Cross-branch
  fixes go through `dev/fvfd-core` first.
- `slbm-paper` carries the source-of-truth plan doc and continues
  the SLBM publication line. It does not depend on Kraken-E.
