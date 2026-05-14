# Kraken Branch Contract: AMR-D Conservative Tree

Date: 2026-05-08
Active branch observed: `slbm-paper`

This branch contains unrelated dirty work. For AMR-D tasks, treat this file as
the local operating contract and stage only files that belong to the AMR-D
change being validated.

## Objective

Close voie D as Kraken's publishable static AMR feature before starting a
replacement AMR architecture.

Voie D means:

- fixed conservative-tree AMR, built from `.krk` `Refine` blocks or explicit
  conservative-tree specs;
- route-native D2Q9 first, then D3Q19;
- static topology during a run: no online adaptation in the publication path;
- nested 2D levels up to at least level 4 with adjacent leaves differing by no
  more than one refinement level;
- reproducible macro-flow dashboards and CSV outputs from `.krk` files;
- CPU correctness first, Metal for local coarse GPU debug, Aqua H100 for real
  GPU performance claims.

Do not mix AMR-D with the future AMR-E/Basilisk-like single-time-step idea. AMR-E
may be documented as a future track, but it must not change AMR-D gates or APIs
until AMR-D reaches the publication milestone.

## Production Architecture

AMR-D uses conservative integrated populations on a cell-centered tree.

- 2D production path:
  - `src/refinement/conservative_tree_spec_2d.jl`
  - `src/refinement/conservative_tree_routes_2d.jl`
  - `src/refinement/conservative_tree_subcycling_2d.jl` (umbrella, 6 lines)
    includes:
      - `subcycling_schedule_2d.jl`
      - `subcycling_explosion_2d.jl`
      - `subcycling_coalesce_2d.jl`
      - `subcycling_streaming_2d.jl`
      - `subcycling_wall_phase_2d.jl`
      - `subcycling_driver_2d.jl`
  - Split landed 2026-05-14, commits 63698a3..44f54b1. No semantic
    change; baseline test signature preserved.
  - `src/refinement/conservative_tree_macroflows_subcycled_2d.jl`
  - `src/refinement/conservative_tree_gpu_pack_2d.jl`
  - `src/refinement/conservative_tree_krk_validation_2d.jl`
- KRK and benchmarks:
  - `benchmarks/krk/amr_d_*`
  - `benchmarks/amr_d_*`
  - `docs/design/amr_d_*`
- Tests:
  - `test/test_conservative_tree_*`
  - `test/test_amr_d_krk_validation_2d.jl`

The current GPU route is KernelAbstractions-based. It must use precomputed
compact route packs, bounded slots, and backend-compatible kernels. Host-side
allocation and dynamic route construction are allowed in topology/pack builders,
not inside hot kernels.

## Current AMR-D Status

Green enough to build on:

- one-level AMR-D route-native channels and simple solid masks;
- nested 2D channel scheduler for Poiseuille/Couette through CPU and local
  Metal smoke paths;
- KRK dispatch for AMR-D nested channel cases;
- local Metal single-step dashboards for nested channel debug cases.

Not closed:

- nested BFS/open-channel validation;
- nested square/cylinder validation and drag convergence;
- solid-interface GPU ledgers;
- production-grade GPU performance;
- H100 benchmark versus CPU and Cartesian baselines;
- D3Q19 conservative-tree publication path.

### Recent changes (2026-05-14)

- Subcycling monolith split into 6 cohesive files (<=700 lines each).
- 6-marche validation ladder added: test/test_amr_d_ladder.jl.
- Ladder runs: marche 1 pure stream; 2 BGK no force;
  3 BGK + constant force, periodic; 4 BB Poiseuille; 5 one-level
  refinement; 6 nested-4 refinement.
- Current status: marches 1-2 pass; marche 3 fails with
  max_err ~= gx/2 (suspected half-force convention ambiguity,
  parked pending disambiguation).

## Validation Ladder

Every AMR-D change climbs this ladder:

1. Analytical/unit canary.
2. Surgical patch test on the smallest tree that exercises the operator.
3. KRK smoke with a small number of steps.
4. Coarse macro-flow dashboard.
5. Longer Metal or Aqua/H100 run.
6. Publication-scale comparison and plots.

When a macro-flow fails, stop macro debugging and reduce to the smallest route,
ledger, boundary, or collision canary that can explain the failure.

Required local canaries before committing AMR-D scheduler/GPU/KRK changes:

```bash
julia --project=. -e 'using Test; using Kraken; include("test/test_amr_d_ladder.jl")'
julia --project=. -e 'using Test; using Kraken; include("test/test_conservative_tree_gpu_pack_2d.jl")'
julia --project=. -e 'using Test; using Kraken; include("test/test_amr_d_krk_validation_2d.jl")'
```

The ladder above is the primary gate. Work on marche N is blocked while any
marche M < N is red. Tolerances are explicit and tied to physics (see the file
header). Do not relax tolerances to make the suite green.

Optional local Metal smoke when GPU route code changes:

```bash
KRAKEN_TEST_METAL=1 julia --project=. -e 'using Test; using Kraken; include("test/test_conservative_tree_gpu_pack_2d.jl")'
```

Dashboard smoke for nested channel `.krk` cases:

```bash
KRK_AMR_D_TEMP_BACKEND=metal \
KRK_AMR_D_TEMP_T=float32 \
KRK_AMR_D_TEMP_SINGLE_STEP=1 \
KRK_AMR_D_TEMP_MAX_STEPS=3200 \
KRK_AMR_D_TEMP_CASES=poiseuille_yband_nested4_debug.krk \
julia --project=. benchmarks/amr_d_macroflow_temporal_convergence_2d.jl
```

## File Size Limits

Files in src/refinement/ must not exceed 700 lines. Before editing any file at
or above this threshold, propose a mechanical split (one module per cohesive
concern, one commit per module, baseline test signature preserved) and wait for
user approval. The 2026-05-14 split of subcycling_2d is the reference protocol.

Other src/ files have no hard cap, but the same principle applies: if a session
keeps adding to a 1000+ line file, the next sustainable move is a split, not
another inline patch.

## Acceptance Gates

Correctness gates:

- active mass drift must be roundoff-level or explicitly reported as
  `max_raw_mass_rel_drift` before correction;
- AMR-D channel profiles must compare against both analytic steady profiles and
  a classic Cartesian transient at the same physical time;
- nested level transitions must not create unexplained jumps in exported
  profiles;
- BFS/open-boundary tests must pass surgical inlet/outlet/open-channel canaries
  before any BFS macro claim;
- square must be validated before cylinder;
- cylinder drag/lift claims require convergence against Cartesian references,
  not just a visually plausible field.

Performance gates:

- no speedup claim from Metal alone;
- local Metal is only a coarse/debug accelerator;
- H100/Aqua runs must report cells/sec or MLUPS, memory footprint, precision,
  active leaf counts, and Cartesian-equivalent baseline;
- avoid per-step host-device synchronization in production performance paths.

## Forbidden Shortcuts

- Do not hide a failed interface/boundary bug behind looser macro tolerances.
- Do not suppress mass guards without exporting the raw drift.
- Do not treat Cd improvement as proof of correctness.
- Do not run long CPU simulations as the main development loop.
- Do not introduce hidden CPU fallbacks in a path advertised as GPU.
- Do not write dynamic allocation, least-squares solves, or host callbacks
  inside GPU kernels.
- Do not modify SLBM/body-fit/multiblock files for an AMR-D fix unless the bug
  is proven cross-cutting and documented.
- Do not stage unrelated dirty files.

## Canary Lifecycle

Canaries are short-lived diagnostic scripts (typically under
benchmarks/results/quicklook/ or as ad hoc test files). They are permitted to
find bugs. They are NOT permitted to accumulate.

Rules:

- Any canary that reveals a real bug must, before the session ends, become a
  permanent @test in test/ that exercises the bug. Once the bug is fixed, the
  @test stays as regression.
- Canaries that reveal no bug are deleted at session end.
- The benchmarks/results/quicklook/amr_d_v14 .. amr_d_v41 series and the
  docs/design/amr_d_* audits prior to 2026-05-14 are archived (see
  _archive_2026-05-14/). They are snapshots of past sessions, not current
  truth. Do not read them to "understand the history" — they will mislead.
- If you find yourself about to create amr_d_v(N+1)_canary, stop. Write a
  @test instead, or ask the user to relax the rule with explicit
  justification.

## Milestones

P0: Contract and hygiene

- This contract exists and is kept current.
- Every AMR-D commit stages only intentional files.

P1: 2D route/ledger canaries

- direct routes, boundary routes, C2F, F2C, corners, moving-wall corrections,
  mass correction, and nested level-native variants have surgical tests.

P2: 2D nested channels

- Poiseuille x-band, y-band, wall-ybands, and Couette nested4 are reproducible
  from `.krk`;
- dashboards export mesh, `u`, `rho`, profiles, analytic/reference curves, and
  CSV values;
- CPU and Metal agree on coarse canaries.

P3: 2D open channel and BFS

- open-boundary/open-channel patch tests pass first;
- BFS macro-flow is then validated against Cartesian fields and profiles.

P4: 2D obstacles

- square obstacle convergence against Cartesian;
- cylinder in channel, including off-center lift probe;
- nested4 cylinder only after square and simple cylinder canaries are green.

P5: GPU production path

- route-native AMR-D GPU avoids per-step host synchronization in benchmark mode;
- local Metal smoke remains available;
- Aqua H100 benchmark compares Cartesian versus AMR-D at equal physical time and
  comparable accuracy.

P6: 3D AMR-D

- D3Q19 conservative-tree spec/routes/ledgers;
- 3D patch canaries;
- 3D Poiseuille/Couette or equivalent channel validation;
- 3D GPU smoke, then H100 benchmark.

P7: Publication package

- KRK files reproduce every figure/table;
- CSV and dashboard outputs are archived under deterministic result folders;
- docs state limitations honestly: static AMR-D, supported BCs, GPU scope, and
  known non-goals.

## Commit Rules

Before an AMR-D commit:

```bash
git diff --check
git diff --cached --name-status
```

Commit only after a green canary, a useful diagnostic freeze, or a durable
contract/design update. Commit messages should be factual, for example:

```text
Add AMR-D branch contract
Fix AMR-D nested channel mass correction
Add AMR-D open-channel boundary canary
```

Never push without explicit user confirmation.
