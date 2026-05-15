# Branch Contract — dev/fvfd-core

Date: 2026-05-15
Branch: `dev/fvfd-core`
Worktree: `/Users/guillaume/Documents/Recherche/Kraken.jl-fvfd-core`
Plan: [kraken_e_fvfd_interface_plan_2026-05-15.md](../../docs/agent/kraken_e_fvfd_interface_plan_2026-05-15.md) (lives on slbm-paper)
Roadmap: [kraken_e_roadmap.md](kraken_e_roadmap.md)
Resource bilan: [fvfd_resource_bilan_2026-05-15.md](fvfd_resource_bilan_2026-05-15.md)

## Branch objective

Own the neutral FVFD operator core for Kraken-E. This branch hosts the
canonical implementation of FV/FD specs, lowering, and 2D/3D operators
used by all consumers (`dev/kraken-e-fvfd-blocks`, `dev-viscoelastic`
post-rebase, future v0.2/v0.3 architecture branches).

## Production architecture

```text
src/fvfd/           # FVFD operator core (this branch owns it)
  FVFD.jl           # module entry / include hub
  specs.jl          # domain BC, field BC, patch, geometry specs
  lowering_2d.jl    # spec → backend array lowering (CPU/Metal/CUDA)
  operators_2d.jl   # gradient, divergence, advection, traction kernels
  (later: lowering_3d.jl, operators_3d.jl, coarse_fine_faces_*.jl)
```

The FVFD core is consumed by Kraken via a single include in
`src/Kraken.jl`. Public names follow `FVFD*` / `fvfd_*` prefix.
`LogFV*` / `logfv_*` aliases for FVFD-core names live in the FVFD
core (they pre-existed in the viscoelastic consumer and must remain
binary-compatible). Consumer-specific `logfv_*` operators (polymer
force, log-FV-specific advection wrappers) stay on `dev-viscoelastic`.

## Allowed edit zones (this branch)

```text
src/fvfd/                       # FVFD core operator/spec/lowering files
test/test_fvfd_*.jl             # FVFD operator tests
docs/design/fvfd_*.md           # operator library design docs
docs/agent/                     # roadmap, bilan, branch contract, derivations
src/Kraken.jl                   # include + export adjustments ONLY, no logic
```

Everything else (`kernels/`, `drivers/`, `io/`, `lattice/`,
`postprocess.jl`, `simulation_runner.jl`, `Project.toml`,
`benchmarks/`, `examples/`, `hpc/`) is read-only on this branch.

## Forbidden shortcuts

- No AMR/LBM/viscoelastic kernel changes here. Promotion of LBM bulk
  changes happens on `dev/kraken-e-fvfd-blocks`. Viscoelastic
  consumer changes happen on `dev-viscoelastic`.
- No benchmark experiments or macro-flow tuning. The FVFD core is
  validated against analytical unit canaries, not Cd/MLUPS targets.
- No GPU work in S1 (CPU only). Metal/CUDA validation is deferred to
  a separate session once the CPU ladder is green.
- No new physical conventions added on this branch. The branch
  inherits the FVFD operator conventions documented in
  `docs/design/fvfd_operator_library.md`. Convention changes go
  through a dedicated derivation doc + governor protocol.

## Validation ladder

CPU first, in this order, for any FVFD operator change:

1. `using Kraken` loads without error.
2. `test/test_fvfd_operators_2d.jl` runs and exits green on CPU.
3. Metal/CUDA smoke test (deferred, separate session).

S1 closes when (1) and (2) hold on CPU.

## Commit and staging rules

- Orchestrator (top-level Claude) commits, not Codex and not the
  session executor subagent.
- One atomic commit per session (S1 = one commit). Use conventional
  commit format:
  ```text
  <type>(fvfd): <subject>

  Session: S<n>
  Pilot: codex (skill: <skill>)
  Exit: <one-line exit criterion satisfied>
  ```
- Stage only files inside the allowed edit zones above.
- Do not stage `Project.toml`, `Manifest.toml`, benchmark logs,
  or unrelated dirty files.

## Documentation expectations

- This branch is the FVFD owner. Operator changes here ripple to
  consumer branches (`dev/kraken-e-fvfd-blocks`, `dev-viscoelastic`)
  via rebase or cherry-pick. Consumers do not independently modify
  FVFD operators.
- Every session updates one row of `docs/agent/kraken_e_roadmap.md`
  (status → `done` with a one-line note pointing at derivation + test).
- Derivation docs go under `docs/agent/kraken_e_S<n>_<topic>.md`.
- The FVFD public-surface contract is `docs/design/fvfd_operator_library.md`.
  Changes there require a derivation doc.

## Stop criteria (inherited from plan §14)

A session immediately stops and the architecture is re-evaluated if
FVFD extraction cannot be isolated from viscoelastic branch
experiments (S1-relevant), or if any of the plan §14 stop criteria
fires on a later session.
