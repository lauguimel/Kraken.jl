---
module: platform
path: src/platform/
owner_concern: method-agnostic-contract
status: scaffold
last_verified: 2026-06-09
depends_on: []
---

# platform — module implication map

The `platform/` module is the **stable, method-agnostic contract** every method and
physics enters behind (the "6 nouns"). **Phase 0 = types + capability introspection
ONLY** — no behaviour, no dependencies, fully additive. The verbs (`solve`, `sample`,
`observe`, `residual`, `fit`) and `ParameterSpace`/closures land in later phases. Design
dossier: `docs/platform/` (00-BILAN … 06-WORKFLOW); free-DOF rationale in
`docs/platform/05-DOF-LIBRES.md`.

## Public surface

Defined in `src/platform/contract.jl`, re-exported by `Kraken`:

- `AbstractProblem` — method-agnostic problem description (domain/BCs/physics).
- `AbstractMethod` — a discretization/solve method; the caller names it (no auto-selection).
- `AbstractSolution` — a queryable solution; internal storage is private (query via `sample`, Phase 1).
- `AbstractObservable` — a quantity comparable to data, defined via `sample`, never via internals.
- `AbstractClosure` — an injectable analytic/learned term, evaluated inside the residual.
- `@enum Capability` = `ForwardSolve`, `GPUExecution`, `SteadyAdjoint`, `TransientAdjoint`,
  `FiniteDiff`, `NeuralClosure`.
- `capabilities(m::AbstractMethod) -> Set{Capability}` — default empty; a method opts in by
  adding a method returning its set.

## Reads from

Nothing. This module is the foundation of the contract: it has **no dependencies** on any
sibling module (`depends_on: []`). It defines only abstract types and one enum; it imports
nothing beyond `Base`/`Core`.

## Writes to

Nothing. No mutation, no global registries, no side effects, no I/O. `capabilities` returns a
fresh empty `Set{Capability}` for the default method. The abstract types carry no fields and
own no state.

## Backend constraints

**Backend-irrelevant by construction.** These are compile-time type definitions; nothing here
enters a kernel or allocates at runtime. Backend portability is expressed *through* the
contract: a method declares `GPUExecution ∈ capabilities(m)` rather than the contract pinning
any backend. No `T`/precision coupling, no GPU/CPU branch.

## Failure modes

- Phase 0 is inert, so there is no runtime failure mode yet. The default
  `capabilities(::AbstractMethod) = Set{Capability}()` is a deliberate **fail-safe**: a method
  that forgets to declare its capabilities appears *incapable* (e.g. `fit` will not assume a
  gradient) rather than silently wrong.
- The real risk is **architectural, not runtime**: over-abstraction. Guardrail (per
  `docs/platform/06-WORKFLOW.md` red-team #1) — the contract generalizes only when a 2nd
  concrete `AbstractMethod` forces it; do NOT add verbs/fields speculatively. Each new method
  must pass the parity test before the contract is widened.

## Touch order

For any contract change, inspect in this order:

1. `src/platform/contract.jl` — the only file today (the 6 types + `Capability` + `capabilities`).
2. `src/Kraken.jl` — the include + export block (`# --- Platform contract ---`), a choke file:
   edits here are serialized on `dev/platform`.
3. Later phases add siblings under `src/platform/` (`solution.jl`, `sample.jl`, `observe.jl`,
   `residual.jl`, `calibration.jl`, `closure.jl`) — see `docs/platform/02-PLAN-IMPLEMENTATION.md`.
