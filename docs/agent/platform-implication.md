---
module: platform
path: src/platform/
owner_concern: method-agnostic-contract
status: phase-0b
last_verified: 2026-06-09
depends_on: []
---

# platform — module implication map

The `platform/` module is the **stable, method-agnostic contract** every method and
physics enters behind (the "6 nouns"). **Phase 0** added the abstract types + capability
introspection. **Phase 0b** adds the first concrete wrapper — `LBM`, `LBMSolution`, and
behaviour-preserving `solve`/`sample` that forward to `run_simulation`. Still additive,
LBM untouched. The remaining verbs (`observe`, `residual`, `fit`) and
`ParameterSpace`/closures land in later phases. Design dossier: `docs/platform/`
(00-BILAN … 06-WORKFLOW); free-DOF rationale in `docs/platform/05-DOF-LIBRES.md`.

## Public surface

Defined under `src/platform/`, re-exported by `Kraken`:

Phase 0 (`contract.jl`):
- `AbstractProblem` — method-agnostic problem description (domain/BCs/physics).
- `AbstractMethod` — a discretization/solve method; the caller names it (no auto-selection).
- `AbstractSolution` — a queryable solution; internal storage is private (query via `sample`).
- `AbstractObservable` — a quantity comparable to data, defined via `sample`, never via internals.
- `AbstractClosure` — an injectable analytic/learned term, evaluated inside the residual.
- `@enum Capability` = `ForwardSolve`, `GPUExecution`, `SteadyAdjoint`, `TransientAdjoint`,
  `FiniteDiff`, `NeuralClosure`.
- `capabilities(m::AbstractMethod) -> Set{Capability}` — default empty; a method opts in by
  adding a method returning its set.

Phase 0b (`solution.jl`, `sample.jl`):
- `LBM <: AbstractMethod` — the lattice-Boltzmann method; `capabilities(::LBM)` =
  `{ForwardSolve, GPUExecution, SteadyAdjoint}`.
- `LBMSolution{R} <: AbstractSolution` — thin wrapper over the `run_simulation` `NamedTuple`
  (single field `result`).
- `solve(problem, ::LBM; kwargs...) -> LBMSolution` — forwards verbatim to `run_simulation`
  (parity-tested); parameters pass as `kwargs`. The `solve(problem, method, p)` form arrives
  with `ParameterSpace` in Phase 2.
- `sample(sol::LBMSolution, field[, query])` — pass-through query: `getproperty(result, field)`,
  with `:` (whole field) and `idx::Tuple` (indexed) variants.

## Reads from

Phase 0 reads nothing. **Phase 0b**: `solve(_, ::LBM)` forwards to `run_simulation`
(`src/simulation_runner.jl`) at call time, and `LBMSolution` wraps the `NamedTuple` it returns
(`ρ`, `ux`, `uy`, `setup`). The reference is **lazy** (resolved when `solve` is *called*), so the
`platform/` includes can precede the runner in `src/Kraken.jl`. No other sibling is read; nothing
is imported beyond `Base`.

## Writes to

Nothing. No mutation, no global registries, no side effects, no I/O. `capabilities` returns a fresh
`Set`; `solve` returns a fresh `LBMSolution` wrapping the runner's result; `sample` returns the
wrapped arrays (by reference — `sample(sol, :ux) === sol.result.ux`) or an indexed scalar. The
abstract types carry no fields/state. `solve` does not mutate `problem`.

## Backend constraints

**Backend-irrelevant by construction.** Type definitions + thin dispatch; nothing here enters a
kernel or allocates in a loop. Backend portability is expressed *through* the contract: a method
declares `GPUExecution ∈ capabilities(m)`. `solve`/`sample` add no backend coupling — whatever
backend `run_simulation` ran on is preserved untouched inside `LBMSolution`.

## Failure modes

- The default `capabilities(::AbstractMethod) = Set{Capability}()` is a deliberate **fail-safe**: a
  method that forgets to declare its capabilities appears *incapable* (e.g. `fit` won't assume a
  gradient) rather than silently wrong.
- `solve`/`sample` are pass-throughs, so they inherit `run_simulation`'s failure modes verbatim
  (the parity test pins "no behaviour change"). `sample(sol, :badfield)` raises the normal
  `getproperty` error — intentional, not swallowed.
- **Name-collision watch**: `solve`/`sample` are exported as Kraken's own generic functions (no dep
  exports them today). When SciML enters (Phase 4), switch `solve` to extend `CommonSolve.solve`
  rather than shadow it.
- The real risk stays **architectural**: over-abstraction. Guardrail (`docs/platform/06-WORKFLOW.md`
  red-team #1) — generalize the contract only when a 2nd concrete `AbstractMethod` forces it.

## Touch order

1. `src/platform/contract.jl` — the 6 types + `Capability` + `capabilities`.
2. `src/platform/solution.jl` — `LBM`, `LBMSolution`, `solve` (forwards to `run_simulation`).
3. `src/platform/sample.jl` — `sample` pass-through.
4. `src/Kraken.jl` — the `# --- Platform contract ---` include + export block (a choke file;
   edits serialized on `dev/platform`).
5. `test/platform/contract_parity_test.jl` — capabilities + bit-for-bit parity vs `run_simulation`.
6. Later phases add siblings under `src/platform/` (`observe.jl`, `residual.jl`, `calibration.jl`,
   `closure.jl`) — see `docs/platform/02-PLAN-IMPLEMENTATION.md`.
