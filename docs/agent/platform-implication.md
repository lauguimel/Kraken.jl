---
module: platform
path: src/platform/
owner_concern: method-agnostic-contract
status: phase-1
last_verified: 2026-06-09
depends_on: []
---

# platform — module implication map

The `platform/` module is the **stable, method-agnostic contract** every method and physics
enters behind (the "6 nouns"). **Phase 0** added the abstract types + capability introspection.
**Phase 0b** added the first concrete wrapper (`LBM`, `LBMSolution`, behaviour-preserving
`solve`/`sample` forwarding to `run_simulation`). **Phase 1** adds observables + `observe`/`predict`
— quantities comparable to data, computed **via `sample`** (the path to calibration). Still
additive, LBM untouched. The remaining verbs (`residual`, `fit`) and `ParameterSpace`/closures land
in later phases. Design dossier: `docs/platform/` (00-BILAN … 06-WORKFLOW); free-DOF rationale in
`docs/platform/05-DOF-LIBRES.md`.

## Public surface

Defined under `src/platform/`, re-exported by `Kraken`:

Phase 0 (`contract.jl`):
- `AbstractProblem`, `AbstractMethod`, `AbstractSolution`, `AbstractObservable`, `AbstractClosure`.
- `@enum Capability` = `ForwardSolve`, `GPUExecution`, `SteadyAdjoint`, `TransientAdjoint`,
  `FiniteDiff`, `NeuralClosure`.
- `capabilities(m::AbstractMethod) -> Set{Capability}` — default empty; a method opts in.

Phase 0b (`solution.jl`, `sample.jl`):
- `LBM <: AbstractMethod` — `capabilities(::LBM)` = `{ForwardSolve, GPUExecution, SteadyAdjoint}`.
- `LBMSolution{R} <: AbstractSolution` — thin wrapper over the `run_simulation` `NamedTuple`.
- `solve(problem, ::LBM; kwargs...) -> LBMSolution` — forwards verbatim to `run_simulation`.
- `sample(sol::LBMSolution, field[, query])` — pass-through query (`getproperty`, `:`, `idx::Tuple`).

Phase 1 (`observe.jl`):
- `observe(sol::AbstractSolution, o::AbstractObservable) -> Prediction` — computes `o` via `sample`
  ONLY (generic on the solution type; new observables add a method here).
- `predict(problem, ::AbstractMethod, o; kwargs...)` = `observe(solve(problem, method; kwargs...), o)`.
- `Prediction{O,V}` (observable + value); observables `FieldProbe(field, index::Tuple)`,
  `LineProfile(field, indices)`, `FieldReduction(field, reducer)`. **Integral QoIs (drag, Nusselt)
  deferred** — they need boundary integration, not pure `sample`.

Second concrete method (`src/methods/inc_ns/method.jl`, mirrors the LBM wrapper):
- `IncNS <: AbstractMethod` — `IncNS(driver)` with driver ∈
  `{:simple, :cavity, :cavity_mg, :projection, :manifold}`;
  `capabilities(::IncNS)` = `{ForwardSolve}` only (CUDA seam is manual-load).
- `IncNSSolution{R} <: AbstractSolution` — thin wrapper over the driver `NamedTuple`.
- `solve(params::NamedTuple, m::IncNS)` — splats `params` verbatim into the matching
  `solve_incns_*` driver; `sample` pass-through methods as for `LBMSolution`.
  Registration map: `docs/agent/incns-platform-implication.md`.

## Reads from

Phase 0 reads nothing. **Phase 0b**: `solve(_, ::LBM)` forwards to `run_simulation`
(`src/simulation_runner.jl`) at call time; `LBMSolution` wraps the `NamedTuple` it returns
(`ρ`, `ux`, `uy`, `setup`). **Phase 1**: `observe` reads a solution only through `sample`; `predict`
also goes through `solve`. No new external read; nothing imported beyond `Base`. The runner
reference is **lazy** (resolved at call time), so the `platform/` includes can precede the runner.

## Writes to

Nothing persistent. No mutation, no global registries, no side effects, no I/O. `capabilities`,
`solve`, `observe`, `predict` each return fresh values; `sample` returns the wrapped arrays by
reference (`sample(sol, :ux) === sol.result.ux`) or an indexed scalar. The abstract/spec types carry
no mutable state. `solve`/`predict` do not mutate `problem`.

## Backend constraints

**Backend-irrelevant by construction.** Type definitions + thin dispatch; nothing here enters a
kernel or allocates in a loop. Backend portability is expressed *through* the contract (a method
declares `GPUExecution ∈ capabilities(m)`). `solve`/`sample`/`observe` add no backend coupling —
whatever backend `run_simulation` ran on is preserved untouched inside `LBMSolution`. `observe`
operates on the host arrays returned by the runner.

## Failure modes

- The default `capabilities(::AbstractMethod) = Set{Capability}()` is a deliberate **fail-safe**:
  an undeclared method appears *incapable* rather than silently wrong.
- `solve`/`sample`/`observe` are pass-throughs/derivations, inheriting `run_simulation`'s failure
  modes verbatim (the parity test pins "no behaviour change"). `sample(sol, :badfield)` raises the
  normal `getproperty` error; an out-of-range probe index raises the normal bounds error — both
  intentional, not swallowed.
- **Name-collision watch**: `solve`/`sample`/`observe`/`predict` are exported as Kraken's own
  generic functions (no dep exports them today). When SciML enters (Phase 4), switch `solve` to
  extend `CommonSolve.solve` rather than shadow it.
- The real risk stays **architectural**: over-abstraction. Guardrail (`docs/platform/06-WORKFLOW.md`
  red-team #1) — generalize the contract only when a 2nd concrete `AbstractMethod` forces it.

## Touch order

1. `src/platform/contract.jl` — the 6 types + `Capability` + `capabilities`.
2. `src/platform/solution.jl` — `LBM`, `LBMSolution`, `solve` (forwards to `run_simulation`).
3. `src/platform/sample.jl` — `sample` pass-through.
4. `src/platform/observe.jl` — `observe`/`predict`/`Prediction` + observables
   (`FieldProbe`/`LineProfile`/`FieldReduction`).
5. `src/Kraken.jl` — the `# --- Platform contract ---` include + export block (a choke file;
   edits serialized on `dev/platform`).
6. `test/platform/contract_parity_test.jl` — capabilities, bit-for-bit parity vs `run_simulation`,
   and observe/predict (Phase 1).
7. Later phases add siblings under `src/platform/` (`residual.jl`, `calibration.jl`, `closure.jl`)
   — see `docs/platform/02-PLAN-IMPLEMENTATION.md`.
