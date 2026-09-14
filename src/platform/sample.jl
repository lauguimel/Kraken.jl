"""
    sample(sol::AbstractSolution, field::Symbol)
    sample(sol::AbstractSolution, field::Symbol, query)

Query a solution by field name (and optional location `query`), defined on the platform
interface rather than the method's internal storage — so the same call works across
methods regardless of how the result is stored. Phase 0 (`LBMSolution`): a faithful
pass-through to the wrapped result.

- `sample(sol, :ux)` / `sample(sol, :ux, :)` → the whole field.
- `sample(sol, :ux, (i, j))` → the value at index `(i, j)`.
"""
sample(sol::LBMSolution, field::Symbol) = getproperty(sol.result, field)
sample(sol::LBMSolution, field::Symbol, ::Colon) = sample(sol, field)
sample(sol::LBMSolution, field::Symbol, idx::Tuple) = sample(sol, field)[idx...]
