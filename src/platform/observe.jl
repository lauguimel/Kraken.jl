"""
    Prediction(observable, value)

The result of [`observe`](@ref): an `AbstractObservable` paired with the value computed for
it from a solution. This is the quantity compared against data in calibration (Phase 2).
"""
struct Prediction{O,V}
    observable::O
    value::V
end

"""
    FieldProbe(field::Symbol, index::Tuple)

Observable = the value of `field` at grid index `index` (a single sensor point). Compared
e.g. against a probe or a PIV point measurement.
"""
struct FieldProbe <: AbstractObservable
    field::Symbol
    index::Tuple
end

"""
    LineProfile(field::Symbol, indices)

Observable = `field` sampled at each index in `indices` (an iterable of index-tuples) — a
profile along a line. Compared e.g. against a PIV line.
"""
struct LineProfile <: AbstractObservable
    field::Symbol
    indices
end

"""
    FieldReduction(field::Symbol, reducer)

Observable = `reducer` applied to the whole `field` (e.g. `sum`, `maximum`) — a scalar
summary comparable to an integrated/aggregate measurement.
"""
struct FieldReduction <: AbstractObservable
    field::Symbol
    reducer
end

"""
    observe(sol::AbstractSolution, o::AbstractObservable) -> Prediction

Compute observable `o` from solution `sol`, using only [`sample`](@ref) — never the method's
internal storage. Returns a [`Prediction`](@ref). New observables add a method here; they must
go through `sample`, keeping observables decoupled from any method's internal representation.
"""
observe(sol::AbstractSolution, o::FieldProbe)     = Prediction(o, sample(sol, o.field, o.index))
observe(sol::AbstractSolution, o::LineProfile)    = Prediction(o, [sample(sol, o.field, i) for i in o.indices])
observe(sol::AbstractSolution, o::FieldReduction) = Prediction(o, o.reducer(sample(sol, o.field)))

"""
    predict(problem, method::AbstractMethod, o::AbstractObservable; kwargs...) -> Prediction

Run `problem` with `method` and observe `o` in one call:
`observe(solve(problem, method; kwargs...), o)`. This is the forward map a calibration loop
will minimise/differentiate against data (Phase 2).
"""
predict(problem, method::AbstractMethod, o::AbstractObservable; kwargs...) =
    observe(solve(problem, method; kwargs...), o)
