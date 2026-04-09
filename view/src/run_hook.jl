# Bridge Kraken.run_simulation → KrakenView Observable updates.

"""
    make_callback(scene::KrakenScene) -> Function

Build a closure that `Kraken.run_simulation` can call every N steps as
`callback(step, state)`. It pushes the currently-selected field to the
scene's Observable (updating the heatmap in place) and stores a CPU
snapshot in `scene.snapshots` so the post-run time slider can scrub.

The `state` NamedTuple is expected to contain `rho`, `ux`, `uy` as CPU
arrays (see `Kraken.run_simulation`).
"""
function make_callback(scene::KrakenScene)
    return function _krakenview_cb(step::Int, state)
        fname = scene.field_name[]
        arr = _extract_field(state, fname)
        scene.field[] = arr
        push!(scene.snapshots, (; step=step, rho=state.rho,
                                  ux=state.ux, uy=state.uy))
        return nothing
    end
end

function _extract_field(state, fname::Symbol)
    if fname === :umag
        return sqrt.(state.ux .^ 2 .+ state.uy .^ 2)
    elseif fname === :ux
        return state.ux
    elseif fname === :uy
        return state.uy
    elseif fname === :rho
        return state.rho
    elseif fname === :T && hasproperty(state, :T)
        return getproperty(state, :T)
    end
    return sqrt.(state.ux .^ 2 .+ state.uy .^ 2)
end

"""
    run_view(path::String; field=:umag, callback_every=100, kwargs...) -> (scene, result)

Convenience wrapper: builds a [`KrakenScene`](@ref) for `path`, then calls
`Kraken.run_simulation` with a live-updating callback. Returns the
`KrakenScene` and the simulation result NamedTuple.

# Arguments
- `path::String`: path to a `.krk` file.
- `field::Symbol`: initial field to show (`:umag`, `:ux`, `:uy`, `:rho`, `:T`).
- `callback_every::Int`: update the viewer every N LBM steps.
- `kwargs...`: forwarded to `Kraken.run_simulation` (e.g. `Re=100`, `N=128`).

# Examples
```julia
using KrakenView
scene, result = run_view("examples/cavity.krk"; field=:umag, callback_every=200)
display(scene.figure)
```
"""
function run_view(path::String; field::Symbol=:umag,
                  callback_every::Int=100, kwargs...)
    scene = view_krk(path; field=field)
    cb = make_callback(scene)
    result = Kraken.run_simulation(path;
                                    callback=cb,
                                    callback_every=callback_every,
                                    kwargs...)
    return scene, result
end
