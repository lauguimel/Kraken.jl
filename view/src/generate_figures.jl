# Batch figure generation entry point.

"""
    generate_figures(spec; output_dir="docs/src/assets/figures") -> Vector{String}

Loop over `spec` (a vector of `NamedTuple`) and generate every described
figure. Each spec entry is a `NamedTuple` of the form:

    (case, figure_type, output, options [, krk, data])

- `case::String` — human-readable case name (used for titles).
- `figure_type::Symbol` — one of `:heatmap`, `:profile`, `:convergence`,
  `:streamlines`.
- `output::String` — filename (relative to `output_dir`).
- `options::NamedTuple` — forwarded kwargs to the figure builder.
- `krk::String` (optional) — path to a `.krk` file to run via
  `Kraken.run_simulation` before plotting.
- `data::NamedTuple` (optional) — precomputed data used instead of running
  a simulation. The required fields depend on `figure_type`:
    * `:heatmap`      → `(; field)`
    * `:profile`      → `(; field, line)`
    * `:convergence`  → `(; N_values, errors)`
    * `:streamlines`  → `(; ux, uy)`

Returns the list of absolute paths of generated figure files.

# Examples
```julia
spec = [
    (case="demo", figure_type=:heatmap, output="demo_hm.png",
     options=(colormap=:viridis,),
     data=(; field=rand(64, 64))),
    (case="order2", figure_type=:convergence, output="order2.png",
     options=(theoretical_order=2.0,),
     data=(; N_values=[32,64,128,256],
             errors=[1.0, 0.25, 0.0625, 0.015625])),
]
paths = generate_figures(spec; output_dir=mktempdir())
```
"""
function generate_figures(spec; output_dir::AbstractString="docs/src/assets/figures")
    isdir(output_dir) || mkpath(output_dir)
    generated = String[]

    for entry in spec
        ftype = entry.figure_type
        opts  = hasproperty(entry, :options) ? entry.options : NamedTuple()
        data  = _resolve_data(entry)

        fig = if ftype === :heatmap
            heatmap_field(data.field; opts...)
        elseif ftype === :profile
            line = hasproperty(data, :line) ? data.line : :vertical
            profile_plot(data.field, line; opts...)
        elseif ftype === :convergence
            convergence_plot(data.N_values, data.errors; opts...)
        elseif ftype === :streamlines
            streamline_plot(data.ux, data.uy; opts...)
        else
            error("generate_figures: unknown figure_type $(ftype)")
        end

        outpath = joinpath(output_dir, entry.output)
        save_figure(fig, outpath)
        push!(generated, abspath(outpath))
    end

    return generated
end

# Resolve data for an entry: prefer `data` field if present; otherwise run
# the simulation from `krk` and build a NamedTuple matching the figure type.
function _resolve_data(entry)
    if hasproperty(entry, :data) && entry.data !== nothing
        return entry.data
    end
    if !hasproperty(entry, :krk) || entry.krk === nothing
        error("generate_figures: entry needs either `data` or `krk`")
    end
    result = Kraken.run_simulation(entry.krk)
    # Assemble a generic NamedTuple from simulation result.
    return (; field=hasproperty(result, :umag) ? result.umag :
                     sqrt.(result.ux .^ 2 .+ result.uy .^ 2),
              ux=result.ux, uy=result.uy,
              line=:vertical)
end
