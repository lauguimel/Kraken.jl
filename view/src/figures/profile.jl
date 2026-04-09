# 1D profile (line cut) figure.

"""
    profile_plot(field::AbstractMatrix, line::Symbol;
                 axis_value=nothing, reference=nothing, ref_label="analytical",
                 title="", xlabel="position", ylabel="value",
                 label="simulation", Lx=nothing, Ly=nothing)

Extract a 1D profile from a 2D field along a line and plot it.

# Arguments
- `field`: 2D array.
- `line`: `:vertical` (line at fixed x, varying y) or `:horizontal`
  (fixed y, varying x).
- `axis_value`: physical or index coordinate of the cut (default midplane).
- `reference`: optional analytical reference. Either a function `f(pos)` or
  a vector of same length as the profile.
- `ref_label`: legend label for the reference curve.
- `Lx`, `Ly`: physical extents (otherwise coordinates use index space).

# Returns
A `Makie.Figure` with one `Axis` and one or two line plots.

# Examples
```julia
ux = rand(64, 64)
fig = profile_plot(ux, :vertical;
                   reference=y -> 4y*(1 - y),
                   title="Poiseuille profile")
```
"""
function profile_plot(field::AbstractMatrix, line::Symbol;
                      axis_value=nothing,
                      reference=nothing,
                      ref_label::AbstractString="analytical",
                      title::AbstractString="",
                      xlabel::AbstractString="position",
                      ylabel::AbstractString="value",
                      label::AbstractString="simulation",
                      Lx=nothing, Ly=nothing)
    Nx, Ny = size(field)

    if line === :vertical
        # varying y at fixed x
        imid = axis_value === nothing ? cld(Nx, 2) : Int(round(axis_value))
        imid = clamp(imid, 1, Nx)
        prof = vec(field[imid, :])
        coords = Ly === nothing ? collect(1:Ny) : collect(range(0.0, Ly; length=Ny))
    elseif line === :horizontal
        jmid = axis_value === nothing ? cld(Ny, 2) : Int(round(axis_value))
        jmid = clamp(jmid, 1, Ny)
        prof = vec(field[:, jmid])
        coords = Lx === nothing ? collect(1:Nx) : collect(range(0.0, Lx; length=Nx))
    else
        error("profile_plot: line must be :vertical or :horizontal, got $line")
    end

    fig = Makie.Figure(; size=(800, 600))
    ax = Makie.Axis(fig[1, 1]; title=title, xlabel=xlabel, ylabel=ylabel)
    Makie.lines!(ax, coords, prof; color=:steelblue, linewidth=2, label=label)

    if reference !== nothing
        refvals = if reference isa Function
            [reference(c) for c in coords]
        else
            collect(reference)
        end
        Makie.lines!(ax, coords, refvals; color=:firebrick, linewidth=2,
                     linestyle=:dash, label=ref_label)
        Makie.axislegend(ax; position=:rt)
    end

    return fig
end
