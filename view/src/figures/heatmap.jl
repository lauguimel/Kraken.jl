# Standalone heatmap figure.

"""
    heatmap_field(field::AbstractMatrix; field_name="|u|", colormap=:viridis,
                  geometry=nothing, refinement=nothing, title="",
                  xlabel="x", ylabel="y", Lx=nothing, Ly=nothing)

Render a 2D scalar field as a heatmap.

# Arguments
- `field`: 2D array to render (Nx × Ny).
- `field_name`: label for the colorbar.
- `colormap`: Makie colormap name (default `:viridis`).
- `geometry`: optional solid mask (`AbstractMatrix{Bool}`) drawn as contour.
- `refinement`: optional vector of tuples `(x0, y0, x1, y1)` drawn as dotted
  rectangles.
- `title`: axis title.
- `Lx`, `Ly`: physical extents (default: 0:Nx, 0:Ny).

# Returns
A `Makie.Figure` with one `Axis` and one heatmap plot.

# Examples
```julia
fig = heatmap_field(rand(64, 64); field_name="rho", colormap=:thermal)
```
"""
function heatmap_field(field::AbstractMatrix;
                       field_name::AbstractString="|u|",
                       colormap=:viridis,
                       geometry=nothing,
                       refinement=nothing,
                       title::AbstractString="",
                       xlabel::AbstractString="x",
                       ylabel::AbstractString="y",
                       Lx=nothing, Ly=nothing)
    Nx, Ny = size(field)
    xs = Lx === nothing ? (1:Nx) : range(0.0, Lx; length=Nx)
    ys = Ly === nothing ? (1:Ny) : range(0.0, Ly; length=Ny)

    fig = Makie.Figure(; size=(800, 600))
    ax = Makie.Axis(fig[1, 1]; title=title, xlabel=xlabel, ylabel=ylabel,
                    aspect=Makie.DataAspect())
    hm = Makie.heatmap!(ax, xs, ys, field; colormap=colormap)
    Makie.Colorbar(fig[1, 2], hm; label=field_name)

    if geometry !== nothing
        try
            Makie.contour!(ax, xs, ys, Float64.(geometry);
                           levels=[0.5], color=:black, linewidth=1.5)
        catch
        end
    end

    if refinement !== nothing
        for r in refinement
            x0, y0, x1, y1 = r
            pts = Makie.Point2f[(x0, y0), (x1, y0), (x1, y1),
                                (x0, y1), (x0, y0)]
            Makie.lines!(ax, pts; color=:orange,
                         linewidth=1.5, linestyle=:dot)
        end
    end

    return fig
end
