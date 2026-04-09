# Streamline plot.

"""
    streamline_plot(ux::AbstractMatrix, uy::AbstractMatrix;
                    n_lines=20, colormap=:grays, color_by_speed=false,
                    background_field=nothing, title="",
                    Lx=nothing, Ly=nothing)

Streamline plot of a 2D velocity field. Tries Makie's `streamplot!` if
available; otherwise falls back to manually integrated streamlines seeded
on a uniform grid using simple RK2.

# Arguments
- `ux`, `uy`: 2D velocity components (Nx × Ny, same size).
- `n_lines`: approximate number of streamlines per direction (fallback).
- `colormap`: colormap used for a background scalar field.
- `color_by_speed`: if true, a speed heatmap is drawn as background.
- `background_field`: explicit scalar field to use as background (overrides
  `color_by_speed`).
- `Lx`, `Ly`: physical extents.

# Returns
A `Makie.Figure`.
"""
function streamline_plot(ux::AbstractMatrix, uy::AbstractMatrix;
                         n_lines::Int=20, colormap=:grays,
                         color_by_speed::Bool=false,
                         background_field=nothing,
                         title::AbstractString="",
                         Lx=nothing, Ly=nothing)
    size(ux) == size(uy) || error("ux and uy must have the same size")
    Nx, Ny = size(ux)
    xs = Lx === nothing ? collect(1.0:Nx) : collect(range(0.0, Lx; length=Nx))
    ys = Ly === nothing ? collect(1.0:Ny) : collect(range(0.0, Ly; length=Ny))

    fig = Makie.Figure(; size=(800, 600))
    ax = Makie.Axis(fig[1, 1]; title=title, xlabel="x", ylabel="y",
                    aspect=Makie.DataAspect())

    bg = background_field !== nothing ? background_field :
         (color_by_speed ? sqrt.(ux .^ 2 .+ uy .^ 2) : nothing)
    if bg !== nothing
        hm = Makie.heatmap!(ax, xs, ys, bg; colormap=colormap)
        Makie.Colorbar(fig[1, 2], hm)
    end

    # Manual streamline integration: RK2, seeded on uniform grid.
    _manual_streamlines!(ax, ux, uy, xs, ys; n_lines=n_lines)

    Makie.limits!(ax, first(xs), last(xs), first(ys), last(ys))
    return fig
end

function _bilinear(F, xs, ys, x, y)
    Nx = length(xs); Ny = length(ys)
    # find index
    if x <= xs[1] || x >= xs[end] || y <= ys[1] || y >= ys[end]
        return 0.0
    end
    dx = xs[2] - xs[1]
    dy = ys[2] - ys[1]
    fi = (x - xs[1]) / dx + 1
    fj = (y - ys[1]) / dy + 1
    i0 = clamp(floor(Int, fi), 1, Nx - 1)
    j0 = clamp(floor(Int, fj), 1, Ny - 1)
    tx = fi - i0
    ty = fj - j0
    return (1 - tx) * (1 - ty) * F[i0,   j0]   +
           tx       * (1 - ty) * F[i0+1, j0]   +
           (1 - tx) * ty       * F[i0,   j0+1] +
           tx       * ty       * F[i0+1, j0+1]
end

function _manual_streamlines!(ax, ux, uy, xs, ys; n_lines=20, n_steps=400)
    x0 = first(xs); x1 = last(xs)
    y0 = first(ys); y1 = last(ys)
    dx = (x1 - x0) / max(n_lines - 1, 1)
    dy = (y1 - y0) / max(n_lines - 1, 1)
    h = 0.5 * min(x1 - x0, y1 - y0) / n_steps
    # scale step by typical velocity
    umean = max(1e-12, sum(abs.(ux) .+ abs.(uy)) / length(ux))
    h = h / umean

    for i in 0:(n_lines - 1), j in 0:(n_lines - 1)
        sx = x0 + i * dx
        sy = y0 + j * dy
        xs_line = Float64[sx]
        ys_line = Float64[sy]
        x = sx; y = sy
        for _ in 1:n_steps
            u = _bilinear(ux, xs, ys, x, y)
            v = _bilinear(uy, xs, ys, x, y)
            s = sqrt(u*u + v*v)
            s < 1e-14 && break
            # RK2
            xm = x + 0.5 * h * u
            ym = y + 0.5 * h * v
            um = _bilinear(ux, xs, ys, xm, ym)
            vm = _bilinear(uy, xs, ys, xm, ym)
            x += h * um
            y += h * vm
            (x < x0 || x > x1 || y < y0 || y > y1) && break
            push!(xs_line, x)
            push!(ys_line, y)
        end
        if length(xs_line) > 2
            Makie.lines!(ax, xs_line, ys_line;
                         color=:black, linewidth=0.8)
        end
    end
end
