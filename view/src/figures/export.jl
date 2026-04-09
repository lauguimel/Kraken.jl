# High-quality figure export.

"""
    save_figure(fig::Makie.Figure, path::String;
                size=(800, 600), dpi=150, format=:auto)

Save a `Makie.Figure` to disk. Format is inferred from the file extension
(`.png`, `.svg`, `.pdf`) unless `format` is set explicitly to `:png`,
`:svg`, or `:pdf`. PNG uses `dpi` to compute the pixel-per-unit factor
(default 150 dpi); SVG and PDF are vector formats and ignore `dpi`.

# Returns
The absolute path of the written file.

# Examples
```julia
fig = heatmap_field(rand(64, 64))
save_figure(fig, "out/heatmap.png")
```
"""
function save_figure(fig, path::AbstractString;
                     size::Tuple{<:Integer,<:Integer}=(800, 600),
                     dpi::Integer=150,
                     format::Symbol=:auto)
    fmt = format
    if fmt === :auto
        ext = lowercase(splitext(path)[2])
        fmt = ext == ".png" ? :png :
              ext == ".svg" ? :svg :
              ext == ".pdf" ? :pdf :
              :png
    end

    dir = dirname(path)
    isempty(dir) || isdir(dir) || mkpath(dir)

    try
        Makie.resize!(fig, size[1], size[2])
    catch
    end

    if fmt === :png
        ppu = dpi / 96
        Makie.save(path, fig; px_per_unit=ppu)
    else
        Makie.save(path, fig)
    end
    return abspath(path)
end
