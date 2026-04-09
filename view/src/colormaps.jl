# Colormaps and BC color mapping for KrakenView.

"""
    field_colormap(field::Symbol) -> Symbol

Return a suggested Makie colormap name for a given field symbol.
Defaults to `:viridis` for unknown fields.
"""
function field_colormap(field::Symbol)
    field === :ux && return :balance
    field === :uy && return :balance
    field === :umag && return :viridis
    field === :rho && return :thermal
    field === :T && return :inferno
    field === :C && return :blues
    return :viridis
end

"""
    bc_color(type::Symbol) -> Colorant

Color-code boundary condition types for overlay rendering.

| BC type      | Color       |
|--------------|-------------|
| `:wall`      | gray        |
| `:velocity`  | steel blue  |
| `:pressure`  | firebrick   |
| `:periodic`  | forest green (dashed in scene) |
"""
function bc_color(type::Symbol)
    type === :wall     && return colorant"gray50"
    type === :velocity && return colorant"steelblue"
    type === :pressure && return colorant"firebrick"
    type === :periodic && return colorant"forestgreen"
    return colorant"black"
end
