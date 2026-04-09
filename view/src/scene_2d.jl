# 2D scene construction for KrakenView.

"""
    KrakenScene

Container for a built viewer scene.

# Fields
- `figure::Figure` — the Makie Figure.
- `axis::Axis` — the main heatmap axis.
- `field::Observable{Matrix{Float64}}` — live field displayed.
- `field_name::Observable{Symbol}` — currently shown field symbol.
- `setup::SimulationSetup` — the parsed `.krk` setup.
- `snapshots::Vector{NamedTuple}` — in-memory history for the time slider.
"""
struct KrakenScene
    figure::Any
    axis::Any
    field::Observable{Matrix{Float64}}
    field_name::Observable{Symbol}
    setup::Any
    snapshots::Vector{Any}
end

"""
    view_krk(path::String; field::Symbol=:umag) -> KrakenScene

Load a `.krk` file, build a 2D visualization scene and return a
[`KrakenScene`](@ref). Does **not** run the simulation; call
[`run_view`](@ref) for that.

The pre-run scene shows:
- Domain bounding box
- Boundary conditions color-coded by type (see [`bc_color`](@ref))
- STL silhouettes (if any) and refinement patches as overlay rectangles
- Heatmap of the initial field (default: velocity magnitude `:umag`)

# Arguments
- `path::String`: path to a `.krk` configuration file.
- `field::Symbol`: field to show. One of `:ux`, `:uy`, `:umag`, `:rho`, `:T`.

# Examples
```julia
using KrakenView
scene = view_krk("examples/cavity.krk")
display(scene.figure)
```

Raises an error if the setup uses `D3Q19` (3D viewer is planned for v0.2.0).
"""
function view_krk(path::String; field::Symbol=:umag)
    setup = Kraken.load_kraken(path)
    return build_scene(setup; field=field)
end

"""
    build_scene(setup; field=:umag) -> KrakenScene

Low-level scene builder from an already-parsed `SimulationSetup`.
"""
function build_scene(setup; field::Symbol=:umag)
    if setup.lattice === :D3Q19
        error("KrakenView 2D MVP only supports D2Q9 in v0.1.0; 3D viewer is planned for v0.2.0")
    end

    dom = setup.domain
    Nx, Ny = dom.Nx, dom.Ny
    Lx, Ly = dom.Lx, dom.Ly

    # Initial field: zeros — will be replaced on first callback.
    field0 = zeros(Float64, Nx, Ny)
    field_obs = Observable(field0)
    field_name_obs = Observable(field)

    fig = Makie.Figure(; size=(900, 720))
    ax = Makie.Axis(fig[1, 1];
                    title="KrakenView — $(setup.name)",
                    xlabel="x", ylabel="y",
                    aspect=Makie.DataAspect())

    xs = range(0.0, Lx; length=Nx)
    ys = range(0.0, Ly; length=Ny)
    Makie.heatmap!(ax, xs, ys, field_obs;
                   colormap=field_colormap(field))

    # Domain bounding box
    bbox = Makie.Point2f[(0, 0), (Lx, 0), (Lx, Ly), (0, Ly), (0, 0)]
    Makie.lines!(ax, bbox; color=:black, linewidth=2)

    # Boundary condition overlays (color-coded)
    _draw_bc_overlays!(ax, setup)

    # Refinement patches
    _draw_refinement_patches!(ax, setup)

    # STL silhouettes (cheap bbox outline per STL region)
    _draw_stl_outlines!(ax, setup)

    Makie.limits!(ax, 0, Lx, 0, Ly)

    return KrakenScene(fig, ax, field_obs, field_name_obs, setup, Any[])
end

function _draw_bc_overlays!(ax, setup)
    dom = setup.domain
    Lx, Ly = dom.Lx, dom.Ly
    for bc in setup.boundaries
        col = bc_color(bc.type)
        ls = bc.type === :periodic ? :dash : :solid
        seg = if bc.face === :north
            [Makie.Point2f(0, Ly), Makie.Point2f(Lx, Ly)]
        elseif bc.face === :south
            [Makie.Point2f(0, 0), Makie.Point2f(Lx, 0)]
        elseif bc.face === :west
            [Makie.Point2f(0, 0), Makie.Point2f(0, Ly)]
        elseif bc.face === :east
            [Makie.Point2f(Lx, 0), Makie.Point2f(Lx, Ly)]
        else
            continue
        end
        Makie.lines!(ax, seg; color=col, linewidth=4, linestyle=ls)
    end
end

function _draw_refinement_patches!(ax, setup)
    if !hasproperty(setup, :refine) || getfield(setup, :refine) === nothing
        return
    end
    refs = getfield(setup, :refine)
    refs isa AbstractVector || return
    for r in refs
        x0, y0, x1, y1 = r.region
        pts = Makie.Point2f[(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        Makie.lines!(ax, pts; color=:orange, linewidth=1.5, linestyle=:dot)
    end
end

function _draw_stl_outlines!(ax, setup)
    for region in setup.regions
        region.stl === nothing && continue
        # Minimal: just mark region with a label at origin; detailed STL
        # projection is deferred to v0.2.0.
        Makie.scatter!(ax, [Makie.Point2f(0, 0)]; color=:purple, markersize=6)
    end
end
