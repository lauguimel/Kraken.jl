# KrakenView guide

`KrakenView` is the visualization companion to Kraken.jl. It has two
roles:

1. **Interactive viewing** of a `.krk` case — you see the domain,
   boundary conditions, mesh, and any STL overlays *before* the solver
   runs, then watch the fields evolve as the simulation steps.
2. **Batch figure generation** for the documentation, reports, and the
   benchmark pipeline — one declarative `spec` in, a folder of PNGs
   out.

It lives under `view/` in the repository as a separate sub-package.
This is deliberate: the main `Kraken` package has no Makie dependency,
so users who only want the solver get a light install.

## Install KrakenView

From the repository root:

```julia
using Pkg
Pkg.activate("view")
Pkg.instantiate()
using KrakenView
```

KrakenView pulls in Makie and a rendering backend. The documentation
build and CI smoke tests use **CairoMakie** (offscreen, headless). For
an interactive on-screen window, swap in **GLMakie** — this path is
supported experimentally in v0.1.0 and will become the default user
backend in v0.2.0.

## Interactive viewing of a `.krk` file

The simplest entry point is `view_krk`, which takes a `.krk` path and
returns a `KrakenScene` containing the Figure and its Observables:

```julia
using KrakenView
scene = view_krk("examples/cavity.krk")
display(scene.figure)
```

Before the run starts, the scene already shows:

- the domain bounding box,
- color-coded boundary-condition overlays (walls, inlets, outlets,
  moving lids),
- any refinement patches declared in the `.krk` file, and
- bounding-box outlines for STL silhouette regions.

To drive the actual simulation with live updates, use `run_view`, which
wires a callback into `Kraken.run_simulation` that pushes new field
snapshots into the Observable behind the heatmap. You can choose which
field to display via the `field` keyword: `:ux`, `:uy`, `:umag`, `:rho`,
or `:T` (thermal cases).

## Generating figures for documentation and reports

For non-interactive use — building the docs, regenerating a benchmark
figure, producing a batch of report images — KrakenView exposes
`generate_figures`. You pass a vector of specs (one per figure) and it
runs whatever simulations are needed, then saves every figure to disk.

```julia
using KrakenView

spec = [
    (case         = "cavity Re=100",
     figure_type  = :heatmap,
     output       = "cavity_umag.png",
     options      = (colormap=:viridis, title="|u|"),
     krk          = "examples/cavity.krk"),

    (case         = "taylor-green convergence",
     figure_type  = :convergence,
     output       = "taylor_green_order.png",
     options      = (theoretical_order=2.0,),
     data         = (; N_values=[32, 64, 128, 256],
                       errors=[1e-2, 2.5e-3, 6.25e-4, 1.56e-4])),
]

paths = generate_figures(spec; output_dir="docs/src/assets/figures")
```

Each spec entry is a `NamedTuple` with:

- `case` — human-readable name, used in titles.
- `figure_type` — one of `:heatmap`, `:profile`, `:convergence`,
  `:streamlines`.
- `output` — filename, relative to `output_dir`.
- `options` — kwargs forwarded to the figure builder.
- `krk` *or* `data` — either a `.krk` file to run, or precomputed data.

## Supported figure types

| `figure_type`    | What it draws                            | Data required            |
|------------------|------------------------------------------|--------------------------|
| `:heatmap`       | 2D scalar field                          | `(; field)`              |
| `:profile`       | Line profile (horizontal or vertical)    | `(; field, line)`        |
| `:convergence`   | Log–log error vs. resolution             | `(; N_values, errors)`   |
| `:streamlines`   | Streamlines over a velocity field        | `(; ux, uy)`             |

All figures are saved through `save_figure`, which picks the format
from the filename extension (PNG and SVG are supported).

## Headless mode

For CI, documentation builds, and batch figure generation, activate
CairoMakie before importing KrakenView so the figures render
offscreen:

```julia
using Pkg; Pkg.activate("view")
using CairoMakie          # headless backend
using KrakenView
```

The main documentation site (the one you are reading) is built this way.

## Limitations in v0.1.0

- **2D only** for the interactive scene: `view_krk` of a `D3Q19`
  setup raises a clear error. The 3D viewer is on the v0.2.0 roadmap.
- **GLMakie experimental**: the default interactive backend will
  become GLMakie in v0.2.0; for now, expect a few rough edges if you
  use it instead of CairoMakie.
- **STL regions** are rendered as bounding-box outlines rather than
  true silhouettes. Accurate STL overlays are also v0.2.0.
