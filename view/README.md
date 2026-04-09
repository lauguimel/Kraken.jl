# KrakenView

Interactive 2D viewer for [Kraken.jl](../) `.krk` simulations, inspired by
GfsView (Gerris) and Basilisk view.

`KrakenView` is a separate Julia sub-project so the main `Kraken` package
stays free of heavy graphics dependencies. The user-facing default is
GLMakie; the test suite runs headless on CairoMakie.

## Install

```julia
using Pkg
Pkg.activate("view")
Pkg.develop(path=".")   # develop parent Kraken
Pkg.instantiate()
```

## Usage

```julia
using KrakenView
scene, result = run_view("examples/cavity.krk"; field=:umag, callback_every=200)
display(scene.figure)
```

## Scope

- v0.1.0 (this MVP): D2Q9 single-phase / thermal, static figures + live
  heatmap via a `run_simulation` callback. CairoMakie tested.
- v0.2.0 (planned): interactive GLMakie widgets (time slider, field menu,
  Run button), full STL projection, D3Q19 viewer, multiphase fields.

TODO: switch default backend to GLMakie once OpenGL availability is
confirmed in the target environments.

## Figures API

KrakenView also ships a set of static figure primitives used by the doc
build and benchmark pipelines. These live under `view/src/figures/` and
are re-exported from `KrakenView`:

| Function | Purpose |
|----------|---------|
| `heatmap_field(field; ...)` | Standalone 2D heatmap with optional geometry / refinement overlays. |
| `profile_plot(field, line; reference=..., ...)` | 1D line cut through a 2D field, with optional analytical overlay. |
| `convergence_plot(N_values, errors; theoretical_order=..., ...)` | Log-log error-vs-N with fitted observed order. |
| `streamline_plot(ux, uy; color_by_speed=..., ...)` | Streamlines (manual RK2 integration) with optional speed background. |
| `save_figure(fig, path; size, dpi, format)` | High-quality PNG/SVG/PDF export. |
| `generate_figures(spec; output_dir)` | Batch driver consuming a `NamedTuple` spec list. |

Every figure function returns a `Makie.Figure` that can be further edited
before saving. `generate_figures` is the single entry point used by the
doc build: the same tool users call interactively produces every figure
in the documentation.
