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
