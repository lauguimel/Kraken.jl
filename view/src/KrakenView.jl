"""
    KrakenView

Interactive 2D viewer for Kraken.jl `.krk` simulations, inspired by
GfsView (Gerris) and Basilisk view.

# Design
- Separate sub-package: the main `Kraken` package does not depend on Makie,
  so core users keep a light install. Visualization is opt-in by activating
  the `view/` project.
- Default user-facing backend is GLMakie (interactive), but the test suite
  and CI use CairoMakie (offscreen) so smoke tests run headless.
- Scope for v0.1.0: D2Q9 single-phase / thermal only. 3D (D3Q19) is stubbed
  and raises a clear error; planned for v0.2.0.

# Public API
- [`view_krk`](@ref) — build a Figure from a `.krk` file.
- [`run_view`](@ref) — build the scene then run the simulation with a
  live-updating callback.
- [`KrakenScene`](@ref) — container for the Figure and its Observables.
"""
module KrakenView

using Kraken
using Makie
using Observables
using Colors
using GeometryBasics
using FileIO

export view_krk, run_view, KrakenScene

include("colormaps.jl")
include("scene_2d.jl")
include("run_hook.jl")

end # module
