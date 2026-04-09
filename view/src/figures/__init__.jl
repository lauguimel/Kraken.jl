# Figure module entry point.
#
# This file is `include`d from KrakenView.jl and pulls in every figure
# primitive (heatmap, profile, convergence, streamlines, export). All
# symbols live in the `KrakenView` namespace.

include("heatmap.jl")
include("profile.jl")
include("convergence.jl")
include("streamlines.jl")
include("export.jl")
