#!/usr/bin/env julia
#
# Serve Pluto notebooks interactively — works locally or over SSH.
#
# Usage:
#   julia --project=docs docs/serve_notebooks.jl              # local
#   julia --project=docs docs/serve_notebooks.jl --host 0.0.0.0  # SSH (expose)
#
# Then open http://localhost:1234 (or SSH-forward the port).
#
# For a static read-only server (no code execution on server):
#   julia --project=docs docs/serve_notebooks.jl --static
#

import Pkg
Pkg.instantiate()

using Pluto

# --- Parse CLI args ---
host   = "127.0.0.1"
port   = 1234
static = false

for (i, arg) in enumerate(ARGS)
    if arg == "--host" && i < length(ARGS)
        host = ARGS[i + 1]
    elseif arg == "--port" && i < length(ARGS)
        port = parse(Int, ARGS[i + 1])
    elseif arg == "--static"
        static = true
    end
end

notebook_dir = joinpath(@__DIR__, "src", "tutorials")

if static
    # PlutoSliderServer: read-only static export with interactivity
    using PlutoSliderServer
    PlutoSliderServer.run_directory(notebook_dir;
        host, port,
        Export_offer_binder = false,
    )
else
    # Full Pluto: editable, reactive notebooks
    Pluto.run(;
        host, port,
        launch_browser = (host == "127.0.0.1"),
        notebook = notebook_dir,
    )
end
