#!/usr/bin/env julia --project=docs
# WP-3D-6 — Figure: sphere 3D Re=20 convergence (uniform vs stretched).
# Julia/CairoMakie version of plot_sphere_3d_convergence.py.
#
# Usage:
#   julia --project=docs scripts/figures/plot_sphere_3d_convergence.jl \
#       --log results/slbm_sphere_3d.log \
#       --out paper/figures/sphere_3d_convergence.pdf

using CairoMakie

const LINE_RE = r"^(?<label>\S.*?)\s+(?<Nx>\d+)×(?<Ny>\d+)×(?<Nz>\d+)\s+\(\s*(?<cells>\d+)\s*cells\)\s+Cd=(?<cd>[-\d.]+)\s+err=(?<err>[-\d.]+)%\s+MLUPS=(?<mlups>[-\d.]+)"

struct Run
    label::String
    cells::Int
    Cd::Float64
    err::Float64
    mlups::Float64
end

function parse_log(path::AbstractString)
    rows = Run[]
    for raw in eachline(path)
        m = match(LINE_RE, strip(raw))
        m === nothing && continue
        push!(rows, Run(strip(m["label"]),
                         parse(Int, m["cells"]),
                         parse(Float64, m["cd"]),
                         parse(Float64, m["err"]),
                         parse(Float64, m["mlups"])))
    end
    return rows
end

function split_runs(rows)
    uni  = filter(r -> startswith(lowercase(r.label), "uniform"),  rows)
    strd = filter(r -> startswith(lowercase(r.label), "stretch"), rows)
    sort!(uni;  by = r -> r.cells)
    sort!(strd; by = r -> r.cells)
    return uni, strd
end

function build_fig(uni::Vector{Run}, strd::Vector{Run}, out_path::AbstractString)
    fig = Figure(; size = (900, 360))

    ax_err = Axis(fig[1, 1];
        xlabel = "cells",
        ylabel = "|Cd − Cd_ref| / Cd_ref [%]",
        title  = "Sphere Re=20 — Cd accuracy",
        xscale = log10, yscale = log10,
        xgridstyle = :dash, ygridstyle = :dash)
    if !isempty(uni)
        scatterlines!(ax_err, [r.cells for r in uni], [r.err for r in uni];
                      color = :steelblue, marker = :circle,
                      markersize = 12, linewidth = 2,
                      label = "Uniform Cartesian")
    end
    if !isempty(strd)
        scatterlines!(ax_err, [r.cells for r in strd], [r.err for r in strd];
                      color = :darkorange, marker = :rect,
                      markersize = 12, linewidth = 2, linestyle = :dash,
                      label = "Stretched (preliminary)")
    end
    axislegend(ax_err; position = :rt)

    ax_mlups = Axis(fig[1, 2];
        xlabel = "cells",
        ylabel = "MLUPS (H100, Float64)",
        title  = "Throughput vs grid size",
        xscale = log10,
        xgridstyle = :dash, ygridstyle = :dash)
    if !isempty(uni)
        scatterlines!(ax_mlups, [r.cells for r in uni], [r.mlups for r in uni];
                      color = :steelblue, marker = :circle,
                      markersize = 12, linewidth = 2,
                      label = "Uniform Cartesian")
    end
    if !isempty(strd)
        scatterlines!(ax_mlups, [r.cells for r in strd], [r.mlups for r in strd];
                      color = :darkorange, marker = :rect,
                      markersize = 12, linewidth = 2, linestyle = :dash,
                      label = "Stretched (preliminary)")
    end
    axislegend(ax_mlups; position = :rb)

    mkpath(dirname(out_path))
    save(out_path, fig)
    @info "wrote" out_path
end

function main(args)
    log_path = "paper/data/sphere_3d_h100_20147745.log"
    out_path = "paper/figures/sphere_3d_convergence.pdf"
    i = 1
    while i ≤ length(args)
        if args[i] == "--log" && i + 1 ≤ length(args)
            log_path = args[i + 1]; i += 2
        elseif args[i] == "--out" && i + 1 ≤ length(args)
            out_path = args[i + 1]; i += 2
        else
            i += 1
        end
    end
    isfile(log_path) || error("log file not found: $log_path")
    rows = parse_log(log_path)
    isempty(rows) && error("no benchmark lines parsed; check log format")
    uni, strd = split_runs(rows)
    println("parsed $(length(rows)) runs ($(length(uni)) uniform, $(length(strd)) stretched)")
    for r in rows
        println("  ", rpad(r.label, 30), "  cells=", lpad(r.cells, 9),
                "  Cd=", round(r.Cd, digits = 3),
                "  err=", round(r.err, digits = 2), "%",
                "  MLUPS=", round(Int, r.mlups))
    end
    build_fig(uni, strd, out_path)
end

main(ARGS)
