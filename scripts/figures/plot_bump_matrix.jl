#!/usr/bin/env julia --project=docs
# WP-MESH-6 figure: convergence of Cd, Cl_RMS, Strouhal on cylinder
# cross-flow Re=100, three baselines × three resolutions, parsed from
# the Aqua log of hpc/wp_mesh_6_bump_aqua.jl.
#
# Expected log line format:
#   [A D=20 cells=13041] Cd=1.40 (err 0.0%) Cl_RMS=0.33 (err 0.0%) ...| 80 MLUPS  10.0s
#
# Usage:
#   julia --project=docs scripts/figures/plot_bump_matrix.jl \
#       --log paper/data/wp_mesh_6_bump_h100.log \
#       --out paper/figures/bump_convergence.pdf
using CairoMakie

const REF_CD  = 1.4       # Williamson 1996, Park 1998 (Re=100 free-stream)
const REF_CLR = 0.33      # Cl_RMS = Cl_max / sqrt(2) ≈ 0.466 / sqrt(2)
const REF_ST  = 0.165     # Henderson 1995 / Williamson 1996

const LINE_RE = r"^\[(?<method>[ABC])\s+D=(?<D>\d+)\s+cells=(?<cells>\d+)\]\s+Cd=(?<cd>\S+).*?Cl_RMS=(?<clr>\S+).*?St=(?<st>\S+).*?\|\s+(?<mlups>\S+)\s+MLUPS"

struct Run
    method::Symbol; D::Int; cells::Int
    cd::Float64; cl_rms::Float64; St::Float64; mlups::Float64
end

parse_log(p) = begin
    rows = Run[]
    for raw in eachline(p)
        m = match(LINE_RE, strip(raw)); m === nothing && continue
        push!(rows, Run(Symbol(m["method"]), parse(Int, m["D"]), parse(Int, m["cells"]),
                         parse(Float64, m["cd"]), parse(Float64, m["clr"]),
                         parse(Float64, m["st"]), parse(Float64, m["mlups"])))
    end
    rows
end

method_label(s) = s===:A ? "Cart + halfway-BB" : s===:B ? "Cart + LI-BB v2" : "gmsh Bump + SLBM + LI-BB v2"
method_color(s) = s===:A ? :steelblue : s===:B ? :firebrick : :darkorange
method_marker(s) = s===:A ? :circle : s===:B ? :rect : :diamond

function build_fig(rows::Vector{Run}, out_path::AbstractString)
    fig = Figure(; size = (1200, 380))
    function panel!(ax, getter, ref, ylabel, title; min_err=0.01)
        for m in (:A, :B, :C)
            sel = sort([r for r in rows if r.method == m]; by = r -> r.cells)
            sel = filter(r -> isfinite(getter(r)), sel); isempty(sel) && continue
            cells = [r.cells for r in sel]
            err = [max(min_err, 100 * abs(getter(r) - ref) / abs(ref)) for r in sel]
            scatterlines!(ax, cells, err;
                color = method_color(m), marker = method_marker(m),
                markersize = 12, linewidth = 2, label = method_label(m))
        end
        ax.xscale = log10; ax.yscale = log10
        ax.xlabel = "cells"; ax.ylabel = ylabel
        ax.title = title; ax.xgridstyle = :dash; ax.ygridstyle = :dash
    end
    ax_cd  = Axis(fig[1, 1])
    ax_clr = Axis(fig[1, 2])
    ax_st  = Axis(fig[1, 3])
    panel!(ax_cd,  r -> r.cd,     REF_CD,  "|Cd − $(REF_CD)| / $(REF_CD) [%]",  "Drag")
    panel!(ax_clr, r -> r.cl_rms, REF_CLR, "|Cl_RMS − $(REF_CLR)| / $(REF_CLR) [%]", "Lift RMS")
    panel!(ax_st,  r -> r.St,     REF_ST,  "|St − $(REF_ST)| / $(REF_ST) [%]",  "Strouhal")
    Legend(fig[2, 1:3], ax_cd; orientation = :horizontal, framevisible = false)
    mkpath(dirname(out_path)); save(out_path, fig); save(replace(out_path, ".pdf"=>".png"), fig; px_per_unit=2)
    @info "wrote" pdf=out_path
end

function main(args)
    log_path = "paper/data/wp_mesh_6_bump_h100.log"
    out_path = "paper/figures/bump_convergence.pdf"
    i = 1
    while i ≤ length(args)
        if args[i] == "--log" && i+1 ≤ length(args); log_path = args[i+1]; i += 2
        elseif args[i] == "--out" && i+1 ≤ length(args); out_path = args[i+1]; i += 2
        else; i += 1; end
    end
    isfile(log_path) || error("log not found: $log_path")
    rows = parse_log(log_path); isempty(rows) && error("no benchmark lines parsed")
    println("parsed $(length(rows)) runs")
    for r in rows
        println("  ", r.method, "  D=", r.D, "  cells=", lpad(r.cells, 8),
                "  Cd=", round(r.cd, digits=3),
                "  Cl_RMS=", round(r.cl_rms, digits=3),
                "  St=", round(r.St, digits=3),
                "  MLUPS=", round(Int, r.mlups))
    end
    build_fig(rows, out_path)
end
main(ARGS)
