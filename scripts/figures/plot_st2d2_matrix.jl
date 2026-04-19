#!/usr/bin/env julia --project=docs
# WP-MESH-5 figure: convergence of Cd, Cl_RMS, Strouhal on Schäfer-Turek
# 2D-2 (Re=100) for the 3 baselines × 3 resolutions, parsed from the
# Aqua log produced by hpc/wp_mesh_5_st2d2_aqua.jl.
#
# Expected log line format (one per run):
#   [A D=20 cells=36603] Cd=3.31 Cl_RMS=0.679 St=0.30 | 40 MLUPS 28.5s
#
# Usage:
#   julia --project=docs scripts/figures/plot_st2d2_matrix.jl \
#       --log paper/data/wp_mesh_5_st2d2_h100.log \
#       --out paper/figures/st2d2_convergence.pdf
using CairoMakie

const REF_CD  = 3.23
const REF_CLR = 0.706
const REF_ST  = 0.30

const LINE_RE = r"^\[(?<method>[ABC])\s+D=(?<D>\d+)\s+cells=(?<cells>\d+)\]\s+Cd=(?<cd>\S+).*?Cl_RMS=(?<clr>\S+).*?St=(?<st>\S+).*?\|\s+(?<mlups>\S+)\s+MLUPS"

struct Run
    method::Symbol      # :A, :B, :C
    D::Int
    cells::Int
    cd::Float64
    cl_rms::Float64
    St::Float64
    mlups::Float64
end

function parse_log(path::AbstractString)
    rows = Run[]
    for raw in eachline(path)
        m = match(LINE_RE, strip(raw))
        m === nothing && continue
        push!(rows, Run(Symbol(m["method"]),
                         parse(Int, m["D"]),
                         parse(Int, m["cells"]),
                         parse(Float64, m["cd"]),
                         parse(Float64, m["clr"]),
                         parse(Float64, m["st"]),
                         parse(Float64, m["mlups"])))
    end
    return rows
end

method_label(s) = s === :A ? "Cart + halfway-BB"  :
                   s === :B ? "Cart + LI-BB v2"   :
                              "gmsh + SLBM + LI-BB v2"
method_color(s) = s === :A ? :steelblue : s === :B ? :firebrick : :darkorange
method_marker(s) = s === :A ? :circle : s === :B ? :rect : :diamond

function build_fig(rows::Vector{Run}, out_path::AbstractString)
    fig = Figure(; size = (1200, 380))

    function panel!(ax, getter, ref, ylabel, title; min_err=0.01)
        for m in (:A, :B, :C)
            sel = sort([r for r in rows if r.method == m]; by = r -> r.cells)
            isempty(sel) && continue
            # Drop runs that diverged (NaN values) from this panel
            sel_clean = filter(r -> isfinite(getter(r)), sel)
            isempty(sel_clean) && continue
            cells = [r.cells for r in sel_clean]
            # Clamp to min_err so log10 is well-defined when the FFT
            # bin resolution returns 0% on St.
            err   = [max(min_err, 100 * abs(getter(r) - ref) / abs(ref)) for r in sel_clean]
            scatterlines!(ax, cells, err;
                color = method_color(m), marker = method_marker(m),
                markersize = 12, linewidth = 2,
                label = method_label(m))
        end
        ax.xscale = log10; ax.yscale = log10
        ax.xlabel = "cells"; ax.ylabel = ylabel
        ax.title = title
        ax.xgridstyle = :dash; ax.ygridstyle = :dash
    end

    ax_cd  = Axis(fig[1, 1])
    ax_clr = Axis(fig[1, 2])
    ax_st  = Axis(fig[1, 3])

    panel!(ax_cd,  r -> r.cd,     REF_CD,  "|Cd − 3.23| / 3.23 [%]",       "Drag")
    panel!(ax_clr, r -> r.cl_rms, REF_CLR, "|Cl_RMS − 0.706| / 0.706 [%]", "Lift RMS")
    panel!(ax_st,  r -> r.St,     REF_ST,  "|St − 0.30| / 0.30 [%]",        "Strouhal")
    Legend(fig[2, 1:3], ax_cd; orientation = :horizontal, framevisible = false)

    mkpath(dirname(out_path))
    save(out_path, fig)
    save(replace(out_path, ".pdf" => ".png"), fig; px_per_unit = 2)
    @info "wrote" pdf=out_path
end

function main(args)
    log_path = "paper/data/wp_mesh_5_st2d2_h100.log"
    out_path = "paper/figures/st2d2_convergence.pdf"
    i = 1
    while i ≤ length(args)
        if args[i] == "--log" && i + 1 ≤ length(args); log_path = args[i+1]; i += 2
        elseif args[i] == "--out" && i + 1 ≤ length(args); out_path = args[i+1]; i += 2
        else; i += 1; end
    end
    isfile(log_path) || error("log not found: $log_path")
    rows = parse_log(log_path)
    isempty(rows) && error("no benchmark lines parsed; check log format")
    println("parsed $(length(rows)) runs")
    for r in rows
        println("  ", r.method, "  D=", r.D,
                "  cells=", lpad(r.cells, 8),
                "  Cd=",     round(r.cd, digits=3),
                "  Cl_RMS=", round(r.cl_rms, digits=3),
                "  St=",     round(r.St, digits=3),
                "  MLUPS=",  round(Int, r.mlups))
    end
    build_fig(rows, out_path)
end

main(ARGS)
