#!/usr/bin/env julia

using CairoMakie
using Printf

function _read_summary(path::AbstractString)
    lines = readlines(path)
    header = split(first(lines), ",")
    rows = NamedTuple[]
    for line in Iterators.drop(lines, 1)
        isempty(strip(line)) && continue
        values = split(line, ",")
        d = Dict(Symbol(header[i]) => values[i] for i in eachindex(header))
        push!(rows, (
            flow=Symbol(d[:flow]),
            method=Symbol(d[:method]),
            scale=parse(Int, d[:scale]),
            ux_abs_error=parse(Float64, d[:ux_abs_error]),
            Cd_rel_error=parse(Float64, d[:Cd_rel_error]),
            speedup_vs_leaf=parse(Float64, d[:speedup_vs_leaf]),
            cell_count_ratio_vs_leaf=parse(Float64, d[:cell_count_ratio_vs_leaf]),
        ))
    end
    return rows
end

function _series(rows, flow::Symbol, method::Symbol, field::Symbol)
    selected = sort(filter(r -> r.flow == flow && r.method == method, rows);
                    by=r -> r.scale)
    return [r.scale for r in selected], [getproperty(r, field) for r in selected]
end

function _lineplot!(ax, rows, flow::Symbol, method::Symbol, field::Symbol;
                    label::String, color, marker)
    x, y = _series(rows, flow, method, field)
    lines!(ax, x, y; label=label, color=color, linewidth=2)
    scatter!(ax, x, y; color=color, marker=marker, markersize=10)
    return ax
end

function main()
    input = get(ENV, "KRK_AMR_D_SUMMARY",
                joinpath(@__DIR__, "results",
                         "amr_d_publication_summary_2d_aqua_D_pub_20260505_long.csv"))
    outdir = get(ENV, "KRK_AMR_D_FIG_DIR",
                 joinpath(@__DIR__, "results", "figures"))
    mkpath(outdir)
    rows = _read_summary(input)

    fig = Figure(size=(1100, 760), fontsize=18)

    ax1 = Axis(fig[1, 1],
               title="Cylinder Cd error vs leaf oracle",
               xlabel="scale",
               ylabel="relative error",
               yscale=log10,
               xticks=[1, 2, 4])
    _lineplot!(ax1, rows, :cylinder, :cartesian_coarse, :Cd_rel_error;
               label="coarse Cartesian", color=:gray35, marker=:rect)
    _lineplot!(ax1, rows, :cylinder, :amr_route_native, :Cd_rel_error;
               label="AMR D", color=:dodgerblue3, marker=:circle)
    axislegend(ax1; position=:rt)

    ax2 = Axis(fig[1, 2],
               title="Mean streamwise velocity error",
               xlabel="scale",
               ylabel="absolute error",
               yscale=log10,
               xticks=[1, 2, 4])
    _lineplot!(ax2, rows, :square, :cartesian_coarse, :ux_abs_error;
               label="square coarse", color=:gray45, marker=:rect)
    _lineplot!(ax2, rows, :square, :amr_route_native, :ux_abs_error;
               label="square AMR D", color=:seagreen4, marker=:circle)
    _lineplot!(ax2, rows, :cylinder, :cartesian_coarse, :ux_abs_error;
               label="cylinder coarse", color=:gray70, marker=:utriangle)
    _lineplot!(ax2, rows, :cylinder, :amr_route_native, :ux_abs_error;
               label="cylinder AMR D", color=:dodgerblue3, marker=:diamond)
    axislegend(ax2; position=:lb)

    ax3 = Axis(fig[2, 1],
               title="Runtime speed relative to leaf oracle",
               xlabel="scale",
               ylabel="speedup",
               xticks=[1, 2, 4])
    hlines!(ax3, [1.0]; color=:black, linestyle=:dash)
    _lineplot!(ax3, rows, :square, :cartesian_coarse, :speedup_vs_leaf;
               label="square coarse", color=:gray45, marker=:rect)
    _lineplot!(ax3, rows, :square, :amr_route_native, :speedup_vs_leaf;
               label="square AMR D", color=:seagreen4, marker=:circle)
    _lineplot!(ax3, rows, :cylinder, :cartesian_coarse, :speedup_vs_leaf;
               label="cylinder coarse", color=:gray70, marker=:utriangle)
    _lineplot!(ax3, rows, :cylinder, :amr_route_native, :speedup_vs_leaf;
               label="cylinder AMR D", color=:dodgerblue3, marker=:diamond)
    axislegend(ax3; position=:rt)

    ax4 = Axis(fig[2, 2],
               title="Active cell count relative to leaf oracle",
               xlabel="scale",
               ylabel="cell-count ratio",
               xticks=[1, 2, 4],
               yticks=0:0.25:1.0)
    _lineplot!(ax4, rows, :cylinder, :cartesian_coarse, :cell_count_ratio_vs_leaf;
               label="coarse Cartesian", color=:gray35, marker=:rect)
    _lineplot!(ax4, rows, :cylinder, :amr_route_native, :cell_count_ratio_vs_leaf;
               label="AMR D", color=:dodgerblue3, marker=:circle)
    axislegend(ax4; position=:rc)

    png_path = joinpath(outdir, "amr_d_publication_2d_summary.png")
    pdf_path = joinpath(outdir, "amr_d_publication_2d_summary.pdf")
    save(png_path, fig)
    save(pdf_path, fig)
    println("wrote ", png_path)
    println("wrote ", pdf_path)
    return png_path, pdf_path
end

main()
