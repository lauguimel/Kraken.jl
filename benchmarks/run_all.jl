#!/usr/bin/env julia
"""
    Kraken.jl — Benchmark Suite

Runs all 7 benchmarks sequentially, prints timing summary and generates figures.
Usage: julia --project=benchmarks benchmarks/run_all.jl
"""

using Kraken
using LinearAlgebra
using Printf
using CairoMakie
using KernelAbstractions

# Include all benchmark scripts
include("heat_diffusion.jl")
include("lid_cavity.jl")
include("taylor_green.jl")
include("poiseuille.jl")
include("couette.jl")
include("rotation_advection.jl")
include("advection_diffusion.jl")
include("rayleigh_benard.jl")

function main()
    println("╔" * "═"^58 * "╗")
    println("║" * lpad("Kraken.jl — Benchmark Suite", 43) * " "^15 * "║")
    println("╚" * "═"^58 * "╝")
    println()

    figdir = joinpath(@__DIR__, "..", "docs", "src", "assets", "figures")
    mkpath(figdir)

    results = Dict{String, Any}[]

    benchmarks = [
        ("Heat Diffusion",       () -> run_heat_diffusion(figdir=figdir)),
        ("Lid-Driven Cavity",    () -> run_lid_cavity(figdir=figdir)),
        ("Taylor-Green Vortex",  () -> run_taylor_green(figdir=figdir)),
        ("Poiseuille Flow",      () -> run_poiseuille(figdir=figdir)),
        ("Couette Flow",         () -> run_couette(figdir=figdir)),
        ("Rotation Advection",   () -> run_rotation_advection(figdir=figdir)),
        ("Advection-Diffusion",  () -> run_advection_diffusion(figdir=figdir)),
        ("Rayleigh-Bénard",     () -> run_rayleigh_benard_benchmark(figdir=figdir)),
    ]

    total_time = @elapsed begin
        for (name, bench_fn) in benchmarks
            println()
            try
                r = bench_fn()
                push!(results, r)
            catch e
                println("  ERROR in $name: $e")
                push!(results, Dict("name" => name, "t_cpu" => NaN, "t_metal" => NaN, "has_metal" => false))
            end
        end
    end

    # --- Summary table ---
    println()
    println("╔" * "═"^58 * "╗")
    println("║" * lpad("TIMING SUMMARY", 36) * " "^22 * "║")
    println("╠" * "═"^58 * "╣")
    @printf("║ %-24s │ %8s │ %8s │ %7s ║\n", "Benchmark", "CPU (s)", "Metal(s)", "Speedup")
    println("╟" * "─"^25 * "┼" * "─"^10 * "┼" * "─"^10 * "┼" * "─"^9 * "╢")

    for r in results
        name = get(r, "name", "?")
        tc = get(r, "t_cpu", NaN)
        tm = get(r, "t_metal", NaN)
        hm = get(r, "has_metal", false)
        tc_str = isnan(tc) ? "ERR" : @sprintf("%.3f", tc)
        tm_str = (hm && !isnan(tm)) ? @sprintf("%.3f", tm) : "N/A"
        sp_str = (hm && !isnan(tm) && !isnan(tc) && tm > 0) ? @sprintf("%.1fx", tc / tm) : "N/A"
        @printf("║ %-24s │ %8s │ %8s │ %7s ║\n", name, tc_str, tm_str, sp_str)
    end

    println("╠" * "═"^58 * "╣")
    @printf("║ Total wall time: %6.1fs%33s║\n", total_time, "")
    println("╚" * "═"^58 * "╝")

    # Count generated figures
    fig_count = 0
    if isdir(figdir)
        fig_count = count(endswith(".png"), readdir(figdir))
    end
    println("\nFigures generated: $fig_count (in $figdir)")
    println("Done.")
end

main()
