#!/usr/bin/env julia
"""
    Kraken.jl — Cross-Solver Comparison

Reads JSON results from external solvers (Gerris, Basilisk, OpenFOAM) in
benchmarks/external/results/ and generates comparison figures overlaying
external solver data on Kraken benchmark plots.

Usage: julia --project=benchmarks benchmarks/compare_solvers.jl

Prerequisites: Run benchmarks/external/run_all.sh first to collect external results.
"""

using Kraken
using LinearAlgebra
using Printf
using CairoMakie
using KernelAbstractions
using JSON3  # for parsing external results

# Include Kraken benchmarks for re-running
include("heat_diffusion.jl")
include("lid_cavity.jl")
include("taylor_green.jl")

const RESULTS_DIR = joinpath(@__DIR__, "external", "results")
const FIGDIR = joinpath(@__DIR__, "..", "docs", "src", "assets", "figures")

# Solver colors and markers
const SOLVER_STYLE = Dict(
    "kraken"   => (color=:royalblue,  marker=:circle,   label="Kraken.jl"),
    "gerris"   => (color=:darkorange, marker=:diamond,   label="Gerris"),
    "basilisk" => (color=:forestgreen, marker=:utriangle, label="Basilisk"),
    "openfoam" => (color=:crimson,    marker=:rect,      label="OpenFOAM v10"),
)

"""Load JSON results for a solver, return nothing if file doesn't exist or is empty."""
function load_results(solver::String)
    path = joinpath(RESULTS_DIR, "$solver.json")
    if !isfile(path)
        @warn "No results found for $solver (expected $path)"
        return nothing
    end
    content = read(path, String)
    if isempty(strip(content))
        @warn "Empty results file for $solver ($path)"
        return nothing
    end
    return JSON3.read(content)
end

"""
    compare_cavity(; save_figures=true)

Generate comparison figure for lid-driven cavity Re=100:
overlays u(y) profiles from Kraken, Gerris, Basilisk, OpenFOAM, and Ghia reference.
Also generates a timing bar chart.
"""
function compare_cavity(; save_figures=true)
    println("=" ^ 60)
    println("Comparison: Lid-Driven Cavity Re=100")
    println("=" ^ 60)

    N = 64

    # --- Ghia reference data ---
    y_ghia = [1.0, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344,
              0.6172, 0.5, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703,
              0.0625, 0.0547, 0.0]
    u_ghia = [1.0, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332,
              -0.13641, -0.20581, -0.21090, -0.15662, -0.10150, -0.06434,
              -0.04775, -0.04192, -0.03717, 0.0]

    # --- Run Kraken (warmup + timed) ---
    println("  Warming up Kraken (small run to compile)...")
    run_cavity(N=8, Re=100.0, cfl=0.2, max_steps=10, tol=1e-1, verbose=false)

    println("  Running Kraken cavity (N=$N, Re=100)...")
    t_kraken = @elapsed begin
        u, v, p, converged = run_cavity(N=N, Re=100.0, cfl=0.2,
                                         max_steps=20000, tol=1e-7, verbose=false)
    end
    println("  Kraken converged: $converged ($(round(t_kraken, digits=2))s, excl. JIT)")

    # Extract Kraken u(y) at x=0.5
    dx = 1.0 / (N - 1)
    i_mid = round(Int, 0.5 / dx) + 1
    y_grid = collect(range(0.0, 1.0, length=N))
    u_kraken = Array(u[i_mid, :])

    # --- Load external results ---
    gerris_data = load_results("gerris")
    basilisk_data = load_results("basilisk")
    openfoam_data = load_results("openfoam")

    # --- Collect profiles and timings ---
    profiles = Dict{String, NamedTuple{(:y, :u), Tuple{Vector{Float64}, Vector{Float64}}}}()
    timings = Dict{String, Float64}()

    profiles["kraken"] = (y=y_grid, u=u_kraken)
    timings["kraken"] = t_kraken

    for (solver, data) in [("gerris", gerris_data), ("basilisk", basilisk_data), ("openfoam", openfoam_data)]
        if data === nothing || !haskey(data, :cases) || !haskey(data.cases, :cavity)
            continue
        end
        cav = data.cases.cavity
        timings[solver] = Float64(cav.time_seconds)
        yp = Float64.(cav.y_profile)
        up = Float64.(cav.u_profile)
        if !isempty(yp) && !isempty(up)
            # Add boundary points if missing (cell-centered solvers)
            if yp[1] > 0.001
                pushfirst!(yp, 0.0); pushfirst!(up, 0.0)  # no-slip bottom
            end
            if yp[end] < 0.999
                push!(yp, 1.0); push!(up, 1.0)  # lid velocity
            end
            profiles[solver] = (y=yp, u=up)
        end
        style = SOLVER_STYLE[solver]
        println("  $(style.label): $(round(timings[solver], digits=2))s" *
                (isempty(yp) ? " (no profile data)" : " ($(length(yp)) points)"))
    end

    # --- Print L2 errors vs Ghia ---
    println()
    println("  L2 errors vs Ghia et al. 1982:")
    for (solver, prof) in profiles
        u_interp = zeros(length(y_ghia))
        for (k, yg) in enumerate(y_ghia)
            if yg <= prof.y[1]
                u_interp[k] = prof.u[1]
            elseif yg >= prof.y[end]
                u_interp[k] = prof.u[end]
            else
                # Linear interpolation
                idx = searchsortedlast(prof.y, yg)
                idx = clamp(idx, 1, length(prof.y) - 1)
                w = (yg - prof.y[idx]) / (prof.y[idx+1] - prof.y[idx])
                u_interp[k] = (1 - w) * prof.u[idx] + w * prof.u[idx+1]
            end
        end
        l2 = norm(u_interp .- u_ghia) / norm(u_ghia)
        @printf("    %-12s: %.2f%%\n", solver, l2 * 100)
    end

    # --- Figures ---
    if save_figures
        mkpath(FIGDIR)

        # (a) Combined u(y) profile comparison
        fig1 = Figure(size=(900, 700))
        ax1 = Axis(fig1[1, 1], xlabel="u", ylabel="y",
                    title="Lid-Driven Cavity Re=100 — u(y) at x=0.5")

        # Plot Ghia reference first (background)
        scatter!(ax1, u_ghia, y_ghia, markersize=10, color=:black,
                 marker=:star5, label="Ghia et al. 1982")

        # Plot each solver
        for solver in ["kraken", "gerris", "basilisk", "openfoam"]
            if haskey(profiles, solver)
                style = SOLVER_STYLE[solver]
                prof = profiles[solver]
                lines!(ax1, prof.u, prof.y, linewidth=2, color=style.color,
                       label=style.label)
            end
        end

        axislegend(ax1, position=:lb)
        save(joinpath(FIGDIR, "comparison_cavity_profile.png"), fig1)
        println("  Saved: comparison_cavity_profile.png")

        # (b) Timing bar chart
        solver_order = ["kraken", "gerris", "basilisk", "openfoam"]
        available = [s for s in solver_order if haskey(timings, s)]
        if length(available) >= 2
            fig2 = Figure(size=(800, 500))
            ax2 = Axis(fig2[1, 1],
                        xlabel="Solver", ylabel="Wall time (s)",
                        title="Lid-Driven Cavity Re=100, N=64 — Execution Time",
                        xticks=(1:length(available),
                                [SOLVER_STYLE[s].label for s in available]))
            colors = [SOLVER_STYLE[s].color for s in available]
            times = [timings[s] for s in available]
            barplot!(ax2, 1:length(available), times, color=colors,
                     bar_labels=:y, label_formatter=x -> @sprintf("%.1fs", x))
            save(joinpath(FIGDIR, "comparison_cavity_timing.png"), fig2)
            println("  Saved: comparison_cavity_timing.png")
        end

        # (c) Update the original lid_cavity_u_profile.png with external data
        fig3 = Figure(size=(800, 600))
        ax3 = Axis(fig3[1, 1], xlabel="u", ylabel="y",
                    title="Lid-Driven Cavity Re=100 — u(y) at x=0.5")
        lines!(ax3, u_kraken, y_grid, linewidth=2, color=:royalblue,
               label="Kraken (N=$N)")
        scatter!(ax3, u_ghia, y_ghia, markersize=8, color=:black,
                 marker=:star5, label="Ghia et al. 1982")

        for solver in ["gerris", "basilisk", "openfoam"]
            if haskey(profiles, solver)
                style = SOLVER_STYLE[solver]
                prof = profiles[solver]
                lines!(ax3, prof.u, prof.y, linewidth=1.5, color=style.color,
                       linestyle=:dash, label=style.label)
            end
        end
        axislegend(ax3, position=:lb)
        save(joinpath(FIGDIR, "lid_cavity_u_profile.png"), fig3)
        println("  Updated: lid_cavity_u_profile.png (with external solvers)")
    end

    return Dict(
        "profiles" => profiles,
        "timings" => timings,
    )
end

"""
    compare_taylor_green(; save_figures=true)

Compare Taylor-Green vortex results between Kraken and Basilisk.
"""
function compare_taylor_green(; save_figures=true)
    println()
    println("=" ^ 60)
    println("Comparison: Taylor-Green Vortex 2D")
    println("=" ^ 60)

    basilisk_data = load_results("basilisk")

    if basilisk_data === nothing || !haskey(basilisk_data, :cases) || !haskey(basilisk_data.cases, :taylor_green)
        println("  No Basilisk Taylor-Green data available. Skipping comparison.")
        return nothing
    end

    tg = basilisk_data.cases.taylor_green
    println("  Basilisk Taylor-Green: L2 error = $(round(Float64(tg.l2_error), sigdigits=3))")

    # Run Kraken Taylor-Green for comparison
    ν = 0.01
    t_final = 1.0
    N = 64
    Nt = N + 2
    L = 2π
    dx_val = L / N

    function apply_periodic!(f, Ntot)
        f[1, :] .= @view f[Ntot-1, :]
        f[Ntot, :] .= @view f[2, :]
        f[:, 1] .= @view f[:, Ntot-1]
        f[:, Ntot] .= @view f[:, 2]
    end

    x = [(i - 1.5) * dx_val for i in 1:Nt]
    y = [(j - 1.5) * dx_val for j in 1:Nt]
    u_tg = [cos(x[i]) * sin(y[j]) for i in 1:Nt, j in 1:Nt]
    v_tg = [-sin(x[i]) * cos(y[j]) for i in 1:Nt, j in 1:Nt]
    p_tg = zeros(Nt, Nt)
    au = zeros(Nt, Nt); av = zeros(Nt, Nt)
    lu = zeros(Nt, Nt); lv = zeros(Nt, Nt)
    df = zeros(Nt, Nt); gx = zeros(Nt, Nt); gy = zeros(Nt, Nt)
    phi = zeros(N, N)

    dt = min(0.2 * dx_val, 0.2 * dx_val^2 / ν)
    nsteps = ceil(Int, t_final / dt)
    dt = t_final / nsteps
    apply_periodic!(u_tg, Nt); apply_periodic!(v_tg, Nt)

    record_every = max(1, nsteps ÷ 50)
    times_rec = Float64[]
    umax_rec = Float64[]

    t_kraken = @elapsed begin
        for step in 1:nsteps
            fill!(au, 0.0); fill!(av, 0.0)
            advect!(au, u_tg, v_tg, u_tg, dx_val)
            advect!(av, u_tg, v_tg, v_tg, dx_val)
            fill!(lu, 0.0); fill!(lv, 0.0)
            laplacian!(lu, u_tg, dx_val)
            laplacian!(lv, v_tg, dx_val)
            for j in 2:Nt-1, i in 2:Nt-1
                u_tg[i, j] += dt * (-au[i, j] + ν * lu[i, j])
                v_tg[i, j] += dt * (-av[i, j] + ν * lv[i, j])
            end
            apply_periodic!(u_tg, Nt); apply_periodic!(v_tg, Nt)
            fill!(df, 0.0); divergence!(df, u_tg, v_tg, dx_val)
            rhs = df[2:Nt-1, 2:Nt-1] ./ dt
            fill!(phi, 0.0); solve_poisson_fft!(phi, rhs, dx_val)
            p_tg[2:Nt-1, 2:Nt-1] .= phi; apply_periodic!(p_tg, Nt)
            fill!(gx, 0.0); fill!(gy, 0.0); gradient!(gx, gy, p_tg, dx_val)
            for j in 2:Nt-1, i in 2:Nt-1
                u_tg[i, j] -= dt * gx[i, j]; v_tg[i, j] -= dt * gy[i, j]
            end
            apply_periodic!(u_tg, Nt); apply_periodic!(v_tg, Nt)
            if step % record_every == 0
                push!(times_rec, step * dt)
                push!(umax_rec, maximum(abs.(u_tg[2:Nt-1, 2:Nt-1])))
            end
        end
    end

    # Kraken L2 error
    decay = exp(-2ν * t_final)
    u_exact = [cos(x[i]) * sin(y[j]) * decay for i in 1:Nt, j in 1:Nt]
    v_exact = [-sin(x[i]) * cos(y[j]) * decay for i in 1:Nt, j in 1:Nt]
    u_int = u_tg[2:Nt-1, 2:Nt-1]; ue_int = u_exact[2:Nt-1, 2:Nt-1]
    v_int = v_tg[2:Nt-1, 2:Nt-1]; ve_int = v_exact[2:Nt-1, 2:Nt-1]
    kraken_err = sqrt(norm(u_int .- ue_int)^2 + norm(v_int .- ve_int)^2) /
                 sqrt(norm(ue_int)^2 + norm(ve_int)^2)

    @printf("  Kraken Taylor-Green: L2 error = %.3e (%.2fs)\n", kraken_err, t_kraken)

    if save_figures
        mkpath(FIGDIR)

        # Update Taylor-Green decay plot with Basilisk data point
        fig = Figure(size=(800, 600))
        ax = Axis(fig[1, 1], xlabel="t", ylabel="max |u|",
                   title="Taylor-Green — Velocity decay (Kraken vs Basilisk)")
        lines!(ax, times_rec, umax_rec, linewidth=2, color=:royalblue,
               label="Kraken (N=64)")
        t_exact = range(0, t_final, length=100)
        lines!(ax, collect(t_exact), exp.(-2ν .* t_exact), linestyle=:dash,
               color=:black, linewidth=2, label="Exact: exp(-2νt)")

        # Add Basilisk final point
        basilisk_umax = haskey(tg, :umax) ? Float64(tg.umax) : NaN
        if !isnan(basilisk_umax)
            scatter!(ax, [t_final], [basilisk_umax], markersize=12,
                     color=:forestgreen, marker=:utriangle, label="Basilisk (N=64)")
        end

        axislegend(ax)
        save(joinpath(FIGDIR, "taylor_green_decay.png"), fig)
        println("  Updated: taylor_green_decay.png")

        # Comparison bar: L2 errors
        basilisk_err = Float64(tg.l2_error)
        fig2 = Figure(size=(600, 400))
        ax2 = Axis(fig2[1, 1],
                    xlabel="Solver", ylabel="L2 relative error",
                    title="Taylor-Green Vortex — L2 Error at t=1.0",
                    xticks=(1:2, ["Kraken.jl", "Basilisk"]),
                    yscale=log10)
        barplot!(ax2, [1, 2], [kraken_err, basilisk_err],
                 color=[:royalblue, :forestgreen],
                 bar_labels=:y, label_formatter=x -> @sprintf("%.1e", x))
        save(joinpath(FIGDIR, "comparison_taylor_green_error.png"), fig2)
        println("  Saved: comparison_taylor_green_error.png")
    end

    return Dict(
        "kraken_error" => kraken_err,
        "basilisk_error" => Float64(tg.l2_error),
        "kraken_time" => t_kraken,
    )
end

function main()
    println("╔" * "═"^58 * "╗")
    println("║" * lpad("Cross-Solver Comparison", 40) * " "^18 * "║")
    println("╚" * "═"^58 * "╝")
    println()

    # Check for available results
    available = String[]
    for solver in ["gerris", "basilisk", "openfoam"]
        path = joinpath(RESULTS_DIR, "$solver.json")
        if isfile(path)
            push!(available, solver)
            println("  Found: $solver.json")
        end
    end

    if isempty(available)
        println()
        println("  No external results found in $RESULTS_DIR/")
        println("  Run benchmarks/external/run_all.sh first.")
        println()
        return
    end

    println()

    # --- Cavity comparison ---
    cavity_results = compare_cavity()

    # --- Taylor-Green comparison ---
    tg_results = compare_taylor_green()

    # --- Summary ---
    println()
    println("╔" * "═"^58 * "╗")
    println("║" * lpad("COMPARISON SUMMARY", 38) * " "^20 * "║")
    println("╠" * "═"^58 * "╣")

    if cavity_results !== nothing
        println("║ Cavity Re=100 — Wall times:                             ║")
        for (solver, t) in sort(collect(cavity_results["timings"]), by=x->x[2])
            style = SOLVER_STYLE[solver]
            @printf("║   %-12s: %8.2fs                                  ║\n", style.label, t)
        end
    end

    println("╠" * "═"^58 * "╣")

    if tg_results !== nothing
        println("║ Taylor-Green — L2 errors:                                ║")
        @printf("║   Kraken.jl:   %.3e                                 ║\n", tg_results["kraken_error"])
        @printf("║   Basilisk:    %.3e                                 ║\n", tg_results["basilisk_error"])
    end

    println("╚" * "═"^58 * "╝")

    # List generated figures
    comparison_figs = filter(f -> startswith(f, "comparison_"), readdir(FIGDIR))
    println("\nNew comparison figures:")
    for f in comparison_figs
        println("  $f")
    end
    println("\nUpdated existing figures:")
    println("  lid_cavity_u_profile.png")
    println("  taylor_green_decay.png")
    println("\nDone.")
end

main()
