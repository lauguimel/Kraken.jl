include(joinpath(@__DIR__, "paper_convergent_singleblock.jl"))

function parse_specs(raw::AbstractString)
    specs = Tuple{Int,Int}[]
    for part in split(raw, ",")
        s = strip(part)
        isempty(s) && continue
        m = match(r"^(\d+)x(\d+)$", s)
        m === nothing && error("bad KRK_CONV_SWEEP_SPECS entry '$s'; expected Nx x Ny")
        push!(specs, (parse(Int, m.captures[1]), parse(Int, m.captures[2])))
    end
    isempty(specs) && error("empty mesh sweep spec list")
    return specs
end

function csv_cell(x)
    if x isa AbstractString
        y = replace(x, "\"" => "\"\"")
        return occursin(r"[,\n\"]", y) ? "\"" * y * "\"" : y
    elseif x === nothing
        return ""
    elseif x isa Real
        return isfinite(x) ? string(x) : "NaN"
    end
    return string(x)
end

function write_rows_csv(path, rows, columns)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(string.(columns), ","))
        for row in rows
            vals = [haskey(row, c) ? row[c] : "" for c in columns]
            println(io, join(csv_cell.(vals), ","))
        end
    end
    return path
end

function case_paths(case_root, Nx, Ny)
    case = "n$(Nx)x$(Ny)"
    base = joinpath(case_root, case)
    msh = joinpath(base, "meshes", "single_block_convergent_$(case).msh")
    krk = joinpath(base, "krk", "single_block_convergent_$(case).krk")
    return (; case, base, msh, krk)
end

function final_metrics(mbm, states, run)
    block = mbm.blocks[1]
    state = states[1]
    flows = edge_flow_rates(block, state)
    wall_bot = edge_normal_velocity(block, state, :south)
    wall_top = edge_normal_velocity(block, state, :north)
    rho_min, rho_max = physical_density_bounds(states)
    data = physical_speeds(mbm, states)
    return (;
        rho_min=Float64(rho_min),
        rho_max=Float64(rho_max),
        rho_span=Float64(rho_max - rho_min),
        Q_in=Float64(flows.Q_in),
        Q_out=Float64(flows.Q_out),
        Q_rel_err=Float64(abs(flows.Q_out - flows.Q_in) /
                          max(abs(flows.Q_in), eps(Float64))),
        max_wall_un=Float64(max(maximum(wall_bot), maximum(wall_top))),
        max_speed=Float64(maximum(data.speed)),
        elapsed_s=Float64(run.elapsed_s),
    )
end

function run_sweep_case(paths; Nx, Ny, steps, ng, ν, u_max,
                        L, H_in, H_out, departure, plot_dir,
                        rho_lo, rho_hi)
    write_convergent_msh(paths.msh; L=L, H_in=H_in, H_out=H_out, Nx=Nx, Ny=Ny)
    write_convergent_krk(paths.krk, paths.msh; L=L, H_in=H_in, H_out=H_out,
                         Nx=Nx, Ny=Ny, u_max=u_max, Re=20.0, steps=steps)

    mbm_raw, _ = load_gmsh_multiblock_2d(paths.msh; FT=FT, layout=:topological)
    mbm = orient_single_convergent(mbm_raw)
    issues = sanity_check_multiblock(mbm; verbose=false)
    n_errors = count(issue -> issue.severity === :error, issues)
    n_errors == 0 || error("mesh sanity has $n_errors error(s)")

    run = run_convergent(mbm; steps=steps, ng=ng, ν=ν, L=L, H_in=H_in,
                         H_out=H_out, u_max=u_max, departure=departure)
    metrics = final_metrics(mbm, run.states, run)
    stable = isfinite(metrics.rho_min) && isfinite(metrics.rho_max) &&
             metrics.rho_min >= rho_lo && metrics.rho_max <= rho_hi

    png = joinpath(plot_dir, "bodyfit_convergent_singleblock_$(paths.case).png")
    plot_convergent(png, mbm, run.states, run;
                    L=L, H_in=H_in, H_out=H_out, u_max=u_max, ν=ν)

    row = merge((;
        case=paths.case, Nx=Nx, Ny=Ny, nodes=Nx * Ny, steps=steps,
        ng=ng, nu=Float64(ν), u_max=Float64(u_max),
        L=Float64(L), H_in=Float64(H_in), H_out=Float64(H_out),
        dx_ref=Float64(run.dx_ref), status="ok", stable=stable,
        msh=relpath(paths.msh, pwd()), krk=relpath(paths.krk, pwd()),
        png=relpath(png, pwd()), error="",
    ), metrics)

    history = [merge((; case=paths.case, Nx=Nx, Ny=Ny), h) for h in run.history]
    return row, history
end

function failed_row(paths; Nx, Ny, steps, ng, ν, u_max, L, H_in, H_out, err)
    return (;
        case=paths.case, Nx=Nx, Ny=Ny, nodes=Nx * Ny, steps=steps,
        ng=ng, nu=Float64(ν), u_max=Float64(u_max),
        L=Float64(L), H_in=Float64(H_in), H_out=Float64(H_out),
        dx_ref=NaN, status="failed", stable=false,
        rho_min=NaN, rho_max=NaN, rho_span=NaN,
        Q_in=NaN, Q_out=NaN, Q_rel_err=NaN, Q_out_rel_to_finest=NaN,
        max_wall_un=NaN, max_speed=NaN, elapsed_s=NaN,
        msh=relpath(paths.msh, pwd()), krk=relpath(paths.krk, pwd()),
        png="", error=sprint(showerror, err))
end

function add_finest_reference(rows)
    ok = filter(row -> row.status == "ok", rows)
    isempty(ok) && return rows
    finest = ok[argmin([row.dx_ref for row in ok])]
    q_ref = finest.Q_out
    out = NamedTuple[]
    for row in rows
        qerr = row.status == "ok" ?
            abs(row.Q_out - q_ref) / max(abs(q_ref), eps(Float64)) : NaN
        push!(out, merge(row, (; Q_out_rel_to_finest=Float64(qerr))))
    end
    return out
end

function write_sweep_plot(path, rows, history)
    HAS_CAIRO || return nothing
    ok = filter(row -> row.status == "ok", rows)
    isempty(ok) && return nothing
    mkpath(dirname(path))
    fig = Figure(size=(1300, 920))
    ax_q = Axis(fig[1, 1], title="grid convergence vs finest",
                xlabel="dx_ref", ylabel="rel |Qout - Qout(finest)|",
                xscale=log10, yscale=log10)
    ax_mass = Axis(fig[1, 2], title="mass balance",
                   xlabel="dx_ref", ylabel="|Qout-Qin|/|Qin|",
                   xscale=log10, yscale=log10)
    ax_rho = Axis(fig[2, 1], title="density bounds over time",
                  xlabel="step", ylabel="rho")
    ax_wall = Axis(fig[2, 2], title="wall leakage",
                   xlabel="dx_ref", ylabel="max |u.n|",
                   xscale=log10, yscale=log10)

    dx = [row.dx_ref for row in ok]
    qerr = [row.Q_out_rel_to_finest for row in ok]
    merr = [row.Q_rel_err for row in ok]
    wun = [row.max_wall_un for row in ok]
    order = sortperm(dx; rev=true)
    lines!(ax_q, dx[order], max.(qerr[order], 1e-12); color=:black, linewidth=2)
    scatter!(ax_q, dx, max.(qerr, 1e-12); color=:black, markersize=10)
    lines!(ax_mass, dx[order], max.(merr[order], 1e-12); color=:firebrick, linewidth=2)
    scatter!(ax_mass, dx, max.(merr, 1e-12); color=:firebrick, markersize=10)
    lines!(ax_wall, dx[order], max.(wun[order], 1e-14); color=:navy, linewidth=2)
    scatter!(ax_wall, dx, max.(wun, 1e-14); color=:navy, markersize=10)

    palette = (:black, :firebrick, :royalblue, :seagreen, :darkorange, :purple)
    for (idx, row) in enumerate(ok)
        h = filter(x -> x.case == row.case, history)
        isempty(h) && continue
        color = palette[mod1(idx, length(palette))]
        steps = [x.step for x in h]
        rho_min = [x.rho_min for x in h]
        rho_max = [x.rho_max for x in h]
        lines!(ax_rho, steps, rho_min; color=color, linewidth=1.5,
               label="$(row.Nx)x$(row.Ny) min")
        lines!(ax_rho, steps, rho_max; color=color, linewidth=1.5,
               linestyle=:dash, label="$(row.Nx)x$(row.Ny) max")
    end
    axislegend(ax_rho; position=:rb, nbanks=2, labelsize=11)

    for row in ok
        text!(ax_q, row.dx_ref, max(row.Q_out_rel_to_finest, 1e-12);
              text="$(row.Nx)x$(row.Ny)", align=(:left, :bottom), fontsize=10)
        text!(ax_mass, row.dx_ref, max(row.Q_rel_err, 1e-12);
              text="$(row.Nx)x$(row.Ny)", align=(:left, :bottom), fontsize=10)
    end

    save(path, fig; px_per_unit=2)
    return path
end

function main()
    L = env_float("KRK_CONV_L", 2.0)
    H_in = env_float("KRK_CONV_H_IN", 1.0)
    H_out = env_float("KRK_CONV_H_OUT", 0.5)
    specs = parse_specs(get(ENV, "KRK_CONV_SWEEP_SPECS",
                            "91x46,121x61,181x91,241x121"))
    steps = env_int("KRK_CONV_SWEEP_STEPS", env_int("KRK_CONV_STEPS", 20000))
    ng = env_int("KRK_CONV_NG", 2)
    u_max = env_float("KRK_CONV_UMAX", 0.02)
    ν = env_float("KRK_CONV_NU", 0.04)
    departure = Symbol(get(ENV, "KRK_CONV_DEPARTURE", "q1_newton"))
    rho_lo = env_float("KRK_CONV_RHO_LO", 0.90)
    rho_hi = env_float("KRK_CONV_RHO_HI", 1.10)

    case_root = joinpath(@__DIR__, "convergence_cases", "single_block_convergent_mesh_sweep")
    plot_dir = joinpath(@__DIR__, "plots")
    table_dir = joinpath(@__DIR__, "paper_tables")
    summary_csv = joinpath(table_dir, "bodyfit_convergent_mesh_sweep.csv")
    history_csv = joinpath(table_dir, "bodyfit_convergent_mesh_sweep_history.csv")
    summary_png = joinpath(plot_dir, "bodyfit_convergent_mesh_sweep.png")

    rows = NamedTuple[]
    history = NamedTuple[]
    println("=== Single-block convergent mesh sweep ===")
    println("specs=", join(["$(a)x$(b)" for (a, b) in specs], ", "),
            " steps=$steps ng=$ng nu=$(ν) u_max=$(u_max)")
    for (Nx, Ny) in specs
        paths = case_paths(case_root, Nx, Ny)
        print("  running $(paths.case) ... ")
        try
            row, hist = run_sweep_case(paths; Nx=Nx, Ny=Ny, steps=steps, ng=ng,
                                       ν=ν, u_max=u_max, L=L, H_in=H_in,
                                       H_out=H_out, departure=departure,
                                       plot_dir=plot_dir,
                                       rho_lo=rho_lo, rho_hi=rho_hi)
            push!(rows, row)
            append!(history, hist)
            println("ok Qerr=$(fmt(row.Q_rel_err)) rho=[$(fmt(row.rho_min)), $(fmt(row.rho_max))]")
        catch err
            row = failed_row(paths; Nx=Nx, Ny=Ny, steps=steps, ng=ng,
                             ν=ν, u_max=u_max, L=L, H_in=H_in,
                             H_out=H_out, err=err)
            push!(rows, row)
            println("failed: ", row.error)
        end
    end

    rows = add_finest_reference(rows)
    columns = (:case, :Nx, :Ny, :nodes, :steps, :ng, :nu, :u_max,
               :L, :H_in, :H_out, :dx_ref, :status, :stable,
               :rho_min, :rho_max, :rho_span,
               :Q_in, :Q_out, :Q_rel_err, :Q_out_rel_to_finest,
               :max_wall_un, :max_speed, :elapsed_s,
               :msh, :krk, :png, :error)
    write_rows_csv(summary_csv, rows, columns)
    write_rows_csv(history_csv, history,
                   (:case, :Nx, :Ny, :step, :rho_min, :rho_max,
                    :Q_in, :Q_out, :Q_rel_err))
    png = write_sweep_plot(summary_png, rows, history)

    nok = count(row -> row.status == "ok", rows)
    nstable = count(row -> get(row, :stable, false) == true, rows)
    println("done: ok=$nok/$(length(rows)) stable=$nstable/$(length(rows))")
    println("wrote:")
    println("  ", relpath(summary_csv, pwd()))
    println("  ", relpath(history_csv, pwd()))
    png !== nothing && println("  ", relpath(png, pwd()))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
