include(joinpath(@__DIR__, "paper_convergent_mesh_sweep.jl"))

cd_height(x; L=FT(4.0), H=FT(1.0), H_throat=FT(0.5)) =
    x <= L / 2 ? H + (H_throat - H) * x / (L / 2) :
                 H_throat + (H - H_throat) * (x - L / 2) / (L / 2)

function write_convergent_divergent_msh(path::AbstractString;
                                        L=FT(4.0), H=FT(1.0),
                                        H_throat=FT(0.5),
                                        Nx::Int=121, Ny::Int=61)
    mkpath(dirname(path))
    Nx_left = fld(Nx, 2) + 1
    Nx_right = Nx - Nx_left + 1
    xmid = L / 2

    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("convergent_divergent_2block")

        p0 = gmsh.model.geo.addPoint(0, 0, 0)
        p1 = gmsh.model.geo.addPoint(xmid, 0, 0)
        p2 = gmsh.model.geo.addPoint(L, 0, 0)
        p3 = gmsh.model.geo.addPoint(L, H, 0)
        p4 = gmsh.model.geo.addPoint(xmid, H_throat, 0)
        p5 = gmsh.model.geo.addPoint(0, H, 0)

        l_bot_l = gmsh.model.geo.addLine(p0, p1)
        l_throat = gmsh.model.geo.addLine(p1, p4)
        l_top_l = gmsh.model.geo.addLine(p4, p5)
        l_in = gmsh.model.geo.addLine(p5, p0)

        l_bot_r = gmsh.model.geo.addLine(p1, p2)
        l_out = gmsh.model.geo.addLine(p2, p3)
        l_top_r = gmsh.model.geo.addLine(p3, p4)

        loop_l = gmsh.model.geo.addCurveLoop([l_bot_l, l_throat, l_top_l, l_in])
        surf_l = gmsh.model.geo.addPlaneSurface([loop_l])
        loop_r = gmsh.model.geo.addCurveLoop([l_bot_r, l_out, l_top_r, -l_throat])
        surf_r = gmsh.model.geo.addPlaneSurface([loop_r])

        for line in (l_bot_l, l_top_l)
            gmsh.model.geo.mesh.setTransfiniteCurve(line, Nx_left)
        end
        for line in (l_bot_r, l_top_r)
            gmsh.model.geo.mesh.setTransfiniteCurve(line, Nx_right)
        end
        for line in (l_in, l_throat, l_out)
            gmsh.model.geo.mesh.setTransfiniteCurve(line, Ny)
        end
        for surface in (surf_l, surf_r)
            gmsh.model.geo.mesh.setTransfiniteSurface(surface)
            gmsh.model.geo.mesh.setRecombine(2, surface)
        end

        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(1, [l_in], -1, "inlet")
        gmsh.model.addPhysicalGroup(1, [l_out], -1, "outlet")
        gmsh.model.addPhysicalGroup(1, [l_bot_l, l_bot_r], -1, "wall_bot")
        gmsh.model.addPhysicalGroup(1, [l_top_l, l_top_r], -1, "wall_top")
        gmsh.model.addPhysicalGroup(1, [l_throat], -1, "interface")
        gmsh.model.addPhysicalGroup(2, [surf_l], -1, "block_convergent")
        gmsh.model.addPhysicalGroup(2, [surf_r], -1, "block_divergent")

        gmsh.model.mesh.generate(2)
        gmsh.write(path)
    finally
        gmsh.finalize()
    end
    return path
end

function write_convergent_divergent_spline_msh(path::AbstractString;
                                               L=FT(4.0), H=FT(1.0),
                                               H_throat=FT(0.5),
                                               Nx::Int=121, Ny::Int=61)
    mkpath(dirname(path))
    gmsh.initialize()
    try
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("convergent_divergent_spline")

        p0 = gmsh.model.geo.addPoint(0, 0, 0)
        p1 = gmsh.model.geo.addPoint(L, 0, 0)
        p2 = gmsh.model.geo.addPoint(L, H, 0)
        p3 = gmsh.model.geo.addPoint(L / 2, H_throat, 0)
        p4 = gmsh.model.geo.addPoint(0, H, 0)

        l_bot = gmsh.model.geo.addLine(p0, p1)
        l_out = gmsh.model.geo.addLine(p1, p2)
        l_top = gmsh.model.geo.addSpline([p2, p3, p4])
        l_in = gmsh.model.geo.addLine(p4, p0)

        loop = gmsh.model.geo.addCurveLoop([l_bot, l_out, l_top, l_in])
        surf = gmsh.model.geo.addPlaneSurface([loop])

        gmsh.model.geo.mesh.setTransfiniteCurve(l_bot, Nx)
        gmsh.model.geo.mesh.setTransfiniteCurve(l_top, Nx)
        gmsh.model.geo.mesh.setTransfiniteCurve(l_in, Ny)
        gmsh.model.geo.mesh.setTransfiniteCurve(l_out, Ny)
        gmsh.model.geo.mesh.setTransfiniteSurface(surf)
        gmsh.model.geo.mesh.setRecombine(2, surf)

        gmsh.model.geo.synchronize()
        gmsh.model.addPhysicalGroup(1, [l_in], -1, "inlet")
        gmsh.model.addPhysicalGroup(1, [l_out], -1, "outlet")
        gmsh.model.addPhysicalGroup(1, [l_bot], -1, "wall_bot")
        gmsh.model.addPhysicalGroup(1, [l_top], -1, "wall_top")
        gmsh.model.addPhysicalGroup(2, [surf], -1, "block_cd_spline")

        gmsh.model.mesh.generate(2)
        gmsh.write(path)
    finally
        gmsh.finalize()
    end
    return path
end

function write_cd_krk(path::AbstractString, msh_path::AbstractString;
                      L=FT(4.0), H=FT(1.0), H_throat=FT(0.5),
                      Nx::Int=121, Ny::Int=61,
                      u_max=FT(0.02), Re=FT(20.0), steps::Int=20000)
    mkpath(dirname(path))
    rel_mesh = relpath(msh_path, dirname(path))
    open(path, "w") do io
        println(io, "Simulation convergent_divergent_2block D2Q9")
        println(io, "Module slbm_drag")
        println(io)
        println(io, "Define L = $(L)")
        println(io, "Define H = $(H)")
        println(io, "Define H_throat = $(H_throat)")
        println(io, "Define U = $(u_max)")
        println(io)
        println(io, "Domain L = L x H  N = $(Nx) x $(Ny)")
        println(io, "Mesh gmsh(file = \"$rel_mesh\", layout = topological, multiblock = true)")
        println(io)
        println(io, "Physics Re = $(Re) u_max = U")
        println(io)
        println(io, "Boundary west velocity(ux = U, uy = 0)")
        println(io, "Boundary east pressure(rho = 1.0)")
        println(io, "Boundary south wall")
        println(io, "Boundary north wall")
        println(io)
        println(io, "Run $(steps) steps")
    end
    return path
end

function orient_convergent_divergent(mbm_raw)
    mbm = autoreorient_blocks(mbm_raw; verbose=false)
    issues = sanity_check_multiblock(mbm; verbose=false)
    errors = count(issue -> issue.severity === :error, issues)
    errors == 0 || error("convergent-divergent sanity check has $errors error(s)")
    tags = Symbol[]
    for block in mbm.blocks, edge in (:west, :east, :south, :north)
        push!(tags, getproperty(block.boundary_tags, edge))
    end
    count(==(:inlet), tags) == 1 || error("expected exactly one inlet edge")
    count(==(:outlet), tags) == 1 || error("expected exactly one outlet edge")
    count(tag -> tag === :interface || tag === INTERFACE_TAG, tags) == 2 ||
        error("expected two interface edge references")
    return mbm
end

function tagged_profiles(mbm, states)
    y_in = FT[]; u_in = FT[]
    y_out = FT[]; u_out = FT[]
    for (k, block) in enumerate(mbm.blocks)
        state = states[k]
        ng = state.n_ghost
        for edge in (:west, :east)
            tag = getproperty(block.boundary_tags, edge)
            (tag === :inlet || tag === :outlet) || continue
            i = edge === :west ? 1 : block.mesh.Nξ
            ii = ng + i
            ys = vec(block.mesh.Y[i, :])
            us = [state.ux[ii, ng + j] for j in 1:block.mesh.Nη]
            if tag === :inlet
                append!(y_in, ys); append!(u_in, us)
            else
                append!(y_out, ys); append!(u_out, us)
            end
        end
    end
    return (; y_in, u_in, y_out, u_out)
end

function max_wall_normal_velocity(mbm, states)
    vmax = 0.0
    for (k, block) in enumerate(mbm.blocks)
        for edge in (:south, :north, :west, :east)
            tag = getproperty(block.boundary_tags, edge)
            (tag === :wall_bot || tag === :wall_top || tag === :wall) || continue
            vals = edge_normal_velocity(block, states[k], edge)
            vmax = max(vmax, maximum(vals))
        end
    end
    return Float64(vmax)
end

function final_metrics_cd(mbm, states, run)
    flows = tagged_vertical_flow_rates(mbm, states)
    rho_min, rho_max = physical_density_bounds(states)
    data = physical_speeds(mbm, states)
    return merge((;
        rho_min=Float64(rho_min),
        rho_max=Float64(rho_max),
        rho_span=Float64(rho_max - rho_min),
        Q_in=Float64(flows.Q_in),
        Q_out=Float64(flows.Q_out),
        Q_rel_err=Float64(abs(flows.Q_out - flows.Q_in) /
                          max(abs(flows.Q_in), eps(Float64))),
        max_wall_un=max_wall_normal_velocity(mbm, states),
        max_speed=Float64(maximum(data.speed)),
        elapsed_s=Float64(run.elapsed_s),
    ), flow_tail_metrics(run.history))
end

function plot_cd(path, mbm, states, run; L=FT(4.0), H=FT(1.0),
                 H_throat=FT(0.5), u_max=FT(0.02), ν=FT(0.04))
    HAS_CAIRO || return nothing
    mkpath(dirname(path))
    data = physical_speeds(mbm, states)
    flows = tagged_vertical_flow_rates(mbm, states)
    prof = tagged_profiles(mbm, states)
    wall_un = max_wall_normal_velocity(mbm, states)

    fig = Figure(size=(1550, 920))
    mesh_title = length(mbm.blocks) == 1 ? "single-block spline convergent-divergent" :
                                           "2-block convergent-divergent"
    ax_mesh = Axis(fig[1, 1], aspect=DataAspect(), title=mesh_title,
                   xlabel="x", ylabel="y")
    ax_ux = Axis(fig[1, 2], aspect=DataAspect(), title="ux", xlabel="x", ylabel="y")
    ax_speed = Axis(fig[1, 3], aspect=DataAspect(), title="|u|", xlabel="x", ylabel="y")
    ax_rho = Axis(fig[2, 1], aspect=DataAspect(), title="rho", xlabel="x", ylabel="y")
    ax_prof = Axis(fig[2, 2], title="inlet/outlet profiles", xlabel="ux", ylabel="y")
    ax_hist = Axis(fig[2, 3], title="density and flux history", xlabel="step", ylabel="value")

    colors = (:steelblue, :darkorange, :seagreen, :purple)
    for (k, block) in enumerate(mbm.blocks)
        mesh = block.mesh
        c = colors[mod1(k, length(colors))]
        for j in 1:mesh.Nη
            lines!(ax_mesh, mesh.X[:, j], mesh.Y[:, j], color=(c, 0.35), linewidth=0.35)
        end
        for i in 1:mesh.Nξ
            lines!(ax_mesh, mesh.X[i, :], mesh.Y[i, :], color=(c, 0.35), linewidth=0.35)
        end
        for edge in (:west, :east, :south, :north)
            tag = getproperty(block.boundary_tags, edge)
            xs, ys = edge === :west  ? (mesh.X[1, :], mesh.Y[1, :]) :
                     edge === :east  ? (mesh.X[end, :], mesh.Y[end, :]) :
                     edge === :south ? (mesh.X[:, 1], mesh.Y[:, 1]) :
                                       (mesh.X[:, end], mesh.Y[:, end])
            color = tag === :inlet ? :seagreen :
                    tag === :outlet ? :royalblue :
                    (tag === :interface || tag === INTERFACE_TAG) ? :purple : :black
            lines!(ax_mesh, xs, ys, color=color, linewidth=3)
        end
    end

    uxlim = max(maximum(abs, data.ux), FT(1e-10))
    sc_ux = scatter!(ax_ux, data.xs, data.ys; color=data.ux, markersize=3,
                     colormap=:balance, colorrange=(-uxlim, uxlim))
    Colorbar(fig[1, 4], sc_ux)
    sc_speed = scatter!(ax_speed, data.xs, data.ys; color=data.speed,
                        markersize=3, colormap=:viridis)
    Colorbar(fig[1, 5], sc_speed)
    rho_delta = max(maximum(abs.(data.rho .- 1)), FT(1e-8))
    sc_rho = scatter!(ax_rho, data.xs, data.ys; color=data.rho, markersize=3,
                      colormap=:balance, colorrange=(1 - rho_delta, 1 + rho_delta))
    Colorbar(fig[2, 4], sc_rho)

    yline = range(0, H; length=300)
    inlet_ref = [4 * u_max * y / H * (1 - y / H) for y in yline]
    lines!(ax_prof, inlet_ref, yline; color=:black, linewidth=2, label="target inlet")
    scatter!(ax_prof, prof.u_in, prof.y_in; color=:seagreen, markersize=4, label="inlet")
    scatter!(ax_prof, prof.u_out, prof.y_out; color=:royalblue, markersize=4, label="outlet")
    axislegend(ax_prof; position=:rt)

    steps = [h.step for h in run.history]
    rho_min = [h.rho_min for h in run.history]
    rho_max = [h.rho_max for h in run.history]
    qerr = [h.Q_rel_err for h in run.history]
    lines!(ax_hist, steps, rho_min; color=:firebrick, linewidth=1.5, label="rho min")
    lines!(ax_hist, steps, rho_max; color=:navy, linewidth=1.5, label="rho max")
    lines!(ax_hist, steps, qerr; color=:black, linewidth=1.5, label="Q rel err")
    axislegend(ax_hist; position=:rt)

    Label(fig[3, 1:3],
          "L=$(L), H=$(H), H_throat=$(H_throat), Umax=$(u_max), nu=$(ν), " *
          "dx_ref=$(fmt(run.dx_ref)), Q_in=$(fmt(flows.Q_in)), Q_out=$(fmt(flows.Q_out)), " *
          "max wall |u.n|=$(fmt(wall_un))";
          tellwidth=false, fontsize=16)
    save(path, fig; px_per_unit=2)
    return path
end

function cd_case_paths(case_root, Nx, Ny; layout::Symbol=:two_block)
    case = "n$(Nx)x$(Ny)"
    base = joinpath(case_root, string(layout), case)
    msh = joinpath(base, "meshes", "convergent_divergent_$(layout)_$(case).msh")
    krk = joinpath(base, "krk", "convergent_divergent_$(layout)_$(case).krk")
    return (; case, base, msh, krk)
end

function run_cd_sweep_case(paths; Nx, Ny, steps, ng, ν, u_max,
                           L, H, H_throat, departure, plot_dir,
                           rho_lo, rho_hi, layout::Symbol=:two_block,
                           shared_exchange::Bool=true,
                           wall_ghost_mode::Symbol=:copy_halfway)
    if layout === :single_spline
        write_convergent_divergent_spline_msh(paths.msh; L=L, H=H,
                                              H_throat=H_throat, Nx=Nx, Ny=Ny)
    else
        write_convergent_divergent_msh(paths.msh; L=L, H=H, H_throat=H_throat,
                                       Nx=Nx, Ny=Ny)
    end
    write_cd_krk(paths.krk, paths.msh; L=L, H=H, H_throat=H_throat,
                 Nx=Nx, Ny=Ny, u_max=u_max, Re=20.0, steps=steps)

    mbm_raw, _ = load_gmsh_multiblock_2d(paths.msh; FT=FT, layout=:topological)
    mbm = layout === :single_spline ? orient_single_convergent(mbm_raw) :
                                      orient_convergent_divergent(mbm_raw)
    hfn = x -> cd_height(x; L=L, H=H, H_throat=H_throat)
    run = run_convergent(mbm; steps=steps, ng=ng, ν=ν, L=L, H_in=H,
                         H_out=H, u_max=u_max, departure=departure,
                         height_fn=hfn, shared_exchange=shared_exchange,
                         wall_ghost_mode=wall_ghost_mode)
    metrics = final_metrics_cd(mbm, run.states, run)
    stable = isfinite(metrics.rho_min) && isfinite(metrics.rho_max) &&
             metrics.rho_min >= rho_lo && metrics.rho_max <= rho_hi

    mode_suffix = wall_ghost_mode === :copy_halfway ? "" : "_$(wall_ghost_mode)"
    png = joinpath(plot_dir, "bodyfit_convergent_divergent_$(layout)$(mode_suffix)_$(paths.case).png")
    plot_cd(png, mbm, run.states, run; L=L, H=H, H_throat=H_throat,
            u_max=u_max, ν=ν)

    row = merge((;
        case=paths.case, Nx=Nx, Ny=Ny, nodes=sum(b.mesh.Nξ * b.mesh.Nη for b in mbm.blocks),
        steps=steps, ng=ng, nu=Float64(ν), u_max=Float64(u_max),
        L=Float64(L), H_in=Float64(H), H_out=Float64(H),
        dx_ref=Float64(run.dx_ref), status="ok", stable=stable,
        msh=relpath(paths.msh, pwd()), krk=relpath(paths.krk, pwd()),
        png=relpath(png, pwd()), error="",
    ), metrics)
    history = [merge((; case=paths.case, Nx=Nx, Ny=Ny), h) for h in run.history]
    return row, history
end

function main()
    L = env_float("KRK_CD_L", 4.0)
    H = env_float("KRK_CD_H", 1.0)
    H_throat = env_float("KRK_CD_H_THROAT", 0.5)
    layout = Symbol(get(ENV, "KRK_CD_LAYOUT", "single_spline"))
    specs = parse_specs(get(ENV, "KRK_CD_SWEEP_SPECS",
                            "91x46,121x61,181x91,241x121"))
    steps = env_int("KRK_CD_SWEEP_STEPS", 20000)
    ng = env_int("KRK_CD_NG", 2)
    u_max = env_float("KRK_CD_UMAX", 0.02)
    ν = env_float("KRK_CD_NU", 0.04)
    departure = Symbol(get(ENV, "KRK_CD_DEPARTURE", "q1_newton"))
    rho_lo = env_float("KRK_CD_RHO_LO", 0.90)
    rho_hi = env_float("KRK_CD_RHO_HI", 1.10)
    shared_exchange = get(ENV, "KRK_CD_SHARED_EXCHANGE", "1") != "0"
    wall_ghost_mode = Symbol(get(ENV, "KRK_CD_WALL_GHOST", "copy_halfway"))

    case_root = joinpath(@__DIR__, "convergence_cases", "convergent_divergent_mesh_sweep")
    plot_dir = joinpath(@__DIR__, "plots")
    table_dir = joinpath(@__DIR__, "paper_tables")
    suffix = (layout === :single_spline && wall_ghost_mode === :copy_halfway) ?
             "" : "_$(layout)_$(wall_ghost_mode)"
    summary_csv = joinpath(table_dir, "bodyfit_convergent_divergent_mesh_sweep$(suffix).csv")
    history_csv = joinpath(table_dir, "bodyfit_convergent_divergent_mesh_sweep$(suffix)_history.csv")
    summary_png = joinpath(plot_dir, "bodyfit_convergent_divergent_mesh_sweep$(suffix).png")

    rows = NamedTuple[]
    history = NamedTuple[]
    println("=== convergent-divergent mesh sweep ($(layout)) ===")
    println("specs=", join(["$(a)x$(b)" for (a, b) in specs], ", "),
            " steps=$steps ng=$ng nu=$(ν) u_max=$(u_max)",
            " wall_ghost=$(wall_ghost_mode)")
    for (Nx, Ny) in specs
        paths = cd_case_paths(case_root, Nx, Ny; layout=layout)
        print("  running $(paths.case) ... ")
        try
            row, hist = run_cd_sweep_case(paths; Nx=Nx, Ny=Ny, steps=steps, ng=ng,
                                          ν=ν, u_max=u_max, L=L, H=H,
                                          H_throat=H_throat, departure=departure,
                                          plot_dir=plot_dir, rho_lo=rho_lo,
                                          rho_hi=rho_hi, layout=layout,
                                          shared_exchange=shared_exchange,
                                          wall_ghost_mode=wall_ghost_mode)
            push!(rows, row)
            append!(history, hist)
            println("ok Qerr=$(fmt(row.Q_rel_err)) Qtail_rms=$(fmt(row.Q_rel_tail_rms)) " *
                    "rho=[$(fmt(row.rho_min)), $(fmt(row.rho_max))]")
        catch err
            row = failed_row(paths; Nx=Nx, Ny=Ny, steps=steps, ng=ng,
                             ν=ν, u_max=u_max, L=L, H_in=H,
                             H_out=H, err=err)
            push!(rows, row)
            println("failed: ", row.error)
        end
    end

    rows = add_finest_reference(rows)
    columns = (:case, :Nx, :Ny, :nodes, :steps, :ng, :nu, :u_max,
               :L, :H_in, :H_out, :dx_ref, :status, :stable,
               :rho_min, :rho_max, :rho_span,
               :Q_in, :Q_out, :Q_rel_err, :Q_out_rel_to_finest,
               :Q_tail_start_step, :Q_tail_n, :Q_rel_tail_mean,
               :Q_rel_tail_mean_abs, :Q_rel_tail_rms, :Q_rel_tail_max_abs,
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
