#!/usr/bin/env julia

# Diagnostics for the body-fitted convergent-divergent SLBM cases.
# This script intentionally lives under tmp/paper_bodyfit_run: it probes
# wall leakage and two-block interface failures without changing the library API.

include(joinpath(@__DIR__, "paper_convergent_divergent_mesh_sweep.jl"))

using Statistics: mean

function edge_xy(block, edge::Symbol)
    mesh = block.mesh
    if edge === :west
        return vec(mesh.X[1, :]), vec(mesh.Y[1, :])
    elseif edge === :east
        return vec(mesh.X[end, :]), vec(mesh.Y[end, :])
    elseif edge === :south
        return vec(mesh.X[:, 1]), vec(mesh.Y[:, 1])
    elseif edge === :north
        return vec(mesh.X[:, end]), vec(mesh.Y[:, end])
    end
    error("unknown edge $edge")
end

function edge_velocity(block, state, edge::Symbol)
    ng = state.n_ghost
    n = edge === :west || edge === :east ? block.mesh.Nη : block.mesh.Nξ
    ux = Vector{FT}(undef, n)
    uy = Vector{FT}(undef, n)
    for r in 1:n
        i, j = edge_node_index(block, edge, r)
        ux[r] = state.ux[ng + i, ng + j]
        uy[r] = state.uy[ng + i, ng + j]
    end
    return ux, uy
end

function outward_segment_normal(xa, ya, xb, yb, xc, yc)
    tx = xb - xa
    ty = yb - ya
    ds = hypot(tx, ty)
    ds == 0 && return (zero(FT), zero(FT), zero(FT))
    nx1 = ty / ds
    ny1 = -tx / ds
    xm = (xa + xb) / 2
    ym = (ya + yb) / 2
    if (xm - xc) * nx1 + (ym - yc) * ny1 < 0
        return (-nx1, -ny1, ds)
    end
    return (nx1, ny1, ds)
end

function signed_edge_flux(block, state, edge::Symbol; domain_center=nothing)
    xs, ys = edge_xy(block, edge)
    ux, uy = edge_velocity(block, state, edge)
    xc, yc = domain_center === nothing ? (mean(xs), mean(ys)) : domain_center
    flux = zero(FT)
    max_abs_un = zero(FT)
    max_x = xs[1]
    max_y = ys[1]
    for r in 1:(length(xs) - 1)
        nx, ny, ds = outward_segment_normal(xs[r], ys[r], xs[r + 1], ys[r + 1], xc, yc)
        umx = (ux[r] + ux[r + 1]) / 2
        umy = (uy[r] + uy[r + 1]) / 2
        un = umx * nx + umy * ny
        flux += un * ds
        if abs(un) > max_abs_un
            max_abs_un = abs(un)
            max_x = (xs[r] + xs[r + 1]) / 2
            max_y = (ys[r] + ys[r + 1]) / 2
        end
    end
    return (; flux=Float64(flux), max_abs_un=Float64(max_abs_un),
            max_x=Float64(max_x), max_y=Float64(max_y))
end

function domain_center(mbm)
    xs = FT[]
    ys = FT[]
    for block in mbm.blocks
        append!(xs, vec(block.mesh.X))
        append!(ys, vec(block.mesh.Y))
    end
    return (mean(xs), mean(ys))
end

function boundary_flux_report(mbm, states)
    center = domain_center(mbm)
    rows = NamedTuple[]
    for (k, block) in enumerate(mbm.blocks)
        for edge in EDGE_SYMBOLS_2D
            tag = getproperty(block.boundary_tags, edge)
            rep = signed_edge_flux(block, states[k], edge; domain_center=center)
            push!(rows, (; block=String(block.id), edge=String(edge), tag=String(tag),
                         flux=rep.flux, max_abs_un=rep.max_abs_un,
                         max_x=rep.max_x, max_y=rep.max_y))
        end
    end
    return rows
end

function write_boundary_flux_csv(path, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, "block,edge,tag,flux,max_abs_un,max_x,max_y")
        for r in rows
            println(io, join((r.block, r.edge, r.tag, r.flux, r.max_abs_un,
                              r.max_x, r.max_y), ","))
        end
    end
    return path
end

function wall_flux_totals(rows)
    inlet = -sum(r.flux for r in rows if r.tag == "inlet")
    outlet = sum(r.flux for r in rows if r.tag == "outlet")
    wall = sum(r.flux for r in rows if startswith(r.tag, "wall"))
    total = sum(r.flux for r in rows if r.tag != "interface")
    max_wall_un = maximum([r.max_abs_un for r in rows if startswith(r.tag, "wall")]; init=0.0)
    return (; inlet, outlet, wall, total, qdiff=outlet - inlet,
            max_wall_un)
end

function edge_macro(state, edge::Symbol, r::Int)
    ng = state.n_ghost
    if edge === :west
        i, j = ng + 1, ng + r
    elseif edge === :east
        i, j = ng + state.Nξ_phys, ng + r
    elseif edge === :south
        i, j = ng + r, ng + 1
    elseif edge === :north
        i, j = ng + r, ng + state.Nη_phys
    else
        error("unknown edge $edge")
    end
    return state.ρ[i, j], state.ux[i, j], state.uy[i, j]
end

function interface_macro_jump(mbm, states)
    jump = 0.0
    for iface in mbm.interfaces
        ia = mbm.block_by_id[iface.from[1]]
        ib = mbm.block_by_id[iface.to[1]]
        block_a = mbm.blocks[ia]
        block_b = mbm.blocks[ib]
        edge_a = iface.from[2]
        edge_b = iface.to[2]
        xa, ya = edge_coords(block_a, edge_a)
        xb, yb = edge_coords(block_b, edge_b)
        same = maximum(hypot.(xa .- xb, ya .- yb))
        flipped = maximum(hypot.(xa .- reverse(xb), ya .- reverse(yb)))
        reverse_b = flipped < same
        nrun = edge_length(block_a, edge_a)
        for r in 1:nrun
            rb = reverse_b ? nrun - r + 1 : r
            a = edge_macro(states[ia], edge_a, r)
            b = edge_macro(states[ib], edge_b, rb)
            jump = max(jump, abs(Float64(a[1] - b[1])),
                       abs(Float64(a[2] - b[2])),
                       abs(Float64(a[3] - b[3])))
        end
    end
    return jump
end

function density_extrema_location(mbm, states)
    rho_min = Inf
    rho_max = -Inf
    min_block = ""
    max_block = ""
    min_x = NaN
    min_y = NaN
    max_x = NaN
    max_y = NaN
    n_nonfinite = 0
    for (k, block) in enumerate(mbm.blocks)
        state = states[k]
        ng = state.n_ghost
        for j in 1:block.mesh.Nη, i in 1:block.mesh.Nξ
            rho = Float64(state.ρ[ng + i, ng + j])
            if !isfinite(rho)
                n_nonfinite += 1
                continue
            end
            if rho < rho_min
                rho_min = rho
                min_block = String(block.id)
                min_x = Float64(block.mesh.X[i, j])
                min_y = Float64(block.mesh.Y[i, j])
            end
            if rho > rho_max
                rho_max = rho
                max_block = String(block.id)
                max_x = Float64(block.mesh.X[i, j])
                max_y = Float64(block.mesh.Y[i, j])
            end
        end
    end
    return (; rho_min, rho_max, min_block, min_x, min_y,
            max_block, max_x, max_y, n_nonfinite)
end

function write_debug_csv(path, rows)
    mkpath(dirname(path))
    cols = (:step, :rho_min, :rho_max, :Q_in, :Q_out, :Q_rel_err,
            :interface_jump, :wall_flux, :boundary_total, :max_wall_un,
            :min_block, :min_x, :min_y, :max_block, :max_x, :max_y,
            :n_nonfinite)
    open(path, "w") do io
        println(io, join(cols, ","))
        for r in rows
            println(io, join((getproperty(r, c) for c in cols), ","))
        end
    end
    return path
end

function run_two_block_debug(; Nx::Int=91, Ny::Int=46, steps::Int=8000,
                             ng::Int=2, ν=FT(0.04), u_max=FT(0.02),
                             L=FT(4.0), H=FT(1.0), H_throat=FT(0.5),
                             departure::Symbol=:q1_newton,
                             shared_exchange::Bool=true,
                             sync_interface::Bool=true,
                             wall_ghost_mode::Symbol=:copy_halfway,
                             check_every::Int=50)
    layout = :two_block
    case_root = joinpath(@__DIR__, "convergence_cases", "cd_investigate")
    paths = cd_case_paths(case_root, Nx, Ny; layout=layout)
    write_convergent_divergent_msh(paths.msh; L=L, H=H, H_throat=H_throat,
                                   Nx=Nx, Ny=Ny)
    write_cd_krk(paths.krk, paths.msh; L=L, H=H, H_throat=H_throat,
                 Nx=Nx, Ny=Ny, u_max=u_max, Re=20.0, steps=steps)
    mbm_raw, _ = load_gmsh_multiblock_2d(paths.msh; FT=FT, layout=:topological)
    mbm = orient_convergent_divergent(mbm_raw)
    hfn = x -> cd_height(x; L=L, H=H, H_throat=H_throat)
    rows = NamedTuple[]
    cb = function (step, mbm_cb, states_cb, h)
        flux_rows = boundary_flux_report(mbm_cb, states_cb)
        totals = wall_flux_totals(flux_rows)
        loc = density_extrema_location(mbm_cb, states_cb)
        push!(rows, merge(h, (;
            interface_jump=interface_macro_jump(mbm_cb, states_cb),
            wall_flux=totals.wall,
            boundary_total=totals.total,
            max_wall_un=totals.max_wall_un,
        ), loc))
    end
    status = "ok"
    errtxt = ""
    try
        run_convergent(mbm; steps=steps, ng=ng, ν=ν, L=L, H_in=H,
                       H_out=H, u_max=u_max, departure=departure,
                       height_fn=hfn, shared_exchange=shared_exchange,
                       sync_interface=sync_interface,
                       wall_ghost_mode=wall_ghost_mode,
                       check_every_override=check_every,
                       diagnostic_cb=cb)
    catch err
        status = "failed"
        errtxt = sprint(showerror, err)
    end
    csv = joinpath(@__DIR__, "paper_tables",
                   "bodyfit_cd_twoblock_debug_$(wall_ghost_mode)_n$(Nx)x$(Ny)_s$(steps).csv")
    write_debug_csv(csv, rows)
    return (; status, error=errtxt, csv, rows,
            msh=paths.msh, krk=paths.krk)
end

function run_flux_diagnostic(; layout::Symbol=:single_spline, Nx::Int=91, Ny::Int=46,
                             steps::Int=4000, ng::Int=2, ν=FT(0.04),
                             u_max=FT(0.02), L=FT(4.0), H=FT(1.0),
                             H_throat=FT(0.5), departure::Symbol=:q1_newton,
                             shared_exchange::Bool=true,
                             wall_ghost_mode::Symbol=:copy_halfway)
    case_root = joinpath(@__DIR__, "convergence_cases", "cd_investigate")
    plot_dir = joinpath(@__DIR__, "plots")
    paths = cd_case_paths(case_root, Nx, Ny; layout=layout)
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
    png = joinpath(plot_dir, "bodyfit_convergent_divergent_$(layout)_$(wall_ghost_mode)_n$(Nx)x$(Ny)_diag.png")
    plot_cd(png, mbm, run.states, run; L=L, H=H, H_throat=H_throat,
            u_max=u_max, ν=ν)
    row = merge((; case="n$(Nx)x$(Ny)", Nx=Nx, Ny=Ny,
                 nodes=sum(b.mesh.Nξ * b.mesh.Nη for b in mbm.blocks),
                 steps=steps, ng=ng, nu=Float64(ν), u_max=Float64(u_max),
                 L=Float64(L), H_in=Float64(H), H_out=Float64(H),
                 dx_ref=Float64(run.dx_ref), status="ok",
                 stable=isfinite(metrics.rho_min) && isfinite(metrics.rho_max),
                 msh=relpath(paths.msh, pwd()), krk=relpath(paths.krk, pwd()),
                 png=relpath(png, pwd()), error=""), metrics)
    rows = boundary_flux_report(mbm, run.states)
    csv = joinpath(@__DIR__, "paper_tables",
                   "bodyfit_cd_flux_$(layout)_$(wall_ghost_mode)_n$(Nx)x$(Ny)_s$(steps).csv")
    write_boundary_flux_csv(csv, rows)
    totals = wall_flux_totals(rows)
    return (; row, rows, totals, csv)
end

function main()
    layout = Symbol(get(ENV, "KRK_CD_DIAG_LAYOUT", "single_spline"))
    Nx, Ny = parse_specs(get(ENV, "KRK_CD_DIAG_SPEC", "91x46"))[1]
    steps = env_int("KRK_CD_DIAG_STEPS", 4000)
    ng = env_int("KRK_CD_DIAG_NG", 2)
    ν = env_float("KRK_CD_DIAG_NU", 0.04)
    u_max = env_float("KRK_CD_DIAG_UMAX", 0.02)
    shared_exchange = get(ENV, "KRK_CD_DIAG_SHARED_EXCHANGE", "1") != "0"
    wall_ghost_mode = Symbol(get(ENV, "KRK_CD_DIAG_WALL_GHOST", "copy_halfway"))
    if layout === :two_block
        result = run_two_block_debug(; Nx=Nx, Ny=Ny, steps=steps,
                                     ng=ng, ν=ν, u_max=u_max,
                                     shared_exchange=shared_exchange,
                                     wall_ghost_mode=wall_ghost_mode)
        println("status=", result.status)
        println("error=", result.error)
        println("csv=", relpath(result.csv, pwd()))
        for r in result.rows[max(1, end - 9):end]
            println(r)
        end
    else
        result = run_flux_diagnostic(; layout=layout, Nx=Nx, Ny=Ny, steps=steps,
                                     ng=ng, ν=ν, u_max=u_max,
                                     shared_exchange=shared_exchange,
                                     wall_ghost_mode=wall_ghost_mode)
        println("row=", result.row)
        println("totals=", result.totals)
        println("csv=", relpath(result.csv, pwd()))
        for r in result.rows
            println(r)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
