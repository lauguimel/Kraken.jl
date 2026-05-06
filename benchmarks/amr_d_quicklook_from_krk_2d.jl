#!/usr/bin/env julia

using CairoMakie
using Dates
using Kraken
using Printf

const DEFAULT_QUICKLOOK_CASE_DIR = joinpath(
    dirname(@__DIR__), "benchmarks", "krk", "amr_d_convergence_2d")
const DEFAULT_QUICKLOOK_OUTDIR = joinpath(
    dirname(@__DIR__), "benchmarks", "results", "quicklook",
    "amr_d_krk_2d_" * Dates.format(now(), "yyyymmdd_HHMMSS"))

struct AMRDQuicklookArtifact2D
    case_name::String
    flow::Symbol
    method::Symbol
    status::Symbol
    outdir::String
    status_csv::String
    mesh_csv::String
    mesh_png::String
    fields_csv::String
    fields_png::String
    profiles_csv::String
    profiles_png::String
end

function _ql_sanitize_name(name::AbstractString)
    cleaned = replace(String(name), r"[^A-Za-z0-9_.-]+" => "_")
    return isempty(cleaned) ? "case" : cleaned
end

function _ql_parse_case_list(raw::AbstractString; case_dir=DEFAULT_QUICKLOOK_CASE_DIR)
    items = String[]
    for item in split(raw, ",")
        s = strip(item)
        isempty(s) && continue
        path = isabspath(s) ? s : joinpath(case_dir, s)
        endswith(path, ".krk") || (path *= ".krk")
        push!(items, path)
    end
    return items
end

function _ql_env_case_paths()
    raw = strip(get(ENV, "KRK_AMR_D_QUICKLOOK_CASES", ""))
    case_dir = get(ENV, "KRK_AMR_D_QUICKLOOK_DIR", DEFAULT_QUICKLOOK_CASE_DIR)
    if isempty(raw)
        return sort(filter(endswith(".krk"), readdir(case_dir; join=true)))
    end
    return _ql_parse_case_list(raw; case_dir=case_dir)
end

function _ql_steps(setup; steps_override=nothing)
    steps_override !== nothing && return Int(steps_override)
    raw = strip(get(ENV, "KRK_AMR_D_QUICKLOOK_STEPS_OVERRIDE", ""))
    isempty(raw) || return parse(Int, raw)
    return getproperty(setup, :max_steps)
end

function _ql_avg_window(setup, steps::Int; avg_window_override=nothing)
    avg_window_override !== nothing && return min(Int(avg_window_override), steps)
    raw = strip(get(ENV, "KRK_AMR_D_QUICKLOOK_AVG_WINDOW_OVERRIDE", ""))
    isempty(raw) || return min(parse(Int, raw), steps)
    vars = getproperty(setup, :user_vars)
    return min(round(Int, get(vars, :avg_window, max(1, div(steps, 4)))), steps)
end

function _ql_var(setup, name::Symbol, default)
    vars = getproperty(setup, :user_vars)
    return haskey(vars, name) ? vars[name] : default
end

function _ql_eval_expr(expr, default)
    try
        return Float64(evaluate(expr))
    catch
        return Float64(default)
    end
end

function _ql_body_force(setup, name::Symbol, default)
    body = getproperty(getproperty(setup, :physics), :body_force)
    haskey(body, name) || return _ql_var(setup, name, default)
    return _ql_eval_expr(body[name], _ql_var(setup, name, default))
end

function _ql_boundary_value(setup,
                            face::Symbol,
                            bc_type::Symbol,
                            key::Symbol,
                            default)
    for bc in getproperty(setup, :boundaries)
        getproperty(bc, :face) == face || continue
        getproperty(bc, :type) == bc_type || continue
        values = getproperty(bc, :values)
        haskey(values, key) || continue
        return _ql_eval_expr(values[key], default)
    end
    return default
end

function _ql_patch_ranges(setup)
    ranges = conservative_tree_patch_ranges_from_krk_refines_2d(
        getproperty(setup, :domain), getproperty(setup, :refinements))
    length(ranges) == 1 ||
        throw(ArgumentError("quicklook single-patch route needs one Refine block"))
    return only(ranges)
end

function _ql_mass_rel_drift(result)
    mass_initial = getproperty(result, :mass_initial)
    mass_drift = getproperty(result, :mass_drift)
    return abs(mass_drift) / max(abs(mass_initial),
                                 eps(typeof(float(mass_initial))))
end

function _ql_leaf_fields(F, is_solid; volume=0.25, force_x=0.0, force_y=0.0)
    ux = fill(NaN, size(F, 1), size(F, 2))
    uy = similar(ux)
    rho = similar(ux)
    speed = similar(ux)
    @inbounds for j in axes(F, 2), i in axes(F, 1)
        is_solid[i, j] && continue
        cell = @view F[i, j, :]
        rho_ij = mass_F(cell) / volume
        mx, my = momentum_F(cell)
        ux_ij = (mx / volume + force_x / 2) / rho_ij
        uy_ij = (my / volume + force_y / 2) / rho_ij
        rho[i, j] = rho_ij
        ux[i, j] = ux_ij
        uy[i, j] = uy_ij
        speed[i, j] = hypot(ux_ij, uy_ij)
    end
    return (; rho, ux, uy, speed)
end

function _ql_single_patch_level_map(Nx::Int, Ny::Int, patch)
    level = zeros(Int, 2 * Nx, 2 * Ny)
    for J in patch.parent_j_range, I in patch.parent_i_range
        level[(2 * I - 1):(2 * I), (2 * J - 1):(2 * J)] .= 1
    end
    return level
end

function _ql_state_from_composite_result(result; force_x=0.0, force_y=0.0)
    coarse = getproperty(result, :coarse_F)
    patch = getproperty(result, :patch)
    leaf = zeros(eltype(coarse), 2 * size(coarse, 1), 2 * size(coarse, 2), 9)
    composite_to_leaf_F_2d!(leaf, coarse, patch)
    is_solid = hasproperty(result, :is_solid_leaf) ?
        getproperty(result, :is_solid_leaf) : falses(size(leaf, 1), size(leaf, 2))
    fields = _ql_leaf_fields(leaf, is_solid; volume=0.25,
                             force_x=force_x, force_y=force_y)
    level = _ql_single_patch_level_map(size(coarse, 1), size(coarse, 2), patch)
    return (; fields, is_solid, level, patch, leaf_nx=size(leaf, 1),
            leaf_ny=size(leaf, 2))
end

function _ql_state_from_spec_result(result; force_x=0.0, force_y=0.0)
    spec = getproperty(result, :spec)
    F = getproperty(result, :F)
    leaf_nx = spec.Nx << spec.max_level
    leaf_ny = spec.Ny << spec.max_level
    ux = fill(NaN, leaf_nx, leaf_ny)
    uy = similar(ux)
    rho = similar(ux)
    speed = similar(ux)
    level = fill(-1, leaf_nx, leaf_ny)

    @inbounds for cell_id in spec.active_cells
        cell = spec.cells[cell_id]
        scale = 1 << (spec.max_level - cell.level)
        mass = zero(eltype(F))
        mx = zero(eltype(F))
        my = zero(eltype(F))
        for q in 1:9
            Fq = F[cell_id, q]
            mass += Fq
            mx += d2q9_cx(q) * Fq
            my += d2q9_cy(q) * Fq
        end
        volume = eltype(F)(cell.metrics.volume)
        rho_c = mass / volume
        ux_c = (mx / volume + force_x / 2) / rho_c
        uy_c = (my / volume + force_y / 2) / rho_c
        i0 = (cell.i - 1) * scale + 1
        i1 = cell.i * scale
        j0 = (cell.j - 1) * scale + 1
        j1 = cell.j * scale
        rho[i0:i1, j0:j1] .= rho_c
        ux[i0:i1, j0:j1] .= ux_c
        uy[i0:i1, j0:j1] .= uy_c
        speed[i0:i1, j0:j1] .= hypot(ux_c, uy_c)
        level[i0:i1, j0:j1] .= cell.level
    end

    return (; fields=(; rho, ux, uy, speed), is_solid=falses(leaf_nx, leaf_ny),
            level, patch=nothing, leaf_nx, leaf_ny)
end

function _ql_static_solid_mask(setup, case, leaf_nx::Int, leaf_ny::Int)
    case.geometry == :cylinder || return falses(leaf_nx, leaf_ny)
    vars = getproperty(setup, :user_vars)
    haskey(vars, :cx_leaf) && haskey(vars, :cy_leaf) &&
        haskey(vars, :radius_leaf) || return falses(leaf_nx, leaf_ny)
    return cylinder_solid_mask_leaf_2d(
        leaf_nx, leaf_ny, vars[:cx_leaf], vars[:cy_leaf], vars[:radius_leaf])
end

function _ql_mesh_cells_from_patch(Nx::Int, Ny::Int, patch)
    rows = NamedTuple[]
    for J in 1:Ny, I in 1:Nx
        inside = I in patch.parent_i_range && J in patch.parent_j_range
        if inside
            for jf in (2 * J - 1):(2 * J), ifine in (2 * I - 1):(2 * I)
                push!(rows, (; level=1, i=ifine, j=jf,
                             leaf_i_min=ifine, leaf_i_max=ifine,
                             leaf_j_min=jf, leaf_j_max=jf, active=true))
            end
        else
            push!(rows, (; level=0, i=I, j=J,
                         leaf_i_min=2 * I - 1, leaf_i_max=2 * I,
                         leaf_j_min=2 * J - 1, leaf_j_max=2 * J,
                         active=true))
        end
    end
    return rows
end

function _ql_mesh_cells_from_spec(spec)
    rows = NamedTuple[]
    for cell_id in spec.active_cells
        cell = spec.cells[cell_id]
        scale = 1 << (spec.max_level - cell.level)
        push!(rows, (; level=cell.level, i=cell.i, j=cell.j,
                     leaf_i_min=(cell.i - 1) * scale + 1,
                     leaf_i_max=cell.i * scale,
                     leaf_j_min=(cell.j - 1) * scale + 1,
                     leaf_j_max=cell.j * scale,
                     active=true))
    end
    return rows
end

function _ql_write_status_csv(path, setup, case)
    open(path, "w") do io
        println(io, "name,flow,geometry,boundary_policy,wall_model,max_level,refine_count,spec_supported,runtime_supported,runtime_status,reason")
        println(io, join((
            case.name,
            case.flow,
            case.geometry,
            case.boundary_policy,
            case.wall_model,
            case.max_level,
            case.refine_count,
            case.spec_supported,
            case.runtime_supported,
            case.runtime_status,
            repr(case.reason),
        ), ","))
    end
    return path
end

function _ql_write_mesh_csv(path, rows)
    open(path, "w") do io
        println(io, "level,i,j,leaf_i_min,leaf_i_max,leaf_j_min,leaf_j_max,active")
        for r in rows
            println(io, join((r.level, r.i, r.j, r.leaf_i_min, r.leaf_i_max,
                              r.leaf_j_min, r.leaf_j_max, r.active), ","))
        end
    end
    return path
end

function _ql_write_fields_csv(path, state)
    open(path, "w") do io
        println(io, "i_leaf,j_leaf,level,is_solid,rho,ux,uy,speed")
        for j in 1:state.leaf_ny, i in 1:state.leaf_nx
            @printf(io, "%d,%d,%d,%d,%.16e,%.16e,%.16e,%.16e\n",
                    i, j, state.level[i, j], state.is_solid[i, j] ? 1 : 0,
                    state.fields.rho[i, j], state.fields.ux[i, j],
                    state.fields.uy[i, j], state.fields.speed[i, j])
        end
    end
    return path
end

function _ql_profile_vectors(result, state)
    profile = hasproperty(result, :ux_profile) ? getproperty(result, :ux_profile) :
        _ql_mean_ux_by_y(state)
    analytic = hasproperty(result, :analytic_ux_profile) ?
        getproperty(result, :analytic_ux_profile) :
        hasproperty(result, :analytic_profile) ?
        getproperty(result, :analytic_profile) :
        fill(NaN, length(profile))
    return Float64.(profile), Float64.(analytic)
end

function _ql_mean_ux_by_y(state)
    profile = fill(NaN, state.leaf_ny)
    for j in 1:state.leaf_ny
        sum_ux = 0.0
        count = 0
        for i in 1:state.leaf_nx
            value = state.fields.ux[i, j]
            isfinite(value) || continue
            sum_ux += value
            count += 1
        end
        count > 0 && (profile[j] = sum_ux / count)
    end
    return profile
end

function _ql_write_profiles_csv(path, result, state)
    profile, analytic = _ql_profile_vectors(result, state)
    nx = state.leaf_nx
    ny = state.leaf_ny
    j_mid = cld(ny, 2)
    i_probe = clamp(round(Int, 0.75 * nx), 1, nx)
    open(path, "w") do io
        println(io, "kind,index,coord,ux,rho,uy,speed,analytic_ux")
        for j in 1:ny
            coord = ny == 1 ? 0.0 : (j - 1) / (ny - 1)
            @printf(io, "mean_y,%d,%.16e,%.16e,NaN,NaN,NaN,%.16e\n",
                    j, coord, profile[j], analytic[j])
        end
        for i in 1:nx
            coord = (i - 0.5) / nx
            @printf(io, "centerline_x,%d,%.16e,%.16e,%.16e,%.16e,%.16e,NaN\n",
                    i, coord, state.fields.ux[i, j_mid],
                    state.fields.rho[i, j_mid], state.fields.uy[i, j_mid],
                    state.fields.speed[i, j_mid])
        end
        for j in 1:ny
            coord = (j - 0.5) / ny
            @printf(io, "vertical_probe,%d,%.16e,%.16e,%.16e,%.16e,%.16e,NaN\n",
                    j, coord, state.fields.ux[i_probe, j],
                    state.fields.rho[i_probe, j], state.fields.uy[i_probe, j],
                    state.fields.speed[i_probe, j])
        end
    end
    return path
end

function _ql_finite_colorrange(A; symmetric=false)
    vals = Float64[x for x in A if isfinite(x)]
    isempty(vals) && return (0.0, 1.0)
    if symmetric
        m = maximum(abs, vals)
        m == 0 && (m = 1.0)
        return (-m, m)
    end
    lo = minimum(vals)
    hi = maximum(vals)
    lo == hi && (hi = lo + max(abs(lo), 1.0) * 1e-12)
    return (lo, hi)
end

function _ql_rect!(ax, i0, i1, j0, j1; color, linewidth)
    lines!(ax, [i0 - 0.5, i1 + 0.5, i1 + 0.5, i0 - 0.5, i0 - 0.5],
              [j0 - 0.5, j0 - 0.5, j1 + 0.5, j1 + 0.5, j0 - 0.5];
           color=color, linewidth=linewidth)
    return ax
end

function _ql_level_color(level::Int)
    palette = (:gray35, :dodgerblue4, :seagreen4, :orange3, :crimson)
    return palette[clamp(level + 1, 1, length(palette))]
end

function _ql_overlay_solid!(ax, is_solid)
    isempty(is_solid) && return ax
    any(is_solid) || return ax
    heatmap!(ax, 1:size(is_solid, 1), 1:size(is_solid, 2),
             Float64.(is_solid);
             colormap=[RGBAf(0, 0, 0, 0), RGBAf(0.8, 0.03, 0.02, 0.65)],
             colorrange=(0.0, 1.0))
    return ax
end

function _ql_plot_mesh(path, rows; title, leaf_nx, leaf_ny, is_solid=falses(0, 0))
    fig = Figure(size=(1100, 760), fontsize=16)
    ax = Axis(fig[1, 1]; title=title, xlabel="x leaf", ylabel="y leaf",
              aspect=DataAspect())
    for r in rows
        _ql_rect!(ax, r.leaf_i_min, r.leaf_i_max, r.leaf_j_min, r.leaf_j_max;
                  color=(_ql_level_color(r.level), 0.68),
                  linewidth=r.level == 0 ? 0.45 : 0.65)
    end
    _ql_overlay_solid!(ax, is_solid)
    xlims!(ax, 0.5, leaf_nx + 0.5)
    ylims!(ax, 0.5, leaf_ny + 0.5)
    save(path, fig)
    return path
end

function _ql_heatmap!(fig, row, col, title, A; colormap=:viridis,
                      colorrange=nothing)
    ax = Axis(fig[row, col]; title=title, xlabel="x leaf", ylabel="y leaf",
              aspect=DataAspect())
    cr = colorrange === nothing ? _ql_finite_colorrange(A) : colorrange
    Aplot = map(x -> isfinite(x) ? x : cr[1], A)
    hm = heatmap!(ax, 1:size(A, 1), 1:size(A, 2), Aplot;
                  colormap=colormap, colorrange=cr)
    Colorbar(fig[row, col + 1], hm)
    return ax
end

function _ql_plot_fields(path, state; title)
    speed_range = _ql_finite_colorrange(state.fields.speed)
    rho_range = _ql_finite_colorrange(state.fields.rho)
    ux_range = _ql_finite_colorrange(state.fields.ux; symmetric=true)
    fig = Figure(size=(1500, 950), fontsize=16)
    ax1 = _ql_heatmap!(fig, 1, 1, "$title |u|", state.fields.speed;
                       colormap=:viridis, colorrange=speed_range)
    _ql_overlay_solid!(ax1, state.is_solid)
    ax2 = _ql_heatmap!(fig, 1, 3, "$title ux", state.fields.ux;
                       colormap=:balance, colorrange=ux_range)
    _ql_overlay_solid!(ax2, state.is_solid)
    ax3 = _ql_heatmap!(fig, 2, 1, "$title rho", state.fields.rho;
                       colormap=:viridis, colorrange=rho_range)
    _ql_overlay_solid!(ax3, state.is_solid)
    ax4 = _ql_heatmap!(fig, 2, 3, "$title level", Float64.(state.level);
                       colormap=:viridis,
                       colorrange=(minimum(state.level), maximum(state.level)))
    _ql_overlay_solid!(ax4, state.is_solid)
    save(path, fig)
    return path
end

function _ql_lines_finite!(ax, xs, ys; label=nothing, kwargs...)
    first_i = nothing
    label_pending = label
    for k in 1:(length(xs) + 1)
        finite = k <= length(xs) && isfinite(xs[k]) && isfinite(ys[k])
        if finite && first_i === nothing
            first_i = k
        elseif (!finite || k == length(xs) + 1) && first_i !== nothing
            last_i = k - 1
            if last_i >= first_i
                if label_pending === nothing
                    lines!(ax, xs[first_i:last_i], ys[first_i:last_i]; kwargs...)
                else
                    lines!(ax, xs[first_i:last_i], ys[first_i:last_i];
                           label=label_pending, kwargs...)
                    label_pending = nothing
                end
            end
            first_i = nothing
        end
    end
    return ax
end

function _ql_plot_profiles(path, result, state; title)
    profile, analytic = _ql_profile_vectors(result, state)
    nx = state.leaf_nx
    ny = state.leaf_ny
    j_mid = cld(ny, 2)
    i_probe = clamp(round(Int, 0.75 * nx), 1, nx)
    x = (collect(1:nx) .- 0.5) ./ nx
    y = (collect(1:ny) .- 0.5) ./ ny
    y_mean = length(profile) == 1 ? [0.0] :
        [(j - 1) / (length(profile) - 1) for j in eachindex(profile)]

    fig = Figure(size=(1350, 920), fontsize=16)
    ax1 = Axis(fig[1, 1]; title="$title mean ux profile",
               xlabel="ux", ylabel="y/Ly")
    _ql_lines_finite!(ax1, profile, y_mean; label="AMR-D",
                      color=:orangered, linewidth=2.8)
    _ql_lines_finite!(ax1, analytic, y_mean; label="analytic",
                      color=:black, linestyle=:dash, linewidth=2.2)
    axislegend(ax1, position=:rb)

    ax2 = Axis(fig[1, 2]; title="$title centerline ux",
               xlabel="x/Lx", ylabel="ux")
    _ql_lines_finite!(ax2, x, state.fields.ux[:, j_mid];
                      color=:orangered, linewidth=2.5)

    ax3 = Axis(fig[2, 1]; title="$title vertical ux probe",
               xlabel="ux", ylabel="y/Ly")
    _ql_lines_finite!(ax3, state.fields.ux[i_probe, :], y;
                      color=:orangered, linewidth=2.5)

    ax4 = Axis(fig[2, 2]; title="$title centerline rho",
               xlabel="x/Lx", ylabel="rho")
    _ql_lines_finite!(ax4, x, state.fields.rho[:, j_mid];
                      color=:dodgerblue4, linewidth=2.5)
    save(path, fig)
    return path
end

function _ql_run_one_level_case(setup, case; steps, avg_window, method, T)
    domain = getproperty(setup, :domain)
    Nx = Int(domain.Nx)
    Ny = Int(domain.Ny)
    rho = _ql_var(setup, :rho0, 1.0)
    omega = _ql_var(setup, :omega, 1.0)
    patch_i, patch_j = method == :leaf_oracle ? (1:Nx, 1:Ny) :
        _ql_patch_ranges(setup)

    if case.flow == :poiseuille
        Fx = _ql_body_force(setup, :Fx, _ql_var(setup, :Fx, 5e-5))
        runner = method == :leaf_oracle ?
            run_conservative_tree_poiseuille_macroflow_2d :
            run_conservative_tree_poiseuille_route_native_2d
        result = runner(; Nx=Nx, Ny=Ny, patch_i_range=patch_i,
                        patch_j_range=patch_j, Fx=Fx, omega=omega,
                        rho=rho, steps=steps, T=T)
        return result, (force_x=Fx, force_y=0.0)
    elseif case.flow == :couette
        U = _ql_boundary_value(setup, :north, :velocity, :ux,
                               _ql_var(setup, :U, 1e-3))
        runner = method == :leaf_oracle ?
            run_conservative_tree_couette_macroflow_2d :
            run_conservative_tree_couette_route_native_2d
        result = runner(; Nx=Nx, Ny=Ny, patch_i_range=patch_i,
                        patch_j_range=patch_j, U=U, omega=omega,
                        rho=rho, steps=steps, T=T)
        return result, (force_x=0.0, force_y=0.0)
    elseif case.flow == :bfs
        u_in = _ql_boundary_value(setup, :west, :velocity, :ux,
                                  _ql_var(setup, :u_in, 0.03))
        rho_out = _ql_boundary_value(setup, :east, :pressure, :rho, 1.0)
        kwargs = (; Nx=Nx, Ny=Ny, patch_i_range=patch_i,
                  patch_j_range=patch_j,
                  step_i_leaf=round(Int, _ql_var(setup, :step_i_leaf, 16)),
                  step_height_leaf=round(Int, _ql_var(setup, :step_height_leaf, 8)),
                  u_in=u_in, omega=omega, rho=rho, steps=steps, T=T)
        result = method == :leaf_oracle ?
            run_conservative_tree_bfs_macroflow_2d(; kwargs...) :
            run_conservative_tree_bfs_route_native_2d(; kwargs...,
                                                      rho_out=rho_out)
        return result, (force_x=0.0, force_y=0.0)
    elseif case.flow == :square
        Fx = _ql_body_force(setup, :Fx, _ql_var(setup, :Fx, 2e-5))
        Fy = _ql_body_force(setup, :Fy, _ql_var(setup, :Fy, 0.0))
        runner = method == :leaf_oracle ?
            run_conservative_tree_square_obstacle_macroflow_2d :
            run_conservative_tree_square_obstacle_route_native_2d
        result = runner(; Nx=Nx, Ny=Ny, patch_i_range=patch_i,
                        patch_j_range=patch_j,
                        obstacle_i_range=round(Int, _ql_var(setup, :obstacle_i0, 22)):
                                         round(Int, _ql_var(setup, :obstacle_i1, 27)),
                        obstacle_j_range=round(Int, _ql_var(setup, :obstacle_j0, 12)):
                                         round(Int, _ql_var(setup, :obstacle_j1, 17)),
                        Fx=Fx, Fy=Fy, omega=omega, rho=rho, steps=steps, T=T)
        return result, (force_x=Fx, force_y=Fy)
    elseif case.flow == :cylinder
        Fx = _ql_body_force(setup, :Fx, _ql_var(setup, :Fx, 2e-5))
        runner = method == :leaf_oracle ?
            run_conservative_tree_cylinder_macroflow_2d :
            run_conservative_tree_cylinder_obstacle_route_native_2d
        result = runner(; Nx=Nx, Ny=Ny, patch_i_range=patch_i,
                        patch_j_range=patch_j,
                        cx_leaf=_ql_var(setup, :cx_leaf, (2 * Nx + 1) / 2),
                        cy_leaf=_ql_var(setup, :cy_leaf, (2 * Ny + 1) / 2),
                        radius_leaf=_ql_var(setup, :radius_leaf, 3.0),
                        Fx=Fx, omega=omega, rho=rho, steps=steps,
                        avg_window=avg_window, T=T)
        return result, (force_x=Fx, force_y=0.0)
    end

    throw(ArgumentError("quicklook does not dispatch flow $(case.flow)"))
end

function _ql_write_summary_csv(path, artifacts)
    open(path, "w") do io
        println(io, "case,flow,method,status,outdir,status_csv,mesh_csv,mesh_png,fields_csv,fields_png,profiles_csv,profiles_png")
        for a in artifacts
            println(io, join((a.case_name, a.flow, a.method, a.status, a.outdir,
                              a.status_csv, a.mesh_csv, a.mesh_png,
                              a.fields_csv, a.fields_png,
                              a.profiles_csv, a.profiles_png), ","))
        end
    end
    return path
end

function _ql_empty_artifact(setup, case, outdir, status_csv, mesh_csv, mesh_png)
    return AMRDQuicklookArtifact2D(
        case.name, case.flow, :none, case.runtime_status, outdir, status_csv,
        mesh_csv, mesh_png, "", "", "", "")
end

function _ql_artifact(setup, case, method, outdir, status_csv,
                      mesh_csv, mesh_png, fields_csv, fields_png,
                      profiles_csv, profiles_png)
    return AMRDQuicklookArtifact2D(
        case.name, case.flow, method, :ok, outdir, status_csv, mesh_csv,
        mesh_png, fields_csv, fields_png, profiles_csv, profiles_png)
end

function run_amr_d_quicklook_from_krk_2d(paths;
        outdir::AbstractString=DEFAULT_QUICKLOOK_OUTDIR,
        steps_override=nothing,
        avg_window_override=nothing,
        include_reference::Bool=true,
        make_plots::Bool=true,
        T::Type{<:AbstractFloat}=Float64)
    mkpath(outdir)
    artifacts = AMRDQuicklookArtifact2D[]

    for raw_path in paths
        setup = load_kraken(String(raw_path))
        case = conservative_tree_amr_d_case_from_krk_2d(setup)
        case_dir = joinpath(outdir, _ql_sanitize_name(case.name))
        mkpath(case_dir)
        status_csv = _ql_write_status_csv(joinpath(case_dir, "status.csv"),
                                          setup, case)

        mesh_csv = ""
        mesh_png = ""
        if case.spec_supported
            spec = create_conservative_tree_spec_from_krk_2d(setup)
            mesh_rows = _ql_mesh_cells_from_spec(spec)
            leaf_nx = spec.Nx << spec.max_level
            leaf_ny = spec.Ny << spec.max_level
            solid = _ql_static_solid_mask(setup, case, leaf_nx, leaf_ny)
            mesh_csv = _ql_write_mesh_csv(joinpath(case_dir, "mesh_static.csv"),
                                          mesh_rows)
            mesh_png = joinpath(case_dir, "mesh_static.png")
            make_plots && _ql_plot_mesh(
                mesh_png, mesh_rows; title="$(case.name) static AMR mesh",
                leaf_nx=leaf_nx, leaf_ny=leaf_ny, is_solid=solid)
        end

        if !case.runtime_supported
            push!(artifacts, _ql_empty_artifact(
                setup, case, case_dir, status_csv, mesh_csv, mesh_png))
            continue
        end

        steps = _ql_steps(setup; steps_override=steps_override)
        avg = _ql_avg_window(setup, steps; avg_window_override=avg_window_override)
        methods = case.max_level <= 1 && include_reference ?
            (:amr_d, :leaf_oracle) : (:amr_d,)

        for method in methods
            result = nothing
            force = (force_x=0.0, force_y=0.0)
            if case.runtime_status == :subcycled_nested_channel
                result = run_conservative_tree_amr_d_case_from_krk_2d(
                    setup; steps_override=steps, T=T)
                force = (force_x=_ql_body_force(setup, :Fx, 0.0), force_y=0.0)
            else
                result, force = _ql_run_one_level_case(
                    setup, case; steps=steps, avg_window=avg,
                    method=method == :leaf_oracle ? :leaf_oracle : :amr_d,
                    T=T)
            end

            state = hasproperty(result, :spec) ?
                _ql_state_from_spec_result(result; force_x=force.force_x,
                                           force_y=force.force_y) :
                _ql_state_from_composite_result(result; force_x=force.force_x,
                                                force_y=force.force_y)
            mesh_rows = state.patch === nothing ?
                _ql_mesh_cells_from_spec(getproperty(result, :spec)) :
                _ql_mesh_cells_from_patch(
                    div(state.leaf_nx, 2), div(state.leaf_ny, 2), state.patch)

            prefix = String(method)
            mesh_csv_m = _ql_write_mesh_csv(
                joinpath(case_dir, "mesh_$(prefix).csv"), mesh_rows)
            fields_csv = _ql_write_fields_csv(
                joinpath(case_dir, "fields_$(prefix).csv"), state)
            profiles_csv = _ql_write_profiles_csv(
                joinpath(case_dir, "profiles_$(prefix).csv"), result, state)
            mesh_png_m = joinpath(case_dir, "mesh_$(prefix).png")
            fields_png = joinpath(case_dir, "fields_$(prefix).png")
            profiles_png = joinpath(case_dir, "profiles_$(prefix).png")
            if make_plots
                _ql_plot_mesh(mesh_png_m, mesh_rows;
                              title="$(case.name) $(prefix) mesh",
                              leaf_nx=state.leaf_nx, leaf_ny=state.leaf_ny,
                              is_solid=state.is_solid)
                _ql_plot_fields(fields_png, state;
                                title="$(case.name) $(prefix)")
                _ql_plot_profiles(profiles_png, result, state;
                                  title="$(case.name) $(prefix)")
            end
            push!(artifacts, _ql_artifact(
                setup, case, method, case_dir, status_csv, mesh_csv_m,
                mesh_png_m, fields_csv, fields_png, profiles_csv, profiles_png))
        end
    end

    _ql_write_summary_csv(joinpath(outdir, "summary.csv"), artifacts)
    return artifacts
end

run_amr_d_quicklook_from_krk_2d(path::AbstractString; kwargs...) =
    run_amr_d_quicklook_from_krk_2d([path]; kwargs...)

function main()
    paths = _ql_env_case_paths()
    outdir = get(ENV, "KRK_AMR_D_QUICKLOOK_OUTDIR", DEFAULT_QUICKLOOK_OUTDIR)
    include_reference = lowercase(get(ENV, "KRK_AMR_D_QUICKLOOK_REFERENCE", "true")) != "false"
    artifacts = run_amr_d_quicklook_from_krk_2d(
        paths; outdir=outdir, include_reference=include_reference)
    println("wrote ", joinpath(outdir, "summary.csv"))
    for artifact in artifacts
        println("wrote ", artifact.outdir)
    end
    return artifacts
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
