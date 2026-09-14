#!/usr/bin/env julia

# Axis-aligned 4:1 contraction comparison for the log-FV polymer backend.
# The geometry intentionally avoids curved cut cells; this is the discriminator
# between core polymer-pipeline issues and cylinder-only curved-wall issues.

using Dates
using KernelAbstractions
using Printf
using Serialization

using Kraken

const DEFAULT_RHEOTOOL_CASE = joinpath("bench", "rheotool", "contraction41_oldroydb_log")

const SUMMARY_COLUMNS = (
    :case,
    :status,
    :error,
    :backend,
    :float_type,
    :H_out,
    :H_in,
    :Nx,
    :Ny,
    :L_up,
    :L_down,
    :beta,
    :Re,
    :Wi,
    :u_mean,
    :nu_total,
    :nu_s,
    :nu_p,
    :lambda,
    :steps,
    :completed_steps,
    :polymer_substeps,
    :raw_substeps,
    :substeps_clamped,
    :min_c_eig,
    :max_c_trace,
    :rho_min,
    :rho_max,
    :max_speed,
    :tau_xx_peak,
    :centerline_u_peak,
    :centerline_u_overshoot,
    :throat_tau_xx_peak,
    :vortex_length_top,
    :first_nonfinite_step,
    :profile_centerline,
    :profile_throat,
    :field_dump,
)

const CUDA_MOD = try
    @eval using CUDA
    getfield(Main, :CUDA)
catch
    nothing
end

const METAL_MOD = if Sys.isapple()
    try
        @eval using Metal
        getfield(Main, :Metal)
    catch
        nothing
    end
else
    nothing
end

function parse_list(::Type{T}, raw::AbstractString) where {T}
    values = T[]
    for part in split(replace(raw, ';' => ','), ',')
        text = strip(part)
        isempty(text) && continue
        push!(values, parse(T, text))
    end
    isempty(values) && error("empty list")
    return values
end

function parse_bool(name::AbstractString, default::Bool)
    raw = lowercase(strip(get(ENV, name, default ? "1" : "0")))
    raw in ("1", "true", "yes", "on") && return true
    raw in ("0", "false", "no", "off") && return false
    error("$(name) must be boolean-like, got $(raw)")
end

function parse_polymer_substeps(raw::AbstractString)
    text = lowercase(strip(raw))
    text == "auto" && return :auto
    return parse(Int, text)
end

function select_backend()
    requested = lowercase(get(ENV, "KRAKEN_BACKEND", "auto"))
    if requested in ("auto", "cuda") && CUDA_MOD !== nothing
        try
            if Base.invokelatest(getfield(CUDA_MOD, :functional))
                return Base.invokelatest(getfield(CUDA_MOD, :CUDABackend)), "cuda"
            end
        catch err
            requested == "cuda" && rethrow(err)
        end
    end
    if requested in ("auto", "metal") && METAL_MOD !== nothing
        try
            if Base.invokelatest(getfield(METAL_MOD, :functional))
                return Base.invokelatest(getfield(METAL_MOD, :MetalBackend)), "metal"
            end
        catch err
            requested == "metal" && rethrow(err)
        end
    end
    requested in ("auto", "cpu") || error("unknown or unavailable KRAKEN_BACKEND=$(requested)")
    return KernelAbstractions.CPU(), "cpu"
end

function select_float_type(backend_name::AbstractString)
    raw = lowercase(get(ENV, "KRAKEN_FT", "auto"))
    if raw == "auto"
        return backend_name == "metal" ? Float32 : Float64
    elseif raw in ("float32", "f32", "single")
        return Float32
    elseif raw in ("float64", "f64", "double")
        backend_name == "metal" && error("Metal uses Float32 for local smoke runs")
        return Float64
    end
    error("unknown KRAKEN_FT=$(raw)")
end

function csv_cell(x)
    x === nothing && return ""
    if x isa AbstractFloat
        return isfinite(x) ? @sprintf("%.16g", x) : string(x)
    end
    s = string(x)
    if occursin(',', s) || occursin('"', s) || occursin('\n', s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function append_summary(path::AbstractString, row::Dict{Symbol,Any})
    first = !isfile(path)
    open(path, first ? "w" : "a") do io
        first && println(io, join(string.(SUMMARY_COLUMNS), ","))
        println(io, join((csv_cell(get(row, col, "")) for col in SUMMARY_COLUMNS), ","))
    end
end

function guard_local_run!(backend_name, Nx, Ny, steps)
    backend_name == "cuda" && return nothing
    parse_bool("KRAKEN_ALLOW_LONG_LOCAL", false) && return nothing
    max_updates = parse(Float64, get(ENV, "KRAKEN_MAX_LOCAL_UPDATES", "5e7"))
    updates = Float64(Nx) * Float64(Ny) * Float64(steps)
    updates <= max_updates && return nothing
    error("refusing local run with $(updates) cell-steps; set KRAKEN_ALLOW_LONG_LOCAL=1")
end

function center_rows(Ny::Int)
    iseven(Ny) && return (Ny ÷ 2, Ny ÷ 2 + 1)
    c = cld(Ny, 2)
    return (c, c)
end

function center_value(field, i, j1, j2)
    return j1 == j2 ? Float64(field[i, j1]) : 0.5 * (Float64(field[i, j1]) + Float64(field[i, j2]))
end

function write_centerline_profile(path, result)
    geom = result.geometry
    j1, j2 = center_rows(result.Ny)
    H = Float64(geom.H_ref)
    open(path, "w") do io
        println(io, "x_over_H,ux,uy,tau_xx,tau_xy,tau_yy,is_solid")
        for i in 1:result.Nx
            x = (Float64(i) - Float64(geom.i_step)) / H
            solid = result.is_solid[i, j1] || result.is_solid[i, j2]
            vals = (
                x,
                center_value(result.ux, i, j1, j2),
                center_value(result.uy, i, j1, j2),
                center_value(result.tauxx, i, j1, j2),
                center_value(result.tauxy, i, j1, j2),
                center_value(result.tauyy, i, j1, j2),
                solid ? 1 : 0,
            )
            println(io, join(csv_cell.(vals), ","))
        end
    end
end

function write_throat_profile(path, result)
    geom = result.geometry
    i = min(result.Nx, geom.i_step + 1)
    H = Float64(geom.H_ref)
    y0 = 0.5 * (Float64(result.Ny) + 1.0)
    open(path, "w") do io
        println(io, "y_over_H,ux,uy,tau_xx,tau_xy,tau_yy,is_solid")
        for j in 1:result.Ny
            vals = (
                (Float64(j) - y0) / H,
                Float64(result.ux[i, j]),
                Float64(result.uy[i, j]),
                Float64(result.tauxx[i, j]),
                Float64(result.tauxy[i, j]),
                Float64(result.tauyy[i, j]),
                result.is_solid[i, j] ? 1 : 0,
            )
            println(io, join(csv_cell.(vals), ","))
        end
    end
end

function top_vortex_length(result)
    geom = result.geometry
    i0 = geom.i_step
    top_open = last(geom.inlet_open)
    j = max(1, min(result.Ny, top_open - 1))
    H = Float64(geom.H_ref)
    prev_i = i0 - 1
    prev_u = Float64(result.ux[prev_i, j])
    for i in (i0 - 2):-1:1
        result.is_solid[i, j] && continue
        u = Float64(result.ux[i, j])
        if prev_u * u < 0
            frac = abs(prev_u) / max(abs(prev_u - u), eps(Float64))
            x_cross = Float64(prev_i) - frac
            return (Float64(i0) - x_cross) / H
        end
        prev_i = i
        prev_u = u
    end
    return 0.0
end

function result_metrics(result, u_mean)
    j1, j2 = center_rows(result.Ny)
    center_u = [center_value(result.ux, i, j1, j2) for i in 1:result.Nx
                if !(result.is_solid[i, j1] || result.is_solid[i, j2])]
    center_txx = [center_value(result.tauxx, i, j1, j2) for i in 1:result.Nx
                  if !(result.is_solid[i, j1] || result.is_solid[i, j2])]
    i_throat = min(result.Nx, result.geometry.i_step + 1)
    throat_mask = .!result.is_solid[i_throat, :]
    throat_txx = Float64.(result.tauxx[i_throat, :])[throat_mask]
    return (;
        tau_xx_peak=isempty(center_txx) ? NaN : maximum(center_txx),
        centerline_u_peak=isempty(center_u) ? NaN : maximum(center_u),
        centerline_u_overshoot=isempty(center_u) ? NaN : maximum(center_u) / Float64(u_mean),
        throat_tau_xx_peak=isempty(throat_txx) ? NaN : maximum(throat_txx),
        vortex_length_top=top_vortex_length(result),
    )
end

function save_fields(path, result, row)
    fields = Dict{String,Matrix{Float64}}(
        "rho" => Float64.(result.rho),
        "ux" => Float64.(result.ux),
        "uy" => Float64.(result.uy),
        "solid" => Float64.(result.is_solid),
        "tau_xx" => Float64.(result.tauxx),
        "tau_xy" => Float64.(result.tauxy),
        "tau_yy" => Float64.(result.tauyy),
        "fx_poly" => Float64.(result.fx_poly),
        "fy_poly" => Float64.(result.fy_poly),
        "fx_total" => Float64.(result.fx_total),
        "fy_total" => Float64.(result.fy_total),
    )
    open(path, "w") do io
        serialize(io, (; metadata=Dict(row), fields))
    end
end

function rheotool_reference_links(case_dir::AbstractString)
    links = String[]
    isdir(case_dir) || return links
    for (root, _, files) in walkdir(case_dir)
        for f in files
            if occursin("centerline", f) || occursin("lAfterx0", f) ||
               occursin("Xr_", f) || occursin("Xl_", f)
                push!(links, joinpath(root, f))
            end
        end
    end
    sort!(links)
    return links
end

function html_escape(s::AbstractString)
    return replace(replace(replace(s, "&" => "&amp;"), "<" => "&lt;"), ">" => "&gt;")
end

function write_dashboard(path, rows, reference_links)
    open(path, "w") do io
        println(io, "<!doctype html><meta charset='utf-8'><title>Kraken contraction41 log-FV</title>")
        println(io, "<style>body{font-family:system-ui,sans-serif;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:4px 8px;text-align:right}td:first-child,th:first-child{text-align:left}</style>")
        println(io, "<h1>Axis-aligned 4:1 contraction log-FV comparison</h1>")
        println(io, "<table><tr><th>case</th><th>status</th><th>Hout</th><th>steps</th><th>substeps</th><th>min C eig</th><th>max speed</th><th>tau_xx peak</th><th>U peak/Umean</th><th>throat tau_xx</th><th>Xr top</th><th>centerline</th><th>throat</th></tr>")
        for row in rows
            @printf(io, "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%.4e</td><td>%.4e</td><td>%.4e</td><td>%.4e</td><td>%.4e</td><td>%.4e</td><td><a href='%s'>csv</a></td><td><a href='%s'>csv</a></td></tr>\n",
                html_escape(string(row[:case])), html_escape(string(row[:status])),
                row[:H_out], row[:steps], row[:polymer_substeps],
                Float64(get(row, :min_c_eig, NaN)),
                Float64(get(row, :max_speed, NaN)),
                Float64(get(row, :tau_xx_peak, NaN)),
                Float64(get(row, :centerline_u_overshoot, NaN)),
                Float64(get(row, :throat_tau_xx_peak, NaN)),
                Float64(get(row, :vortex_length_top, NaN)),
                basename(string(get(row, :profile_centerline, ""))),
                basename(string(get(row, :profile_throat, ""))))
        end
        println(io, "</table>")
        println(io, "<h2>RheoTool reference files</h2>")
        if isempty(reference_links)
            println(io, "<p>No sampled RheoTool reference files found yet. Run <code>bench/rheotool/contraction41_oldroydb_log/run_docker.sh</code>.</p>")
        else
            println(io, "<ul>")
            for link in reference_links
                println(io, "<li><code>", html_escape(link), "</code></li>")
            end
            println(io, "</ul>")
        end
    end
end

function write_verdict(path, rows, reference_links)
    open(path, "w") do io
        println(io, "# Contraction41 Axis-Aligned Replay — $(Dates.format(today(), "yyyy-mm-dd"))")
        println(io)
        println(io, "## Kraken Summary")
        println(io)
        println(io, "| case | status | H_out | steps | min C eig | tau_xx peak | U peak/Umean | Xr top |")
        println(io, "|---|---|---:|---:|---:|---:|---:|---:|")
        for row in rows
            @printf(io, "| %s | %s | %s | %s | %.4e | %.4e | %.4e | %.4e |\n",
                row[:case], row[:status], row[:H_out], row[:steps],
                Float64(get(row, :min_c_eig, NaN)),
                Float64(get(row, :tau_xx_peak, NaN)),
                Float64(get(row, :centerline_u_overshoot, NaN)),
                Float64(get(row, :vortex_length_top, NaN)))
        end
        println(io)
        println(io, "## RheoTool Reference")
        println(io)
        if isempty(reference_links)
            println(io, "RheoTool sampled reference files were not found in the project-local case yet.")
        else
            println(io, "Reference files found:")
            for link in reference_links
                println(io, "- `$(link)`")
            end
        end
        println(io)
        println(io, "## Verdict")
        println(io)
        if any(get(row, :status, "") != "ok" for row in rows)
            println(io, "At least one Kraken contraction run failed or diverged. Follow the decision tree: tighten polymer subcycling first, then isolate source/advection/force under the axis-aligned contraction.")
        elseif isempty(reference_links)
            println(io, "Kraken outputs were produced, but no RheoTool reference was available for the four-panel comparison yet.")
        else
            println(io, "Comparison artifacts are available. Fill in the quantitative RheoTool deltas from the sampled reference before promoting the cut-cell/cylinder verdict.")
        end
    end
end

function main()
    backend, backend_name = select_backend()
    FT = select_float_type(backend_name)
    out_dir = get(
        ENV,
        "KRAKEN_OUTPUT_DIR",
        joinpath("tmp", "contraction41_axis_aligned", Dates.format(now(), "yyyymmdd_HHMMSS")),
    )
    mkpath(out_dir)
    field_dir = joinpath(out_dir, "fields")
    mkpath(field_dir)

    H_out_values = parse_list(Int, get(ENV, "KRAKEN_CONTRACTION_H_OUT_LIST", "4,8,16"))
    β_c = parse(Int, get(ENV, "KRAKEN_CONTRACTION_RATIO", "4"))
    L_up = parse(Int, get(ENV, "KRAKEN_L_UP", "64"))
    L_down = parse(Int, get(ENV, "KRAKEN_L_DOWN", "64"))
    Re = parse(Float64, get(ENV, "KRAKEN_RE", "0.02"))
    Wi = parse(Float64, get(ENV, "KRAKEN_WI", "0.125"))
    beta = parse(Float64, get(ENV, "KRAKEN_BETA", "0.11111"))
    u_mean = parse(Float64, get(ENV, "KRAKEN_U_MEAN", "0.01"))
    max_steps = parse(Int, get(ENV, "KRAKEN_STEPS", "200"))
    avg_window = parse(Int, get(ENV, "KRAKEN_AVG_WINDOW", string(max(1, max_steps ÷ 4))))
    polymer_substeps = parse_polymer_substeps(get(ENV, "KRAKEN_POLYMER_SUBSTEPS", "auto"))
    max_polymer_substeps = parse(Int, get(ENV, "KRAKEN_MAX_POLYMER_SUBSTEPS", "256"))
    bsd_fraction = parse(Float64, get(ENV, "KRAKEN_LOGFV_BSD_FRACTION", "1.0"))
    diagnostic_stride = parse(Int, get(ENV, "KRAKEN_DIAGNOSTIC_STRIDE", "0"))
    rheotool_case = get(ENV, "KRAKEN_RHEOTOOL_CASE", DEFAULT_RHEOTOOL_CASE)

    summary_path = joinpath(out_dir, "summary.csv")
    rows = Dict{Symbol,Any}[]
    for H_out in H_out_values
        H_in = β_c * H_out
        nu_total = u_mean * H_out / Re
        nu_s = beta * nu_total
        nu_p = (1.0 - beta) * nu_total
        lambda = Wi * H_out / u_mean
        Nx = (L_up + L_down) * H_out
        Ny = H_in
        case = "kraken_Hout$(H_out)"
        row = Dict{Symbol,Any}(
            :case => case,
            :status => "pending",
            :error => "",
            :backend => backend_name,
            :float_type => FT,
            :H_out => H_out,
            :H_in => H_in,
            :Nx => Nx,
            :Ny => Ny,
            :L_up => L_up,
            :L_down => L_down,
            :beta => beta,
            :Re => Re,
            :Wi => Wi,
            :u_mean => u_mean,
            :nu_total => nu_total,
            :nu_s => nu_s,
            :nu_p => nu_p,
            :lambda => lambda,
            :steps => max_steps,
        )
        push!(rows, row)
        try
            guard_local_run!(backend_name, Nx, Ny, max_steps)
            @printf("Running %s Nx=%d Ny=%d steps=%d nu=%.6g lambda=%.6g\n",
                case, Nx, Ny, max_steps, nu_total, lambda)
            result = Kraken.run_viscoelastic_logfv_contraction_coupled_2d(;
                H_out,
                β_c,
                L_up,
                L_down,
                nu_s=FT(nu_s),
                nu_p=FT(nu_p),
                lambda=FT(lambda),
                u_mean=FT(u_mean),
                Fx_body=FT(0),
                bsd_fraction=FT(bsd_fraction),
                polymer_substeps,
                max_polymer_substeps,
                max_steps,
                avg_window,
                drag_stride=max(1, avg_window),
                diagnostic_stride,
                backend,
                T=FT,
            )
            metrics = result_metrics(result, u_mean)
            centerline_path = joinpath(out_dir, "$(case)_centerline.csv")
            throat_path = joinpath(out_dir, "$(case)_throat_x0.csv")
            fields_path = joinpath(field_dir, "$(case)_fields.jls")
            write_centerline_profile(centerline_path, result)
            write_throat_profile(throat_path, result)
            save_fields(fields_path, result, row)
            merge!(row, Dict{Symbol,Any}(
                :status => "ok",
                :completed_steps => result.completed_steps,
                :polymer_substeps => result.polymer_substeps,
                :raw_substeps => result.subcycle_estimate.raw_substeps,
                :substeps_clamped => result.subcycle_estimate.clamped,
                :min_c_eig => result.min_c_eig,
                :max_c_trace => result.max_c_trace,
                :rho_min => result.rho_min,
                :rho_max => result.rho_max,
                :max_speed => result.max_speed,
                :first_nonfinite_step => result.first_nonfinite_step,
                :profile_centerline => centerline_path,
                :profile_throat => throat_path,
                :field_dump => fields_path,
            ))
            merge!(row, pairs(metrics))
            if result.first_nonfinite_step != 0
                row[:status] = "nonfinite"
            end
        catch err
            row[:status] = "error"
            row[:error] = sprint(showerror, err)
            @warn "contraction run failed" case error=row[:error]
        end
        append_summary(summary_path, row)
    end

    reference_links = rheotool_reference_links(rheotool_case)
    dashboard = joinpath(out_dir, "dashboard.html")
    verdict = joinpath(out_dir, "CONTRACTION41_AXIS_ALIGNED_$(Dates.format(today(), "yyyymmdd")).md")
    write_dashboard(dashboard, rows, reference_links)
    write_verdict(verdict, rows, reference_links)

    println("Summary: ", summary_path)
    println("Dashboard: ", dashboard)
    println("Verdict: ", verdict)
end

main()
