#!/usr/bin/env julia

using Dates
using KernelAbstractions
using Printf

using Kraken

const _METAL_BACKEND_STATUS = if Sys.isapple()
    try
        @eval using Metal
        Metal.functional() ? (:Metal, Metal.MetalBackend(), Float32, nothing) :
            (:none, nothing, Float32, "Metal is not functional")
    catch err
        (:none, nothing, Float32, err)
    end
else
    (:none, nothing, Float32, "Metal is only supported on macOS")
end

const _CUDA_BACKEND_STATUS = try
    @eval using CUDA
    if Base.invokelatest(CUDA.functional)
        backend = Base.invokelatest(CUDA.CUDABackend)
        (:CUDA, backend, Float64, nothing)
    else
        (:none, nothing, Float64, "CUDA is not functional")
    end
catch err
    (:none, nothing, Float64, err)
end

function _backend_from_env()
    requested = lowercase(get(ENV, "KRAKEN_BACKEND", "cpu"))
    if requested == "cuda"
        cuda_name, cuda_backend, cuda_ft, cuda_error = _CUDA_BACKEND_STATUS
        if cuda_backend !== nothing
            return (cuda_name, cuda_backend, cuda_ft)
        end
        @warn "CUDA requested but unavailable; falling back to CPU" reason=cuda_error
    elseif requested == "metal"
        metal_name, metal_backend, metal_ft, metal_error = _METAL_BACKEND_STATUS
        if metal_backend !== nothing
            return (metal_name, metal_backend, metal_ft)
        end
        @warn "Metal requested but unavailable; falling back to CPU" reason=metal_error
    elseif requested != "cpu"
        @warn "Unsupported KRAKEN_BACKEND=$(requested); expected cpu, metal, or cuda; falling back to CPU"
    end
    return (:CPU, KernelAbstractions.CPU(), Float64)
end

function _parse_resolutions()
    raw = get(ENV, "KRAKEN_SIMPLE_LADDER_NY", "16,32,64")
    values = Int[]
    for part in split(raw, ',')
        text = strip(part)
        isempty(text) && continue
        push!(values, parse(Int, text))
    end
    isempty(values) && throw(ArgumentError("KRAKEN_SIMPLE_LADDER_NY produced no resolutions"))
    return values
end

function _csv_cell(x)
    x === nothing && return ""
    s = string(x)
    if occursin(',', s) || occursin('"', s) || occursin('\n', s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function _write_summary(path, rows)
    columns = (
        :case,
        :backend,
        :T,
        :flow,
        :Nx,
        :Ny,
        :initial,
        :steps,
        :polymer_substeps,
        :lambda,
        :prefactor,
        :bsd_fraction,
        :max_c_error,
        :max_psi_error,
        :max_tau_error,
        :max_poly_force_error,
        :max_total_force_error,
        :max_transverse_force,
        :min_c_eig,
        :profile_csv,
    )
    open(path, "w") do io
        println(io, join(string.(columns), ","))
        for row in rows
            println(io, join((_csv_cell(get(row, col, "")) for col in columns), ","))
        end
    end
end

function _write_profile(path, result)
    i = cld(result.Nx, 2)
    open(path, "w") do io
        println(io, "y,ux,uy,cxx,cxy,cyy,cxx_ref,cxy_ref,cyy_ref,tauxx,tauxy,tauyy,tauxx_ref,tauxy_ref,tauyy_ref,fx_poly,fy_poly,fx_total,fy_total")
        for j in 1:result.Ny
            y = (j - 0.5) * result.dy
            values = (
                y,
                result.ux[i, j],
                result.uy[i, j],
                result.cxx[i, j],
                result.cxy[i, j],
                result.cyy[i, j],
                result.reference.cxx[i, j],
                result.reference.cxy[i, j],
                result.reference.cyy[i, j],
                result.tauxx[i, j],
                result.tauxy[i, j],
                result.tauyy[i, j],
                result.reference.tauxx[i, j],
                result.reference.tauxy[i, j],
                result.reference.tauyy[i, j],
                result.fx_poly[i, j],
                result.fy_poly[i, j],
                result.fx_total[i, j],
                result.fy_total[i, j],
            )
            println(io, join((_csv_cell(v) for v in values), ","))
        end
    end
end

function _write_dashboard(path, rows)
    open(path, "w") do io
        println(io, "<!doctype html><meta charset=\"utf-8\"><title>Kraken log-FV simple ladder</title>")
        println(io, "<style>body{font-family:system-ui,sans-serif;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px;text-align:right}th:first-child,td:first-child{text-align:left}</style>")
        println(io, "<h1>Kraken log-FV quantitative simple ladder</h1>")
        println(io, "<table>")
        println(io, "<tr><th>case</th><th>flow</th><th>Nx</th><th>Ny</th><th>substeps</th><th>max C err</th><th>max tau err</th><th>max force err</th><th>min eig C</th><th>profile</th></tr>")
        for row in rows
            println(io, "<tr>",
                "<td>", row[:case], "</td>",
                "<td>", row[:flow], "</td>",
                "<td>", row[:Nx], "</td>",
                "<td>", row[:Ny], "</td>",
                "<td>", row[:polymer_substeps], "</td>",
                "<td>", @sprintf("%.6e", row[:max_c_error]), "</td>",
                "<td>", @sprintf("%.6e", row[:max_tau_error]), "</td>",
                "<td>", @sprintf("%.6e", row[:max_total_force_error]), "</td>",
                "<td>", @sprintf("%.6e", row[:min_c_eig]), "</td>",
                "<td><a href=\"", basename(row[:profile_csv]), "\">csv</a></td>",
                "</tr>")
        end
        println(io, "</table>")
    end
end

function main()
    backend_name, backend, FT = _backend_from_env()
    resolutions = _parse_resolutions()
    steps = parse(Int, get(ENV, "KRAKEN_SIMPLE_LADDER_STEPS", "1"))
    substeps = parse(Int, get(ENV, "KRAKEN_SIMPLE_LADDER_SUBSTEPS", "128"))
    output_root = get(
        ENV,
        "KRAKEN_OUTPUT_DIR",
        joinpath("tmp", "logfv_quantitative_simple_ladder", Dates.format(now(), "yyyymmdd_HHMMSS")),
    )
    mkpath(output_root)

    rows = Dict{Symbol,Any}[]
    for flow in (:couette, :poiseuille), Ny in resolutions
        Nx = max(8, Ny ÷ 2)
        result = Kraken.run_viscoelastic_logfv_frozen_channel_cde_2d(;
            Nx,
            Ny,
            flow,
            height=1.0,
            width=1.0,
            umax=0.02,
            uwall=0.02,
            lambda=2.0,
            prefactor=0.03,
            bsd_fraction=1.0,
            initial=:steady,
            max_steps=steps,
            polymer_substeps=substeps,
            backend,
            T=FT,
        )
        case_name = string(flow, "_Ny", Ny)
        profile_csv = joinpath(output_root, case_name * "_profile.csv")
        _write_profile(profile_csv, result)
        row = Dict{Symbol,Any}(
            :case => case_name,
            :backend => backend_name,
            :T => FT,
            :flow => flow,
            :Nx => result.Nx,
            :Ny => result.Ny,
            :initial => result.initial,
            :steps => result.max_steps,
            :polymer_substeps => result.polymer_substeps,
            :lambda => result.lambda,
            :prefactor => result.prefactor,
            :bsd_fraction => result.bsd_fraction,
            :max_c_error => result.max_c_error,
            :max_psi_error => result.max_psi_error,
            :max_tau_error => result.max_tau_error,
            :max_poly_force_error => result.max_poly_force_error,
            :max_total_force_error => result.max_total_force_error,
            :max_transverse_force => result.max_transverse_force,
            :min_c_eig => result.min_c_eig,
            :profile_csv => profile_csv,
        )
        push!(rows, row)
        @printf(
            "%-16s %-8s Nx=%3d Ny=%3d sub=%3d maxC=% .6e maxTau=% .6e maxForce=% .6e minC=% .6e\n",
            case_name,
            string(backend_name),
            result.Nx,
            result.Ny,
            result.polymer_substeps,
            result.max_c_error,
            result.max_tau_error,
            result.max_total_force_error,
            result.min_c_eig,
        )
    end

    summary_csv = joinpath(output_root, "summary.csv")
    dashboard = joinpath(output_root, "dashboard.html")
    _write_summary(summary_csv, rows)
    _write_dashboard(dashboard, rows)
    println("Output directory: ", output_root)
    println("Summary CSV: ", summary_csv)
    println("Dashboard: ", dashboard)
end

main()
