#!/usr/bin/env julia

using Dates
using KernelAbstractions
using Kraken
using Printf

function _arg_value(argv, name, default)
    prefix = name * "="
    for arg in argv
        startswith(arg, prefix) && return split(arg, "="; limit=2)[2]
    end
    return default
end

function _parse_grid(s)
    parts = split(lowercase(String(s)), 'x')
    length(parts) == 2 || throw(ArgumentError("grid must be Nx x Ny, e.g. 197x321"))
    return parse(Int, parts[1]), parse(Int, parts[2])
end

function _parse_T_list(s)
    return [parse(Float64, x) for x in split(String(s), ',') if !isempty(strip(x))]
end

function _parse_symbol(s)
    return Symbol(lowercase(String(s)))
end

function _cuda_backend()
    # The functional() check lives INSIDE the @eval and the backend is built via
    # invokelatest: referencing CUDA from this function's original world age
    # throws UndefVarError ("binding may be too new").
    try
        @eval Main begin
            using CUDA, CUDSS
            CUDA.functional() || error("CUDA requested but CUDA.functional() is false")
        end
    catch err
        error("CUDA sweep requested but CUDA/CUDSS could not be loaded/functional. Co-install sibling weakdeps as noted in docs/agent/solve-linear-implication.md. Original error: $(sprint(showerror, err))")
    end
    return Base.invokelatest(() -> Main.CUDA.CUDABackend())
end

function _growth_series(steps, speeds)
    growth = similar(speeds, Float64)
    first_positive = findfirst(>(0.0), speeds)
    if first_positive === nothing
        fill!(growth, 0.0)
        return growth
    end
    s0 = steps[first_positive]
    u0 = max(speeds[first_positive], floatmin(Float64))
    for k in eachindex(speeds)
        if steps[k] == s0
            growth[k] = 0.0
        else
            growth[k] = log(max(speeds[k], floatmin(Float64)) / u0) / (steps[k] - s0)
        end
    end
    return growth
end

function _write_history(path, steps, speeds)
    growth = _growth_series(steps, speeds)
    open(path, "w") do io
        println(io, "step,max_abs_u,growth_rate_estimate")
        for k in eachindex(steps)
            @printf(io, "%d,%.12g,%.12g\n", steps[k], speeds[k], growth[k])
        end
    end
    return isempty(growth) ? 0.0 : growth[end]
end

function _write_summary(csv_path, md_path, rows)
    open(csv_path, "w") do io
        println(io, "T,ns_scheme,charge_scheme,phi_scheme,steps,ms_per_step,final_max_abs_u,growth_rate_estimate,history_csv")
        for row in rows
            @printf(io, "%.12g,%s,%s,%s,%d,%.12g,%.12g,%.12g,%s\n",
                    row.T, row.ns_scheme, row.charge_scheme, row.phi_scheme, row.steps,
                    row.ms_per_step, row.final_max_abs_u, row.growth_rate_estimate,
                    row.history_csv)
        end
    end
    open(md_path, "w") do io
        println(io, "# EHD Tc Sweep")
        println(io)
        println(io, "| T | NS | charge | phi | steps | ms/step | final max|u| | growth estimate |")
        println(io, "|---:|:---|:---|:---|---:|---:|---:|---:|")
        for row in rows
            @printf(io, "| %.6g | %s | %s | %s | %d | %.6g | %.6g | %.6g |\n",
                    row.T, row.ns_scheme, row.charge_scheme, row.phi_scheme, row.steps,
                    row.ms_per_step, row.final_max_abs_u, row.growth_rate_estimate)
        end
    end
end

function main(argv=ARGS)
    if "--help" in argv || "-h" in argv
        println("usage: julia --project=. benchmarks/ehd/tc_sweep.jl [--smoke] [--gpu] [--grid=197x321] [--T=150,160,163.5,165,170,190] [--cycles=50000] [--ns-scheme=mrt] [--charge-scheme=regularized] [--phi-scheme=lbm|direct] [--phi-substeps=1|auto] [--phi-tol=1e-4] [--phi-max-iter=10000]")
        return nothing
    end

    smoke = "--smoke" in argv
    use_gpu = "--gpu" in argv && !smoke

    Nx, Ny = smoke ? (31, 48) : _parse_grid(_arg_value(argv, "--grid", "197x321"))
    T_values = smoke ? [150.0, 220.0] : _parse_T_list(_arg_value(argv, "--T", "150,160,163.5,165,170,190"))
    cycles = parse(Int, _arg_value(argv, "--cycles", smoke ? "500" : "50000"))
    ns_scheme = smoke ? :mrt : _parse_symbol(_arg_value(argv, "--ns-scheme", "mrt"))
    charge_scheme = _parse_symbol(_arg_value(argv, "--charge-scheme", "regularized"))
    phi_scheme = _parse_symbol(_arg_value(argv, "--phi-scheme", "lbm"))
    phi_substeps_arg = _arg_value(argv, "--phi-substeps", "1")
    phi_substeps = lowercase(String(phi_substeps_arg)) == "auto" ? nothing : parse(Int, phi_substeps_arg)
    phi_tol = parse(Float64, _arg_value(argv, "--phi-tol", "1e-4"))
    phi_max_iter = parse(Int, _arg_value(argv, "--phi-max-iter", "10000"))
    history_interval = parse(Int, _arg_value(argv, "--history-interval", smoke ? "10" : "100"))
    force_projection = _parse_symbol(_arg_value(argv, "--force-projection", "none"))
    outdir = String(_arg_value(argv, "--output-dir", joinpath(dirname(@__DIR__), "results", "ehd")))
    mkpath(outdir)

    backend = use_gpu ? _cuda_backend() : KernelAbstractions.CPU()
    tag = Dates.format(now(), "yyyymmdd_HHMMSS")
    rows = NamedTuple[]

    for T_ehd in T_values
        result = Kraken.run_electroconvection_2d(;
            Nx=Nx, Ny=Ny, T=T_ehd, max_cycles=cycles,
            ns_scheme=ns_scheme, charge_scheme=charge_scheme,
            phi_scheme=phi_scheme, phi_substeps=phi_substeps,
            phi_tol=phi_tol, phi_max_iter=phi_max_iter,
            history_interval=history_interval, force_projection=force_projection,
            backend=backend, FT=Float64)

        T_tag = replace(@sprintf("%.6g", T_ehd), "." => "p", "-" => "m")
        history_name = "tc_sweep_T$(T_tag)_$(ns_scheme)_$(tag).csv"
        history_path = joinpath(outdir, history_name)
        growth = _write_history(history_path, result.cycle_history, result.umax_history)
        push!(rows, (T=T_ehd, ns_scheme=String(ns_scheme), charge_scheme=String(charge_scheme),
                     phi_scheme=String(phi_scheme),
                     steps=result.steps, ms_per_step=result.loop_ms_per_step,
                     final_max_abs_u=last(result.umax_history),
                     growth_rate_estimate=growth, history_csv=history_name))
        @info "EHD Tc sweep case complete" T=T_ehd ns_scheme steps=result.steps ms_per_step=result.loop_ms_per_step maxu=last(result.umax_history)
    end

    summary_csv = joinpath(outdir, "tc_sweep_summary_$(tag).csv")
    summary_md = joinpath(outdir, "tc_sweep_summary_$(tag).md")
    _write_summary(summary_csv, summary_md, rows)
    println("summary_csv=$(summary_csv)")
    println("summary_md=$(summary_md)")
end

main()
