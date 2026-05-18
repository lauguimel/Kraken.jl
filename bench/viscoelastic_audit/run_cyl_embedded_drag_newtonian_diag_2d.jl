#!/usr/bin/env julia
# Strict-Newtonian embedded-cylinder drag diagnostic for M26.
#
# Cases:
#   0000_qwall   embedded gradient/advection/force/drag off, q_wall geometry
#   0001_qwall   embedded drag on only, q_wall geometry
#   0000_circle  embedded flags off, circle FVFD geometry
#   1111_circle  embedded gradient/advection/force/drag on, circle FVFD geometry
#
# Isolates whether the suspected 1111_circle +8.8 Cd_s ghost-drag signature
# comes from embedded drag accounting, circle geometry lowering, or the
# embedded gradient/advection/force machinery.
#
# Env-var configurable:
#   KRAKEN_BACKEND     "cuda" | "metal" | "cpu"  (auto-selected if not set)
#   KRAKEN_FT          "float64" | "float32"     (Metal forces Float32)
#   KRAKEN_OUTPUT_DIR  "results/viscoelastic_audit/cyl_embedded_drag_newtonian_diag"
#
# Output:
#   $KRAKEN_OUTPUT_DIR/cyl_embedded_drag_newtonian_diag.csv
#   In --self-test mode, the default output dir is
#   bench/scratch/cyl_embedded_drag_newtonian_diag_selftest unless overridden.
#
# Self-test:
#   julia --project=. bench/viscoelastic_audit/run_cyl_embedded_drag_newtonian_diag_2d.jl --self-test
#   Runs R=20, L_up=L_down=4, max_steps=1000, avg_window=200 and asserts one
#   CSV header plus four data rows with finite walltime_s.

using Kraken
using Printf
using Dates
using KernelAbstractions

function detect_backend()
    req = lowercase(get(ENV, "KRAKEN_BACKEND", "auto"))
    ft  = lowercase(get(ENV, "KRAKEN_FT", "float64"))
    FT  = ft in ("f32", "float32", "single") ? Float32 : Float64

    if (req == "auto" || req == "cuda")
        try
            @eval using CUDA
            CUDAMod = getfield(Main, :CUDA)
            if Base.invokelatest(getfield(CUDAMod, :functional))
                return CUDAMod.CUDABackend(), "cuda", FT
            end
        catch end
    end
    if (req == "auto" || req == "metal") && Sys.isapple()
        try
            @eval using Metal
            MetalMod = getfield(Main, :Metal)
            if Base.invokelatest(getfield(MetalMod, :functional))
                return MetalMod.MetalBackend(), "metal", FT == Float64 ? Float32 : FT
            end
        catch end
    end
    if req != "auto" && req != "cpu"
        @warn "Requested KRAKEN_BACKEND=$req but detection failed; falling back to CPU."
    end
    return KernelAbstractions.CPU(), "cpu", FT
end

const SELF_TEST = "--self-test" in ARGS
const BACKEND, BACKEND_LABEL, FT = detect_backend()

const FULL_OUTPUT_DIR = "results/viscoelastic_audit/cyl_embedded_drag_newtonian_diag"
const SELF_TEST_OUTPUT_DIR = "bench/scratch/cyl_embedded_drag_newtonian_diag_selftest"
const OUTPUT_DIR = get(ENV, "KRAKEN_OUTPUT_DIR", SELF_TEST ? SELF_TEST_OUTPUT_DIR : FULL_OUTPUT_DIR)
const CSV_PATH = joinpath(OUTPUT_DIR, "cyl_embedded_drag_newtonian_diag.csv")

const CSV_COLUMNS = [
    :timestamp, :backend, :FT, :case_label,
    :embedded_gradient, :embedded_advection, :embedded_force, :embedded_drag,
    :embedded_geometry, :R, :L_up, :L_down, :nu_s, :nu_p, :lambda, :max_steps,
    :avg_window, :completed_steps, :Cd_kraken, :Cd_s, :Cd_p, :Cd_bsd,
    :u_max_abs, :walltime_s, :nan_step, :nan_field,
]

const CASES = [
    (label="0000_qwall",  embedded_gradient=false, embedded_advection=false,
     embedded_force=false, embedded_drag=false, embedded_geometry=:qwall),
    (label="0001_qwall",  embedded_gradient=false, embedded_advection=false,
     embedded_force=false, embedded_drag=true, embedded_geometry=:qwall),
    (label="0000_circle", embedded_gradient=false, embedded_advection=false,
     embedded_force=false, embedded_drag=false, embedded_geometry=:circle),
    (label="1111_circle", embedded_gradient=true, embedded_advection=true,
     embedded_force=true, embedded_drag=true, embedded_geometry=:circle),
]

function csv_cell(x)
    x === nothing && return ""
    if x isa AbstractFloat
        isnan(x) && return "NaN"
        return @sprintf("%.16g", Float64(x))
    end
    s = string(x)
    if any(c in s for c in (',', '"', '\n'))
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function write_rows_csv(path, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(string.(CSV_COLUMNS), ","))
        for row in rows
            println(io, join((csv_cell(get(row, c, "")) for c in CSV_COLUMNS), ","))
        end
    end
end

function base_row(case_cfg, R, L_up, L_down, nu_s, nu_p, lambda, max_steps, avg_window)
    row = Dict{Symbol,Any}(
        :timestamp => string(now()),
        :backend => BACKEND_LABEL,
        :FT => string(FT),
        :case_label => case_cfg.label,
        :embedded_gradient => Int(case_cfg.embedded_gradient),
        :embedded_advection => Int(case_cfg.embedded_advection),
        :embedded_force => Int(case_cfg.embedded_force),
        :embedded_drag => Int(case_cfg.embedded_drag),
        :embedded_geometry => string(case_cfg.embedded_geometry),
        :R => R,
        :L_up => L_up,
        :L_down => L_down,
        :nu_s => nu_s,
        :nu_p => nu_p,
        :lambda => lambda,
        :max_steps => max_steps,
        :avg_window => avg_window,
        :completed_steps => 0,
        :walltime_s => NaN,
        :nan_step => 0,
        :nan_field => "none",
    )
    for c in CSV_COLUMNS
        haskey(row, c) || (row[c] = NaN)
    end
    return row
end

function copy_result_fields!(row, result, max_steps; self_test::Bool)
    row[:completed_steps] = result.completed_steps
    row[:u_max_abs] = Float64(result.max_speed)
    row[:nan_step] = result.first_nonfinite_step
    row[:nan_field] = string(result.first_nonfinite_field)

    trusted_cd = true
    if !self_test
        completed_full = result.completed_steps >= max_steps
        late_nonfinite = result.first_nonfinite_step > 0 &&
            result.first_nonfinite_step > 0.5 * max_steps
        trusted_cd = completed_full || late_nonfinite
        if !trusted_cd
            row[:nan_step] = result.first_nonfinite_step
            row[:nan_field] = string(result.first_nonfinite_field)
        end
    end

    if trusted_cd
        row[:Cd_kraken] = Float64(result.Cd)
        row[:Cd_s] = Float64(result.Cd_s)
        row[:Cd_p] = Float64(result.Cd_p)
        row[:Cd_bsd] = Float64(result.Cd_bsd)
    end
    return row
end

function run_case(case_cfg; R, L_up, L_down, max_steps, avg_window)
    u_mean = 0.005
    re_target = 1.0
    nu_p = 0.0
    nu_s = u_mean * R / re_target
    lambda = 1.0
    bsd_fraction = 1.0
    H = 4 * R
    row = base_row(case_cfg, R, L_up, L_down, nu_s, nu_p, lambda, max_steps, avg_window)

    @printf("[%s] starting %s backend=%s FT=%s R=%d max_steps=%d\n",
        string(now()), case_cfg.label, BACKEND_LABEL, string(FT), R, max_steps)
    flush(stdout)

    t0 = time()
    result = nothing
    status = :ok
    try
        result = Kraken.run_viscoelastic_logfv_cylinder_coupled_2d(;
            radius=R, H=H, L_up=L_up, L_down=L_down,
            nu_s=FT(nu_s), nu_p=FT(nu_p), lambda=FT(lambda),
            polymer_model=:oldroydb, L_max=FT(10.0),
            u_mean=FT(u_mean), Fx_body=FT(0.0),
            bsd_fraction=FT(bsd_fraction), polymer_substeps=:auto,
            subcycle_relative_tolerance=FT(0.01),
            max_deformation_increment=FT(0.05),
            max_memory_deformation_increment=FT(0.07),
            max_polymer_substeps=64, max_steps=max_steps,
            avg_window=avg_window, drag_stride=200, diagnostic_stride=0,
            embedded_geometry=case_cfg.embedded_geometry,
            embedded_gradient=case_cfg.embedded_gradient,
            embedded_advection=case_cfg.embedded_advection,
            embedded_force=case_cfg.embedded_force,
            embedded_drag=case_cfg.embedded_drag,
            embedded_circle_samples=32, force_boundary_fill=:bc_aware,
            backend=BACKEND, T=FT,
        )
    catch err
        if err isa DomainError || err isa BoundsError
            status = :spd_or_bounds
        else
            rethrow(err)
        end
    end
    row[:walltime_s] = time() - t0

    if status === :ok && result !== nothing
        copy_result_fields!(row, result, max_steps; self_test=SELF_TEST)
    else
        row[:completed_steps] = 0
        row[:nan_step] = 0
        row[:nan_field] = "spd_or_bounds"
    end

    @printf("[%s] finished %s Cd_s=%s completed=%s nan_step=%s walltime=%.3fs\n",
        string(now()),
        case_cfg.label,
        csv_cell(row[:Cd_s]),
        string(row[:completed_steps]),
        string(row[:nan_step]),
        Float64(row[:walltime_s]))
    flush(stdout)
    return row
end

function validate_self_test_csv(path)
    lines = readlines(path)
    @assert length(lines) == 1 + length(CASES) "expected one header plus four data rows"
    header = split(lines[1], ",")
    @assert header == string.(CSV_COLUMNS) "unexpected CSV header"
    label_idx = findfirst(==("case_label"), header)
    walltime_idx = findfirst(==("walltime_s"), header)
    labels = String[]
    for line in lines[2:end]
        fields = split(line, ",")
        push!(labels, fields[label_idx])
        @assert isfinite(parse(Float64, fields[walltime_idx])) "non-finite walltime_s"
    end
    expected = [case.label for case in CASES]
    @assert sort(labels) == sort(expected) "unexpected case labels"
    println("[M26-impl self-test PASS] " * join(labels, ","))
end

function main()
    R = SELF_TEST ? 20 : 30
    L_up = 4.0
    L_down = 4.0
    max_steps = SELF_TEST ? 1000 : 100000
    avg_window = SELF_TEST ? 200 : round(Int, 0.2 * max_steps)

    println("=== cyl_embedded_drag_newtonian_diag backend=$(BACKEND_LABEL) FT=$(FT) ===")
    println("self_test=$(SELF_TEST) R=$(R) L_up=$(L_up) L_down=$(L_down)")
    println("max_steps=$(max_steps) avg_window=$(avg_window)")
    println("output=$(CSV_PATH)")
    flush(stdout)

    rows = Dict{Symbol,Any}[]
    for case in CASES
        push!(rows, run_case(case; R, L_up, L_down, max_steps, avg_window))
        write_rows_csv(CSV_PATH, rows)
    end

    if SELF_TEST
        validate_self_test_csv(CSV_PATH)
    end
    println("=== cyl_embedded_drag_newtonian_diag DONE at $(now()) ===")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
