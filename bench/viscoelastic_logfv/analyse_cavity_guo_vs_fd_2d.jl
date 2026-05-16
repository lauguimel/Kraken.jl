#!/usr/bin/env julia

using Serialization
using DelimitedFiles
using Printf
using Statistics
using Test

const FIELDS_JLS = "fields.jls"
const SUMMARY_CSV = "summary.csv"
const DIFF_MAP_CSV = "guo_vs_fd_diff_map.csv"

const EXPECTED_FIELD_NAMES = (
    :step,
    :t_phys,
    :ux,
    :uy,
    :psixx,
    :psixy,
    :psiyy,
    :tauxx,
    :tauxy,
    :tauyy,
)

struct AuditResult
    leaf_dir::String
    N::Int
    nu_p::Float64
    bsd_fraction::Float64
    l2_fd::Float64
    l2_guo::Float64
    l2_delta::Float64
    relative_l2_delta::Float64
    max_abs_delta::Float64
    max_i::Int
    max_j::Int
    l2_laplacian_u::Float64
    mean_abs_delta::Float64
    diff_csv::String
end

function find_leaf_dir(path::AbstractString)
    root = abspath(path)
    isdir(root) || error("not a directory: $(path)")
    if isfile(joinpath(root, FIELDS_JLS)) && isfile(joinpath(root, SUMMARY_CSV))
        return root
    end

    candidates = String[]
    for child in readdir(root; join=true)
        isdir(child) || continue
        if isfile(joinpath(child, FIELDS_JLS)) && isfile(joinpath(child, SUMMARY_CSV))
            push!(candidates, abspath(child))
        end
    end
    isempty(candidates) && error("no one-level case directory with fields.jls and summary.csv under $(path)")
    length(candidates) == 1 || error("expected one case directory under $(path), found $(length(candidates))")
    return only(candidates)
end

function display_path(path::AbstractString)
    rel = relpath(path, pwd())
    return startswith(rel, "..") ? abspath(path) : rel
end

function read_summary(path::AbstractString)
    isfile(path) || error("missing CSV: $(path)")
    data = readdlm(path, ',', String)
    size(data, 2) >= 2 || error("expected key,value columns in $(path)")
    start_row = lowercase(strip(data[1, 1])) == "key" ? 2 : 1
    values = Dict{String,String}()
    for row in start_row:size(data, 1)
        key = strip(data[row, 1])
        isempty(key) && continue
        values[key] = strip(data[row, 2])
    end
    return values
end

function parse_bsd_fraction_from_runner_default()
    runner_path = joinpath(@__DIR__, "run_cavity_remismatch_sweep.pbs")
    isfile(runner_path) || return nothing
    for line in eachline(runner_path)
        matched = match(r"KRAKEN_BSD_FRACTION:-([0-9eE+\-.]+)", line)
        matched === nothing && continue
        parsed = tryparse(Float64, matched.captures[1])
        parsed === nothing || return parsed
    end
    return nothing
end

function summary_float(values::Dict{String,String}, key::AbstractString, summary_path::AbstractString)
    if haskey(values, key)
        parsed = tryparse(Float64, values[key])
        parsed === nothing && error("summary key $(key) is not a Float64 in $(summary_path): $(values[key])")
        return parsed
    end

    if key == "bsd_fraction"
        env_value = get(ENV, "KRAKEN_BSD_FRACTION", "")
        env_parsed = tryparse(Float64, env_value)
        if env_parsed !== nothing
            println(stderr, "summary.csv missing bsd_fraction; using KRAKEN_BSD_FRACTION from the environment")
            return env_parsed
        end

        runner_default = parse_bsd_fraction_from_runner_default()
        if runner_default !== nothing
            println(stderr, "summary.csv missing bsd_fraction; using KRAKEN_BSD_FRACTION default from run_cavity_remismatch_sweep.pbs")
            return runner_default
        end
    end

    error("missing required summary key $(key) in $(summary_path)")
end

function field_layout_error(payload)
    found = payload isa NamedTuple ? Tuple(propertynames(payload)) : Symbol[]
    expected_text = join(string.(EXPECTED_FIELD_NAMES), ", ")
    found_text = payload isa NamedTuple ? join(string.(found), ", ") : "not a NamedTuple ($(typeof(payload)))"
    return error("""
fields.jls layout mismatch.
expected fields: $(expected_text)
found fields   : $(found_text)
minimum additional fields for a true f-based Guo accumulator audit:
  f::Array{T,3} (Nx, Ny, 9)
  fx_total::Matrix{T} (Nx, Ny)
  fy_total::Matrix{T} (Nx, Ny)
""")
end

function load_fields(path::AbstractString)
    isfile(path) || error("missing fields snapshot: $(path)")
    payload = open(path, "r") do io
        deserialize(io)
    end
    payload isa NamedTuple || field_layout_error(payload)
    Tuple(propertynames(payload)) == EXPECTED_FIELD_NAMES || field_layout_error(payload)
    payload.step isa Int64 || error("fields.step must be Int64; found $(typeof(payload.step))")
    payload.t_phys isa Float64 || error("fields.t_phys must be Float64; found $(typeof(payload.t_phys))")

    matrix_names = (:ux, :uy, :psixx, :psixy, :psiyy, :tauxx, :tauxy, :tauyy)
    first_size = size(payload.ux)
    length(first_size) == 2 || error("fields.ux must be a matrix")
    first_size[1] >= 3 && first_size[2] >= 3 || error("snapshot grid must be at least 3x3")
    for name in matrix_names
        value = getproperty(payload, name)
        value isa Matrix{Float64} || error("fields.$(name) must be Matrix{Float64}; found $(typeof(value))")
        size(value) == first_size || error("fields.$(name) size $(size(value)) does not match ux size $(first_size)")
    end
    return payload
end

@inline function derivative_x(field, i::Int, j::Int, Nx::Int)
    if 1 < i < Nx
        return (field[i + 1, j] - field[i - 1, j]) * 0.5
    elseif i == 1
        return Nx >= 3 ? (-3.0 * field[i, j] + 4.0 * field[i + 1, j] - field[i + 2, j]) * 0.5 :
               field[i + 1, j] - field[i, j]
    else
        return Nx >= 3 ? (3.0 * field[i, j] - 4.0 * field[i - 1, j] + field[i - 2, j]) * 0.5 :
               field[i, j] - field[i - 1, j]
    end
end

@inline function derivative_y(field, i::Int, j::Int, Ny::Int)
    if 1 < j < Ny
        return (field[i, j + 1] - field[i, j - 1]) * 0.5
    elseif j == 1
        return Ny >= 3 ? (-3.0 * field[i, j] + 4.0 * field[i, j + 1] - field[i, j + 2]) * 0.5 :
               field[i, j + 1] - field[i, j]
    else
        return Ny >= 3 ? (3.0 * field[i, j] - 4.0 * field[i, j - 1] + field[i, j - 2]) * 0.5 :
               field[i, j] - field[i, j - 1]
    end
end

@inline function second_derivative_x(field, i::Int, j::Int, Nx::Int)
    if 1 < i < Nx
        return field[i + 1, j] - 2.0 * field[i, j] + field[i - 1, j]
    elseif i == 1 && Nx >= 3
        return field[i, j] - 2.0 * field[i + 1, j] + field[i + 2, j]
    elseif i == Nx && Nx >= 3
        return field[i, j] - 2.0 * field[i - 1, j] + field[i - 2, j]
    else
        return 0.0
    end
end

@inline function second_derivative_y(field, i::Int, j::Int, Ny::Int)
    if 1 < j < Ny
        return field[i, j + 1] - 2.0 * field[i, j] + field[i, j - 1]
    elseif j == 1 && Ny >= 3
        return field[i, j] - 2.0 * field[i, j + 1] + field[i, j + 2]
    elseif j == Ny && Ny >= 3
        return field[i, j] - 2.0 * field[i, j - 1] + field[i, j - 2]
    else
        return 0.0
    end
end

function compute_force_fields(fields, nu_p::Float64, bsd_fraction::Float64)
    Nx, Ny = size(fields.ux)
    F_FD_x = zeros(Float64, Nx, Ny)
    F_FD_y = zeros(Float64, Nx, Ny)
    F_Guo_x = zeros(Float64, Nx, Ny)
    F_Guo_y = zeros(Float64, Nx, Ny)
    lap_ux = zeros(Float64, Nx, Ny)
    lap_uy = zeros(Float64, Nx, Ny)
    zeta_nu_p = bsd_fraction * nu_p

    for j in 1:Ny, i in 1:Nx
        F_FD_x[i, j] = derivative_x(fields.tauxx, i, j, Nx) + derivative_y(fields.tauxy, i, j, Ny)
        F_FD_y[i, j] = derivative_x(fields.tauxy, i, j, Nx) + derivative_y(fields.tauyy, i, j, Ny)
        lap_ux[i, j] = second_derivative_x(fields.ux, i, j, Nx) + second_derivative_y(fields.ux, i, j, Ny)
        lap_uy[i, j] = second_derivative_x(fields.uy, i, j, Nx) + second_derivative_y(fields.uy, i, j, Ny)
        F_Guo_x[i, j] = F_FD_x[i, j] - zeta_nu_p * lap_ux[i, j]
        F_Guo_y[i, j] = F_FD_y[i, j] - zeta_nu_p * lap_uy[i, j]
    end

    return F_FD_x, F_FD_y, F_Guo_x, F_Guo_y, lap_ux, lap_uy
end

function write_diff_map(path::AbstractString, F_FD_x, F_FD_y, F_Guo_x, F_Guo_y)
    Nx, Ny = size(F_FD_x)
    open(path, "w") do io
        write(io, "i,j,F_FD_x,F_FD_y,F_Guo_x,F_Guo_y,dFx,dFy\n")
        for j in 2:(Ny - 1), i in 2:(Nx - 1)
            dFx = F_FD_x[i, j] - F_Guo_x[i, j]
            dFy = F_FD_y[i, j] - F_Guo_y[i, j]
            @printf(io, "%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                    i, j, F_FD_x[i, j], F_FD_y[i, j], F_Guo_x[i, j], F_Guo_y[i, j], dFx, dFy)
        end
    end
    return path
end

function reduce_interior(leaf_dir::AbstractString, nu_p::Float64, bsd_fraction::Float64,
                         F_FD_x, F_FD_y, F_Guo_x, F_Guo_y, lap_ux, lap_uy)
    Nx, Ny = size(F_FD_x)
    fd_sum = 0.0
    guo_sum = 0.0
    delta_sum = 0.0
    lap_sum = 0.0
    abs_delta_sum = 0.0
    max_abs_delta = -Inf
    max_i = 0
    max_j = 0
    count = 0

    for j in 2:(Ny - 1), i in 2:(Nx - 1)
        fd2 = F_FD_x[i, j]^2 + F_FD_y[i, j]^2
        guo2 = F_Guo_x[i, j]^2 + F_Guo_y[i, j]^2
        dFx = F_FD_x[i, j] - F_Guo_x[i, j]
        dFy = F_FD_y[i, j] - F_Guo_y[i, j]
        delta2 = dFx^2 + dFy^2
        lap2 = lap_ux[i, j]^2 + lap_uy[i, j]^2
        delta = sqrt(delta2)

        fd_sum += fd2
        guo_sum += guo2
        delta_sum += delta2
        lap_sum += lap2
        abs_delta_sum += delta
        count += 1
        if delta > max_abs_delta
            max_abs_delta = delta
            max_i = i
            max_j = j
        end
    end

    l2_fd = sqrt(fd_sum)
    l2_fd > 0.0 || error("F_FD interior L2 norm is zero; cannot compute relative difference")
    l2_guo = sqrt(guo_sum)
    l2_delta = sqrt(delta_sum)
    relative_l2_delta = l2_delta / l2_fd
    l2_laplacian_u = sqrt(lap_sum)
    mean_abs_delta = mean((abs_delta_sum / count,))
    diff_csv = write_diff_map(joinpath(leaf_dir, DIFF_MAP_CSV), F_FD_x, F_FD_y, F_Guo_x, F_Guo_y)

    outputs = (l2_fd, l2_guo, relative_l2_delta, max_abs_delta)
    @assert all(isfinite, outputs)

    return AuditResult(
        leaf_dir, Nx, nu_p, bsd_fraction, l2_fd, l2_guo, l2_delta, relative_l2_delta,
        max_abs_delta, max_i, max_j, l2_laplacian_u, mean_abs_delta, diff_csv,
    )
end

function analyse_leaf(leaf_dir::AbstractString)
    fields = load_fields(joinpath(leaf_dir, FIELDS_JLS))
    summary = read_summary(joinpath(leaf_dir, SUMMARY_CSV))
    nu_p = summary_float(summary, "nu_p", joinpath(leaf_dir, SUMMARY_CSV))
    bsd_fraction = summary_float(summary, "bsd_fraction", joinpath(leaf_dir, SUMMARY_CSV))

    force_fields = compute_force_fields(fields, nu_p, bsd_fraction)
    return reduce_interior(leaf_dir, nu_p, bsd_fraction, force_fields...)
end

function analyse_path(path::AbstractString)
    return analyse_leaf(find_leaf_dir(path))
end

function print_summary(result::AuditResult; io::IO=stdout)
    println(io, "== M4 Guo-vs-FD audit ==")
    @printf(io, "leaf_dir    : %s\n", display_path(result.leaf_dir))
    @printf(io, "N           : %d\n", result.N)
    @printf(io, "nu_p        : %.10g\n", result.nu_p)
    @printf(io, "bsd_fraction: %.10g\n", result.bsd_fraction)
    @printf(io, "F_FD  L2_norm : %.6e\n", result.l2_fd)
    @printf(io, "F_Guo L2_norm : %.6e\n", result.l2_guo)
    @printf(io, "relative L2(F_FD - F_Guo) / L2(F_FD) : %.6e\n", result.relative_l2_delta)
    @printf(io, "max|F_FD - F_Guo| = %.6e at (i, j) = (%d, %d)\n",
            result.max_abs_delta, result.max_i, result.max_j)
    @printf(io, "diff_map    : %s\n", display_path(result.diff_csv))
end

function write_synthetic_case(leaf_dir::AbstractString)
    mkpath(leaf_dir)
    N = 16
    x = collect(range(0.0, 1.0; length=N))
    y = collect(range(0.0, 1.0; length=N))

    ux = [0.07 * sin(2pi * x[i]) * sin(pi * y[j]) for i in 1:N, j in 1:N]
    uy = [0.05 * cos(pi * x[i]) * sin(2pi * y[j]) for i in 1:N, j in 1:N]
    psixx = [0.01 * sin(pi * x[i]) for i in 1:N, j in 1:N]
    psixy = [0.01 * cos(pi * x[i]) * sin(pi * y[j]) for i in 1:N, j in 1:N]
    psiyy = [0.01 * cos(pi * y[j]) for i in 1:N, j in 1:N]
    tauxx = [sin(2pi * x[i]) * cos(2pi * y[j]) for i in 1:N, j in 1:N]
    tauxy = [0.25 * cos(pi * x[i]) * sin(2pi * y[j]) for i in 1:N, j in 1:N]
    tauyy = [0.5 * sin(2pi * x[i]) * sin(pi * y[j]) for i in 1:N, j in 1:N]

    snapshot = (;
        step=Int64(160),
        t_phys=1.0,
        ux,
        uy,
        psixx,
        psixy,
        psiyy,
        tauxx,
        tauxy,
        tauyy,
    )

    open(joinpath(leaf_dir, FIELDS_JLS), "w") do io
        serialize(io, snapshot)
    end

    open(joinpath(leaf_dir, SUMMARY_CSV), "w") do io
        write(io, "key,value\n")
        @printf(io, "N,%d\n", N)
        @printf(io, "nu_s,%.10g\n", 0.1)
        @printf(io, "nu_p,%.10g\n", 0.1)
        @printf(io, "u_max,%.10g\n", 0.005)
        @printf(io, "bsd_fraction,%.10g\n", 0.75)
        @printf(io, "lambda_phys,%.10g\n", 1.0)
        @printf(io, "lambda_lu,%.10g\n", 12800.0)
        @printf(io, "dt_phys,%.10g\n", 7.8125e-5)
    end

    return leaf_dir
end

function run_self_test()
    mktempdir() do root
        leaf = write_synthetic_case(joinpath(root, "u0.005", "kraken_N16_synthetic"))
        @testset "M4 Guo-vs-FD audit" begin
            result = analyse_path(dirname(leaf))
            buffer = IOBuffer()
            print_summary(result; io=buffer)
            summary = String(take!(buffer))

            @test occursin("M4 Guo-vs-FD audit", summary)
            @test occursin("F_FD  L2_norm", summary)
            @test all(isfinite, (
                result.l2_fd,
                result.l2_guo,
                result.relative_l2_delta,
                result.max_abs_delta,
            ))
            @test result.l2_delta <= result.bsd_fraction * result.nu_p * result.l2_laplacian_u +
                                      max(1e-12, 1e-12 * result.l2_delta)
            @test 2 <= result.max_i <= result.N - 1
            @test 2 <= result.max_j <= result.N - 1
            @test isfile(result.diff_csv)

            print(summary)
        end
        println("SELF-TEST PASSED")
    end
    return nothing
end

function main(args::Vector{String}=ARGS)
    if isempty(args) || first(args) == "--self-test" || first(args) == "-t"
        run_self_test()
        exit(0)
    end
    length(args) == 1 || error("usage: julia --project=. $(PROGRAM_FILE) <output_dir> | --self-test")

    result = analyse_path(only(args))
    print_summary(result)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
