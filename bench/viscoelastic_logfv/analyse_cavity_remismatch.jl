#!/usr/bin/env julia

using DelimitedFiles
using Printf
using Test

const VERTICAL_CSV = "profile_vertical_x0.5.csv"
const HORIZONTAL_CSV = "profile_horizontal_y0.75.csv"

struct AnalysisRow
    label::String
    sort_value::Float64
    leaf_dir::String
    l2_u::Float64
    l2_psixy::Float64
end

function relative_l2(values::AbstractVector, ref::AbstractVector)
    length(values) == length(ref) || error("vector length mismatch")
    denom = sqrt(sum(abs2, ref))
    denom < eps() && return NaN
    return sqrt(sum(abs2, values .- ref)) / denom
end

function read_csv_matrix(path::AbstractString)
    isfile(path) || error("missing CSV: $(path)")
    return readdlm(path, ',', Float64; skipstart=1)
end

function find_leaf_dir(path::AbstractString)
    root = abspath(path)
    isdir(root) || error("not a directory: $(path)")
    if isfile(joinpath(root, VERTICAL_CSV)) && isfile(joinpath(root, HORIZONTAL_CSV))
        return root
    end

    candidates = String[]
    for child in readdir(root; join=true)
        isdir(child) || continue
        if isfile(joinpath(child, VERTICAL_CSV)) && isfile(joinpath(child, HORIZONTAL_CSV))
            push!(candidates, abspath(child))
        end
    end
    isempty(candidates) && error("no one-level case directory with profile CSVs under $(path)")
    length(candidates) == 1 || error("expected one case directory under $(path), found $(length(candidates))")
    return only(candidates)
end

function infer_label(leaf_dir::AbstractString)
    parent = basename(dirname(leaf_dir))
    matched = match(r"^u([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)$", parent)
    if matched !== nothing
        label = matched.captures[1]
        return label, parse(Float64, label)
    end
    fallback = basename(leaf_dir)
    parsed = tryparse(Float64, fallback)
    return fallback, parsed === nothing ? -Inf : parsed
end

function analyse_leaf(leaf_dir::AbstractString)
    vertical = read_csv_matrix(joinpath(leaf_dir, VERTICAL_CSV))
    horizontal = read_csv_matrix(joinpath(leaf_dir, HORIZONTAL_CSV))
    size(vertical, 2) >= 4 || error("expected at least four columns in $(VERTICAL_CSV)")
    size(horizontal, 2) >= 4 || error("expected at least four columns in $(HORIZONTAL_CSV)")

    l2_u = relative_l2(view(vertical, :, 2), view(vertical, :, 4))
    l2_psixy = relative_l2(view(horizontal, :, 2), view(horizontal, :, 4))
    label, sort_value = infer_label(leaf_dir)
    return AnalysisRow(label, sort_value, leaf_dir, l2_u, l2_psixy)
end

function analyse_dirs(paths::AbstractVector{<:AbstractString})
    isempty(paths) && error("provide one or more output directories, or pass --self-test")
    rows = [analyse_leaf(find_leaf_dir(path)) for path in paths]
    return sort(rows; by=row -> (-row.sort_value, row.label))
end

function print_table(rows::AbstractVector{AnalysisRow}; io::IO=stdout)
    println(io, "u_max    | L2(u_centerline) | L2(psi_xy_y=0.75)")
    println(io, "-------- | ---------------- | -----------------")
    for row in rows
        @printf(io, "%-8s | %-16.3e | %-17.3e\n", row.label, row.l2_u, row.l2_psixy)
    end
end

function write_synthetic_case(leaf_dir::AbstractString; scale::Float64)
    mkpath(leaf_dir)
    y = collect(range(0.0, 1.0; length=17))
    x = collect(range(0.0, 1.0; length=17))
    ref_u = sin.(pi .* y)
    ref_v = 0.25 .* cos.(pi .* y)
    ref_theta = cos.(pi .* x)
    ref_tau = 0.5 .* sin.(pi .* x)

    open(joinpath(leaf_dir, VERTICAL_CSV), "w") do io
        write(io, "y,kraken_ux,kraken_uy,rheotool_ux,rheotool_uy\n")
        for k in eachindex(y)
            @printf(io, "%.10g,%.10g,%.10g,%.10g,%.10g\n",
                    y[k], scale * ref_u[k], scale * ref_v[k], ref_u[k], ref_v[k])
        end
    end

    open(joinpath(leaf_dir, HORIZONTAL_CSV), "w") do io
        write(io, "x,kraken_psixy,kraken_tauxy,rheotool_thetaxy,rheotool_tauxy\n")
        for k in eachindex(x)
            @printf(io, "%.10g,%.10g,%.10g,%.10g,%.10g\n",
                    x[k], scale * ref_theta[k], scale * ref_tau[k], ref_theta[k], ref_tau[k])
        end
    end
end

function run_self_test()
    mktempdir() do root
        identity_leaf = joinpath(root, "u0.005", "case")
        perturbed_leaf = joinpath(root, "u0.002", "case")
        write_synthetic_case(identity_leaf; scale=1.0)
        write_synthetic_case(perturbed_leaf; scale=1.1)

        rows = analyse_dirs([joinpath(root, "u0.005"), joinpath(root, "u0.002")])
        buffer = IOBuffer()
        print_table(rows; io=buffer)
        table = String(take!(buffer))

        by_label = Dict(row.label => row for row in rows)
        @test haskey(by_label, "0.005")
        @test haskey(by_label, "0.002")
        @test by_label["0.005"].l2_u < 1e-10
        @test by_label["0.005"].l2_psixy < 1e-10
        @test 0.05 < by_label["0.002"].l2_u < 0.15
        @test 0.05 < by_label["0.002"].l2_psixy < 0.15
        @test occursin("L2(u_centerline)", table)

        print(table)
        println("SELF-TEST PASSED")
    end
    return nothing
end

function main(args::Vector{String}=ARGS)
    if any(arg -> arg == "--self-test" || arg == "-t", args)
        run_self_test()
        return nothing
    end

    rows = analyse_dirs(args)
    print_table(rows)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
