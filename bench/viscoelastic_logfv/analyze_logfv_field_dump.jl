#!/usr/bin/env julia

using Printf
using Serialization

if isempty(ARGS)
    println(stderr, "usage: julia --project=. bench/viscoelastic_logfv/analyze_logfv_field_dump.jl FIELD_DUMP.jls [summary.csv]")
    exit(2)
end

dump_path = ARGS[1]
out_csv = length(ARGS) >= 2 ? ARGS[2] :
          joinpath(dirname(dump_path), "field_dump_summary.csv")

payload = open(deserialize, dump_path)
metadata = payload.metadata
fields = payload.fields

solid = haskey(fields, "solid") ? fields["solid"] .> 0.5 :
        falses(size(first(values(fields))))
Nx, Ny = size(solid)
R = Float64(get(metadata, :R, NaN))
L_up = Float64(get(metadata, :L_up, NaN))
cx = isfinite(R) && isfinite(L_up) ? L_up * R : NaN
cy = (Ny - 1) / 2

function surface_distance(i::Integer, j::Integer)
    isfinite(cx) && isfinite(R) || return NaN
    x = Float64(i - 1)
    y = Float64(j - 1)
    return hypot(x - cx, y - cy) - R
end

function adjacent_solid(i::Integer, j::Integer)
    i1 = max(1, i - 1)
    i2 = min(Nx, i + 1)
    j1 = max(1, j - 1)
    j2 = min(Ny, j + 1)
    @inbounds for jj in j1:j2, ii in i1:i2
        solid[ii, jj] && return true
    end
    return false
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

function summarize_field(name::AbstractString, values)
    n_fluid = 0
    n_nonfinite = 0
    first_i = 0
    first_j = 0
    min_v = Inf
    max_v = -Inf
    max_abs = -Inf
    max_abs_i = 0
    max_abs_j = 0

    @inbounds for j in 1:Ny, i in 1:Nx
        solid[i, j] && continue
        n_fluid += 1
        v = Float64(values[i, j])
        if !isfinite(v)
            n_nonfinite += 1
            if first_i == 0
                first_i = i
                first_j = j
            end
            continue
        end
        min_v = min(min_v, v)
        max_v = max(max_v, v)
        av = abs(v)
        if av > max_abs
            max_abs = av
            max_abs_i = i
            max_abs_j = j
        end
    end

    finite_count = n_fluid - n_nonfinite
    finite_count == 0 && begin
        min_v = NaN
        max_v = NaN
        max_abs = NaN
    end

    return Dict{Symbol,Any}(
        :field => name,
        :n_fluid => n_fluid,
        :n_nonfinite => n_nonfinite,
        :first_i => first_i,
        :first_j => first_j,
        :first_surface_distance => first_i == 0 ? NaN : surface_distance(first_i, first_j),
        :first_adjacent_solid => first_i != 0 && adjacent_solid(first_i, first_j),
        :min => min_v,
        :max => max_v,
        :max_abs => max_abs,
        :max_abs_i => max_abs_i,
        :max_abs_j => max_abs_j,
        :max_abs_surface_distance => max_abs_i == 0 ? NaN : surface_distance(max_abs_i, max_abs_j),
        :max_abs_adjacent_solid => max_abs_i != 0 && adjacent_solid(max_abs_i, max_abs_j),
    )
end

columns = [
    :field, :n_fluid, :n_nonfinite,
    :first_i, :first_j, :first_surface_distance, :first_adjacent_solid,
    :min, :max, :max_abs,
    :max_abs_i, :max_abs_j, :max_abs_surface_distance, :max_abs_adjacent_solid,
]

preferred = [
    "rho", "ux", "uy", "u_mag",
    "psi_xx", "psi_xy", "psi_yy",
    "C_xx", "C_xy", "C_yy", "C_trace",
    "tau_xx", "tau_xy", "tau_yy",
    "fx_poly", "fy_poly", "fx_total", "fy_total",
]
field_names = [name for name in preferred if haskey(fields, name)]
append!(field_names, sort(setdiff(collect(keys(fields)), field_names)))

rows = [summarize_field(name, fields[name]) for name in field_names]

open(out_csv, "w") do io
    println(io, join(string.(columns), ","))
    for row in rows
        println(io, join((csv_cell(get(row, col, "")) for col in columns), ","))
    end
end

println("="^78)
println("Log-FV field dump diagnostic")
println("dump=$(dump_path)")
println("summary=$(out_csv)")
println("R=$(get(metadata, :R, "")) Wi=$(get(metadata, :Wi, "")) case=$(get(metadata, :case_name, "")) completed_steps=$(get(metadata, :completed_steps, ""))")
println("first_nonfinite_step=$(get(metadata, :first_nonfinite_step, "")) field=$(get(metadata, :first_nonfinite_field, "")) i=$(get(metadata, :first_nonfinite_i, "")) j=$(get(metadata, :first_nonfinite_j, ""))")
println("-"^78)
for row in rows
    if row[:n_nonfinite] > 0
        @printf("%-12s nonfinite=%8d first=(%d,%d) d_wall=% .4g adjacent=%s\n",
                row[:field], row[:n_nonfinite], row[:first_i], row[:first_j],
                row[:first_surface_distance], string(row[:first_adjacent_solid]))
    else
        @printf("%-12s finite      maxabs=% .6g at=(%d,%d) d_wall=% .4g adjacent=%s\n",
                row[:field], row[:max_abs], row[:max_abs_i], row[:max_abs_j],
                row[:max_abs_surface_distance], string(row[:max_abs_adjacent_solid]))
    end
end
