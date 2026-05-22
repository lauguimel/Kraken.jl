#!/usr/bin/env julia

using CairoMakie
using Serialization

const HERE = @__DIR__
const DEFAULT_ROOT = normpath(joinpath(HERE, "..", "..", "..", "tmp", "m32_phase4_R60"))

ntget(nt, key::Symbol, default) = hasproperty(nt, key) ? getproperty(nt, key) : default

function newest_event(root)
    candidates = String[]
    if isdir(root)
        for (dir, _, files) in walkdir(root)
            if "nan_event.jls" in files
                push!(candidates, joinpath(dir, "nan_event.jls"))
            end
        end
    end
    isempty(candidates) && error("no nan_event.jls found under $(root)")
    return candidates[argmax(mtime.(candidates))]
end

function event_path_from_args()
    if !isempty(ARGS)
        return ARGS[1]
    end
    env_path = get(ENV, "KRAKEN_NAN_EVENT", "")
    !isempty(env_path) && return env_path
    direct = joinpath(DEFAULT_ROOT, "nan_event.jls")
    isfile(direct) && return direct
    return newest_event(DEFAULT_ROOT)
end

function boundary_points(is_solid)
    xs = Float64[]
    ys = Float64[]
    nx, ny = size(is_solid)
    @inbounds for j in 2:(ny - 1), i in 2:(nx - 1)
        is_solid[i, j] || continue
        if !is_solid[i - 1, j] || !is_solid[i + 1, j] ||
           !is_solid[i, j - 1] || !is_solid[i, j + 1]
            push!(xs, i)
            push!(ys, j)
        end
    end
    return xs, ys
end

event_path = event_path_from_args()
event = deserialize(event_path)
params = event.case_parameters
snapshot = event.nan_snapshot
is_solid = snapshot.is_solid
nx, ny = size(is_solid)
i = Int(event.first_nonfinite_i)
j = Int(event.first_nonfinite_j)
cx = Float64(ntget(params, :cylinder_x_lbm, 900.0))
cy = Float64(ntget(params, :cylinder_y_lbm, 119.5))
radius = Float64(ntget(params, :radius_lbm, 60.0))
theta = range(0, 2pi; length=361)
circle_x = cx .+ radius .* cos.(theta)
circle_y = cy .+ radius .* sin.(theta)
ring_x = cx .+ (radius + 1.0) .* cos.(theta)
ring_y = cy .+ (radius + 1.0) .* sin.(theta)
bx, by = boundary_points(is_solid)

fig = Figure(size=(900, 620))
ax = Axis(
    fig[1, 1],
    xlabel="i",
    ylabel="j",
    title="first nonfinite $(event.first_nonfinite_field) at step $(event.first_nonfinite_step)",
    aspect=DataAspect(),
)
scatter!(ax, bx, by, markersize=2, color=(:gray35, 0.45), label="staircase boundary")
lines!(ax, circle_x, circle_y, color=:black, linewidth=2, label="R")
lines!(ax, ring_x, ring_y, color=:dodgerblue4, linewidth=1.5, linestyle=:dash, label="R+1")
scatter!(ax, [i], [j], color=:red, markersize=16, label="first nonfinite")
xlims!(ax, max(1, i - 150), min(nx, i + 150))
ylims!(ax, max(1, j - 100), min(ny, j + 100))
axislegend(ax, position=:rt)

out = joinpath(HERE, "first_nan_location.png")
save(out, fig)
println("wrote $(out)")
