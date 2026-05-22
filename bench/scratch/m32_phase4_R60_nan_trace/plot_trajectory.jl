#!/usr/bin/env julia

using CairoMakie

const HERE = @__DIR__
const DEFAULT_ROOT = normpath(joinpath(HERE, "..", "..", "..", "tmp", "m32_phase4_R60"))

function newest_csv(root)
    candidates = String[]
    if isdir(root)
        for (dir, _, files) in walkdir(root)
            if "nan_probe.csv" in files
                push!(candidates, joinpath(dir, "nan_probe.csv"))
            end
        end
    end
    isempty(candidates) && error("no nan_probe.csv found under $(root)")
    return candidates[argmax(mtime.(candidates))]
end

function csv_path_from_args()
    if !isempty(ARGS)
        return ARGS[1]
    end
    env_path = get(ENV, "KRAKEN_NAN_PROBE_CSV", "")
    !isempty(env_path) && return env_path
    direct = joinpath(DEFAULT_ROOT, "nan_probe.csv")
    isfile(direct) && return direct
    return newest_csv(DEFAULT_ROOT)
end

function read_probe_csv(path)
    lines = filter(!isempty, readlines(path))
    length(lines) >= 2 || error("nan_probe.csv has no data rows: $(path)")
    names = Symbol.(split(lines[1], ","))
    cols = Dict(name => Float64[] for name in names)
    for line in lines[2:end]
        vals = split(line, ",")
        for (name, raw) in zip(names, vals)
            push!(cols[name], parse(Float64, raw))
        end
    end
    return cols
end

probe_csv = csv_path_from_args()
cols = read_probe_csv(probe_csv)
steps = cols[:step]
max_psi = max.(cols[:max_abs_psixx], cols[:max_abs_psixy], cols[:max_abs_psiyy])
max_tau = cols[:max_tr_tau]
max_div = cols[:max_div_u]
floor_positive(v) = max.(abs.(v), eps(Float64))

fig = Figure(size=(1000, 620))
ax = Axis(
    fig[1, 1],
    xlabel="step",
    ylabel="global diagnostic",
    yscale=log10,
)
lines!(ax, steps, floor_positive(max_psi), label="max |Psi|", linewidth=2.5)
lines!(ax, steps, floor_positive(max_tau), label="max tr(tau)", linewidth=2.5)
lines!(ax, steps, floor_positive(max_div), label="max |div u|", linewidth=2.5)
axislegend(ax, position=:lt)
fig[0, 1] = Label(fig, basename(probe_csv), fontsize=16)

out = joinpath(HERE, "nan_probe_trajectory.png")
save(out, fig)
println("wrote $(out)")
