# Pseudo-PIV demo on the converged J1 fields (CPU only).
#
#   julia --project=. benchmarks/inverse_rheology/pseudo_piv_demo.jl [runs=run1_m64_nu0.340,run7_m64_nu0.340]
#         [in=<KRAKEN_REFERENCE_DATA>/j1_fields] [out=benchmarks/inverse_rheology/out/piv] [ref=<KRAKEN_REFERENCE_DATA>/reference_fields]
#         [seed=20261020]
#
# For each run folder: rebuilds the solid mask at Kraken nodes with
# `case_solid_mask` (obstacle_channel_case.jl) and checks its node count against
# manifest.toml; converts the stored u* fields back to lattice units; applies
# `Kraken.pseudo_piv` with the default settings; projects onto the reference grid;
# writes .npy files, summary.json, profiles.csv and (if CairoMakie loads) a PNG.
# Defaults of every setting: see `Kraken.PseudoPIVSettings`. Input folders are
# read-only.

using Kraken
using Random
using Printf
using TOML

include(joinpath(@__DIR__, "obstacle_channel_case.jl"))

const PIV = Kraken

function parse_args(args)
    d = Dict{String,String}()
    for a in args
        k, v = split(a, '='; limit = 2)
        d[k] = v
    end
    return d
end

# Minimal JSON writer (no dependency): numbers, strings, bools, nested Dict/NamedTuple/Vector.
json(x::Bool) = x ? "true" : "false"
json(x::Integer) = string(x)
json(x::AbstractFloat) = isfinite(x) ? @sprintf("%.10g", x) : "null"
json(x::AbstractString) = "\"" * replace(x, "\"" => "\\\"") * "\""
json(::Nothing) = "null"
json(x::AbstractVector) = "[" * join(json.(x), ", ") * "]"
json(x::NamedTuple) = json(Dict(String(k) => v for (k, v) in pairs(x)))
json(x::AbstractDict) = "{" * join(("\"$k\": " * json(x[k]) for k in sort!(collect(keys(x)))), ", ") * "}"

rrmse(a, b) = sqrt(sum(abs2, a .- b) / sum(abs2, b))

function run_one(rundir, outroot, reffields, seed)
    name = basename(rundir)
    man = TOML.parsefile(joinpath(rundir, "manifest.toml"))
    res = TOML.parsefile(joinpath(rundir, "manifest.toml"))
    m = Int(man["m"]); nu0 = Float64(man["nu0_lat"])
    solid = case_solid_mask(CaseUnits(m, nu0))
    @assert count(solid) == man["solid_nodes"] "solid node count $(count(solid)) != manifest.toml $(man["solid_nodes"])"
    uxs = Float64.(PIV.read_npy(joinpath(rundir, "ux_star_nodes.npy")))
    uys = Float64.(PIV.read_npy(joinpath(rundir, "uy_star_nodes.npy")))
    uref = nu0 / m                                    # u_lat = u* * nu0 / m
    ux_lat = uxs .* uref; uy_lat = uys .* uref
    outdir = joinpath(outroot, name); mkpath(outdir)

    st = PIV.PseudoPIVSettings(seed = seed)
    t_cpu = @elapsed f = PIV.pseudo_piv(ux_lat, uy_lat, solid, m, nu0, st; rng = Xoshiro(st.seed))
    # same pipeline without noise nor dropout: window means only (lost band identical)
    st0 = PIV.PseudoPIVSettings(seed = seed, sigma_rel = 0.0, dropout = 0.0)
    f0 = PIV.pseudo_piv(ux_lat, uy_lat, solid, m, nu0, st0; rng = Xoshiro(st.seed))

    ux, uy, val = PIV.to_reference_grid(f)
    ux0, uy0, val0 = PIV.to_reference_grid(f0)
    PIV.write_reference_npy(outdir, ux, uy, val)

    # (a) noise alone: observed vs window means on the same valid vectors
    v = f.valid
    ex = f.ux[v] .- f.ux_clean[v]; ey = f.uy[v] .- f.uy_clean[v]
    cx = f.ux_clean[v]; cy = f.uy_clean[v]
    rr_noise = sqrt((sum(abs2, ex) + sum(abs2, ey)) / (sum(abs2, cx) + sum(abs2, cy)))
    # (b) observed vs reference field at the nodes holding an observation
    zfolder = joinpath(reffields, man["reference_folder"])
    zx = Float64.(PIV.read_npy(joinpath(zfolder, "reference_velocity_x.npy")))
    zy = Float64.(PIV.read_npy(joinpath(zfolder, "reference_velocity_y.npy")))
    obs(a, va, z) = (Float64.(a[va]), z[va])
    ox, rx = obs(ux, val.x, zx); oy, ry = obs(uy, val.y, zy)
    rr_zen = sqrt((sum(abs2, ox .- rx) + sum(abs2, oy .- ry)) / (sum(abs2, rx) + sum(abs2, ry)))
    ox0, rx0 = obs(ux0, val0.x, zx); oy0, ry0 = obs(uy0, val0.y, zy)
    rr_zen_clean = sqrt((sum(abs2, ox0 .- rx0) + sum(abs2, oy0 .- ry0)) / (sum(abs2, rx0) + sum(abs2, ry0)))

    c = f.counts
    summary = Dict(
        "run" => name, "m" => m, "nu0_lat" => nu0, "seed" => seed,
        "solid_nodes" => count(solid), "solid_nodes_manifest.toml" => man["solid_nodes"],
        "settings" => Dict(String(k) => getfield(st, k) for k in fieldnames(PIV.PseudoPIVSettings)),
        "lost_band_effective" => PIV.piv_lost_band_value(st),
        "vectors_total" => c.total, "vectors_valid" => c.valid,
        "lost_band" => c.lost_band, "no_fluid" => c.no_fluid, "dropout" => c.dropout, "outliers" => c.outliers,
        "frac_lost_band" => c.lost_band / c.total, "frac_dropout" => c.dropout / c.total,
        "frac_valid" => c.valid / c.total,
        "u_ref_star_max" => f.u_ref, "sigma_target_star" => st.sigma_rel * f.u_ref,
        "sigma_eff_star" => f.sigma_eff,
        "rrmse_noise_vs_window_means" => rr_noise,
        "rrmse_vs_reference_valid_nodes" => rr_zen,
        "rrmse_window_means_vs_reference_same_nodes" => rr_zen_clean,
        "reference_nodes_x" => count(val.x), "reference_nodes_y" => count(val.y),
        "cpu_seconds_pseudo_piv" => t_cpu,
        "stagger" => "mac (assumed, H17 not verified)",
    )
    open(joinpath(outdir, "summary.json"), "w") do io
        println(io, json(summary))
    end

    # profile at the throat column (vector column nearest x = 4)
    a = argmin(abs.(f.x .- 4.0))
    open(joinpath(outdir, "profile_throat.csv"), "w") do io
        println(io, "x,y,ux_obs,ux_window_mean,valid")
        for b in eachindex(f.y)
            @printf(io, "%.6f,%.6f,%.6f,%.6f,%d\n", f.x[a], f.y[b], f.ux[a, b], f.ux_clean[a, b], f.valid[a, b])
        end
    end
    # PNG if CairoMakie can be loaded in this environment, otherwise the CSV above is the figure data
    png = false
    try
        @eval using CairoMakie
        fig = Base.invokelatest() do
            fg = CairoMakie.Figure(size = (1000, 700))
            ax = CairoMakie.Axis(fg[1, 1]; title = "$name: pseudo-PIV ux (u*)", aspect = CairoMakie.DataAspect())
            CairoMakie.heatmap!(ax, f.x, f.y, f.ux)
            ax2 = CairoMakie.Axis(fg[2, 1]; xlabel = "y", ylabel = "ux (u*) at x = $(round(f.x[a]; digits = 3))")
            CairoMakie.lines!(ax2, f.y, f.ux_clean[a, :])
            CairoMakie.scatter!(ax2, f.y, f.ux[a, :]; markersize = 4)
            fg
        end
        Base.invokelatest(CairoMakie.save, joinpath(outdir, "piv_demo.png"), fig)
        png = true
    catch err
        @info "CairoMakie not available, CSV profile only" err = sprint(showerror, err)
    end
    @printf("%s: valid %d/%d (band %.1f%%, dropout %.1f%%), sigma_eff %.4g (target %.4g), rrmse noise %.4f, vs reference %.4f (window means alone %.4f), cpu %.2fs, png=%s\n",
            name, c.valid, c.total, 100c.lost_band / c.total, 100c.dropout / c.total, f.sigma_eff,
            st.sigma_rel * f.u_ref, rr_noise, rr_zen, rr_zen_clean, t_cpu, png)
    return summary
end

function main(args)
    a = parse_args(args)
    inroot = expanduser(get(a, "in", something(Kraken.reference_data_path("j1_fields"), "j1_fields")))
    outroot = get(a, "out", joinpath(@__DIR__, "out", "piv"))
    reffields = expanduser(get(a, "ref", something(Kraken.reference_data_path("reference_fields"), "reference_fields")))
    seed = parse(Int, get(a, "seed", "20261020"))
    runs = split(get(a, "runs", "run1_m64_nu0.340,run7_m64_nu0.340,run4_m72_nu0.340"), ',')
    for r in runs
        run_one(joinpath(inroot, r), outroot, reffields, seed)
    end
end

main(ARGS)
