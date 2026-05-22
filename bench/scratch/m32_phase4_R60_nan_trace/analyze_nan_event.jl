#!/usr/bin/env julia

using Serialization
using Printf

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

function sym2_min_eig(a, b, c)
    tr = a + c
    disc = sqrt((a - c)^2 + 4b^2)
    return 0.5 * (tr - disc)
end

function cell_metrics(s, i, j)
    psi_vals = (
        Float64(s.psixx[i, j]),
        Float64(s.psixy[i, j]),
        Float64(s.psiyy[i, j]),
    )
    tau_trace = Float64(s.tauxx[i, j]) + Float64(s.tauyy[i, j])
    div_u = Float64(s.dudx[i, j]) + Float64(s.dvdy[i, j])
    return (;
        step=Int(s.step),
        max_abs_psi=maximum(abs, psi_vals),
        trace_tau=tau_trace,
        abs_div_u=abs(div_u),
        speed=hypot(Float64(s.ux[i, j]), Float64(s.uy[i, j])),
        psi_min_eig=sym2_min_eig(psi_vals[1], psi_vals[2], psi_vals[3]),
    )
end

function neighbor_psi_min(s, i, j)
    vals = Float64[]
    nx, ny = size(s.psixx)
    for jj in max(1, j - 1):min(ny, j + 1), ii in max(1, i - 1):min(nx, i + 1)
        ii == i && jj == j && continue
        s.is_solid[ii, jj] && continue
        v = sym2_min_eig(
            Float64(s.psixx[ii, jj]),
            Float64(s.psixy[ii, jj]),
            Float64(s.psiyy[ii, jj]),
        )
        isfinite(v) && push!(vals, v)
    end
    isempty(vals) && return (min=NaN, mean=NaN, n=0)
    return (min=minimum(vals), mean=sum(vals) / length(vals), n=length(vals))
end

function ratio(last, first)
    !(isfinite(last) && isfinite(first)) && return NaN
    abs(first) <= eps(Float64) && return last == first ? 1.0 : Inf
    return last / first
end

function position_label(theta_deg)
    if -30.0 <= theta_deg <= 30.0
        return "front-pole"
    elseif (60.0 <= theta_deg <= 120.0) || (-120.0 <= theta_deg <= -60.0)
        return "shoulder"
    elseif theta_deg >= 150.0 || theta_deg <= -150.0
        return "wake"
    else
        return "front-shoulder"
    end
end

function classify(field, label, wall_offset, theta_deg, metrics)
    first_field = Symbol(field)
    psi_fields = (:psixx, :psixy, :psiyy)
    tau_fields = (:tauxx, :tauxy, :tauyy)
    force_fields = (:fx_total, :fy_total)
    near_pole = abs(theta_deg) <= 30.0 || abs(abs(theta_deg) - 180.0) <= 30.0
    in_ring = -1.5 <= wall_offset <= 3.0
    first_metric = first(metrics)
    last_clean = length(metrics) >= 2 ? metrics[end - 1] : first(metrics)
    psi_growth = ratio(last_clean.max_abs_psi, first_metric.max_abs_psi)
    tau_growth = ratio(last_clean.trace_tau, first_metric.trace_tau)
    div_growth = ratio(last_clean.abs_div_u, first_metric.abs_div_u)

    if near_pole && in_ring
        return :bc_pole_pathology
    elseif first_field in force_fields
        return :bsd_coupling
    elseif first_field in tau_fields && tau_growth > max(psi_growth, div_growth)
        return :logconf_singularity
    elseif first_field in psi_fields && label in ("shoulder", "wake", "front-shoulder")
        return :rusanov_overshoot
    else
        return :other
    end
end

function write_cell_trajectory(path, metrics)
    open(path, "w") do io
        println(io, "step,max_abs_psi,trace_tau,abs_div_u,speed,psi_min_eig")
        for m in metrics
            @printf(
                io,
                "%d,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                m.step,
                m.max_abs_psi,
                m.trace_tau,
                m.abs_div_u,
                m.speed,
                m.psi_min_eig,
            )
        end
    end
end

event_path = event_path_from_args()
event = deserialize(event_path)
params = event.case_parameters
i = Int(event.first_nonfinite_i)
j = Int(event.first_nonfinite_j)
cx = Float64(ntget(params, :cylinder_x_lbm, 900.0))
cy = Float64(ntget(params, :cylinder_y_lbm, 119.5))
radius = Float64(ntget(params, :radius_lbm, 60.0))
theta = atan(j - cy, i - cx)
theta_deg = theta * 180.0 / pi
r = hypot(i - cx, j - cy)
label = position_label(theta_deg)
wall_offset = r - radius
ring_row = floor(Int, wall_offset)
snapshots = [event.clean_snapshots...; event.nan_snapshot]
metrics = [cell_metrics(s, i, j) for s in snapshots]
neigh = isempty(event.clean_snapshots) ?
    (min=NaN, mean=NaN, n=0) :
    neighbor_psi_min(event.clean_snapshots[end], i, j)
classification = classify(
    event.first_nonfinite_field, label, wall_offset, theta_deg, metrics,
)

mkpath(HERE)
trajectory_path = joinpath(HERE, "trajectory_at_first_nan_cell.csv")
summary_path = joinpath(HERE, "nan_event_summary.txt")
write_cell_trajectory(trajectory_path, metrics)

first_metric = first(metrics)
last_clean = length(metrics) >= 2 ? metrics[end - 1] : first(metrics)

open(summary_path, "w") do io
    println(io, "classification=$(classification)")
    println(io, "event_path=$(event_path)")
    println(io, "first_nonfinite_step=$(event.first_nonfinite_step)")
    println(io, "first_nonfinite_field=$(event.first_nonfinite_field)")
    println(io, "first_nonfinite_cell=$(i),$(j)")
    @printf(io, "theta_deg=%.6f\n", theta_deg)
    @printf(io, "radius=%.6f\n", r)
    @printf(io, "wall_offset=%.6f\n", wall_offset)
    println(io, "position_label=$(label)")
    println(io, "wall_ring_row=$(ring_row)")
    println(io, "polymer_substeps_used=$(ntget(params, :polymer_substeps_used, missing))")
    println(io, "max_polymer_substeps=$(ntget(params, :max_polymer_substeps, missing))")
    @printf(io, "first_metric_max_abs_psi=%.17g\n", first_metric.max_abs_psi)
    @printf(io, "last_clean_max_abs_psi=%.17g\n", last_clean.max_abs_psi)
    @printf(io, "first_metric_trace_tau=%.17g\n", first_metric.trace_tau)
    @printf(io, "last_clean_trace_tau=%.17g\n", last_clean.trace_tau)
    @printf(io, "first_metric_abs_div_u=%.17g\n", first_metric.abs_div_u)
    @printf(io, "last_clean_abs_div_u=%.17g\n", last_clean.abs_div_u)
    @printf(io, "last_clean_speed=%.17g\n", last_clean.speed)
    @printf(io, "last_clean_psi_min_eig=%.17g\n", last_clean.psi_min_eig)
    @printf(io, "neighbor_psi_min=%.17g\n", neigh.min)
    @printf(io, "neighbor_psi_mean=%.17g\n", neigh.mean)
    println(io, "neighbor_count=$(neigh.n)")
end

println("wrote $(summary_path)")
println("wrote $(trajectory_path)")
