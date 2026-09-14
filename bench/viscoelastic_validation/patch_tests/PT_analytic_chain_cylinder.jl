#!/usr/bin/env julia

using Printf
using Statistics
using Kraken

const OUTDIR = joinpath("scratch", "M58")
const PER_CELL_CSV = joinpath(OUTDIR, "M58_per_cell_errors.csv")
const AGG_CSV = joinpath(OUTDIR, "M58_aggregate_matrix.csv")

const D2Q9_QS = 2:9
const FIELDS = ("F_shear", "F_rotation", "F_poiseuille")
const THETAS = (
    ("0", 0.0),
    ("pi/4", pi / 4),
    ("pi/2", pi / 2),
    ("3pi/4", 3pi / 4),
)
const COMPONENTS_G = ("dudx", "dudy", "dvdx", "dvdy")
const COMPONENTS_S = ("xx", "xy", "yy")
const COMPONENTS_F = ("fx", "fy")

function base_field_gradient(field, xr, yr, radius)
    if field == "F_shear"
        return (yr, 0.0), (0.0, 1.0, 0.0, 0.0)
    elseif field == "F_rotation"
        return (-yr, xr), (0.0, -1.0, 1.0, 0.0)
    elseif field == "F_poiseuille"
        return ((yr * yr) / (radius * radius) - 1.0, 0.0),
               (0.0, 2.0 * yr / (radius * radius), 0.0, 0.0)
    end
    error("unknown field $(field)")
end

function rotate_gradient(g, c, s)
    a, b, d, e = g
    # R(theta) * grad * R(-theta), with grad = [a b; d e].
    m11 = c * a - s * d
    m12 = c * b - s * e
    m21 = s * a + c * d
    m22 = s * b + c * e
    return (
        m11 * c - m12 * s,
        m11 * s + m12 * c,
        m21 * c - m22 * s,
        m21 * s + m22 * c,
    )
end

function fill_analytic!(ux, uy, grad, source, stress, force, field, theta, cx, cy, radius, alpha)
    c = cos(theta)
    s = sin(theta)
    force_base = field == "F_poiseuille" ? 2.0 * alpha / (radius * radius) : 0.0
    @inbounds for j in axes(ux, 2), i in axes(ux, 1)
        xr = (i - 1.0) - cx
        yr = (j - 1.0) - cy
        xo = c * xr + s * yr
        yo = -s * xr + c * yr
        u0, g0 = base_field_gradient(field, xo, yo, radius)
        ux[i, j] = c * u0[1] - s * u0[2]
        uy[i, j] = s * u0[1] + c * u0[2]
        g = rotate_gradient(g0, c, s)
        grad[1][i, j], grad[2][i, j], grad[3][i, j], grad[4][i, j] = g
        src = (2.0 * g[1], g[2] + g[3], 2.0 * g[4])
        source[1][i, j], source[2][i, j], source[3][i, j] = src
        stress[1][i, j], stress[2][i, j], stress[3][i, j] = alpha .* src
        force[1][i, j] = c * force_base
        force[2][i, j] = s * force_base
    end
    return nothing
end

function cell_bin(embedded, i, j)
    qmax = 0.0
    @inbounds for q in D2Q9_QS
        qw = embedded.wall_q[i, j, q]
        if 0.1 <= qw <= 0.9
            qmax = max(qmax, qw)
        end
    end
    qmax == 0.0 && return "AXIS_INTERIOR"
    qmax <= 0.3 && return "CUT_Q_0.1_0.3"
    qmax <= 0.5 && return "CUT_Q_0.3_0.5"
    qmax <= 0.7 && return "CUT_Q_0.5_0.7"
    return "CUT_Q_0.7_0.9"
end

function infnorm(arrs, is_solid)
    m = 0.0
    @inbounds for j in axes(is_solid, 2), i in axes(is_solid, 1)
        is_solid[i, j] && continue
        for a in arrs
            m = max(m, abs(a[i, j]))
        end
    end
    return m
end

threshold(normval) = max(1.0e-12, 0.1 * normval)

function push_component_rows!(rows, field, rot, gate, helper, comps, got, expected, bins, is_solid, th)
    @inbounds for j in axes(is_solid, 2), i in axes(is_solid, 1)
        is_solid[i, j] && continue
        for k in eachindex(comps)
            err = abs(got[k][i, j] - expected[k][i, j])
            push!(rows, (
                field=field, rotation=rot, gate=gate, cell_i=i, cell_j=j,
                helper=helper, bin=bins[i, j], component=comps[k],
                computed=got[k][i, j], analytic=expected[k][i, j],
                abs_err=err, threshold=th, verdict=err > th ? "RED" : "OK",
            ))
        end
    end
    return nothing
end

function oldroydb_source_arrays(grads, lambda)
    out = ntuple(_ -> similar(grads[1]), 3)
    @inbounds for j in axes(grads[1], 2), i in axes(grads[1], 1)
        out[1][i, j], out[2][i, j], out[3][i, j] = Kraken.logfv_oldroydb_source_c_2d(
            1.0, 0.0, 1.0, grads[1][i, j], grads[2][i, j], grads[3][i, j], grads[4][i, j], lambda,
        )
    end
    return out
end

function stress_from_source(source, dt, model)
    cxx = ones(Float64, size(source[1]))
    cxy = zeros(Float64, size(source[1]))
    cyy = ones(Float64, size(source[1]))
    @. cxx += dt * source[1]
    @. cxy += dt * source[2]
    @. cyy += dt * source[3]
    tau = ntuple(_ -> zeros(Float64, size(source[1])), 3)
    Kraken.update_polymer_stress!(tau[1], tau[2], tau[3], cxx, cxy, cyy, model)
    return tau
end

function run_case!(rows, field, rot_label, theta, q_wall, is_solid, embedded, bc)
    Nx, Ny = size(is_solid)
    dx = dy = 1.0
    cx = cy = 16.5
    radius = 4.0
    lambda = 1.0
    dt = 0.01 * lambda
    G = 2.0
    alpha = G * dt
    model = Kraken.OldroydB(G, lambda)

    ux = zeros(Float64, Nx, Ny)
    uy = similar(ux)
    grad_an = ntuple(_ -> zeros(Float64, Nx, Ny), 4)
    source_an = ntuple(_ -> zeros(Float64, Nx, Ny), 3)
    stress_an = ntuple(_ -> zeros(Float64, Nx, Ny), 3)
    force_an = ntuple(_ -> zeros(Float64, Nx, Ny), 2)
    fill_analytic!(ux, uy, grad_an, source_an, stress_an, force_an, field, theta, cx, cy, radius, alpha)

    bins = Array{String}(undef, Nx, Ny)
    @inbounds for j in 1:Ny, i in 1:Nx
        bins[i, j] = is_solid[i, j] ? "SOLID" : cell_bin(embedded, i, j)
    end

    grad_fo = ntuple(_ -> zeros(Float64, Nx, Ny), 4)
    grad_m55 = ntuple(_ -> zeros(Float64, Nx, Ny), 4)
    Kraken.fvfd_velocity_gradient_2d!(grad_fo..., ux, uy, is_solid, dx, dy, bc)
    Kraken.fvfd_velocity_gradient_embedded_2d!(grad_m55..., ux, uy, is_solid, dx, dy, bc, embedded)

    th_g = threshold(infnorm(grad_an, is_solid))
    push_component_rows!(rows, field, rot_label, "G1_gradient", "FO", COMPONENTS_G, grad_fo, grad_an, bins, is_solid, th_g)
    push_component_rows!(rows, field, rot_label, "G1_gradient", "M55b", COMPONENTS_G, grad_m55, grad_an, bins, is_solid, th_g)

    src_fo = oldroydb_source_arrays(grad_fo, lambda)
    src_m55 = oldroydb_source_arrays(grad_m55, lambda)
    th_s = threshold(infnorm(source_an, is_solid))
    push_component_rows!(rows, field, rot_label, "G2_source", "FO", COMPONENTS_S, src_fo, source_an, bins, is_solid, th_s)
    push_component_rows!(rows, field, rot_label, "G2_source", "M55b", COMPONENTS_S, src_m55, source_an, bins, is_solid, th_s)

    tau_fo = stress_from_source(src_fo, dt, model)
    tau_m55 = stress_from_source(src_m55, dt, model)
    th_t = threshold(infnorm(stress_an, is_solid))
    push_component_rows!(rows, field, rot_label, "G3_stress", "FO", COMPONENTS_S, tau_fo, stress_an, bins, is_solid, th_t)
    push_component_rows!(rows, field, rot_label, "G3_stress", "M55b", COMPONENTS_S, tau_m55, stress_an, bins, is_solid, th_t)

    force_fo = ntuple(_ -> zeros(Float64, Nx, Ny), 2)
    force_m55 = ntuple(_ -> zeros(Float64, Nx, Ny), 2)
    Kraken.fvfd_tensor_divergence_2d!(force_fo..., tau_fo..., is_solid, dx, dy, bc)
    Kraken.fvfd_tensor_divergence_embedded_2d!(force_m55..., tau_m55..., is_solid, dx, dy, bc, embedded)
    th_f = threshold(infnorm(force_an, is_solid))
    push_component_rows!(rows, field, rot_label, "G4_divergence", "FO", COMPONENTS_F, force_fo, force_an, bins, is_solid, th_f)
    push_component_rows!(rows, field, rot_label, "G4_divergence", "M55b", COMPONENTS_F, force_m55, force_an, bins, is_solid, th_f)
    return nothing
end

function write_per_cell(rows)
    mkpath(OUTDIR)
    open(PER_CELL_CSV, "w") do io
        println(io, "field,rotation,gate,cell_i,cell_j,helper,bin,component,computed,analytic,abs_err,threshold,verdict")
        for r in rows
            @printf(io, "%s,%s,%s,%d,%d,%s,%s,%s,%.17g,%.17g,%.17g,%.17g,%s\n",
                    r.field, r.rotation, r.gate, r.cell_i, r.cell_j, r.helper, r.bin, r.component,
                    r.computed, r.analytic, r.abs_err, r.threshold, r.verdict)
        end
    end
end

safe_ratio(a, b) = b == 0.0 ? (a == 0.0 ? 1.0 : Inf) : a / b

function write_aggregate(rows)
    vals = Dict{NTuple{5,String}, Vector{Float64}}()
    ths = Dict{NTuple{5,String}, Float64}()
    for r in rows
        key = (r.field, r.rotation, r.gate, r.bin, r.helper)
        push!(get!(vals, key, Float64[]), r.abs_err)
        ths[key] = r.threshold
    end
    stats = Dict{NTuple{5,String}, NamedTuple}()
    for (key, xs) in vals
        stats[key] = (n=length(xs), mean=mean(xs), max=maximum(xs), threshold=ths[key])
    end
    open(AGG_CSV, "w") do io
        println(io, "field,rotation,gate,bin,helper,n,mean_err,max_err,threshold,verdict,mean_ratio_m55_over_fo,max_ratio_m55_over_fo")
        for key in sort(collect(keys(stats)))
            field, rot, gate, bin, helper = key
            st = stats[key]
            fo = get(stats, (field, rot, gate, bin, "FO"), nothing)
            m55 = get(stats, (field, rot, gate, bin, "M55b"), nothing)
            mean_ratio = fo === nothing || m55 === nothing ? NaN : safe_ratio(m55.mean, fo.mean)
            max_ratio = fo === nothing || m55 === nothing ? NaN : safe_ratio(m55.max, fo.max)
            verdict = st.max > st.threshold ? "RED" : "OK"
            @printf(io, "%s,%s,%s,%s,%s,%d,%.17g,%.17g,%.17g,%s,%.17g,%.17g\n",
                    field, rot, gate, bin, helper, st.n, st.mean, st.max, st.threshold,
                    verdict, mean_ratio, max_ratio)
        end
    end
end

function print_summary(rows)
    worst = sort(rows; by=r -> r.abs_err, rev=true)[1]
    red = count(r -> r.verdict == "RED", rows)
    @printf("M58 rows=%d red_component_rows=%d worst=%s/%s/%s/%s/%s cell=(%d,%d) err=%.6g th=%.6g\n",
            length(rows), red, worst.field, worst.rotation, worst.gate, worst.helper,
            worst.component, worst.cell_i, worst.cell_j, worst.abs_err, worst.threshold)
    @printf("M58 per-cell csv: %s\nM58 aggregate csv: %s\n", PER_CELL_CSV, AGG_CSV)
end

function main()
    Nx = Ny = 32
    cx = cy = 16.5
    radius = 4.0
    q_wall, is_solid = Kraken.precompute_q_wall_cylinder(Nx, Ny, cx, cy, radius; FT=Float64)
    embedded = Kraken.fvfd_embedded_boundary_from_qwall_2d(q_wall; FT=Float64)
    bc = Kraken.fvfd_wallxwally_bcspec_2d()
    rows = NamedTuple[]
    for field in FIELDS, (rot_label, theta) in THETAS
        run_case!(rows, field, rot_label, theta, q_wall, is_solid, embedded, bc)
    end
    write_per_cell(rows)
    write_aggregate(rows)
    print_summary(rows)
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
