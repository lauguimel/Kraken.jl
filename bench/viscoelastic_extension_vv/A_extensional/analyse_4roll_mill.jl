#!/usr/bin/env julia

using Printf
using Statistics

include(joinpath(@__DIR__, "..", "ref", "analytic_4roll_mill.jl"))
include(joinpath(@__DIR__, "simple_png.jl"))

const M44_WI_A = (0.1, 0.3, 0.5, 1.0)
const M44_N_A = (32, 64, 96, 128)
const M44_BETA_A = (0.59, 0.80, 0.90)
const M44_ROOT_A = normpath(joinpath(@__DIR__, "..", "..", ".."))
const M44_PROFILES_A = joinpath(M44_ROOT_A, "bench", "scratch", "m44_vv_a", "profiles")
const M44_CASES_A = joinpath(@__DIR__, "csv", "cases")
const M44_RESULTS_A = joinpath(@__DIR__, "csv", "M44_VV_A_results.csv")
const M44_PLOTS_A = joinpath(M44_ROOT_A, "bench", "viscoelastic_extension_vv", "plots")

function read_numeric_csv(path)
    lines = filter(!isempty, readlines(path))
    header = split(lines[1], ",")
    data = zeros(Float64, length(lines) - 1, length(header))
    for (r, line) in enumerate(lines[2:end])
        vals = split(line, ",")
        for c in eachindex(header)
            data[r, c] = parse(Float64, vals[c])
        end
    end
    return header, data
end

function rowdict(path)
    h, d = read_numeric_csv(path)
    return Dict(h[i] => d[1, i] for i in eachindex(h))
end

function col(header, data, name)
    idx = findfirst(==(name), header)
    idx === nothing && error("missing column $(name)")
    return data[:, idx]
end

function r2_score(y, yhat)
    n = min(length(y), length(yhat))
    n == 0 && return NaN
    yy = y[1:n]; rr = yhat[1:n]
    m = mean(yy)
    den = sum((v - m)^2 for v in yy)
    num = sum((yy[i] - rr[i])^2 for i in 1:n)
    return den <= 1.0e-30 ? (num <= 1.0e-30 ? 1.0 : -Inf) : 1.0 - num / den
end

function interp_periodic(xs_ref, ys_ref, xs)
    n = length(xs_ref)
    dx = M44_TWO_PI / n
    out = similar(xs)
    for k in eachindex(xs)
        x = mod(xs[k], M44_TWO_PI)
        a = x / dx
        i0 = floor(Int, a) + 1
        frac = a - floor(a)
        i1 = mod1(i0, n)
        i2 = mod1(i0 + 1, n)
        out[k] = (1 - frac) * ys_ref[i1] + frac * ys_ref[i2]
    end
    return out
end

function profile_path(kind, wi, n, beta)
    return joinpath(M44_PROFILES_A, "$(kind)_$(m44_case_tag(wi, n, beta)).csv")
end

function load_profile(kind, wi, n, beta)
    return read_numeric_csv(profile_path(kind, wi, n, beta))
end

function profile_r2s(wi, n, beta)
    n == 128 && return Dict("ux" => NaN, "p" => NaN, "tauxx" => NaN, "tauxy" => NaN)
    out = Dict{String,Float64}()
    for field in ("ux", "p", "tauxx", "tauxy")
        vals = Float64[]
        for kind in ("centerline_x", "centerline_y")
            h, d = load_profile(kind, wi, n, beta)
            hr, dr = load_profile(kind, wi, 128, beta)
            xs = col(h, d, "s")
            y = col(h, d, field)
            yr = interp_periodic(col(hr, dr, "s"), col(hr, dr, field), xs)
            push!(vals, r2_score(y, yr))
        end
        out[field] = minimum(vals)
    end
    return out
end

function combined_l2_to_ref(wi, n, beta)
    n == 128 && return 0.0
    total = 0.0
    count = 0
    for kind in ("centerline_x", "centerline_y"), field in ("ux", "p", "tauxx", "tauxy")
        h, d = load_profile(kind, wi, n, beta)
        hr, dr = load_profile(kind, wi, 128, beta)
        xs = col(h, d, "s")
        y = col(h, d, field)
        yr = interp_periodic(col(hr, dr, "s"), col(hr, dr, field), xs)
        scale = max(std(yr), 1.0e-12)
        total += sum(((y .- yr) ./ scale) .^ 2)
        count += length(y)
    end
    return sqrt(total / max(count, 1))
end

function fit_convergence_p(wi, beta)
    ns = Float64[]
    errs = Float64[]
    for n in M44_N_A
        n == 128 && continue
        e = combined_l2_to_ref(wi, n, beta)
        if isfinite(e) && e > 0
            push!(ns, n)
            push!(errs, e)
        end
    end
    length(errs) < 2 && return NaN, Dict{Int,Float64}()
    x = log.(1.0 ./ ns)
    y = log.(errs)
    xm = mean(x); ym = mean(y)
    p = sum((x .- xm) .* (y .- ym)) / sum((x .- xm) .^ 2)
    return p, Dict(Int(ns[i]) => errs[i] for i in eachindex(ns))
end

function analytic_r2(wi, n, beta)
    wi > 0.3 && return NaN
    h, d = read_numeric_csv(profile_path("stagnation", wi, n, beta))
    t = col(h, d, "t")
    y = col(h, d, "psixx")
    ref = [analytic_psixx_stagnation(wi, ti) for ti in t]
    return r2_score(y, ref)
end

function make_rows()
    conv = Dict{Tuple{Float64,Float64},Float64}()
    errs = Dict{Tuple{Float64,Float64},Dict{Int,Float64}}()
    for wi in M44_WI_A, beta in M44_BETA_A
        p, ed = fit_convergence_p(wi, beta)
        conv[(wi, beta)] = p
        errs[(wi, beta)] = ed
    end
    rows = Vector{Dict{String,Any}}()
    for wi in M44_WI_A, beta in M44_BETA_A, n in M44_N_A
        tag = m44_case_tag(wi, n, beta)
        c = rowdict(joinpath(M44_CASES_A, "case_$(tag).csv"))
        r2p = profile_r2s(wi, n, beta)
        r2a = analytic_r2(wi, n, beta)
        cp = conv[(wi, beta)]
        finite = c["first_nonfinite_step"] == -1 && c["min_det_C"] > 0
        r2_ok = (wi > 0.3 || r2a >= 0.99) &&
                (n == 128 || minimum(values(r2p)) >= 0.99)
        p_ok = isfinite(cp) && cp >= 1.5
        status = finite ? (r2_ok && p_ok ? "PASS" : "FAIL") : "NaN"
        row = Dict{String,Any}(
            "wi" => wi, "n" => n, "beta" => beta,
            "max_u" => c["max_u"], "max_p" => c["max_p"],
            "max_tauxx" => c["max_tauxx"], "max_tauxy" => c["max_tauxy"],
            "max_tauyy" => c["max_tauyy"], "max_psixx" => c["max_psixx"],
            "max_N1" => c["max_N1"], "avg_u" => c["avg_u"], "avg_p" => c["avg_p"],
            "avg_tauxx" => c["avg_tauxx"], "avg_tauxy" => c["avg_tauxy"],
            "avg_tauyy" => c["avg_tauyy"], "avg_psixx" => c["avg_psixx"],
            "avg_N1" => c["avg_N1"], "min_det_C" => c["min_det_C"],
            "first_nonfinite_step" => c["first_nonfinite_step"],
            "R2_psi_xx_stag_vs_analytic" => r2a,
            "R2_centerline_u" => r2p["ux"], "R2_centerline_p" => r2p["p"],
            "R2_centerline_tauxx" => r2p["tauxx"], "R2_centerline_tauxy" => r2p["tauxy"],
            "convergence_p" => cp, "status" => status,
        )
        push!(rows, row)
    end
    return rows, conv, errs
end

function fmtval(x)
    x isa AbstractString && return x
    x isa Integer && return string(x)
    x isa Real || return string(x)
    isfinite(Float64(x)) || return "NaN"
    return @sprintf("%.10g", x)
end

function write_results(rows)
    mkpath(dirname(M44_RESULTS_A))
    cols = ["wi","n","beta","max_u","max_p","max_tauxx","max_tauxy","max_tauyy",
            "max_psixx","max_N1","avg_u","avg_p","avg_tauxx","avg_tauxy","avg_tauyy",
            "avg_psixx","avg_N1","min_det_C","first_nonfinite_step",
            "R2_psi_xx_stag_vs_analytic","R2_centerline_u","R2_centerline_p",
            "R2_centerline_tauxx","R2_centerline_tauxy","convergence_p","status"]
    open(M44_RESULTS_A, "w") do io
        println(io, join(cols, ","))
        for r in rows
            println(io, join((fmtval(r[c]) for c in cols), ","))
        end
    end
end

function panel_map(vals, lo, hi, a, b)
    hi <= lo && return (a + b) / 2
    return a + (vals - lo) * (b - a) / (hi - lo)
end

function convergence_plot(errs)
    mkpath(M44_PLOTS_A)
    img = m44_canvas(960, 720)
    for (ri, wi) in enumerate(M44_WI_A), (ci, beta) in enumerate(M44_BETA_A)
        x0 = 30 + (ci - 1) * 310; y0 = 30 + (ri - 1) * 170
        x1 = x0 + 280; y1 = y0 + 135
        m44_rect!(img, x0, y0, x1, y1, (0x99, 0x99, 0x99))
        ed = errs[(wi, beta)]
        ns = sort(collect(keys(ed)))
        isempty(ns) && continue
        xs = log.(1.0 ./ Float64.(ns)); ys = log.([ed[n] for n in ns])
        xl, xh = extrema(xs); yl, yh = extrema(ys)
        last = nothing
        for k in eachindex(ns)
            px = panel_map(xs[k], xl, xh, x0 + 20, x1 - 12)
            py = panel_map(ys[k], yl, yh, y1 - 18, y0 + 12)
            m44_line!(img, px - 3, py, px + 3, py, (0x00, 0x44, 0x99), width=3)
            last !== nothing && m44_line!(img, last[1], last[2], px, py, (0x00, 0x44, 0x99), width=2)
            last = (px, py)
        end
    end
    return m44_save_rgb_png(joinpath(M44_PLOTS_A, "M44_VV_A_convergence_grid.png"), img)
end

function profile_plot()
    img = m44_canvas(900, 720)
    for (ri, wi) in enumerate(M44_WI_A), (ci, kind) in enumerate(("centerline_x", "centerline_y"))
        x0 = 35 + (ci - 1) * 430; y0 = 28 + (ri - 1) * 170
        x1 = x0 + 390; y1 = y0 + 135
        m44_rect!(img, x0, y0, x1, y1, (0x99, 0x99, 0x99))
        h, d = load_profile(kind, wi, 128, 0.59)
        xs = col(h, d, "s")
        y = col(h, d, "tauxx")
        yl, yh = extrema(y)
        if wi <= 0.3
            ref = brief_steady_cxx_minus_one(wi) * (1.0 - 0.59) / wi
            yl = min(yl, ref); yh = max(yh, ref)
        end
        last = nothing
        for k in eachindex(xs)
            px = panel_map(xs[k], 0.0, M44_TWO_PI, x0 + 12, x1 - 12)
            py = panel_map(y[k], yl, yh, y1 - 15, y0 + 12)
            last !== nothing && m44_line!(img, last[1], last[2], px, py, (0x00, 0x77, 0x55), width=2)
            last = (px, py)
        end
        if wi <= 0.3
            ref = brief_steady_cxx_minus_one(wi) * (1.0 - 0.59) / wi
            py = panel_map(ref, yl, yh, y1 - 15, y0 + 12)
            m44_line!(img, x0 + 12, py, x1 - 12, py, (0xaa, 0x33, 0x33), width=1)
        end
    end
    return m44_save_rgb_png(joinpath(M44_PLOTS_A, "M44_VV_A_profiles_beta0p59_n128.png"), img)
end

function write_verdict(rows, conv, errs)
    pass_count = count(r -> r["status"] == "PASS", rows)
    fail_count = count(r -> r["status"] == "FAIL", rows)
    nan_count = count(r -> r["status"] == "NaN", rows)
    conv_png = convergence_plot(errs)
    prof_png = profile_plot()
    open(joinpath(@__DIR__, "VERDICT.md"), "w") do io
        println(io, "# M44-VV-A VERDICT")
        println(io)
        println(io, "Setup: periodic [0, 2pi]^2 4-roll mill, log-FV Oldroyd-B polymer chain, no walls, no solid mask, no BSD/cut-cell path.")
        println(io, "Matrix: Wi={0.1,0.3,0.5,1.0}, N={32,64,96,128}, beta={0.59,0.80,0.90}; 48 cases.")
        println(io, "Note: for the specified velocity formula, the listed pi/2 points are elliptic; the analytic gate samples the actual x-extensional stagnation point at (0,0).")
        println(io)
        println(io, "## 48-case matrix")
        println(io, "| Wi | beta | PASS | FAIL | NaN | convergence p |")
        println(io, "|---:|---:|---:|---:|---:|---:|")
        for wi in M44_WI_A, beta in M44_BETA_A
            sub = filter(r -> r["wi"] == wi && r["beta"] == beta, rows)
            println(io, @sprintf("| %.2f | %.2f | %d | %d | %d | %.3f |",
                wi, beta, count(r -> r["status"] == "PASS", sub),
                count(r -> r["status"] == "FAIL", sub), count(r -> r["status"] == "NaN", sub),
                conv[(wi, beta)]))
        end
        println(io)
        println(io, "## Plots")
        println(io, "- Convergence grid: `$(relpath(conv_png, @__DIR__))`")
        println(io, "- Profile grid: `$(relpath(prof_png, @__DIR__))`")
        println(io)
        println(io, "## Verdict synthesis")
        println(io, "- Overall: $(pass_count)/48 PASS, $(fail_count) FAIL, $(nan_count) NaN.")
        for wi in M44_WI_A
            vals = [conv[(wi, b)] for b in M44_BETA_A]
            println(io, @sprintf("- Wi %.2f convergence p range: %.3f to %.3f.", wi, minimum(vals), maximum(vals)))
        end
        high_fail = count(r -> r["wi"] >= 0.5 && r["status"] != "PASS", rows)
        low_fail = count(r -> r["wi"] <= 0.3 && r["status"] != "PASS", rows)
        println(io, "- Wi-dependent signature: low-Wi non-PASS count $(low_fail); Wi>=0.5 non-PASS count $(high_fail).")
        println(io, "- beta-dependent signature: inspect PASS fractions above against (1-beta); stronger low-beta failures indicate polymer-coupling scaling.")
        println(io, "- De Gennes note: Wi=1 is above the 0.5 coil-stretch threshold; stagnation growth is treated as a risk signature, not an automatic NaN failure.")
        rec = pass_count == 48 ? "chain is GREEN in extension -> proceed to Tests B + C" :
              "chain is RED in extension -> fundamental polymer-pressure coupling/advection issue should be isolated before Tests B + C"
        println(io, "- Recommendation to Boss: $(rec).")
    end
end

function main()
    rows, conv, errs = make_rows()
    write_results(rows)
    write_verdict(rows, conv, errs)
    println("analysed $(length(rows)) cases")
    println("PASS rows: $(count(r -> r["status"] == "PASS", rows))")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
