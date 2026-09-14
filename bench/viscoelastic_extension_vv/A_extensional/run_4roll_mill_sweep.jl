#!/usr/bin/env julia

using KernelAbstractions
using Printf

using Kraken

include(joinpath(@__DIR__, "..", "ref", "analytic_4roll_mill.jl"))

const M44_WI = (0.1, 0.3, 0.5, 1.0)
const M44_N = (32, 64, 96, 128)
const M44_BETA = (0.59, 0.80, 0.90)
const M44_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const M44_SCRATCH = joinpath(M44_ROOT, "bench", "scratch", "m44_vv_a")
const M44_PROFILE_DIR = joinpath(M44_SCRATCH, "profiles")
const M44_CASE_DIR = joinpath(@__DIR__, "csv", "cases")

function parse_args(args)
    opts = Dict{String,Any}("all" => false, "smoke" => false)
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--all"
            opts["all"] = true
        elseif a == "--smoke"
            opts["smoke"] = true
        elseif a in ("--wi", "--n", "--beta", "--max-time")
            i == length(args) && error("missing value for $(a)")
            opts[a[3:end]] = args[i + 1]
            i += 1
        else
            error("unrecognised argument: $(a)")
        end
        i += 1
    end
    return opts
end

function ensure_dirs()
    mkpath(M44_PROFILE_DIR)
    mkpath(joinpath(M44_SCRATCH, "tmp"))
    mkpath(M44_CASE_DIR)
    mkpath(joinpath(M44_ROOT, "tmp", "m44_vv_a"))
    return nothing
end

function init_fourroll!(ux, uy, n)
    for j in 1:n, i in 1:n
        x = fourroll_x(i, n)
        y = fourroll_x(j, n)
        ux[i, j], uy[i, j] = fourroll_velocity(x, y)
    end
    return nothing
end

function init_lbm_equilibrium!(f, ux_phys, uy_phys, mach)
    n = size(ux_phys, 1)
    lat = Kraken.D2Q9()
    for q in 1:9, j in 1:n, i in 1:n
        ux = mach * ux_phys[i, j]
        uy = mach * uy_phys[i, j]
        f[i, j, q] = Kraken.equilibrium(lat, 1.0, ux, uy, q)
    end
    return nothing
end

function scale_field!(out, a, s)
    @inbounds for idx in eachindex(out, a)
        out[idx] = s * a[idx]
    end
    return nothing
end

function exp_c_components(psixx, psixy, psiyy, i, j)
    return Kraken.logfv_exp_sym2_2d(psixx[i, j], psixy[i, j], psiyy[i, j])
end

function field_stats(ux, uy, p, psixx, psixy, psiyy, tauxx, tauxy, tauyy)
    n = size(ux, 1)
    max_u = max_p = max_txx = max_txy = max_tyy = max_psixx = max_n1 = 0.0
    sum_u = sum_p = sum_txx = sum_txy = sum_tyy = sum_psixx = sum_n1 = 0.0
    min_det_c = Inf
    for j in 1:n, i in 1:n
        u = hypot(ux[i, j], uy[i, j])
        n1 = tauxx[i, j] - tauyy[i, j]
        cxx, cxy, cyy = exp_c_components(psixx, psixy, psiyy, i, j)
        detc = cxx * cyy - cxy * cxy
        max_u = max(max_u, abs(u)); sum_u += abs(u)
        max_p = max(max_p, abs(p[i, j])); sum_p += abs(p[i, j])
        max_txx = max(max_txx, abs(tauxx[i, j])); sum_txx += abs(tauxx[i, j])
        max_txy = max(max_txy, abs(tauxy[i, j])); sum_txy += abs(tauxy[i, j])
        max_tyy = max(max_tyy, abs(tauyy[i, j])); sum_tyy += abs(tauyy[i, j])
        max_psixx = max(max_psixx, abs(psixx[i, j])); sum_psixx += abs(psixx[i, j])
        max_n1 = max(max_n1, abs(n1)); sum_n1 += abs(n1)
        min_det_c = min(min_det_c, detc)
    end
    inv_count = inv(float(n * n))
    return (;
        max_u, max_p, max_tauxx=max_txx, max_tauxy=max_txy, max_tauyy=max_tyy,
        max_psixx, max_N1=max_n1,
        avg_u=sum_u * inv_count, avg_p=sum_p * inv_count,
        avg_tauxx=sum_txx * inv_count, avg_tauxy=sum_txy * inv_count,
        avg_tauyy=sum_tyy * inv_count, avg_psixx=sum_psixx * inv_count,
        avg_N1=sum_n1 * inv_count, min_det_C=min_det_c,
    )
end

function pressure_from_rho(rho, mach)
    p = similar(rho)
    rho0 = sum(rho) / length(rho)
    @inbounds for idx in eachindex(rho)
        p[idx] = (rho[idx] - rho0) / (3.0 * mach * mach)
    end
    return p
end

function write_profile(path, s, ux, p, tauxx, tauxy)
    open(path, "w") do io
        println(io, "s,ux,p,tauxx,tauxy")
        for k in eachindex(s)
            @printf(io, "%.17g,%.17g,%.17g,%.17g,%.17g\n", s[k], ux[k], p[k], tauxx[k], tauxy[k])
        end
    end
    return path
end

function write_outputs(tag, wi, n, beta, max_time, steps, walltime, stats, first_bad,
                       ux, uy, p, psixx, psixy, psiyy, tauxx, tauxy, tauyy, stag_rows)
    dx = M44_TWO_PI / n
    s = [fourroll_x(i, n) for i in 1:n]
    ix = div(n, 4) + 1
    iy = div(n, 4) + 1
    diag_ux = [ux[i, i] for i in 1:n]
    diag_p = [p[i, i] for i in 1:n]
    diag_txx = [tauxx[i, i] for i in 1:n]
    diag_txy = [tauxy[i, i] for i in 1:n]
    write_profile(joinpath(M44_PROFILE_DIR, "centerline_x_$(tag).csv"), s,
                  [ux[i, iy] for i in 1:n], [p[i, iy] for i in 1:n],
                  [tauxx[i, iy] for i in 1:n], [tauxy[i, iy] for i in 1:n])
    write_profile(joinpath(M44_PROFILE_DIR, "centerline_y_$(tag).csv"), s,
                  [ux[ix, j] for j in 1:n], [p[ix, j] for j in 1:n],
                  [tauxx[ix, j] for j in 1:n], [tauxy[ix, j] for j in 1:n])
    write_profile(joinpath(M44_PROFILE_DIR, "diagonal_$(tag).csv"), s, diag_ux, diag_p, diag_txx, diag_txy)
    open(joinpath(M44_PROFILE_DIR, "stagnation_$(tag).csv"), "w") do io
        println(io, "t,psixx,cxx_minus_one,analytic_psixx,analytic_cxx_minus_one")
        for r in stag_rows
            @printf(io, "%.17g,%.17g,%.17g,%.17g,%.17g\n", r...)
        end
    end
    open(joinpath(M44_CASE_DIR, "case_$(tag).csv"), "w") do io
        println(io, join(("wi","n","beta","max_time","steps","walltime_s","max_u","max_p",
                          "max_tauxx","max_tauxy","max_tauyy","max_psixx","max_N1",
                          "avg_u","avg_p","avg_tauxx","avg_tauxy","avg_tauyy",
                          "avg_psixx","avg_N1","min_det_C","first_nonfinite_step"), ","))
        vals = (wi, n, beta, max_time, steps, walltime, stats.max_u, stats.max_p,
                stats.max_tauxx, stats.max_tauxy, stats.max_tauyy, stats.max_psixx, stats.max_N1,
                stats.avg_u, stats.avg_p, stats.avg_tauxx, stats.avg_tauxy, stats.avg_tauyy,
                stats.avg_psixx, stats.avg_N1, stats.min_det_C, first_bad)
        println(io, join(map(x -> @sprintf("%.17g", x), vals), ","))
    end
    return dx
end

function run_case(; wi, n, beta, max_time=nothing, smoke=false)
    ensure_dirs()
    t_end = max_time === nothing ? 5.0 * wi : Float64(max_time)
    tag = m44_case_tag(wi, n, beta)
    t0 = time()
    dx = M44_TWO_PI / n
    dt_base = min(0.05 * wi, 0.25 * dx)
    sample_dt = 0.1 * wi
    mach = 0.04
    nu_lu = mach / dx
    omega = 1.0 / (3.0 * nu_lu + 0.5)
    prefactor = (1.0 - beta) / wi
    backend = KernelAbstractions.CPU()
    bc = Kraken.FVFDDomainBC2D(; west=:periodic, east=:periodic, south=:periodic, north=:periodic)

    is_solid = falses(n, n)
    ux = zeros(Float64, n, n); uy = zeros(Float64, n, n)
    init_fourroll!(ux, uy, n)
    f = zeros(Float64, n, n, 9); fbuf = similar(f)
    init_lbm_equilibrium!(f, ux, uy, mach)
    rho = ones(Float64, n, n); ux_lbm = zeros(Float64, n, n); uy_lbm = zeros(Float64, n, n)
    psixx = zeros(Float64, n, n); psixy = zeros(Float64, n, n); psiyy = zeros(Float64, n, n)
    psixx_adv = similar(psixx); psixy_adv = similar(psixx); psiyy_adv = similar(psixx)
    psixx_next = similar(psixx); psixy_next = similar(psixx); psiyy_next = similar(psixx)
    ux_face = zeros(Float64, n + 1, n); uy_face = zeros(Float64, n, n + 1)
    dudx = similar(psixx); dudy = similar(psixx); dvdx = similar(psixx); dvdy = similar(psixx)
    tauxx = similar(psixx); tauxy = similar(psixx); tauyy = similar(psixx)
    tauxx_lu = similar(psixx); tauxy_lu = similar(psixx); tauyy_lu = similar(psixx)
    dummy_y = zeros(Float64, n); dummy_x = zeros(Float64, n)
    stag = extensional_stagnation_indices(n)[1]
    stag_rows = Vector{NTuple{5,Float64}}()
    next_sample = sample_dt
    first_bad = -1
    steps = 0
    t = 0.0
    while t < t_end - 1.0e-12
        dt = min(dt_base, t_end - t, next_sample - t)
        Kraken.logfv_cell_velocity_to_faces_bc_aware_2d!(
            ux_face, uy_face, ux, uy, is_solid, dummy_y, dummy_y, dummy_x, dummy_x, bc; sync=false)
        Kraken.logfv_advect_upwind_bc_aware_2d!(
            psixx_adv, psixy_adv, psiyy_adv, psixx, psixy, psiyy,
            dummy_y, dummy_y, dummy_y, dummy_y, dummy_y, dummy_y,
            dummy_x, dummy_x, dummy_x, dummy_x, dummy_x, dummy_x,
            ux_face, uy_face, is_solid, dx, dx, bc, dt; sync=false)
        Kraken.fvfd_velocity_gradient_2d!(dudx, dudy, dvdx, dvdy, ux, uy, is_solid, dx, dx, bc; sync=false)
        nsub = Kraken.logfv_recommended_oldroydb_substeps(1.0, wi, dt; relative_tolerance=0.005)
        wx, wy, wz = psixx_adv, psixy_adv, psiyy_adv
        subdt = dt / nsub
        for _ in 1:nsub
            Kraken.logfv_step_constitutive_log_2d!(
                psixx_next, psixy_next, psiyy_next, wx, wy, wz,
                dudx, dudy, dvdx, dvdy, wi, subdt, Kraken.LOGFV_MODEL_OLDROYDB, 0.0; sync=false)
            wx, psixx_next = psixx_next, wx
            wy, psixy_next = psixy_next, wy
            wz, psiyy_next = psiyy_next, wz
        end
        psixx, psixx_adv = wx, psixx
        psixy, psixy_adv = wy, psixy
        psiyy, psiyy_adv = wz, psiyy
        Kraken.logfv_stress_from_log_2d!(tauxx, tauxy, tauyy, psixx, psixy, psiyy, prefactor; sync=true)
        scale_field!(tauxx_lu, tauxx, mach * mach)
        scale_field!(tauxy_lu, tauxy, mach * mach)
        scale_field!(tauyy_lu, tauyy, mach * mach)
        Kraken.stream_fully_periodic_2d!(fbuf, f, n, n)
        Kraken.collide_viscoelastic_source_2d!(fbuf, is_solid, omega, tauxx_lu, tauxy_lu, tauyy_lu)
        Kraken.compute_macroscopic_2d!(rho, ux_lbm, uy_lbm, fbuf; sync=true)
        f, fbuf = fbuf, f
        t += dt
        steps += 1
        if first_bad == -1 && !(all(isfinite, psixx) && all(isfinite, rho))
            first_bad = steps
            break
        end
        if abs(t - next_sample) < 1.0e-10 || t > next_sample
            i, j = stag
            cxx, _, _ = exp_c_components(psixx, psixy, psiyy, i, j)
            push!(stag_rows, (t, psixx[i, j], cxx - 1.0,
                              analytic_psixx_stagnation(wi, t),
                              analytic_cxx_minus_one_stagnation(wi, t)))
            next_sample += sample_dt
        end
    end
    p = pressure_from_rho(rho, mach)
    stats = field_stats(ux, uy, p, psixx, psixy, psiyy, tauxx, tauxy, tauyy)
    walltime = time() - t0
    write_outputs(tag, wi, n, beta, t_end, steps, walltime, stats, first_bad,
                  ux, uy, p, psixx, psixy, psiyy, tauxx, tauxy, tauyy, stag_rows)
    if smoke
        vals = [r[2] for r in stag_rows]
        monotone = length(vals) > 1 && all(diff(vals) .>= -1.0e-10) && vals[end] > vals[1]
        finite = first_bad == -1 && isfinite(stats.min_det_C) && stats.min_det_C > 0
        qualitative = stats.max_tauxx > 0 && stats.max_u > 0 && length(stag_rows) >= 4
        println("SMOKE no_nan ", finite ? "PASS" : "FAIL")
        println("SMOKE psixx_monotone ", monotone ? "PASS" : "FAIL")
        println("SMOKE profiles_qualitative ", qualitative ? "PASS" : "FAIL")
        return finite && monotone && qualitative
    end
    println(@sprintf("case %s steps=%d walltime=%.2fs first_nonfinite=%d min_det_C=%.6e",
                     tag, steps, walltime, first_bad, stats.min_det_C))
    return first_bad == -1
end

function main(args=ARGS)
    opts = parse_args(args)
    if opts["smoke"]
        ok = run_case(; wi=0.1, n=32, beta=0.59, max_time=0.05, smoke=true)
        exit(ok ? 0 : 1)
    elseif opts["all"]
        ok = true
        for wi in M44_WI, beta in M44_BETA, n in M44_N
            ok &= run_case(; wi, n, beta)
        end
        exit(ok ? 0 : 1)
    else
        for k in ("wi", "n", "beta")
            haskey(opts, k) || error("missing --$(k)")
        end
        wi = parse(Float64, opts["wi"])
        n = parse(Int, opts["n"])
        beta = parse(Float64, opts["beta"])
        max_time = haskey(opts, "max-time") ? parse(Float64, opts["max-time"]) : nothing
        ok = run_case(; wi, n, beta, max_time)
        exit(ok ? 0 : 1)
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
