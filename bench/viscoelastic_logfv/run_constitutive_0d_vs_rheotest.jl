#!/usr/bin/env julia

# 0D constitutive integration of Kraken's log-FV Oldroyd-B step on a single
# cell under prescribed shear rate gamma. Compares trajectory and steady
# state against:
#   1. Closed-form Oldroyd-B simple-shear (from rest, C(0) = I):
#        Cyy(t) = 1
#        Cxy(t) = gamma * lambda * (1 - exp(-t/lambda))
#        Cxx(t) = 1 + 2*(gamma*lambda)^2 * (1 - exp(-t/lambda) - (t/lambda)*exp(-t/lambda))
#   2. rheoTool rheoTestFoam output (project-local case in
#      bench/rheotool/rheotest_oldroydb), loaded if present.
#
# Purpose: bisect the cavity discrepancy. If Kraken's constitutive matches
# both the analytical and rheoTool at every gamma, the bug is in spatial
# coupling. If it doesn't, the bug is in the constitutive itself or its
# substep scheme.

using Printf
using DelimitedFiles
using Dates

using Kraken

# ---------------------------------------------------------------------
# Analytical Oldroyd-B simple-shear (homogeneous, from C(0) = I)
# ---------------------------------------------------------------------

function oldroydb_simple_shear_analytical(gamma::Real, lambda::Real, t::Real)
    if t <= 0
        return (Cxx=1.0, Cxy=0.0, Cyy=1.0)
    end
    s = t / lambda
    exps = exp(-s)
    Cxy = gamma * lambda * (1 - exps)
    Cxx = 1.0 + 2.0 * (gamma * lambda)^2 * (1 - exps - s * exps)
    Cyy = 1.0
    return (Cxx=Cxx, Cxy=Cxy, Cyy=Cyy)
end

# Planar elongation: gradU = epsilon_dot * diag(1, -1) in 2D (incompressible).
# Closed-form solution from rest C(0) = I requires De = epsilon_dot * lambda < 0.5
# (otherwise polymer stretches without bound for Oldroyd-B).
function oldroydb_planar_extension_analytical(eps_dot::Real, lambda::Real, t::Real)
    if t <= 0
        return (Cxx=1.0, Cxy=0.0, Cyy=1.0)
    end
    De = eps_dot * lambda
    abs(De) < 0.5 - 1e-12 ||
        error("oldroydb_planar_extension_analytical: De=$(De) outside |De|<0.5 stability bound for Oldroyd-B")
    inv_t_xx = (1.0 / lambda) - 2.0 * eps_dot
    inv_t_yy = (1.0 / lambda) + 2.0 * eps_dot
    Cxx_ss = 1.0 / (1.0 - 2.0 * De)
    Cyy_ss = 1.0 / (1.0 + 2.0 * De)
    Cxx = Cxx_ss + (1.0 - Cxx_ss) * exp(-inv_t_xx * t)
    Cyy = Cyy_ss + (1.0 - Cyy_ss) * exp(-inv_t_yy * t)
    return (Cxx=Cxx, Cxy=0.0, Cyy=Cyy)
end

# ---------------------------------------------------------------------
# Kraken 0D loop
# ---------------------------------------------------------------------

function kraken_0d_shear(gamma::Real, lambda::Real, t_end::Real;
                         dt_target::Real=0.0, max_substep_factor::Real=0.01,
                         sample_every::Int=100, T::Type=Float64,
                         flow::Symbol=:shear)
    psixx = zero(T); psixy = zero(T); psiyy = zero(T)
    if flow === :shear
        dudx = zero(T); dudy = T(gamma); dvdx = zero(T); dvdy = zero(T)
    elseif flow === :planar_extension
        # gradU = gamma * diag(1, -1), incompressible 2D planar elongation
        dudx = T(gamma); dudy = zero(T); dvdx = zero(T); dvdy = -T(gamma)
    else
        error("unsupported flow=$(flow); expected :shear or :planar_extension")
    end
    lam = T(lambda)

    # Pick a dt small enough that the polymer relaxation per step is
    # well-resolved and the memory deformation stays bounded:
    # dt_relax_cap = max_substep_factor * lambda
    # dt_def_cap   = max_substep_factor / max(|gamma|, eps)
    safe_dt = T(max_substep_factor) * min(lam, one(T) / max(abs(T(gamma)), T(1e-12)))
    dt = dt_target > 0 ? T(dt_target) : safe_dt

    n_steps = max(1, ceil(Int, t_end / dt))
    dt = T(t_end) / T(n_steps)

    Cxx_traj = Float64[1.0]
    Cxy_traj = Float64[0.0]
    Cyy_traj = Float64[1.0]
    times = Float64[0.0]

    t_start = time()
    for k in 1:n_steps
        psixx, psixy, psiyy = Kraken.logfv_oldroydb_step_log_2d(
            psixx, psixy, psiyy, dudx, dudy, dvdx, dvdy, lam, dt,
        )
        if k == n_steps || (k % sample_every == 0)
            Cxx, Cxy, Cyy = Kraken.logfv_exp_sym2_2d(psixx, psixy, psiyy)
            push!(times, Float64(k * dt))
            push!(Cxx_traj, Float64(Cxx))
            push!(Cxy_traj, Float64(Cxy))
            push!(Cyy_traj, Float64(Cyy))
        end
    end
    elapsed = time() - t_start

    return (
        gamma=gamma, lambda=lambda, t_end=t_end,
        n_steps=n_steps, dt=Float64(dt),
        elapsed_s=elapsed,
        steps_per_sec=n_steps / elapsed,
        times=times, Cxx=Cxx_traj, Cxy=Cxy_traj, Cyy=Cyy_traj,
    )
end

function kraken_0d_steady_state(gamma::Real, lambda::Real;
                                 settle_factor::Real=10.0, kwargs...)
    res = kraken_0d_shear(gamma, lambda, settle_factor * lambda; kwargs...)
    return (
        gamma=gamma, lambda=lambda,
        Cxx=res.Cxx[end], Cxy=res.Cxy[end], Cyy=res.Cyy[end],
        elapsed_s=res.elapsed_s, n_steps=res.n_steps, dt=res.dt,
        steps_per_sec=res.steps_per_sec,
    )
end

# ---------------------------------------------------------------------
# Compare against analytical at sample times
# ---------------------------------------------------------------------

function compare_kraken_vs_analytical(kraken_run, gamma::Real, lambda::Real;
                                       flow::Symbol=:shear)
    analytical_fn = if flow === :shear
        oldroydb_simple_shear_analytical
    elseif flow === :planar_extension
        oldroydb_planar_extension_analytical
    else
        error("unsupported flow=$(flow)")
    end
    n = length(kraken_run.times)
    Cxx_an = Vector{Float64}(undef, n)
    Cxy_an = Vector{Float64}(undef, n)
    Cyy_an = Vector{Float64}(undef, n)
    for (k, t) in enumerate(kraken_run.times)
        an = analytical_fn(gamma, lambda, t)
        Cxx_an[k] = an.Cxx
        Cxy_an[k] = an.Cxy
        Cyy_an[k] = an.Cyy
    end

    function rel_err(kr, an)
        scale = maximum(abs, an)
        scale < eps() && return 0.0
        return maximum(abs.(kr .- an)) ./ scale
    end

    return (
        Cxx_an=Cxx_an, Cxy_an=Cxy_an, Cyy_an=Cyy_an,
        rel_linf_Cxx=rel_err(kraken_run.Cxx, Cxx_an),
        rel_linf_Cxy=rel_err(kraken_run.Cxy, Cxy_an),
        rel_linf_Cyy=rel_err(kraken_run.Cyy, Cyy_an),
    )
end

# ---------------------------------------------------------------------
# rheoTool Report loader (output of rheoTestFoam ramp=true)
# ---------------------------------------------------------------------

function load_rheotool_report(case_dir::AbstractString)
    report_path = joinpath(case_dir, "Report")
    isfile(report_path) || return nothing
    raw = try
        readdlm(report_path, comments=true, comment_char='#')
    catch err
        @warn "Failed to parse rheoTool Report at $(report_path): $(err)"
        return nothing
    end
    return raw
end

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

function main()
    lambda_val = parse(Float64, get(ENV, "KRAKEN_OLDROYDB_LAMBDA", "1.0"))
    gamma_list = parse.(Float64, split(get(ENV, "KRAKEN_GAMMA_LIST", "0.01,0.1,1.0,10.0,100.0"), ','))
    t_factor = parse(Float64, get(ENV, "KRAKEN_TEND_FACTOR", "12.0"))
    output_dir = get(ENV, "KRAKEN_OUTPUT_DIR", joinpath("tmp", "constitutive_0d"))
    rheotool_dir = get(ENV, "KRAKEN_RHEOTOOL_CASE", joinpath("bench", "rheotool", "rheotest_oldroydb"))
    # Optional explicit dt/lambda override (used to match production cavity
    # substep cadence dt_sub/lambda_LU ~= 4e-8).
    dt_over_lambda_raw = get(ENV, "KRAKEN_DT_OVER_LAMBDA", "")
    dt_over_lambda = isempty(dt_over_lambda_raw) ? 0.0 : parse(Float64, dt_over_lambda_raw)
    # Optional sample stride
    sample_every = parse(Int, get(ENV, "KRAKEN_SAMPLE_EVERY", "0"))
    # Flow type: :shear or :planar_extension
    flow_sym = Symbol(get(ENV, "KRAKEN_FLOW", "shear"))

    mkpath(output_dir)
    stamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    run_dir = joinpath(output_dir, "constitutive_0d_$(stamp)")
    mkpath(run_dir)

    println("=== Kraken Oldroyd-B 0D constitutive vs analytical sweep ===")
    @printf("lambda=%.3g  t_end=%.3g*lambda  gamma_list=%s\n",
            lambda_val, t_factor, string(gamma_list))

    summary_rows = NamedTuple{(:gamma, :Cxx_kraken, :Cxx_analytical, :Cxy_kraken, :Cxy_analytical,
                                :rel_linf_Cxx, :rel_linf_Cxy, :rel_linf_Cyy,
                                :n_steps, :dt, :elapsed_s, :steps_per_sec),
                               Tuple{Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Int,Float64,Float64,Float64}}[]

    for gamma in gamma_list
        dt_explicit = dt_over_lambda > 0 ? dt_over_lambda * lambda_val : 0.0
        sample_every_use = sample_every > 0 ? sample_every : max(1, div(round(Int, t_factor * lambda_val / (dt_explicit > 0 ? dt_explicit : 0.01)), 200))
        res = kraken_0d_shear(gamma, lambda_val, t_factor * lambda_val;
                              dt_target=dt_explicit, sample_every=sample_every_use,
                              flow=flow_sym)
        cmp = compare_kraken_vs_analytical(res, gamma, lambda_val; flow=flow_sym)
        an_end = if flow_sym === :shear
            oldroydb_simple_shear_analytical(gamma, lambda_val, t_factor * lambda_val)
        else
            oldroydb_planar_extension_analytical(gamma, lambda_val, t_factor * lambda_val)
        end
        push!(summary_rows, (
            gamma=gamma,
            Cxx_kraken=res.Cxx[end],
            Cxx_analytical=an_end.Cxx,
            Cxy_kraken=res.Cxy[end],
            Cxy_analytical=an_end.Cxy,
            rel_linf_Cxx=cmp.rel_linf_Cxx,
            rel_linf_Cxy=cmp.rel_linf_Cxy,
            rel_linf_Cyy=cmp.rel_linf_Cyy,
            n_steps=res.n_steps,
            dt=res.dt,
            elapsed_s=res.elapsed_s,
            steps_per_sec=res.steps_per_sec,
        ))
        @printf("  gamma=%-8g | Cxx_K=%-14.6g Cxx_A=%-14.6g | Cxy_K=%-12.6g Cxy_A=%-12.6g | err_Linf Cxx=%.2e Cxy=%.2e | dt=%.2e steps=%d wall=%.3fs\n",
                gamma, res.Cxx[end], an_end.Cxx, res.Cxy[end], an_end.Cxy,
                cmp.rel_linf_Cxx, cmp.rel_linf_Cxy,
                res.dt, res.n_steps, res.elapsed_s)

        # Dump trajectory CSV
        csv_path = joinpath(run_dir, @sprintf("trajectory_gamma_%g.csv", gamma))
        open(csv_path, "w") do io
            write(io, "t,Cxx_kraken,Cxy_kraken,Cyy_kraken,Cxx_analytical,Cxy_analytical,Cyy_analytical\n")
            for k in eachindex(res.times)
                @printf(io, "%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g\n",
                        res.times[k], res.Cxx[k], res.Cxy[k], res.Cyy[k],
                        cmp.Cxx_an[k], cmp.Cxy_an[k], cmp.Cyy_an[k])
            end
        end
    end

    summary_csv = joinpath(run_dir, "summary.csv")
    open(summary_csv, "w") do io
        write(io, "gamma,Cxx_kraken,Cxx_analytical,Cxy_kraken,Cxy_analytical,rel_linf_Cxx,rel_linf_Cxy,rel_linf_Cyy,n_steps,dt,elapsed_s,steps_per_sec\n")
        for r in summary_rows
            @printf(io, "%.10g,%.10g,%.10g,%.10g,%.10g,%.10e,%.10e,%.10e,%d,%.10g,%.6g,%.6g\n",
                    r.gamma, r.Cxx_kraken, r.Cxx_analytical, r.Cxy_kraken, r.Cxy_analytical,
                    r.rel_linf_Cxx, r.rel_linf_Cxy, r.rel_linf_Cyy,
                    r.n_steps, r.dt, r.elapsed_s, r.steps_per_sec)
        end
    end

    # Try loading rheoTool Report
    rt = load_rheotool_report(rheotool_dir)
    if rt !== nothing
        rt_path = joinpath(run_dir, "rheotool_report_copy.dat")
        writedlm(rt_path, rt)
        println("\nrheoTool Report copied to $(rt_path) ($(size(rt, 1)) rows)")
    else
        missing_path = joinpath(rheotool_dir, "Report")
        println("\nrheoTool Report not found at $(missing_path); run the case first to compare.")
    end

    println("\nOutput directory: $(run_dir)")
    println("Summary: $(summary_csv)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
