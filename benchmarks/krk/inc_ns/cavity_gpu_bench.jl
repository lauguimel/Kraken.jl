# Aqua-ready GPU ABLATION benchmark for the backend-parametric SIMPLE cavity
# solver (src/methods/inc_ns/cavity_mg.jl): per-change contribution table for
# the GPU-efficiency features, Re=1000 on 512^2 (A100).
#
# WHAT IT DOES
# ------------
# One CPU REFERENCE run (legacy kwargs) provides the speed-up denominator, the
# parity reference fields, and the Ghia sanity anchor. Then CUMULATIVE GPU
# configs (each adds ONE feature on top of the previous):
#   C0  baseline-legacy        norm_stride=1, mg_cycles=0 (tolerance-driven
#                              inner MG), no mixed precision, no graph
#   C1  +norm_stride=25        outer convergence norms every 25 iters
#   C2  +fixed MG cycles       mg_cycles=3 (pressure), mom_mg_cycles=1
#   C3  +CUDA graph            solve_incns_cavity_mg_cuda_graph (off-stride
#                              outer iterations replayed from a captured graph;
#                              forces static_gauge=true)
#   C4  +mixed precision       mg_mixed_precision + mom_mg_mixed_precision
#                              (F32 V-cycles + F64 defect correction), on top
#                              of the graph path
# Per config it reports: wall time, outer iters, converged?, speed-up vs the
# CPU reference, max-RELATIVE parity vs the CPU reference fields u/v/p
# (target < 1e-3), Ghia (1982) centreline max error %, and GPU utilization
# (mean + peak, sampled at 1 Hz by a background nvidia-smi process started/
# stopped around the TIMED SOLVE WINDOW only; per-config log files in CWD).
# A final machine-parseable markdown table goes to stdout.
#
# The graph wrapper (src/methods/inc_ns/cavity_mg_cuda.jl) is NOT part of the
# default solver include; this bench `include`s it at top level AFTER
# `using CUDA` (world-age pattern below). If graph capture fails on the A100
# the wrapper degrades to plain per-launch execution (surfaced in the `path`
# column); if the C3/C4 call itself throws, the row reports FAILED(<err>) and
# the bench continues with the remaining configs.
#
# RUNNING
# -------
#   GPU (Aqua):    julia --project=. benchmarks/krk/inc_ns/cavity_gpu_bench.jl
#   local smoke:   CAVITY_BENCH_SMOKE=1 julia --project=. benchmarks/krk/inc_ns/cavity_gpu_bench.jl
#       smoke = 64^2 Re=100 CPU-only; runs C0->C2 and C4 (C3 + util sampling
#       are skipped gracefully without CUDA), same summary table, < 2 min.
#   CAVITY_BENCH_1024=1     additionally run 1024^2 GPU-only with the best
#       config (C4, falling back to C3/C2 if C4 failed). No CPU 1024^2.
#   CAVITY_BENCH_1024_C0=1  also run C0 at 1024^2 for contrast (~2 h alone;
#       only enable when the walltime budget allows).

using LinearAlgebra
using Printf

const _SRC = joinpath(@__DIR__, "..", "..", "..", "src")

# The backend-parametric SIMPLE cavity solver. It includes the matrix-free MG
# (src/solve/poisson_mg.jl) itself if not present. KA + stdlib only.
include(joinpath(_SRC, "methods", "inc_ns", "cavity_mg.jl"))   # solve_incns_cavity_mg

# ---------------------------------------------------------------------------
# Conditionally load CUDA at TOP LEVEL. This MUST be done in top-level
# statements SEPARATE from where the bindings are first used: referencing `CUDA`
# in the same world age as its `using` raises `UndefVarError: CUDA ... binding
# too new`. On a CPU-only box the `using` throws and the GPU half stays disabled.
# ---------------------------------------------------------------------------
const _CUDA_LOADED = try
    @eval using CUDA
    true
catch err
    @info "GPU configs disabled: CUDA not loadable" error = err
    false
end

# Separate top-level statement -> world age has advanced past the `using` above,
# so `CUDA` now resolves. Short-circuits (never touches `CUDA`) when not loaded.
const _GPU_OK = _CUDA_LOADED && CUDA.functional()

# The CUDA-graph wrapper REQUIRES `using CUDA` in scope, so it is included here
# (top level, after the CUDA block) and only under a functional CUDA. Its entry
# point is reached via Base.invokelatest (freshly-included method).
const _GRAPH_LOADED = if _GPU_OK
    try
        include(joinpath(_SRC, "methods", "inc_ns", "cavity_mg_cuda.jl"))
        true
    catch err
        @warn "CUDA-graph wrapper not loadable; C3 will be skipped, C4 runs plain" error = err
        false
    end
else
    false
end

# ---------------------------------------------------------------------------
# Ghia, Ghia & Shin (1982): u-velocity on the VERTICAL centreline (x=0.5) at
# y/L positions, normalised by U_lid. Reused from benchmarks/convergence_cavity.jl
# (Re=100 for the smoke case, Re=1000 for the heavy set).
# ---------------------------------------------------------------------------
const GHIA_Y = [0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719,
                0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516,
                0.9531, 0.9609, 0.9688, 0.9766, 1.0000]
const GHIA_RE100_UX = [0.0000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
                       -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
                       0.68717, 0.73722, 0.78871, 0.84123, 1.00000]
const GHIA_RE1000_UX = [0.0000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289,
                        -0.27805, -0.10648, -0.06080, 0.05702, 0.18719, 0.33304,
                        0.46604, 0.51117, 0.57492, 0.65928, 1.00000]
const GHIA_DATA = Dict(100 => GHIA_RE100_UX, 1000 => GHIA_RE1000_UX)

const SMOKE = get(ENV, "CAVITY_BENCH_SMOKE", "") in ("1", "true", "TRUE")
const WANT_1024 = get(ENV, "CAVITY_BENCH_1024", "") in ("1", "true", "TRUE")
const WANT_1024_C0 = get(ENV, "CAVITY_BENCH_1024_C0", "") in ("1", "true", "TRUE")

# Max-relative parity over the solution fields u, v, p vs the CPU reference.
const PARITY_TOL = 1e-3

# ---------------------------------------------------------------------------
# Cases. SMOKE = 64^2 Re=100 CPU-only (structure validation, < 2 min). Heavy =
# the A100 ablation case 512^2 Re=1000 (+ optional 1024^2, GPU-only).
# ---------------------------------------------------------------------------
function bench_case()
    SMOKE && return (N=64, Re=100.0, relax=(u=0.7, p=0.3),
                     tol=1e-6, vel_tol=1e-5, maxiter=1200)
    return (N=512, Re=1000.0, relax=(u=0.5, p=0.2),
            tol=1e-7, vel_tol=1e-6, maxiter=60000)
end

case_1024() = (N=1024, Re=1000.0, relax=(u=0.5, p=0.2),
               tol=1e-7, vel_tol=1e-6, maxiter=80000)

# ---------------------------------------------------------------------------
# The CUMULATIVE ablation ladder. `solver` are the kwargs layered on top of the
# physical case; `graph` routes through solve_incns_cavity_mg_cuda_graph;
# `mp` adds mg_mixed_precision + mom_mg_mixed_precision.
# ---------------------------------------------------------------------------
ablation_configs() = [
    (id="C0", delta="legacy (norm_stride=1, mg_cycles=0)",
     solver=(norm_stride=1, mg_cycles=0), graph=false, mp=false),
    (id="C1", delta="+norm_stride=25",
     solver=(norm_stride=25, mg_cycles=0), graph=false, mp=false),
    (id="C2", delta="+fixed MG cycles (3/1)",
     solver=(norm_stride=25, mg_cycles=3, mom_mg_cycles=1), graph=false, mp=false),
    (id="C3", delta="+CUDA graph",
     solver=(norm_stride=25, mg_cycles=3, mom_mg_cycles=1), graph=true, mp=false),
    (id="C4", delta="+mixed precision (p+mom)",
     solver=(norm_stride=25, mg_cycles=3, mom_mg_cycles=1), graph=true, mp=true),
]

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
# Centreline u(y) at x=L/2 interpolated onto the Ghia y-positions (u/U_lid).
function centreline_u(res)
    xc = res.xcenters; yc = res.ycenters
    mid_i = argmin(abs.(xc .- 0.5 * res.L))
    ucol = res.u[mid_i, :] ./ res.U_lid
    interp(yq) = begin
        if yq <= yc[1]
            return ucol[1]
        elseif yq >= yc[end]
            return ucol[end]
        end
        k = searchsortedlast(yc, yq)
        t = (yq - yc[k]) / (yc[k+1] - yc[k])
        (1 - t) * ucol[k] + t * ucol[k+1]
    end
    return [interp(yg * res.L) for yg in GHIA_Y]
end

# Max |centreline u - Ghia| in % of U_lid; NaN when no Ghia table for this Re.
function ghia_error_pct(res)
    ref = get(GHIA_DATA, Int(round(res.Re)), nothing)
    ref === nothing && return NaN
    return 100.0 * maximum(abs.(centreline_u(res) .- ref))
end

# Max-RELATIVE parity over the full solution fields u, v, p (host arrays):
# max_f ||f - f_ref||_inf / ||f_ref||_inf. Target < PARITY_TOL.
function field_parity(res, ref)
    rel(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), eps())
    return max(rel(res.u, ref.u), rel(res.v, ref.v), rel(res.p, ref.p))
end

# ---------------------------------------------------------------------------
# GPU utilization sampler: background `nvidia-smi -l 1` started right before
# and killed right after the timed solve, so mean/peak cover the solve window
# only. Gracefully disabled when nvidia-smi (or CUDA) is absent.
# ---------------------------------------------------------------------------
function start_gpu_sampler(logpath)
    (_GPU_OK && Sys.which("nvidia-smi") !== nothing) || return nothing
    cmd = `nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader -l 1`
    proc = try
        run(pipeline(cmd; stdout=logpath, stderr=devnull); wait=false)
    catch err
        @warn "GPU utilization sampler failed to start" error = err
        nothing
    end
    return proc
end

function stop_gpu_sampler(proc, logpath)
    proc === nothing && return (mean=NaN, peak=NaN, n=0)
    try
        kill(proc)
        wait(proc)
    catch
    end
    vals = Int[]
    try
        for line in eachline(logpath)
            m = match(r"(\d+)", line)
            m === nothing || push!(vals, parse(Int, m.captures[1]))
        end
    catch
    end
    isempty(vals) && return (mean=NaN, peak=NaN, n=0)
    return (mean=sum(vals) / length(vals), peak=maximum(vals), n=length(vals))
end

# ---------------------------------------------------------------------------
# Solve runners. CUDA bindings (CUDABackend / CuArray) and the freshly-included
# graph entry point are reached via Base.invokelatest (world-age pattern; same
# convention as the previous bench).
# ---------------------------------------------------------------------------
_case_kwargs(c) = (nx=c.N, ny=c.N, Re=c.Re, U_lid=1.0, relax=c.relax,
                   tol=c.tol, vel_tol=c.vel_tol, maxiter=c.maxiter)

_mp_kwargs(mp) = mp ? (mg_mixed_precision=true, mom_mg_mixed_precision=true) : (;)

function solve_cpu(c, extra)
    return solve_incns_cavity_mg(; _case_kwargs(c)..., extra...,
                                 backend_ka=KernelAbstractions.CPU(),
                                 atype=Array{Float64})
end

function solve_gpu(c, extra; graph::Bool=false)
    kab   = Base.invokelatest(CUDABackend)
    atype = Base.invokelatest(() -> CuArray{Float64})
    res = if graph
        f = getproperty(Main, :solve_incns_cavity_mg_cuda_graph)
        Base.invokelatest(f; _case_kwargs(c)..., extra...,
                          backend_ka=kab, atype=atype)
    else
        Base.invokelatest(solve_incns_cavity_mg; _case_kwargs(c)..., extra...,
                          backend_ka=kab, atype=atype)
    end
    Base.invokelatest(CUDA.synchronize)            # device sync before stopping clock
    return res
end

# Per-config JIT warmup OUTSIDE the timed/sampled window: tiny 64^2 run through
# the SAME code path (plain/graph, F64/mixed) so first-touch compilation does
# not pollute the config's wall time or its GPU-utilization stats.
function warmup_config(extra; graph::Bool, gpu::Bool)
    w = (N=64, Re=100.0, relax=(u=0.7, p=0.3), tol=1e-6, vel_tol=1e-5, maxiter=60)
    if gpu
        solve_gpu(w, extra; graph=graph)
    else
        solve_cpu(w, extra)
    end
    return nothing
end

# Which execution path actually ran (surface the graph wrapper's fallback).
function path_string(res; graph::Bool, gpu::Bool)
    gpu || return "cpu"
    graph || return "gpu"
    if get(res, :graph_captured, false)
        return "gpu+graph(launches=$(res.graph_launches))"
    elseif get(res, :graph_fallback, false)
        return "gpu(graph-FALLBACK:plain)"
    else
        return "gpu(graph-not-engaged)"
    end
end

# ---------------------------------------------------------------------------
# Run one ablation config: warmup -> start sampler -> timed solve -> stop
# sampler -> metrics. `ref` is the CPU reference result at the same N (or
# `nothing` -> parity/speed-up reported as "-", e.g. the 1024^2 rows).
# Any escaping exception (e.g. graph capture machinery) yields a FAILED row and
# the bench continues.
# ---------------------------------------------------------------------------
function run_config(cfg, c; ref, gpu::Bool)
    extra = (; cfg.solver..., _mp_kwargs(cfg.mp)...)
    use_graph = cfg.graph && gpu && _GRAPH_LOADED
    util_log = joinpath(pwd(), "gpu_util_$(cfg.id)_N$(c.N).log")
    @printf("[%s] N=%d Re=%d %s (%s) ...\n", cfg.id, c.N, Int(c.Re), cfg.delta,
            gpu ? (use_graph ? "GPU+graph" : "GPU") : "CPU")
    local res, t
    util = (mean=NaN, peak=NaN, n=0)
    try
        warmup_config(extra; graph=use_graph, gpu=gpu)
        sampler = gpu ? start_gpu_sampler(util_log) : nothing
        try
            t0 = time_ns()
            res = gpu ? solve_gpu(c, extra; graph=use_graph) : solve_cpu(c, extra)
            t = (time_ns() - t0) / 1e9
        finally
            util = stop_gpu_sampler(sampler, util_log)   # solve window only
        end
        parity = ref === nothing ? NaN : field_parity(res, ref.res)
        speedup = ref === nothing ? NaN : ref.t / t
        row = (; id=cfg.id, delta=cfg.delta, N=c.N, t=t, iters=res.iters,
               converged=res.converged, speedup=speedup, parity=parity,
               ghia=ghia_error_pct(res),
               util_mean=util.mean, util_peak=util.peak,
               path=path_string(res; graph=use_graph, gpu=gpu), res=res,
               status=:ok)
        @printf("     done: %.2fs, iters=%d, converged=%s, parity=%s, Ghia=%.3f%%, path=%s\n",
                row.t, row.iters, row.converged,
                isnan(parity) ? "-" : @sprintf("%.2e", parity), row.ghia, row.path)
        return row
    catch err
        msg = sprint(showerror, err)
        msg = length(msg) > 80 ? msg[1:80] * "..." : msg
        @warn "config $(cfg.id) FAILED; continuing with remaining configs" error = err
        return (; id=cfg.id, delta=cfg.delta, N=c.N, t=NaN, iters=0,
                converged=false, speedup=NaN, parity=NaN, ghia=NaN,
                util_mean=NaN, util_peak=NaN,
                path="FAILED($msg)", res=nothing, status=:fail)
    end
end

skip_row(cfg, c, why) = (; id=cfg.id, delta=cfg.delta, N=c.N, t=NaN, iters=0,
                         converged=false, speedup=NaN, parity=NaN, ghia=NaN,
                         util_mean=NaN, util_peak=NaN, path="SKIP($why)",
                         res=nothing, status=:skip)

# ---------------------------------------------------------------------------
# Markdown summary table (machine-parseable, printed to stdout).
# ---------------------------------------------------------------------------
_fmt(x; digits=2) = isnan(x) ? "-" : string(round(x; digits=digits))
_fmte(x) = isnan(x) ? "-" : @sprintf("%.2e", x)

function print_table(rows)
    println()
    println("## ABLATION SUMMARY (markdown)")
    println()
    println("| config | delta | N | wall_s | iters | converged | speedup_vs_cpu | parity_maxrel | ghia_err_pct | gpu_util_mean_pct | gpu_util_peak_pct | path |")
    println("|--------|-------|---|--------|-------|-----------|----------------|---------------|--------------|-------------------|-------------------|------|")
    for r in rows
        @printf("| %s | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n",
                r.id, r.delta, r.N,
                _fmt(r.t), r.status == :ok ? string(r.iters) : "-",
                r.status == :ok ? string(r.converged) : "-",
                isnan(r.speedup) ? "-" : @sprintf("%.2fx", r.speedup),
                _fmte(r.parity), _fmt(r.ghia; digits=3),
                _fmt(r.util_mean; digits=1), _fmt(r.util_peak; digits=0),
                r.path)
    end
    println()
end

# ---------------------------------------------------------------------------
function main()
    println("=== SIMPLE cavity GPU ABLATION bench (per-change contribution) ===")
    c = bench_case()
    if SMOKE
        println("    MODE: SMOKE — 64^2 Re=100, CPU-only, C0->C2 + C4 (graph + util sampling skipped)")
    else
        println("    MODE: HEAVY — 512^2 Re=1000 ablation C0->C4 on GPU + one CPU reference")
        WANT_1024 && println("    1024^2 GPU-only runs ENABLED (best config" *
                             (WANT_1024_C0 ? " + C0 contrast)" : ")"))
    end
    @printf("    case: N=%d Re=%d relax=(u=%.2f,p=%.2f) tol=%.0e vel_tol=%.0e maxiter=%d\n",
            c.N, Int(c.Re), c.relax.u, c.relax.p, c.tol, c.vel_tol, c.maxiter)
    println("    parity = max-rel over u/v/p fields vs CPU reference, target < $(PARITY_TOL)")
    println("    GPU available: $(_GPU_OK), graph wrapper loaded: $(_GRAPH_LOADED)")
    println()

    # --- CPU reference (legacy kwargs): speed-up denominator + parity fields ---
    legacy = (norm_stride=1, mg_cycles=0)
    println("[CPUref] legacy CPU reference, N=$(c.N) Re=$(Int(c.Re)) ...")
    warmup_config(legacy; graph=false, gpu=false)
    t0 = time_ns()
    refres = solve_cpu(c, legacy)
    tref = (time_ns() - t0) / 1e9
    ref = (; res=refres, t=tref)
    @printf("     done: %.2fs, iters=%d, converged=%s, Ghia=%.3f%%\n",
            tref, refres.iters, refres.converged, ghia_error_pct(refres))
    rows = Any[(; id="CPUref", delta="legacy CPU reference", N=c.N, t=tref,
                iters=refres.iters, converged=refres.converged, speedup=1.0,
                parity=0.0, ghia=ghia_error_pct(refres),
                util_mean=NaN, util_peak=NaN, path="cpu", res=refres, status=:ok)]

    # --- ablation ladder ---
    # SMOKE: CPU-only ladder (C0->C2, C4-plain; C3 skipped — its only delta is
    # the graph). HEAVY: GPU ladder; run_config surfaces the wrapper's fallback
    # path, and a heavy run WITHOUT CUDA skips the whole ladder (a 512^2 CPU
    # ablation would take ~1 h per config — run it on Aqua instead).
    gpu_mode = _GPU_OK && !SMOKE
    graph_possible = gpu_mode && _GRAPH_LOADED
    for cfg in ablation_configs()
        if !SMOKE && !gpu_mode
            push!(rows, skip_row(cfg, c, "no CUDA — heavy ablation is GPU-only"))
            continue
        end
        if cfg.graph && !cfg.mp && !graph_possible
            push!(rows, skip_row(cfg, c, gpu_mode ? "graph wrapper not loadable" :
                                                    "no CUDA"))
            continue
        end
        push!(rows, run_config(cfg, c; ref=ref, gpu=gpu_mode))
    end

    # --- optional 1024^2, GPU-only (no CPU reference: too slow) ---
    if WANT_1024 && !SMOKE
        if gpu_mode
            c1k = case_1024()
            cfgs = ablation_configs()
            # best = last config of [C4, C3, C2] whose 512^2 run is :ok
            okids = Set(r.id for r in rows if r.status == :ok)
            best_id = "C4" in okids ? "C4" : ("C3" in okids ? "C3" : "C2")
            best = cfgs[findfirst(x -> x.id == best_id, cfgs)]
            println("\n-- 1024^2 GPU-only (best config = $(best.id); no CPU ref) --")
            push!(rows, run_config(best, c1k; ref=nothing, gpu=true))
            if WANT_1024_C0
                push!(rows, run_config(cfgs[1], c1k; ref=nothing, gpu=true))
            end
        else
            println("\n-- 1024^2 requested but no functional CUDA: skipped --")
        end
    end

    print_table(rows)

    # --- status ---
    ran = [r for r in rows if r.status == :ok && r.id != "CPUref"]
    parity_rows = [r for r in ran if !isnan(r.parity)]
    parity_ok = all(r.parity < PARITY_TOL for r in parity_rows)
    conv_ok = all(r.converged for r in ran)
    fails = [r.id for r in rows if r.status == :fail]
    ghia_ref = rows[1].ghia

    if SMOKE
        println("SMOKE: structure OK — table printed, parity/Ghia columns populated.")
        println(parity_ok ? "SMOKE parity: GREEN (all max-rel < $(PARITY_TOL))." :
                            "SMOKE parity: RED (a config exceeded $(PARITY_TOL)).")
        return parity_ok ? 0 : 1
    end

    if parity_ok && conv_ok && isempty(fails) && ghia_ref < 8.0
        println("STATUS: GREEN — all configs converged, parity < $(PARITY_TOL), " *
                "CPU Ghia-err $(@sprintf("%.2f", ghia_ref))% (< 8%).")
    else
        reasons = String[]
        parity_ok || push!(reasons, "parity FAIL")
        conv_ok || push!(reasons, "non-converged config")
        isempty(fails) || push!(reasons, "FAILED: $(join(fails, ","))")
        ghia_ref < 8.0 || push!(reasons, "Ghia-err $(@sprintf("%.2f", ghia_ref))% >= 8%")
        println("STATUS: RED — $(join(reasons, "; ")). Inspect the table above.")
    end
    return 0
end

exit(main())
