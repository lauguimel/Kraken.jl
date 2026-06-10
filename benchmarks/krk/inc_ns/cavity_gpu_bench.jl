# Aqua-ready GPU benchmark: the FULL backend-parametric SIMPLE lid-driven cavity
# solver (src/methods/inc_ns/cavity_mg.jl) run on CUDA (GPU) vs CPU at Re=1000 on
# fine grids, with GPU<->CPU parity, Ghia (1982) validation, and timing.
#
# WHAT IT DOES
# ------------
# For each grid N in BENCH_NS (default {256, 512}, Re=1000) it:
#   * solves the cavity on the CPU backend (backend_ka=CPU(), Array{Float64})   ALWAYS
#   * solves the cavity on the GPU backend (backend_ka=CUDABackend(),
#     atype=CuArray{Float64}) — only when _GPU_OK                                GPU only
#   * asserts GPU<->CPU centreline parity ‖Δu‖∞ < PARITY_TOL (SIMPLE is iterative
#     so this is a loose-but-meaningful tolerance, not bit parity)
#   * reports BOTH against Ghia Re=1000 (max centreline error, in % of U_lid)
#   * times CPU + GPU wall clock, computes GPU/CPU speed-up.
# A table summarises: N, wall-time s (CPU/GPU), iters, GPU/CPU speed-up,
# max-Ghia-error %, parity. A clear GREEN/RED summary line closes the run.
#
# The SAME source file runs on both backends — a CUDA run is purely
# backend_ka=CUDABackend(), atype=CuArray{Float64} (see cavity_mg.jl header).
#
# RUNNING
# -------
#   CPU only (local, this machine):  julia --project=. benchmarks/krk/inc_ns/cavity_gpu_bench.jl
#   GPU (Aqua, CUDA in proj):        same command — the GPU columns auto-activate.
# The CPU half is the source of truth and runs standalone; the GPU columns
# activate only when CUDA is functional, so this file is safe on a CPU-only box.
#
#   CAVITY_BENCH_SMOKE=1 ... -> run ONLY a small quick case (Re=100, 64^2, a few
#       hundred SIMPLE iters) so the STRUCTURE can be validated fast locally.
#   default (no env)        -> the heavy Re=1000 fine grids {256,512} (for Aqua).
#   CAVITY_BENCH_1024=1     -> additionally append N=1024 to the heavy set.

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
    @eval using CUDA.CUSPARSE
    @eval using CUDSS
    true
catch err
    @info "GPU half disabled: CUDA/CUDSS not loadable" error = err
    false
end

# Separate top-level statement → world age has advanced past the `using` above,
# so `CUDA` now resolves. Short-circuits (never touches `CUDA`) when not loaded.
const _GPU_OK = _CUDA_LOADED && CUDA.functional()

# ---------------------------------------------------------------------------
# Ghia, Ghia & Shin (1982) reference, Re=1000: u-velocity on the VERTICAL
# centreline (x=0.5) sampled at y/L positions (u normalised by U_lid).
# ---------------------------------------------------------------------------
const GHIA_Y = [0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719,
                0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516,
                0.9531, 0.9609, 0.9688, 0.9766, 1.0000]
const GHIA_RE1000_UX = [0.0000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289,
                        -0.27805, -0.10648, -0.06080, 0.05702, 0.18719, 0.33304,
                        0.46604, 0.51117, 0.57492, 0.65928, 1.00000]

# ---------------------------------------------------------------------------
# Bench cases. SMOKE = one small fast case so the STRUCTURE is validated locally
# without the (slow) Re=1000 fine grids. Default = the heavy Aqua set.
# ---------------------------------------------------------------------------
const SMOKE = get(ENV, "CAVITY_BENCH_SMOKE", "") in ("1", "true", "TRUE")

# SIMPLE solver knobs. Re=1000 on fine grids needs many SIMPLE sweeps; the
# loose vel-settle gate matches cavity_mg.jl's explicit under-relaxation.
function bench_cases()
    if SMOKE
        # Small, quick: Re=100 on 64^2, capped iterations so it finishes fast.
        return [(N=64, Re=100.0, relax=(u=0.7, p=0.3), tol=1e-6, vel_tol=1e-5,
                 maxiter=400)]
    end
    cases = [
        (N=256, Re=1000.0, relax=(u=0.5, p=0.2), tol=1e-7, vel_tol=1e-6, maxiter=40000),
        (N=512, Re=1000.0, relax=(u=0.5, p=0.2), tol=1e-7, vel_tol=1e-6, maxiter=60000),
    ]
    if get(ENV, "CAVITY_BENCH_1024", "") in ("1", "true", "TRUE")
        push!(cases, (N=1024, Re=1000.0, relax=(u=0.5, p=0.2),
                      tol=1e-7, vel_tol=1e-6, maxiter=80000))
    end
    return cases
end

# GPU<->CPU centreline parity tolerance. SIMPLE is iterative and the two backends
# may take slightly different reduction/round-off paths, so we require agreement
# of the converged centreline to 1e-3 (absolute, u normalised by U_lid), NOT bit
# parity. This is a physically meaningful "same solution" check.
const PARITY_TOL = 1e-3

# ---------------------------------------------------------------------------
# Centreline extraction. The solver returns u,v as nx×ny host Arrays with
# u[i,j], i=x-index, j=y-index, cell centres at xcenters/ycenters. The Ghia
# profile is the u-velocity on the VERTICAL centreline x=0.5, as a function of y.
# We pick the column of cells nearest x=0.5 and linearly interpolate u(y) onto
# the GHIA_Y positions (normalising by U_lid).
# ---------------------------------------------------------------------------
function centreline_u(res)
    xc = res.xcenters; yc = res.ycenters
    mid_i = argmin(abs.(xc .- 0.5 * res.L))      # column nearest x = L/2
    ucol = res.u[mid_i, :] ./ res.U_lid          # u(y) normalised, length N
    # linear interpolation u(y) onto a query y in [0,L]
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

# Max |centreline u(this Ghia y) - Ghia ref|, in % of U_lid (profile is already
# normalised, so this is directly a percentage).
function ghia_error_pct(res)
    up = centreline_u(res)
    return 100.0 * maximum(abs.(up .- GHIA_RE1000_UX))
end

# Parity: ‖Δu‖∞ over the interpolated centreline, absolute (normalised by U_lid).
function centreline_parity(res_a, res_b)
    return maximum(abs.(centreline_u(res_a) .- centreline_u(res_b)))
end

# ----- CPU solve: always runs, standalone ----------------------------------
function run_cavity_cpu(c)
    t0 = time_ns()
    res = solve_incns_cavity_mg(; nx=c.N, ny=c.N, Re=c.Re, U_lid=1.0,
                                relax=c.relax, tol=c.tol, vel_tol=c.vel_tol,
                                maxiter=c.maxiter,
                                backend_ka=KernelAbstractions.CPU(),
                                atype=Array{Float64})
    t = (time_ns() - t0) / 1e9
    return (; res, t, c.N, c.Re,
            iters=res.iters, converged=res.converged,
            ghia_pct=ghia_error_pct(res))
end

# ----- GPU solve: activates only under a functional CUDA --------------------
# Same source, same call, but backend_ka=CUDABackend() and atype=CuArray{Float64}.
# Uses Base.invokelatest for the freshly-loaded-CUDA references (CUDABackend /
# CuArray) to avoid world-age errors. Returns host solution + timing.
function run_cavity_gpu(c)
    kab   = Base.invokelatest(CUDABackend)
    atype = Base.invokelatest(() -> CuArray{Float64})
    solve_gpu() = Base.invokelatest(solve_incns_cavity_mg;
                                    nx=c.N, ny=c.N, Re=c.Re, U_lid=1.0,
                                    relax=c.relax, tol=c.tol, vel_tol=c.vel_tol,
                                    maxiter=c.maxiter, backend_ka=kab, atype=atype)
    t0 = time_ns()
    res = solve_gpu()
    Base.invokelatest(CUDA.synchronize)            # device sync before stopping clock
    t = (time_ns() - t0) / 1e9
    return (; res, t, c.N, c.Re,
            iters=res.iters, converged=res.converged,
            ghia_pct=ghia_error_pct(res))
end

function main()
    println("=== full SIMPLE cavity bench: GPU (CUDA) vs CPU, Re=1000 (Ghia 1982) ===")
    cs = bench_cases()
    if SMOKE
        println("    MODE: SMOKE (small fast case; structure validation only)")
    else
        println("    MODE: HEAVY (Re=1000 fine grids for Aqua)")
    end
    @printf("    cases: %s\n", join(["N=$(c.N),Re=$(Int(c.Re))" for c in cs], "  "))
    println("    parity tol ‖Δu_centreline‖∞ < $(PARITY_TOL) (SIMPLE is iterative)")
    println()

    # --- CPU column (always; source of truth) ---
    cpu = NamedTuple[]
    for c in cs
        @printf("[CPU] solving N=%d Re=%d ...\n", c.N, Int(c.Re))
        r = run_cavity_cpu(c)
        @printf("      done: %.2fs, iters=%d, converged=%s, Ghia-err=%.3f%%\n",
                r.t, r.iters, r.converged, r.ghia_pct)
        push!(cpu, r)
    end

    # --- GPU column (only under functional CUDA) ---
    gpu = _GPU_OK ? NamedTuple[] : nothing
    all_parity_ok = true
    if _GPU_OK
        for (idx, c) in enumerate(cs)
            @printf("[GPU] solving N=%d Re=%d ...\n", c.N, Int(c.Re))
            g = run_cavity_gpu(c)
            par = centreline_parity(g.res, cpu[idx].res)
            ok = par < PARITY_TOL
            all_parity_ok &= ok
            @printf("      done: %.2fs, iters=%d, converged=%s, Ghia-err=%.3f%%, parity=%.2e (%s)\n",
                    g.t, g.iters, g.converged, g.ghia_pct, par, ok ? "OK" : "FAIL")
            push!(gpu, (; g..., parity=par, parity_ok=ok))
        end
    end

    # --- Results table ---
    println()
    println("-- results (wall time in s of the full SIMPLE solve) --")
    @printf("%-6s %-7s %-11s %-11s %-9s %-9s %-9s %-10s %-11s\n",
            "N", "Re", "CPU_s", "GPU_s", "speedup", "iters", "Ghia%CPU", "Ghia%GPU", "parity_inf")
    for (idx, r) in enumerate(cpu)
        if _GPU_OK
            g = gpu[idx]
            speedup = @sprintf("%.2fx", r.t / g.t)
            @printf("%-6d %-7d %-11.2f %-11.2f %-9s %-9d %-9.3f %-10.3f %-11.2e\n",
                    r.N, Int(r.Re), r.t, g.t, speedup, r.iters,
                    r.ghia_pct, g.ghia_pct, g.parity)
        else
            @printf("%-6d %-7d %-11.2f %-11s %-9s %-9d %-9.3f %-10s %-11s\n",
                    r.N, Int(r.Re), r.t, "skip", "-", r.iters,
                    r.ghia_pct, "n/a", "n/a")
        end
    end

    # --- Status line ---
    println()
    # Ghia accuracy gate: on a properly resolved Re=1000 cavity the centreline
    # should track Ghia to a few % of U_lid. We flag if CPU is way off (the CPU
    # path is the source of truth). SMOKE is Re=100 so the Ghia-Re1000 column is
    # not physically meaningful there; we only check structure/exit there.
    cpu_max_ghia = maximum(r.ghia_pct for r in cpu)
    if SMOKE
        println("SMOKE: structure OK — solver ran on CPU, table printed.")
        println("  (Ghia%% column is Re=1000 ref vs a Re=100 smoke case — ignore its value.)")
        if _GPU_OK
            println(all_parity_ok ? "GPU half: GREEN (parity < $(PARITY_TOL))." :
                                    "GPU half: RED (centreline parity exceeded $(PARITY_TOL)).")
        else
            println("GPU half: SKIPPED (no functional CUDA).")
        end
        return 0
    end

    if _GPU_OK
        if all_parity_ok && cpu_max_ghia < 8.0
            println("STATUS: GREEN — GPU<->CPU centreline parity < $(PARITY_TOL) at all N, " *
                    "max CPU Ghia-err = $(@sprintf("%.2f", cpu_max_ghia))% (< 8%).")
        else
            reason = !all_parity_ok ? "parity FAIL" : "Ghia-err $(@sprintf("%.2f", cpu_max_ghia))% >= 8%"
            println("STATUS: RED — $(reason). Inspect the table above.")
        end
    else
        if cpu_max_ghia < 8.0
            println("STATUS: GREEN (CPU-only) — max CPU Ghia-err = $(@sprintf("%.2f", cpu_max_ghia))% (< 8%). " *
                    "GPU columns SKIPPED (no functional CUDA); run on Aqua for GPU/CPU timing + parity.")
        else
            println("STATUS: RED (CPU-only) — max CPU Ghia-err = $(@sprintf("%.2f", cpu_max_ghia))% >= 8%. " *
                    "GPU columns SKIPPED (no functional CUDA).")
        end
    end

    return 0
end

exit(main())
