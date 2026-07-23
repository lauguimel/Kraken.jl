# Aqua-ready GPU benchmark: matrix-free geometric multigrid (V-cycle) Poisson
# vs cuDSS sparse-direct, CPU vs GPU, up to N=4096 (16M DOF).
#
# WHAT IT DOES
# ------------
# Manufactured Dirichlet Poisson on the unit square,
#     u(x,y) = sin(pi x) sin(pi y),   f = -∇²u = 2 pi² sin(pi x) sin(pi y),
# with homogeneous Dirichlet data (u = 0 on the boundary, so nothing is folded
# into the RHS). For N in {128, 256, 512, 1024, 2048, 4096} it solves with:
#   * MG on CPU   (backend_ka = KernelAbstractions.CPU(), Array{Float64})   ALWAYS
#   * MG on GPU   (backend_ka = CUDABackend(), CuArray{Float64})            only if _GPU_OK
#   * cuDSS F64   (assemble the regular Poisson, factor + solve)            only if _GPU_OK
# and reports a table: N, DOF, MG-CPU ms, MG-GPU ms, MG speed-up, cuDSS solve ms,
# V-cycles, cells/s, parity.
#
# WHY THIS EXISTS
# ---------------
# cuDSS sparse-direct under-uses the A100 (~9% averaged GPU utilization on the
# pressure-Poisson solve). The hypothesis: a matrix-free geometric multigrid whose
# smoothers are 5-point stencils SATURATES the GPU and is O(N) per V-cycle with a
# V-cycle count that is asymptotically INDEPENDENT of N. This bench measures it,
# and the companion .pbs samples nvidia-smi utilization during the run to confirm.
#
# RUNNING
# -------
#   CPU only (local, this machine):  julia --project=. benchmarks/krk/inc_ns/poisson_mg_gpu_bench.jl
#   GPU (Aqua, CUDA+CUDSS in proj):  same command — the GPU columns auto-activate.
# The CPU half (MG-CPU + analytic order) is the source of truth and runs
# standalone; the GPU/cuDSS columns activate only when CUDA is functional, so this
# file is safe to run on a CPU-only box.

using LinearAlgebra
using SparseArrays
using Printf

const _SRC = joinpath(@__DIR__, "..", "..", "..", "src", "solve")

# CPU pieces: regular Poisson assembly + analytic L2 error (poisson.jl) and the
# matrix-free multigrid solver (poisson_mg.jl). poisson_mg.jl pulls in
# linear_solve.jl (the backend tags + CPU seam) itself if not present.
include(joinpath(_SRC, "poisson.jl"))      # assemble_poisson_dirichlet, l2_error, tags
include(joinpath(_SRC, "poisson_mg.jl"))   # solve_poisson_mg + MG hierarchy

# ---------------------------------------------------------------------------
# Conditionally load CUDA/CUDSS at TOP LEVEL. This MUST be done in top-level
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

const BENCH_NS = (128, 256, 512, 1024, 2048, 4096)

# Manufactured Dirichlet problem (homogeneous boundary -> no RHS folding needed).
mg_exact(x, y) = sin(pi * x) * sin(pi * y)
mg_rhs(x, y)   = 2.0 * pi^2 * sin(pi * x) * sin(pi * y)

# V-cycle stopping tolerance: a moderate relative residual that lets the V-cycle
# count plateau (the multigrid hallmark) and keeps the cuDSS comparison fair (the
# direct solve is exact, MG is iterated to a fixed relative residual).
const MG_TOL = 1e-8
const MG_MAXCYCLES = 60

# Median of repeated timings (in seconds) of `f()`, after one warmup.
function timed_median(f; reps::Int = 3)
    f()                                  # warmup (JIT + first-touch)
    ts = Float64[]
    for _ in 1:reps
        t0 = time_ns()
        f()
        push!(ts, (time_ns() - t0) / 1e9)
    end
    sort!(ts)
    return ts[(length(ts) + 1) ÷ 2]
end

# Grid-function L2 error of an MG solution vs the analytic field.
mg_l2_error(u::AbstractMatrix, N::Integer) = l2_error(Array(u), mg_exact, N)

# ----- MG on CPU: always runs, standalone ----------------------------------
function run_mg_cpu(N::Integer)
    # Solve once (untimed) to capture V-cycles + the solution for parity/accuracy.
    u, ncycles, _ = solve_poisson_mg(mg_rhs, N; bc = :dirichlet,
                                     tol = MG_TOL, maxcycles = MG_MAXCYCLES,
                                     smoother = :rbgs)
    t = timed_median(() -> solve_poisson_mg(mg_rhs, N; bc = :dirichlet,
                                            tol = MG_TOL, maxcycles = MG_MAXCYCLES,
                                            smoother = :rbgs))
    err = mg_l2_error(u, N)
    n = N * N
    return (; N, n, u = Array(u), ncycles, err,
            t_ms = 1e3 * t, cells_per_s = n / t)
end

# ----- MG on GPU: activates only under a functional CUDA --------------------
# Same source, same call, but backend_ka = CUDABackend() and atype = CuArray.
# Uses Base.invokelatest for the freshly-loaded-CUDA reference (CUDABackend /
# CuArray) to avoid world-age errors. Returns the host solution + timing.
function run_mg_gpu(N::Integer)
    kab   = Base.invokelatest(CUDABackend)
    atype = Base.invokelatest(() -> CuArray{Float64})
    solve_gpu() = Base.invokelatest(solve_poisson_mg, mg_rhs, N;
                                    bc = :dirichlet, backend_ka = kab,
                                    atype = atype, tol = MG_TOL,
                                    maxcycles = MG_MAXCYCLES, smoother = :rbgs)
    u, ncycles, _ = solve_gpu()
    # Time the full solve including device sync (synchronize is inside the kernels).
    t = timed_median(() -> (r = solve_gpu();
                            Base.invokelatest(CUDA.synchronize); r))
    n = N * N
    return (; N, n, u = Array(u), ncycles, t_ms = 1e3 * t, cells_per_s = n / t)
end

# ----- cuDSS sparse-direct: comparison column, GPU only ---------------------
# Assemble the REGULAR (non-embedded) Poisson, upload CPU CSC as device CSR (F64),
# factor once (cuDSS Cholesky) and time the SOLVE separately (the column that
# matters for the saturation comparison). Returns host solution + solve timing.
function run_cudss(N::Integer)
    A, b = assemble_poisson_dirichlet(N, mg_rhs)            # SparseMatrixCSC, F64

    A_gpu = Base.invokelatest(CUDA.CUSPARSE.CuSparseMatrixCSR, A)
    b_gpu = Base.invokelatest(CuVector{Float64}, b)

    # Factorize ONCE via the CUDA seam (loaded lazily below in the caller).
    cache = Base.invokelatest(lin_factorize, CUDABackendTag(), A_gpu; spd = true)

    t_solve = timed_median(() ->
        Base.invokelatest(() -> begin
            x = lin_solve!(cache, b_gpu); CUDA.synchronize(); x
        end))
    x_gpu = Base.invokelatest(lin_solve!, cache, b_gpu)

    n = N * N
    return (; N, n, x = Array(x_gpu),
            t_solve_ms = 1e3 * t_solve, cells_per_s = n / t_solve)
end

function main()
    println("=== matrix-free multigrid Poisson bench (MG-CPU / MG-GPU / cuDSS F64) ===")
    println("    MMS: u = sin(pi x) sin(pi y),  f = 2 pi^2 sin sin,  Dirichlet, tol=$(MG_TOL)")
    println()

    # --- CPU columns (always; source of truth) ---
    cpu = NamedTuple[]
    for N in BENCH_NS
        push!(cpu, run_mg_cpu(N))
    end

    # --- GPU columns (only under functional CUDA) ---
    gpu_mg = _GPU_OK ? NamedTuple[] : nothing
    gpu_ds = _GPU_OK ? NamedTuple[] : nothing
    if _GPU_OK
        # Load the CUDA seam methods (CuSparseMatrixCSR lin_factorize/lin_solve!).
        include(joinpath(_SRC, "linear_solve_cuda.jl"))
        for r in cpu
            mg = run_mg_gpu(r.N)
            # Parity MG-GPU vs MG-CPU (same iterative method, same tol).
            parity = maximum(abs.(mg.u .- r.u))
            @assert parity < 1e-6 "MG GPU/CPU parity failed at N=$(r.N): ‖Δ‖∞ = $parity"
            push!(gpu_mg, (; mg..., parity))

            ds = run_cudss(r.N)
            # cuDSS is a different (direct) solver; sanity-check it lands near MG.
            ds_vs_mg = maximum(abs.(reshape(ds.x, r.N, r.N) .- r.u))
            push!(gpu_ds, (; ds..., ds_vs_mg))
        end
    end

    # --- Combined results table ---
    println("-- results (ms = median wall time of full solve / cuDSS solve) --")
    @printf("%-6s %-12s %-12s %-12s %-9s %-13s %-9s %-13s %-11s\n",
            "N", "DOF", "MGcpu_ms", "MGgpu_ms", "speedup",
            "cuDSS_ms", "Vcyc", "cells/s(MG)", "parity_inf")
    for (idx, r) in enumerate(cpu)
        if _GPU_OK
            mg = gpu_mg[idx]
            ds = gpu_ds[idx]
            speedup = @sprintf("%.2fx", r.t_ms / mg.t_ms)
            cps = _GPU_OK ? mg.cells_per_s : r.cells_per_s
            @printf("%-6d %-12d %-12.3f %-12.3f %-9s %-13.3f %-9d %-13.3e %-11.2e\n",
                    r.N, r.n, r.t_ms, mg.t_ms, speedup, ds.t_solve_ms,
                    r.ncycles, cps, mg.parity)
        else
            @printf("%-6d %-12d %-12.3f %-12s %-9s %-13s %-9d %-13.3e %-11s\n",
                    r.N, r.n, r.t_ms, "skip", "-", "skip",
                    r.ncycles, r.cells_per_s, "n/a")
        end
    end

    # --- MG-CPU analytic accuracy (expect ~2nd order) ---
    println()
    println("-- MG-CPU analytic accuracy (MMS Dirichlet, expect order ~2) --")
    prev_err = NaN
    for r in cpu
        ord = isnan(prev_err) ? NaN : log2(prev_err / r.err)
        prev_err = r.err
        @printf("  N=%-6d L2=%.6e  order=%s  Vcycles=%d\n",
                r.N, r.err, isnan(ord) ? "-" : @sprintf("%.3f", ord), r.ncycles)
    end

    # --- V-cycle flatness check (the multigrid hallmark) ---
    counts = [r.ncycles for r in cpu]
    spread = maximum(counts) - minimum(counts)
    println()
    @printf("MG V-cycle counts across N: %s  (max-min = %d -> %s)\n",
            string(counts), spread, spread <= 4 ? "FLAT (O(1), multigrid hallmark)" : "GROWING?")

    # --- Status line ---
    println()
    if _GPU_OK
        println("GPU half: GREEN (MG GPU/CPU parity < 1e-6 at all N; cuDSS column present).")
        println("  Compare MG-GPU cells/s and the captured nvidia-smi utilization (gpu_util_mg.log)")
        println("  against cuDSS's ~9% to test the saturation hypothesis.")
    else
        println("GPU half: SKIPPED (no functional CUDA). MG-CPU + analytic benchmark complete.")
    end

    return 0
end

exit(main())
