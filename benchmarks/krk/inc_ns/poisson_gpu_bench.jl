# Aqua-ready GPU Poisson benchmark for the factorize-once linear-solve seam.
#
# WHAT IT DOES
# ------------
# Assembles the cut-cell (embedded) pressure-Poisson operator on the unit square
# at N in {128, 256, 512, 1024} and solves it with:
#   * CPU CHOLMOD  (always; standalone, no CUDA needed)         via lin_factorize/lin_solve!
#   * cuDSS F64    (only if `using CUDA, CUDSS` succeeds)        via the CUDA seam
# It times the symbolic+numeric FACTORIZE and the SOLVE separately, asserts GPU
# parity `‖x_cpu − x_gpu‖∞ < 1e-8`, and reports cells/s, factorize/solve ms, and
# scaling vs N.
#
# RUNNING
# -------
#   CPU only (local, this machine):   julia --project=. benchmarks/krk/inc_ns/poisson_gpu_bench.jl
#   GPU (Aqua, CUDA+CUDSS in project): same command — the GPU half auto-activates.
# The CPU half is the source of truth for the parity check; the GPU half only
# runs when CUDA is functional, so this file is safe to run on a CPU-only box.

using LinearAlgebra
using SparseArrays
using Printf

const _SRC = joinpath(@__DIR__, "..", "..", "..", "src", "solve")

# CPU seam (factorize-once) + the embedded cut-cell Poisson assembly.
include(joinpath(_SRC, "linear_solve.jl"))
include(joinpath(_SRC, "poisson_embedded.jl"))

const BENCH_NS = (128, 256, 512, 1024)

# Manufactured Dirichlet problem on the unit square with an embedded tilted wall,
# so the matrices are genuine cut-cell Poisson operators (the GPU target).
bench_exact(x, y) = x^2 + y^2
bench_rhs(x, y)   = -4.0

# Assemble the cut-cell Poisson operator A and RHS b at resolution N.
function assemble_bench_system(N::Integer)
    fx, fy, vf = tilted_half_plane_fractions(N)
    A, b = assemble_poisson_embedded(
        N, fx, fy, vf, bench_rhs;
        outer_bc = :dirichlet,
        embedded_bc = :dirichlet,
        outer_dirichlet = bench_exact,
        embedded_dirichlet = bench_exact,
    )
    return A, b, vf
end

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

# ----- CPU half: always runs, standalone -----------------------------------
function run_cpu(N::Integer)
    A, b, _ = assemble_bench_system(N)
    n = size(A, 1)

    # Factorize ONCE (CHOLMOD via the seam), timed separately from the solve.
    t_fact = timed_median(() -> lin_factorize(A; backend = CPUBackendTag(), spd = true))
    cache  = lin_factorize(A; backend = CPUBackendTag(), spd = true)

    # Reuse the cached factorization for the solve, timed separately.
    t_solve = timed_median(() -> lin_solve!(cache, b))
    x = lin_solve!(cache, b)

    return (; N, n, x, A, b,
            t_fact_ms = 1e3 * t_fact,
            t_solve_ms = 1e3 * t_solve,
            cells_per_s_solve = n / t_solve)
end

# ----- GPU half: activates only under a functional CUDA + CUDSS -------------
# Returns `nothing` if CUDA/CUDSS are unavailable (CPU-only run).
function try_run_gpu(cpu_results)
    have_cuda = false
    try
        @eval using CUDA
        @eval using CUDA.CUSPARSE
        @eval using CUDSS
        have_cuda = CUDA.functional()
    catch err
        @info "GPU half skipped: CUDA/CUDSS not available" error = err
        return nothing
    end
    if !have_cuda
        @info "GPU half skipped: CUDA.functional() == false (no device)"
        return nothing
    end

    include(joinpath(_SRC, "linear_solve_cuda.jl"))

    gpu_rows = NamedTuple[]
    for r in cpu_results
        # Upload the CPU CSC operator as a device CSR (F64) and solve with cuDSS.
        A_gpu = CUDA.CUSPARSE.CuSparseMatrixCSR(r.A)
        b_gpu = CuVector{Float64}(r.b)

        t_fact = timed_median(() ->
            Base.invokelatest(lin_factorize, CUDABackendTag(), A_gpu; spd = true))
        cache = Base.invokelatest(lin_factorize, CUDABackendTag(), A_gpu; spd = true)

        t_solve = timed_median(() -> Base.invokelatest(lin_solve!, cache, b_gpu))
        x_gpu = Base.invokelatest(lin_solve!, cache, b_gpu)

        # Parity against the CPU CHOLMOD solution.
        x_gpu_host = Array(x_gpu)
        parity = maximum(abs.(x_gpu_host .- r.x))
        @assert parity < 1e-8 "GPU/CPU parity failed at N=$(r.N): ‖Δ‖∞ = $parity"

        push!(gpu_rows, (; N = r.N, n = r.n,
                         t_fact_ms = 1e3 * t_fact,
                         t_solve_ms = 1e3 * t_solve,
                         cells_per_s_solve = r.n / t_solve,
                         parity))
    end
    return gpu_rows
end

function main()
    println("=== cut-cell Poisson factorize-once benchmark (CPU CHOLMOD / cuDSS F64) ===")
    @printf("%-6s %-10s %-14s %-14s %-16s %-12s\n",
            "N", "cells", "factor_ms", "solve_ms", "cells/s(solve)", "scale_solve")

    cpu_results = NamedTuple[]
    prev_solve = NaN
    for N in BENCH_NS
        r = run_cpu(N)
        scale = isnan(prev_solve) ? "-" : @sprintf("%.2fx", r.t_solve_ms / prev_solve)
        prev_solve = r.t_solve_ms
        @printf("%-6d %-10d %-14.3f %-14.3f %-16.3e %-12s\n",
                r.N, r.n, r.t_fact_ms, r.t_solve_ms, r.cells_per_s_solve, scale)
        push!(cpu_results, r)
    end

    # Convergence sanity: the cut-cell Dirichlet solve should be ~2nd order, a
    # cheap guard that the assembled matrices are correct (not just fast).
    println()
    println("-- CPU solution accuracy (cut-cell Dirichlet MMS, expect ~2nd order) --")
    prev_err = NaN
    for r in cpu_results
        _, _, vf = assemble_bench_system(r.N)
        u = reshape(r.x, r.N, r.N)
        err = fluid_l2_error(u, bench_exact, r.N, vf)
        ord = isnan(prev_err) ? NaN : log2(prev_err / err)
        prev_err = err
        @printf("  N=%-6d L2=%.6e  order=%s\n", r.N, err,
                isnan(ord) ? "-" : @sprintf("%.3f", ord))
    end

    # GPU half (no-op without CUDA).
    println()
    gpu_rows = try_run_gpu(cpu_results)
    if gpu_rows === nothing
        println("GPU half: SKIPPED (no CUDA). CPU CHOLMOD benchmark complete.")
    else
        println("=== cuDSS F64 GPU results (parity vs CPU enforced < 1e-8) ===")
        @printf("%-6s %-10s %-14s %-14s %-16s %-12s\n",
                "N", "cells", "factor_ms", "solve_ms", "cells/s(solve)", "parity_inf")
        for g in gpu_rows
            @printf("%-6d %-10d %-14.3f %-14.3f %-16.3e %-12.3e\n",
                    g.N, g.n, g.t_fact_ms, g.t_solve_ms, g.cells_per_s_solve, g.parity)
        end
        println("GPU half: GREEN (cuDSS parity < 1e-8 at all N).")
    end

    return 0
end

exit(main())
