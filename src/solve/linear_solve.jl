# Backend-parametric linear-solve seam with factorize-once caching.
#
# WHY THIS EXISTS
# ---------------
# The pressure-Poisson matrix on a fixed grid is GEOMETRY-ONLY: it is constant
# across every SIMPLE outer iteration (the lid-driven cavity ran ~3000 of them).
# Re-factorizing it each call is pure waste. This module factorizes the constant
# operator ONCE and reuses the factorization for every right-hand side.
#
# THE SEAM
# --------
# A `LinearSolveCache` holds a backend tag plus a cached factorization object.
# Two functions form the entire contract:
#
#   lin_factorize(A; backend, spd, pin_k0) -> LinearSolveCache
#       Factorize the (constant) operator A ONCE for the given backend.
#         * spd=true  -> Cholesky        (SPD operators: viscous / pressure Lap)
#         * spd=false -> LU / LDLᵀ        (non-symmetric or indefinite fallback)
#         * pin_k0>0  -> the operator was pinned at reference dof k0 (singular
#                        all-Neumann pressure); the RHS is pinned per-solve.
#
#   lin_solve!(cache, b) -> x
#       Reuse the cached factorization to solve A x = b for a fresh RHS b.
#       Called once per outer iteration; NEVER re-factorizes.
#
# Dispatch is keyed on the backend tag (a singleton type). The CPU method lives
# here and uses ONLY LinearAlgebra + SparseArrays (CHOLMOD / UMFPACK). The CUDA
# method (`CUDABackendTag`) lives in `linear_solve_cuda.jl` and is loaded ONLY
# inside a job that has `using CUDA, CUDSS` — it implements the SAME two
# functions for `CuSparseMatrixCSR`, so a GPU run drops in with no call-site
# change. Keep this file CUDA-free so it stays include-able on a CPU-only box.
#
# DESIGN NOTE — pinning lives in the cache, not the call site.
# The singular pressure operator is pinned (row/col k0 replaced by identity) at
# factorize time via `pin_reference_dof`. The cache remembers k0 and the ORIGINAL
# unpinned matrix so each `lin_solve!` can pin the RHS consistently. This keeps
# the seam identical for pinned and non-pinned operators.

using LinearAlgebra
using SparseArrays

# `pin_reference_dof` lives in poisson.jl. Pull it in if not already present so
# this file is standalone-include-able.
if !isdefined(@__MODULE__, :pin_reference_dof)
    include(joinpath(@__DIR__, "poisson.jl"))
end

# --------------------------------------------------------------------------
# Backend tags. These are the DISPATCH KEY for the seam. `CPUBackendTag` is the
# default and is handled here; `CUDABackendTag` is handled in linear_solve_cuda.jl
# (loaded only under `using CUDA, CUDSS`). New backends = new tag + two methods.
# --------------------------------------------------------------------------
abstract type LinearSolveBackend end
struct CPUBackendTag <: LinearSolveBackend end
struct CUDABackendTag <: LinearSolveBackend end

"""
    LinearSolveCache

Holds a factorize-once cache for a CONSTANT sparse operator.

Fields:
  `backend`  the backend tag (`CPUBackendTag()` / `CUDABackendTag()`).
  `factor`   the cached factorization object (CHOLMOD/UMFPACK on CPU, a CUDSS
             solver on GPU). Built ONCE; reused for every RHS.
  `A`        the operator actually factorized (already pinned if `pin_k0 > 0`).
  `A_unpinned` the ORIGINAL operator before pinning (== `A` when not pinned);
             used to pin the RHS consistently per solve.
  `pin_k0`   reference dof for a singular (all-Neumann/periodic) operator, or 0.
  `spd`      whether a Cholesky factorization was used.
"""
struct LinearSolveCache{B<:LinearSolveBackend, F, M}
    backend::B
    factor::F
    A::M
    A_unpinned::M
    pin_k0::Int
    spd::Bool
end

# --------------------------------------------------------------------------
# Public entry points. Default `backend = CPUBackendTag()`; pass a tag to route.
# --------------------------------------------------------------------------

"""
    lin_factorize(A; backend=CPUBackendTag(), spd=true, pin_k0=0) -> LinearSolveCache

Factorize the constant operator `A` ONCE for `backend` and return a reusable
cache. `spd=true` selects Cholesky; `spd=false` selects an LU/LDLᵀ fallback.
`pin_k0>0` pins reference dof `k0` (singular operator); the RHS is pinned per
solve inside `lin_solve!`.
"""
function lin_factorize(A::SparseMatrixCSC{Float64,Int};
                       backend::LinearSolveBackend = CPUBackendTag(),
                       spd::Bool = true,
                       pin_k0::Integer = 0)
    return lin_factorize(backend, A; spd = spd, pin_k0 = Int(pin_k0))
end

"""
    lin_solve!(cache, b) -> x

Solve `A x = b` reusing the cached factorization. `b` may be a vector; returns a
`Vector{Float64}`. NEVER re-factorizes. For a pinned operator the RHS is adjusted
to enforce `x[k0] = pin value` consistently with the pinned matrix.
"""
function lin_solve!(cache::LinearSolveCache, b::AbstractVector)
    return lin_solve!(cache.backend, cache, b)
end

# --------------------------------------------------------------------------
# CPU method (CHOLMOD for SPD, LU/LDLᵀ fallback). LinearAlgebra + SparseArrays
# only. This is the reference path; the CUDA method mirrors it exactly.
# --------------------------------------------------------------------------
function lin_factorize(::CPUBackendTag, A::SparseMatrixCSC{Float64,Int};
                       spd::Bool = true, pin_k0::Int = 0)
    A_unpinned = A
    if pin_k0 > 0
        Apin, _ = pin_reference_dof(A, zeros(Float64, size(A, 1)), pin_k0, 0.0)
        A_used = Apin
    else
        A_used = A
    end

    if spd
        factor = cholesky(Symmetric(A_used); check = true)
    else
        # Non-symmetric / indefinite fallback. Try LDLᵀ first (symmetric
        # indefinite), fall back to LU for the fully general case.
        factor = try
            ldlt(Symmetric(A_used))
        catch
            lu(A_used)
        end
    end

    return LinearSolveCache(CPUBackendTag(), factor, A_used, A_unpinned,
                            Int(pin_k0), spd)
end

function lin_solve!(::CPUBackendTag, cache::LinearSolveCache, b::AbstractVector)
    bvec = Vector{Float64}(b)
    if cache.pin_k0 > 0
        _, bpin = pin_reference_dof(cache.A_unpinned, bvec, cache.pin_k0, 0.0)
        return cache.factor \ bpin
    else
        return cache.factor \ bvec
    end
end
