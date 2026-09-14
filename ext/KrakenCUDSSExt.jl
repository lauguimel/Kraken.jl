module KrakenCUDSSExt

# cuDSS (GPU direct sparse) backend for the factorize-once linear-solve seam
# (issue #8). Loaded by `using CUDSS` (weakdep trigger). CUDA is a strong dep
# of Kraken, usable here directly (Julia >= 1.11 lets extensions use the
# parent's strong deps — do NOT also list CUDA in [weakdeps], that breaks the
# ext precompile env). Zero impact when CUDA/CUDSS are absent.
#
# ARCHITECTURE NOTE — CUDSS without LinearSolve. This ext routes through
# Kraken's OWN seam (`lin_factorize`/`lin_solve!` on `CUDABackendTag`), not
# through LinearSolve.jl's CUDSS bindings: the two exts stay independent
# (either can load without the other) and the GPU path mirrors the proven
# manual-load file `src/solve/linear_solve_cuda.jl` (Aqua GPU jobs), which it
# supersedes for package users. The manual file attaches its methods to the
# INCLUDING module (bench scripts' Main), so both can coexist without method
# clashes.

using Kraken
using CUDA
using CUDA.CUSPARSE
using CUDSS
using SparseArrays
using LinearAlgebra

import Kraken: lin_factorize, lin_solve!, _cudss_factorize

# cuDSS operates on CSR; host CSC operators are converted on upload.
const _CUDSS_GPU_MAT = CuSparseMatrixCSR{Float64,Int32}

"""
    lin_factorize(A::CuSparseMatrixCSR; backend=CUDABackendTag(), spd=true,
                  pin_k0=0, A_unpinned=A) -> LinearSolveCache

Device-matrix entry of the factorize-once seam (generic keyword form, mirrors
the host `SparseMatrixCSC` entry in src/solve/linear_solve.jl). Because
`Kraken.pin_reference_dof` is a CPU CSC routine, callers pass the
ALREADY-PINNED matrix as `A` plus the unpinned matrix via `A_unpinned`;
`pin_k0 > 0` then only drives the per-solve RHS pinning (done on device).
Prefer the host-matrix entry (`lin_factorize(A_csc; backend=CUDABackendTag(),
pin_k0=k0)`) which handles pinning and upload for you.
"""
function lin_factorize(A::_CUDSS_GPU_MAT;
                       backend::Kraken.LinearSolveBackend = Kraken.CUDABackendTag(),
                       spd::Bool = true, pin_k0::Integer = 0,
                       A_unpinned::_CUDSS_GPU_MAT = A)
    return lin_factorize(backend, A; spd = spd, pin_k0 = Int(pin_k0),
                         A_unpinned = A_unpinned)
end

# CUDA `lin_factorize`: cuDSS symbolic + numeric factorization done ONCE.
# Mirror of src/solve/linear_solve_cuda.jl (see that file's header for the
# CUDSS.jl API notes). `spd=false` goes straight to LU — cuDSS LDLᵀ silently
# mis-factorizes non-symmetric matrices, same trap as the CPU seam.
function lin_factorize(::Kraken.CUDABackendTag, A::_CUDSS_GPU_MAT;
                       spd::Bool = true, pin_k0::Int = 0,
                       A_unpinned::_CUDSS_GPU_MAT = A)
    factor = spd ? cholesky(A; view = 'F') : lu(A)
    return Kraken.LinearSolveCache(Kraken.CUDABackendTag(), factor, A,
                                   A_unpinned, Int(pin_k0), spd)
end

"""
    _cudss_factorize(::CUDABackendTag, A::SparseMatrixCSC; spd=true, pin_k0=0)
        -> LinearSolveCache

Host-matrix cuDSS factorize (the `lin_factorize(A_csc; backend=
CUDABackendTag())` route declared in src/solve/linear_solve_frontend.jl):
pins reference DOF `pin_k0` on the CPU (geometry-only, done once), uploads
BOTH matrices as `CuSparseMatrixCSR{Float64,Int32}`, and factorizes with cuDSS
(Cholesky for `spd=true`, LU otherwise). This is what makes
[`Kraken.solve_poisson_direct`](@ref) run on GPU with `method=CUDABackendTag()`
and no call-site change.
"""
function _cudss_factorize(::Kraken.CUDABackendTag, A::SparseMatrixCSC{Float64,Int};
                          spd::Bool = true, pin_k0::Int = 0)
    if pin_k0 > 0
        Apin, _ = Kraken.pin_reference_dof(A, zeros(Float64, size(A, 1)),
                                           pin_k0, 0.0)
        A_dev = _CUDSS_GPU_MAT(Apin)
        A_unpinned_dev = _CUDSS_GPU_MAT(A)
    else
        A_dev = _CUDSS_GPU_MAT(A)
        A_unpinned_dev = A_dev
    end
    return lin_factorize(Kraken.CUDABackendTag(), A_dev;
                         spd = spd, pin_k0 = pin_k0,
                         A_unpinned = A_unpinned_dev)
end

"""
    lin_solve!(::CUDABackendTag, cache, b) -> CuVector{Float64}

cuDSS per-RHS solve of the factorize-once seam: reuses the cached symbolic +
numeric factors (`ldiv!`), accepts a host or device `b`, applies the
reference-DOF RHS pinning on device when `cache.pin_k0 > 0` (pin value 0, so
only the pinned entry is forced — no host round-trip in the outer loop), and
returns a `CuVector{Float64}`. Callers needing host data `Array(...)` it back
(as `Kraken.solve_poisson_direct` does).
"""
function lin_solve!(::Kraken.CUDABackendTag, cache::Kraken.LinearSolveCache,
                    b::AbstractVector)
    bdev = b isa CuArray ? CuVector{Float64}(b) :
                           CuVector{Float64}(Vector{Float64}(b))

    if cache.pin_k0 > 0
        CUDA.@allowscalar bdev[cache.pin_k0] = 0.0
    end

    xdev = similar(bdev)
    ldiv!(xdev, cache.factor, bdev)   # reuse symbolic + numeric factors
    return xdev
end

end # module KrakenCUDSSExt
