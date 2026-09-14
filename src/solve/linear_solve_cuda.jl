# CUDA / cuDSS method for the backend-parametric linear-solve seam.
#
# DO NOT LOAD THIS ON A CPU-ONLY MACHINE. It requires `using CUDA, CUDSS` to be
# in scope BEFORE it is included (the Aqua GPU job's project, not the main
# Project.toml). The CPU seam in `linear_solve.jl` stays CUDA-free; this file
# adds the `CUDABackendTag` methods of `lin_factorize` / `lin_solve!` so a GPU
# run drops in with NO call-site change.
#
# CUDSS.jl API (NVIDIA cuDSS direct sparse solver, F64), per the generic
# LinearAlgebra interface it exposes for `CuSparseMatrixCSR`:
#   * cholesky(A; view='F')  -> CudssSolver   (symbolic ANALYSIS + numeric
#                                               FACTORIZATION done once)
#   * ldlt(A; view='F')      -> CudssSolver   (symmetric indefinite)
#   * lu(A)                  -> CudssSolver   (general)
#   * x = F \ b   /   ldiv!(x, F, b)          -> reuse the factors for each RHS
#   * cholesky!(F, A_new)                     -> re-numeric-factorize, reuse symbolic
# These mirror the CPU CHOLMOD/UMFPACK API exactly, which is why the seam is the
# same two functions with the same signatures.
#
# Usage from the Aqua bench driver:
#   using CUDA, CUDSS
#   include("src/solve/linear_solve.jl")          # CPU seam + tags
#   include("src/solve/linear_solve_cuda.jl")     # this file (CUDA methods)
#   A_gpu = CUSPARSE.CuSparseMatrixCSR(A_cpu)      # F64
#   cache = lin_factorize(A_gpu; backend=CUDABackendTag(), spd=true)
#   x_gpu = lin_solve!(cache, b_gpu)

using LinearAlgebra
using SparseArrays
using CUDA
using CUDA.CUSPARSE
using CUDSS

# Tags + the LinearSolveCache type live in linear_solve.jl; require it first.
if !isdefined(@__MODULE__, :CUDABackendTag)
    include(joinpath(@__DIR__, "linear_solve.jl"))
end

# Accepted GPU matrix type. cuDSS works on CSR; the bench converts CPU CSC ->
# CuSparseMatrixCSR before calling here.
const _CUDSS_GPU_MAT = CuSparseMatrixCSR{Float64,Int32}

# --------------------------------------------------------------------------
# CUDA `lin_factorize`: symbolic + numeric factorization done ONCE.
#
# For the pinned singular pressure operator we expect the CALLER to pass the
# ALREADY-PINNED matrix as `A` and the unpinned matrix as `A_unpinned` (a GPU
# CSR), because `pin_reference_dof` is a CPU SparseMatrixCSC routine. The bench
# pins on the CPU once (geometry-only, cheap) and uploads both. `pin_k0`>0 then
# only drives the per-solve RHS pinning, which we do on the GPU below.
# --------------------------------------------------------------------------
function lin_factorize(::CUDABackendTag, A::_CUDSS_GPU_MAT;
                       spd::Bool = true, pin_k0::Int = 0,
                       A_unpinned::_CUDSS_GPU_MAT = A)
    if spd
        # 'F' = full matrix supplied; cuDSS reads the SPD structure. cholesky()
        # performs the symbolic analysis AND the numeric factorization now.
        factor = cholesky(A; view = 'F')
    else
        # General (non-symmetric / indefinite) path: cuDSS LU. Do NOT try
        # ldlt first — cuDSS LDLᵀ assumes a symmetric structure and does not
        # throw on a non-symmetric matrix (mirror of the CPU-seam bug: it
        # would silently factorize the wrong operator). Callers with a known
        # symmetric-indefinite GPU system can call ldlt explicitly.
        factor = lu(A)
    end

    return LinearSolveCache(CUDABackendTag(), factor, A, A_unpinned,
                            Int(pin_k0), spd)
end

# --------------------------------------------------------------------------
# CUDA `lin_solve!`: reuse the cached factors for a fresh RHS.
#
# `b` is a CPU vector OR a CuVector. We materialise a device RHS, apply the
# reference-dof RHS pinning on the GPU when pin_k0>0 (mirrors the CPU
# pin_reference_dof RHS adjustment: subtract column k0 * pin_value, then force
# the pinned row to the pin value; here pin_value = 0), solve in place, and
# return a CuVector. The bench copies it back to host for the parity assert.
# --------------------------------------------------------------------------
function lin_solve!(::CUDABackendTag, cache::LinearSolveCache, b::AbstractVector)
    bdev = b isa CuArray ? CuVector{Float64}(b) : CuVector{Float64}(Vector{Float64}(b))

    if cache.pin_k0 > 0
        k0 = cache.pin_k0
        # RHS pinning consistent with a pin value of 0 (the operator was pinned
        # to identity at row/col k0). With pin_value = 0 the column-subtraction
        # term vanishes; we only force the pinned entry to 0. Done on device to
        # avoid a host round-trip inside the outer loop.
        CUDA.@allowscalar bdev[k0] = 0.0
    end

    xdev = similar(bdev)
    ldiv!(xdev, cache.factor, bdev)   # reuse symbolic + numeric factors
    return xdev
end
