using Krylov
using LinearAlgebra
using KernelAbstractions
using FFTW

"""
    HelmholtzOperator{T}

Matrix-free linear operator that applies the Helmholtz operator (I - σ·∇²)
with homogeneous Neumann boundary conditions on a 2D uniform grid.

The Helmholtz equation arises from semi-implicit treatment of diffusion:
    (I - σ·∇²) φ = rhs,  where σ = dt·ν

Neumann BCs are implemented by ghost-point reflection. Unlike the pure
Neumann Laplacian, this operator is SPD for σ>0 (the identity removes
the null space), so no pinning is required.

# Fields
- `N::Int`: total grid size (including boundary points)
- `dx::T`: grid spacing
- `sigma::T`: diffusion coefficient σ = dt·ν
"""
struct HelmholtzOperator{T}
    N::Int
    dx::T
    sigma::T
end

Base.size(op::HelmholtzOperator) = (op.N^2, op.N^2)
Base.size(op::HelmholtzOperator, d::Int) = op.N^2
Base.eltype(::HelmholtzOperator{T}) where {T} = T

@kernel function helmholtz_kernel!(Y, @Const(X), inv_dx2, sigma, N)
    idx = @index(Global)
    @inbounds begin
        i = mod1(idx, N)
        j = div(idx - 1, N) + 1

        # Neumann BC: ghost values = interior neighbor (zero gradient)
        xim = i > 1 ? X[idx - 1] : X[idx + 1]
        xip = i < N ? X[idx + 1] : X[idx - 1]
        xjm = j > 1 ? X[idx - N] : X[idx + N]
        xjp = j < N ? X[idx + N] : X[idx - N]

        # (I - σ·∇²)x = x + σ·(-∇²x)/dx²
        neg_lap = (4 * X[idx] - xim - xip - xjm - xjp) * inv_dx2
        Y[idx] = X[idx] + sigma * neg_lap
    end
end

function LinearAlgebra.mul!(y::AbstractVector, op::HelmholtzOperator, x::AbstractVector)
    N = op.N
    inv_dx2 = one(op.dx) / (op.dx * op.dx)
    backend = KernelAbstractions.get_backend(x)

    kernel! = helmholtz_kernel!(backend)
    kernel!(y, x, inv_dx2, op.sigma, N; ndrange=N * N)
    KernelAbstractions.synchronize(backend)

    return y
end

"""
    HelmholtzJacobiPreconditioner{T}

Jacobi preconditioner for the Helmholtz operator (I - σ·∇²).
The diagonal is 1 + 4σ/dx², so the preconditioner multiplies by 1/(1 + 4σ/dx²).

# Fields
- `inv_diag::T`: the value 1/(1 + 4σ/dx²)
"""
struct HelmholtzJacobiPreconditioner{T}
    inv_diag::T
end

Base.size(p::HelmholtzJacobiPreconditioner) = error("Not needed")
Base.eltype(::HelmholtzJacobiPreconditioner{T}) where {T} = T

function LinearAlgebra.mul!(y::AbstractVector, M::HelmholtzJacobiPreconditioner, x::AbstractVector)
    y .= M.inv_diag .* x
    return y
end

"""
    solve_helmholtz!(phi, rhs, dx, sigma; maxiter=2000, rtol=1e-3, solver=nothing, work_b=nothing)

Solve the 2D Helmholtz equation (I - σ·∇²)φ = rhs with Neumann BCs using CG.

This is used for semi-implicit diffusion in the projection method.
When σ=0, the equation reduces to the identity: φ = rhs.

Works on CPU arrays and GPU arrays (CUDA, Metal) automatically.

# Arguments
- `phi`: output array (N × N), will be overwritten with the solution.
- `rhs`: right-hand side array (N × N).
- `dx`: uniform grid spacing.
- `sigma::Real`: diffusion parameter σ = dt·ν (must be ≥ 0).

# Keyword Arguments
- `maxiter::Int`: maximum CG iterations (default: 2000).
- `rtol`: relative tolerance (default: 1e-3).
- `solver`: pre-allocated `CgWorkspace` to avoid per-call allocation (default: nothing).
- `work_b`: pre-allocated work vector of length N² (default: nothing).

# Returns
- `(phi, niter)`: solution array and iteration count.

See also: [`solve_poisson_neumann!`](@ref), [`projection_step_implicit!`](@ref)
"""
function solve_helmholtz!(phi, rhs, dx, sigma; maxiter=2000, rtol=1e-3,
                          solver=nothing, work_b=nothing)
    N = size(rhs, 1)
    T = eltype(rhs)

    # Special case: sigma=0 => identity, phi = rhs
    if sigma ≈ zero(T)
        copyto!(phi, rhs)
        return phi, 0
    end

    # Use pre-allocated work_b or allocate
    b = work_b === nothing ? similar(rhs, N * N) : work_b
    b .= vec(rhs)

    A = HelmholtzOperator{T}(N, dx, T(sigma))
    inv_diag_val = T(1) / (T(1) + T(4) * T(sigma) / (dx * dx))
    M = HelmholtzJacobiPreconditioner{T}(inv_diag_val)

    if solver !== nothing
        # In-place solve with pre-allocated workspace (avoids per-call allocation)
        solver.warm_start = false
        cg!(solver, A, b; M=M, ldiv=false, itmax=maxiter, rtol=T(rtol), atol=zero(T))
        phi .= reshape(solver.x, N, N)
        return phi, solver.stats.niter
    else
        (x, stats) = cg(A, b; M=M, ldiv=false, itmax=maxiter, rtol=T(rtol), atol=zero(T))
        phi .= reshape(x, N, N)
        return phi, stats.niter
    end
end

"""
    solve_helmholtz_dct!(phi, rhs, dx, sigma; poisson_eigenvalues=nothing)

Solve the 2D Helmholtz equation (I - σ·∇²)φ = rhs with Neumann BCs using DCT-I.

This is a direct solver (no iterations) that exploits the fact that the DCT-I
diagonalizes the discrete Neumann Laplacian. The Helmholtz eigenvalues are
`1 - σ·λ_{k,l}` where `λ_{k,l}` are the Poisson (Laplacian) eigenvalues.

Since Poisson eigenvalues are ≤ 0 and σ > 0, all Helmholtz eigenvalues are ≥ 1,
so no division by zero or near-zero occurs. This makes DCT Helmholtz even more
robust than DCT Poisson (which needs special handling of the zero mode).

# Arguments
- `phi`: output array (N × N), will be overwritten with the solution.
- `rhs`: right-hand side array (N × N).
- `dx`: uniform grid spacing.
- `sigma::Real`: diffusion parameter σ = dt·ν (must be > 0).

# Keyword Arguments
- `poisson_eigenvalues`: pre-computed Poisson eigenvalues matrix (N × N).
  If not provided, eigenvalues are computed on the fly.

# Returns
- `(phi, 0)`: solution array and iteration count (always 0 for direct solver).

See also: [`solve_helmholtz!`](@ref), [`solve_poisson_neumann_dct!`](@ref)
"""
function solve_helmholtz_dct!(phi, rhs, dx, sigma; poisson_eigenvalues=nothing)
    N = size(rhs, 1)
    T = eltype(rhs)

    # Forward DCT-I (REDFT00) — matches Neumann BCs with values at grid points
    f_hat = FFTW.r2r(rhs, FFTW.REDFT00)

    # Build or use cached Poisson eigenvalues
    if poisson_eigenvalues !== nothing
        eig = poisson_eigenvalues
    else
        eig = zeros(T, N, N)
        inv_dx2 = one(T) / (dx * dx)
        for l in 1:N, k in 1:N
            eig[k, l] = T(2) * inv_dx2 * (cos(T(π) * T(k - 1) / T(N - 1)) - one(T)) +
                         T(2) * inv_dx2 * (cos(T(π) * T(l - 1) / T(N - 1)) - one(T))
        end
    end

    # Divide by Helmholtz eigenvalues: 1 - sigma * laplacian_eigenvalue
    # Since eig[k,l] ≤ 0, helmholtz_eig = 1 - sigma*eig ≥ 1 (always positive, no singularity)
    @inbounds for l in 1:N, k in 1:N
        helmholtz_eig = one(T) - T(sigma) * eig[k, l]
        f_hat[k, l] /= helmholtz_eig
    end

    # Inverse DCT-I (REDFT00 is self-inverse, normalization = 2*(N-1) per dim)
    norm_factor = T(4) * T(N - 1) * T(N - 1)
    phi .= FFTW.r2r(f_hat, FFTW.REDFT00) ./ norm_factor

    return phi, 0
end
