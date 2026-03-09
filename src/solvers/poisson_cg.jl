using Krylov
using LinearAlgebra

"""
    NegLaplacianOperator{T}

Matrix-free linear operator that applies the negative discrete Laplacian (-∇²)
with Dirichlet boundary conditions (phi=0 on boundary) on a 2D uniform grid.

The negation makes the operator symmetric positive definite (SPD), as required
by the Conjugate Gradient method. The operator works on vectors (reshaped
internally to 2D grids). Only interior points are part of the linear system;
boundary values are implicitly zero.

# Fields
- `n_interior::Int`: number of interior points per dimension
- `dx::T`: grid spacing
"""
struct NegLaplacianOperator{T}
    n_interior::Int
    dx::T
end

Base.size(op::NegLaplacianOperator) = (op.n_interior^2, op.n_interior^2)
Base.size(op::NegLaplacianOperator, d::Int) = op.n_interior^2
Base.eltype(::NegLaplacianOperator{T}) where {T} = T

function LinearAlgebra.mul!(y::AbstractVector, op::NegLaplacianOperator, x::AbstractVector)
    n = op.n_interior
    inv_dx2 = one(op.dx) / (op.dx * op.dx)

    # Reshape to 2D for stencil application
    X = reshape(x, n, n)
    Y = reshape(y, n, n)

    # Apply -∇² (negated Laplacian) so eigenvalues are positive
    @inbounds for j in 1:n, i in 1:n
        # Neighbors with Dirichlet BC (phi=0 outside interior)
        xim = i > 1 ? X[i-1, j] : zero(eltype(x))
        xip = i < n ? X[i+1, j] : zero(eltype(x))
        xjm = j > 1 ? X[i, j-1] : zero(eltype(x))
        xjp = j < n ? X[i, j+1] : zero(eltype(x))

        Y[i, j] = (4 * X[i, j] - xim - xip - xjm - xjp) * inv_dx2
    end

    return y
end

"""
    JacobiPreconditioner{T}

Jacobi (diagonal) preconditioner for the negative discrete Laplacian (-∇²).
The diagonal of -∇² is 4/dx², so the preconditioner multiplies by dx²/4.

# Fields
- `inv_diag::T`: the value dx²/4
"""
struct JacobiPreconditioner{T}
    inv_diag::T
end

Base.size(p::JacobiPreconditioner) = error("Not needed")
Base.eltype(::JacobiPreconditioner{T}) where {T} = T

function LinearAlgebra.mul!(y::AbstractVector, M::JacobiPreconditioner, x::AbstractVector)
    y .= M.inv_diag .* x
    return y
end

"""
    solve_poisson_cg!(phi, f, dx; maxiter=1000, rtol=1e-10)

Solve the 2D Poisson equation nabla^2 phi = f with Dirichlet boundary conditions
(phi=0 on boundary) using the Conjugate Gradient method (Krylov.jl).

The solver uses a matrix-free Laplacian operator and a Jacobi preconditioner.
Only interior points are solved; boundary values in `phi` are set to zero.

# Arguments
- `phi`: output array (N x N), will be overwritten with the solution
- `f`: right-hand side array (N x N)
- `dx`: uniform grid spacing

# Keyword Arguments
- `maxiter::Int`: maximum CG iterations (default: 1000)
- `rtol`: relative tolerance (default: 1e-10)

# Returns
- `(phi, niter)`: the solution array and the number of CG iterations

# Example
```julia
N = 66; dx = 1.0 / (N - 1)
phi = zeros(N, N)
f = zeros(N, N)
phi, niter = solve_poisson_cg!(phi, f, dx)
```
"""
function solve_poisson_cg!(phi, f, dx; maxiter=1000, rtol=1e-10)
    N = size(f, 1)
    T = eltype(f)
    n = N - 2  # interior points per dimension

    # Extract interior RHS as a vector, negated for SPD system: -∇²φ = -f
    b = vec(-f[2:N-1, 2:N-1])

    # Create matrix-free operator (-∇², SPD) and Jacobi preconditioner
    A = NegLaplacianOperator{T}(n, dx)
    M = JacobiPreconditioner{T}(T(dx * dx / 4))

    # Solve with CG
    (x, stats) = cg(A, b; M=M, ldiv=false, itmax=maxiter, rtol=rtol, atol=zero(T))

    # Write solution back to phi (boundary stays zero)
    phi .= zero(T)
    phi[2:N-1, 2:N-1] .= reshape(x, n, n)

    return phi, stats.niter
end
