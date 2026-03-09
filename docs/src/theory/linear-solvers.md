# Linear Solvers

## What is this?

At every time step, the [projection method](@ref "Projection Method") requires solving a **Poisson equation** for pressure: ``\nabla^2 p = f``. After discretization on an ``N \times N`` grid, this becomes a large sparse linear system ``A\mathbf{x} = \mathbf{b}`` with ``N^2`` unknowns. The pressure solver is typically the most expensive part of an incompressible flow simulation, so choosing the right algorithm matters enormously.

## The pressure Poisson equation

The discrete Laplacian with the [5-point stencil](@ref "Finite Differences") produces a matrix ``A`` that is:

- **Sparse**: each row has at most 5 non-zero entries
- **Symmetric**: ``A = A^T``
- **Negative semi-definite**: all eigenvalues ``\leq 0`` (with one zero eigenvalue for Neumann BCs)

For an ``N \times N`` grid, ``A`` is an ``N^2 \times N^2`` matrix, but only ``\sim 5N^2`` entries are non-zero.

## Direct solvers

Gaussian elimination or LU factorization solve ``A\mathbf{x} = \mathbf{b}`` exactly, but cost ``O(N^3)`` for a dense matrix and ``O(N^{3/2})`` for a 2D sparse matrix (using nested dissection). For a ``256 \times 256`` grid, that is 65,536 unknowns — direct solvers become impractical. More importantly, they require assembling the matrix explicitly, which conflicts with GPU memory patterns.

## FFT spectral solver

For **periodic boundary conditions**, the Poisson equation can be solved in ``O(N^2 \log N)`` using the Fast Fourier Transform. This exploits the fact that the eigenvectors of the discrete Laplacian on a periodic grid are exactly the Fourier modes.

Transform to Fourier space, divide by the eigenvalues, transform back:

```math
\hat{p}_{k,l} = \frac{\hat{f}_{k,l}}{\lambda_{k,l}}
```

where the eigenvalues of the 2D discrete Laplacian are:

```math
\lambda_{k,l} = \frac{2(\cos(2\pi k/N) - 1) + 2(\cos(2\pi l/N) - 1)}{h^2}
```

**Zero mode treatment**: For a purely Neumann problem, ``\lambda_{0,0} = 0`` (the constant mode). Since pressure is only defined up to a constant, we set ``\hat{p}_{0,0} = 0``.

The FFT solver is:
- Exact (no iteration, no convergence issues)
- Very fast: ``O(N^2 \log N)``
- GPU-friendly: FFT libraries (cuFFT, Metal FFT) are highly optimized
- Limited to periodic or special boundary conditions

## Conjugate Gradient (CG)

For **general boundary conditions** (Dirichlet, Neumann, mixed), we use the Conjugate Gradient method — an iterative solver for symmetric positive definite (SPD) systems.

Since the discrete Laplacian is negative semi-definite, we solve ``-A\mathbf{x} = -\mathbf{b}`` to get an SPD system (or positive semi-definite for Neumann BCs, where we pin one value).

The CG algorithm:

1. Start with an initial guess ``\mathbf{x}_0`` (typically the previous time step's pressure)
2. Compute residual ``\mathbf{r}_0 = \mathbf{b} - A\mathbf{x}_0``
3. Iterate: update ``\mathbf{x}`` along conjugate directions that minimize the error
4. Stop when ``\|\mathbf{r}_k\| / \|\mathbf{r}_0\| < \epsilon``

Key properties:

- **Matrix-free**: only needs the operation ``\mathbf{y} = A\mathbf{x}`` (a stencil sweep), never assembles ``A``
- **Convergence**: at most ``N^2`` iterations for exact solution, but typically ``O(\sqrt{\kappa})`` iterations suffice, where ``\kappa`` is the condition number
- **GPU-friendly**: each iteration is a stencil apply + dot products + vector updates — all highly parallel

### Condition number and preconditioning

For the discrete Laplacian on an ``N \times N`` grid, ``\kappa \sim N^2``, so CG needs ``O(N)`` iterations. Preconditioning reduces this. Common choices:

| Preconditioner | Iterations | GPU-friendly | Complexity |
|----------------|-----------|--------------|------------|
| None | ``O(N)`` | Yes | Simple |
| Jacobi (diagonal) | ``O(N)`` (better constant) | Yes | Trivial |
| ILU | ``O(\sqrt{N})`` | No (sequential) | Moderate |
| Multigrid | ``O(1)`` | Challenging | Complex |

Kraken uses **Jacobi preconditioning** (dividing by the diagonal of ``A``) — it barely changes the iteration count for the Laplacian (diagonal is constant), but it is the only option that maps well to GPUs. Algebraic multigrid would be ideal but is complex to implement on GPU.

## Implementation in Kraken.jl

Kraken provides two Poisson solvers:

- [`solve_poisson_fft!`](@ref) — FFT-based solver for periodic boundary conditions, ``O(N^2 \log N)``
- [`solve_poisson_cg!`](@ref) — CG solver for Dirichlet boundary conditions
- [`solve_poisson_neumann!`](@ref) — CG solver for Neumann boundary conditions (used in the projection method)

The CG solvers are fully matrix-free and run on any KernelAbstractions.jl backend. The FFT solver uses the AbstractFFTs.jl interface.

## References

- Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM, 2003.
- L.N. Trefethen and D. Bau, *Numerical Linear Algebra*, SIAM, 1997.
- J.W. Cooley and J.W. Tukey, "An algorithm for the machine calculation of complex Fourier series," *Math. Comp.*, 19:297-301, 1965.
