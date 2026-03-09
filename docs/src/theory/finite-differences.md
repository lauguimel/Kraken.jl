# Finite Differences

## What is this?

Finite differences are the simplest way to approximate derivatives on a regular grid. Instead of evaluating a derivative analytically, you estimate it from the values at neighboring grid points. If you know the function values at evenly spaced points, you can compute derivatives with simple arithmetic.

## Grid notation

Consider a 2D uniform grid with spacing ``\Delta x = \Delta y = h``. The value of a field ``\phi`` at grid point ``(i,j)`` is written ``\phi_{i,j}``. The grid has ``N \times N`` interior points, plus ghost cells at the boundaries (see [Boundary Conditions](@ref)).

## Approximating first derivatives

Three classical formulas approximate ``\partial \phi / \partial x`` at point ``i``:

**Forward difference** (uses point ahead):

```math
\left.\frac{\partial \phi}{\partial x}\right|_i \approx \frac{\phi_{i+1} - \phi_i}{h} + O(h)
```

**Backward difference** (uses point behind):

```math
\left.\frac{\partial \phi}{\partial x}\right|_i \approx \frac{\phi_i - \phi_{i-1}}{h} + O(h)
```

**Central difference** (uses both neighbors):

```math
\left.\frac{\partial \phi}{\partial x}\right|_i \approx \frac{\phi_{i+1} - \phi_{i-1}}{2h} + O(h^2)
```

Central differences are second-order accurate — doubling the resolution cuts the error by 4x. Forward and backward differences are only first-order (doubling resolution halves the error). Kraken uses central differences for pressure gradients and diffusion terms.

## Why central is second-order

Taylor-expand ``\phi_{i+1}`` and ``\phi_{i-1}`` around ``\phi_i``:

```math
\phi_{i \pm 1} = \phi_i \pm h\phi'_i + \frac{h^2}{2}\phi''_i \pm \frac{h^3}{6}\phi'''_i + \cdots
```

Subtracting: ``\phi_{i+1} - \phi_{i-1} = 2h\phi'_i + O(h^3)``, so dividing by ``2h`` gives ``\phi'_i + O(h^2)``. The first-order error terms cancel by symmetry.

## The 5-point Laplacian

The Laplacian ``\nabla^2 \phi = \partial^2\phi/\partial x^2 + \partial^2\phi/\partial y^2`` is discretized as:

```math
\nabla^2 \phi_{i,j} \approx \frac{\phi_{i+1,j} + \phi_{i-1,j} + \phi_{i,j+1} + \phi_{i,j-1} - 4\phi_{i,j}}{h^2}
```

This is the **5-point stencil** — each point interacts with its 4 immediate neighbors. It is ``O(h^2)`` accurate and produces a sparse, symmetric matrix when assembled into a linear system (see [Linear Solvers](@ref)).

## Comparison with FEM and FVM

If you come from a finite element (FEM) or finite volume (FVM) background:

| Aspect | FD | FEM | FVM |
|--------|-----|-----|-----|
| Grid | Structured, uniform | Unstructured | Unstructured |
| Basis | Point values | Shape functions | Cell averages |
| Conservation | Not inherent | Weak form | Built-in |
| Implementation | Very simple | Complex assembly | Moderate |
| GPU efficiency | Excellent (regular memory) | Poor (indirect addressing) | Moderate |

FD on uniform grids is the natural choice for GPU computing: the regular memory access pattern maps perfectly to GPU architectures. The trade-off is geometric flexibility — FD needs structured grids, while FEM/FVM handle arbitrary meshes. For the problems Kraken targets (rectangular domains, Cartesian grids), FD is ideal.

## Implementation in Kraken.jl

- [`laplacian!`](@ref) — computes ``\nabla^2 \phi`` using the 5-point stencil on GPU via KernelAbstractions.jl
- [`gradient!`](@ref) — computes ``(\partial p/\partial x, \partial p/\partial y)`` using central differences
- [`divergence!`](@ref) — computes ``\nabla \cdot \mathbf{u}`` using central differences

All operators run on any backend (CPU, CUDA, Metal) thanks to KernelAbstractions.jl. See the [API Reference](@ref) for details.

## References

- R.J. LeVeque, *Finite Difference Methods for Ordinary and Partial Differential Equations*, SIAM, 2007.
- J.C. Strikwerda, *Finite Difference Schemes and PDEs*, 2nd ed., SIAM, 2004.
