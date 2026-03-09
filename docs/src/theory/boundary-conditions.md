# Boundary Conditions

## What is this?

Boundary conditions (BCs) specify what happens at the edges of the computational domain. Without them, the equations are not well-posed — there are infinitely many solutions. In Kraken, BCs are implemented using **ghost cells**: an extra layer of fictitious grid points surrounding the physical domain, whose values are set to enforce the desired condition.

## Ghost cells

Consider a 1D domain with ``N`` interior points indexed ``1, \ldots, N``. We add ghost cells at positions ``0`` and ``N+1``. The physical domain uses indices ``1:N``; the ghost cells provide the neighbor values needed by finite difference stencils at the boundaries.

In 2D, the array has size ``(N+2) \times (N+2)``, with interior points at ``(2:N+1, 2:N+1)`` and ghost cells forming a one-cell-wide border.

The advantage of ghost cells: interior stencils work uniformly everywhere, including at boundaries. No special "boundary stencil" is needed — we simply set the ghost cell values before each stencil evaluation.

## Dirichlet boundary conditions (fixed value)

A Dirichlet BC prescribes the value of a variable at the wall. For a wall at the boundary between interior cell ``1`` and ghost cell ``0``, the wall sits at position ``1/2``. To impose ``\phi_{\text{wall}} = \phi_w``:

```math
\frac{\phi_0 + \phi_1}{2} = \phi_w \quad \Rightarrow \quad \phi_0 = 2\phi_w - \phi_1
```

The ghost cell value is set so that linear interpolation to the wall gives exactly ``\phi_w``.

**Example — no-slip wall**: For velocity, ``u_w = 0`` (the fluid does not slip at a solid wall):

```math
u_{\text{ghost}} = -u_{\text{interior}}
```

The ghost velocity mirrors the interior velocity with opposite sign, ensuring ``u = 0`` at the wall face.

## Neumann boundary conditions (fixed gradient)

A Neumann BC prescribes the normal derivative at the wall. For a zero-gradient condition ``\partial \phi / \partial n = 0``:

```math
\frac{\phi_1 - \phi_0}{h} = 0 \quad \Rightarrow \quad \phi_0 = \phi_1
```

The ghost cell simply copies the nearest interior value. This is used for pressure at solid walls: the normal pressure gradient is zero because there is no flow through the wall (from the normal momentum equation at the boundary).

## Lid-driven cavity BCs

The classic benchmark problem uses the following boundary conditions:

| Boundary | Velocity ``u`` | Velocity ``v`` | Pressure ``p`` |
|----------|---------------|---------------|----------------|
| Top (lid) | ``u = U_{\text{lid}}`` (Dirichlet) | ``v = 0`` (Dirichlet) | ``\partial p / \partial n = 0`` (Neumann) |
| Bottom | ``u = 0`` (no-slip) | ``v = 0`` (no-slip) | ``\partial p / \partial n = 0`` (Neumann) |
| Left | ``u = 0`` (no-slip) | ``v = 0`` (no-slip) | ``\partial p / \partial n = 0`` (Neumann) |
| Right | ``u = 0`` (no-slip) | ``v = 0`` (no-slip) | ``\partial p / \partial n = 0`` (Neumann) |

The lid moves at constant velocity ``U_{\text{lid}}`` (set to 1 in dimensionless form), driving the flow by viscous drag. All other walls are stationary no-slip surfaces.

## Periodic boundary conditions

For flows that repeat in one or both directions (channel flow, homogeneous turbulence), periodic BCs wrap the domain:

```math
\phi_0 = \phi_N, \quad \phi_{N+1} = \phi_1
```

The ghost cell on the left takes the value from the rightmost interior cell, and vice versa. This creates a seamless, infinite-domain effect. Periodic BCs are natural for the [FFT-based Poisson solver](@ref "Linear Solvers").

## Order of operations

In a typical time step, boundary conditions are applied at specific moments:

1. **Before advection/diffusion**: apply velocity BCs so that stencil operations near walls use correct values
2. **After velocity prediction**: re-apply velocity BCs to the predicted velocity ``\mathbf{u}^*``
3. **After pressure solve**: apply pressure Neumann BCs
4. **After velocity correction**: apply velocity BCs to the final ``\mathbf{u}^{n+1}``

Getting this order wrong is a common source of subtle bugs in CFD codes.

## Corner treatment

At corners (e.g., bottom-left of the cavity), two boundary conditions meet. The ghost cell at the corner ``(0,0)`` is typically set by averaging the two adjacent ghost cells, or by extrapolation. In Kraken, corner ghost cells are set to maintain consistency with both adjacent walls.

## Implementation in Kraken.jl

- [`apply_velocity_bc!`](@ref) — sets ghost cell values for ``u`` and ``v`` to enforce no-slip walls and the moving lid
- `apply_pressure_neumann_bc!` — copies interior pressure to ghost cells (zero normal gradient)

These functions are called within [`projection_step!`](@ref) at the appropriate stages. See the [Projection Method](@ref) page for the full algorithm.

## References

- J.H. Ferziger, M. Perić, and R.L. Street, *Computational Methods for Fluid Dynamics*, 4th ed., Springer, 2020, Ch. 9.
- U. Ghia, K.N. Ghia, and C.T. Shin, "High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method," *J. Comp. Phys.*, 48:387-411, 1982.
