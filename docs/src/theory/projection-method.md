# Projection Method

## What is this?

The projection method (Chorin, 1967) is an algorithm for solving the incompressible Navier-Stokes equations by splitting the problem into two steps: first advance the velocity ignoring incompressibility, then "project" the result onto the space of divergence-free fields by solving a pressure equation. Think of it as: move the fluid, then fix it so mass is conserved.

## The incompressibility problem

The momentum equation gives us a way to update velocity, but the continuity equation ``\nabla \cdot \mathbf{u} = 0`` provides no evolution equation — it is a **constraint** that must be satisfied at every time step. Pressure is the Lagrange multiplier that enforces this constraint. We do not have an equation for pressure directly; instead, pressure adjusts itself so that the resulting velocity field is divergence-free.

## Helmholtz decomposition

The mathematical foundation is the **Helmholtz-Hodge decomposition**: any smooth vector field can be uniquely split into a divergence-free part and a gradient part:

```math
\mathbf{w} = \mathbf{u} + \nabla \phi, \quad \text{where } \nabla \cdot \mathbf{u} = 0
```

Given any vector field ``\mathbf{w}`` (possibly with non-zero divergence), we can recover the divergence-free part ``\mathbf{u}`` by subtracting a gradient. The scalar ``\phi`` is found by taking the divergence of both sides:

```math
\nabla^2 \phi = \nabla \cdot \mathbf{w}
```

This is a Poisson equation — and this is why pressure solvers are central to incompressible CFD.

## The algorithm

Starting from ``\mathbf{u}^n`` at time ``t^n``, one time step of the projection method proceeds as:

**Step 1 — Predict (ignore pressure):**

```math
\mathbf{u}^* = \mathbf{u}^n + \Delta t\Big(-(\mathbf{u}^n \cdot \nabla)\mathbf{u}^n + \nu \nabla^2 \mathbf{u}^n + \mathbf{f}\Big)
```

This intermediate velocity ``\mathbf{u}^*`` satisfies momentum but generally has ``\nabla \cdot \mathbf{u}^* \neq 0``.

**Step 2 — Solve pressure Poisson equation:**

```math
\nabla^2 p = \frac{1}{\Delta t}\nabla \cdot \mathbf{u}^*
```

with Neumann boundary conditions ``\partial p / \partial n = 0`` on solid walls (see [Boundary Conditions](@ref)).

**Step 3 — Correct (project onto divergence-free space):**

```math
\mathbf{u}^{n+1} = \mathbf{u}^* - \Delta t \nabla p
```

## Why this works

Take the divergence of Step 3:

```math
\nabla \cdot \mathbf{u}^{n+1} = \nabla \cdot \mathbf{u}^* - \Delta t \nabla^2 p
```

But from Step 2, ``\Delta t \nabla^2 p = \nabla \cdot \mathbf{u}^*``, so:

```math
\nabla \cdot \mathbf{u}^{n+1} = 0
```

The corrected velocity is exactly divergence-free (to machine precision, limited only by the accuracy of the Poisson solve).

## Splitting error

The basic projection method introduces a **splitting error** of ``O(\Delta t)`` because pressure is lagged by one time step. More sophisticated variants reduce this:

| Method | Splitting error | Pressure solves/step | Notes |
|--------|----------------|---------------------|-------|
| Chorin projection | ``O(\Delta t)`` | 1 | Simple, current Kraken default |
| Kim & Moin (incremental) | ``O(\Delta t^2)`` | 1 | Uses pressure increment ``\delta p`` |
| SIMPLE | Iterative | 1+ | Steady-state focus, under-relaxation |
| PISO | ``O(\Delta t^2)`` | 2-3 | Multiple corrector steps |

Kraken currently uses the basic Chorin projection. For most transient simulations at moderate ``\Delta t``, the splitting error is acceptable.

## Implementation in Kraken.jl

- [`projection_step!`](@ref) — performs all three steps (predict, Poisson solve, correct) in a single call
- [`solve_poisson_neumann!`](@ref) — CG-based Poisson solver with Neumann BCs for the pressure equation
- [`apply_velocity_bc!`](@ref) — enforces velocity boundary conditions after correction
- `apply_pressure_neumann_bc!` — enforces ``\partial p / \partial n = 0`` on walls

See [Linear Solvers](@ref) for details on the Poisson solve, and [Collocated Grids](@ref) for how Kraken handles pressure-velocity coupling on collocated grids.

## References

- A.J. Chorin, "Numerical solution of the Navier-Stokes equations," *Math. Comp.*, 22:745-762, 1968.
- J. Kim and P. Moin, "Application of a fractional-step method to incompressible Navier-Stokes equations," *J. Comp. Phys.*, 59:308-323, 1985.
- J.B. Bell, P. Colella, and H.M. Glaz, "A second-order projection method for the incompressible Navier-Stokes equations," *J. Comp. Phys.*, 85:257-283, 1989.
