# Time Integration

## What is this?

Time integration schemes advance the solution from one time step to the next. After discretizing in space (see [Finite Differences](@ref)), we are left with a system of ordinary differential equations (ODEs) in time: ``d\mathbf{u}/dt = \mathbf{F}(\mathbf{u})``. The choice of time integrator determines accuracy, stability, and how large a time step ``\Delta t`` we can take.

## The ODE perspective

After spatial discretization, the Navier-Stokes equations reduce to:

```math
\frac{d\mathbf{u}}{dt} = \mathbf{F}(\mathbf{u}) = -\text{Adv}(\mathbf{u}) + \nu\,\text{Lap}(\mathbf{u}) - \text{Grad}(p) + \mathbf{f}
```

where ``\text{Adv}``, ``\text{Lap}``, and ``\text{Grad}`` are the discrete advection, Laplacian, and gradient operators. We need to march this system forward in time.

## Explicit Euler

The simplest scheme — evaluate the right-hand side at the current time:

```math
\mathbf{u}^{n+1} = \mathbf{u}^n + \Delta t\,\mathbf{F}(\mathbf{u}^n)
```

This is first-order accurate ``O(\Delta t)`` and requires only one evaluation of ``\mathbf{F}`` per step. The downside: it is **conditionally stable**, meaning ``\Delta t`` must be small enough or the solution blows up. This is the current scheme used in Kraken's [`projection_step!`](@ref).

## Runge-Kutta methods

Higher-order explicit schemes reduce the time error without changing the spatial discretization.

**RK2 (Heun's method):**

```math
\mathbf{k}_1 = \mathbf{F}(\mathbf{u}^n)
```
```math
\mathbf{k}_2 = \mathbf{F}(\mathbf{u}^n + \Delta t\,\mathbf{k}_1)
```
```math
\mathbf{u}^{n+1} = \mathbf{u}^n + \frac{\Delta t}{2}(\mathbf{k}_1 + \mathbf{k}_2)
```

Second-order, ``O(\Delta t^2)``, two ``\mathbf{F}`` evaluations per step.

**RK4 (classical):** Fourth-order, ``O(\Delta t^4)``, four evaluations per step. Excellent accuracy but still CFL-limited.

## Implicit Euler

Evaluate ``\mathbf{F}`` at the **new** time level:

```math
\mathbf{u}^{n+1} = \mathbf{u}^n + \Delta t\,\mathbf{F}(\mathbf{u}^{n+1})
```

This is **unconditionally stable** — you can take arbitrarily large time steps without the solution blowing up. The price: you must solve a (potentially nonlinear) system of equations at each step. For CFD, this means solving a large sparse linear system. First-order accurate, ``O(\Delta t)``.

## BDF2 (target for Kraken V1)

The **second-order Backward Differentiation Formula** uses two previous time levels:

```math
\frac{3\mathbf{u}^{n+1} - 4\mathbf{u}^n + \mathbf{u}^{n-1}}{2\Delta t} = \mathbf{F}(\mathbf{u}^{n+1})
```

This is second-order ``O(\Delta t^2)`` and unconditionally stable (A-stable). It is the workhorse of industrial CFD codes (OpenFOAM's `backward` scheme, Fluent's second-order implicit). BDF2 is planned for a future Kraken release.

## CFL condition

Explicit schemes require ``\Delta t`` below a stability limit. Two constraints arise:

**Advective CFL** (Courant-Friedrichs-Lewy):

```math
\Delta t < \frac{\Delta x}{|\mathbf{u}|_{\max}} \cdot C
```

where ``C \leq 1`` is the Courant number. Physically: information must not travel more than one cell per time step.

**Diffusive stability:**

```math
\Delta t < \frac{\Delta x^2}{2d\,\nu}
```

where ``d`` is the number of spatial dimensions. This is typically more restrictive at low Reynolds numbers or fine grids.

## Automatic time stepping

In practice, ``\Delta t`` is chosen as the minimum of both constraints:

```math
\Delta t = \min\left(C \cdot \frac{\Delta x}{|\mathbf{u}|_{\max}},\; Fo \cdot \frac{\Delta x^2}{\nu}\right)
```

where ``C`` is the target Courant number (typically 0.5) and ``Fo`` is the target Fourier number (typically 0.25). This ensures stability while taking the largest safe step.

## Comparison

| Scheme | Order | Stable | Evaluations/step | Status in Kraken |
|--------|-------|--------|-------------------|-----------------|
| Explicit Euler | 1 | CFL-limited | 1 | Current default |
| RK2 | 2 | CFL-limited | 2 | Planned |
| RK4 | 4 | CFL-limited | 4 | Planned |
| Implicit Euler | 1 | Unconditional | 1 + linear solve | Not planned |
| BDF2 | 2 | Unconditional | 1 + linear solve | Planned (V1) |

## Implementation in Kraken.jl

- [`projection_step!`](@ref) — currently uses explicit Euler for the velocity prediction step

The time step ``\Delta t`` is specified by the user in the simulation configuration (see [`SimulationConfig`](@ref)). Automatic CFL-based time stepping is planned for a future release.

## References

- E. Hairer and G. Wanner, *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*, Springer, 1996.
- U. Ascher and L. Petzold, *Computer Methods for Ordinary Differential Equations and Differential-Algebraic Equations*, SIAM, 1998.
