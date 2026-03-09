# Governing Equations

## What is this?

Kraken.jl solves the **incompressible Navier-Stokes equations** — the fundamental laws governing the motion of fluids like water, air (at low Mach number), and polymeric solutions. These equations express conservation of mass and momentum for a fluid whose density stays constant.

## The equations

For an incompressible, Newtonian fluid with constant density ``\rho`` and kinematic viscosity ``\nu``, two equations govern the velocity field ``\mathbf{u} = (u,v)`` and the pressure field ``p``:

**Continuity (mass conservation):**

```math
\nabla \cdot \mathbf{u} = 0
```

This states that fluid is neither created nor destroyed — what flows in must flow out. In 2D:

```math
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
```

**Momentum conservation:**

```math
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{f}
```

Each term has a physical meaning:

| Term | Name | Physical meaning |
|------|------|------------------|
| ``\partial \mathbf{u}/\partial t`` | Unsteady | Rate of change of velocity |
| ``(\mathbf{u} \cdot \nabla)\mathbf{u}`` | Advection | Fluid carrying its own momentum |
| ``-\nabla p / \rho`` | Pressure gradient | Force pushing fluid from high to low pressure |
| ``\nu \nabla^2 \mathbf{u}`` | Diffusion | Viscous friction smoothing velocity gradients |
| ``\mathbf{f}`` | Body force | External forces (gravity, buoyancy, ...) |

Think of the momentum equation as Newton's second law (``F = ma``) applied to a tiny fluid parcel, where inertia (left side) balances pressure, viscosity, and external forces (right side).

## Non-dimensionalization

By choosing a reference length ``L``, velocity ``U``, and using ``L/U`` as reference time and ``\rho U^2`` as reference pressure, the equations become:

```math
\frac{\partial \mathbf{u}^*}{\partial t^*} + (\mathbf{u}^* \cdot \nabla^*)\mathbf{u}^* = -\nabla^* p^* + \frac{1}{Re}\nabla^{*2} \mathbf{u}^*
```

where the **Reynolds number** ``Re = UL/\nu`` is the single dimensionless parameter controlling the flow. Low ``Re`` (viscous, laminar) vs. high ``Re`` (inertial, turbulent) — this is why the lid-driven cavity benchmark is characterized entirely by ``Re``.

## Boussinesq approximation (future)

For buoyancy-driven flows (natural convection), density variations are small and only matter in the gravity term:

```math
\rho \approx \rho_0\big(1 - \beta(T - T_0)\big)
```

where ``\beta`` is the thermal expansion coefficient. This adds a buoyancy force ``\mathbf{f} = -\beta(T - T_0)\mathbf{g}`` to the momentum equation, coupled with an energy equation for temperature ``T``. This will be part of a future Kraken release.

## Implementation in Kraken.jl

Kraken.jl solves these equations using:

- Spatial derivatives via finite differences: [`laplacian!`](@ref), [`gradient!`](@ref), [`divergence!`](@ref)
- Advection with upwind schemes: [`advect!`](@ref)
- Pressure-velocity coupling via the projection method: [`projection_step!`](@ref)

See [Finite Differences](@ref) and [Projection Method](@ref) for details on the discretization.

## References

- A. Chorin and J.E. Marsden, *A Mathematical Introduction to Fluid Mechanics*, Springer, 1993.
- J.H. Ferziger, M. Perić, and R.L. Street, *Computational Methods for Fluid Dynamics*, 4th ed., Springer, 2020.
- S.B. Pope, *Turbulent Flows*, Cambridge University Press, 2000.
