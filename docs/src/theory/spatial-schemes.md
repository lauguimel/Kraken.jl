# Spatial Schemes

## What is this?

Spatial schemes determine how we approximate the advection (transport) term ``(\mathbf{u} \cdot \nabla)\phi`` in the Navier-Stokes equations. The choice of scheme controls the trade-off between accuracy, stability, and computational cost. Getting this right is critical: a bad advection scheme either blows up (unstable) or smears out all the interesting flow features (too diffusive).

## The advection problem

Consider the 1D advection equation — the simplest transport model:

```math
\frac{\partial \phi}{\partial t} + u\frac{\partial \phi}{\partial x} = 0
```

This says that ``\phi`` is carried along by velocity ``u`` without changing its shape. A good numerical scheme should preserve this property. In 2D, the advection of a field ``\phi`` by velocity ``(u,v)`` is:

```math
(\mathbf{u} \cdot \nabla)\phi = u\frac{\partial \phi}{\partial x} + v\frac{\partial \phi}{\partial y}
```

## Central difference scheme

Using central differences for ``\partial \phi / \partial x``:

```math
u_i \frac{\phi_{i+1} - \phi_{i-1}}{2h}
```

This is second-order accurate (``O(h^2)``), but has a fatal flaw: it produces **oscillations** when the local Peclet number ``Pe = |u|h/\nu`` exceeds 2. In convection-dominated flows (high ``Re``), central differences for advection generate unphysical wiggles that can crash the simulation.

## First-order upwind

The upwind scheme uses information from the direction the flow is coming from:

```math
u_i \frac{\partial \phi}{\partial x} \approx \begin{cases} u_i \dfrac{\phi_i - \phi_{i-1}}{h} & \text{if } u_i > 0 \\[6pt] u_i \dfrac{\phi_{i+1} - \phi_i}{h} & \text{if } u_i < 0 \end{cases}
```

This is unconditionally stable (no oscillations) but only first-order accurate. It adds **numerical diffusion** proportional to ``|u|h/2`` — like adding artificial viscosity. Sharp gradients get smeared out over time. This is the current default in Kraken.

## TVD schemes (future)

**Total Variation Diminishing** schemes achieve the best of both worlds: second-order accuracy in smooth regions, with no spurious oscillations near sharp gradients. The key idea is **flux limiters** — functions that blend between upwind (safe) and central (accurate) depending on the local solution smoothness.

For a quantity ``\phi`` transported at velocity ``u > 0``, the TVD flux at face ``i+1/2`` is:

```math
F_{i+1/2} = u\left(\phi_i + \frac{1}{2}\psi(r)(\phi_i - \phi_{i-1})\right)
```

where ``r = (\phi_{i+1} - \phi_i)/(\phi_i - \phi_{i-1})`` measures the ratio of successive gradients, and ``\psi(r)`` is the limiter function.

Common limiters:

| Limiter | Formula | Character |
|---------|---------|-----------|
| Minmod | ``\max(0, \min(1, r))`` | Most diffusive TVD scheme |
| Van Leer | ``(r + |r|)/(1 + |r|)`` | Smooth, good all-rounder |
| Superbee | ``\max(0, \min(2r,1), \min(r,2))`` | Least diffusive, can over-compress |
| MC (monotonized central) | ``\max(0, \min(2r, (1+r)/2, 2))`` | Good balance |

## QUICK scheme

The **Quadratic Upstream Interpolation for Convective Kinematics** (Leonard, 1979) uses a 3-point upstream-biased stencil:

```math
\phi_{i+1/2} = \frac{3}{8}\phi_{i+1} + \frac{6}{8}\phi_i - \frac{1}{8}\phi_{i-1} \quad (u > 0)
```

QUICK is third-order accurate ``O(h^3)`` but **not TVD** — it can still produce small overshoots near discontinuities.

## Comparison

| Scheme | Order | Bounded | Diffusive | Cost | Status in Kraken |
|--------|-------|---------|-----------|------|-----------------|
| Central | 2 | No | No | Low | Used for diffusion |
| Upwind | 1 | Yes | Very | Low | Current advection default |
| Van Leer TVD | 2 | Yes | Low | Medium | Planned |
| QUICK | 3 | No | Low | Medium | Not planned |

## Implementation in Kraken.jl

- [`advect!`](@ref) — computes the advection term ``(\mathbf{u} \cdot \nabla)\phi`` using first-order upwind

The upwind scheme is implemented as a GPU kernel that checks the sign of each velocity component to select the appropriate one-sided difference. See [Finite Differences](@ref) for the underlying stencil operations.

## References

- B.P. Leonard, "A stable and accurate convective modelling procedure based on quadratic upstream interpolation," *Comput. Methods Appl. Mech. Eng.*, 19:59-98, 1979.
- H.K. Versteeg and W. Malalasekera, *An Introduction to Computational Fluid Dynamics: The Finite Volume Method*, 2nd ed., Pearson, 2007.
- P.K. Sweby, "High resolution schemes using flux limiters for hyperbolic conservation laws," *SIAM J. Numer. Anal.*, 21:995-1011, 1984.
