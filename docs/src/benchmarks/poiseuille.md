# Poiseuille Flow

## Problem Description

Pressure-driven channel flow between two parallel plates with no-slip walls. A constant body force ``f_x = 8\nu U_{\max}/H^2`` drives the flow to a steady parabolic velocity profile. Domain ``[0,1]^2`` with periodic boundaries in ``x`` and no-slip walls at ``y=0`` and ``y=H``.

```
       periodic ←────────────→ periodic
       ┌──────────────────────┐ y = H (wall, u=0)
       │   ───→  ───→  ───→  │
       │ ════→  ════→  ════→  │  ← parabolic profile
       │   ───→  ───→  ───→  │
       └──────────────────────┘ y = 0 (wall, u=0)
              f_x →  (body force)
```

## Equations

At steady state, the momentum equation reduces to:

```math
\nu \frac{\partial^2 u}{\partial y^2} + f_x = 0
```

with ``u(0) = u(H) = 0`` and ``\nu = 0.01``, ``U_{\max} = 1``, ``H = 1``.

## Exact Solution

```math
u(y) = \frac{4 U_{\max}}{H^2} \, y(H - y)
```

This parabolic profile has maximum velocity ``U_{\max} = 1`` at the channel centerline ``y = H/2``.

## Implementation

The simulation uses only [`laplacian!`](@ref) (pure diffusion with body force) and marches to steady state:

```julia
for step in 1:nsteps
    fill!(lap, 0.0)
    laplacian!(lap, u, dx)
    for j in 2:N-1, i in 2:N-1
        u[i, j] += dt * (ν * lap[i, j] + f_x)
    end
    apply_poiseuille_bc!(u, v, N)
end
```

The time step satisfies the diffusion stability criterion ``\Delta t = 0.2 \, \Delta x^2/\nu`` and the simulation runs for ``t = 50`` (about 5 diffusion time scales ``H^2/\nu``).

## Results

### Velocity Profile

![u(y) profile compared to exact parabola](../assets/figures/poiseuille_profile.png)

### Error Profile

![Absolute error along the channel height](../assets/figures/poiseuille_error.png)

### Performance

| Grid | CPU time (s) | Metal time (s) | Speedup |
|------|-------------|----------------|---------|
| 64x64 | TBD | TBD | TBD |

*Measured on Apple M-series, Julia 1.12*

## References

- [1] Poiseuille, J. L. M. (1846). Recherches experimentales sur le mouvement des liquides dans les tubes de tres-petits diametres. *Memoires presentes par divers savants a l'Academie Royale des Sciences de l'Institut de France*, 9, 433-544.
