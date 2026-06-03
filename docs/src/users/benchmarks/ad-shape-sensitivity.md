# Steady Shape Sensitivity (AD)

Kraken can compute the steady shape sensitivity of cylinder drag, `dCd/dR`, for
the validated D2Q9 TRT/Li-BB confined-cylinder path. The public entry point is
`steady_shape_sensitivity`; the `.krk` surface is a `Sensitivity { qoi = drag,
wrt = radius }` block. The current validated pair is exactly `qoi=:drag` and
`wrt=:radius`.

The result is a derivative of Kraken's own discrete operator at a residual-
converged steady state, not a transient tape. The forward solve first converges
the fixed point

```text
f* = G(f*, R)
```

and the adjoint then solves

```text
(I - dG/df)^T lambda = dJ/df
```

with one-step reverse-mode vector-Jacobian products. Each VJP re-linearises the
same single LBM step at `f*`, so memory is O(1) in the number of forward steps:
there is no unrolled trajectory and no tape of the transient history. Richardson
iteration is attempted first; GMRES is used on the same matrix-free operator when
the contraction is too slow.

## Shape Chain

The radius enters through the cut-link geometry,

```text
R -> q_wall(R) -> {G, J}
```

with the Boolean `is_solid` mask held fixed inside one smooth cut-set interval.
Kraken does not differentiate through the ray-circle precompute. Instead it
contracts Enzyme cotangents with the analytic `dq_wall_dR_cylinder` derivative:

```text
dCd/dR =
  <dCd/dq_wall, dq_wall/dR>
  - Cd/R
  + <d(lambda^T G)/dq_wall, dq_wall/dR>
```

The middle `-Cd/R` term is the direct derivative of the drag normalization
`D = 2R`. The gradient should be read as the smooth-piece derivative; if changing
`R` changes the set of cut links, the discrete `Cd(R)` can kink.

## CPU AD, GPU Production

The differentiated path is deliberately CPU Float64. Enzyme is a weak dependency:
`using Kraken` loads the core AD stubs without Enzyme, and `using Enzyme` activates
`ext/KrakenADExt.jl`, which supplies the reverse passes. Calling the sensitivity
API without loading Enzyme raises a clear error.

This path does not differentiate the fused GPU kernel. The AD step in `src/ad/`
is an unfused, plain-Julia bit mirror of the production TRT/Li-BB step and drag
sum. The permanent anti-drift checks compare the inline AD drag with
`compute_drag_libb_mei_2d` and the inline steady state with the production
`run_cylinder_libb_2d` path; the validated bridge is `Cd` inline equivalent to
production (`Delta = 0`).

## Julia Example

```julia
using Kraken
using Enzyme

result = steady_shape_sensitivity(;
    Nx=48,
    Ny=16,
    cx=12,
    cy=8,
    radius=3.75,
    u_in=0.05,
    ν=0.05,
    qoi=:drag,
    wrt=:radius,
    tol=1e-9,
    max_steps=60_000,
    gmres_tol=1e-9,
    adjoint_tol=1e-8,
    fd_check=true,
    fd_h=0.05,
)

result.gradient        # dCd/dR
result.qoi_value       # Cd at the converged state
result.value           # current alias of the Cd value
result.terms           # explicit q_wall, direct_D, implicit q_wall pieces
result.fd_check.relerr # populated only when fd_check=true
```

The production default tolerance is tighter (`tol=1e-12`). The example above uses
the small `.krk` smoke geometry so the full AD path is cheap enough for local
tests.

## .krk Example

`examples/sensitivity_cylinder.krk` uses the same request declaratively:

```krk
Simulation sensitivity_cylinder D2Q9
Domain L = 48 x 16  N = 48 x 16

Define U = 0.05
Define H = 16
Define cx = 12
Define cy = 8
Define R = 3.75
Define tol = 1e-9
Define gmres_tol = 1e-9
Define adjoint_tol = 1e-8

Physics nu = 0.05

Obstacle cylinder wall(radius = R) { (x - cx)^2 + (y - cy)^2 <= R^2 }

Boundary west  velocity(ux = 4*U*y*(H - y)/H^2, uy = 0)
Boundary east  pressure(rho = 1.0)
Boundary south wall
Boundary north wall

Run 60000 steps

Sensitivity { qoi = drag, wrt = radius }
```

Running this file through `run_simulation` dispatches to the same AD API:

```julia
using Kraken
using Enzyme

result = run_simulation("examples/sensitivity_cylinder.krk")
result.gradient
```

## Validation

The validation ladder checks the derivative machinery before the cylinder case is
trusted:

| Rung | Check | Gate |
|---|---|---|
| C0 | one-step VJP against dense finite-difference Jacobian transpose | Enzyme VJP matches the discrete step |
| C1 | steady fixed-point adjoint on a Poiseuille/body-force case | adjoint solve and finite-difference derivative agree |
| C2 | geometry chain on planar Poiseuille flow rate | analytic anchor `Q proportional to H^3` |
| C3 | confined-cylinder `d(Cd)/dR` | central finite difference agreement to `9e-5` |

The C2 anchor is the substitute reference for the geometry derivative: RheoTool
can compute a drag value for its own finite-volume operator, but it cannot
differentiate Kraken's LBM cut-link operator with respect to `R`. For derivatives
of this discrete operator, the meaningful references are the analytic Poiseuille
law `Q proportional to H^3` and central finite differences of Kraken's own
residual-converged forward solve.

## Caveats

- Only CPU Float64 is supported on the AD path.
- Only `qoi=:drag`, `wrt=:radius` is validated.
- `using Enzyme` is required before calling the API.
- The derivative is a smooth-piece cut-link derivative; avoid radius values where
  a cut link enters or leaves the mask when comparing to finite differences.
- The GPU production path remains the production path for forward simulations;
  this page documents the separate differentiable path and its anti-drift checks.
