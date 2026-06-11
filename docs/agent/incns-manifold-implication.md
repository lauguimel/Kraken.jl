---
module: incns-manifold
path: src/methods/inc_ns/manifold_flow.jl
owner_concern: localized-inlet-outlet-simple
status: implemented
last_verified: 2026-06-11
depends_on:
  - solve-poisson
  - solve-linear
  - fvfd-operators-2d-grad-div-laplacian
  - scalar-transport
---

# incns-manifold — module implication map

The second black-box staging rung for steady incompressible SIMPLE: localized
west/east velocity inlets, localized pressure-reference outlets, and full-cell
immersed solid plates. It keeps the cavity convention for collocated fields and
returns face velocities for scalar-transport handoff. **Registered in
`src/Kraken.jl`** and exported (`solve_incns_manifold`,
`manifold_full_cell_mask`); also reachable as
`solve(params, IncNS(:manifold))` (`src/methods/inc_ns/method.jl`).

## Public surface

- `solve_incns_manifold(; nx, ny, Lx, Ly, Re, U_in, is_solid=nothing, inlet,
  outlet, mu=nothing, relax=(u=0.7,p=0.3), tol=1e-7, maxiter=4000,
  backend=CPU(), verbose=false) -> NamedTuple`. `inlet` is
  `(; side::Symbol, j0::Int, j1::Int, u::Float64)`; `outlet` is
  `(; side::Symbol, j0::Int, j1::Int)`. Current localized spans are west/east.
- Returns `u, v, p, uf, vf, is_solid, dx, dy, nx, ny, xcenters, ycenters,
  residual_history, iters, converged, vel_change, mass_imbalance, dp, Re, mu,
  U_in, Lx, Ly, checkerboard`.
- `uf[i,j]` is the east face of cell `(i,j)` and `vf[i,j]` is the north face,
  matching `cavity.jl` and the `solve_scalar_transport` consumer contract.
- `manifold_full_cell_mask(nx, ny, Lx, Ly, plates)` builds axis-aligned
  full-cell plate masks from grid-line-aligned coordinate ranges.

## Reads from

- `src/solve/poisson.jl`: `pin_reference_dof`, indirectly through the linear
  seam include guard.
- `src/solve/linear_solve.jl`: `lin_factorize` / `lin_solve!` /
  `CPUBackendTag`. Momentum and pressure systems both use `spd=true`
  Cholesky; pressure uses `pin_k0=0`.
- `src/fvfd/operators_2d_grad_div_laplacian.jl`: guarded include for the
  existing FVFD operator context/constants. The manifold solver mirrors the
  regular full-cell `is_solid` semantics but assembles its own SIMPLE matrices.
- `src/methods/scalar_transport/thermal_transport.jl` consumes the returned
  `uf`, `vf`, `is_solid`, `dx`, `dy`, `nx`, and `ny` in the smoke handoff.

## Writes to

Nothing global. The solver allocates fields and factorization caches inside the
call, mutates them across SIMPLE iterations, and returns a NamedTuple. Scratch
driver output is printed by `test/scratch/incns_manifold_driver.jl`; no files
are written.

## Backend constraints

CPU Float64 path only. Sparse systems are assembled as `SparseMatrixCSC` and
factorized once with CHOLMOD via the linear-solve seam. `backend=CPU()` is kept
for API consistency with the existing standalone bricks; GPU lowering is
deferred to a future seam-compatible implementation.

## Failure modes

- Outlet Dirichlet is the pressure reference. Do **not** additionally pin the
  pressure-correction matrix; `pin_k0` must stay `0`.
- Inlet faces must stay homogeneous Neumann on the pressure-correction side and
  must be skipped by face correction. Correcting the inlet changes the imposed
  inlet flux and breaks the global mass balance.
- Full-cell geometry is an assumption, not a cut-cell approximation. Plate
  coordinates must lie on grid lines; a mid-cell plate edge silently loses the
  partial face physics and must be rejected.
- Fluid-solid pressure faces are Neumann drops; fluid-solid momentum faces are
  no-slip Dirichlet `+2μ/h²` contributions. Reusing the pressure stencil for
  momentum would create slip along plates.
- Re≈48 manifold SIMPLE can be sensitive to under-relaxation. The scratch
  battery driver uses `relax.u≈0.2`; `relax.u≈0.5` may diverge on the localized
  inlet/outlet case.
- West-boundary inlet flux is part of the internal projection but is not stored
  in `uf` because the cavity face layout only stores east faces. Consumers that
  need a west boundary advective flux must impose it through their own boundary
  condition.

## Touch order

1. `src/methods/inc_ns/manifold_flow.jl` — boundary masks, matrix assembly,
   Rhie-Chow faces, projection, diagnostics.
2. `test/analytical/incns_manifold.jl` — Poiseuille profile/Δp/order, plate
   sanity, scalar-transport handoff.
3. `test/scratch/incns_manifold_driver.jl` — battery-manifold OpenFOAM
   comparison and per-gap split printout.
4. `src/methods/inc_ns/cavity.jl` — sibling convention reference only; do not
   edit for this rung.
5. `src/solve/linear_solve.jl` — solve seam reference only; both manifold
   systems should remain SPD.
