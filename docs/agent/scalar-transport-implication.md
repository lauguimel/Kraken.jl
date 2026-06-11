---
module: scalar-transport
path: src/methods/scalar_transport/thermal_transport.jl
owner_concern: method
status: implemented
last_verified: 2026-06-11
depends_on:
  - fvfd
  - linear-solve
  - inc_ns
---

# scalar-transport — module implication map

The scalar-transport module is the **single owner** of the DECOUPLED STEADY
scalar advection–diffusion ("thermal transport") brick: given a FROZEN
face-normal velocity field `(uf, vf)`, it solves `∇·(u T) − DT ∇²T = 0` to
steady state. Because `u` is frozen the equation is LINEAR in `T`, so the steady
state is a SINGLE direct solve `A·T = b` — there is NO outer iteration loop. It
assembles a 5-point diffusion stencil plus a first-order UPWIND advection
operator into one sparse CSC, applies Dirichlet / Neumann-flux / zero-gradient
BCs, and solves through the shared factorize-once linear-solve seam. Validated by
`test/analytical/scalar_transport_heated_channel.jl`: pure conduction recovered to
machine precision (l2_rel ≈ 5e-14), second-order diffusion convergence
(order ≈ 2.00), and the constant-flux parallel-plate developed Nusselt number
`Nu = 8.2357` (analytic 8.235, rel error 8.5e-5) on a 400×60 grid, Lx=40.

## Public surface

The brick is a standalone include (NOT exported into `Kraken`, NOT subtyping
`AbstractMethod`), mirroring the `inc_ns/simple.jl` and `inc_ns/cavity.jl`
pattern:

- `solve_scalar_transport(; nx, ny, dx, dy, uf, vf, DT, is_solid=falses(nx,ny), bc, source=nothing, backend=nothing) -> NamedTuple` — the single entry point. Returns `(; T, residual_history, iters, converged, dx, dy, ycenters, nx, ny, DT, Pe_cell)`. `T` is the steady `nx×ny` field; `residual_history` is the one-element `[‖A·T−b‖]` (≈ machine eps); `iters == 1` always (one linear solve); `Pe_cell = max|u|·min(dx,dy)/DT`.
- `bc` is a NamedTuple keyed `west/east/south/north`, each `(kind::Symbol, value)` with `kind ∈ {:dirichlet, :flux, :outflow}`. `:dirichlet` imposes `T = value`; `:flux` imposes a Neumann wall heat flux `q = value` injected as a SOURCE in `b` (not in `A`); `:outflow` is zero-gradient (ghost = interior, advection carries the interior upwind value).
- `source` (optional) is a cell-centred volumetric source field `S[i,j]`; the balance becomes `div(uT) − DT·lap(T) = S` (used by the manufactured second-order convergence test). `nothing` = no source.

Internal (file-private) assembly helper: `_st_assemble_system(nx, ny, dx, dy, uf, vf, DT, is_solid; bc, source) -> (A::SparseMatrixCSC, b::Vector)`.

## Reads from

- `fvfd` (collocated cell-centred FVFD geometry) — the brick mirrors the
  `inc_ns/simple.jl` sparse assembly directly: the linear index `k = i + (j−1)*nx`,
  the 5-point Laplacian coefficients `±DT/h²`, and the `uf[i,j]=east face` /
  `vf[i,j]=north face` velocity layout. It reimplements NONE of the matrix-free KA
  operators — it transcribes their stencil into CSC coefficients.
- `inc_ns` — the UPWIND advection logic is transcribed from `_cavity_convection!`
  (`src/methods/inc_ns/cavity.jl`): first-order donor/acceptor on the frozen face
  fluxes `Fe=uf[i,j]`, `Fw=uf[i-1,j]`, `Fn=vf[i,j]`, `Fs=vf[i,j-1]`, turned into
  matrix coefficients (donor → `+F/h` on the diagonal, acceptor → `−F/h`
  off-diagonal). The Dirichlet wall-value source convention (`+2·DT/h²·T_wall` to
  `b`) mirrors `_cavity_dirichlet_rhs!`.
- `linear-solve` — the factorize-once seam (`src/solve/linear_solve.jl`):
  `LinearSolveCache`, `CPUBackendTag`, and `lin_solve!`. The brick reads the seam's
  CPU solve contract (`cache.factor \ b`).

## Writes to

- **Returns a fresh `T::Matrix{Float64}` (nx×ny)** and the NamedTuple above; it
  mutates none of its inputs (`uf`, `vf`, `is_solid`, `bc`, `source` are read-only).
- **Allocates per call**: the sparse triplet vectors `I/J/V`, the assembled
  `A::SparseMatrixCSC`, the RHS `b`, the LU factorization, and the solution
  vector. There is no per-iteration reuse because there is no iteration — one
  assemble + one factorize + one solve.
- No files written, no global registry mutated, no `using Kraken` side effects.

## Backend constraints

- **CPU-first, KA + stdlib only.** Default `backend=nothing` routes to
  `CPUBackendTag()`; the CPU path uses `lu(A)` (sparse UMFPACK) because the
  upwind advection makes `A` NON-symmetric. The solve dispatches through the
  seam's `lin_solve!`.
- **The seam's `lin_factorize(A; spd=false)` is NOT used for the CPU path** — see
  Failure modes. The brick builds the `LinearSolveCache` with the correct `lu(A)`
  factor directly (the branch the seam INTENDED for non-symmetric operators).
- A non-CPU `backend` tag is forwarded to `lin_factorize(A; backend, spd=false)`;
  the GPU sparse-LU path (cuDSS) is the seam's responsibility, untested here.
- The validated path uses the REGULAR (non-embedded) stencil with
  `is_solid=falses`; cut-cell support is a signature placeholder only.

## Failure modes

- **Seam `spd=false` silently returns a WRONG field for non-symmetric A.** The CPU
  `spd=false` branch in `src/solve/linear_solve.jl` is
  `try ldlt(Symmetric(A)) catch lu(A)`. For a genuinely non-symmetric `A`,
  `ldlt(Symmetric(A))` does NOT throw — it factorizes the SYMMETRIZED matrix (a
  different operator), so the solve returns garbage (verified: residual ~1e2 vs
  ~1e-13 for `lu`). The brick therefore constructs the cache with `lu(A)` itself.
  If this is ever "simplified" back to `lin_factorize(A; spd=false)`, the Nusselt
  rung breaks (Nu reads ~12 instead of 8.24) and `converged` flips false.
- **Outlet contamination of the developed Nu.** The `:outflow` zero-gradient BC
  biases the last ~5% of streamwise cells (Nu drifts to ~8.7); the test averages
  `Nu` over 50%–95% of the channel to stay in the developed region. Averaging
  including the very last cells inflates the reported Nu.
- **First-order upwind numerical diffusion.** At larger cell Péclet the upwind
  scheme adds artificial cross-stream diffusion that biases Nu; the channel uses a
  fine streamwise grid (nx=400, Lx=40, Pe_cell≈0.17) so the developed value is
  stable. Coarsening x without re-checking Nu is the trap.
- **Wall-flux sign convention.** `:flux value=q` injects `+q/h` into `b` on the
  wall row (conductive flux INTO the domain). A sign flip silently inverts the
  near-wall gradient; the conduction rung (machine-precision linear profile) is the
  guard.
- **`source` units.** `source[i,j]` is the RHS of `div(uT) − DT·lap(T) = S`, NOT a
  temperature; passing a temperature field corrupts the manufactured-solution
  convergence order.

## Touch order

For a scalar-transport bug (wrong steady field, non-machine residual, wrong Nu,
bad convergence order), inspect in this order:

1. `src/methods/scalar_transport/thermal_transport.jl` — `_st_assemble_system`:
   the per-face diffusion `±DT/h²` and upwind advection `±F/h` coefficients, the
   per-wall BC branches (`:dirichlet` diagonal `+2DT/h²` + `b` source, `:flux`
   `b += q/h`, `:outflow` interior-upwind diagonal). 90% of wrong-field and
   wrong-Nu bugs are a face coefficient or a BC sign here.
2. `solve_scalar_transport` — the LU-cache construction. If the residual is ~1e2
   instead of ~1e-13, the seam's broken `spd=false` path has been reintroduced;
   confirm the cache holds an `lu(A)` factor.
3. `src/solve/linear_solve.jl` — the seam contract (`LinearSolveCache`,
   `lin_solve!`, `CPUBackendTag`). Only relevant if the `lin_solve!` dispatch or
   the struct fields change; the brick depends on `cache.factor \ b`.
4. `src/methods/inc_ns/cavity.jl` / `simple.jl` — the SOURCE patterns this brick
   mirrors (upwind donor/acceptor, Dirichlet wall-value source, 5-point Laplacian
   assembly). Cross-check here when porting a new BC kind or a cut-cell stencil.
5. `test/analytical/scalar_transport_heated_channel.jl` — the three validation
   rungs (conduction machine precision, manufactured second-order order, developed
   Nu window). Adjust the Nu averaging window or grid here if the developed value
   drifts.
