# lattice/

D2Q9 and D3Q19 lattice definitions — velocity vectors `c_q`, weights
`w_q`, lattice speed of sound, and MRT collision matrices. The numerical
primitives every other module builds on.

## Key entry points

| File | Symbol | Purpose |
|---|---|---|
| `lattice.jl` | `AbstractLattice`, common interface | Shared types and helper functions |
| `d2q9.jl` | `D2Q9`, `weights_d2q9`, `c_d2q9` | 9-velocity 2D lattice (the workhorse) |
| `d3q19.jl` | `D3Q19`, `weights_d3q19`, `c_d3q19` | 19-velocity 3D lattice |

## Critical invariants

- **`Σ_q w_q = 1`** (probability normalisation).
- **`Σ_q w_q c_q = 0`** (no net drift in equilibrium).
- **`c_s² = 1/3`** in lattice units (isothermal compressible LBM).
- **`opposite[q]` is involutive**: `opposite[opposite[q]] == q` (used by
  every bounce-back and half-way reflection).
- **MRT matrices** (where defined) are Hermitian; eigenvalues map to the
  relaxation rates.

## Cross-module dependencies

Reads from: nothing — leaf module.
Provides to: every other physics module (`kernels`, `refinement`,
`multiblock`, `curvilinear`, `rheology`). When a new lattice
(D2Q7, D2Q15, …) is added, this is the only module touched if the
abstraction is honoured.

## Status / scope notes

- Layout assumption: AoS `f[i, j, q]` (or `f[i, j, k, q]` in 3D) — the
  weights are arranged accordingly. The abandoned SoA layout `f[q, i, j]`
  is documented in `feedback_soa_layout` and MUST NOT be reintroduced
  here.
- All values are `Float64` by default; runtime `eltype(...)` is propagated
  from the calling kernel (Float32 on Apple Silicon Metal by convention).
