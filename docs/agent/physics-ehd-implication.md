---
module: physics-ehd
path: src/kernels/ehd_2d.jl
owner_concern: constitutive
status: implemented
last_verified: 2026-07-29
depends_on:
  - lbm
  - bc
---

# physics-ehd — module implication map

The EHD physics module owns the uncoupled electric potential and charge-density
D2Q9 scalar populations for the hydrostatic Coulomb-driven electroconvection
base state. It implements the Jiachen/Luo-Wu-Yi-Tan standard LBM formulas for
`phi` pseudo-time Poisson collision, electric-field recovery from the potential
population first moment, and charge drift-diffusion collision with `u=0`.

## Public surface

- `run_ehd_hydrostatic_2d(; Nx, Ny, C, M, Ma_E, alpha, charge_scheme, backend, FT)` — standalone hydrostatic EHD validation driver. Returns 2D fields, x-averaged profiles, analytic profiles, relative L2 errors, convergence metadata, and lattice parameters.
- Kernel-level surface: `collide_electric_potential_2d!`, `compute_electric_field_2d!`, `collide_electric_charge_srt_2d!`, `collide_electric_charge_regularized_2d!`, `compute_ehd_scalar_2d!`, `apply_phi_nee_walls_2d!`, and `apply_charge_nee_walls_2d!`.

## Reads from

- `lbm` — D2Q9 ordering and the existing `stream_periodic_x_wall_y_2d!` topology: periodic in x, bounded in y.
- `bc` — only the convention that wall-node scalar BCs are applied after streaming. EHD wall values are local non-equilibrium extrapolation kernels owned by this module.

## Writes to

- Mutates potential populations `phi_f`, charge populations `q_f`, scalar moments `phi`/`q`, and electric fields `Ex`/`Ey` in place.
- Allocates per run in the driver: two ping-pong DDF arrays for each scalar, scalar moment arrays, electric-field arrays, and small host copies for convergence tests and returned profiles.
- Does not mutate flow fields, parser state, units registries, or files.

## Backend constraints

- Kernels are KernelAbstractions `@kernel` functions and use `@Const` for read-only arrays.
- Hot kernels are allocation-free and unroll the D2Q9 operations.
- The driver is backend-generic for arrays, but convergence checks copy the small validation fields to the host each step.
- No Coulomb force, Navier-Stokes coupling, MRT/TRT charge collision, `.krk` parser branch, or simulation-runner path is included.

## Failure modes

- `tau_q` is close to `0.5` at `alpha=1e-4`; the hydrostatic validation passes with SRT, but the regularized charge collision is available for later perturbed/coupled cases where the MATLAB driver commonly uses it.
- The wall convention is wall-node non-equilibrium extrapolation. Wall values live at `y*=0,1`; interior DDF profiles are compared on the effective half-link samples `y*=(j-3/2)/(Ny-1)`, matching the discrete charge population location near the injector.
- The analytic E profile is positive upward; if `Ey` changes sign, inspect the potential source sign and D2Q9 direction ordering before changing analytics.
- The current validation assumes x-invariance and periodic x; adding sidewalls or perturbations belongs to later coupled EHD missions.

## Touch order

1. `src/kernels/ehd_2d.jl` — collision formulas, E moment, and EHD-specific wall extrapolation.
2. `src/drivers/ehd.jl` — parameter mapping, analytic profiles, initialization, convergence criteria, and error metrics.
3. `test/analytical/ehd_hydrostatic_2d.jl` — CPU hydrostatic fixed-point validation.
4. `src/Kraken.jl` — include registration only.
