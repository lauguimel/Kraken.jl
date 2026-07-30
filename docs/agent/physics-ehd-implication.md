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

The EHD physics module owns the electric potential and charge-density D2Q9
scalar populations for the Luo-Wu-Yi-Tan electroconvection model, plus the
CPU coupled canary that feeds Coulomb force density `F=qE` into the existing
Navier-Stokes BGK+Guo kernel. It implements the Jiachen standard-LBM formulas
for `phi` pseudo-time Poisson collision, electric-field recovery from the
potential population first moment, charge drift-diffusion collision, and the
coarse electroconvection onset bracket.

## Public surface

- `run_ehd_hydrostatic_2d(; Nx, Ny, C, M, Ma_E, alpha, charge_scheme, backend, FT)` — standalone hydrostatic EHD validation driver. Returns 2D fields, x-averaged profiles, analytic profiles, relative L2 errors, convergence metadata, and lattice parameters.
- `run_electroconvection_2d(; Nx, Ny, C, M, T, Ma_E, alpha, max_cycles, phi_substeps, force_projection, backend, FT)` — coupled CPU canary. Returns flow, charge, potential, electric field, Coulomb force, velocity history, lattice mapping, and loop diagnostics.
- Kernel-level surface: `collide_electric_potential_2d!`, `compute_electric_field_2d!`, `collide_electric_charge_srt_2d!`, `collide_electric_charge_regularized_2d!`, `compute_ehd_scalar_2d!`, scalar NEE wall/box BC kernels, EHD-local non-periodic stream, free-slip sidewall port, Coulomb force helper, and Guo-corrected macro recovery.

## Reads from

- `lbm` — D2Q9 ordering and `collide_guo_field_2d!` for forced Navier-Stokes.
- `bc` — only the convention that wall-node scalar BCs are applied after streaming. EHD wall values are local non-equilibrium extrapolation kernels owned by this module.

## Writes to

- Mutates potential populations `phi_f`, charge populations `q_f`, scalar moments `phi`/`q`, and electric fields `Ex`/`Ey` in place.
- The coupled driver also mutates local NS populations, `rho/ux/uy`, and `Fx/Fy`.
- Allocates per run in the driver: two ping-pong DDF arrays for each scalar, scalar moment arrays, electric-field arrays, and small host copies for convergence tests and returned profiles.
- Does not mutate parser state, units registries, generic BC framework, or GPU-specific paths.

## Backend constraints

- Kernels are KernelAbstractions `@kernel` functions and use `@Const` for read-only arrays.
- Hot kernels are allocation-free and unroll the D2Q9 operations.
- The hydrostatic driver is backend-generic for arrays, but convergence checks copy the small validation fields to the host each step. The coupled electroconvection driver is a CPU canary path; force projection uses per-row host reductions.
- No MRT/TRT charge collision, `.krk` parser branch, simulation-runner path, or GPU-specific EHD path is included.

## Coupled Loop Conventions

- PRE/Jiachen lattice mapping: `K = Ma_E*H*cs/delta_U`, `nu = M^2*K*delta_U/T`, `tau = 0.5 + 3*nu`, `eps = (M*K)^2`, `q_inj = C*eps*delta_U/H^2`, `D = alpha*K*delta_U`, and `tau_q = 0.5 + 3*D`.
- Each outer cycle solves or substeps `phi`, computes `E`, computes charge-advection macros from the previous force, advances charge with equilibrium drift `u + K*E`, forms current `F=qE`, collides NS through `collide_guo_field_2d!`, streams, applies free-slip sidewall mirroring, then recovers current Guo-corrected macros.
- Guo convention: `collide_guo_field_2d!` receives force density and computes the internal equilibrium velocity with `+F/2`. Driver macro recovery also adds `+F/2` from the matching time level: previous force before charge advection and current force after the forced NS step.
- Side scalar BCs are EHD-local NEE extrapolation. `phi` uses fixed bottom/top potential and zero-gradient sides; `q` uses fixed bottom injection, zero-gradient top, and zero-gradient sides. Wall/side corners are owned by side BCs, matching the MATLAB plate-mask order.
- Free-slip flow sidewalls are not reused from a generic BC path. The MATLAB population mirror is ported explicitly after streaming, with `ux=0` on side columns and `uy` copied from the adjacent interior column for macro enforcement.
- Optional force projection ports Jiachen's per-row mean subtraction over fluid nodes. `:xy` subtracts both components; `:y` subtracts only vertical force; default is `:none`.
- NS coupling uses BGK+Guo rather than Jiachen's MRT NS collision. This is sufficient for the analytical canary but is not a production onset benchmark claim.

## Failure modes

- `tau_q` is close to `0.5` at `alpha=1e-4`; the hydrostatic validation passes with SRT, while the coupled onset canary uses the regularized charge collision as in Jiachen's default validation path.
- The wall convention is wall-node non-equilibrium extrapolation. Wall values live at `y*=0,1`; interior DDF profiles are compared on the effective half-link samples `y*=(j-3/2)/(Ny-1)`, matching the discrete charge population location near the injector.
- The analytic E profile is positive upward; if `Ey` changes sign, inspect the potential source sign and D2Q9 direction ordering before changing analytics.
- The hydrostatic validation assumes x-invariance and periodic x. The onset validation uses sidewalls and a sinusoidal charge perturbation.
- `test/analytical/ehd_onset_2d.jl` uses `Nx=59`, `Ny=96`, `A≈0.611`, `C=10`, `M=10`, `alpha=1e-4`, `Ma_E=0.01`, `phi_substeps=1`, and 50k cycles per branch. Setup runs measured about 21 s per branch. `T=150` decayed from `max|u|≈2.40e-3` to `5.08e-6`; `T=190` was marginal at `1.89e-4`; `T=220` grew to `1.62e-3`, over 100x the `T=150` final value. The bracket is a coarse BGK+Guo canary with shifted `T_c`, not a PRE-accurate critical-number measurement.

## Touch order

1. `src/kernels/ehd_2d.jl` — collision formulas, E moment, and EHD-specific wall extrapolation.
2. `src/kernels/ehd_bc_2d.jl` — EHD-local sidewall, force, Guo-macro, and free-slip canary helpers.
3. `src/drivers/ehd.jl` / `src/drivers/ehd_ec.jl` — parameter mapping, analytic profiles, initialization, convergence criteria, coupled loop, and error/history metrics.
4. `test/analytical/ehd_hydrostatic_2d.jl` and `test/analytical/ehd_onset_2d.jl` — CPU analytical validations.
5. `src/Kraken.jl` — include/export registration only.
