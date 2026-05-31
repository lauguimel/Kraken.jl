# Viscoelastic flow past a confined cylinder (Oldroyd-B)

This benchmark validates Kraken's Oldroyd-B viscoelastic solver against
**RheoTool** (rheoFoam, OpenFOAM-9, log-conformation) on the canonical
**confined-cylinder** problem — the standard High-Weissenberg-Number-Problem
(HWNP) testbed of Alves, Oliveira & Pinho (2001).

- **Geometry**: cylinder of radius `R` on the channel centreline, blockage
  ratio `D/H = 0.5` (half-height `2R`), upstream/downstream length `15R`.
- **Fluid**: Oldroyd-B, `Re = 1`, solvent fraction `β = η_s/η₀ = 0.59`
  (Boger-fluid value used throughout the cylinder literature).
- **Scaling**: diffusive (fixed lattice viscosity `ν_total = 0.15`, `τ = 0.95`);
  `u_mean = ν_total·Re/R`, `λ = Wi·R/u_mean`.
- **Solver**: TRT-LBM + log-conformation finite-volume polymer transport
  (`run_viscoelastic_logfv_cylinder_coupled_2d`), MUSCL–superbee advection of
  the conformation field, half-way bounce-back cylinder wall.

All Kraken numbers below are **Float64 on an NVIDIA A100** (CUDA), 300 000
steps. RheoTool numbers are rheoFoam at steady state (`t = 20`), 2-core MPI.

## Drag coefficient vs Weissenberg number

At the production resolution `R = 50`, Kraken matches RheoTool to **better than
1 %** across the validated range `Wi ≤ 1`:

| Wi  | Kraken Cd | RheoTool Cd | rel. error |
|-----|-----------|-------------|------------|
| 0.1 | 129.92    | 130.43      | **−0.40 %** |
| 0.5 | 118.68    | 119.71      | **−0.86 %** |
| 1.0 | 119.24    | 120.38      | **−0.96 %** |

The drag coefficient follows the **characteristic elastic signature** reported
across the cylinder literature (Alves–Oliveira–Pinho 2001; Hulsen et al. 2005;
Claus & Phillips 2013): a shallow **minimum near `Wi ≈ 0.5`** followed by an
**elastic upturn**. Kraken reproduces this shape (Cd drops 129.9 → 118.7 to the
minimum, then rises again past `Wi ≈ 1`), not just a single matched value.

### Mesh convergence

The agreement improves monotonically as the cylinder is resolved, confirming
the residual gap is discretisation (not a modelling error):

| Wi = 1.0 | R = 10 | R = 30 | R = 50 | RheoTool |
|----------|--------|--------|--------|----------|
| Cd       | 112.93 | 118.10 | 119.24 | 120.38   |
| rel. err | −6.2 % | −1.9 % | −0.96 % | —        |

## Solvent-fraction (β) sensitivity

Sweeping the solvent fraction at moderate `Wi` (R = 50) cross-validates the two
codes over the full range where **both** converge, and reveals a shared
stability boundary:

| β    | Kraken Wi=0.5 | RheoTool Wi=0.5 | Kraken Wi=1.0 | RheoTool Wi=1.0 |
|------|---------------|-----------------|---------------|-----------------|
| 0.59 | 118.68        | 119.71          | 119.24        | 120.38 (−0.96 %) |
| 0.30 | 107.98        | 109.92          | 107.99        | 107.28 (+0.66 %) |
| 0.10 | diverges      | diverges        | diverges      | diverges        |
| 0.01 | diverges      | diverges        | diverges      | diverges        |

Both solvers agree on the trend (drag decreases as the polymer fraction grows
from β = 0.59 to 0.30) **and both diverge for β ≤ 0.1** on this mesh: Kraken
returns NaN, while RheoTool's PETSc solver aborts with a floating-point
exception (divide-by-zero) at `t ≈ 5`. The strongly-polymeric corner is a
genuine physical/numerical boundary, reproduced identically by two independent
methods (LBM + log-FV vs FV + log-conformation) — not a solver-specific
weakness.

## First normal-stress difference (N1)

The wake first normal-stress difference `N1 = τ_xx − τ_yy` (polymer extra-stress,
`τ_p = (ν_p/λ)(C − I)` in both codes) is **qualitatively concordant** between
Kraken and RheoTool — N1 rises with `Wi` and rises as `β` decreases (more
polymer) on both sides — but the **absolute** wake-N1 maxima differ by up to
~30–44 %, and the discrepancy **changes sign** with the parameters:

| β    | Wi  | Kraken/RheoTool N1 ratio |
|------|-----|--------------------------|
| 0.59 | 0.5 | 0.77 |
| 0.59 | 1.0 | 0.70 |
| 0.30 | 0.5 | 0.89 |
| 0.30 | 1.0 | 1.44 |

This is **not** a unit-conversion or stress-convention artifact: an independent
two-method derivation confirmed that every candidate reference stress
(η₀U/R, ρU², η_pU/R, G = η_p/λ) cancels identically in the Kraken/RheoTool
ratio, so no nondimensionalization can collapse the four points. Both codes were
verified to report the same quantity (polymer extra-stress, ρ = 1, β = η_s/η₀).
The residual is a **genuine difference in the resolved wake stress field**
(under-prediction growing with Wi, sign-flipping at low β) — consistent with
either under-resolution of the steep wake-stress gradient at R = 50 or a
front-shoulder polymer back-force difference. The integrated **drag** (Cd),
which is the mandate's primary integrated-quantity gate, matches to <1 %; N1 is
reported here as a documented open difference, not a passing ≤5 % comparison.

## Stability envelope (high Weissenberg)

Pushing past the validated range, Kraken's half-way-bounce-back cylinder
remains **NaN-free up to `Wi = 10`** at both R = 30 and R = 50:

| Wi   | 1.5   | 2.0   | 3.0   | 5.0   | 7.0   | 10    |
|------|-------|-------|-------|-------|-------|-------|
| R=50 | 123.6 | 127.8 | 141.8 | 138.1 | 123.0 | 114.7 |
| R=30 | 121.6 | 119.8 | 116.3 | 111.3 | 106.2 | 104.3 |

This stability range is at or beyond the state of the art for **LBM-based**
viscoelastic solvers, which typically report ceilings near `Wi ≈ 1` on this
benchmark (Kuron et al. 2021).

!!! warning "Stability is not accuracy"
    The high-`Wi` drag values above are **stable but not converged**. The two
    resolutions diverge increasingly with `Wi` (e.g. 141.8 vs 116.3 at Wi = 3),
    because the polymer relaxation time `λ = Wi·R/u_mean` grows very large
    (≈ 1.7×10⁵ lattice units at Wi = 10, R = 50) and 300 000 steps no longer
    reach steady state. Only `Wi ≤ 1` is quantitatively validated here; the
    `Wi ≤ 10` figures demonstrate robustness, not benchmark-grade drag.

## Wall boundary condition

A sub-cell linear-interpolation wall (Bouzidi–Filippova–Hänel, `wall = bouzidi_fl`)
was evaluated as an alternative to half-way bounce-back. It **diverges (NaN) at
the production resolution** for `Wi ≥ 0.5` (R ≥ 50) and over-predicts drag where
it survives on coarse grids, so half-way bounce-back is the validated default for
this benchmark.

## Reproducing this benchmark

```julia
using Kraken
result = run_simulation("benchmarks/krk/viscoelastic/cylinder_oldroyd_b.krk")
@show result.Cd
```

The `.krk` file declares the channel, the cylinder obstacle, the Oldroyd-B
rheology (`Rheology oldroyd_b { nu_s, nu_p, lambda }`) and `Re`/`Wi`; the runner
dispatches to the coupled log-FV cylinder driver. The Aqua sweep scripts that
produced the tables above are
`bench/viscoelastic_logfv/run_cyl_m8_diffusive_matrix_a100.pbs` (Wi sweep × wall
BC × R) and `run_cyl_m8_beta_highwi_a100.pbs` (β sweep + Wi staircase); the
RheoTool references are `bench/rheotool/run_cyl_beta_sweep_aqua.pbs`.

## References

- M. A. Alves, P. J. Oliveira, F. T. Pinho, *The flow of viscoelastic fluids
  past a cylinder: finite-volume high-resolution methods*, J. Non-Newtonian
  Fluid Mech. **97** (2001) 207–232.
- M. A. Hulsen, R. Fattal, R. Kupferman, *Flow of viscoelastic fluids past a
  cylinder at high Weissenberg number: stabilized simulations using matrix
  logarithms*, J. Non-Newtonian Fluid Mech. **127** (2005) 27–39.
- R. Fattal, R. Kupferman, *Constitutive laws for the matrix-logarithm of the
  conformation tensor*, J. Non-Newtonian Fluid Mech. **123** (2004) 281–285.
- S. Claus, T. N. Phillips, *Viscoelastic flow around a confined cylinder using
  spectral/hp element methods*, J. Non-Newtonian Fluid Mech. **200** (2013)
  131–146.
- J. Kuron et al., *An extensible lattice Boltzmann method for viscoelastic
  flows: complex and moving boundaries in Oldroyd-B fluids*, Eur. Phys. J. E
  **44** (2021) 1.
